from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
warnings.filterwarnings(
    "ignore",
    message="MiniBatchKMeans is known to have a memory leak.*",
)

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=ConvergenceWarning)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
DATA_HOME = ROOT / ".sklearn_data"
TABLES.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
DEFAULT_OUTPUT_STEM = "vacs_transformer_embedding_probe"
DEFAULT_DATASET_NAME = "20newsgroups_20class_deberta"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 256


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def env_torch_dtype(name: str) -> torch.dtype | None:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    if value not in mapping:
        raise ValueError(
            f"Unsupported {name}={value!r}. Use auto, float16/fp16, bfloat16/bf16, float32/fp32, or float64/fp64."
        )
    return mapping[value] if mapping[value] != "auto" else None


MODEL_NAME = env_str("VACS_EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
MODEL_PATH = env_str("VACS_EMBEDDING_MODEL_PATH", "")
OUTPUT_STEM = env_str("VACS_EMBEDDING_OUTPUT_STEM", DEFAULT_OUTPUT_STEM)
DATASET_NAME = env_str("VACS_EMBEDDING_DATASET_NAME", DEFAULT_DATASET_NAME)
BATCH_SIZE = env_int("VACS_EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE)
MAX_LENGTH = env_int("VACS_EMBEDDING_MAX_LENGTH", DEFAULT_MAX_LENGTH)
MODEL_DTYPE = env_torch_dtype("VACS_EMBEDDING_MODEL_DTYPE")
MODEL_ATTENTION_IMPL = env_str("VACS_EMBEDDING_ATTENTION_IMPL", "")
MODEL_DEVICE_MAP = env_str("VACS_EMBEDDING_DEVICE_MAP", "")
MODEL_MAX_MEMORY_GPU_GB = env_str("VACS_EMBEDDING_MAX_MEMORY_GPU_GB", "")
MODEL_MAX_MEMORY_CPU_GB = env_str("VACS_EMBEDDING_MAX_MEMORY_CPU_GB", "")
MODEL_OFFLOAD_FOLDER = env_str("VACS_EMBEDDING_OFFLOAD_FOLDER", "")
MODEL_SOURCE = MODEL_PATH or MODEL_NAME

if Path(MODEL_SOURCE).exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SEEDS = list(range(8))
BUDGETS = [1, 2]
BASE_SELECTORS = ["random", "herding", "kcenter", "boundary", "kmeans", "marc"]
METHODS = BASE_SELECTORS + ["vacs"]
METHOD_NAMES = {
    "random": "Random",
    "herding": "Herding",
    "kcenter": "K-center",
    "boundary": "Boundary",
    "kmeans": "K-means",
    "marc": "MARC",
    "vacs": "VACS",
}
def l2_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    Y = np.asarray(Y, dtype=np.float32, order="C")
    x2 = np.einsum("ij,ij->i", X, X, optimize=True)[:, None]
    y2 = np.einsum("ij,ij->i", Y, Y, optimize=True)[None, :]
    return np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)


def class_centroids(X: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    return {int(label): X[y == label].mean(axis=0) for label in np.unique(y)}


def margins_to_centroids(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    labels = np.unique(y)
    centroids = class_centroids(X, y)
    C = np.vstack([centroids[int(label)] for label in labels])
    distances = l2_distances(X, C)
    own_pos = np.array([np.where(labels == label)[0][0] for label in y])
    own = distances[np.arange(len(y)), own_pos]
    distances[np.arange(len(y)), own_pos] = np.inf
    nearest_other = distances.min(axis=1)
    return nearest_other - own


def farthest_first(X: np.ndarray, candidate_indices: np.ndarray, k: int) -> list[int]:
    candidate_indices = list(map(int, candidate_indices))
    if k <= 0 or not candidate_indices:
        return []
    centroid = X[candidate_indices].mean(axis=0, keepdims=True)
    first = candidate_indices[int(np.argmin(l2_distances(X[candidate_indices], centroid).ravel()))]
    selected = [first]
    remaining = [idx for idx in candidate_indices if idx != first]
    min_distances = None
    while len(selected) < k and remaining:
        distances = l2_distances(X[remaining], X[selected[-1]][None, :]).ravel()
        min_distances = distances if min_distances is None else np.minimum(min_distances, distances)
        best_pos = int(np.argmax(min_distances))
        selected.append(remaining.pop(best_pos))
        min_distances = np.delete(min_distances, best_pos)
    return selected[:k]


def kmeans_select_class(X: np.ndarray, class_indices: np.ndarray, k: int, seed: int, weights=None) -> list[int]:
    if k <= 0:
        return []
    if k >= len(class_indices):
        return class_indices.tolist()
    if k == 1:
        centroid = X[class_indices].mean(axis=0, keepdims=True)
        return [int(class_indices[int(np.argmin(l2_distances(X[class_indices], centroid).ravel()))])]

    local_X = X[class_indices]
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        init="random",
        n_init=5,
        max_iter=80,
        batch_size=min(256, max(32, len(class_indices))),
        reassignment_ratio=0.0,
    )
    km.fit(local_X, sample_weight=weights)
    distances = l2_distances(km.cluster_centers_, local_X)
    chosen = []
    used = set()
    for row in distances:
        for pos in np.argsort(row):
            idx = int(class_indices[int(pos)])
            if idx not in used:
                chosen.append(idx)
                used.add(idx)
                break
    if len(chosen) < k:
        chosen = farthest_first(X, class_indices, k)
    return chosen[:k]


def select_indices(X: np.ndarray, y: np.ndarray, budget_per_class: int, method: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = class_centroids(X, y)
    margins = margins_to_centroids(X, y)
    selected: list[int] = []
    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        k = min(int(budget_per_class), len(class_indices))
        if method == "random":
            selected.extend(rng.choice(class_indices, size=k, replace=False).tolist())
        elif method == "herding":
            distances = l2_distances(X[class_indices], centroids[int(label)][None, :]).ravel()
            selected.extend(class_indices[np.argsort(distances)[:k]].tolist())
        elif method == "kcenter":
            selected.extend(farthest_first(X, class_indices, k))
        elif method == "boundary":
            selected.extend(class_indices[np.argsort(margins[class_indices])[:k]].tolist())
        elif method == "kmeans":
            selected.extend(kmeans_select_class(X, class_indices, k, seed))
        elif method == "marc":
            class_margins = margins[class_indices]
            scale = np.std(class_margins) + 1e-8
            z = (class_margins - np.median(class_margins)) / scale
            weights = 0.2 + 1.8 / (1.0 + np.exp(-z))
            selected.extend(kmeans_select_class(X, class_indices, k, seed, weights=weights))
        else:
            raise ValueError(method)
    return np.array(sorted(set(selected)), dtype=int)


def evaluate_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subset: np.ndarray,
    seed: int,
) -> dict[str, float]:
    n_neighbors = min(3, len(subset))
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        metric="euclidean",
    )
    clf.fit(X_train[subset], y_train[subset])
    pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def validation_aligned_select(
    X: np.ndarray,
    y: np.ndarray,
    budget_per_class: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    indices = np.arange(len(y))
    pool_idx, val_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=10_000 + 37 * seed + budget_per_class,
        stratify=y,
    )
    best = None
    for selector in BASE_SELECTORS:
        local_subset = select_indices(X[pool_idx], y[pool_idx], budget_per_class, selector, seed)
        subset = pool_idx[local_subset]
        metrics = evaluate_subset(X, y, X[val_idx], y[val_idx], subset, seed)
        score = (metrics["accuracy"], metrics["macro_f1"], -BASE_SELECTORS.index(selector))
        if best is None or score > best["score"]:
            best = {"selector": selector, "score": score}
    final_subset = select_indices(X, y, budget_per_class, best["selector"], seed)
    return final_subset, str(best["selector"])


def load_20ng() -> tuple[list[str], np.ndarray]:
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home=DATA_HOME,
    )
    return list(data.data), np.asarray(data.target)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def embed_texts(texts: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, local_files_only=True)
    blank_texts = sum(1 for text in texts if not text.strip())
    placeholder = tokenizer.eos_token or tokenizer.pad_token or tokenizer.unk_token or " "
    normalized_texts = [text if text.strip() else placeholder for text in texts]
    model_kwargs = {"local_files_only": True}
    if MODEL_DTYPE is not None:
        model_kwargs["torch_dtype"] = MODEL_DTYPE
    model_kwargs["low_cpu_mem_usage"] = True
    if MODEL_ATTENTION_IMPL:
        model_kwargs["attn_implementation"] = MODEL_ATTENTION_IMPL
    if MODEL_DEVICE_MAP:
        model_kwargs["device_map"] = MODEL_DEVICE_MAP
        max_memory = {}
        if MODEL_MAX_MEMORY_GPU_GB:
            max_memory[0] = f"{MODEL_MAX_MEMORY_GPU_GB}GiB"
        if MODEL_MAX_MEMORY_CPU_GB:
            max_memory["cpu"] = f"{MODEL_MAX_MEMORY_CPU_GB}GiB"
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if MODEL_OFFLOAD_FOLDER:
            model_kwargs["offload_folder"] = MODEL_OFFLOAD_FOLDER
        model_kwargs["offload_state_dict"] = True
        model = AutoModel.from_pretrained(MODEL_SOURCE, **model_kwargs).eval()
        input_device = getattr(model, "device", torch.device("cpu"))
    else:
        model = AutoModel.from_pretrained(MODEL_SOURCE, **model_kwargs).to("cuda").eval()
        input_device = torch.device("cuda")
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    embeddings = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(normalized_texts), BATCH_SIZE):
            batch_texts = normalized_texts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = pooled.detach().to(torch.float32).cpu().numpy()
            if not np.isfinite(pooled).all():
                raise ValueError("Non-finite values detected in the pooled embeddings.")
            embeddings.append(pooled)
    elapsed = time.time() - started
    X = np.vstack(embeddings)
    X = Normalizer().fit_transform(X).astype(np.float32)
    meta = {
        "model": MODEL_NAME,
        "model_source": MODEL_SOURCE,
        "output_stem": OUTPUT_STEM,
        "dataset": DATASET_NAME,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model_dtype": str(MODEL_DTYPE) if MODEL_DTYPE is not None else "default",
        "attention_impl": MODEL_ATTENTION_IMPL or "default",
        "embedding_dim": int(X.shape[1]),
        "documents": int(X.shape[0]),
        "blank_texts": int(blank_texts),
        "embedding_runtime_sec": float(elapsed),
        "batch_size": int(BATCH_SIZE),
        "max_length": int(MAX_LENGTH),
    }
    return X, meta


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def fmt_signed(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def write_outputs(rows: list[dict[str, object]], meta: dict[str, object]) -> None:
    out_csv = TABLES / f"{OUTPUT_STEM}_results.csv"
    fields = [
        "dataset",
        "seed",
        "budget_per_class",
        "method",
        "selector_detail",
        "selected",
        "accuracy",
        "macro_f1",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            handle.write(",".join(str(row[field]) for field in fields) + "\n")

    methods = sorted(set(str(row["method"]) for row in rows), key=lambda m: METHODS.index(m))
    summary_rows = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary_rows.append({
            "method": method,
            "accuracy_mean": float(np.mean([float(row["accuracy"]) for row in method_rows])),
            "macro_f1_mean": float(np.mean([float(row["macro_f1"]) for row in method_rows])),
        })
    summary_rows.sort(key=lambda row: row["accuracy_mean"], reverse=True)

    out_summary = TABLES / f"{OUTPUT_STEM}_summary.csv"
    with out_summary.open("w", encoding="utf-8", newline="") as handle:
        handle.write("method,accuracy_mean,macro_f1_mean\n")
        for row in summary_rows:
            handle.write(f"{row['method']},{row['accuracy_mean']},{row['macro_f1_mean']}\n")

    vacs_acc = next(row for row in summary_rows if row["method"] == "vacs")["accuracy_mean"]
    best_static = max(
        (row for row in summary_rows if row["method"] != "vacs"),
        key=lambda row: row["accuracy_mean"],
    )
    meta["headline"] = {
        "vacs_accuracy": float(vacs_acc),
        "best_static_method": str(best_static["method"]),
        "best_static_accuracy": float(best_static["accuracy_mean"]),
        "absolute_gain": float(vacs_acc - best_static["accuracy_mean"]),
    }
    (RESULTS / f"{OUTPUT_STEM}_manifest.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Acc. & Macro-F1 \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{METHOD_NAMES[row['method']]} & {pct(row['accuracy_mean'])} & {pct(row['macro_f1_mean'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / f"{OUTPUT_STEM}_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run() -> None:
    started = time.time()
    texts, y = load_20ng()
    X, meta = embed_texts(texts)
    rows = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=seed,
            stratify=y,
        )
        for budget in BUDGETS:
            for method in METHODS:
                if method == "vacs":
                    subset, selector_detail = validation_aligned_select(X_train, y_train, budget, seed)
                else:
                    subset = select_indices(X_train, y_train, budget, method, seed)
                    selector_detail = method
                metrics = evaluate_subset(X_train, y_train, X_test, y_test, subset, seed)
                rows.append({
        "dataset": DATASET_NAME,
                    "seed": seed,
                    "budget_per_class": budget,
                    "method": method,
                    "selector_detail": selector_detail,
                    "selected": int(len(subset)),
                    **metrics,
                })
    meta.update({
        "seeds": SEEDS,
        "budgets_per_class": BUDGETS,
        "selectors": METHODS,
        "runtime_sec": float(time.time() - started),
        "protocol": (
            f"Frozen {MODEL_NAME} mean-pooled document embeddings on the full 20 Newsgroups corpus, "
            "70/30 outer split, 75/25 internal VACS validation, 3-NN downstream classifier."
        ),
    })
    write_outputs(rows, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    run()
