from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
warnings.filterwarnings("ignore", message="MiniBatchKMeans is known to have a memory leak.*")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import run_vacs_experiments as vacs_experiments
from run_vacs_experiments import (
    BASE_SELECTORS,
    DATASET_NAMES,
    evaluate_subset,
    preprocess_dense,
    select_indices,
    validation_aligned_select,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
DATA_HOME = ROOT / ".torchvision_data"
TABLES.mkdir(parents=True, exist_ok=True)
DATA_HOME.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASETS = "cifar10,cifar100,fashion_mnist"
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7"
DEFAULT_BUDGETS = "1,2"
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 4
DEFAULT_BACKBONES = "convnext_tiny,resnet50,resnet18"

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

DATASET_NAMES_VISION = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "fashion_mnist": "Fashion-MNIST",
}


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_dataset_config(name: str) -> dict[str, object]:
    if name == "cifar10":
        return {
            "name": name,
            "display": DATASET_NAMES_VISION[name],
            "kind": "rgb",
            "train_ctor": lambda transform: datasets.CIFAR10(
                root=str(DATA_HOME), train=True, download=True, transform=transform
            ),
            "test_ctor": lambda transform: datasets.CIFAR10(
                root=str(DATA_HOME), train=False, download=True, transform=transform
            ),
        }
    if name == "cifar100":
        return {
            "name": name,
            "display": DATASET_NAMES_VISION[name],
            "kind": "rgb",
            "train_ctor": lambda transform: datasets.CIFAR100(
                root=str(DATA_HOME), train=True, download=True, transform=transform
            ),
            "test_ctor": lambda transform: datasets.CIFAR100(
                root=str(DATA_HOME), train=False, download=True, transform=transform
            ),
        }
    if name == "fashion_mnist":
        rgbify = transforms.Lambda(lambda img: img.convert("RGB"))
        return {
            "name": name,
            "display": DATASET_NAMES_VISION[name],
            "kind": "grayscale",
            "prefix": rgbify,
            "train_ctor": lambda transform: datasets.FashionMNIST(
                root=str(DATA_HOME), train=True, download=True, transform=transform
            ),
            "test_ctor": lambda transform: datasets.FashionMNIST(
                root=str(DATA_HOME), train=False, download=True, transform=transform
            ),
        }
    raise ValueError(f"Unsupported dataset: {name}")


def resolve_backbone(allowed_backbones: list[str] | None = None) -> tuple[str, torch.nn.Module, object]:
    candidates = [
        ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
        ("resnet50", "ResNet50_Weights"),
        ("resnet18", "ResNet18_Weights"),
    ]
    allowed = None if allowed_backbones is None else {name for name in allowed_backbones if name}
    for model_name, weight_enum_name in candidates:
        if allowed is not None and model_name not in allowed:
            continue
        if not hasattr(models, model_name) or not hasattr(models, weight_enum_name):
            continue
        weights_enum = getattr(models, weight_enum_name)
        weights = weights_enum.DEFAULT
        model = getattr(models, model_name)(weights=weights)
        model.eval()
        return model_name, model, weights
    if allowed is not None:
        raise RuntimeError(f"No supported torchvision backbone found in allowed set {sorted(allowed)!r}.")
    raise RuntimeError("No supported torchvision backbone found.")


def build_transform(weights, kind: str, prefix=None):
    base = [weights.transforms()]
    if prefix is not None:
        base.insert(0, prefix)
    return transforms.Compose(base)


def make_embedder(backbone_name: str, model: torch.nn.Module):
    if backbone_name.startswith("convnext"):
        features = model.features
        pool = model.avgpool

        def embed(batch: torch.Tensor) -> torch.Tensor:
            x = features(batch)
            x = pool(x)
            return torch.flatten(x, 1)

        return embed

    if backbone_name.startswith("resnet"):
        trunk = torch.nn.Sequential(*list(model.children())[:-1])

        def embed(batch: torch.Tensor) -> torch.Tensor:
            x = trunk(batch)
            return torch.flatten(x, 1)

        return embed

    raise RuntimeError(f"Unsupported backbone for feature extraction: {backbone_name}")


def extract_embeddings(dataset, backbone_name: str, model: torch.nn.Module, device: torch.device, batch_size: int, num_workers: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    embed = make_embedder(backbone_name, model)
    embeddings = []
    labels = []
    started = time.time()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            feats = embed(images).detach().to(torch.float32).cpu().numpy()
            embeddings.append(feats)
            labels.append(np.asarray(targets))
    elapsed = time.time() - started
    X = np.vstack(embeddings)
    X = Normalizer().fit_transform(X).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(int, copy=False)
    return X, y, float(elapsed)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def fmt_signed(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def write_outputs(rows: list[dict[str, object]], meta: dict[str, object]) -> None:
    out_csv = TABLES / "vacs_torchvision_embedding_probe_results.csv"
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

    summary_rows = []
    df = pd.DataFrame(rows)
    for dataset, sub in df.groupby("dataset", sort=True):
        vacs = sub[sub["method"] == "vacs"]
        statics = sub[sub["method"] != "vacs"]
        vacs_acc = float(vacs["accuracy"].mean())
        vacs_f1 = float(vacs["macro_f1"].mean())
        static_means = statics.groupby("method")["accuracy"].mean()
        static_f1 = statics.groupby("method")["macro_f1"].mean()
        best_static_method = str(static_means.idxmax())
        summary_rows.append(
            {
                "dataset": dataset,
                "display": DATASET_NAMES_VISION[dataset],
                "units": int(len(vacs)),
                "vacs_accuracy": vacs_acc,
                "vacs_macro_f1": vacs_f1,
                "best_static_method": best_static_method,
                "best_static_accuracy": float(static_means.loc[best_static_method]),
                "best_static_macro_f1": float(static_f1.loc[best_static_method]),
                "gain": vacs_acc - float(static_means.loc[best_static_method]),
            }
        )

    pooled = df.copy()
    vacs = pooled[pooled["method"] == "vacs"]
    statics = pooled[pooled["method"] != "vacs"]
    pooled_static_means = statics.groupby("method")["accuracy"].mean()
    pooled_static_f1 = statics.groupby("method")["macro_f1"].mean()
    pooled_best_static = str(pooled_static_means.idxmax())
    summary_rows.append(
        {
            "dataset": "pooled",
            "display": "Pooled",
            "units": int(len(vacs)),
            "vacs_accuracy": float(vacs["accuracy"].mean()),
            "vacs_macro_f1": float(vacs["macro_f1"].mean()),
            "best_static_method": pooled_best_static,
            "best_static_accuracy": float(pooled_static_means.loc[pooled_best_static]),
            "best_static_macro_f1": float(pooled_static_f1.loc[pooled_best_static]),
            "gain": float(vacs["accuracy"].mean() - pooled_static_means.loc[pooled_best_static]),
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES / "vacs_torchvision_embedding_probe_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & VACS & Best static & Gain & Backbone \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        if row["dataset"] == "pooled":
            label = "Pooled"
        else:
            label = str(row["display"])
        lines.append(
            f"{label} & {pct(float(row['vacs_accuracy']))} & "
            f"{METHOD_NAMES[str(row['best_static_method'])]} {pct(float(row['best_static_accuracy']))} & "
            f"{fmt_signed(float(row['gain']))} & {meta['backbone_name']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_torchvision_embedding_probe_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    meta.update(
        {
            "probe_count": int(summary_df[summary_df["dataset"] != "pooled"].shape[0]),
            "positive_dataset_count": int(
                ((summary_df["dataset"] != "pooled") & (summary_df["gain"] > 0)).sum()
            ),
            "pooled_gain": float(summary_df.loc[summary_df["dataset"] == "pooled", "gain"].iloc[0]),
            "summary_rows": summary_rows,
        }
    )
    (RESULTS / "vacs_torchvision_embedding_probe_manifest.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def run() -> None:
    start = time.time()
    dataset_names = parse_csv_strings(env_str("VACS_VISION_DATASETS", DEFAULT_DATASETS))
    seeds = parse_csv_ints(env_str("VACS_VISION_SEEDS", DEFAULT_SEEDS))
    budgets = parse_csv_ints(env_str("VACS_VISION_BUDGETS", DEFAULT_BUDGETS))
    batch_size = env_int("VACS_VISION_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    num_workers = env_int("VACS_VISION_NUM_WORKERS", DEFAULT_NUM_WORKERS)
    backbone_filters = parse_csv_strings(env_str("VACS_VISION_BACKBONES", DEFAULT_BACKBONES))

    dataset_configs = [make_dataset_config(name) for name in dataset_names]
    backbone_name, model, weights = resolve_backbone(backbone_filters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    rows = []
    dataset_meta = []
    backbone_runtime_total = 0.0

    original_make_classifier = vacs_experiments.make_classifier

    def memory_safe_make_classifier(learner: str, subset_size: int, seed: int):
        clf = original_make_classifier(learner, subset_size, seed)
        if learner == "knn3":
            clf.set_params(algorithm="kd_tree")
        return clf

    vacs_experiments.make_classifier = memory_safe_make_classifier
    try:
        for cfg in dataset_configs:
            name = str(cfg["name"])
            display = str(cfg["display"])
            transform = build_transform(weights, str(cfg["kind"]), cfg.get("prefix"))
            train_dataset = cfg["train_ctor"](transform)
            test_dataset = cfg["test_ctor"](transform)
            X_train_full, y_train_full, train_embed_sec = extract_embeddings(
                train_dataset,
                backbone_name,
                model,
                device,
                batch_size,
                num_workers,
            )
            X_test, y_test, test_embed_sec = extract_embeddings(
                test_dataset,
                backbone_name,
                model,
                device,
                batch_size,
                num_workers,
            )
            backbone_runtime_total += train_embed_sec + test_embed_sec
            dataset_meta.append(
                {
                    "dataset": name,
                    "display": display,
                    "train_examples": int(len(y_train_full)),
                    "test_examples": int(len(y_test)),
                    "embedding_dim": int(X_train_full.shape[1]),
                    "train_embedding_runtime_sec": float(train_embed_sec),
                    "test_embedding_runtime_sec": float(test_embed_sec),
                }
            )

            for seed in seeds:
                X_train, X_test_proc = preprocess_dense(X_train_full, X_test)

                for budget in budgets:
                    for method in METHODS:
                        if method == "vacs":
                            subset, selector_detail = validation_aligned_select(
                                X_train,
                                y_train_full,
                                budget,
                                seed,
                            )
                        else:
                            subset = select_indices(X_train, y_train_full, budget, method, seed)
                            selector_detail = method
                        metrics = evaluate_subset(
                            X_train,
                            y_train_full,
                            X_test_proc,
                            y_test,
                            subset,
                            learner="knn3",
                            seed=seed,
                        )
                        rows.append(
                            {
                                "dataset": name,
                                "seed": int(seed),
                                "budget_per_class": int(budget),
                                "method": method,
                                "selector_detail": selector_detail,
                                "selected": int(len(subset)),
                                **metrics,
                            }
                        )
    finally:
        vacs_experiments.make_classifier = original_make_classifier

    meta = {
        "backbone_name": backbone_name,
        "backbone_class": model.__class__.__name__,
        "backbone_weights": weights.__class__.__name__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "datasets": dataset_meta,
        "seeds": seeds,
        "budgets_per_class": budgets,
        "methods": METHODS,
        "runtime_sec": float(time.time() - start),
        "embedding_runtime_sec": float(backbone_runtime_total),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "interpretation": (
            "Frozen torchvision image embeddings are reported as supplemental boundary evidence. "
            "The suite is intended to widen benchmark coverage beyond classical tabular and text "
            "features, not to establish a new state-of-the-art claim."
        ),
    }
    write_outputs(rows, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    run()
