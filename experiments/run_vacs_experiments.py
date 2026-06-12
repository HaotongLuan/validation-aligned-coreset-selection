import gc
import json
import os
import time
import warnings
from math import comb
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="MiniBatchKMeans is known to have a memory leak.*",
)

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, fetch_20newsgroups
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
DATA_HOME = ROOT / ".sklearn_data"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


METHODS = ["random", "herding", "kcenter", "boundary", "kmeans", "marc", "badge", "vacs"]
BASE_SELECTORS = ["random", "herding", "kcenter", "boundary", "kmeans", "marc"]
METHOD_NAMES = {
    "random": "Random",
    "herding": "Herding",
    "kcenter": "K-center",
    "boundary": "Boundary",
    "kmeans": "K-means",
    "marc": "MARC",
    "badge": "BADGE",
    "vacs": "VACS",
}
DATASET_NAMES = {
    "20newsgroups": "20 Newsgroups",
    "breast_cancer": "Breast cancer",
    "digits": "Digits",
    "iris": "Iris",
    "wine": "Wine",
}
DATASET_ORDER = ["20newsgroups", "breast_cancer", "digits", "iris", "wine"]
LOW_BUDGETS = [1, 2]
BUDGETS = [1, 2, 4, 8]
SEEDS = list(range(8))
ROBUSTNESS_LEARNERS = ["knn3", "logreg"]
VALIDATION_FRACTIONS = [0.20, 0.25, 0.33]
REPEATED_VALIDATION_REPEATS = 5
_BADGE_RANK_CACHE = {}


def l2_distances(X, Y):
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2 * X @ Y.T, 0.0)


def row_mean(X, indices, chunk_size=64):
    indices = np.asarray(indices, dtype=int)
    total = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        total += X[indices[start:start + chunk_size]].sum(axis=0, dtype=np.float64)
    return (total / max(len(indices), 1)).astype(X.dtype, copy=False)


def squared_distances_to_vector(X, indices, vector, chunk_size=64):
    indices = np.asarray(indices, dtype=int)
    distances = np.empty(len(indices), dtype=np.float64)
    vector = np.asarray(vector)
    vector_norm = float(np.dot(vector, vector))
    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start:start + chunk_size]
        chunk = X[chunk_idx]
        distances[start:start + len(chunk_idx)] = np.maximum(
            np.einsum("ij,ij->i", chunk, chunk) + vector_norm - 2.0 * (chunk @ vector),
            0.0,
        )
    return distances


def preprocess_dense(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    return X_train.astype(np.float32, copy=False), X_test.astype(np.float32, copy=False)


def preprocess_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        stop_words="english",
        norm="l2",
    )
    X_train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    X_test = vectorizer.transform(test_texts).toarray().astype(np.float32)
    return X_train, X_test


def load_datasets():
    datasets = []

    digits = load_digits()
    datasets.append(("digits", digits.data.astype(np.float32), digits.target, "dense"))

    iris = load_iris()
    datasets.append(("iris", iris.data.astype(np.float32), iris.target, "dense"))

    wine = load_wine()
    datasets.append(("wine", wine.data.astype(np.float32), wine.target, "dense"))

    cancer = load_breast_cancer()
    datasets.append(("breast_cancer", cancer.data.astype(np.float32), cancer.target, "dense"))

    try:
        train = fetch_20newsgroups(
            subset="train",
            remove=("headers", "footers", "quotes"),
            data_home=DATA_HOME,
        )
        test = fetch_20newsgroups(
            subset="test",
            remove=("headers", "footers", "quotes"),
            data_home=DATA_HOME,
        )
        X = list(train.data) + list(test.data)
        y = np.concatenate([train.target, test.target])
        datasets.append(("20newsgroups", X, y, "sparse_text"))
    except Exception as exc:
        print(f"[warn] skipping 20newsgroups: {exc}")

    return datasets


def per_class_budget(y, budget_per_class):
    labels, counts = np.unique(y, return_counts=True)
    return {label: min(int(budget_per_class), int(count)) for label, count in zip(labels, counts)}


def class_centroids(X, y):
    centroids = {}
    for label in np.unique(y):
        centroids[label] = X[y == label].mean(axis=0)
    return centroids


def margins_to_centroids(X, y):
    labels = np.unique(y)
    centroids = class_centroids(X, y)
    C = np.vstack([centroids[label] for label in labels])
    dists = l2_distances(X, C)
    own_pos = np.array([np.where(labels == label)[0][0] for label in y])
    own = dists[np.arange(len(y)), own_pos]
    dists[np.arange(len(y)), own_pos] = np.inf
    nearest_other = dists.min(axis=1)
    return nearest_other - own


def badge_order_from_embeddings(embeddings, class_indices, max_k):
    if max_k <= 0 or len(class_indices) == 0:
        return []
    max_k = min(int(max_k), len(class_indices))
    sq_norm = np.einsum("ij,ij->i", embeddings, embeddings)
    selected_positions = [int(np.argmax(sq_norm))]
    min_distances = None

    while len(selected_positions) < max_k:
        latest = selected_positions[-1]
        distances = np.maximum(
            sq_norm + sq_norm[latest] - 2.0 * (embeddings @ embeddings[latest]),
            0.0,
        )
        if min_distances is None:
            min_distances = distances
        else:
            min_distances = np.minimum(min_distances, distances)
        min_distances[selected_positions] = -np.inf
        next_pos = int(np.argmax(min_distances))
        selected_positions.append(next_pos)

    return [int(class_indices[pos]) for pos in selected_positions]


def badge_rankings(X, y, seed, max_k):
    cache_key = ("badge", id(X), id(y), tuple(X.shape), int(seed), int(max_k))
    if cache_key in _BADGE_RANK_CACHE:
        return _BADGE_RANK_CACHE[cache_key]

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X, y)
    probabilities = clf.predict_proba(X).astype(np.float32, copy=False)
    pseudo_labels = np.argmax(probabilities, axis=1)

    rankings = {}
    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        local_X = X[class_indices].astype(np.float32, copy=False)
        coefficients = probabilities[class_indices].copy()
        coefficients[np.arange(len(class_indices)), pseudo_labels[class_indices]] -= 1.0
        embeddings = (coefficients[:, :, None] * local_X[:, None, :]).reshape(
            len(class_indices), -1
        )
        rankings[label] = badge_order_from_embeddings(embeddings, class_indices, max_k)

    _BADGE_RANK_CACHE[cache_key] = rankings
    return rankings


def farthest_first(X, candidate_indices, k, initial_indices=None):
    candidate_indices = list(candidate_indices)
    if k <= 0 or not candidate_indices:
        return []

    selected = [] if initial_indices is None else list(initial_indices)
    selected_set = set(selected)
    remaining = [idx for idx in candidate_indices if idx not in selected_set]
    if not selected:
        centroid = row_mean(X, candidate_indices)
        distances = squared_distances_to_vector(X, candidate_indices, centroid)
        first = candidate_indices[int(np.argmin(distances))]
        selected.append(first)
        selected_set.add(first)
        remaining.remove(first)

    min_distances = None
    for initial_idx in selected[:-1]:
        distances = squared_distances_to_vector(X, remaining, X[initial_idx])
        min_distances = distances if min_distances is None else np.minimum(min_distances, distances)
    while len(selected) < k and remaining:
        distances = squared_distances_to_vector(X, remaining, X[selected[-1]])
        if min_distances is None:
            min_distances = distances
        else:
            min_distances = np.minimum(min_distances, distances)
        best_pos = int(np.argmax(min_distances))
        chosen = remaining[best_pos]
        selected.append(chosen)
        selected_set.add(chosen)
        remaining.pop(best_pos)
        if min_distances is not None:
            min_distances = np.delete(min_distances, best_pos)
    return selected[:k]


def stable_order(values, indices, reverse=False):
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    primary = -values if reverse else values
    return np.lexsort((indices, primary))


def stable_lloyd_medoids(X, class_indices, k, sample_weight=None, max_iter=50):
    if k <= 0:
        return []
    if k >= len(class_indices):
        return class_indices.tolist()

    initial = farthest_first(X, class_indices, k)
    local_X = np.asarray(X[class_indices], dtype=np.float64)
    centers = np.asarray(X[initial], dtype=np.float64)
    weights = (
        np.ones(len(class_indices), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )

    assignments = None
    for _ in range(max_iter):
        distances = l2_distances(local_X, centers)
        next_assignments = np.argmin(distances, axis=1)
        if assignments is not None and np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments

        next_centers = centers.copy()
        nearest_center_distance = distances.min(axis=1)
        for cluster_id in range(k):
            members = assignments == cluster_id
            if members.any():
                member_weights = weights[members]
                next_centers[cluster_id] = np.average(
                    local_X[members],
                    axis=0,
                    weights=member_weights,
                )
            else:
                order = stable_order(nearest_center_distance, class_indices, reverse=True)
                next_centers[cluster_id] = local_X[order[0]]
        if np.allclose(centers, next_centers, rtol=1e-10, atol=1e-12):
            centers = next_centers
            break
        centers = next_centers

    distances = l2_distances(centers, local_X)
    chosen = []
    used = set()
    for row in distances:
        for pos in stable_order(row, class_indices):
            idx = int(class_indices[int(pos)])
            if idx not in used:
                chosen.append(idx)
                used.add(idx)
                break

    if len(chosen) < k:
        chosen = farthest_first(X, class_indices, k, initial_indices=chosen)
    return chosen[:k]


def marc_select_class(X, class_indices, class_margins, k, seed):
    del seed
    if k <= 0:
        return []
    if k >= len(class_indices):
        return class_indices.tolist()
    if k == 1:
        centroid = X[class_indices].mean(axis=0, keepdims=True)
        distances = l2_distances(X[class_indices], centroid).ravel()
        return [int(class_indices[int(np.argmin(distances))])]

    scale = np.std(class_margins) + 1e-8
    z = (class_margins - np.median(class_margins)) / scale
    weights = 0.2 + 1.8 / (1.0 + np.exp(-z))
    return stable_lloyd_medoids(X, class_indices, k, sample_weight=weights)


def kmeans_select_class(X, class_indices, k, seed):
    del seed
    if k <= 0:
        return []
    if k >= len(class_indices):
        return class_indices.tolist()
    if k == 1:
        centroid = X[class_indices].mean(axis=0, keepdims=True)
        distances = l2_distances(X[class_indices], centroid).ravel()
        return [int(class_indices[int(np.argmin(distances))])]

    return stable_lloyd_medoids(X, class_indices, k)


def select_indices(X, y, budget_per_class, method, seed):
    rng = np.random.default_rng(seed)
    budgets = per_class_budget(y, budget_per_class)
    centroids = class_centroids(X, y)
    margins = margins_to_centroids(X, y)
    badge_orders = badge_rankings(X, y, seed, max(BUDGETS)) if method == "badge" else None
    selected = []

    for label in np.unique(y):
        class_indices = np.where(y == label)[0]
        k = budgets[label]
        if method == "random":
            selected.extend(rng.choice(class_indices, size=k, replace=False).tolist())
        elif method == "herding":
            centroid = centroids[label][None, :]
            distances = l2_distances(X[class_indices], centroid).ravel()
            selected.extend(class_indices[np.argsort(distances)[:k]].tolist())
        elif method == "kcenter":
            selected.extend(farthest_first(X, class_indices, k))
        elif method == "boundary":
            selected.extend(class_indices[np.argsort(margins[class_indices])[:k]].tolist())
        elif method == "kmeans":
            selected.extend(kmeans_select_class(X, class_indices, k, seed))
        elif method == "marc":
            selected.extend(marc_select_class(X, class_indices, margins[class_indices], k, seed))
        elif method == "badge":
            selected.extend(badge_orders[label][:k])
        else:
            raise ValueError(method)

    return np.array(sorted(set(selected)), dtype=int)


def make_classifier(learner, subset_size, seed):
    if learner == "knn3":
        n_neighbors = min(3, subset_size)
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="euclidean",
        )
    if learner == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            random_state=seed,
        )
    raise ValueError(learner)


def evaluate_subset(X_train, y_train, X_test, y_test, subset, learner="knn3", seed=0):
    clf = make_classifier(learner, len(subset), seed)
    clf.fit(X_train[subset], y_train[subset])
    pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
    }


def validation_aligned_select(
    X,
    y,
    budget_per_class,
    seed,
    selectors=None,
    rebuild=True,
    learner="knn3",
    collect_diagnostics=False,
    collect_timing=False,
    val_fraction=0.25,
    val_seed_offset=0,
):
    selectors = BASE_SELECTORS if selectors is None else list(selectors)
    indices = np.arange(len(y))
    pool_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=10_000 + 37 * seed + budget_per_class + 1009 * val_seed_offset,
        stratify=y,
    )

    best = None
    diagnostics = []
    candidate_selection_sec = 0.0
    candidate_evaluation_sec = 0.0
    validation_phase_start = time.perf_counter()
    for selector in selectors:
        candidate_start = time.perf_counter()
        local_subset = select_indices(X[pool_idx], y[pool_idx], budget_per_class, selector, seed)
        subset = pool_idx[local_subset]
        candidate_selection_sec += time.perf_counter() - candidate_start
        evaluation_start = time.perf_counter()
        metrics = evaluate_subset(X, y, X[val_idx], y[val_idx], subset, learner, seed)
        candidate_evaluation_sec += time.perf_counter() - evaluation_start
        score = (metrics["accuracy"], metrics["macro_f1"], -selectors.index(selector))
        if collect_diagnostics:
            diagnostics.append({
                "selector": selector,
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_macro_f1": float(metrics["macro_f1"]),
                "validation_selected": int(len(subset)),
            })
        if best is None or score > best["score"]:
            best = {"selector": selector, "score": score, "subset": subset}
    validation_phase_sec = time.perf_counter() - validation_phase_start

    final_rebuild_sec = 0.0
    if rebuild:
        rebuild_start = time.perf_counter()
        final_subset = select_indices(X, y, budget_per_class, best["selector"], seed)
        final_rebuild_sec = time.perf_counter() - rebuild_start
    else:
        final_subset = best["subset"]
    timing = {
        "candidate_selectors": int(len(selectors)),
        "validation_candidate_selection_sec": float(candidate_selection_sec),
        "validation_candidate_evaluation_sec": float(candidate_evaluation_sec),
        "validation_phase_sec": float(validation_phase_sec),
        "final_rebuild_sec": float(final_rebuild_sec),
        "vacs_selection_total_sec": float(validation_phase_sec + final_rebuild_sec),
        "rebuild": bool(rebuild),
        "validation_fraction": float(val_fraction),
    }
    if collect_diagnostics and collect_timing:
        return final_subset, best["selector"], diagnostics, timing
    if collect_diagnostics:
        return final_subset, best["selector"], diagnostics
    if collect_timing:
        return final_subset, best["selector"], timing
    return final_subset, best["selector"]


def repeated_validation_aligned_select(
    X,
    y,
    budget_per_class,
    seed,
    selectors=None,
    learner="knn3",
    val_fraction=0.25,
    repeats=REPEATED_VALIDATION_REPEATS,
):
    selectors = BASE_SELECTORS if selectors is None else list(selectors)
    scores = {
        selector: {
            "accuracy": [],
            "macro_f1": [],
            "validation_selected": [],
        }
        for selector in selectors
    }

    for repeat in range(repeats):
        indices = np.arange(len(y))
        pool_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=10_000 + 37 * seed + budget_per_class + 1009 * repeat,
            stratify=y,
        )
        for selector in selectors:
            local_subset = select_indices(X[pool_idx], y[pool_idx], budget_per_class, selector, seed)
            subset = pool_idx[local_subset]
            metrics = evaluate_subset(X, y, X[val_idx], y[val_idx], subset, learner, seed)
            scores[selector]["accuracy"].append(float(metrics["accuracy"]))
            scores[selector]["macro_f1"].append(float(metrics["macro_f1"]))
            scores[selector]["validation_selected"].append(int(len(subset)))

    best = None
    diagnostics = []
    for selector in selectors:
        mean_accuracy = float(np.mean(scores[selector]["accuracy"]))
        mean_macro_f1 = float(np.mean(scores[selector]["macro_f1"]))
        score = (mean_accuracy, mean_macro_f1, -selectors.index(selector))
        diagnostics.append({
            "selector": selector,
            "mean_validation_accuracy": mean_accuracy,
            "mean_validation_macro_f1": mean_macro_f1,
            "std_validation_accuracy": float(np.std(scores[selector]["accuracy"], ddof=0)),
            "validation_selected_mean": float(np.mean(scores[selector]["validation_selected"])),
            "repeats": int(repeats),
            "validation_fraction": float(val_fraction),
        })
        if best is None or score > best["score"]:
            best = {"selector": selector, "score": score}

    final_subset = select_indices(X, y, budget_per_class, best["selector"], seed)
    return final_subset, best["selector"], diagnostics


def run():
    start = time.time()
    rows = []
    ablation_rows = []
    downstream_rows = []
    validation_rows = []
    split_sensitivity_rows = []
    repeated_validation_rows = []
    repeated_validation_score_rows = []
    selection_cost_rows = []
    static_selection_cost_rows = []
    manifest = {
        "methods": METHODS,
        "ablation_methods": [
            "vacs",
            "vacs_no_rebuild",
            "vacs_no_kmeans",
            "vacs_no_marc",
            "vacs_no_random",
        ],
        "robustness_learners": ROBUSTNESS_LEARNERS,
        "validation_fractions": VALIDATION_FRACTIONS,
        "repeated_validation_repeats": REPEATED_VALIDATION_REPEATS,
        "budgets_per_class": BUDGETS,
        "seeds": SEEDS,
        "datasets": [],
    }

    for name, X, y, kind in load_datasets():
        manifest["datasets"].append({
            "name": name,
            "n": int(len(y)),
            "classes": int(len(np.unique(y))),
            "kind": kind,
        })
        for seed in SEEDS:
            gc.collect()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            if kind == "dense":
                X_train, X_test = preprocess_dense(X_train, X_test)
            elif kind == "sparse_text":
                X_train, X_test = preprocess_text(X_train, X_test)
            else:
                raise ValueError(f"unknown dataset kind: {kind}")

            for budget in BUDGETS:
                for method in METHODS:
                    selector_detail = method
                    if method == "vacs":
                        if budget in LOW_BUDGETS:
                            subset, selector_detail, validation_diag, timing_diag = validation_aligned_select(
                                X_train,
                                y_train,
                                budget,
                                seed,
                                collect_diagnostics=True,
                                collect_timing=True,
                            )
                        else:
                            subset, selector_detail = validation_aligned_select(
                                X_train, y_train, budget, seed
                            )
                    else:
                        subset = select_indices(X_train, y_train, budget, method, seed)
                    metrics = evaluate_subset(X_train, y_train, X_test, y_test, subset)
                    rows.append({
                        "dataset": name,
                        "seed": seed,
                        "budget_per_class": budget,
                        "learner": "knn3",
                        "method": method,
                        "selector_detail": selector_detail,
                        "selected": int(len(subset)),
                        **metrics,
                    })
                    if method == "vacs" and budget in LOW_BUDGETS:
                        for diag in validation_diag:
                            validation_rows.append({
                                "dataset": name,
                                "seed": seed,
                                "budget_per_class": budget,
                                "learner": "knn3",
                                **diag,
                            })
                        selection_cost_rows.append({
                            "dataset": name,
                            "seed": seed,
                            "budget_per_class": budget,
                            "learner": "knn3",
                            "selector_detail": selector_detail,
                            "selected": int(len(subset)),
                            **timing_diag,
                        })
                    if budget in LOW_BUDGETS:
                        downstream_rows.append({
                            "dataset": name,
                            "seed": seed,
                            "budget_per_class": budget,
                            "learner": "knn3",
                            "method": method,
                            "selector_detail": selector_detail,
                            "selected": int(len(subset)),
                            **metrics,
                        })
                    if method == "vacs" and budget in LOW_BUDGETS:
                        ablation_rows.append({
                            "dataset": name,
                            "seed": seed,
                            "budget_per_class": budget,
                            "ablation_method": "vacs",
                            "selector_detail": selector_detail,
                            "selected": int(len(subset)),
                            **metrics,
                        })

                if budget in LOW_BUDGETS:
                    for method in METHODS:
                        selector_detail = method
                        if method == "vacs":
                            subset, selector_detail = validation_aligned_select(
                                X_train, y_train, budget, seed, learner="logreg"
                            )
                        else:
                            selection_start = time.perf_counter()
                            subset = select_indices(X_train, y_train, budget, method, seed)
                            selection_sec = time.perf_counter() - selection_start
                            if budget in LOW_BUDGETS:
                                static_selection_cost_rows.append({
                                    "dataset": name,
                                    "seed": seed,
                                    "budget_per_class": budget,
                                    "learner": "knn3",
                                    "method": method,
                                    "selected": int(len(subset)),
                                    "selection_sec": float(selection_sec),
                                })
                        metrics = evaluate_subset(
                            X_train, y_train, X_test, y_test, subset, learner="logreg", seed=seed
                        )
                        downstream_rows.append({
                            "dataset": name,
                            "seed": seed,
                            "budget_per_class": budget,
                            "learner": "logreg",
                            "method": method,
                            "selector_detail": selector_detail,
                            "selected": int(len(subset)),
                            **metrics,
                        })

                    no_rebuild_subset, no_rebuild_selector = validation_aligned_select(
                        X_train, y_train, budget, seed, rebuild=False
                    )
                    no_rebuild_metrics = evaluate_subset(
                        X_train, y_train, X_test, y_test, no_rebuild_subset
                    )
                    ablation_rows.append({
                        "dataset": name,
                        "seed": seed,
                        "budget_per_class": budget,
                        "ablation_method": "vacs_no_rebuild",
                        "selector_detail": no_rebuild_selector,
                        "selected": int(len(no_rebuild_subset)),
                        **no_rebuild_metrics,
                    })

                    no_marc_subset, no_marc_selector = validation_aligned_select(
                        X_train,
                        y_train,
                        budget,
                        seed,
                        selectors=["random", "herding", "kcenter", "boundary", "kmeans"],
                    )
                    no_marc_metrics = evaluate_subset(
                        X_train, y_train, X_test, y_test, no_marc_subset
                    )
                    ablation_rows.append({
                        "dataset": name,
                        "seed": seed,
                        "budget_per_class": budget,
                        "ablation_method": "vacs_no_marc",
                        "selector_detail": no_marc_selector,
                        "selected": int(len(no_marc_subset)),
                        **no_marc_metrics,
                    })

                    no_kmeans_subset, no_kmeans_selector = validation_aligned_select(
                        X_train,
                        y_train,
                        budget,
                        seed,
                        selectors=["random", "herding", "kcenter", "boundary", "marc"],
                    )
                    no_kmeans_metrics = evaluate_subset(
                        X_train, y_train, X_test, y_test, no_kmeans_subset
                    )
                    ablation_rows.append({
                        "dataset": name,
                        "seed": seed,
                        "budget_per_class": budget,
                        "ablation_method": "vacs_no_kmeans",
                        "selector_detail": no_kmeans_selector,
                        "selected": int(len(no_kmeans_subset)),
                        **no_kmeans_metrics,
                    })

                    no_random_subset, no_random_selector = validation_aligned_select(
                        X_train,
                        y_train,
                        budget,
                        seed,
                        selectors=["herding", "kcenter", "boundary", "kmeans", "marc"],
                    )
                    no_random_metrics = evaluate_subset(
                        X_train, y_train, X_test, y_test, no_random_subset
                    )
                    ablation_rows.append({
                        "dataset": name,
                        "seed": seed,
                        "budget_per_class": budget,
                        "ablation_method": "vacs_no_random",
                        "selector_detail": no_random_selector,
                        "selected": int(len(no_random_subset)),
                        **no_random_metrics,
                    })

                    for val_fraction in VALIDATION_FRACTIONS:
                        sensitivity_subset, sensitivity_selector = validation_aligned_select(
                            X_train,
                            y_train,
                            budget,
                            seed,
                            val_fraction=val_fraction,
                        )
                        sensitivity_metrics = evaluate_subset(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            sensitivity_subset,
                        )
                        split_sensitivity_rows.append({
                            "dataset": name,
                            "seed": seed,
                            "budget_per_class": budget,
                            "validation_fraction": float(val_fraction),
                            "selector_detail": sensitivity_selector,
                            "selected": int(len(sensitivity_subset)),
                            **sensitivity_metrics,
                        })

                    repeated_subset, repeated_selector, repeated_diagnostics = repeated_validation_aligned_select(
                        X_train,
                        y_train,
                        budget,
                        seed,
                    )
                    repeated_metrics = evaluate_subset(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        repeated_subset,
                    )
                    repeated_validation_rows.append({
                        "dataset": name,
                        "seed": seed,
                        "budget_per_class": budget,
                        "validation_fraction": 0.25,
                        "repeats": REPEATED_VALIDATION_REPEATS,
                        "selector_detail": repeated_selector,
                        "selected": int(len(repeated_subset)),
                        **repeated_metrics,
                    })
                    for diag in repeated_diagnostics:
                        repeated_validation_score_rows.append({
                            "dataset": name,
                            "seed": seed,
                            "budget_per_class": budget,
                            **diag,
                        })

    df = pd.DataFrame(rows)
    ablation_df = pd.DataFrame(ablation_rows)
    downstream_df = pd.DataFrame(downstream_rows)
    validation_df = pd.DataFrame(validation_rows)
    split_sensitivity_df = pd.DataFrame(split_sensitivity_rows)
    repeated_validation_df = pd.DataFrame(repeated_validation_rows)
    repeated_validation_score_df = pd.DataFrame(repeated_validation_score_rows)
    selection_cost_df = pd.DataFrame(selection_cost_rows)
    static_selection_cost_df = pd.DataFrame(static_selection_cost_rows)
    df.to_csv(TABLES / "vacs_all_results.csv", index=False)
    ablation_df.to_csv(TABLES / "vacs_ablation_results.csv", index=False)
    downstream_df.to_csv(TABLES / "vacs_downstream_learner_results.csv", index=False)
    validation_df.to_csv(TABLES / "vacs_validation_scores.csv", index=False)
    split_sensitivity_df.to_csv(
        TABLES / "vacs_validation_split_sensitivity_results.csv",
        index=False,
    )
    repeated_validation_df.to_csv(
        TABLES / "vacs_repeated_validation_results.csv",
        index=False,
    )
    repeated_validation_score_df.to_csv(
        TABLES / "vacs_repeated_validation_scores.csv",
        index=False,
    )
    selection_cost_df.to_csv(TABLES / "vacs_selection_cost_results.csv", index=False)
    static_selection_cost_df.to_csv(TABLES / "vacs_static_selection_cost_results.csv", index=False)

    summary = (
        df.groupby(["dataset", "budget_per_class", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            selected_mean=("selected", "mean"),
        )
    )
    summary.to_csv(TABLES / "vacs_summary.csv", index=False)

    # Dataset-averaged rank and score summary.
    avg = (
        summary.groupby(["budget_per_class", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy_mean", "mean"),
            macro_f1_mean=("macro_f1_mean", "mean"),
        )
    )
    avg.to_csv(TABLES / "vacs_budget_average.csv", index=False)

    # Low-budget headline table: budgets 1 and 2.
    low = df[df["budget_per_class"].isin([1, 2])]
    low_summary = (
        low.groupby("method", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )
    low_summary.to_csv(TABLES / "vacs_low_budget_summary.csv", index=False)

    selector_mix = (
        df[df["method"] == "vacs"]
        .groupby(["budget_per_class", "selector_detail"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    selector_mix.to_csv(TABLES / "vacs_selector_mix.csv", index=False)

    best_static_row = low_summary[~low_summary["method"].isin(["vacs"])].iloc[0]
    best_static = best_static_row["accuracy_mean"]
    vacs_acc = low_summary[low_summary["method"] == "vacs"]["accuracy_mean"].iloc[0]
    manifest["headline"] = {
        "vacs_low_budget_accuracy": float(vacs_acc),
        "best_static_method": str(best_static_row["method"]),
        "best_static_low_budget_accuracy": float(best_static),
        "absolute_gain": float(vacs_acc - best_static),
        "relative_error_reduction": float(((1 - best_static) - (1 - vacs_acc)) / (1 - best_static)),
    }
    manifest["runtime_sec"] = time.time() - start

    # LaTeX tables and diagnostic summaries.
    write_latex_tables(
        df,
        low_summary,
        avg,
        summary,
        ablation_df,
        downstream_df,
        validation_df,
        split_sensitivity_df,
        repeated_validation_df,
        selection_cost_df,
        static_selection_cost_df,
        manifest,
    )
    (RESULTS / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_figures(avg, df)
    print(json.dumps(manifest, indent=2))


def pct(x):
    return f"{100*x:.1f}"


def pct2(x):
    return f"{100*x:.2f}"


def fmt_signed(x):
    return f"{100*x:+.2f}"


def fmt_sec(x):
    return f"{x:.3f}"


def bootstrap_mean_ci(values, seed=20260531, reps=20000):
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def sign_test_p_value(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    cumulative = sum(comb(n, i) for i in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * cumulative)


def fmt_p_value(p_value):
    if p_value < 0.001:
        return "$<.001$"
    return f"{p_value:.3f}"


def write_stability_table(df, comparator):
    paired = (
        df[df["method"].isin(["vacs", comparator]) & df["budget_per_class"].isin([1, 2])]
        .pivot_table(
            index=["dataset", "seed", "budget_per_class"],
            columns="method",
            values="accuracy",
        )
        .reset_index()
    )
    paired["delta"] = paired["vacs"] - paired[comparator]

    slice_rows = [
        ("Pooled", paired),
        ("20NG", paired[paired["dataset"] == "20newsgroups"]),
        ("Breast cancer", paired[paired["dataset"] == "breast_cancer"]),
        ("Digits", paired[paired["dataset"] == "digits"]),
        ("Iris", paired[paired["dataset"] == "iris"]),
        ("Wine", paired[paired["dataset"] == "wine"]),
    ]

    rows = []
    for label, sub in slice_rows:
        delta = sub["delta"].to_numpy()
        lo, hi = bootstrap_mean_ci(delta)
        wins = int((delta > 1e-12).sum())
        ties = int((np.abs(delta) <= 1e-12).sum())
        losses = int((delta < -1e-12).sum())
        p_value = sign_test_p_value(wins, losses)
        rows.append(
            {
                "slice": label,
                "n": int(len(delta)),
                "delta": fmt_signed(delta.mean()),
                "ci": f"[{fmt_signed(lo)},{fmt_signed(hi)}]",
                "wtl": f"{wins}/{ties}/{losses}",
                "sign_p": p_value,
                "sign_p_tex": fmt_p_value(p_value),
            }
        )

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Slice & $n$ & $\\Delta$ Acc. & 95\\% CI & W/T/L & Sign $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['slice']} & {row['n']} & {row['delta']} & {row['ci']} & "
            f"{row['wtl']} & {row['sign_p_tex']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_stability_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return rows


def write_seed_block_stability_table(df, comparator):
    paired = (
        df[
            df["method"].isin(["vacs", comparator])
            & (df["learner"] == "knn3")
            & df["budget_per_class"].isin(LOW_BUDGETS)
        ]
        .pivot_table(
            index=["seed", "dataset", "budget_per_class"],
            columns="method",
            values="accuracy",
        )
        .reset_index()
    )
    paired["delta"] = paired["vacs"] - paired[comparator]

    by_seed = (
        paired.groupby("seed", as_index=False)
        .agg(
            units=("delta", "size"),
            vacs_accuracy=("vacs", "mean"),
            comparator_accuracy=(comparator, "mean"),
            delta=("delta", "mean"),
        )
        .sort_values("seed")
    )
    by_seed.to_csv(TABLES / "vacs_seed_block_stability.csv", index=False)

    values = by_seed["delta"].to_numpy()
    lo, hi = bootstrap_mean_ci(values)
    summary = {
        "comparator": comparator,
        "blocks": int(len(values)),
        "units_per_block": int(by_seed["units"].iloc[0]),
        "delta": float(values.mean()),
        "ci": f"[{fmt_signed(lo)},{fmt_signed(hi)}]",
        "positive_blocks": int((values > 1e-12).sum()),
        "tied_blocks": int((np.abs(values) <= 1e-12).sum()),
        "negative_blocks": int((values < -1e-12).sum()),
        "min_delta": float(values.min()),
        "max_delta": float(values.max()),
    }
    pd.DataFrame([summary]).to_csv(
        TABLES / "vacs_seed_block_stability_summary.csv", index=False
    )

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Blocks & $\\Delta$ Acc. & 95\\% CI & Pos./Tie/Neg & Range \\\\",
        "\\midrule",
        (
            f"{summary['blocks']} seeds & {fmt_signed(summary['delta'])} & {summary['ci']} & "
            f"{summary['positive_blocks']}/{summary['tied_blocks']}/{summary['negative_blocks']} & "
            f"{fmt_signed(summary['min_delta'])}--{fmt_signed(summary['max_delta'])} \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    (TABLES / "vacs_seed_block_stability_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def write_oracle_regret_summary(df):
    base = [method for method in METHODS if method != "vacs"]
    oracle = (
        df[df["method"].isin(base)]
        .groupby(["dataset", "seed", "budget_per_class"], as_index=False)["accuracy"]
        .max()
        .rename(columns={"accuracy": "oracle_accuracy"})
    )
    merged = df.merge(oracle, on=["dataset", "seed", "budget_per_class"], how="left")
    merged["regret"] = merged["oracle_accuracy"] - merged["accuracy"]
    merged["match"] = np.abs(merged["regret"]) <= 1e-12
    merged["severe"] = merged["regret"] > 0.05

    low = merged[merged["budget_per_class"].isin([1, 2])]
    rows = []
    for method in METHODS:
        sub = low[low["method"] == method]
        rows.append(
            {
                "method": method,
                "regret": float(sub["regret"].mean()),
                "match_rate": float(sub["match"].mean()),
                "severe_rate": float(sub["severe"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "vacs_oracle_regret.csv", index=False)

    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Regret & Match & Severe \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{METHOD_NAMES[row['method']]} & {pct2(row['regret'])} & "
            f"{pct(row['match_rate'])} & {pct(row['severe_rate'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_oracle_regret_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return rows


def write_ablation_table(ablation_df):
    names = {
        "vacs": "Full VACS",
        "vacs_no_rebuild": "No final rebuild",
        "vacs_no_kmeans": "No K-means candidate",
        "vacs_no_marc": "No MARC candidate",
        "vacs_no_random": "No random candidate",
    }
    full = (
        ablation_df[ablation_df["ablation_method"] == "vacs"]
        [["dataset", "seed", "budget_per_class", "accuracy"]]
        .rename(columns={"accuracy": "full_accuracy"})
    )
    merged = ablation_df.merge(
        full, on=["dataset", "seed", "budget_per_class"], how="left"
    )
    merged["delta_vs_full"] = merged["accuracy"] - merged["full_accuracy"]

    rows = []
    for method in ["vacs", "vacs_no_rebuild", "vacs_no_kmeans", "vacs_no_marc", "vacs_no_random"]:
        sub = merged[merged["ablation_method"] == method]
        rows.append(
            {
                "ablation_method": method,
                "accuracy_mean": float(sub["accuracy"].mean()),
                "delta_vs_full": float(sub["delta_vs_full"].mean()),
                "macro_f1_mean": float(sub["macro_f1"].mean()),
                "selected_mean": float(sub["selected"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "vacs_ablation_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & Acc. & $\\Delta$ & Macro-F1 & Selected \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{names[row['ablation_method']]} & {pct(row['accuracy_mean'])} & "
            f"{fmt_signed(row['delta_vs_full'])} & {pct(row['macro_f1_mean'])} & "
            f"{row['selected_mean']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return rows


def write_leave_one_dataset_out_table(df):
    low = df[df["budget_per_class"].isin(LOW_BUDGETS)]
    datasets = sorted(low["dataset"].unique())
    static_methods = [method for method in METHODS if method != "vacs"]

    rows = []
    for held_out in ["None"] + datasets:
        sub = low if held_out == "None" else low[low["dataset"] != held_out]
        means = sub.groupby("method")["accuracy"].mean()
        best_static = max(static_methods, key=lambda method: means[method])
        rows.append(
            {
                "held_out": held_out,
                "units": int(len(sub[["dataset", "seed", "budget_per_class"]].drop_duplicates())),
                "vacs_accuracy": float(means["vacs"]),
                "best_static_method": best_static,
                "best_static_accuracy": float(means[best_static]),
                "gain": float(means["vacs"] - means[best_static]),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "vacs_leave_one_dataset_out.csv", index=False)

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Held out & Units & VACS & Best static & Gain \\\\",
        "\\midrule",
    ]
    for row in rows:
        held_out = "None" if row["held_out"] == "None" else row["held_out"].replace("_", "\\_")
        best = f"{METHOD_NAMES[row['best_static_method']]} {pct(row['best_static_accuracy'])}"
        lines.append(
            f"{held_out} & {row['units']} & {pct(row['vacs_accuracy'])} & "
            f"{best} & {fmt_signed(row['gain'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_leave_one_dataset_out_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def write_dataset_low_budget_table(df):
    low = df[(df["learner"] == "knn3") & df["budget_per_class"].isin(LOW_BUDGETS)]
    static_methods = [method for method in METHODS if method != "vacs"]

    rows = []
    for dataset in sorted(low["dataset"].unique()):
        sub = low[low["dataset"] == dataset]
        means = sub.groupby("method")["accuracy"].mean()
        best_static = max(static_methods, key=lambda method: means[method])
        vacs_rows = sub[sub["method"] == "vacs"]
        selector_counts = vacs_rows["selector_detail"].value_counts()
        top_selector = str(selector_counts.index[0])
        top_count = int(selector_counts.iloc[0])
        units = int(len(vacs_rows[["seed", "budget_per_class"]].drop_duplicates()))
        rows.append({
            "dataset": dataset,
            "units": units,
            "vacs_accuracy": float(means["vacs"]),
            "best_static_method": best_static,
            "best_static_accuracy": float(means[best_static]),
            "gain": float(means["vacs"] - means[best_static]),
            "top_selector": top_selector,
            "top_selector_count": top_count,
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "vacs_dataset_low_budget_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & VACS & Best static & Gain & Top VACS choice \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = row["dataset"].replace("_", "\\_")
        best = f"{METHOD_NAMES[row['best_static_method']]} {pct(row['best_static_accuracy'])}"
        top_choice = (
            f"{METHOD_NAMES[row['top_selector']]} "
            f"{row['top_selector_count']}/{row['units']}"
        )
        lines.append(
            f"{dataset} & {pct(row['vacs_accuracy'])} & {best} & "
            f"{fmt_signed(row['gain'])} & {top_choice} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_dataset_low_budget_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def write_full20news_stress_table(df):
    low = df[
        (df["dataset"] == "20newsgroups")
        & (df["learner"] == "knn3")
        & df["budget_per_class"].isin(LOW_BUDGETS)
    ].copy()
    if low.empty:
        return []

    low.to_csv(TABLES / "vacs_full20news_stress_results.csv", index=False)

    static_methods = [method for method in METHODS if method != "vacs"]
    summary = (
        low.groupby("method", as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .rename(columns={"accuracy": "accuracy_mean", "macro_f1": "macro_f1_mean"})
        .sort_values("accuracy_mean", ascending=False)
    )
    summary.to_csv(TABLES / "vacs_full20news_stress_summary.csv", index=False)

    means = summary.set_index("method")
    best_static = max(static_methods, key=lambda method: float(means.loc[method, "accuracy_mean"]))
    vacs_rows = low[low["method"] == "vacs"]
    selector_counts = vacs_rows["selector_detail"].value_counts()
    selector_mix = {
        selector: int(selector_counts.get(selector, 0))
        for selector in BASE_SELECTORS
    }
    top_selector = max(BASE_SELECTORS, key=lambda selector: selector_mix[selector])
    units = int(len(vacs_rows[["seed", "budget_per_class"]].drop_duplicates()))
    row = {
        "dataset": "20newsgroups",
        "display_name": "20 Newsgroups (20-class)",
        "classes": 20,
        "methods": ",".join(METHODS),
        "units": units,
        "vacs_accuracy": float(means.loc["vacs", "accuracy_mean"]),
        "best_static_method": best_static,
        "best_static_accuracy": float(means.loc[best_static, "accuracy_mean"]),
        "gain": float(
            means.loc["vacs", "accuracy_mean"] - means.loc[best_static, "accuracy_mean"]
        ),
        "relative_error_reduction": float(
            (
                (1.0 - means.loc[best_static, "accuracy_mean"])
                - (1.0 - means.loc["vacs", "accuracy_mean"])
            )
            / (1.0 - means.loc[best_static, "accuracy_mean"])
        ),
        "top_selector": top_selector,
        "top_selector_count": selector_mix[top_selector],
        "selector_mix": json.dumps(selector_mix, sort_keys=True),
        "seed_blocks": int(vacs_rows["seed"].nunique()),
    }
    pd.DataFrame([row]).to_csv(TABLES / "vacs_full20news_stress_summary_row.csv", index=False)

    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Task & Classes & Units & VACS & Best static & Gain & Top VACS choice \\\\",
        "\\midrule",
    ]
    best = f"{METHOD_NAMES[row['best_static_method']]} {pct(row['best_static_accuracy'])}"
    top_choice = (
        f"{METHOD_NAMES[row['top_selector']]} "
        f"{row['top_selector_count']}/{row['units']}"
    )
    lines.append(
        f"{row['display_name']} & {row['classes']} & {row['units']} & "
        f"{pct(row['vacs_accuracy'])} & {best} & {fmt_signed(row['gain'])} & "
        f"{top_choice} \\\\"
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_full20news_stress_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return [row]


def write_selector_mix_table(df):
    low_vacs = df[
        (df["learner"] == "knn3")
        & (df["method"] == "vacs")
        & (df["budget_per_class"].isin(LOW_BUDGETS))
    ]
    chosen_selectors = BASE_SELECTORS
    counts = (
        low_vacs.groupby(["dataset", "selector_detail"])
        .size()
        .unstack(fill_value=0)
    )
    for selector in chosen_selectors:
        if selector not in counts.columns:
            counts[selector] = 0
    counts = counts[chosen_selectors]

    rows = []
    ordered_datasets = [
        dataset for dataset in DATASET_ORDER if dataset in set(counts.index)
    ] + [
        dataset for dataset in sorted(counts.index) if dataset not in DATASET_ORDER
    ]
    for dataset in ordered_datasets:
        row = {selector: int(counts.loc[dataset, selector]) for selector in chosen_selectors}
        row["dataset"] = dataset
        row["units"] = int(sum(row[selector] for selector in chosen_selectors))
        rows.append(row)

    pooled = {selector: int(counts[selector].sum()) for selector in chosen_selectors}
    pooled["dataset"] = "Pooled"
    pooled["units"] = int(sum(pooled[selector] for selector in chosen_selectors))
    rows.append(pooled)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "vacs_selector_mix_by_dataset.csv", index=False)

    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Dataset & Units & Random & Herding & K-center & Boundary & K-means & MARC \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = DATASET_NAMES.get(row["dataset"], row["dataset"].replace("_", "\\_"))
        if row["dataset"] == "Pooled":
            dataset = "\\textbf{Pooled}"
        lines.append(
            f"{dataset} & {row['units']} & {row['random']} & {row['herding']} & "
            f"{row['kcenter']} & {row['boundary']} & {row['kmeans']} & {row['marc']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_selector_mix_by_dataset_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def write_downstream_learner_table(downstream_df):
    static_methods = [method for method in METHODS if method != "vacs"]
    learner_names = {
        "knn3": "3-NN",
        "logreg": "LogReg",
    }

    rows = []
    for learner in ROBUSTNESS_LEARNERS:
        sub = downstream_df[downstream_df["learner"] == learner]
        means = sub.groupby("method").agg(
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
        )
        best_static = max(static_methods, key=lambda method: means.loc[method, "accuracy_mean"])
        rows.append({
            "learner": learner,
            "vacs_accuracy": float(means.loc["vacs", "accuracy_mean"]),
            "vacs_macro_f1": float(means.loc["vacs", "macro_f1_mean"]),
            "best_static_method": best_static,
            "best_static_accuracy": float(means.loc[best_static, "accuracy_mean"]),
            "gain": float(
                means.loc["vacs", "accuracy_mean"] - means.loc[best_static, "accuracy_mean"]
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "vacs_downstream_learner_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Learner & VACS & Best static & Gain & VACS F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        best = f"{METHOD_NAMES[row['best_static_method']]} {pct(row['best_static_accuracy'])}"
        lines.append(
            f"{learner_names[row['learner']]} & {pct(row['vacs_accuracy'])} & "
            f"{best} & {fmt_signed(row['gain'])} & {pct(row['vacs_macro_f1'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_downstream_learner_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return rows


def write_validation_reliability_table(validation_df, df):
    test_df = (
        df[(df["learner"] == "knn3") & df["method"].isin(BASE_SELECTORS) & df["budget_per_class"].isin(LOW_BUDGETS)]
        [["dataset", "seed", "budget_per_class", "method", "accuracy", "macro_f1"]]
        .rename(columns={"accuracy": "test_accuracy", "macro_f1": "test_macro_f1"})
    )
    merged = validation_df.rename(columns={"selector": "method"}).merge(
        test_df,
        on=["dataset", "seed", "budget_per_class", "method"],
        how="left",
    )

    order_map = {method: i for i, method in enumerate(BASE_SELECTORS)}
    slice_specs = [
        ("Pooled", None),
        ("20NG", "20newsgroups"),
        ("Breast cancer", "breast_cancer"),
        ("Digits", "digits"),
        ("Iris", "iris"),
        ("Wine", "wine"),
    ]

    detail_rows = []
    summary_rows = []
    for label, dataset_name in slice_specs:
        sub = merged if dataset_name is None else merged[merged["dataset"] == dataset_name]
        if sub.empty:
            continue

        unit_rows = []
        for (dataset, seed, budget), unit in sub.groupby(["dataset", "seed", "budget_per_class"]):
            unit = unit.copy()
            unit["selector_order"] = unit["method"].map(order_map)
            val_sorted = unit.sort_values(
                ["validation_accuracy", "validation_macro_f1", "selector_order"],
                ascending=[False, False, True],
            )
            test_sorted = unit.sort_values(
                ["test_accuracy", "test_macro_f1", "selector_order"],
                ascending=[False, False, True],
            )
            val_order = {method: rank + 1 for rank, method in enumerate(val_sorted["method"].tolist())}
            test_order = {method: rank + 1 for rank, method in enumerate(test_sorted["method"].tolist())}
            val_ranks = np.array([val_order[m] for m in BASE_SELECTORS], dtype=float)
            test_ranks = np.array([test_order[m] for m in BASE_SELECTORS], dtype=float)
            rho = 1.0 if np.std(val_ranks) == 0.0 or np.std(test_ranks) == 0.0 else float(np.corrcoef(val_ranks, test_ranks)[0, 1])
            validation_top = val_sorted.iloc[0]["method"]
            test_top = test_sorted.iloc[0]["method"]
            regret = float(test_sorted.iloc[0]["test_accuracy"] - unit.loc[unit["method"] == validation_top, "test_accuracy"].iloc[0])
            severe = regret > 0.05

            detail_rows.append({
                "slice": label,
                "dataset": dataset,
                "seed": int(seed),
                "budget_per_class": int(budget),
                "validation_top": validation_top,
                "test_top": test_top,
                "top1_agree": bool(validation_top == test_top),
                "rank_correlation": rho,
                "regret": regret,
                "severe": severe,
            })
            unit_rows.append({
                "top1_agree": bool(validation_top == test_top),
                "rank_correlation": rho,
                "regret": regret,
                "severe": severe,
            })

        unit_df = pd.DataFrame(unit_rows)
        summary_rows.append({
            "slice": label,
            "n": int(len(unit_df)),
            "top1_agreement": float(unit_df["top1_agree"].mean()),
            "rank_correlation": float(unit_df["rank_correlation"].mean()),
            "regret": float(unit_df["regret"].mean()),
            "severe_rate": float(unit_df["severe"].mean()),
        })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(TABLES / "vacs_validation_reliability_results.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES / "vacs_validation_reliability_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Slice & $n$ & Top-1 & $\\rho$ & Regret & Severe \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['slice']} & {int(row['n'])} & {pct(row['top1_agreement'])} & "
            f"{row['rank_correlation']:.2f} & {pct2(row['regret'])} & {pct(row['severe_rate'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_validation_reliability_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return summary_df.to_dict(orient="records")


def write_validation_gap_table(validation_df):
    order_map = {method: i for i, method in enumerate(BASE_SELECTORS)}
    slice_specs = [
        ("Pooled", None),
        ("20NG", "20newsgroups"),
        ("Breast cancer", "breast_cancer"),
        ("Digits", "digits"),
        ("Iris", "iris"),
        ("Wine", "wine"),
    ]

    detail_rows = []
    summary_rows = []
    for label, dataset_name in slice_specs:
        sub = validation_df if dataset_name is None else validation_df[validation_df["dataset"] == dataset_name]
        if sub.empty:
            continue

        unit_rows = []
        for (dataset, seed, budget), unit in sub.groupby(["dataset", "seed", "budget_per_class"]):
            unit = unit.copy()
            unit["selector_order"] = unit["selector"].map(order_map)
            sorted_unit = unit.sort_values(
                ["validation_accuracy", "validation_macro_f1", "selector_order"],
                ascending=[False, False, True],
            )
            top = sorted_unit.iloc[0]
            runner_up = sorted_unit.iloc[1]
            gap = float(top["validation_accuracy"] - runner_up["validation_accuracy"])
            low_gap = gap < 0.02
            very_low_gap = gap < 0.05

            detail_rows.append({
                "slice": label,
                "dataset": dataset,
                "seed": int(seed),
                "budget_per_class": int(budget),
                "validation_top": top["selector"],
                "validation_runner_up": runner_up["selector"],
                "gap": gap,
                "low_gap": low_gap,
                "very_low_gap": very_low_gap,
            })
            unit_rows.append({
                "gap": gap,
                "low_gap": low_gap,
                "very_low_gap": very_low_gap,
            })

        unit_df = pd.DataFrame(unit_rows)
        summary_rows.append({
            "slice": label,
            "n": int(len(unit_df)),
            "mean_gap": float(unit_df["gap"].mean()),
            "median_gap": float(unit_df["gap"].median()),
            "low_gap_rate": float(unit_df["low_gap"].mean()),
            "very_low_gap_rate": float(unit_df["very_low_gap"].mean()),
        })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(TABLES / "vacs_validation_gap_results.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES / "vacs_validation_gap_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Slice & $n$ & Mean gap & Median gap & $<2$ pt & $<5$ pt \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['slice']} & {int(row['n'])} & {pct2(row['mean_gap'])} & "
            f"{pct2(row['median_gap'])} & {pct(row['low_gap_rate'])} & {pct(row['very_low_gap_rate'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_validation_gap_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return summary_df.to_dict(orient="records")


def write_validation_gap_fallback_table(validation_df, df):
    test_df = (
        df[(df["learner"] == "knn3") & df["method"].isin(BASE_SELECTORS) & df["budget_per_class"].isin(LOW_BUDGETS)]
        [["dataset", "seed", "budget_per_class", "method", "accuracy", "macro_f1"]]
        .rename(columns={"accuracy": "test_accuracy", "macro_f1": "test_macro_f1"})
    )
    merged = validation_df.rename(columns={"selector": "method"}).merge(
        test_df,
        on=["dataset", "seed", "budget_per_class", "method"],
        how="left",
    )

    order_map = {method: i for i, method in enumerate(BASE_SELECTORS)}
    detail_rows = []
    for (dataset, seed, budget), unit in merged.groupby(["dataset", "seed", "budget_per_class"]):
        unit = unit.copy()
        unit["selector_order"] = unit["method"].map(order_map)
        sorted_unit = unit.sort_values(
            ["validation_accuracy", "validation_macro_f1", "selector_order"],
            ascending=[False, False, True],
        )
        top = sorted_unit.iloc[0]
        runner_up = sorted_unit.iloc[1]
        gap = float(top["validation_accuracy"] - runner_up["validation_accuracy"])
        herding_acc = float(unit.loc[unit["method"] == "herding", "test_accuracy"].iloc[0])
        detail_rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "budget_per_class": int(budget),
            "is_dense": bool(dataset != "20newsgroups"),
            "gap": gap,
            "vacs_accuracy": float(top["test_accuracy"]),
            "herding_accuracy": herding_acc,
        })

    detail_df = pd.DataFrame(detail_rows)
    vacs_mean = float(detail_df["vacs_accuracy"].mean())
    variant_specs = [
        ("VACS", None),
        ("Static herding", detail_df["gap"] >= 0.0),
        ("All gap < 2pt -> herding", detail_df["gap"] < 0.02),
        ("Dense gap < 0.5pt -> herding", detail_df["is_dense"] & (detail_df["gap"] < 0.005)),
        ("Dense gap < 2pt -> herding", detail_df["is_dense"] & (detail_df["gap"] < 0.02)),
    ]

    summary_rows = []
    for name, mask in variant_specs:
        if mask is None:
            chosen = detail_df["vacs_accuracy"].to_numpy()
            fallback_count = 0
        else:
            chosen = np.where(mask, detail_df["herding_accuracy"], detail_df["vacs_accuracy"])
            fallback_count = int(mask.sum())
        summary_rows.append({
            "variant": name,
            "fallback_count": fallback_count,
            "accuracy": float(chosen.mean()),
            "delta_vs_vacs": float(chosen.mean() - vacs_mean),
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df.to_csv(TABLES / "vacs_validation_gap_fallback_results.csv", index=False)
    summary_df.to_csv(TABLES / "vacs_validation_gap_fallback_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Variant & Fallbacks & Acc. & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['variant']} & {int(row['fallback_count'])} & "
            f"{pct(row['accuracy'])} & {fmt_signed(row['delta_vs_vacs'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_validation_gap_fallback_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return summary_df.to_dict(orient="records")


def write_selection_cost_table(selection_cost_df, static_selection_cost_df, reference_method):
    slice_specs = [
        ("Pooled", None),
        ("20NG", "20newsgroups"),
        ("Breast cancer", "breast_cancer"),
        ("Digits", "digits"),
        ("Iris", "iris"),
        ("Wine", "wine"),
    ]
    summary_rows = []
    for label, dataset_name in slice_specs:
        sub = selection_cost_df if dataset_name is None else selection_cost_df[selection_cost_df["dataset"] == dataset_name]
        static_sub = static_selection_cost_df
        if dataset_name is not None:
            static_sub = static_sub[static_sub["dataset"] == dataset_name]
        static_sub = static_sub[static_sub["method"] == reference_method]
        if sub.empty:
            continue
        reference_sec = float(static_sub["selection_sec"].sum()) if not static_sub.empty else 0.0
        validation_select = float(sub["validation_candidate_selection_sec"].sum())
        validation_score = float(sub["validation_candidate_evaluation_sec"].sum())
        validation_phase = float(sub["validation_phase_sec"].sum())
        rebuild = float(sub["final_rebuild_sec"].sum())
        total = float(sub["vacs_selection_total_sec"].sum())
        summary_rows.append({
            "slice": label,
            "n": int(len(sub)),
            "reference_method": reference_method,
            "reference_selection_sec": reference_sec,
            "candidate_selectors_mean": float(sub["candidate_selectors"].mean()),
            "validation_candidate_selection_sec": validation_select,
            "validation_candidate_evaluation_sec": validation_score,
            "validation_phase_sec": validation_phase,
            "final_rebuild_sec": rebuild,
            "vacs_selection_total_sec": total,
            "absolute_overhead_vs_reference_sec": float(total - reference_sec),
            "validation_share": float(validation_phase / total) if total > 0 else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES / "vacs_selection_cost_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        f"Slice & $n$ & {METHOD_NAMES.get(reference_method, reference_method)} & VACS val. & Rebuild & Total \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['slice']} & {int(row['n'])} & "
            f"{fmt_sec(row['reference_selection_sec'])} & "
            f"{fmt_sec(row['validation_phase_sec'])} & "
            f"{fmt_sec(row['final_rebuild_sec'])} & "
            f"{fmt_sec(row['vacs_selection_total_sec'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_selection_cost_table.tex").write_text("\n".join(lines), encoding="utf-8")
    return summary_df.to_dict(orient="records")


def write_validation_split_sensitivity_table(split_sensitivity_df, df):
    low = df[(df["learner"] == "knn3") & df["budget_per_class"].isin(LOW_BUDGETS)]
    static_methods = [method for method in METHODS if method != "vacs"]
    static_means = (
        low[low["method"].isin(static_methods)]
        .groupby("method")["accuracy"]
        .mean()
    )
    best_static = static_means.idxmax()
    best_static_accuracy = float(static_means.loc[best_static])

    key_cols = ["dataset", "seed", "budget_per_class"]
    default = (
        split_sensitivity_df[np.isclose(split_sensitivity_df["validation_fraction"], 0.25)]
        [key_cols + ["selector_detail"]]
        .rename(columns={"selector_detail": "default_selector"})
    )
    merged = split_sensitivity_df.merge(default, on=key_cols, how="left")
    merged["selector_agree_default"] = (
        merged["selector_detail"] == merged["default_selector"]
    )

    rows = []
    for val_fraction, sub in merged.groupby("validation_fraction", sort=True):
        vacs_accuracy = float(sub["accuracy"].mean())
        rows.append({
            "validation_fraction": float(val_fraction),
            "units": int(len(sub)),
            "vacs_accuracy": vacs_accuracy,
            "best_static_method": str(best_static),
            "best_static_accuracy": best_static_accuracy,
            "gain": float(vacs_accuracy - best_static_accuracy),
            "selector_agreement": float(sub["selector_agree_default"].mean()),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        TABLES / "vacs_validation_split_sensitivity_summary.csv",
        index=False,
    )

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Val. frac. & Units & VACS & Best static & Gain & Agree \\\\",
        "\\midrule",
    ]
    for row in rows:
        best = f"{METHOD_NAMES[row['best_static_method']]} {pct(row['best_static_accuracy'])}"
        lines.append(
            f"{row['validation_fraction']:.2f} & {row['units']} & "
            f"{pct(row['vacs_accuracy'])} & {best} & "
            f"{fmt_signed(row['gain'])} & {pct(row['selector_agreement'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_validation_split_sensitivity_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return summary_df.to_dict(orient="records")


def write_repeated_validation_table(repeated_validation_df, df):
    low = df[(df["learner"] == "knn3") & df["budget_per_class"].isin(LOW_BUDGETS)]
    static_methods = [method for method in METHODS if method != "vacs"]
    static_means = (
        low[low["method"].isin(static_methods)]
        .groupby("method")["accuracy"]
        .mean()
    )
    best_static = static_means.idxmax()
    best_static_accuracy = float(static_means.loc[best_static])

    key_cols = ["dataset", "seed", "budget_per_class"]
    default = (
        low[low["method"] == "vacs"][key_cols + ["selector_detail"]]
        .rename(columns={"selector_detail": "default_selector"})
    )
    merged = repeated_validation_df.merge(default, on=key_cols, how="left")
    merged["selector_agree_default"] = (
        merged["selector_detail"] == merged["default_selector"]
    )

    slice_specs = [
        ("Pooled", None),
        ("20NG", "20newsgroups"),
        ("Breast cancer", "breast_cancer"),
        ("Digits", "digits"),
        ("Iris", "iris"),
        ("Wine", "wine"),
    ]

    summary_rows = []
    for label, dataset_name in slice_specs:
        sub = merged if dataset_name is None else merged[merged["dataset"] == dataset_name]
        if sub.empty:
            continue
        ref_sub = (
            low[low["method"] == best_static]
            if dataset_name is None
            else low[(low["method"] == best_static) & (low["dataset"] == dataset_name)]
        )
        reference_accuracy = float(ref_sub["accuracy"].mean())
        summary_rows.append({
            "slice": label,
            "units": int(len(sub)),
            "repeats": int(sub["repeats"].iloc[0]),
            "repeated_accuracy": float(sub["accuracy"].mean()),
            "default_accuracy": 0.0,
            "best_static_method": str(best_static),
            "best_static_accuracy": reference_accuracy,
            "pooled_best_static_accuracy": best_static_accuracy,
            "gain_vs_best_static": float(sub["accuracy"].mean() - reference_accuracy),
            "selector_agreement": float(sub["selector_agree_default"].mean()),
        })

    # Fill default accuracy per slice from the single-split VACS baseline.
    default_means = (
        low[low["method"] == "vacs"]
        .groupby("dataset")["accuracy"]
        .mean()
    )
    pooled_default = float(low[low["method"] == "vacs"]["accuracy"].mean())
    for row in summary_rows:
        if row["slice"] == "Pooled":
            row["default_accuracy"] = pooled_default
        else:
            dataset_name = {
                "20NG": "20newsgroups",
                "Breast cancer": "breast_cancer",
                "Digits": "digits",
                "Iris": "iris",
                "Wine": "wine",
            }[row["slice"]]
            row["default_accuracy"] = float(default_means.loc[dataset_name])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES / "vacs_repeated_validation_summary.csv", index=False)
    merged.to_csv(TABLES / "vacs_repeated_validation_results.csv", index=False)

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Slice & Units & Repeated VACS & Default VACS & Best static & Agree \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        best = f"{METHOD_NAMES[row['best_static_method']]} {pct(row['best_static_accuracy'])}"
        lines.append(
            f"{row['slice']} & {row['units']} & {pct(row['repeated_accuracy'])} & "
            f"{pct(row['default_accuracy'])} & {best} & {pct(row['selector_agreement'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_repeated_validation_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return summary_rows


def write_latex_tables(
    df,
    low_summary,
    avg,
    summary,
    ablation_df,
    downstream_df,
    validation_df,
    split_sensitivity_df,
    repeated_validation_df,
    selection_cost_df,
    static_selection_cost_df,
    manifest,
):
    static_low = low_summary[low_summary["method"] != "vacs"]
    stability_comparator = static_low.sort_values("accuracy_mean", ascending=False).iloc[0]["method"]
    stability_rows = write_stability_table(df, stability_comparator)
    seed_block_stability = write_seed_block_stability_table(df, stability_comparator)
    oracle_rows = write_oracle_regret_summary(df)
    ablation_rows = write_ablation_table(ablation_df)
    leave_one_rows = write_leave_one_dataset_out_table(df)
    dataset_low_budget_rows = write_dataset_low_budget_table(df)
    full20news_stress_rows = write_full20news_stress_table(df)
    selector_mix_rows = write_selector_mix_table(df)
    downstream_rows = write_downstream_learner_table(downstream_df)
    validation_rows = write_validation_reliability_table(validation_df, df)
    validation_gap_rows = write_validation_gap_table(validation_df)
    validation_gap_fallback_rows = write_validation_gap_fallback_table(validation_df, df)
    validation_split_rows = write_validation_split_sensitivity_table(split_sensitivity_df, df)
    repeated_validation_rows = write_repeated_validation_table(repeated_validation_df, df)
    selection_cost_rows = write_selection_cost_table(
        selection_cost_df,
        static_selection_cost_df,
        stability_comparator,
    )

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Acc. & Std. & Macro-F1 & F1 Std. \\\\",
        "\\midrule",
    ]
    for _, row in low_summary.iterrows():
        name = METHOD_NAMES[row["method"]]
        lines.append(
            f"{name} & {pct(row['accuracy_mean'])} & {pct(row['accuracy_std'])} & "
            f"{pct(row['macro_f1_mean'])} & {pct(row['macro_f1_std'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_low_budget_table.tex").write_text("\n".join(lines), encoding="utf-8")

    pivot = avg.pivot(index="budget_per_class", columns="method", values="accuracy_mean")
    lines = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Budget/class & Random & Herding & K-center & Boundary & K-means & MARC & BADGE & VACS \\\\",
        "\\midrule",
    ]
    for budget, row in pivot.iterrows():
        lines.append(
            f"{int(budget)} & {pct(row['random'])} & {pct(row['herding'])} & "
            f"{pct(row['kcenter'])} & {pct(row['boundary'])} & {pct(row['kmeans'])} & "
            f"{pct(row['marc'])} & {pct(row['badge'])} & "
            f"{pct(row['vacs'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_budget_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # Per-dataset budget=2 table for paper detail.
    detail = summary[summary["budget_per_class"] == 2]
    pivot = detail.pivot(index="dataset", columns="method", values="accuracy_mean")
    lines = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Dataset & Random & Herding & K-center & Boundary & K-means & MARC & BADGE & VACS \\\\",
        "\\midrule",
    ]
    for dataset, row in pivot.iterrows():
        pretty = dataset.replace("_", "\\_")
        lines.append(
            f"{pretty} & {pct(row['random'])} & {pct(row['herding'])} & "
            f"{pct(row['kcenter'])} & {pct(row['boundary'])} & {pct(row['kmeans'])} & "
            f"{pct(row['marc'])} & {pct(row['badge'])} & "
            f"{pct(row['vacs'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_dataset_budget2_table.tex").write_text("\n".join(lines), encoding="utf-8")

    (TABLES / "headline_numbers.tex").write_text(
        "\\newcommand{\\vacsLowAcc}{" + pct(manifest["headline"]["vacs_low_budget_accuracy"]) + "\\%}\n"
        "\\newcommand{\\bestStaticLowAcc}{" + pct(manifest["headline"]["best_static_low_budget_accuracy"]) + "\\%}\n"
        "\\newcommand{\\vacsGain}{" + pct(manifest["headline"]["absolute_gain"]) + "\\ points}\n",
        encoding="utf-8",
    )

    manifest["stability"] = {
        "comparator": stability_comparator,
        "pooled": {
            "n": stability_rows[0]["n"],
            "delta": float(stability_rows[0]["delta"]),
            "ci": stability_rows[0]["ci"],
            "wtl": stability_rows[0]["wtl"],
            "sign_p": float(stability_rows[0]["sign_p"]),
        },
        "slices": {
            row["slice"]: {
                "n": row["n"],
                "delta": row["delta"],
                "ci": row["ci"],
                "wtl": row["wtl"],
                "sign_p": float(row["sign_p"]),
            }
            for row in stability_rows[1:]
        },
    }
    manifest["seed_block_stability"] = seed_block_stability
    manifest["oracle_regret"] = {row["method"]: row for row in oracle_rows}
    manifest["ablation"] = {
        row["ablation_method"]: row
        for row in ablation_rows
    }
    manifest["leave_one_dataset_out"] = leave_one_rows
    manifest["dataset_low_budget"] = dataset_low_budget_rows
    manifest["full20news_stress"] = full20news_stress_rows
    manifest["selector_mix_by_dataset"] = selector_mix_rows
    manifest["downstream_learners"] = downstream_rows
    manifest["validation_reliability"] = validation_rows
    manifest["validation_confidence"] = validation_gap_rows
    manifest["validation_gap_fallback"] = validation_gap_fallback_rows
    manifest["validation_split_sensitivity"] = validation_split_rows
    manifest["repeated_validation"] = repeated_validation_rows
    manifest["selection_cost"] = selection_cost_rows


def write_figures(avg, df):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator

    times_path = Path("C:/Windows/Fonts/times.ttf")
    times_prop = None
    if times_path.exists():
        font_manager.fontManager.addfont(str(times_path))
        times_prop = font_manager.FontProperties(fname=str(times_path))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    def apply_times(texts):
        for text in texts:
            if times_prop is not None:
                text.set_fontproperties(times_prop)
            else:
                text.set_fontname("Times New Roman")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for method in METHODS:
        sub = avg[avg["method"] == method].sort_values("budget_per_class")
        marker = "o" if method == "vacs" else "s"
        linewidth = 2.5 if method == "vacs" else 1.4
        ax.plot(
            sub["budget_per_class"],
            100 * sub["accuracy_mean"],
            marker=marker,
            linewidth=linewidth,
            label=METHOD_NAMES[method],
        )
    ax.set_xlabel("Selected examples per class", fontproperties=times_prop)
    ax.set_ylabel("Mean accuracy across datasets (%)", fontproperties=times_prop)
    ax.set_xticks(BUDGETS)
    apply_times(ax.get_xticklabels() + ax.get_yticklabels())
    ax.grid(True, alpha=0.25)
    legend = ax.legend(frameon=False, prop=times_prop, ncol=2)
    apply_times(legend.get_texts())
    fig.tight_layout()
    fig.savefig(FIGURES / "vacs_budget_curve.pdf")
    fig.savefig(FIGURES / "vacs_budget_curve.png", dpi=220)
    plt.close(fig)

    low_vacs = df[
        (df["learner"] == "knn3")
        & (df["method"] == "vacs")
        & (df["budget_per_class"].isin(LOW_BUDGETS))
    ]
    if low_vacs.empty:
        return

    counts = (
        low_vacs.groupby(["dataset", "selector_detail"])
        .size()
        .unstack(fill_value=0)
    )
    for selector in BASE_SELECTORS:
        if selector not in counts.columns:
            counts[selector] = 0
    counts = counts[BASE_SELECTORS]
    dataset_order = [
        dataset for dataset in DATASET_ORDER if dataset in set(counts.index)
    ] + [
        dataset for dataset in sorted(counts.index) if dataset not in DATASET_ORDER
    ]
    counts = counts.loc[dataset_order]

    selector_colors = {
        "random": "#4C78A8",
        "herding": "#F58518",
        "kcenter": "#54A24B",
        "boundary": "#E45756",
        "kmeans": "#72B7B2",
        "marc": "#B279A2",
    }
    display_names = {
        "20newsgroups": "20NG",
        "breast_cancer": "Breast cancer",
        "digits": "Digits",
        "iris": "Iris",
        "wine": "Wine",
    }
    x = np.arange(len(counts.index))
    bottoms = np.zeros(len(counts.index))
    fig, ax = plt.subplots(figsize=(5.2, 3.15))
    for selector in BASE_SELECTORS:
        values = counts[selector].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=METHOD_NAMES[selector],
            color=selector_colors[selector],
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(dataset, dataset) for dataset in counts.index])
    ax.set_xlabel("Dataset", fontproperties=times_prop)
    ax.set_ylabel("Low-budget VACS choices", fontproperties=times_prop)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(bottoms) * 1.10)
    ax.grid(axis="y", alpha=0.22)
    apply_times(ax.get_xticklabels() + ax.get_yticklabels())
    legend = ax.legend(
        frameon=False,
        prop=times_prop,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    apply_times(legend.get_texts())
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.76)
    fig.savefig(FIGURES / "vacs_selector_mix.pdf")
    fig.savefig(FIGURES / "vacs_selector_mix.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    run()
