from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import run_vacs_experiments as vacs_experiments
from run_vacs_experiments import (
    BASE_SELECTORS,
    DATASET_NAMES,
    LOW_BUDGETS,
    METHOD_NAMES,
    RESULTS,
    SEEDS,
    TABLES,
    evaluate_subset,
    load_datasets,
    preprocess_dense,
    preprocess_text,
    validation_aligned_select,
)


VALIDATION_SIZE_FRACTIONS = [0.10, 0.20, 0.25, 0.33]
SEVERE_REGRET_THRESHOLD = 0.05


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def pct2(value: float) -> str:
    return f"{100.0 * value:.2f}"


def fmt_signed(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def load_static_scores() -> tuple[dict[tuple[str, int, int, str], dict[str, float]], str, float]:
    all_results_path = TABLES / "vacs_all_results.csv"
    if not all_results_path.is_file():
        raise FileNotFoundError(
            "Expected results/tables/vacs_all_results.csv. Run experiments/run_vacs_experiments.py first."
        )

    df = pd.read_csv(all_results_path)
    low = df[
        (df["learner"] == "knn3")
        & (df["budget_per_class"].isin(LOW_BUDGETS))
        & (df["method"].isin(BASE_SELECTORS))
    ].copy()
    if low.empty:
        raise ValueError("No low-budget static selector rows found in vacs_all_results.csv.")

    score_map: dict[tuple[str, int, int, str], dict[str, float]] = {}
    for _, row in low.iterrows():
        key = (
            str(row["dataset"]),
            int(row["seed"]),
            int(row["budget_per_class"]),
            str(row["method"]),
        )
        score_map[key] = {
            "test_accuracy": float(row["accuracy"]),
            "test_macro_f1": float(row["macro_f1"]),
        }

    static_means = low.groupby("method")["accuracy"].mean()
    best_static = str(static_means.idxmax())
    return score_map, best_static, float(static_means.loc[best_static])


def rank_reliability(
    diagnostics: list[dict[str, float | int | str]],
    static_scores: dict[tuple[str, int, int, str], dict[str, float]],
    dataset: str,
    seed: int,
    budget: int,
) -> dict[str, float | bool | str]:
    order_map = {method: i for i, method in enumerate(BASE_SELECTORS)}
    unit_rows = []
    for diag in diagnostics:
        method = str(diag["selector"])
        scores = static_scores[(dataset, seed, budget, method)]
        unit_rows.append({
            "method": method,
            "selector_order": order_map[method],
            "validation_accuracy": float(diag["validation_accuracy"]),
            "validation_macro_f1": float(diag["validation_macro_f1"]),
            "test_accuracy": float(scores["test_accuracy"]),
            "test_macro_f1": float(scores["test_macro_f1"]),
        })

    unit = pd.DataFrame(unit_rows)
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
    val_ranks = np.array([val_order[method] for method in BASE_SELECTORS], dtype=float)
    test_ranks = np.array([test_order[method] for method in BASE_SELECTORS], dtype=float)
    rho = 1.0 if np.std(val_ranks) == 0.0 or np.std(test_ranks) == 0.0 else float(np.corrcoef(val_ranks, test_ranks)[0, 1])

    validation_top = str(val_sorted.iloc[0]["method"])
    test_top = str(test_sorted.iloc[0]["method"])
    selected_test_accuracy = float(
        unit.loc[unit["method"] == validation_top, "test_accuracy"].iloc[0]
    )
    best_test_accuracy = float(test_sorted.iloc[0]["test_accuracy"])
    regret = best_test_accuracy - selected_test_accuracy
    return {
        "validation_top": validation_top,
        "test_top": test_top,
        "top1_agree": validation_top == test_top,
        "rank_correlation": rho,
        "selected_test_accuracy": selected_test_accuracy,
        "best_test_accuracy": best_test_accuracy,
        "regret": regret,
        "severe": regret > SEVERE_REGRET_THRESHOLD,
    }


def write_table(summary_df: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Val. frac. & Units & VACS & Gain & Top-1 & $\\rho$ & Regret & Severe \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.sort_values("validation_fraction").iterrows():
        lines.append(
            f"{row['validation_fraction']:.2f} & {int(row['units'])} & "
            f"{pct(float(row['vacs_accuracy']))} & {fmt_signed(float(row['gain']))} & "
            f"{pct(float(row['top1_agreement']))} & {float(row['rank_correlation']):.2f} & "
            f"{pct2(float(row['regret']))} & {pct(float(row['severe_rate']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_validation_size_reliability_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run() -> None:
    start = time.time()
    static_scores, best_static, best_static_accuracy = load_static_scores()
    result_rows = []
    validation_score_rows = []

    original_make_classifier = vacs_experiments.make_classifier

    def memory_safe_make_classifier(learner: str, subset_size: int, seed: int):
        clf = original_make_classifier(learner, subset_size, seed)
        if learner == "knn3":
            clf.set_params(algorithm="kd_tree")
        return clf

    vacs_experiments.make_classifier = memory_safe_make_classifier

    try:
        for name, X, y, kind in load_datasets():
            for seed in SEEDS:
                gc.collect()
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.3,
                    random_state=seed,
                    stratify=y,
                )
                if kind == "dense":
                    X_train, X_test = preprocess_dense(X_train, X_test)
                elif kind == "sparse_text":
                    X_train, X_test = preprocess_text(X_train, X_test)
                else:
                    raise ValueError(f"unknown dataset kind: {kind}")

                for budget in LOW_BUDGETS:
                    for val_fraction in VALIDATION_SIZE_FRACTIONS:
                        subset, selector, diagnostics = validation_aligned_select(
                            X_train,
                            y_train,
                            budget,
                            seed,
                            collect_diagnostics=True,
                            val_fraction=val_fraction,
                        )
                        metrics = evaluate_subset(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            subset,
                        )
                        reliability = rank_reliability(
                            diagnostics,
                            static_scores,
                            name,
                            seed,
                            budget,
                        )
                        result_rows.append({
                            "dataset": name,
                            "seed": int(seed),
                            "budget_per_class": int(budget),
                            "validation_fraction": float(val_fraction),
                            "selector_detail": selector,
                            "selected": int(len(subset)),
                            "accuracy": float(metrics["accuracy"]),
                            "macro_f1": float(metrics["macro_f1"]),
                            **reliability,
                        })
                        for diag in diagnostics:
                            validation_score_rows.append({
                                "dataset": name,
                                "seed": int(seed),
                                "budget_per_class": int(budget),
                                "validation_fraction": float(val_fraction),
                                **diag,
                            })
    finally:
        vacs_experiments.make_classifier = original_make_classifier

    results_df = pd.DataFrame(result_rows)
    key_cols = ["dataset", "seed", "budget_per_class"]
    default = (
        results_df[np.isclose(results_df["validation_fraction"], 0.25)]
        [key_cols + ["selector_detail"]]
        .rename(columns={"selector_detail": "default_selector"})
    )
    results_df = results_df.merge(default, on=key_cols, how="left")
    results_df["selector_agree_default"] = (
        results_df["selector_detail"] == results_df["default_selector"]
    )

    summary_rows = []
    for val_fraction, sub in results_df.groupby("validation_fraction", sort=True):
        summary_rows.append({
            "validation_fraction": float(val_fraction),
            "units": int(len(sub)),
            "vacs_accuracy": float(sub["accuracy"].mean()),
            "best_static_method": best_static,
            "best_static_accuracy": best_static_accuracy,
            "gain": float(sub["accuracy"].mean() - best_static_accuracy),
            "top1_agreement": float(sub["top1_agree"].mean()),
            "rank_correlation": float(sub["rank_correlation"].mean()),
            "regret": float(sub["regret"].mean()),
            "severe_rate": float(sub["severe"].mean()),
            "selector_agreement": float(sub["selector_agree_default"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(
        TABLES / "vacs_validation_size_reliability_results.csv",
        index=False,
    )
    pd.DataFrame(validation_score_rows).to_csv(
        TABLES / "vacs_validation_size_reliability_scores.csv",
        index=False,
    )
    summary_df.to_csv(
        TABLES / "vacs_validation_size_reliability_summary.csv",
        index=False,
    )
    write_table(summary_df)

    manifest = {
        "validation_fractions": VALIDATION_SIZE_FRACTIONS,
        "budgets_per_class": LOW_BUDGETS,
        "seeds": SEEDS,
        "selectors": BASE_SELECTORS,
        "best_static_method": best_static,
        "best_static_label": METHOD_NAMES[best_static],
        "best_static_accuracy": best_static_accuracy,
        "severe_regret_threshold": SEVERE_REGRET_THRESHOLD,
        "runtime_seconds": time.time() - start,
        "datasets": [
            {
                "name": name,
                "label": DATASET_NAMES.get(name, name),
                "units": int(len(results_df[results_df["dataset"] == name]) / len(VALIDATION_SIZE_FRACTIONS)),
            }
            for name in sorted(results_df["dataset"].unique())
        ],
    }
    (RESULTS / "validation_size_reliability_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))
    print(f"Wrote validation-size reliability diagnostic in {time.time() - start:.2f}s")


if __name__ == "__main__":
    run()
