from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_vacs_experiments import (
    DATASET_ORDER,
    LOW_BUDGETS,
    METHOD_NAMES,
    METHODS,
    bootstrap_mean_ci,
    fmt_p_value,
    fmt_signed,
    pct,
    sign_test_p_value,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
ALL_RESULTS = TABLES / "vacs_all_results.csv"
REPEATED_RESULTS = TABLES / "vacs_repeated_validation_results.csv"


def dataset_label(dataset: str) -> str:
    return {
        "20newsgroups": "20NG",
        "breast_cancer": "Breast cancer",
        "digits": "Digits",
        "iris": "Iris",
        "wine": "Wine",
    }.get(dataset, dataset.replace("_", " "))


def summarize(label: str, comparator_label: str, paired: pd.DataFrame) -> dict[str, object]:
    delta = paired["delta_accuracy"].to_numpy(dtype=float)
    lo, hi = bootstrap_mean_ci(delta)
    wins = int((delta > 1e-12).sum())
    ties = int((np.abs(delta) <= 1e-12).sum())
    losses = int((delta < -1e-12).sum())
    sign_p = sign_test_p_value(wins, losses)
    return {
        "slice": label,
        "comparator": comparator_label,
        "n": int(len(paired)),
        "repeated_vacs_accuracy": float(paired["repeated_vacs_accuracy"].mean()),
        "comparator_accuracy": float(paired["comparator_accuracy"].mean()),
        "delta_accuracy": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_p": float(sign_p),
        "delta_points": fmt_signed(float(delta.mean())),
        "ci_points": f"[{fmt_signed(lo)},{fmt_signed(hi)}]",
        "wtl": f"{wins}/{ties}/{losses}",
        "sign_p_tex": fmt_p_value(sign_p),
    }


def best_pooled_static_method(low: pd.DataFrame) -> str:
    static_methods = [method for method in METHODS if method != "vacs"]
    means = low.groupby("method")["accuracy"].mean()
    return max(static_methods, key=lambda method: float(means.loc[method]))


def paired_against_method(repeated: pd.DataFrame, low: pd.DataFrame, method: str) -> pd.DataFrame:
    comparator = low[low["method"] == method][
        ["dataset", "seed", "budget_per_class", "accuracy"]
    ].rename(columns={"accuracy": "comparator_accuracy"})
    paired = repeated.merge(
        comparator, on=["dataset", "seed", "budget_per_class"], how="inner"
    )
    paired["comparator"] = method
    paired["delta_accuracy"] = (
        paired["repeated_vacs_accuracy"] - paired["comparator_accuracy"]
    )
    return paired


def paired_against_dataset_best(repeated: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    static_methods = [method for method in METHODS if method != "vacs"]
    paired_frames: list[pd.DataFrame] = []
    for dataset in sorted(low["dataset"].unique()):
        sub = low[low["dataset"] == dataset]
        means = sub.groupby("method")["accuracy"].mean()
        best_method = max(static_methods, key=lambda method: float(means.loc[method]))
        comparator = sub[sub["method"] == best_method][
            ["dataset", "seed", "budget_per_class", "accuracy"]
        ].rename(columns={"accuracy": "comparator_accuracy"})
        paired = repeated[repeated["dataset"] == dataset].merge(
            comparator, on=["dataset", "seed", "budget_per_class"], how="inner"
        )
        paired["comparator"] = best_method
        paired["delta_accuracy"] = (
            paired["repeated_vacs_accuracy"] - paired["comparator_accuracy"]
        )
        paired_frames.append(paired)
    return pd.concat(paired_frames, ignore_index=True)


def summarize_slices(paired: pd.DataFrame, comparator_label: str) -> list[dict[str, object]]:
    dataset_order = [
        dataset for dataset in DATASET_ORDER if dataset in set(paired["dataset"])
    ] + [
        dataset for dataset in sorted(paired["dataset"].unique()) if dataset not in DATASET_ORDER
    ]
    rows = [summarize("Pooled", comparator_label, paired)]
    for dataset in dataset_order:
        rows.append(
            summarize(
                dataset_label(dataset),
                comparator_label,
                paired[paired["dataset"] == dataset],
            )
        )
    return rows


def write_table(rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Reference & $n$ & Repeated VACS & Ref. & $\\Delta$ [95\\% CI] & W/T/L \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparator']} & {row['n']} & "
            f"{pct(float(row['repeated_vacs_accuracy']))} & "
            f"{pct(float(row['comparator_accuracy']))} & "
            f"{row['delta_points']} {row['ci_points']} & {row['wtl']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_repeated_validation_comparison_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    low = pd.read_csv(ALL_RESULTS)
    low = low[
        (low["learner"] == "knn3")
        & low["budget_per_class"].isin(LOW_BUDGETS)
    ].copy()
    repeated = pd.read_csv(REPEATED_RESULTS)
    repeated = repeated.rename(columns={"accuracy": "repeated_vacs_accuracy"})

    best_static_method = best_pooled_static_method(low)
    best_static_paired = paired_against_method(repeated, low, best_static_method)
    dataset_best_paired = paired_against_dataset_best(repeated, low)

    detail = pd.concat(
        [
            best_static_paired.assign(comparison="best_static"),
            dataset_best_paired.assign(comparison="dataset_best"),
        ],
        ignore_index=True,
    )
    detail[
        [
            "comparison",
            "dataset",
            "seed",
            "budget_per_class",
            "comparator",
            "repeated_vacs_accuracy",
            "comparator_accuracy",
            "delta_accuracy",
        ]
    ].to_csv(TABLES / "vacs_repeated_validation_comparison.csv", index=False)

    best_static_label = METHOD_NAMES.get(best_static_method, best_static_method)
    best_static_rows = summarize_slices(best_static_paired, best_static_label)
    dataset_best_rows = summarize_slices(dataset_best_paired, "Per-dataset static")
    all_rows = [
        {**row, "comparison": "best_static"} for row in best_static_rows
    ] + [
        {**row, "comparison": "dataset_best"} for row in dataset_best_rows
    ]
    pd.DataFrame(all_rows).to_csv(
        TABLES / "vacs_repeated_validation_comparison_summary.csv",
        index=False,
    )

    pooled_rows = [best_static_rows[0], dataset_best_rows[0]]
    write_table(pooled_rows)

    manifest = {
        "inputs": [
            str(ALL_RESULTS.relative_to(ROOT)).replace("\\", "/"),
            str(REPEATED_RESULTS.relative_to(ROOT)).replace("\\", "/"),
        ],
        "filter": {
            "learner": "knn3",
            "budget_per_class": LOW_BUDGETS,
            "unit": ["dataset", "seed", "budget_per_class"],
            "best_pooled_static_method": best_static_method,
        },
        "bootstrap_reps": 20000,
        "summary": all_rows,
        "outputs": [
            "results/tables/vacs_repeated_validation_comparison.csv",
            "results/tables/vacs_repeated_validation_comparison_summary.csv",
            "results/tables/vacs_repeated_validation_comparison_table.tex",
        ],
    }
    (RESULTS / "repeated_validation_comparison_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(pooled_rows, indent=2))
    print("REPEATED_VALIDATION_COMPARISON_AUDIT_OK")


if __name__ == "__main__":
    main()
