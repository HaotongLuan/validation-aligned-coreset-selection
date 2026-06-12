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
INPUT = TABLES / "vacs_all_results.csv"


def dataset_label(dataset: str) -> str:
    return {
        "20newsgroups": "20NG",
        "breast_cancer": "Breast cancer",
        "digits": "Digits",
        "wine": "Wine",
    }.get(dataset, dataset.replace("_", " "))


def summarize_slice(label: str, best_method: str, paired: pd.DataFrame) -> dict[str, object]:
    delta = paired["delta_accuracy"].to_numpy(dtype=float)
    lo, hi = bootstrap_mean_ci(delta)
    wins = int((delta > 1e-12).sum())
    ties = int((np.abs(delta) <= 1e-12).sum())
    losses = int((delta < -1e-12).sum())
    sign_p = sign_test_p_value(wins, losses)
    return {
        "slice": label,
        "n": int(len(paired)),
        "best_static_method": best_method,
        "vacs_accuracy": float(paired["vacs_accuracy"].mean()),
        "best_static_accuracy": float(paired["best_static_accuracy"].mean()),
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


def main() -> None:
    df = pd.read_csv(INPUT)
    low = df[(df["learner"] == "knn3") & df["budget_per_class"].isin(LOW_BUDGETS)].copy()
    static_methods = [method for method in METHODS if method != "vacs"]

    paired_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    dataset_order = [
        dataset for dataset in DATASET_ORDER if dataset in set(low["dataset"])
    ] + [
        dataset for dataset in sorted(low["dataset"].unique()) if dataset not in DATASET_ORDER
    ]

    for dataset in dataset_order:
        sub = low[low["dataset"] == dataset]
        means = sub.groupby("method")["accuracy"].mean()
        best_method = max(static_methods, key=lambda method: float(means.loc[method]))
        vacs = sub[sub["method"] == "vacs"][
            ["dataset", "seed", "budget_per_class", "accuracy"]
        ].rename(columns={"accuracy": "vacs_accuracy"})
        static = sub[sub["method"] == best_method][
            ["dataset", "seed", "budget_per_class", "accuracy"]
        ].rename(columns={"accuracy": "best_static_accuracy"})
        paired = vacs.merge(static, on=["dataset", "seed", "budget_per_class"], how="inner")
        paired["best_static_method"] = best_method
        paired["delta_accuracy"] = paired["vacs_accuracy"] - paired["best_static_accuracy"]
        paired_frames.append(paired)
        summary_rows.append(summarize_slice(dataset_label(dataset), best_method, paired))

    pooled = pd.concat(paired_frames, ignore_index=True)
    summary_rows.insert(0, summarize_slice("Pooled per-dataset-best", "per-dataset", pooled))

    paired_out = pooled[
        [
            "dataset",
            "seed",
            "budget_per_class",
            "best_static_method",
            "vacs_accuracy",
            "best_static_accuracy",
            "delta_accuracy",
        ]
    ]
    paired_out.to_csv(TABLES / "vacs_dataset_best_static_paired.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "vacs_dataset_best_static_paired_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Slice & Static ref. & VACS & Static & $\\Delta$ [95\\% CI] & W/T/L \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        method = row["best_static_method"]
        ref = (
            "Per-dataset"
            if method == "per-dataset"
            else METHOD_NAMES.get(str(method), str(method))
        )
        lines.append(
            f"{row['slice']} & {ref} & {pct(float(row['vacs_accuracy']))} & "
            f"{pct(float(row['best_static_accuracy']))} & "
            f"{row['delta_points']} {row['ci_points']} & {row['wtl']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_dataset_best_static_paired_table.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    manifest = {
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "filter": {
            "learner": "knn3",
            "budget_per_class": LOW_BUDGETS,
            "static_methods": static_methods,
        },
        "unit": ["dataset", "seed", "budget_per_class"],
        "bootstrap_reps": 20000,
        "summary": summary_rows,
        "outputs": [
            "results/tables/vacs_dataset_best_static_paired.csv",
            "results/tables/vacs_dataset_best_static_paired_summary.csv",
            "results/tables/vacs_dataset_best_static_paired_table.tex",
        ],
    }
    (RESULTS / "dataset_best_static_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"][0], indent=2))
    print("DATASET_BEST_STATIC_AUDIT_OK")


if __name__ == "__main__":
    main()
