from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from run_covtype_external_audit import (
    BASE_SELECTORS,
    DATA_PATH,
    METHOD_DISPLAY,
    METHODS,
    RESULTS,
    ROOT,
    TABLES,
    load_covtype,
    paired_summary,
    pct,
    preprocess_dense,
    run_method,
    stratified_cap,
    summarize_method_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirmatory Covertype audit with a larger cap and fresh sample seed."
    )
    parser.add_argument("--max-per-class", type=int, default=10000)
    parser.add_argument("--sample-seed", type=int, default=20270610)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def write_tex_table(path: Path, summary: pd.DataFrame, paired_rows: list[dict[str, object]]) -> None:
    ordered = summary.set_index("method").loc[METHODS].reset_index()
    notes = {
        row["target"]: (
            f"vs. {METHOD_DISPLAY[row['comparator']]} "
            f"{row['delta_points']} {row['ci_points']}"
        )
        for row in paired_rows
        if row["comparator"] == "herding"
    }
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Acc. & Macro-F1 & Selected & Paired note \\\\",
        "\\midrule",
    ]
    for row in ordered.itertuples(index=False):
        method = str(row.method)
        lines.append(
            f"{METHOD_DISPLAY[method]} & {pct(float(row.accuracy_mean))} & "
            f"{pct(float(row.macro_f1_mean))} & {float(row.selected_mean):.1f} & "
            f"{notes.get(method, '')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_per_class = min(args.max_per_class, 500)
        args.seeds = args.seeds[:2]

    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    start = time.time()

    X, y = load_covtype(DATA_PATH)
    X, y, sample_manifest = stratified_cap(
        X, y, max_per_class=args.max_per_class, seed=args.sample_seed
    )

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        X_train, X_test = preprocess_dense(X_train_raw, X_test_raw)
        for budget in args.budgets:
            for method in METHODS:
                method_start = time.perf_counter()
                detail, subset, metrics = run_method(
                    method, X_train, y_train, X_test, y_test, budget, seed
                )
                rows.append(
                    {
                        "dataset": "covertype_confirmatory",
                        "seed": int(seed),
                        "budget_per_class": int(budget),
                        "method": method,
                        "selector_detail": detail,
                        "selected": int(len(subset)),
                        "seconds": float(time.perf_counter() - method_start),
                        **metrics,
                    }
                )

    results = pd.DataFrame(rows)
    results_path = TABLES / "vacs_covtype_confirmatory_results.csv"
    summary_path = TABLES / "vacs_covtype_confirmatory_summary.csv"
    paired_path = TABLES / "vacs_covtype_confirmatory_paired_summary.csv"
    table_path = TABLES / "vacs_covtype_confirmatory_table.tex"
    manifest_path = RESULTS / "covtype_confirmatory_audit_manifest.json"

    results.to_csv(results_path, index=False)
    summary = summarize_method_results(results)
    summary.to_csv(summary_path, index=False)

    static_summary = summary[summary["method"].isin(BASE_SELECTORS)]
    best_static = str(static_summary.iloc[0]["method"])
    paired_rows = [
        paired_summary(results, "vacs_f", "herding"),
        paired_summary(results, "vacs_r", "herding"),
        paired_summary(results, "vacs_f", best_static),
        paired_summary(results, "vacs_r", best_static),
    ]
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)
    write_tex_table(table_path, summary, paired_rows)

    manifest = {
        "dataset": "Covertype",
        "purpose": "confirmatory_larger_cap_fresh_sample_seed",
        "source": "UCI covtype.data.gz",
        "data_path": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sample": sample_manifest,
        "seeds": [int(seed) for seed in args.seeds],
        "budgets_per_class": [int(budget) for budget in args.budgets],
        "methods": METHODS,
        "best_static_method": best_static,
        "runtime_sec": float(time.time() - start),
        "smoke": bool(args.smoke),
        "outputs": [
            str(results_path.relative_to(ROOT)).replace("\\", "/"),
            str(summary_path.relative_to(ROOT)).replace("\\", "/"),
            str(paired_path.relative_to(ROOT)).replace("\\", "/"),
            str(table_path.relative_to(ROOT)).replace("\\", "/"),
        ],
        "paired_summary": paired_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("COVTYPE_CONFIRMATORY_AUDIT_OK")


if __name__ == "__main__":
    main()
