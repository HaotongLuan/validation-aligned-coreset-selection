from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from run_vacs_experiments import (
    BASE_SELECTORS,
    LOW_BUDGETS,
    METHOD_NAMES,
    bootstrap_mean_ci,
    evaluate_subset,
    fmt_signed,
    pct,
    preprocess_dense,
    repeated_validation_aligned_select,
    select_indices,
    sign_test_p_value,
    validation_aligned_select,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_HOME = ROOT / ".sklearn_data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
DATA_PATH = DATA_HOME / "covtype.data.gz"
METHODS = BASE_SELECTORS + ["vacs_f", "vacs_r"]
METHOD_DISPLAY = {
    **METHOD_NAMES,
    "vacs_f": "VACS-F",
    "vacs_r": "VACS-R",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External Covertype audit for VACS selector robustness."
    )
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--sample-seed", type=int, default=20270609)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--budgets", type=int, nargs="+", default=LOW_BUDGETS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_covtype(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download it from the UCI Covertype archive "
            "before running this audit."
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        df = pd.read_csv(handle, header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32, copy=True)
    y = df.iloc[:, -1].to_numpy(dtype=np.int64, copy=True) - 1
    return X, y


def stratified_cap(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    original_counts: dict[int, int] = {}
    sampled_counts: dict[int, int] = {}
    for label in sorted(np.unique(y)):
        idx = np.flatnonzero(y == label)
        original_counts[int(label)] = int(len(idx))
        take = min(int(max_per_class), int(len(idx)))
        chosen = rng.choice(idx, size=take, replace=False)
        selected.append(chosen)
        sampled_counts[int(label)] = int(take)
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    manifest = {
        "sample_seed": int(seed),
        "max_per_class": int(max_per_class),
        "original_n": int(len(y)),
        "sampled_n": int(len(indices)),
        "classes": int(len(original_counts)),
        "original_counts": original_counts,
        "sampled_counts": sampled_counts,
    }
    return X[indices], y[indices], manifest


def run_method(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    budget: int,
    seed: int,
) -> tuple[str, np.ndarray, dict[str, float]]:
    if method in BASE_SELECTORS:
        subset = select_indices(X_train, y_train, budget, method, seed)
        detail = method
    elif method == "vacs_f":
        subset, detail = validation_aligned_select(X_train, y_train, budget, seed)
    elif method == "vacs_r":
        subset, detail, _diagnostics = repeated_validation_aligned_select(
            X_train, y_train, budget, seed
        )
    else:
        raise ValueError(method)
    metrics = evaluate_subset(X_train, y_train, X_test, y_test, subset, seed=seed)
    return detail, subset, metrics


def summarize_method_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("method", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            selected_mean=("selected", "mean"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )


def paired_summary(results: pd.DataFrame, vacs_method: str, comparator: str) -> dict[str, object]:
    paired = (
        results[results["method"].isin([vacs_method, comparator])]
        .pivot_table(
            index=["seed", "budget_per_class"],
            columns="method",
            values="accuracy",
        )
        .reset_index()
    )
    delta = (paired[vacs_method] - paired[comparator]).to_numpy(dtype=float)
    lo, hi = bootstrap_mean_ci(delta, reps=20000)
    wins = int((delta > 1e-12).sum())
    ties = int((np.abs(delta) <= 1e-12).sum())
    losses = int((delta < -1e-12).sum())
    return {
        "target": vacs_method,
        "comparator": comparator,
        "n": int(len(delta)),
        "target_accuracy": float(paired[vacs_method].mean()),
        "comparator_accuracy": float(paired[comparator].mean()),
        "delta_accuracy": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_p": float(sign_test_p_value(wins, losses)),
        "delta_points": fmt_signed(float(delta.mean())),
        "ci_points": f"[{fmt_signed(lo)},{fmt_signed(hi)}]",
        "wtl": f"{wins}/{ties}/{losses}",
    }


def write_tex_table(summary: pd.DataFrame, paired_rows: list[dict[str, object]]) -> None:
    ordered = summary.set_index("method").loc[METHODS].reset_index()
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Acc. & Macro-F1 & Selected & Paired note \\\\",
        "\\midrule",
    ]
    notes = {
        row["target"]: (
            f"vs. {METHOD_DISPLAY[row['comparator']]} "
            f"{row['delta_points']} {row['ci_points']}"
        )
        for row in paired_rows
        if row["comparator"] == "herding"
    }
    for row in ordered.itertuples(index=False):
        method = str(row.method)
        lines.append(
            f"{METHOD_DISPLAY[method]} & {pct(float(row.accuracy_mean))} & "
            f"{pct(float(row.macro_f1_mean))} & {float(row.selected_mean):.1f} & "
            f"{notes.get(method, '')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "vacs_covtype_external_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_per_class = min(args.max_per_class, 250)
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
                        "dataset": "covertype_external",
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
    results.to_csv(TABLES / "vacs_covtype_external_results.csv", index=False)
    summary = summarize_method_results(results)
    summary.to_csv(TABLES / "vacs_covtype_external_summary.csv", index=False)

    static_summary = summary[summary["method"].isin(BASE_SELECTORS)]
    best_static = str(static_summary.iloc[0]["method"])
    paired_rows = [
        paired_summary(results, "vacs_f", "herding"),
        paired_summary(results, "vacs_r", "herding"),
        paired_summary(results, "vacs_f", best_static),
        paired_summary(results, "vacs_r", best_static),
    ]
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(TABLES / "vacs_covtype_external_paired_summary.csv", index=False)
    write_tex_table(summary, paired_rows)

    manifest = {
        "dataset": "Covertype",
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
            "results/tables/vacs_covtype_external_results.csv",
            "results/tables/vacs_covtype_external_summary.csv",
            "results/tables/vacs_covtype_external_paired_summary.csv",
            "results/tables/vacs_covtype_external_table.tex",
        ],
        "paired_summary": paired_rows,
    }
    (RESULTS / "covtype_external_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    print("COVTYPE_EXTERNAL_AUDIT_OK")


if __name__ == "__main__":
    main()
