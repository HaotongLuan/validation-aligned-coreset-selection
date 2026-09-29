from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis"

KEYS = ["dataset", "seed", "budget_per_class"]
DATASET_ORDER = [
    "20newsgroups",
    "breast_cancer",
    "digits",
    "iris",
    "wine",
]
LOW_BUDGETS = [1, 2]
BOOTSTRAP_SEED = 20260531
BOOTSTRAP_REPS = 20000
W_T_L_TOLERANCE = 1e-12
SUMMARY_FLOAT_TOLERANCE = 1e-15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_source_label(
    path: Path, results_root: Path, experiments_root: Path
) -> str:
    """Return a stable repository-relative label for a hashed input file."""
    resolved = path.resolve()
    for root, label in (
        (results_root.resolve(), "results"),
        (experiments_root.resolve(), "experiments"),
    ):
        try:
            return (Path(label) / resolved.relative_to(root)).as_posix()
        except ValueError:
            continue
    return path.name


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    """Exact copy of the original function's RNG, shape, and quantiles."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = rng.choice(
        values,
        size=(BOOTSTRAP_REPS, len(values)),
        replace=True,
    ).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def signed_points(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def ci_points(lo: float, hi: float) -> str:
    return f"[{signed_points(lo)},{signed_points(hi)}]"


def summarize_pair(
    paired: pd.DataFrame,
    target_col: str,
    comparator_col: str,
    comparison: str,
    target: str,
    comparator: str,
    slice_name: str,
) -> dict[str, object]:
    delta = (
        paired[target_col].to_numpy(dtype=float)
        - paired[comparator_col].to_numpy(dtype=float)
    )
    lo, hi = bootstrap_mean_ci(delta)
    wins = int((delta > W_T_L_TOLERANCE).sum())
    ties = int((np.abs(delta) <= W_T_L_TOLERANCE).sum())
    losses = int((delta < -W_T_L_TOLERANCE).sum())
    return {
        "comparison": comparison,
        "target": target,
        "comparator": comparator,
        "slice": slice_name,
        "n": int(len(delta)),
        "target_mean": float(paired[target_col].mean()),
        "comparator_mean": float(paired[comparator_col].mean()),
        "delta_mean": float(delta.mean()),
        "delta_points": signed_points(float(delta.mean())),
        "ci_low": lo,
        "ci_high": hi,
        "ci_points": ci_points(lo, hi),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wtl": f"{wins}/{ties}/{losses}",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "wtl_tolerance": W_T_L_TOLERANCE,
    }


def read_inputs(results_root: Path) -> dict[str, pd.DataFrame]:
    tables = results_root / "tables"
    return {
        "all_results": pd.read_csv(tables / "vacs_all_results.csv"),
        "repeated_results": pd.read_csv(
            tables / "vacs_repeated_validation_results.csv"
        ),
        "low_budget_summary": pd.read_csv(
            tables / "vacs_low_budget_summary.csv"
        ),
        "dataset_low_budget_summary": pd.read_csv(
            tables / "vacs_dataset_low_budget_summary.csv"
        ),
        "full20_stress_results": pd.read_csv(
            tables / "vacs_full20news_stress_results.csv"
        ),
        "full20_stress_summary": pd.read_csv(
            tables / "vacs_full20news_stress_summary.csv"
        ),
        "full20_stress_summary_row": pd.read_csv(
            tables / "vacs_full20news_stress_summary_row.csv"
        ),
    }


def main_low_budget(all_results: pd.DataFrame) -> pd.DataFrame:
    low = all_results[
        (all_results["learner"] == "knn3")
        & all_results["budget_per_class"].isin(LOW_BUDGETS)
    ].copy()
    expected = 5 * 8 * 2 * 8
    if len(low) != expected:
        raise AssertionError(f"Expected {expected} main low-budget rows, got {len(low)}")
    unit_count = low[KEYS].drop_duplicates().shape[0]
    if unit_count != 80:
        raise AssertionError(f"Expected 80 main units, got {unit_count}")
    counts = low.groupby(KEYS).size()
    if not bool((counts == 8).all()):
        raise AssertionError("Every main unit must contain exactly eight methods")
    return low


def pivot_pair(
    low: pd.DataFrame, target_method: str, comparator_method: str
) -> pd.DataFrame:
    """Match like the original write_stability_table pivot_table path."""
    methods = [target_method, comparator_method]
    pair = (
        low[low["method"].isin(methods)]
        .pivot_table(
            index=KEYS,
            columns="method",
            values="accuracy",
        )
        .reset_index()
    )
    if len(pair) != 80:
        raise AssertionError(
            f"Expected 80 matched units for {target_method} vs {comparator_method}, "
            f"got {len(pair)}"
        )
    if pair[[target_method, comparator_method]].isna().any().any():
        raise AssertionError("Matched main pair contains missing accuracy")
    return pair


def repeated_pair(
    low: pd.DataFrame, repeated_results: pd.DataFrame, comparator_method: str
) -> pd.DataFrame:
    repeated = repeated_results[
        repeated_results["budget_per_class"].isin(LOW_BUDGETS)
    ].copy()
    if len(repeated) != 80:
        raise AssertionError(f"Expected 80 repeated VACS rows, got {len(repeated)}")
    repeated = repeated.rename(columns={"accuracy": "vacs_r"})
    comparator = low[low["method"] == comparator_method][
        KEYS + ["accuracy"]
    ].rename(columns={"accuracy": comparator_method})
    pair = repeated[KEYS + ["vacs_r"]].merge(
        comparator,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(pair) != 80:
        raise AssertionError(
            f"Expected 80 repeated matches for VACS-R vs {comparator_method}, "
            f"got {len(pair)}"
        )
    return pair


def add_unit_deltas(
    unit_rows: pd.DataFrame,
    vacs_f: pd.DataFrame,
    vacs_r: pd.DataFrame,
    marc: pd.DataFrame,
    herding: pd.DataFrame,
) -> pd.DataFrame:
    out = unit_rows.copy()
    for frame, method, col in [
        (vacs_f, "vacs_f", "vacs_f_accuracy"),
        (vacs_r, "vacs_r", "vacs_r_accuracy"),
        (marc, "marc", "marc_accuracy"),
        (herding, "herding", "herding_accuracy"),
    ]:
        values = frame[KEYS + [method]].rename(columns={method: col})
        out = out.merge(values, on=KEYS, how="left", validate="one_to_one")
    out["delta_f_vs_marc"] = out["vacs_f_accuracy"] - out["marc_accuracy"]
    out["delta_r_vs_marc"] = out["vacs_r_accuracy"] - out["marc_accuracy"]
    out["delta_f_vs_herding"] = (
        out["vacs_f_accuracy"] - out["herding_accuracy"]
    )
    out["delta_r_vs_herding"] = (
        out["vacs_r_accuracy"] - out["herding_accuracy"]
    )
    return out


def slice_rows(paired: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    rows = [("Pooled", paired)]
    for dataset in DATASET_ORDER:
        sub = paired[paired["dataset"] == dataset]
        if not sub.empty:
            rows.append((dataset, sub))
    return rows


def direct_summary(
    paired: pd.DataFrame,
    target_col: str,
    comparator_col: str,
    target: str,
    comparator: str,
    comparison: str,
) -> list[dict[str, object]]:
    rows = [
        summarize_pair(
            sub,
            target_col,
            comparator_col,
            comparison,
            target,
            comparator,
            label,
        )
        for label, sub in slice_rows(paired)
    ]
    leave_out = paired[paired["dataset"] != "20newsgroups"]
    rows.append(
        summarize_pair(
            leave_out,
            target_col,
            comparator_col,
            comparison,
            target,
            comparator,
            "Leave-out-20NG",
        )
    )
    return rows


def reported_herding_checks(
    f_pair: pd.DataFrame,
    r_pair: pd.DataFrame,
    manifest: dict[str, object],
    reported_repeated: pd.DataFrame,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    f_reported = manifest["stability"]
    for label, sub in slice_rows(f_pair):
        computed = summarize_pair(
            sub,
            "vacs",
            "herding",
            "existing_reported_herding",
            "VACS-F",
            "herding",
            label,
        )
        source = f_reported["pooled"] if label == "Pooled" else f_reported["slices"][label_map(label)]
        reported_ci = str(source["ci"])
        reported_wtl = str(source["wtl"])
        checks.append(
            {
                "target": "VACS-F",
                "slice": label,
                "source": "results/experiment_manifest.json:stability",
                "computed_ci_points": computed["ci_points"],
                "reported_ci_points": reported_ci,
                "ci_match": computed["ci_points"] == reported_ci,
                "computed_wtl": computed["wtl"],
                "reported_wtl": reported_wtl,
                "wtl_match": computed["wtl"] == reported_wtl,
            }
        )

    for label, sub in slice_rows(r_pair):
        computed = summarize_pair(
            sub,
            "vacs_r",
            "herding",
            "existing_reported_herding",
            "VACS-R",
            "herding",
            label,
        )
        reported = reported_repeated[
            (reported_repeated["comparison"] == "best_static")
            & (reported_repeated["slice"] == reported_slice_label(label))
        ]
        if len(reported) != 1:
            raise AssertionError(f"Missing repeated herding report for {label}")
        source = reported.iloc[0]
        checks.append(
            {
                "target": "VACS-R",
                "slice": label,
                "source": "results/tables/vacs_repeated_validation_comparison_summary.csv:best_static",
                "computed_ci_points": computed["ci_points"],
                "reported_ci_points": str(source["ci_points"]),
                "ci_match": computed["ci_points"] == str(source["ci_points"]),
                "computed_wtl": computed["wtl"],
                "reported_wtl": str(source["wtl"]),
                "wtl_match": computed["wtl"] == str(source["wtl"]),
            }
        )
    return checks


def label_map(label: str) -> str:
    return {
        "20newsgroups": "20NG",
        "breast_cancer": "Breast cancer",
        "digits": "Digits",
        "iris": "Iris",
        "wine": "Wine",
    }[label]


def reported_slice_label(label: str) -> str:
    return "Pooled" if label == "Pooled" else label_map(label)


def table1_sd_audit(low: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    computed = (
        low.groupby("method", as_index=False)
        .agg(
            computed_accuracy_mean=("accuracy", "mean"),
            computed_accuracy_sd_ddof1=("accuracy", lambda x: x.std(ddof=1)),
            computed_macro_f1_mean=("macro_f1", "mean"),
            computed_macro_f1_sd_ddof1=("macro_f1", lambda x: x.std(ddof=1)),
            n=("accuracy", "size"),
        )
    )
    reported = reported.rename(
        columns={
            "accuracy_mean": "reported_accuracy_mean",
            "accuracy_std": "reported_accuracy_sd",
            "macro_f1_mean": "reported_macro_f1_mean",
            "macro_f1_std": "reported_macro_f1_sd",
        }
    )
    out = computed.merge(reported, on="method", how="outer", validate="one_to_one")
    out["accuracy_sd_abs_diff"] = (
        out["computed_accuracy_sd_ddof1"] - out["reported_accuracy_sd"]
    ).abs()
    out["macro_f1_sd_abs_diff"] = (
        out["computed_macro_f1_sd_ddof1"] - out["reported_macro_f1_sd"]
    ).abs()
    out["accuracy_sd_match"] = np.isclose(
        out["computed_accuracy_sd_ddof1"], out["reported_accuracy_sd"], atol=1e-15, rtol=0
    )
    out["macro_f1_sd_match"] = np.isclose(
        out["computed_macro_f1_sd_ddof1"], out["reported_macro_f1_sd"], atol=1e-15, rtol=0
    )
    return out.sort_values("method").reset_index(drop=True)


def compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    numeric: list[str],
    categorical: list[str],
) -> dict[str, object]:
    left_sorted = left.sort_values(keys).reset_index(drop=True)
    right_sorted = right.sort_values(keys).reset_index(drop=True)
    same_keys = left_sorted[keys].equals(right_sorted[keys])
    numeric_max_abs: dict[str, float] = {}
    numeric_match = True
    for column in numeric:
        diff = (
            left_sorted[column].to_numpy(dtype=float)
            - right_sorted[column].to_numpy(dtype=float)
        )
        numeric_max_abs[column] = float(np.max(np.abs(diff))) if len(diff) else 0.0
        numeric_match = numeric_match and bool(np.array_equal(diff, np.zeros_like(diff)))
    categorical_match = all(
        left_sorted[column].astype(str).tolist()
        == right_sorted[column].astype(str).tolist()
        for column in categorical
    )
    return {
        "left_rows": int(len(left_sorted)),
        "right_rows": int(len(right_sorted)),
        "same_keys": bool(same_keys),
        "numeric_max_abs_diff": numeric_max_abs,
        "numeric_exact_match": bool(numeric_match),
        "categorical_exact_match": bool(categorical_match),
        "raw_values_identical": bool(
            same_keys and numeric_match and categorical_match
        ),
    }


def full20_trace(
    low: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> tuple[dict[str, object], pd.DataFrame]:
    main_20 = low[low["dataset"] == "20newsgroups"].copy()
    stress = data["full20_stress_results"].copy()
    raw_trace = compare_frames(
        main_20,
        stress,
        KEYS + ["method"],
        ["selected", "accuracy", "macro_f1"],
        ["learner", "selector_detail"],
    )

    recomputed_summary = (
        stress.groupby("method", as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .rename(
            columns={
                "accuracy": "recomputed_accuracy_mean",
                "macro_f1": "recomputed_macro_f1_mean",
            }
        )
    )
    stored_summary = data["full20_stress_summary"].rename(
        columns={
            "accuracy_mean": "stored_accuracy_mean",
            "macro_f1_mean": "stored_macro_f1_mean",
        }
    )
    summary_join = recomputed_summary.merge(
        stored_summary,
        on="method",
        how="outer",
        validate="one_to_one",
    )
    summary_join["accuracy_abs_diff"] = (
        summary_join["recomputed_accuracy_mean"]
        - summary_join["stored_accuracy_mean"]
    ).abs()
    summary_join["macro_f1_abs_diff"] = (
        summary_join["recomputed_macro_f1_mean"]
        - summary_join["stored_macro_f1_mean"]
    ).abs()
    summary_join["stored_summary_match"] = (
        (summary_join["accuracy_abs_diff"] <= SUMMARY_FLOAT_TOLERANCE)
        & (summary_join["macro_f1_abs_diff"] <= SUMMARY_FLOAT_TOLERANCE)
    )

    main_row = data["dataset_low_budget_summary"]
    main_row = main_row[main_row["dataset"] == "20newsgroups"].iloc[0]
    stress_row = data["full20_stress_summary_row"].iloc[0]
    row_fields = [
        "vacs_accuracy",
        "best_static_accuracy",
        "gain",
    ]
    row_diffs = {
        field: float(main_row[field] - stress_row[field]) for field in row_fields
    }
    row_match = (
        main_row["best_static_method"] == stress_row["best_static_method"]
        and all(value == 0.0 for value in row_diffs.values())
    )
    trace = {
        "raw_main_vs_stress": raw_trace,
        "stored_stress_summary_vs_raw_recompute": {
            "rows": int(len(summary_join)),
            "all_match": bool(summary_join["stored_summary_match"].all()),
            "max_accuracy_abs_diff": float(summary_join["accuracy_abs_diff"].max()),
            "max_macro_f1_abs_diff": float(summary_join["macro_f1_abs_diff"].max()),
            "float_tolerance": SUMMARY_FLOAT_TOLERANCE,
        },
        "main_dataset_row_vs_stress_summary_row": {
            "main_best_static_method": str(main_row["best_static_method"]),
            "stress_best_static_method": str(stress_row["best_static_method"]),
            "field_differences": row_diffs,
            "all_match": bool(row_match),
        },
        "identical_experiments": bool(
            raw_trace["raw_values_identical"]
            and bool(summary_join["stored_summary_match"].all())
            and row_match
        ),
    }
    return trace, summary_join


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit VACS revision-round statistics.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    experiments_root = args.experiments_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_inputs(results_root)
    low = main_low_budget(data["all_results"])

    f_marc = pivot_pair(low, "vacs", "marc").rename(
        columns={"vacs": "vacs_f", "marc": "marc"}
    )
    f_herding = pivot_pair(low, "vacs", "herding").rename(
        columns={"vacs": "vacs", "herding": "herding"}
    )
    f_marc_summary = direct_summary(
        f_marc,
        "vacs_f",
        "marc",
        "VACS-F",
        "MARC",
        "direct_marc",
    )

    r_marc = repeated_pair(low, data["repeated_results"], "marc")
    r_herding = repeated_pair(low, data["repeated_results"], "herding")
    r_marc_summary = direct_summary(
        r_marc,
        "vacs_r",
        "marc",
        "VACS-R",
        "MARC",
        "direct_marc",
    )

    units = f_marc[KEYS].copy()
    units = add_unit_deltas(
        units,
        f_marc.rename(columns={"vacs_f": "vacs_f"}),
        r_marc,
        f_marc[[*KEYS, "marc"]],
        f_herding,
    )
    units = units.sort_values(KEYS).reset_index(drop=True)

    manifest_path = results_root / "experiment_manifest.json"
    repeated_summary_path = (
        results_root / "tables" / "vacs_repeated_validation_comparison_summary.csv"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reported_repeated = pd.read_csv(repeated_summary_path)
    herding_checks = reported_herding_checks(
        f_herding,
        r_herding,
        manifest,
        reported_repeated,
    )

    sd_audit = table1_sd_audit(low, data["low_budget_summary"])
    stress_trace, stress_summary_trace = full20_trace(low, data)

    direct_summary_df = pd.DataFrame(f_marc_summary + r_marc_summary)
    direct_summary_df.to_csv(output_dir / "vacs_direct_marc_summary.csv", index=False)
    units.to_csv(output_dir / "vacs_direct_marc_units.csv", index=False)
    pd.DataFrame(herding_checks).to_csv(
        output_dir / "vacs_herding_ci_verification.csv", index=False
    )
    sd_audit.to_csv(output_dir / "vacs_table1_sd_verification.csv", index=False)
    stress_summary_trace.to_csv(
        output_dir / "vacs_full20news_trace.csv", index=False
    )

    f_pooled = next(row for row in f_marc_summary if row["slice"] == "Pooled")
    r_pooled = next(row for row in r_marc_summary if row["slice"] == "Pooled")
    f_20ng = next(row for row in f_marc_summary if row["slice"] == "20newsgroups")
    r_20ng = next(row for row in r_marc_summary if row["slice"] == "20newsgroups")
    main_20ng = data["dataset_low_budget_summary"]
    main_20ng = main_20ng[main_20ng["dataset"] == "20newsgroups"].iloc[0]

    vacs_r_marc_diagnostic = {
        "comparison": "VACS-R minus MARC on matched main units",
        "scope": "80 main low-budget knn3 units",
        "positive_unit_count": int((units["delta_r_vs_marc"] > W_T_L_TOLERANCE).sum()),
        "tie_unit_count": int(
            (units["delta_r_vs_marc"].abs() <= W_T_L_TOLERANCE).sum()
        ),
        "negative_unit_count": int(
            (units["delta_r_vs_marc"] < -W_T_L_TOLERANCE).sum()
        ),
        "pooled_delta": r_pooled["delta_mean"],
        "pooled_ci_points": r_pooled["ci_points"],
        "pooled_wtl": r_pooled["wtl"],
    }
    same_20ng_different_comparator = {
        "setting": "20newsgroups, knn3, budgets 1 and 2, 16 matched units",
        "main_dataset_row_comparator": str(main_20ng["best_static_method"]),
        "main_row_vacs_f_minus_comparator": float(main_20ng["gain"]),
        "direct_vacs_f_minus_marc": f_20ng["delta_mean"],
        "direct_vacs_r_minus_marc": r_20ng["delta_mean"],
        "explanation": (
            "The main dataset row uses the per-dataset best static comparator "
            "(random for 20NG), whereas the direct audit uses MARC. The signs "
            "therefore differ without changing the underlying 20NG experiment."
        ),
    }

    source_files = [
        results_root / "tables" / "vacs_all_results.csv",
        results_root / "tables" / "vacs_repeated_validation_results.csv",
        results_root / "tables" / "vacs_low_budget_summary.csv",
        results_root / "tables" / "vacs_dataset_low_budget_summary.csv",
        results_root / "tables" / "vacs_full20news_stress_results.csv",
        results_root / "tables" / "vacs_full20news_stress_summary.csv",
        results_root / "tables" / "vacs_full20news_stress_summary_row.csv",
        results_root / "tables" / "vacs_repeated_validation_comparison_summary.csv",
        manifest_path,
        experiments_root / "run_vacs_experiments.py",
        experiments_root / "build_repeated_validation_comparison_audit.py",
    ]
    source_hashes = {
        portable_source_label(path, results_root, experiments_root): sha256(path)
        for path in source_files
        if path.is_file()
    }

    checks = [
        {
            "check": "main_unit_count",
            "status": "PASS" if len(units) == 80 else "FAIL",
            "value": int(len(units)),
            "expected": 80,
        },
        {
            "check": "direct_marc_summary_rows",
            "status": "PASS" if len(direct_summary_df) == 14 else "FAIL",
            "value": int(len(direct_summary_df)),
            "expected": 14,
        },
        {
            "check": "existing_herding_ci_and_wtl",
            "status": "PASS" if all(
                row["ci_match"] and row["wtl_match"] for row in herding_checks
            ) else "FAIL",
            "value": int(len(herding_checks)),
            "expected": "all checks match",
        },
        {
            "check": "table1_accuracy_and_macro_f1_sample_sd_ddof1",
            "status": "PASS" if bool(
                sd_audit["accuracy_sd_match"].all()
                and sd_audit["macro_f1_sd_match"].all()
            ) else "FAIL",
            "value": int(len(sd_audit)),
            "expected": "all eight methods match",
        },
        {
            "check": "full20news_raw_trace",
            "status": "PASS" if stress_trace["identical_experiments"] else "FAIL",
            "value": stress_trace["identical_experiments"],
            "expected": True,
        },
        {
            "check": "vacs_r_marc_matched_unit_diagnostic",
            "status": (
                "PASS"
                if sum(
                    vacs_r_marc_diagnostic[key]
                    for key in (
                        "positive_unit_count",
                        "tie_unit_count",
                        "negative_unit_count",
                    )
                )
                == len(units)
                else "FAIL"
            ),
            "value": "positive, tied, and negative matched-unit deltas",
            "expected": "three delta categories partition the 80 matched units",
        },
    ]
    pd.DataFrame(checks).to_csv(output_dir / "vacs_revision_audit_checks.csv", index=False)

    audit = {
        "audit_name": "VACS revision round 1 matched MARC audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "results_root": "results",
            "experiments_root": "experiments",
            "source_sha256": source_hashes,
        },
        "scope": {
            "datasets": DATASET_ORDER,
            "seeds": list(range(8)),
            "budget_per_class": LOW_BUDGETS,
            "learner": "knn3",
            "main_units": 80,
            "unit_key": KEYS,
            "vacs_f_definition": "method=vacs rows from vacs_all_results.csv (default single validation split)",
            "vacs_r_definition": "vacs_repeated_validation_results.csv (five validation repeats, fraction 0.25)",
            "marc_definition": "method=marc rows from the same matched main units",
        },
        "reproduction": {
            "bootstrap_function": "numpy.random.default_rng(seed).choice(values, size=(reps, len(values)), replace=True).mean(axis=1); np.quantile(..., [0.025, 0.975])",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "wtl_tolerance": W_T_L_TOLERANCE,
            "summary_float_tolerance": SUMMARY_FLOAT_TOLERANCE,
            "vacs_f_pair_order": "pandas pivot_table(index=[dataset, seed, budget_per_class], columns=method, values=accuracy), default sorted index",
            "vacs_r_pair_order": "left-preserving merge of repeated_validation_results onto MARC rows, as in original comparison audit",
        },
        "direct_vacs_vs_marc": direct_summary_df.to_dict(orient="records"),
        "existing_herding_ci_verification": herding_checks,
        "table1_sample_sd_ddof1": sd_audit.to_dict(orient="records"),
        "full20news_trace": stress_trace,
        "findings": {
            "vacs_r_marc_matched_unit_diagnostic": vacs_r_marc_diagnostic,
            "same_20ng_different_comparator": same_20ng_different_comparator,
            "leave_out_20ng_direct_marc": {
                "vacs_f": next(
                    row for row in f_marc_summary if row["slice"] == "Leave-out-20NG"
                ),
                "vacs_r": next(
                    row for row in r_marc_summary if row["slice"] == "Leave-out-20NG"
                ),
            },
        },
        "checks": checks,
        "outputs": [
            "vacs_revision_audit.py",
            "vacs_direct_marc_summary.csv",
            "vacs_direct_marc_units.csv",
            "vacs_herding_ci_verification.csv",
            "vacs_table1_sd_verification.csv",
            "vacs_full20news_trace.csv",
            "vacs_revision_audit_checks.csv",
            "vacs_revision_audit.json",
        ],
        "caveats": [
            "This is a read-only audit of existing result tables; it runs no new experiments.",
            "VACS-F and VACS-R are compared with MARC on the same dataset/seed/budget keys, but their selector protocols differ by design.",
            "The reported CI checks compare the original displayed two-decimal percentage strings; the audit also retains full-precision computed bounds in the direct summary.",
            "The full-20NG stress files are a raw subset and summary of the same main 20NG low-budget knn3 rows, not an independent rerun.",
        ],
    }
    (output_dir / "vacs_revision_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
