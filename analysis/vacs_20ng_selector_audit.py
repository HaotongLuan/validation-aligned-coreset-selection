from __future__ import annotations

import ast
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# This program reads saved scores and parses source only; it never imports or
# executes the experiment module, loads the dataset, or fits a classifier.

BASE_SELECTORS = ["random", "herding", "kcenter", "boundary", "kmeans", "marc"]
KEYS = ["dataset", "seed", "budget_per_class"]
SEEDS = list(range(8))
BUDGETS = [1, 2]
DATASET = "20newsgroups"
TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit saved full-20-class 20 Newsgroups VACS selector results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Experiment results directory containing tables/.",
    )
    parser.add_argument(
        "--experiment-code",
        type=Path,
        required=True,
        help="Selector experiment source file used for the implementation audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for vacs_20ng_selector_* outputs.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(tables: Path, name: str) -> pd.DataFrame:
    path = tables / name
    require(path.is_file(), f"Missing required input: {name}")
    return pd.read_csv(path)


def scope_filter(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["dataset"] == DATASET)
        & frame["budget_per_class"].isin(BUDGETS)
    ].copy()


def validate_scope(
    all_results: pd.DataFrame,
    repeated_results: pd.DataFrame,
    validation_scores: pd.DataFrame,
    repeated_scores: pd.DataFrame,
    stress_results: pd.DataFrame,
    summary_row: pd.DataFrame,
) -> dict[str, Any]:
    main = scope_filter(all_results)
    stress = scope_filter(stress_results)
    require(len(main) == 128, f"Expected 128 main rows, got {len(main)}")
    require(len(stress) == 128, f"Expected 128 stress rows, got {len(stress)}")
    require(len(stress_results) == 128, "Stress CSV contains rows outside the matched slice")
    require(set(main["learner"]) == {"knn3"}, "Main learner is not knn3")
    expected_keys = {(DATASET, seed, budget) for seed in SEEDS for budget in BUDGETS}
    for frame, label, category in [
        (main, "main", "method"),
        (stress, "stress", "method"),
        (scope_filter(repeated_results), "repeated", None),
        (scope_filter(validation_scores), "validation", "selector"),
        (scope_filter(repeated_scores), "repeated scores", "selector"),
    ]:
        require(set(frame[KEYS].itertuples(index=False, name=None)) == expected_keys,
                f"{label}: seed-budget keys are not exactly seeds 0-7 and budgets 1,2")
        row_key = KEYS + ([category] if category else [])
        require(not frame.duplicated(row_key).any(), f"{label}: duplicate result keys")
        count_column = next((name for name in ["selected", "validation_selected", "validation_selected_mean"]
                             if name in frame.columns), None)
        require(count_column is not None, f"{label}: missing coreset size")
        require(frame[count_column].eq(20 * frame["budget_per_class"]).all(),
                f"{label}: coreset size is not 20 times the per-class budget")
        score_columns = [name for name in frame.columns if name.endswith("accuracy") or name.endswith("macro_f1")]
        require(np.isfinite(frame[score_columns].to_numpy(dtype=float)).all(), f"{label}: nonfinite metrics")
    require(main[KEYS].drop_duplicates().shape[0] == 16, "Main scope is not 16 units")
    require(stress[KEYS].drop_duplicates().shape[0] == 16, "Stress scope is not 16 units")
    require(
        main.groupby(KEYS).size().eq(8).all(),
        "Every main unit must contain eight methods",
    )
    require(
        stress.groupby(KEYS).size().eq(8).all(),
        "Every stress unit must contain eight methods",
    )
    require(
        set(main["method"]) == {"random", "herding", "kcenter", "boundary", "kmeans", "marc", "badge", "vacs"},
        "Unexpected method set in main scope",
    )

    compare_cols = KEYS + ["learner", "method", "selector_detail", "selected"]
    left = main[compare_cols + ["accuracy", "macro_f1"]].sort_values(compare_cols).reset_index(drop=True)
    right = stress[compare_cols + ["accuracy", "macro_f1"]].sort_values(compare_cols).reset_index(drop=True)
    require(left[compare_cols].equals(right[compare_cols]), "Stress and main row identities differ")
    require(
        np.allclose(left[["accuracy", "macro_f1"]], right[["accuracy", "macro_f1"]], atol=TOL, rtol=0.0),
        "Stress and main metrics differ",
    )

    repeated = scope_filter(repeated_results)
    require(len(repeated) == 16, f"Expected 16 repeated-protocol rows, got {len(repeated)}")
    require(repeated[KEYS].drop_duplicates().shape[0] == 16, "Repeated scope is not 16 units")
    require(set(repeated["repeats"]) == {5}, "Repeated protocol does not use five repeats")
    require(set(repeated["validation_fraction"]) == {0.25}, "Repeated protocol is not 25% validation")

    validation = scope_filter(validation_scores)
    repeated_score = scope_filter(repeated_scores)
    require(set(validation["learner"]) == {"knn3"}, "Validation learner is not knn3")
    require(set(repeated_score["repeats"]) == {5}, "Repeated scores do not use five repeats")
    require(set(repeated_score["validation_fraction"]) == {0.25}, "Repeated scores use another fraction")
    require(len(validation) == 96, f"Expected 96 single-split validation rows, got {len(validation)}")
    require(len(repeated_score) == 96, f"Expected 96 repeated validation rows, got {len(repeated_score)}")
    for frame, label in [(validation, "single-split validation"), (repeated_score, "repeated validation")]:
        require(frame[KEYS].drop_duplicates().shape[0] == 16, f"{label} is not 16 units")
        require(frame.groupby(KEYS).size().eq(6).all(), f"{label} must contain six selectors per unit")
        require(set(frame["selector"]) == set(BASE_SELECTORS), f"Unexpected selectors in {label}")

    require(len(summary_row) == 1, "Expected one full-20 summary row")
    require(int(summary_row.iloc[0]["classes"]) == 20, "Summary row is not full 20-class")
    require(int(summary_row.iloc[0]["units"]) == 16, "Summary row is not 16 units")

    return {
        "dataset": DATASET,
        "classes": 20,
        "seeds": SEEDS,
        "budgets_per_class": BUDGETS,
        "units": 16,
        "main_rows": 128,
        "stress_rows": 128,
        "stress_matches_main_rows": True,
        "repeated_validation_repeats": 5,
        "repeated_validation_fraction": 0.25,
    }


def source_contract(code: str) -> tuple[dict[str, bool], dict[str, Any]]:
    tree = ast.parse(code)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def contains(function: str, statement: str) -> bool:
        expected = ast.dump(ast.parse(statement).body[0])
        return any(ast.dump(node) == expected for node in ast.walk(functions[function]))

    def default(function: str, argument: str) -> Any:
        args = functions[function].args
        values = dict(zip([a.arg for a in args.args[-len(args.defaults):]], args.defaults))
        return ast.literal_eval(values[argument])

    constants = {
        node.targets[0].id: node.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    rebuild = 'final_subset = select_indices(X, y, budget_per_class, best["selector"], seed)'
    checks = {
        "fresh_random_rng_in_select_indices": contains("select_indices", "rng = np.random.default_rng(seed)"),
        "random_without_replacement": contains("select_indices", "selected.extend(rng.choice(class_indices, size=k, replace=False).tolist())"),
        "deterministic_class_order": any(isinstance(node, ast.For) and ast.unparse(node.iter) == "np.unique(y)"
                                        for node in ast.walk(functions["select_indices"])),
        "deterministic_subset_order": contains("select_indices", "return np.array(sorted(set(selected)), dtype=int)"),
        "f_full_training_rebuild_same_seed": contains("validation_aligned_select", rebuild),
        "r_full_training_rebuild_same_seed": contains("repeated_validation_aligned_select", rebuild),
        "f_rebuild_default_true": default("validation_aligned_select", "rebuild") is True,
        "f_validation_fraction": default("validation_aligned_select", "val_fraction") == 0.25,
        "r_validation_fraction": default("repeated_validation_aligned_select", "val_fraction") == 0.25,
        "default_classifier_knn3": default("evaluate_subset", "learner") == "knn3",
        "outer_split_shared_seed": contains("run", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)"),
        "static_shared_training_inputs": contains("run", "subset = select_indices(X_train, y_train, budget, method, seed)"),
        "repeated_shared_training_inputs": contains("run", "repeated_subset, repeated_selector, repeated_diagnostics = repeated_validation_aligned_select(X_train, y_train, budget, seed)"),
        "no_rebuild_same_f_inputs": contains("run", "no_rebuild_subset, no_rebuild_selector = validation_aligned_select(X_train, y_train, budget, seed, rebuild=False)"),
        "f_ranking_rule": contains("validation_aligned_select", 'score = (metrics["accuracy"], metrics["macro_f1"], -selectors.index(selector))'),
        "r_ranking_rule": contains("repeated_validation_aligned_select", "score = (mean_accuracy, mean_macro_f1, -selectors.index(selector))"),
        "candidate_order": ast.literal_eval(constants["BASE_SELECTORS"]) == BASE_SELECTORS,
        "five_repeats": ast.literal_eval(constants["REPEATED_VALIDATION_REPEATS"]) == 5,
        "full_categories_no_category_filter": all(not any(k.arg == "categories" for k in node.keywords)
            for node in ast.walk(functions["load_datasets"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fetch_20newsgroups"),
    }
    evidence = {
        name: {"start_line": functions[name].lineno, "end_line": functions[name].end_lineno}
        for name in ["load_datasets", "preprocess_text", "select_indices", "make_classifier",
                     "validation_aligned_select", "repeated_validation_aligned_select", "run"]
    }
    return checks, evidence


def add_ranks(
    scores: pd.DataFrame,
    static: pd.DataFrame,
    protocol: str,
    validation_accuracy: str,
    validation_macro_f1: str,
    selected_by_unit: dict[tuple[int, int], str],
) -> tuple[pd.DataFrame, dict[tuple[int, int], dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    details: dict[tuple[int, int], dict[str, Any]] = {}
    score_scope = scope_filter(scores)
    static_scope = scope_filter(static)

    for (seed, budget), unit_scores in score_scope.groupby(["seed", "budget_per_class"], sort=True):
        unit_scores = unit_scores.copy()
        unit_scores["selector_order"] = unit_scores["selector"].map(
            {selector: index for index, selector in enumerate(BASE_SELECTORS)}
        )
        val_sorted = unit_scores.sort_values(
            [validation_accuracy, validation_macro_f1, "selector_order"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        unit_static = static_scope[
            (static_scope["seed"] == seed)
            & (static_scope["budget_per_class"] == budget)
            & static_scope["method"].isin(BASE_SELECTORS)
        ].copy()
        unit_static["selector_order"] = unit_static["method"].map(
            {selector: index for index, selector in enumerate(BASE_SELECTORS)}
        )
        test_sorted = unit_static.sort_values(
            ["accuracy", "macro_f1", "selector_order"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        require(len(val_sorted) == 6 and len(test_sorted) == 6, "Ranking unit is incomplete")

        validation_order = {row.selector: index + 1 for index, row in val_sorted.iterrows()}
        test_order = {row.method: index + 1 for index, row in test_sorted.iterrows()}
        val_ranks = np.array([validation_order[selector] for selector in BASE_SELECTORS], dtype=float)
        test_ranks = np.array([test_order[selector] for selector in BASE_SELECTORS], dtype=float)
        rho = float(np.corrcoef(val_ranks, test_ranks)[0, 1])
        validation_top = str(val_sorted.iloc[0]["selector"])
        test_top = str(test_sorted.iloc[0]["method"])
        selected = selected_by_unit[(int(seed), int(budget))]
        require(selected == validation_top, f"{protocol} selected selector disagrees with validation top")
        top_test_accuracy = float(test_sorted.iloc[0]["accuracy"])
        selected_test_accuracy = float(
            unit_static.loc[unit_static["method"] == validation_top, "accuracy"].iloc[0]
        )
        details[(int(seed), int(budget))] = {
            "validation_top": validation_top,
            "test_top": test_top,
            "top1_agree": bool(validation_top == test_top),
            "rank_correlation": rho,
            "test_regret": top_test_accuracy - selected_test_accuracy,
        }

        for rank, row in val_sorted.iterrows():
            selector = str(row["selector"])
            test_row = unit_static.loc[unit_static["method"] == selector].iloc[0]
            records.append(
                {
                    "protocol": protocol,
                    "seed": int(seed),
                    "budget_per_class": int(budget),
                    "selector": selector,
                    "validation_accuracy": float(row[validation_accuracy]),
                    "validation_macro_f1": float(row[validation_macro_f1]),
                    "validation_repeats": 5 if protocol == "VACS-R" else 1,
                    "validation_rank": int(rank + 1),
                    "test_accuracy": float(test_row["accuracy"]),
                    "test_macro_f1": float(test_row["macro_f1"]),
                    "test_rank": int(test_order[selector]),
                    "validation_top": validation_top,
                    "test_top": test_top,
                    "selected_by_protocol": bool(selector == selected),
                }
            )

    return pd.DataFrame(records), details


def metric_equal(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, atol=TOL, rtol=0.0))


def build_units(
    all_results: pd.DataFrame,
    repeated_results: pd.DataFrame,
    f_rank_details: dict[tuple[int, int], dict[str, Any]],
    r_rank_details: dict[tuple[int, int], dict[str, Any]],
) -> pd.DataFrame:
    main = scope_filter(all_results)
    repeated = scope_filter(repeated_results)
    static = main[main["method"].isin(BASE_SELECTORS)].set_index(["seed", "budget_per_class", "method"])
    f = main[main["method"] == "vacs"].set_index(["seed", "budget_per_class"])
    r = repeated.set_index(["seed", "budget_per_class"])
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        for budget in BUDGETS:
            key = (seed, budget)
            f_row = f.loc[key]
            r_row = r.loc[key]
            random_row = static.loc[(seed, budget, "random")]
            herding_row = static.loc[(seed, budget, "herding")]
            f_selector = str(f_row["selector_detail"])
            r_selector = str(r_row["selector_detail"])
            f_static = static.loc[(seed, budget, f_selector)]
            r_static = static.loc[(seed, budget, r_selector)]
            f_detail = f_rank_details[key]
            r_detail = r_rank_details[key]
            rows.append(
                {
                    "seed": seed,
                    "budget_per_class": budget,
                    "unit_id": f"seed{seed}_budget{budget}",
                    "f_selector": f_selector,
                    "f_accuracy": float(f_row["accuracy"]),
                    "random_accuracy": float(random_row["accuracy"]),
                    "herding_accuracy": float(herding_row["accuracy"]),
                    "marc_accuracy": float(static.loc[(seed, budget, "marc"), "accuracy"]),
                    "boundary_accuracy": float(static.loc[(seed, budget, "boundary"), "accuracy"]),
                    "f_delta_vs_herding_pp": 100.0 * (float(f_row["accuracy"]) - float(herding_row["accuracy"])),
                    "f_routing_contribution_vs_random_pp": 100.0 * (float(f_static["accuracy"]) - float(random_row["accuracy"])),
                    "f_rebuild_residual_vs_selected_static_pp": 100.0 * (float(f_row["accuracy"]) - float(f_static["accuracy"])),
                    "f_delta_vs_random_pp": 100.0 * (float(f_row["accuracy"]) - float(random_row["accuracy"])),
                    "f_macro_f1_delta_vs_random_pp": 100.0 * (float(f_row["macro_f1"]) - float(random_row["macro_f1"])),
                    "f_matches_selected_static_accuracy": metric_equal(float(f_row["accuracy"]), float(f_static["accuracy"])),
                    "f_matches_selected_static_macro_f1": metric_equal(float(f_row["macro_f1"]), float(f_static["macro_f1"])),
                    "f_matches_static_random_accuracy": (
                        metric_equal(float(f_row["accuracy"]), float(random_row["accuracy"])) if f_selector == "random" else None
                    ),
                    "f_matches_static_random_macro_f1": (
                        metric_equal(float(f_row["macro_f1"]), float(random_row["macro_f1"])) if f_selector == "random" else None
                    ),
                    "r_selector": r_selector,
                    "r_accuracy": float(r_row["accuracy"]),
                    "r_delta_vs_herding_pp": 100.0 * (float(r_row["accuracy"]) - float(herding_row["accuracy"])),
                    "r_routing_contribution_vs_random_pp": 100.0 * (float(r_static["accuracy"]) - float(random_row["accuracy"])),
                    "r_rebuild_residual_vs_selected_static_pp": 100.0 * (float(r_row["accuracy"]) - float(r_static["accuracy"])),
                    "r_delta_vs_random_pp": 100.0 * (float(r_row["accuracy"]) - float(random_row["accuracy"])),
                    "r_macro_f1_delta_vs_random_pp": 100.0 * (float(r_row["macro_f1"]) - float(random_row["macro_f1"])),
                    "r_matches_selected_static_accuracy": metric_equal(float(r_row["accuracy"]), float(r_static["accuracy"])),
                    "r_matches_selected_static_macro_f1": metric_equal(float(r_row["macro_f1"]), float(r_static["macro_f1"])),
                    "r_matches_static_random_accuracy": (
                        metric_equal(float(r_row["accuracy"]), float(random_row["accuracy"])) if r_selector == "random" else None
                    ),
                    "r_matches_static_random_macro_f1": (
                        metric_equal(float(r_row["macro_f1"]), float(random_row["macro_f1"])) if r_selector == "random" else None
                    ),
                    "f_validation_top": f_detail["validation_top"],
                    "f_test_top": f_detail["test_top"],
                    "f_top1_transfer": f_detail["top1_agree"],
                    "f_rank_correlation": f_detail["rank_correlation"],
                    "f_test_regret_vs_validation_top_pp": 100.0 * f_detail["test_regret"],
                    "r_validation_top": r_detail["validation_top"],
                    "r_test_top": r_detail["test_top"],
                    "r_top1_transfer": r_detail["top1_agree"],
                    "r_rank_correlation": r_detail["rank_correlation"],
                    "r_test_regret_vs_validation_top_pp": 100.0 * r_detail["test_regret"],
                }
            )
    result = pd.DataFrame(rows)
    require(len(result) == 16, "Unit audit did not produce 16 rows")
    return result


def wtl(values: pd.Series) -> str:
    wins = int((values > TOL).sum())
    ties = int(values.abs().le(TOL).sum())
    losses = int((values < -TOL).sum())
    return f"{wins}/{ties}/{losses}"


def protocol_summary(units: pd.DataFrame, protocol: str) -> dict[str, Any]:
    selector = f"{protocol}_selector"
    accuracy = f"{protocol}_accuracy"
    delta = f"{protocol}_delta_vs_random_pp"
    transfer = f"{protocol}_top1_transfer"
    rho = f"{protocol}_rank_correlation"
    regret = f"{protocol}_test_regret_vs_validation_top_pp"
    counts = units[selector].value_counts().to_dict()
    delta_series = units[delta]
    return {
        "selector_counts": {name: int(counts.get(name, 0)) for name in BASE_SELECTORS},
        "mean_accuracy": float(units[accuracy].mean()),
        "mean_random_accuracy": float(units["random_accuracy"].mean()),
        "mean_herding_accuracy": float(units["herding_accuracy"].mean()),
        "mean_delta_vs_herding_pp": float(units[f"{protocol}_delta_vs_herding_pp"].mean()),
        "wins_ties_losses_vs_herding": wtl(units[f"{protocol}_delta_vs_herding_pp"]),
        "mean_delta_vs_random_pp": float(delta_series.mean()),
        "wins_ties_losses_vs_random": wtl(delta_series),
        "top1_transfer_rate": float(units[transfer].mean()),
        "mean_rank_correlation": float(units[rho].mean()),
        "mean_test_regret_vs_validation_top_pp": float(units[regret].mean()),
        "severe_regret_rate_over_5pp": float((units[regret] > 5.0).mean()),
    }


def decompose(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in ["f", "r"]:
        for selector, selected in units.groupby(f"{protocol}_selector", sort=True):
            deltas = selected[f"{protocol}_delta_vs_random_pp"]
            rows.append({
                "protocol": f"VACS-{protocol.upper()}",
                "selected_rule": selector,
                "units": len(selected),
                "conditional_mean_accuracy_percent": 100.0 * selected[f"{protocol}_accuracy"].mean(),
                "conditional_mean_random_accuracy_percent": 100.0 * selected["random_accuracy"].mean(),
                "conditional_delta_vs_random_pp": deltas.mean(),
                "contribution_to_all_16_units_delta_pp": deltas.sum() / len(units),
                "positive_contribution_to_all_16_units_pp": deltas[deltas > TOL].sum() / len(units),
                "negative_contribution_to_all_16_units_pp": deltas[deltas < -TOL].sum() / len(units),
                "wins_ties_losses_vs_random": wtl(deltas),
                "max_abs_rebuild_residual_vs_selected_static_pp": selected[f"{protocol}_rebuild_residual_vs_selected_static_pp"].abs().max(),
            })
    return pd.DataFrame(rows)


def make_summary_rows(
    scope: dict[str, Any],
    f_summary: dict[str, Any],
    r_summary: dict[str, Any],
    units: pd.DataFrame,
    code_contract: dict[str, bool],
    no_rebuild_matches: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, unit: str = "", details: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "unit": unit, "details": details})

    add("scope", "dataset", DATASET)
    add("scope", "classes", scope["classes"], "classes")
    add("scope", "seeds", "0-7", "seed ids")
    add("scope", "budgets_per_class", "1,2", "examples per class")
    add("scope", "units", scope["units"], "seed-budget units")
    add("provenance", "stress_rows_match_main_rows", scope["stress_matches_main_rows"], "boolean")
    add("VACS-F", "boundary_selector_count", f_summary["selector_counts"]["boundary"], "units")
    add("VACS-F", "random_selector_count", f_summary["selector_counts"]["random"], "units")
    add("VACS-F", "mean_accuracy", f_summary["mean_accuracy"], "proportion")
    add("VACS-F", "mean_random_accuracy", f_summary["mean_random_accuracy"], "proportion")
    add("VACS-F", "mean_delta_vs_random", f_summary["mean_delta_vs_random_pp"], "percentage points")
    add("VACS-F", "wins_ties_losses_vs_random", f_summary["wins_ties_losses_vs_random"], "W/T/L")
    add("VACS-F", "mean_delta_vs_herding", f_summary["mean_delta_vs_herding_pp"], "percentage points")
    add("VACS-F", "wins_ties_losses_vs_herding", f_summary["wins_ties_losses_vs_herding"], "W/T/L")
    add("VACS-R", "random_selector_count", r_summary["selector_counts"]["random"], "units")
    add("VACS-R", "boundary_selector_count", r_summary["selector_counts"]["boundary"], "units")
    add("VACS-R", "mean_accuracy", r_summary["mean_accuracy"], "proportion")
    add("VACS-R", "mean_random_accuracy", r_summary["mean_random_accuracy"], "proportion")
    add("VACS-R", "mean_delta_vs_random", r_summary["mean_delta_vs_random_pp"], "percentage points")
    add("VACS-R", "wins_ties_losses_vs_random", r_summary["wins_ties_losses_vs_random"], "W/T/L")
    add("VACS-R", "mean_delta_vs_herding", r_summary["mean_delta_vs_herding_pp"], "percentage points")
    add("VACS-R", "wins_ties_losses_vs_herding", r_summary["wins_ties_losses_vs_herding"], "W/T/L")
    add("rebuild", "F_selected_matches_static_selected", int(units["f_matches_selected_static_accuracy"].all()), "boolean")
    add("rebuild", "R_selected_matches_static_selected", int(units["r_matches_selected_static_accuracy"].all()), "boolean")
    f_random = units[units["f_selector"] == "random"]
    r_random = units[units["r_selector"] == "random"]
    add("rebuild", "F_random_matches_static_random", f"{int(f_random['f_matches_static_random_accuracy'].sum())}/{len(f_random)}", "units")
    add("rebuild", "R_random_matches_static_random", f"{int(r_random['r_matches_static_random_accuracy'].sum())}/{len(r_random)}", "units")
    add("rebuild", "no_rebuild_matches_static_selected", f"{no_rebuild_matches}/16", "accuracy rows")
    add("validation_transfer", "F_top1_agreement", f_summary["top1_transfer_rate"], "proportion")
    add("validation_transfer", "F_mean_rank_correlation", f_summary["mean_rank_correlation"], "mean Pearson rank correlation")
    add("validation_transfer", "F_mean_test_regret", f_summary["mean_test_regret_vs_validation_top_pp"], "percentage points")
    add("validation_transfer", "R_top1_agreement", r_summary["top1_transfer_rate"], "proportion")
    add("validation_transfer", "R_mean_rank_correlation", r_summary["mean_rank_correlation"], "mean Pearson rank correlation")
    add("validation_transfer", "R_mean_test_regret", r_summary["mean_test_regret_vs_validation_top_pp"], "percentage points")
    add("code_contract", "all_required_rebuild_checks", int(all(code_contract.values())), "boolean")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    tables = args.results_root / "tables"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = read_table(tables, "vacs_all_results.csv")
    repeated_results = read_table(tables, "vacs_repeated_validation_results.csv")
    validation_scores = read_table(tables, "vacs_validation_scores.csv")
    repeated_scores = read_table(tables, "vacs_repeated_validation_scores.csv")
    stress_results = read_table(tables, "vacs_full20news_stress_results.csv")
    summary_row = read_table(tables, "vacs_full20news_stress_summary_row.csv")
    ablation_results = read_table(tables, "vacs_ablation_results.csv")

    scope = validate_scope(
        all_results,
        repeated_results,
        validation_scores,
        repeated_scores,
        stress_results,
        summary_row,
    )
    code_contract, code_evidence = source_contract(args.experiment_code.read_text(encoding="utf-8"))
    require(all(code_contract.values()), f"Experiment code contract failed: {code_contract}")

    main_scope = scope_filter(all_results)
    repeated_scope = scope_filter(repeated_results)
    f_selected = {
        (int(row.seed), int(row.budget_per_class)): str(row.selector_detail)
        for row in main_scope[main_scope["method"] == "vacs"].itertuples()
    }
    r_selected = {
        (int(row.seed), int(row.budget_per_class)): str(row.selector_detail)
        for row in repeated_scope.itertuples()
    }
    f_rankings, f_rank_details = add_ranks(
        validation_scores,
        all_results,
        "VACS-F",
        "validation_accuracy",
        "validation_macro_f1",
        f_selected,
    )
    r_rankings, r_rank_details = add_ranks(
        repeated_scores,
        all_results,
        "VACS-R",
        "mean_validation_accuracy",
        "mean_validation_macro_f1",
        r_selected,
    )
    rankings = pd.concat([f_rankings, r_rankings], ignore_index=True)
    units = build_units(all_results, repeated_results, f_rank_details, r_rank_details)

    no_rebuild = scope_filter(ablation_results)
    no_rebuild = no_rebuild[no_rebuild["ablation_method"] == "vacs_no_rebuild"]
    static = main_scope[main_scope["method"].isin(BASE_SELECTORS)]
    no_rebuild_join = no_rebuild.merge(
        static[["seed", "budget_per_class", "method", "accuracy"]],
        left_on=["seed", "budget_per_class", "selector_detail"],
        right_on=["seed", "budget_per_class", "method"],
        suffixes=("", "_static"),
    )
    no_rebuild_matches = int(
        np.isclose(no_rebuild_join["accuracy"], no_rebuild_join["accuracy_static"], atol=TOL, rtol=0.0).sum()
    )
    require(len(no_rebuild_join) == 16, "No-rebuild ablation is incomplete")

    f_summary = protocol_summary(units, "f")
    r_summary = protocol_summary(units, "r")
    summary = make_summary_rows(scope, f_summary, r_summary, units, code_contract, no_rebuild_matches)
    decomposition = decompose(units)

    input_names = [
        "vacs_all_results.csv",
        "vacs_repeated_validation_results.csv",
        "vacs_validation_scores.csv",
        "vacs_repeated_validation_scores.csv",
        "vacs_full20news_stress_results.csv",
        "vacs_full20news_stress_summary_row.csv",
        "vacs_ablation_results.csv",
    ]
    input_hashes = {name: sha256(tables / name) for name in input_names}
    audit = {
        "audit_name": "20ng_selector_audit",
        "scope": scope,
        "inputs": {"files": input_names, "sha256": input_hashes},
        "source_code": {
            "file": args.experiment_code.name,
            "contract_checks": code_contract,
            "function_line_evidence": code_evidence,
            "rebuild_interpretation": (
                "The selected rule is rebuilt on the full outer training split; the random selector "
                "creates a fresh default_rng(seed) inside each call. Therefore a selected random rebuild "
                "uses the same random coreset as static random for the same seed and budget."
            ),
        },
        "selector_choices": {"VACS-F": f_summary["selector_counts"], "VACS-R": r_summary["selector_counts"]},
        "comparisons_vs_random": {"VACS-F": f_summary, "VACS-R": r_summary},
        "rebuild_decomposition": {
            "F_selected_matches_corresponding_static_metrics": bool(units["f_matches_selected_static_accuracy"].all()),
            "R_selected_matches_corresponding_static_metrics": bool(units["r_matches_selected_static_accuracy"].all()),
            "F_selected_random_matches_static_random": f"{int(units.loc[units.f_selector == 'random', 'f_matches_static_random_accuracy'].sum())}/7",
            "R_selected_random_matches_static_random": f"{int(units.loc[units.r_selector == 'random', 'r_matches_static_random_accuracy'].sum())}/14",
            "no_rebuild_matches_corresponding_static_metrics": f"{no_rebuild_matches}/16",
            "selector_conditioned_decomposition": decomposition.to_dict(orient="records"),
            "aggregate_csv_limitation": "The saved CSVs contain metrics and selector labels, not selected row indices; code semantics establish identity, while metric equality is an independent consistency check.",
        },
        "validation_ranking_transfer": {
            "VACS-F": {
                "top1_agreement": f_summary["top1_transfer_rate"],
                "mean_rank_correlation": f_summary["mean_rank_correlation"],
                "mean_test_regret_pp": f_summary["mean_test_regret_vs_validation_top_pp"],
                "severe_regret_rate_over_5pp": f_summary["severe_regret_rate_over_5pp"],
            },
            "VACS-R": {
                "top1_agreement": r_summary["top1_transfer_rate"],
                "mean_rank_correlation": r_summary["mean_rank_correlation"],
                "mean_test_regret_pp": r_summary["mean_test_regret_vs_validation_top_pp"],
                "severe_regret_rate_over_5pp": r_summary["severe_regret_rate_over_5pp"],
            },
            "supports": "Within these 16 saved units, the logs quantify validation top choice, test top choice, rank correlation, and test regret.",
            "does_not_support": [
                "A general validation-to-test ranking law beyond this dataset, budgets, seeds, and learner.",
                "A causal mechanism for why boundary or random is selected.",
                "Independent replication across 16 datasets; the 16 rows are seed-budget units from one dataset.",
            ],
        },
        "outputs": {
            "units_csv": "vacs_20ng_selector_units.csv",
            "rankings_csv": "vacs_20ng_selector_rankings.csv",
            "summary_csv": "vacs_20ng_selector_summary.csv",
            "decomposition_csv": "vacs_20ng_selector_decomposition.csv",
        },
    }

    units.to_csv(output_dir / "vacs_20ng_selector_units.csv", index=False, float_format="%.12g")
    rankings.to_csv(output_dir / "vacs_20ng_selector_rankings.csv", index=False, float_format="%.12g")
    summary.to_csv(output_dir / "vacs_20ng_selector_summary.csv", index=False, float_format="%.12g")
    decomposition.to_csv(output_dir / "vacs_20ng_selector_decomposition.csv", index=False, float_format="%.12g")
    (output_dir / "vacs_20ng_selector_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
