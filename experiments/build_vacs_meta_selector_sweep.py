from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from run_vacs_experiments import (
    BASE_SELECTORS,
    LOW_BUDGETS,
    METHODS,
    bootstrap_mean_ci,
    fmt_signed,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

UNIT_COLS = ["dataset", "seed", "budget_per_class"]
STATIC_METHODS = [method for method in METHODS if method not in {"vacs", "badge"}]

PORTFOLIOS = {
    "all6": ["random", "herding", "kcenter", "boundary", "kmeans", "marc"],
    "no_boundary": ["random", "herding", "kcenter", "kmeans", "marc"],
    "no_random": ["herding", "kcenter", "boundary", "kmeans", "marc"],
    "representative4": ["herding", "kcenter", "kmeans", "marc"],
    "centroid3": ["herding", "kmeans", "marc"],
    "coverage3": ["herding", "kcenter", "kmeans"],
}

TIE_ORDERS = {
    "default": BASE_SELECTORS,
    "herding_first": ["herding", "kmeans", "marc", "kcenter", "random", "boundary"],
    "kmeans_first": ["kmeans", "herding", "marc", "kcenter", "random", "boundary"],
    "marc_first": ["marc", "kmeans", "herding", "kcenter", "random", "boundary"],
}


def read_low_results() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "vacs_all_results.csv")
    return df[
        (df["learner"] == "knn3")
        & (df["budget_per_class"].isin(LOW_BUDGETS))
        & (df["method"].isin(STATIC_METHODS + ["vacs"]))
    ].copy()


def selector_accuracy_map(low: pd.DataFrame) -> pd.DataFrame:
    static = low[low["method"].isin(STATIC_METHODS)][
        UNIT_COLS + ["method", "accuracy", "macro_f1"]
    ].copy()
    return static.rename(
        columns={
            "method": "selector",
            "accuracy": "test_accuracy",
            "macro_f1": "test_macro_f1",
        }
    )


def load_score_sources() -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}

    single = pd.read_csv(TABLES / "vacs_validation_scores.csv")
    single = single.rename(
        columns={
            "validation_accuracy": "score_accuracy",
            "validation_macro_f1": "score_macro_f1",
        }
    )
    single["score_std_accuracy"] = 0.0
    sources["single_0.25"] = single[UNIT_COLS + ["selector", "score_accuracy", "score_macro_f1", "score_std_accuracy"]]

    repeated = pd.read_csv(TABLES / "vacs_repeated_validation_scores.csv")
    repeated = repeated.rename(
        columns={
            "mean_validation_accuracy": "score_accuracy",
            "mean_validation_macro_f1": "score_macro_f1",
            "std_validation_accuracy": "score_std_accuracy",
        }
    )
    sources["repeat5_0.25"] = repeated[UNIT_COLS + ["selector", "score_accuracy", "score_macro_f1", "score_std_accuracy"]]

    size_path = TABLES / "vacs_validation_size_reliability_scores.csv"
    if size_path.is_file():
        size_scores = pd.read_csv(size_path)
        for frac, sub in size_scores.groupby("validation_fraction"):
            renamed = sub.rename(
                columns={
                    "validation_accuracy": "score_accuracy",
                    "validation_macro_f1": "score_macro_f1",
                }
            ).copy()
            renamed["score_std_accuracy"] = 0.0
            sources[f"single_{float(frac):.2f}"] = renamed[
                UNIT_COLS + ["selector", "score_accuracy", "score_macro_f1", "score_std_accuracy"]
            ]

    return sources


def choose_for_unit(
    group: pd.DataFrame,
    portfolio: list[str],
    tie_order: list[str],
    lcb_lambda: float,
    fallback: str | None,
    gap_threshold: float,
) -> dict[str, object]:
    candidates = group[group["selector"].isin(portfolio)].copy()
    if candidates.empty:
        raise ValueError("empty candidate set")
    order_map = {selector: i for i, selector in enumerate(tie_order)}
    candidates["primary"] = candidates["score_accuracy"] - lcb_lambda * candidates["score_std_accuracy"]
    candidates["tie_rank"] = candidates["selector"].map(order_map).fillna(len(order_map)).astype(int)
    candidates = candidates.sort_values(
        ["primary", "score_macro_f1", "tie_rank"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    top = candidates.iloc[0]
    second_primary = float(candidates.iloc[1]["primary"]) if len(candidates) > 1 else -np.inf
    gap = float(top["primary"] - second_primary)
    selector = str(top["selector"])
    used_fallback = False
    if fallback is not None and fallback in portfolio and gap <= gap_threshold:
        selector = fallback
        used_fallback = True
    return {
        "chosen_selector": selector,
        "validation_primary": float(top["primary"]),
        "validation_gap": gap,
        "used_fallback": used_fallback,
    }


def build_variant_predictions(scores: pd.DataFrame, static_scores: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    available_units = scores[UNIT_COLS].drop_duplicates()
    unit_count = len(available_units)

    lambdas = [0.0]
    if float(scores["score_std_accuracy"].abs().max()) > 0:
        lambdas = [0.0, 0.25, 0.5, 1.0]
    fallbacks = [None, "herding", "kmeans"]
    gap_thresholds = [0.0, 0.005, 0.01, 0.02, 0.05]

    for portfolio_name, portfolio in PORTFOLIOS.items():
        for tie_name, tie_order in TIE_ORDERS.items():
            for lcb_lambda, fallback, gap_threshold in product(lambdas, fallbacks, gap_thresholds):
                if fallback is None and gap_threshold != 0.0:
                    continue
                chosen_rows = []
                for keys, group in scores.groupby(UNIT_COLS, sort=False):
                    choice = choose_for_unit(
                        group,
                        portfolio=portfolio,
                        tie_order=tie_order,
                        lcb_lambda=float(lcb_lambda),
                        fallback=fallback,
                        gap_threshold=float(gap_threshold),
                    )
                    chosen_rows.append({
                        "dataset": keys[0],
                        "seed": int(keys[1]),
                        "budget_per_class": int(keys[2]),
                        **choice,
                    })
                chosen = pd.DataFrame(chosen_rows)
                merged = chosen.merge(
                    static_scores,
                    left_on=UNIT_COLS + ["chosen_selector"],
                    right_on=UNIT_COLS + ["selector"],
                    how="inner",
                )
                if len(merged) != unit_count:
                    continue
                merged["portfolio"] = portfolio_name
                merged["tie_order"] = tie_name
                merged["lcb_lambda"] = float(lcb_lambda)
                merged["fallback"] = "none" if fallback is None else fallback
                merged["gap_threshold"] = float(gap_threshold)
                merged["variant"] = (
                    portfolio_name
                    + "|"
                    + tie_name
                    + f"|lcb={lcb_lambda:g}|fb={merged['fallback'].iloc[0]}|gap={gap_threshold:g}"
                )
                frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def summarize_predictions(predictions: pd.DataFrame, low: pd.DataFrame, source: str) -> pd.DataFrame:
    static_low = low[low["method"].isin(STATIC_METHODS)]
    pooled_static = static_low.groupby("method")["accuracy"].mean()
    best_pooled_method = str(pooled_static.idxmax())
    best_pooled_acc = float(pooled_static.max())

    dataset_best_parts = []
    for dataset, sub in static_low.groupby("dataset"):
        means = sub.groupby("method")["accuracy"].mean()
        method = str(means.idxmax())
        dataset_best_parts.append(sub[sub["method"] == method][UNIT_COLS + ["accuracy"]])
    dataset_best = pd.concat(dataset_best_parts, ignore_index=True).rename(
        columns={"accuracy": "dataset_best_accuracy"}
    )

    rows = []
    for variant, sub in predictions.groupby("variant"):
        merged = sub.merge(dataset_best, on=UNIT_COLS, how="left")
        deltas = (merged["test_accuracy"] - merged["dataset_best_accuracy"]).to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(deltas)
        selector_counts = (
            sub["chosen_selector"].value_counts().reindex(STATIC_METHODS, fill_value=0).to_dict()
        )
        rows.append({
            "source": source,
            "variant": variant,
            "units": int(len(sub)),
            "accuracy": float(sub["test_accuracy"].mean()),
            "macro_f1": float(sub["test_macro_f1"].mean()),
            "best_pooled_static_method": best_pooled_method,
            "best_pooled_static_accuracy": best_pooled_acc,
            "delta_vs_best_pooled_static": float(sub["test_accuracy"].mean() - best_pooled_acc),
            "delta_vs_dataset_best": float(deltas.mean()),
            "dataset_best_ci": f"[{fmt_signed(lo)},{fmt_signed(hi)}]",
            "fallback_rate": float(sub["used_fallback"].mean()),
            "median_validation_gap": float(sub["validation_gap"].median()),
            "selector_counts": json.dumps(selector_counts, sort_keys=True),
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def leave_one_dataset_out(summary: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for held_out in sorted(predictions["dataset"].unique()):
        train = predictions[predictions["dataset"] != held_out]
        test = predictions[predictions["dataset"] == held_out]
        train_scores = train.groupby("variant")["test_accuracy"].mean()
        best_variant = str(train_scores.idxmax())
        chosen = test[test["variant"] == best_variant]
        rows.append({
            "held_out_dataset": held_out,
            "selected_variant": best_variant,
            "train_accuracy": float(train_scores.loc[best_variant]),
            "heldout_accuracy": float(chosen["test_accuracy"].mean()),
            "heldout_macro_f1": float(chosen["test_macro_f1"].mean()),
            "heldout_units": int(len(chosen)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    low = read_low_results()
    static_scores = selector_accuracy_map(low)
    sources = load_score_sources()

    all_summary = []
    all_lodo = []
    for source, scores in sources.items():
        unit_count = scores[UNIT_COLS].drop_duplicates().shape[0]
        if unit_count < 80:
            print(f"[warn] {source} has {unit_count} units; keeping it as a coverage-limited diagnostic.")
        predictions = build_variant_predictions(scores, static_scores)
        predictions["source"] = source
        summary = summarize_predictions(predictions, low, source)
        prediction_path = TABLES / f"vacs_meta_selector_predictions_{source}.csv"
        summary_path = TABLES / f"vacs_meta_selector_sweep_{source}.csv"
        predictions.to_csv(prediction_path, index=False)
        summary.to_csv(summary_path, index=False)
        all_summary.append(summary)

        lodo = leave_one_dataset_out(summary, predictions)
        lodo["source"] = source
        all_lodo.append(lodo)

    combined = pd.concat(all_summary, ignore_index=True).sort_values("accuracy", ascending=False)
    combined.to_csv(TABLES / "vacs_meta_selector_sweep_summary.csv", index=False)
    lodo_combined = pd.concat(all_lodo, ignore_index=True)
    lodo_combined.to_csv(TABLES / "vacs_meta_selector_leave_one_dataset_out.csv", index=False)

    top = combined.head(12).copy()
    manifest = {
        "warning": (
            "This is an exploratory meta-selector sweep over saved validation scores. "
            "Rows selected by pooled test accuracy are not submission-safe claims unless "
            "validated by a pre-registered or leave-one-dataset-out protocol."
        ),
        "outputs": [
            "results/tables/vacs_meta_selector_sweep_summary.csv",
            "results/tables/vacs_meta_selector_leave_one_dataset_out.csv",
        ],
        "top_exploratory": top.to_dict(orient="records"),
        "leave_one_dataset_out_mean": float(lodo_combined["heldout_accuracy"].mean()),
    }
    (RESULTS / "vacs_meta_selector_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(top[[
        "source",
        "variant",
        "units",
        "accuracy",
        "delta_vs_best_pooled_static",
        "delta_vs_dataset_best",
        "fallback_rate",
        "selector_counts",
    ]].to_string(index=False))
    print("META_SELECTOR_SWEEP_OK")


if __name__ == "__main__":
    main()
