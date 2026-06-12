from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SR_MANUSCRIPT = ROOT / "scientific_reports" / "source" / "manuscript_sr.tex"
SR_MANUSCRIPT_PDF = ROOT / "scientific_reports" / "source" / "manuscript_sr.pdf"
SR_SUPPLEMENTARY_PDF = ROOT / "scientific_reports" / "source" / "supplementary_information.pdf"
RESULTS = ROOT / "results"

METHOD_LABELS = {
    "random": "Random",
    "herding": "Herding",
    "kcenter": "K-center",
    "boundary": "Boundary",
    "kmeans": "K-means",
    "marc": "MARC",
    "badge": "BADGE",
    "vacs": "VACS",
}

METHOD_PROSE = {
    **METHOD_LABELS,
    "random": "random",
    "herding": "herding",
    "kcenter": "k-center",
    "boundary": "boundary",
    "kmeans": "k-means",
    "marc": "MARC",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}"


def signed_points(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:+.{digits}f}"


def points(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}"


def english_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"OK {label}")


def require_contains(text: str, needle: str, label: str) -> None:
    require(needle in text, f"{label}: found `{needle}`")


def require_absent(text: str, needle: str, label: str) -> None:
    require(needle not in text, f"{label}: absent `{needle}`")


def main_scientific_reports() -> None:
    """Lightweight consistency audit for the public Scientific Reports package."""
    manuscript = read_text(SR_MANUSCRIPT)
    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))

    require(SR_MANUSCRIPT_PDF.is_file(), "Scientific Reports manuscript PDF exists")
    require(SR_SUPPLEMENTARY_PDF.is_file(), "Scientific Reports supplementary PDF exists")
    require(
        (ROOT / "scientific_reports" / "submission_materials" / "reference_verification_report.md").is_file(),
        "reference verification report exists",
    )

    headline = manifest["headline"]
    require_contains(
        manuscript,
        f"VACS-F reaches {pct(headline['vacs_low_budget_accuracy'])}\\% mean accuracy",
        "SR VACS-F headline accuracy",
    )
    require_contains(
        manuscript,
        "numerically tying the strongest pooled static selector",
        "SR VACS-F near-tie framing",
    )
    require_contains(
        manuscript,
        "VACS supports a narrow conclusion",
        "SR bounded abstract claim",
    )
    require_contains(
        manuscript,
        "not a broadly superior coreset rule",
        "SR no broad-superiority claim",
    )

    repeated_rows = read_csv(RESULTS / "tables" / "vacs_repeated_validation_summary.csv")
    pooled_repeated = next(row for row in repeated_rows if row["slice"] == "Pooled")
    repeated_comparison_rows = read_csv(
        RESULTS / "tables" / "vacs_repeated_validation_comparison_summary.csv"
    )
    repeated_vs_best_static = next(
        row
        for row in repeated_comparison_rows
        if row["comparison"] == "best_static" and row["slice"] == "Pooled"
    )
    repeated_vs_dataset_best = next(
        row
        for row in repeated_comparison_rows
        if row["comparison"] == "dataset_best" and row["slice"] == "Pooled"
    )
    require_contains(
        manuscript,
        f"VACS-R reaches {pct(float(pooled_repeated['repeated_accuracy']))}\\% pooled accuracy",
        "SR VACS-R pooled accuracy",
    )
    require_contains(
        manuscript,
        (
            f"improves over herding by {repeated_vs_best_static['delta_points']} points, "
            f"with a 95\\% bootstrap interval of {repeated_vs_best_static['ci_points']}"
        ),
        "SR VACS-R paired gain",
    )
    require_contains(
        manuscript,
        (
            f"essentially tied at {repeated_vs_dataset_best['delta_points']} points "
            f"with interval {repeated_vs_dataset_best['ci_points']}"
        ),
        "SR VACS-R hindsight-static boundary",
    )

    covtype_confirmatory_rows = read_csv(
        RESULTS / "tables" / "vacs_covtype_confirmatory_paired_summary.csv"
    )
    covtype_vs_herding = next(
        row
        for row in covtype_confirmatory_rows
        if row["target"] == "vacs_r" and row["comparator"] == "herding"
    )
    covtype_vs_marc = next(
        row
        for row in covtype_confirmatory_rows
        if row["target"] == "vacs_r" and row["comparator"] == "marc"
    )
    require_contains(
        manuscript,
        (
            f"VACS-R reaches {pct(float(covtype_vs_herding['target_accuracy']))}\\% mean accuracy, "
            f"improves over herding by {covtype_vs_herding['delta_points']} points"
        ),
        "SR Covertype confirmatory herding result",
    )
    require_contains(
        manuscript,
        (
            f"exactly matches MARC at {covtype_vs_marc['delta_points']} points "
            f"with a {covtype_vs_marc['wtl']} pattern"
        ),
        "SR Covertype MARC boundary",
    )

    require_contains(
        manuscript,
        "Frozen text-embedding probes on full 20-class 20 Newsgroups are boundary evidence",
        "SR frozen text boundary wording",
    )
    require_contains(
        manuscript,
        "Frozen image-embedding probes also tie herding",
        "SR frozen image boundary wording",
    )
    require_contains(
        manuscript,
        "https://github.com/HaotongLuan/validation-aligned-coreset-selection",
        "SR GitHub code availability URL",
    )
    print("CLAIM_CONSISTENCY_OK")


def main() -> None:
    paper_section_paths = sorted((PAPER / "sections").glob("*.tex"))
    if not paper_section_paths:
        main_scientific_reports()
        return

    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
    transformer_manifest = json.loads(
        (RESULTS / "transformer_embedding_probe_manifest.json").read_text(encoding="utf-8")
    )
    torchvision_manifest = json.loads(
        (RESULTS / "vacs_torchvision_embedding_probe_manifest.json").read_text(encoding="utf-8")
    )
    cifar_manifest = json.loads(
        (RESULTS / "cifar_probe_resnet18_manifest.json").read_text(encoding="utf-8")
    )
    meta_sweep_manifest = json.loads(
        (RESULTS / "vacs_meta_selector_sweep_manifest.json").read_text(encoding="utf-8")
    )
    section_text = "\n".join(read_text(path) for path in paper_section_paths)

    headline = manifest["headline"]
    vacs_acc = pct(headline["vacs_low_budget_accuracy"])
    best_static_acc = pct(headline["best_static_low_budget_accuracy"])
    gain_points = signed_points(headline["absolute_gain"], digits=2)
    best_static_method = headline["best_static_method"]
    best_static_label = METHOD_LABELS[best_static_method]
    best_static_prose = METHOD_PROSE[best_static_method]

    require_contains(section_text, f"VACS-F reaches {vacs_acc}\\% mean accuracy", "paper headline VACS-F accuracy")
    require_contains(section_text, f"{best_static_acc}\\% for the best pooled static selector", "paper headline static accuracy")
    require_contains(section_text, f"{gain_points} points", "paper headline gain")
    require_contains(
        section_text,
        "a numerical near-tie (+0.02 points; 95\\% bootstrap interval [-1.38,+1.20])",
        "paper near-tie headline wording",
    )

    stability = manifest["stability"]["pooled"]
    require(manifest["stability"]["comparator"] == best_static_method, "stability comparator matches headline best static method")
    require_contains(section_text, f"improves over {best_static_prose} by {stability['delta']:.2f} points", "paper paired stability comparator and delta")
    require_contains(section_text, f"{stability['delta']:.2f} points on average", "paper paired stability delta")
    require_contains(section_text, stability["ci"], "paper paired stability CI")
    require_contains(section_text, stability["wtl"], "paper paired W/T/L")

    dataset_best_rows = read_csv(RESULTS / "tables" / "vacs_dataset_best_static_paired_summary.csv")
    dataset_best = next(row for row in dataset_best_rows if row["slice"] == "Pooled per-dataset-best")
    require(
        (RESULTS / "tables" / "vacs_dataset_best_static_paired_table.tex").is_file(),
        "dataset-tuned static audit LaTeX table exists",
    )
    require(
        (RESULTS / "dataset_best_static_audit_manifest.json").is_file(),
        "dataset-tuned static audit manifest exists",
    )
    require_contains(
        section_text,
        (
            f"pooled VACS-F is {pct(float(dataset_best['vacs_accuracy']))}\\% versus "
            f"{pct(float(dataset_best['best_static_accuracy']))}\\% for the per-dataset best static selector"
        ),
        "paper dataset-tuned static pooled accuracies",
    )
    require_contains(
        section_text,
        f"a {signed_points(float(dataset_best['delta_accuracy']))} point paired difference",
        "paper dataset-tuned static delta",
    )
    require_contains(
        section_text,
        f"interval of {dataset_best['ci_points']}",
        "paper dataset-tuned static CI",
    )
    require_contains(
        section_text,
        f"{dataset_best['wtl']} win/tie/loss pattern",
        "paper dataset-tuned static W/T/L",
    )
    seed_block = manifest["seed_block_stability"]
    require(
        (RESULTS / "tables" / "vacs_seed_block_stability_table.tex").is_file(),
        "seed-block stability LaTeX table exists",
    )
    require_contains(
        section_text,
        f"five seed blocks remain positive and three are negative",
        "paper seed-block mixed-sign wording",
    )
    require_contains(
        section_text,
        f"seed-block mean is {signed_points(float(seed_block['delta']))} points",
        "paper seed-block mean",
    )
    require_contains(section_text, seed_block["ci"], "paper seed-block CI")
    require_contains(
        section_text,
        f"{signed_points(float(seed_block['min_delta']))} to {signed_points(float(seed_block['max_delta']))} points",
        "paper seed-block range",
    )
    downstream = {
        row["learner"]: row for row in read_csv(RESULTS / "tables" / "vacs_downstream_learner_summary.csv")
    }
    logreg_gain = signed_points(float(downstream["logreg"]["gain"]))
    knn_gain = signed_points(float(downstream["knn3"]["gain"]))
    require_contains(section_text, f"{knn_gain} points with the 3-NN classifier", "paper 3-NN gain")
    require_contains(section_text, f"{logreg_gain} points with logistic regression", "paper logreg gain")

    reliability_rows = read_csv(RESULTS / "tables" / "vacs_validation_reliability_summary.csv")
    pooled_reliability = next(row for row in reliability_rows if row["slice"] == "Pooled")
    require_contains(section_text, f"Pooled top-1 agreement is {pct(float(pooled_reliability['top1_agreement']))}\\%", "paper validation top-1")
    require_contains(section_text, f"mean rank correlation is {float(pooled_reliability['rank_correlation']):.2f}", "paper validation rank correlation")
    require_contains(section_text, f"mean regret is {points(float(pooled_reliability['regret']))} points", "paper validation regret")
    require_contains(
        section_text,
        f"Pooled top-1 agreement is {pct(float(pooled_reliability['top1_agreement']))}\\%",
        "paper validation agreement phrasing",
    )

    validation_size_rows = read_csv(RESULTS / "tables" / "vacs_validation_size_reliability_summary.csv")
    validation_size_by_frac = {row["validation_fraction"]: row for row in validation_size_rows}
    require(
        (RESULTS / "tables" / "vacs_validation_size_reliability_table.tex").is_file(),
        "validation-size reliability LaTeX table exists",
    )
    require(
        (RESULTS / "figures" / "vacs_protocol_robustness.pdf").is_file(),
        "protocol-robustness PDF figure exists",
    )
    validation_size_order = ["0.1", "0.2", "0.25", "0.33"]
    validation_size_gains = [signed_points(float(validation_size_by_frac[frac]["gain"])) for frac in validation_size_order]
    require_contains(
        section_text,
        (
            "In this diagnostic the gains are "
            f"{validation_size_gains[0]}, {validation_size_gains[1]}, {validation_size_gains[2]}, "
            f"and {validation_size_gains[3]} points, respectively."
        ),
        "paper validation-size gains",
    )
    require_contains(
        section_text,
        "The 20\\% split is nearly tied and also has the weakest top-1 agreement, rank correlation, regret, and severe-regret rate; the 25\\% default is the strongest among the tested fractions.",
        "paper validation-size interpretation",
    )
    require_contains(
        section_text,
        "Figure~\\ref{fig:protocol_robustness} summarizes the validation-fraction sensitivity and the repeated-validation uplift in one place.",
        "paper protocol-robustness bridge sentence",
    )
    require_contains(
        section_text,
        "Protocol robustness under validation choice and repetition",
        "paper protocol-robustness figure caption",
    )
    gap_rows = read_csv(RESULTS / "tables" / "vacs_validation_gap_summary.csv")
    pooled_gap = next(row for row in gap_rows if row["slice"] == "Pooled")
    low_gap_count = round(float(pooled_gap["n"]) * float(pooled_gap["low_gap_rate"]))
    require_contains(section_text, f"{low_gap_count} of {pooled_gap['n']} low-budget units", "paper validation-gap count")
    require_contains(section_text, f"median gap is only {points(float(pooled_gap['median_gap']))} points", "paper validation-gap median")

    split_rows = read_csv(RESULTS / "tables" / "vacs_validation_split_sensitivity_summary.csv")
    split_gains = english_list([signed_points(float(row["gain"])) for row in split_rows])
    split_best_static = split_rows[0]["best_static_method"]
    require_contains(
        section_text,
        f"gains over {METHOD_PROSE[split_best_static]} of {split_gains} points",
        "paper split-sensitivity gains",
    )

    repeated_rows = read_csv(RESULTS / "tables" / "vacs_repeated_validation_summary.csv")
    pooled_repeated = next(row for row in repeated_rows if row["slice"] == "Pooled")
    repeated_comparison_rows = read_csv(
        RESULTS / "tables" / "vacs_repeated_validation_comparison_summary.csv"
    )
    repeated_vs_best_static = next(
        row
        for row in repeated_comparison_rows
        if row["comparison"] == "best_static" and row["slice"] == "Pooled"
    )
    repeated_vs_dataset_best = next(
        row
        for row in repeated_comparison_rows
        if row["comparison"] == "dataset_best" and row["slice"] == "Pooled"
    )
    require(
        (RESULTS / "tables" / "vacs_repeated_validation_table.tex").is_file(),
        "repeated-validation LaTeX table exists",
    )
    require(
        (RESULTS / "tables" / "vacs_repeated_validation_comparison_table.tex").is_file(),
        "repeated-validation comparison LaTeX table exists",
    )
    require(
        (RESULTS / "repeated_validation_comparison_audit_manifest.json").is_file(),
        "repeated-validation comparison manifest exists",
    )
    require(
        (RESULTS / "figures" / "vacs_selector_mix.pdf").is_file(),
        "selector-mix PDF figure exists",
    )
    require_contains(
        section_text,
        f"VACS-R obtains {pct(float(pooled_repeated['repeated_accuracy']))}\\% pooled low-budget accuracy",
        "paper repeated-validation accuracy",
    )
    require_contains(
        section_text,
        "VACS-R is the recommended deployment protocol when reliability is more important than selector-choice time",
        "paper VACS-R deployment framing",
    )
    require_contains(
        section_text,
        (
            f"improves over {METHOD_PROSE[pooled_repeated['best_static_method']]} by "
            f"{repeated_vs_best_static['delta_points']} points with a 95\\% bootstrap interval "
            f"of {repeated_vs_best_static['ci_points']}"
        ),
        "paper repeated-validation gain",
    )
    require_contains(
        section_text,
        (
            f"it is essentially tied at {repeated_vs_dataset_best['delta_points']} points "
            f"with interval {repeated_vs_dataset_best['ci_points']}"
        ),
        "paper repeated-validation dataset-best comparison",
    )
    require_contains(
        section_text,
        f"agrees with VACS-F on {pct(float(pooled_repeated['selector_agreement']))}\\%",
        "paper repeated-validation selector agreement",
    )

    covtype_manifest = json.loads((RESULTS / "covtype_external_audit_manifest.json").read_text(encoding="utf-8"))
    covtype_confirmatory_manifest = json.loads(
        (RESULTS / "covtype_confirmatory_audit_manifest.json").read_text(encoding="utf-8")
    )
    covtype_paired_rows = read_csv(RESULTS / "tables" / "vacs_covtype_external_paired_summary.csv")
    covtype_vs_herding = next(
        row
        for row in covtype_paired_rows
        if row["target"] == "vacs_f" and row["comparator"] == "herding"
    )
    covtype_vs_marc = next(
        row
        for row in covtype_paired_rows
        if row["target"] == "vacs_f" and row["comparator"] == "marc"
    )
    covtype_confirmatory_paired_rows = read_csv(
        RESULTS / "tables" / "vacs_covtype_confirmatory_paired_summary.csv"
    )
    covtype_confirmatory_vs_herding = next(
        row
        for row in covtype_confirmatory_paired_rows
        if row["target"] == "vacs_r" and row["comparator"] == "herding"
    )
    covtype_confirmatory_vs_marc = next(
        row
        for row in covtype_confirmatory_paired_rows
        if row["target"] == "vacs_r" and row["comparator"] == "marc"
    )
    require(
        (RESULTS / "tables" / "vacs_covtype_external_table.tex").is_file(),
        "Covertype external audit LaTeX table exists",
    )
    require(
        (RESULTS / "tables" / "vacs_covtype_external_results.csv").is_file(),
        "Covertype external audit result CSV exists",
    )
    require(
        (RESULTS / "tables" / "vacs_covtype_confirmatory_table.tex").is_file(),
        "Covertype confirmatory audit LaTeX table exists",
    )
    require(
        (RESULTS / "tables" / "vacs_covtype_confirmatory_results.csv").is_file(),
        "Covertype confirmatory audit result CSV exists",
    )
    require(
        (RESULTS / "tables" / "vacs_covtype_confirmatory_summary.csv").is_file(),
        "Covertype confirmatory audit summary CSV exists",
    )
    require(
        (RESULTS / "tables" / "vacs_covtype_confirmatory_paired_summary.csv").is_file(),
        "Covertype confirmatory audit paired summary CSV exists",
    )
    require(covtype_manifest["dataset"] == "Covertype", "Covertype manifest dataset matches")
    require(covtype_manifest["sample"]["sampled_n"] == 32747, "Covertype sampled n matches")
    require(covtype_manifest["sample"]["original_n"] == 581012, "Covertype original n matches")
    require(covtype_manifest["sample"]["classes"] == 7, "Covertype class count matches")
    require(covtype_manifest["best_static_method"] == "marc", "Covertype best static method is MARC")
    require(covtype_confirmatory_manifest["dataset"] == "Covertype", "Covertype confirmatory manifest dataset matches")
    require(
        covtype_confirmatory_manifest["sample"]["sampled_n"] == 62240,
        "Covertype confirmatory sampled n matches",
    )
    require(
        covtype_confirmatory_manifest["sample"]["original_n"] == 581012,
        "Covertype confirmatory original n matches",
    )
    require(
        covtype_confirmatory_manifest["sample"]["classes"] == 7,
        "Covertype confirmatory class count matches",
    )
    require(
        covtype_confirmatory_manifest["best_static_method"] == "marc",
        "Covertype confirmatory best static method is MARC",
    )
    require(
        covtype_confirmatory_manifest["seeds"] == list(range(16)),
        "Covertype confirmatory seed list matches",
    )
    require(
        covtype_confirmatory_manifest["budgets_per_class"] == [1, 2],
        "Covertype confirmatory budgets match",
    )
    require_contains(
        section_text,
        f"Covertype stress audit on {covtype_manifest['sample']['sampled_n']:,} sampled examples",
        "paper Covertype sampled-size framing",
    )
    require_contains(
        section_text,
        f"from the original {covtype_manifest['sample']['original_n']:,} examples",
        "paper Covertype original-size framing",
    )
    require_contains(
        section_text,
        (
            f"VACS-F and VACS-R reach {pct(float(covtype_vs_herding['target_accuracy']))}\\%, "
            f"improve over herding by {covtype_vs_herding['delta_points']} points with interval "
            f"{covtype_vs_herding['ci_points']}"
        ),
        "paper Covertype intro result",
    )
    require_contains(
        section_text,
        (
            f"both reach {pct(float(covtype_vs_herding['target_accuracy']))}\\% mean accuracy "
            f"and {pct(0.435)}\\% macro-F1 over {covtype_vs_herding['n']} matched seed--budget units"
        ),
        "paper Covertype result accuracy and macro-F1",
    )
    require_contains(
        section_text,
        (
            f"the paired gain is {covtype_vs_herding['delta_points']} points with interval "
            f"{covtype_vs_herding['ci_points']} and an {covtype_vs_herding['wtl']} win/tie/loss pattern"
        ),
        "paper Covertype herding paired result",
    )
    require_contains(
        section_text,
        (
            f"exactly tied at {covtype_vs_marc['delta_points']} points with a "
            f"{covtype_vs_marc['wtl']} pattern"
        ),
        "paper Covertype MARC tie",
    )
    require_contains(
        section_text,
        (
            "A confirmatory larger-cap rerun on "
            f"{covtype_confirmatory_manifest['sample']['sampled_n']:,} examples and "
            f"{covtype_confirmatory_vs_herding['n']} matched seed--budget units repeats the pattern"
        ),
        "paper Covertype confirmatory framing",
    )
    require_contains(
        section_text,
        (
            f"VACS-R reaches {pct(float(covtype_confirmatory_vs_herding['target_accuracy']))}\\%, "
            f"improves over herding by {covtype_confirmatory_vs_herding['delta_points']} points "
            f"with interval {covtype_confirmatory_vs_herding['ci_points']}"
        ),
        "paper Covertype confirmatory herding result",
    )
    require_contains(
        section_text,
        (
            f"exactly ties MARC at {covtype_confirmatory_vs_marc['delta_points']} points "
            f"with a {covtype_confirmatory_vs_marc['wtl']} pattern"
        ),
        "paper Covertype confirmatory MARC tie",
    )
    require_contains(
        section_text,
        "Covertype exactly matches MARC",
        "paper Covertype limitation",
    )

    full20_rows = read_csv(RESULTS / "tables" / "vacs_full20news_stress_summary_row.csv")
    full20 = full20_rows[0]
    require(
        (RESULTS / "tables" / "vacs_full20news_stress_table.tex").is_file(),
        "full-20NG stress LaTeX table exists",
    )
    require_contains(section_text, "compact full-20-class TF-IDF stress summary", "paper full-20NG stress mention")
    require_contains(
        section_text,
        f"VACS reaches {pct(float(full20['vacs_accuracy']))}\\% on 20 Newsgroups",
        "paper full20NG VACS accuracy",
    )
    require_contains(
        section_text,
        f"static random reaches {pct(float(full20['best_static_accuracy']))}\\%",
        "paper full20NG random accuracy",
    )
    require_contains(
        section_text,
        f"gap is {signed_points(float(full20['gain']))} points",
        "paper full20NG gap",
    )
    require_contains(
        section_text,
        f"boundary selected in {int(full20['top_selector_count'])} of {int(full20['units'])} low-budget VACS decisions",
        "paper full20NG boundary count",
    )
    cost_rows = read_csv(RESULTS / "tables" / "vacs_selection_cost_summary.csv")
    pooled_cost = next(row for row in cost_rows if row["slice"] == "Pooled")
    reference_method = pooled_cost["reference_method"]
    require(reference_method == best_static_method, "selection-cost reference matches headline best static method")
    require_contains(section_text, f"{METHOD_PROSE[reference_method]} selection alone takes {float(pooled_cost['reference_selection_sec']):.2f} seconds", "paper reference selector cost")
    require_contains(section_text, f"VACS-F spends {float(pooled_cost['validation_phase_sec']):.2f} seconds in validation", "paper validation cost")
    require_contains(section_text, f"{float(pooled_cost['final_rebuild_sec']):.2f} seconds in the final rebuild", "paper rebuild cost")
    require_contains(section_text, f"{float(pooled_cost['vacs_selection_total_sec']):.2f} seconds total", "paper total VACS cost")

    transformer_headline = transformer_manifest["headline"]
    transformer_vacs = float(transformer_headline["vacs_accuracy"])
    transformer_best = float(transformer_headline["best_static_accuracy"])
    transformer_best_method = transformer_headline["best_static_method"]
    require(transformer_best_method == "marc", "Transformer probe best static method is MARC")
    require(abs(transformer_vacs - transformer_best) <= 0.001, "Transformer probe remains a near-tie")
    require_contains(
        section_text,
        f"{100.0 * transformer_vacs:.2f}\\% vs. {100.0 * transformer_best:.2f}\\%",
        "paper Transformer-probe boundary result",
    )

    qwen_manifest = json.loads(
        (RESULTS / "vacs_transformer_embedding_probe_qwen2_5_0_5b_local_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    qwen_rows = {
        row["method"]: row
        for row in read_csv(RESULTS / "tables" / "vacs_transformer_embedding_probe_qwen2_5_0_5b_local_summary.csv")
    }
    qwen_vacs = float(qwen_rows["vacs"]["accuracy_mean"])
    qwen_best = float(qwen_rows["herding"]["accuracy_mean"])
    require(qwen_manifest["model"] == "Qwen/Qwen2.5-0.5B-Instruct", "Qwen probe model name matches")
    require(qwen_manifest["headline"]["best_static_method"] == "herding", "Qwen probe best static method is herding")
    require(abs(qwen_vacs - float(qwen_manifest["headline"]["vacs_accuracy"])) <= 1e-12, "Qwen summary matches manifest VACS")
    require(abs(qwen_best - float(qwen_manifest["headline"]["best_static_accuracy"])) <= 1e-12, "Qwen summary matches manifest herding")
    require_contains(
        section_text,
        (
            "The local Qwen2.5-0.5B-Instruct run is another near-tie boundary check at "
            f"{100.0 * qwen_vacs:.2f}\\% versus {100.0 * qwen_best:.2f}\\% for herding"
        ),
        "paper Qwen boundary result",
    )

    require_contains(section_text, "Dataset-level low-budget heterogeneity", "paper dataset heterogeneity table")
    selector_mix_rows = read_csv(RESULTS / "tables" / "vacs_selector_mix_by_dataset.csv")
    selector_mix = {row["dataset"]: row for row in selector_mix_rows}
    ng_mix = selector_mix["20newsgroups"]
    pooled_mix = selector_mix["Pooled"]
    require(
        (RESULTS / "tables" / "vacs_selector_mix_by_dataset_table.tex").is_file(),
        "selector-mix LaTeX table exists",
    )
    random_count = int(ng_mix["random"])
    boundary_count = int(ng_mix["boundary"])
    units_count = int(ng_mix["units"])
    selector_mix_phrase = (
        f"boundary is the top VACS choice in {boundary_count} of {units_count} low-budget units "
        f"and random is chosen in the other {random_count}"
    )
    require_contains(section_text, selector_mix_phrase, "paper 20NG selector-mix count")
    pooled_phrase = (
        f"{int(pooled_mix['herding'])} herding, {int(pooled_mix['random'])} random, "
        f"{int(pooled_mix['kmeans'])} k-means, {int(pooled_mix['marc'])} MARC, "
        f"{int(pooled_mix['kcenter'])} k-center, and {int(pooled_mix['boundary'])} boundary"
    )
    require_contains(section_text, pooled_phrase, "paper pooled selector-mix counts")
    leave_one_rows = manifest["leave_one_dataset_out"]
    holdout_20ng = next(row for row in leave_one_rows if row["held_out"] == "20newsgroups")
    dense_holdout_gains = [
        float(row["gain"])
        for row in leave_one_rows
        if row["held_out"] in {"breast_cancer", "digits", "wine"}
    ]
    require_contains(
        section_text,
        (
            f"holding out 20 Newsgroups leaves VACS below {METHOD_PROSE[holdout_20ng['best_static_method']]} "
            f"({pct(float(holdout_20ng['vacs_accuracy']))}\\% vs. {pct(float(holdout_20ng['best_static_accuracy']))}\\%)"
        ),
        "paper leave-one-dataset-out 20NG",
    )
    require_contains(
        section_text,
        f"mixed deltas from {signed_points(min(dense_holdout_gains))} to {signed_points(max(dense_holdout_gains))} points",
        "paper leave-one-dataset-out dense range",
    )
    require_contains(section_text, "k-means medoids", "paper k-means medoid wording")
    require_contains(section_text, "full 20-class 20 Newsgroups", "paper full 20NG wording")
    require_contains(section_text, "frozen text/image embedding probes are boundary evidence rather than a positive deep-embedding claim", "paper abstract embedding-boundary wording")
    require_contains(section_text, "Supplemental frozen Transformer text probes and two frozen torchvision image probes are included as boundary evidence rather than as a positive deep-embedding claim", "paper introduction image-probe boundary wording")
    require_contains(section_text, "The remote frozen torchvision image-embedding probe uses an NVIDIA GeForce RTX 3090 and is boundary evidence only; the local CIFAR-10 frozen torchvision probe uses an NVIDIA GeForce RTX 4060 Laptop GPU and is boundary evidence only", "paper compute image-probe wording")
    require_contains(section_text, "boundary evidence rather than as a positive deep-embedding claim", "paper deep-embedding boundary wording")
    require_contains(section_text, "BADGE-inspired uncertainty-diversity", "paper BADGE scoped wording")
    require_contains(section_text, "Hindsight Static-Portfolio Regret", "paper hindsight-regret heading")

    require(torchvision_manifest["backbone_name"] == "convnext_tiny", "torchvision probe backbone matches")
    require(torchvision_manifest["device"] == "NVIDIA GeForce RTX 3090", "torchvision probe device matches")
    require(torchvision_manifest["datasets"][0]["dataset"] == "fashion_mnist", "torchvision probe dataset matches")
    require(torchvision_manifest["seeds"] == [0], "torchvision probe seed matches")
    require(torchvision_manifest["budgets_per_class"] == [1], "torchvision probe budget matches")
    require(torchvision_manifest["positive_dataset_count"] == 0, "torchvision probe has no positive dataset")
    require(abs(float(torchvision_manifest["pooled_gain"])) <= 1e-12, "torchvision probe gain is zero")
    require(
        (RESULTS / "tables" / "vacs_torchvision_embedding_probe_table.tex").is_file(),
        "torchvision probe LaTeX table exists",
    )
    torchvision_summary = read_csv(RESULTS / "tables" / "vacs_torchvision_embedding_probe_summary.csv")
    torchvision_pooled = next(row for row in torchvision_summary if row["dataset"] == "pooled")
    require(torchvision_pooled["best_static_method"] == "herding", "torchvision best static method is herding")
    require(abs(float(torchvision_pooled["vacs_accuracy"]) - float(torchvision_pooled["best_static_accuracy"])) <= 1e-12, "torchvision VACS ties best static")
    require_contains(
        section_text,
        (
            f"A frozen torchvision probe on Fashion-MNIST with ConvNeXt-Tiny features also exactly ties herding at "
            f"{100.0 * float(torchvision_pooled['vacs_accuracy']):.2f}\\% vs. "
            f"{100.0 * float(torchvision_pooled['best_static_accuracy']):.2f}\\%"
        ),
        "paper torchvision boundary result",
    )

    require(cifar_manifest["backbone_name"] == "resnet18", "cifar probe backbone matches")
    require(cifar_manifest["device"] == "NVIDIA GeForce RTX 4060 Laptop GPU", "cifar probe device matches")
    require(cifar_manifest["datasets"][0]["dataset"] == "cifar10", "cifar probe dataset matches")
    require(cifar_manifest["seeds"] == [0], "cifar probe seed matches")
    require(cifar_manifest["budgets_per_class"] == [1], "cifar probe budget matches")
    require(cifar_manifest["positive_dataset_count"] == 0, "cifar probe has no positive dataset")
    require(abs(float(cifar_manifest["pooled_gain"])) <= 1e-12, "cifar probe gain is zero")
    require(
        (RESULTS / "tables" / "cifar_probe_resnet18_table.tex").is_file(),
        "cifar probe LaTeX table exists",
    )
    cifar_summary = read_csv(RESULTS / "tables" / "cifar_probe_resnet18_summary.csv")
    cifar_pooled = next(row for row in cifar_summary if row["dataset"] == "pooled")
    require(cifar_pooled["best_static_method"] == "herding", "cifar best static method is herding")
    require(abs(float(cifar_pooled["vacs_accuracy"]) - float(cifar_pooled["best_static_accuracy"])) <= 1e-12, "cifar VACS ties best static")
    require_contains(
        section_text,
        (
            f"A local CIFAR-10 ResNet18 probe also exactly ties herding at "
            f"{pct(float(cifar_pooled['vacs_accuracy']), digits=2)}\\% vs. "
            f"{pct(float(cifar_pooled['best_static_accuracy']), digits=2)}\\%"
        ),
        "paper cifar boundary result",
    )

    require(
        (RESULTS / "tables" / "vacs_meta_selector_sweep_summary.csv").is_file(),
        "meta-selector sweep summary exists",
    )
    require(
        (RESULTS / "tables" / "vacs_meta_selector_leave_one_dataset_out.csv").is_file(),
        "meta-selector LODO summary exists",
    )
    top_meta = meta_sweep_manifest["top_exploratory"][0]
    require_contains(
        section_text,
        f"reaching {pct(float(top_meta['accuracy']), digits=2)}\\% pooled low-budget accuracy",
        "paper meta-selector exploratory top accuracy",
    )
    meta_lodo_rows = read_csv(RESULTS / "tables" / "vacs_meta_selector_leave_one_dataset_out.csv")
    repeat5_lodo = [
        float(row["heldout_accuracy"]) for row in meta_lodo_rows if row["source"] == "repeat5_0.25"
    ]
    repeat5_lodo_mean = sum(repeat5_lodo) / len(repeat5_lodo)
    require_contains(
        section_text,
        f"the five-repeat source averages {pct(repeat5_lodo_mean, digits=2)}\\% on the held-out datasets",
        "paper meta-selector LODO mean",
    )
    require_contains(
        section_text,
        "not as evidence that the current method dominates a tuned protocol family",
        "paper meta-selector non-claim wording",
    )

    require_absent(section_text, "Portfolio Oracle Regret", "old oracle heading")
    require_absent(section_text, "stronger external comparator", "old BADGE comparator wording")
    require_absent(section_text, "stronger active-learning comparator", "old BADGE active-learning wording")
    require_absent(section_text, "+3.42 points with the 3-NN classifier", "stale 3-NN gain")
    require_absent(section_text, "+3.37 points", "stale logreg gain")
    require_absent(section_text, "+3.56 points with logistic regression", "stale logreg gain")
    require_absent(section_text, "76.2--76.8\\%", "stale split-sensitivity range")
    require_absent(section_text, "76.5--76.8\\%", "stale split-sensitivity range")
    require_absent(section_text, "76.8\\% mean accuracy", "stale low-budget headline")
    require_absent(section_text, "73.4\\%", "stale static headline")
    require_absent(section_text, "four-class 20 Newsgroups", "stale 20NG subset wording")
    require_absent(section_text, "33.21\\% vs. 33.22\\%", "stale transformer probe")
    require_absent(section_text, "9.75\\%", "stale Transformer probe accuracy")
    require_absent(section_text, "9.94\\%", "stale Transformer probe accuracy")
    require_absent(section_text, "27.58 seconds total", "stale selection cost")
    require_absent(section_text, "7.68 seconds", "stale k-means cost")
    require_absent(section_text, "75.3 seconds", "stale method cost")
    require_absent(section_text, "64 low-budget decisions", "stale low-budget decision count")
    require_absent(section_text, "13.3\\% relative error reduction", "stale relative error")
    require_absent(section_text, "12.9\\% relative error reduction", "stale relative error")
    require_absent(section_text, "14.2\\% relative error reduction", "stale relative error")
    require_absent(section_text, "herding selection alone takes 1.62 seconds", "stale herding cost")
    require_absent(section_text, "k-means selection alone takes 5.00 seconds", "stale k-means cost")
    print("CLAIM_CONSISTENCY_OK")


if __name__ == "__main__":
    main()
