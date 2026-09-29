# Validation-Aligned Coreset Selection

This repository contains the reproducibility package for the manuscript
**"Validation-Aligned Coreset Selection for Budgeted Few-Shot Classification"**.

The code evaluates Validation-Aligned Coreset Selection (VACS), a training-only
finite-portfolio selector protocol for extremely small class-balanced budgets.
The main claim is intentionally bounded: repeated internal validation can make
selector choice competitive in the evaluated low-budget settings, but the
repository does not claim broad state-of-the-art performance.

## Repository Contents

- `experiments/`: experiment, audit, figure-generation, and consistency-check scripts.
- `analysis/`: saved-result revision audits, including the direct-MARC audit and the full-20-class 20 Newsgroups selector-transfer audit.
- `results/tables/`: generated CSV and LaTeX result tables used by the manuscript.
- `results/figures/`: generated figures and audit plots.
- `results/*_manifest.json`: run manifests and headline result metadata.
- `scientific_reports/source/`: main manuscript and supplementary information source/PDF.
- `scientific_reports/figures/` and `scientific_reports/tables/`: journal figure and table files.
- `scientific_reports/submission_materials/reference_verification_report.md`: reference verification trail.

## Quick Checks

Create an environment and install the core dependencies:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

Verify that the manuscript claims match the generated result artifacts:

```bash
python experiments/check_claim_consistency.py
```

Run the read-only revision audits from the repository root:

```bash
python analysis/vacs_revision_audit.py
python analysis/vacs_20ng_selector_audit.py \\
  --results-root results \\
  --experiment-code experiments/run_vacs_experiments.py
```

These audits consume saved results and do not run new classifier experiments.

Regenerate the workflow and protocol-robustness figures:

```bash
python experiments/make_vacs_workflow_figure.py
python experiments/make_vacs_protocol_robustness_figure.py
```

## Main Experiment

The main CPU experiment can be rerun with:

```bash
python experiments/run_vacs_experiments.py
```

The larger Covertype confirmatory audit can be rerun with:

```bash
python experiments/run_covtype_confirmatory_audit.py --max-per-class 10000
```

The scikit-learn datasets are public. The 20 Newsgroups dataset may be
downloaded by scikit-learn on first use. Covertype is available from the UCI
Machine Learning Repository and is cached locally by the audit script if needed.

## Optional GPU Probes

The frozen transformer and torchvision embedding probes are boundary checks,
not main claims. They require additional GPU-oriented dependencies:

```bash
pip install -r requirements-optional-gpu.txt
python experiments/run_transformer_embedding_probe.py
python experiments/run_torchvision_embedding_probe.py
```

## Code Availability

The public repository URL for review is:

https://github.com/HaotongLuan/validation-aligned-coreset-selection

Generated result tables, manifests, and figure files are included so reviewers
can inspect the evidence without rerunning every experiment.
