# VACS Revision Audit

Run the audit from the repository root:

```bash
python analysis/vacs_revision_audit.py
```

The default paths are relative to the repository root: `results/`,
`experiments/`, and `analysis/`. To write a disposable staging copy, use:

```bash
python analysis/vacs_revision_audit.py --output-dir analysis/staging
```

## Scope

This is a read-only audit of saved result tables and manifests. It runs no new
experiments and does not download data. The main low-budget slice contains 80
matched units: five datasets, eight seeds, and budgets 1 and 2 per class,
evaluated with the `knn3` learner.

The direct MARC comparisons report both VACS-F and VACS-R for pooled units,
each dataset, and the leave-out-20NG slice. The leave-out-20NG result is a
comparison limit, not an additional experiment: it excludes the 20 Newsgroups
units from the same matched 80-unit table. The audit also traces the saved
20NG stress subset back to the corresponding main rows.

## Reproduction Details

- Bootstrap confidence intervals use seed `20260531` and `20000` resamples.
- Win/tie/loss counts use an absolute-difference tolerance of `1e-12`.
- Table 1 accuracy and macro-F1 variability uses sample standard deviation
  with `ddof=1`.
- Existing displayed confidence intervals are checked at their reported
  two-decimal percentage precision; full-precision bounds are retained in the
  direct summary output.

The generated CSV and JSON files contain portable result labels and hashes,
not machine-specific input paths.
