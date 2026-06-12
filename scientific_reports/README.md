# Scientific Reports Submission Package

This directory contains the Scientific Reports submission version of:

**Validation-Aligned Coreset Selection for Budgeted Few-Shot Classification**

The manuscript is organized for Scientific Reports: non-anonymous front matter,
Results before Methods, explicit data/code availability statements,
competing-interest and author-contribution sections, and bounded empirical
claims rather than broad state-of-the-art claims.

The current clean archive is `Scientific_Reports_VACS_submission_clean.zip` in
the project root.

## Core Files

- `source/manuscript_sr.tex`: main manuscript source.
- `source/manuscript_sr.pdf`: compiled main manuscript.
- `source/supplementary_information.tex`: supplementary information source.
- `source/supplementary_information.pdf`: compiled supplementary information.
- References are embedded in `source/manuscript_sr.tex` so the manuscript does
  not require a separate BibTeX upload file.
- `figures/Figure_*.pdf`: separate figure files for upload.
- `tables/*.tex`: table source used by the manuscript and supplementary file.
- `submission_materials/cover_letter.md`: cover letter draft.
- `submission_materials/author_info_template.md`: title page and author metadata template.
- `submission_materials/declarations_and_statements.md`: required declaration text.
- `submission_materials/data_code_availability.md`: standalone availability statements.
- `submission_materials/reference_verification_report.md`: three-source verification trail for all 31 manuscript references.
- `submission_materials/submission_checklist.md`: pre-submission checklist.
- `submission_materials/suggested_reviewers_template.csv`: optional reviewer suggestion template.

## Required Human Edits Before Upload

Scientific Reports is not anonymous. Before submission, complete the author
names, affiliations, corresponding-author email, author contributions, funding,
and acknowledgements. Human authors must also verify all AI-assisted drafting,
code, data, numbers, and claims before upload.

## Reproduction

The experiment code and generated result artifacts remain in the project-level
`experiments` and `results` directories. The clean Scientific Reports archive
includes those reproducibility files together with the manuscript package.

To rebuild the main manuscript from `scientific_reports/source`:

```powershell
pdflatex -interaction=nonstopmode manuscript_sr.tex
pdflatex -interaction=nonstopmode manuscript_sr.tex
```

To rebuild the supplementary information:

```powershell
pdflatex -interaction=nonstopmode supplementary_information.tex
pdflatex -interaction=nonstopmode supplementary_information.tex
```

To run the claim consistency check from the project root:

```powershell
python experiments\check_claim_consistency.py
```
