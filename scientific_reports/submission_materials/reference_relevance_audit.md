# Reference Relevance Audit

Date: 2026-06-13

Scope: `scientific_reports/source/manuscript_sr.tex`

Purpose: confirm that every cited reference supports a manuscript claim, method choice,
evaluation protocol, or dataset/software provenance.

Verdict: all 35 manuscript references are contextually relevant. No citation is used as
decorative padding.

## Relevance Map

| Key | Used in | Why it is relevant |
|---|---|---|
| `lewis1994sequential` | Introduction: active-learning contrast | Foundational sequential text-classifier sampling. |
| `freund1997query` | Introduction: active-learning contrast | Classic query-by-committee background. |
| `settles2009active` | Introduction: active-learning contrast | Standard active-learning survey reference. |
| `sener2018active` | Introduction: core-set contrast | Core-set active learning baseline context. |
| `ash2020deep` | Introduction; Methods comparator | BADGE-style uncertainty/diversity comparator. |
| `feldman2020coresets` | Introduction: coreset framing | General coreset survey and terminology. |
| `wei2015submodularity` | Introduction: subset-selection framing | Submodular subset selection background. |
| `bachem2017practical` | Introduction: coreset framing | Practical coreset construction context. |
| `mirzasoleiman2020craig` | Introduction: coreset framing | Data-efficient training via coresets. |
| `killamsetty2021glister` | Introduction: modern subset selection | Generalization-based subset selection. |
| `killamsetty2021gradmatch` | Introduction: modern subset selection | Gradient-matching data subset selection. |
| `guo2022deepcore` | Introduction: recent deep coreset work | Recent benchmark/library context. |
| `xia2023moderate` | Introduction: recent deep coreset work | Scenario-robust data selection context. |
| `xia2024refined` | Introduction: recent deep coreset work | Minimal coreset-size objective under constraints. |
| `chen2025squaredloss` | Introduction: recent deep coreset work | Recent loss-based coreset objective. |
| `cover1967nearest` | Introduction: nearest-neighbor framing | Foundation for instance-based prediction. |
| `hart1968condensed` | Introduction: nearest-neighbor framing | Classic condensed NN storage reduction. |
| `wilson1972edited` | Introduction: nearest-neighbor framing | Edited NN for reducing stored examples. |
| `brighton2002advances` | Introduction: instance-selection framing | Instance-selection review background. |
| `olvera2010review` | Introduction: instance-selection framing | Instance-selection survey background. |
| `garcia2012prototype` | Introduction: prototype-selection framing | Prototype-selection taxonomy and study. |
| `vinyals2016matching` | Introduction: few-shot context | Matching-network few-shot baseline. |
| `finn2017maml` | Introduction: few-shot context | Meta-learning baseline for few-shot tasks. |
| `snell2017prototypical` | Introduction: few-shot context | Prototypical-network baseline. |
| `sung2018relation` | Introduction: few-shot context | Relation-network baseline. |
| `stone1974cross` | Introduction: validation principles | Cross-validation foundation. |
| `kohavi1995study` | Introduction: validation principles | Cross-validation and bootstrap model selection. |
| `dietterich1998statistical` | Introduction: validation principles | Statistical comparison of learning algorithms. |
| `nadeau2003inference` | Introduction: validation principles | Corrected inference for generalization error. |
| `dror2018hitchhiker` | Introduction: validation principles | Significance-testing guidance for ML/NLP. |
| `rice1976algorithm` | Introduction: algorithm-selection framing | Algorithm-selection foundation. |
| `kotthoff2014algorithm` | Introduction: algorithm-selection framing | Survey-level algorithm-selection context. |
| `thornton2013autoweka` | Introduction: algorithm-selection framing | Algorithm selection plus hyperparameter optimization. |
| `blackard1998covertype` | Results; Methods | Provenance for the Covertype audit dataset. |
| `pedregosa2011scikit` | Methods | Provenance for the main benchmark and software stack. |

## Notes

- The recent 2022-2025 references are included because the manuscript explicitly
  situates VACS against current coreset selection work.
- The older references are kept because they supply the foundational context for
  active learning, nearest-neighbor reduction, validation, and algorithm selection.
- The bibliography is therefore mixed by design: historical foundations plus a small
  set of recent coreset papers that directly support the related-work paragraph.
