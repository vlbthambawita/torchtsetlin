# Literature database v2: tasks, TM configuration, evaluation protocol

Status (2026-11-05): §2 filter fix done; step 1 (vocabulary + validator) and step 2 (proposal CSVs) done,
**waiting for the user's review of `benchmarks/literature/mappings/*.csv`**. HF Space: skipped for now.

Decisions (user, 2026-11-05): clause counts stored **as reported** (`n_clauses` + `clauses_scope`, no
conversion); mechanical backfill first, then PDF re-read; filter-count fix now; Space later. Scope: `benchmarks/literature/` (records, `build_catalog.py`),
`docs/literature.md`, `docs/javascripts/literature.js`, `space/app.py`.

## 1. What the database has today (measured on the 55 records)

| Information | Where | State |
|---|---|---|
| ML task | `datasets[].task` (per paper x dataset), free text | 252 usages, **57 distinct strings**. They mix the task ("classification") with the setting ("federated", "robustness evaluation", "counterfactual recourse"). No paper-level task. |
| Number of classes | `datasets[].n_classes` | 193/252 filled. **Bug:** 11 datasets use different class counts in different papers (20 Newsgroups 4 vs 20, EMNIST 26 vs 62, Iris 2 vs 3, ...). The `datasets` table keeps whichever record is read first, and `dataset_usages` has no `n_classes` column. |
| Regression type, generation type | none | Regression (7 usages) and "forecasting / sequence generation" (4) are only labels. Target, range, horizon and output type are not recorded. |
| Data type | `datasets[].data_type`, free text | 25 strings; `text` (88 usages) sits next to `text (clinical EHR, ...)` (1). |
| Booleanization | `datasets[].booleanization`, free text | 60/252 null. Method, bits per feature and literal count are not separate fields. |
| Hyper-parameters (clauses, T, s, ...) | `results[].config`, free text | 644/3295 null. "clause" appears in 1360 rows and `T=` in 1008. They cannot be queried and it is unclear whether a clause count is per class or in total. |
| Metric | `results[].metric` | **204 distinct names**, with synonyms (`Spearman correlation` / `Spearman rho`, `macro-F1` / `macro F1`, `Kendall correlation` / `Kendall tau`). Synonyms are never ranked against each other. |
| Evaluation protocol | `split` + free text | Runs/seeds, mean vs peak vs final, CV folds and model selection are not recorded. The Pre-Sorted ⚠ case is exactly this gap. |

So: **yes, there is a task field, but it is not usable as a filter.** It also has no paper-level
view, no subtypes and no task-specific attributes.

## 2. Review of the numbers in the filters

The `(n)` after each dropdown option (`fill()` in `literature.js`) is unclear in five ways:

1. **The unit changes and is never shown.** `r-dtype`, `r-split`, `r-category` and `b-dtype` count
   *results*, `d-dtype` counts *datasets*, and `p-category` counts *papers*. So "foundations (16)" on
   the Papers tab and "foundations (1,1xx)" on the Results tab label the same category.
2. **The counts are static.** They are computed once at load time over the whole database. After
   picking dataset = MNIST, "tabular (700)" is still offered and returns 0 rows.
3. **The Best tab counts the wrong thing.** Its rows are dataset x metric groups, but `b-dtype`
   counts raw results.
4. **Free-text inputs and dropdowns differ.** Dataset and metric counts are computed but not shown
   (a `<datalist>` cannot show them), while the dropdowns show theirs.
5. **The vocabulary is fragmented** (section 1), so a count describes one spelling of a value.

Fix (one change in `literature.js`):
- **Faceted counts:** recount every facet against all the *other* active filters on each change
  (`SELECT col, COUNT(*), COUNT(DISTINCT paper_id) FROM base WHERE <filters except col> GROUP BY col`).
  `base` has 3.3k rows, so this costs under 5 ms.
- **State the unit** in the option text and a legend: `image — 31 papers · 702 results`.
  Show the number of papers first, because that is what the user means by "how much evidence".
- Grey out options with zero matches and keep them selectable, so a filter never vanishes.
- On the Best tab, count groups (rows of that tab).
- Show the active filters as removable chips above the table, with "N results from M papers".

## 3. Schema v2

Controlled vocabularies live in a single `benchmarks/literature/vocab.json` (JSON: the build stays stdlib-only), and a JSON Schema
check in `build_catalog.py` **rejects unknown values**. Free-text fields stay for nuance. Every new
field accepts `null` (not extracted yet) or `"not_reported"` (the paper does not say). These two must
differ, because "the paper omits the split" is itself a finding.

### 3.1 Task (per dataset usage, rolled up to the paper)

Four facets that do not overlap:

- `task_family` (enum): classification · regression · generation · clustering ·
  representation_learning · structure_learning · recommendation_ranking ·
  anomaly_novelty_detection · segmentation · reinforcement_learning_games · other.
- `modality` (enum): image · text · tabular · time_series · audio · graph · sequence · multimodal.
  Plus `synthetic: bool` (Noisy XOR, sampled from a Bayesian network, ...).
- `setting_tags` (multi, enum): federated · decentralized · robustness · uncertainty ·
  interpretability · counterfactual · verification · hardware · privacy · continual · transfer.
  This removes most of the 60 task strings.
- `domain` (enum): medical · nlp · vision · security · speech · games · finance · science · generic.

Attributes that depend on `task_family`, in a `task_detail` object:

| Family | Fields |
|---|---|
| classification | `n_classes`, `label_type` (binary/multiclass/multilabel/ordinal), `class_balance` (balanced / imbalanced + majority-class share), `discretized_from` (for example "regression target, 5 bins") |
| regression | `target` (what is predicted, unit), `target_range`, `output_dim` (uni/multi), `subtype` (point / ordinal / forecasting), `horizon` (forecasting), `tm_output_mechanism` (RTM vote sum, IRTM, ...) |
| generation | `output_modality` (text/image/sequence/symbolic), `conditioning` (unconditional / class / prompt / autoregressive context), `mechanism` (next-token classification, clause sampling, autoencoder decode), `output_length_or_size`, `eval_kind` (perplexity, BLEU, FID, human, task accuracy) |
| clustering / representation | `n_clusters`, `embedding_dim`, `downstream_eval` (intrinsic similarity, probe classifier) |
| structure_learning | `n_nodes`, `n_edges`, `graph_type` |
| segmentation | `n_classes`, `resolution`, `per_pixel_or_patch` |
| anomaly / novelty | `normal_classes`, `contamination` |

`n_classes` moves to the **usage** level, which fixes the 20 Newsgroups bug. The `datasets` table
then reports the range (`4–20`).

### 3.2 Input pipeline (per dataset usage)

- `preprocessing` (ordered list of enums plus params): resize, crop, grayscale, normalize,
  augmentation, stop-word removal, stemming, vocabulary cap (`k`), feature selection
  (`method`: chi2 / mutual information / ..., `k`), windowing (time series), imputation.
- `booleanization`: `method` (native_binary · fixed_threshold · adaptive_threshold · otsu ·
  thermometer · quantile_thermometer · one_hot · bag_of_words_presence · n_gram_presence ·
  embedding_augmented · gray_code · fingerprint · learned · other), `bits_per_feature`,
  `n_bins` / `threshold`, `n_boolean_features` (before negation), `n_literals`,
  `fit_on` (train / all data / not stated). `fit_on` is a **leakage** flag.
- `split`: `protocol` (official · random holdout · k-fold CV (k) · leave-one-out · temporal),
  sizes as today, `validation_used_for` (tuning / early stopping / none / not stated).

### 3.3 TM configuration (per result, structured; `config` stays as the verbatim source)

- `tm_variant` (multi, enum): vanilla · convolutional · weighted · coalesced · regression ·
  integer_weighted · multi_output / multi_label · autoencoder · multilayer/deep · contracting ·
  sparse · set_of_words · graph TM · REDRESS · other. It is mapped from `tm_variants_tested`.
- `n_clauses` plus **`clauses_scope` (per_class / total / per_output)**. Without the scope,
  clause counts cannot be compared.
- `T`, `s` (plus `s_schedule`), `n_states` or `state_bits`, `weighted: bool`,
  `max_included_literals`, `boost_true_positive_feedback`, `drop_clause_p`,
  `focused_negative_sampling`, `feedback_mode` (sequential / batched + batch size).
- Convolution: `patch_size`, `stride`, `position_encoding`, `n_patches`.
- Training: `epochs`, `early_stopping`, `implementation` (pyTsetlinMachine · tmu ·
  PyTsetlinMachineCUDA · torchtsetlin · FPGA/ASIC · custom), `hardware` (for timing rows).
- Derived and recorded: `model_size_literals` (included literals, when reported), `n_parameters`
  for non-TM baselines.

The first pass parses these from `config` with regexes, sets a `parsed: true` flag and keeps
`config`. A person or agent then confirms them against the PDF.

### 3.4 Result semantics (per result)

- `metric` maps to a canonical name in `vocab.yaml` (`metric_canonical` + `averaging`
  (macro/micro/weighted/binary) + `unit_scale` (0–1 vs %)). The original name is kept.
  Ranking uses the canonical name.
- `statistic`: mean · median · final_epoch · peak_over_epochs · best_of_n · single_run.
- `n_runs`, `std_or_range` (as today), `selection_on_test: bool`. The ⚠ flag becomes data,
  not prose.
- `kind`: quality · cost (time/energy/memory/area) · interpretability. Cost rows need `hardware`.

### 3.5 Paper level

`primary_task_family` and `task_families` (rolled up), `contribution_type` (already
`feature_type`), `reproducibility` (code available, seeds stated, hyper-parameters complete:
a score derived from the fields above), `interpretability_evidence` (rules shown? / clause
analysis / none).

## 4. Extraction plan

1. **Vocabulary and validator (1 day).** Write `vocab.json` and the JSON Schema. Make
   `build_catalog.py` fail on unknown enum values and print a coverage report per field.
2. **Mechanical backfill (1 day).** Build a reviewable `mappings/*.csv` from each existing free-text
   value to the enums (60 task strings, 25 data types, 204 metric names), plus a regex parse of
   `config`. **The user reviews these CSVs before anything is written into the records.**
3. **Re-read the PDFs (the bulk).** One agent per paper (55, in batches of about 8), each given the
   schema, the current record and the PDF. Each agent fills only new or `null` fields. Every value
   carries a `source_location` (table/section/page). An agent never overwrites a hand-checked value
   and instead lists the disagreement in `extraction_notes`.
4. **Independent second pass** on about 20% of the papers (stratified by category, plus every paper
   whose results are ranked best), done blind to the first pass. Measure agreement per field (§5)
   and resolve disagreements against the PDF.
5. **Rebuild** the DB, docs page and Space. Add facets for task family, modality, setting tags,
   class-count range, TM variant, Booleanization method, statistic and implementation.

## 5. How to evaluate the database

- **Validity:** 100% of records pass the schema (CI-style check in `build_catalog.py`).
- **Coverage:** fill rate per field, split into `filled` / `not_reported` / `null`. `null` should
  reach 0 for the core fields (task, n_classes, booleanization method, n_clauses + scope, T, s,
  statistic).
- **Accuracy:** inter-extractor agreement on the 20% double-extracted sample. Cohen's kappa ≥ 0.8
  on enums, ≥ 95% exact match on numbers. Plus a small **gold set** (about 5 papers, checked fully
  by a person against the PDFs) to measure the error rate per field.
- **Consistency checks** (automatic, in the build):
  - accuracy ≤ 100 and ≥ about 1/n_classes for TM rows (otherwise the row is suspicious);
  - `n_clauses` per class × n_classes ≈ total, when both are stated;
  - train + val + test ≈ dataset size;
  - metrics that a family does not use (accuracy on a regression usage) are flagged;
  - the same model and dataset should not get two different values from one source location.
- **Competency questions.** The database is good if it answers these in one SQL query each, and
  they become tests:
  - Best TM accuracy per dataset with ≤ X clauses per class.
  - Every regression usage with its target and the TM vs baseline MAE.
  - Which Booleanization gives the best CIFAR-10 TM result?
  - Which results are peak-over-epochs or selected on test?
  - Which papers report enough hyper-parameters to reproduce?
  - How do clause counts scale with n_classes?
- **Optional reproduction anchor** (later, and only with your go-ahead for runs): re-run 3–5
  catalogued configs in `torchtsetlin` and compare them with the reported values.

## 6. Decisions for the user

1. ~~Normalise clause counts?~~ No: as reported, with `clauses_scope`.
2. ~~Backfill first?~~ Yes.
3. ~~Filter fix now?~~ Done.
4. ~~HF Space?~~ Skipped for now.
