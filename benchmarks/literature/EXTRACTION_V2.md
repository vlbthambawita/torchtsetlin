# Schema-v2 extraction from the PDFs (step 3 of plans/literature-db-v2.md)

You fill the **structured** fields for one paper by reading its PDF. You write one file,
`benchmarks/literature/papers_v2/<paper_id>.json`, and nothing else. Never edit `papers/*.json`,
the CSVs or the database. If the existing record is wrong, report it under `corrections`.

## Inputs
- The record: `benchmarks/literature/papers/<paper_id>.json`. It has `datasets[]` (index = `usage_index`)
  and `results[]` (index = `result_index`), plus the PDF path in `pdf` (relative to the repo root).
- The PDF. The Read tool cannot render PDFs on this machine (no poppler), so extract the text with
  pymupdf: `python -c "import pymupdf,sys; d=pymupdf.open(sys.argv[1]); [print(f'=== page {i+1}'), print(p.get_text()) for i,p in enumerate(d)]" <pdf> > <scratchpad>/<paper_id>.txt`
  and read that file. Read the experimental sections, tables, figure captions and appendix fully, not
  just the abstract. Figures are not visible as text: if a value seems to live only in a figure or in
  formatting (italics, bold), say so in `notes` instead of guessing.
- The allowed values: `benchmarks/literature/vocab.json`. Use exactly those strings.
- The current automatic proposals, for orientation only: `benchmarks/literature/mappings/*.csv`.
  They are often incomplete or wrong. The PDF decides.

## Value rules
- `"not_reported"`: the paper does not say. Use `null` only for "not applicable".
  Never guess. "Presumably", "typical" or "the library default" are not reported values.
- Numbers are numbers (`2000`, not `"2,000"`). Keep values **as reported**: a clause count is the number
  the paper prints, and `clauses_scope` says what it counts (`per_class`, `total` or `per_output`).
  If the scope is unclear, use `"not_reported"`. Do not convert.
- Every non-trivial value gets an `evidence` entry: field name -> location (`"Table 2"`,
  `"Sec. 4.1, p. 6"`, `"Fig. 3 caption"`).
- Hyper-parameters that differ per dataset or per experiment go in separate `configs` groups.
- `selection_on_test` is `true` when the reported number was chosen by looking at test data:
  a peak/best epoch on the test set (also when such peaks are then averaged over runs), or a
  hyper-parameter (clauses, T, s, p, ...) picked as the best of several test-set results with no
  validation set. `false` only when the paper says selection used a validation set / CV inside the
  training data, or the settings were fixed in advance. Otherwise `"not_reported"`.
- `n_states` is the number of states **per action** (N; the automaton has 2N states). If the paper
  prints the total (2N), store N and list it in `inferred` ("paper prints 2N = 300").
- `split.protocol`: `random_holdout` only if the paper says the split is random; a bare
  "90%/10%" is `"not_reported"` (a split ratio does not say how it was drawn).
- Each correction has a `kind`: `contradicted` (the PDF states something different), `omitted` (the
  record leaves out something that changes how a value reads), or `unsupported`
  (the record states something the PDF does not say, e.g. a well-known class count). Use
  `unsupported` sparingly: a standard fact about a public dataset (IMDb is binary) is not worth one.
- **Inferred values must be labelled.** If you fill a value that the paper does not state for that row,
  but that follows from it (e.g. "Table 3 at 8000 clauses has exactly the Table 4 accuracies, so it
  used the Table 1 settings"), keep it and list it in that entry's `inferred` object: field name ->
  the reasoning, e.g. `"inferred": {"tm_config": "accuracies equal Table 4 means"}`. Anything weaker
  than a clear inference is `"not_reported"`.

## File format
```json
{
  "paper_id": "<exactly the record's paper_id>",
  "extracted": {"by": "agent", "date": "YYYY-MM-DD", "pages_read": "1-14"},
  "paper": {
    "primary_task_family": "<vocab task_family>",
    "code_available": true,
    "seeds_stated": false,
    "hyperparameters_complete": "yes | partial | no",
    "interpretability_evidence": "<vocab interpretability_evidence>",
    "evidence": {"code_available": "p. 1 footnote"}
  },
  "usages": [
    {
      "usage_index": 0,
      "dataset_id": "<copy from record>",
      "task_family": "<vocab task_family>",
      "setting_tags": ["<vocab setting_tags>"],
      "label_type": "<vocab label_type, classification-like only, else null>",
      "n_classes": 4,
      "modality": "<vocab modality>",
      "synthetic": false,
      "domain": "<vocab domain>",
      "task_detail": {},
      "preprocessing": [{"step": "<vocab preprocessing_step>", "params": "free text, e.g. 'chi2, k=8006'"}],
      "booleanization": {
        "method": "<vocab booleanization_method>",
        "bits_per_feature": 8, "n_bins": 8, "threshold": "free text or number",
        "n_boolean_features": 784, "n_literals": 1568,
        "fit_on": "<vocab fit_on or not_reported>",
        "description": "one sentence, as the paper describes it"
      },
      "split": {"protocol": "<vocab split_protocol>", "k_folds": null,
                "validation_used_for": "<vocab validation_used_for>"},
      "evidence": {"booleanization": "Sec. 5.1"}
    }
  ],
  "configs": [
    {
      "result_indices": [0, 1, 2],
      "statistic": "<vocab statistic>",
      "n_runs": 5,
      "selection_on_test": false,
      "tm_config": {
        "n_clauses": 2000, "clauses_scope": "per_class", "T": 50, "s": 10.0, "epochs": 100,
        "n_states": "not_reported", "state_bits": 8, "max_included_literals": null,
        "patch_size": "10x10", "weighted": false,
        "implementation": "<vocab implementation>", "hardware": "free text, for timing rows"
      },
      "evidence": {"tm_config": "Table 1", "statistic": "Sec. 5, 'mean of 5 runs'"},
      "inferred": {}
    }
  ],
  "corrections": [
    {"where": "results[12].value", "kind": "contradicted", "record": 98.1, "pdf": 98.2, "location": "Table 3", "note": ""}
  ],
  "notes": "anything a later reader needs: ambiguities, what was not found"
}
```

`task_detail` keys by `task_family`:

| task_family | keys |
|---|---|
| classification | `n_classes`, `label_type`, `class_balance` (`"balanced"` or e.g. `"imbalanced, majority 70%"`), `discretized_from` |
| regression | `target` (what is predicted, with unit), `target_range`, `output_dim`, `subtype` (vocab `regression_subtype`), `horizon` |
| generation | `output_modality` (vocab `generation_output`), `conditioning` (vocab `generation_conditioning`), `mechanism`, `output_length_or_size`, `eval_kind` |
| clustering / representation_learning | `n_clusters`, `embedding_dim`, `downstream_eval` |
| structure_learning | `n_nodes`, `n_edges`, `graph_type` |
| segmentation | `n_classes`, `resolution`, `per_pixel_or_patch` |
| anomaly_novelty_detection | `normal_classes`, `contamination` |

## Coverage
- One `usages` entry per record dataset (every `usage_index`).
- Every `result_index` in exactly one `configs` group. Baselines (non-TM) get `tm_config: null` but
  still a `statistic`/`n_runs` if the paper says.
- A result whose config is unknown goes into a group with `"tm_config": "not_reported"`.

## Before you finish
Run `python benchmarks/literature/build_catalog.py --check-v2 <paper_id>` and fix every error it
prints. Then reply with: pages read, number of corrections, and the three least certain values.
