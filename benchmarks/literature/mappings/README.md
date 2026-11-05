# Schema-v2 proposals (review these)

`propose_mappings.py` parses the free-text fields of `papers/*.json` into the controlled values of
`vocab.json` and writes these CSVs. `build_catalog.py` reads them as a proposal layer: a value in a
paper record always wins, and every value in the database carries `v2_source`
(`record` / `reviewed` / `proposed`). Plan: `plans/literature-db-v2.md`.

| File | One row per | Feeds |
|---|---|---|
| `tasks.csv` | task string (57) | `task_family`, `setting_tags` of every usage with that string |
| `data_types.csv` | data-type string (25) | `modality`, `synthetic` |
| `metrics.csv` | metric name (204) | `metric_canonical`, `metric_qualifier`, `averaging`, `metric_kind`, `statistic` |
| `usages.csv` | paper x dataset usage (252) | `label_type`, `booleanization_method`, `booleanization_bits` |
| `configs.csv` | result (3,295) | `n_clauses`, `clauses_scope`, `T`, `s`, `epochs`, `n_states`, `state_bits`, `n_runs`, `statistic`, ... |

**How to review.** Correct a `proposed_*` cell in place if it is wrong, then set `review`:

- blank: still a proposal (`v2_source = proposed`);
- `ok`: accepted (`reviewed`);
- `reject`: drop the whole row's proposals (the fields stay empty).

Then run `python benchmarks/literature/build_catalog.py`. It rejects values that are not in
`vocab.json`.

**Do not re-run `propose_mappings.py` after reviewing has started.** It overwrites the CSVs,
`review` column included.
