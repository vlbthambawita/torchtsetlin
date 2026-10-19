---
hide:
  - navigation
  - toc
---

# Literature results

Every dataset and every reported number from the 55 Tsetlin-machine papers in
`source_documents/papers/`: the proposed models *and* the baselines each paper compared against.
The numbers are **as reported by the authors, not reproduced here**.

!!! tip "Faster explorer with SQL: [Tsetlin Machine literature explorer on Hugging Face](https://huggingface.co/spaces/vlbthambawita/TsetlinMachines)"
    The same data as a server-side SQLite database. Filtering and ranking run as SQL, and only the
    shown rows are sent to the browser. It also has a best-per-dataset leaderboard (best overall
    value and best TM value), dataset and paper tables, and a read-only SQL console.

<div class="lit-controls" markdown="0">
  <div class="lit-fields">
  <label>Group by
    <select id="lit-group">
      <option value="dataset,metric" selected>Dataset → metric</option>
      <option value="dataset">Dataset</option>
      <option value="metric">Metric</option>
      <option value="paper">Paper</option>
      <option value="category">Paper category</option>
      <option value="year">Year</option>
      <option value="">No grouping</option>
    </select>
  </label>
  <label>Search <input id="lit-search" type="search" placeholder="model, paper, config, notes…"></label>
  <label>Dataset <input id="lit-dataset" list="lit-dataset-list" placeholder="any"></label>
  <label>Metric <input id="lit-metric" list="lit-metric-list" placeholder="any"></label>
  <label>Data type <select id="lit-dtype"><option value="">any</option></select></label>
  <label>Split <select id="lit-split"><option value="">any</option></select></label>
  <label>Category <select id="lit-category"><option value="">any</option></select></label>
  </div>
  <div class="lit-toggles">
  <label class="lit-check"><input id="lit-tm" type="checkbox"> TM models only</label>
  <label class="lit-check"><input id="lit-proposed" type="checkbox"> Proposed models only</label>
  <label class="lit-check"><input id="lit-caution" type="checkbox"> Hide ⚠ flagged papers</label>
  <label class="lit-check"><input id="lit-best" type="checkbox"> Best per group only</label>
  <span class="lit-buttons">
    <button id="lit-collapse" type="button">Collapse groups</button>
    <button id="lit-reset" type="button">Reset</button>
    <button id="lit-csv" type="button">Download CSV</button>
  </span>
  </div>
  <datalist id="lit-dataset-list"></datalist>
  <datalist id="lit-metric-list"></datalist>
</div>

<p id="lit-status" class="lit-status">Loading…</p>
<div id="lit-results"></div>

**How the highlight works.** <span class="lit-best-cell">★ Highlighted</span> values are the best in
their group: rows are compared only when they share the grouping you chose **and** the same metric,
unit and split, and a group needs at least two such rows. A train-set accuracy is never ranked
against a test-set one, and nJ never against µJ; rows whose paper states no split (mostly hardware
measurements) are compared with test rows. The ↑ / ↓ next to a value says which direction is better.
"Best" means highest for accuracy, F1, AUC, IoU, correlations, speed-ups, throughput and efficiency,
and lowest for errors (MAE, RMSE, ECE), time, latency, energy, power, area, memory and model size.
Only a few count-type metrics, such as the number of clusters, are never highlighted. With no
grouping, every metric/unit/split combination gets one winner across the whole table.

!!! warning "Before comparing across papers"
    Papers use different splits, Booleanizations, clause budgets and epoch counts, and some report
    the *peak* test accuracy over training rather than a final one. The `notes` and `config` of each
    row (click a row to expand them) say how the number was obtained. Results from other
    implementations, or measured before `torchtsetlin` 0.3.0 made the clause budget a hard cap, are
    not directly comparable to `torchtsetlin` runs. Papers marked **⚠** have a specific reliability
    problem (hover the mark, or click the row); for example, the Pre-Sorted TM's top MNIST and
    Kuzushiji-MNIST accuracies are likely optimistic. Use *Hide ⚠ flagged papers* to drop them.

## Datasets

One row per dataset, with the train / val / test sizes each paper reported and a download link.
Click a dataset to filter the results table above to it.

<div id="lit-datasets"></div>

The data is generated: `benchmarks/literature/papers/*.json` holds one record per paper (the source
of truth), and `python benchmarks/literature/build_catalog.py` rebuilds
`docs/assets/literature/*.json`, which this page reads. Edit the records, not the generated files.

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabulator/6.3.1/css/tabulator_simple.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/tabulator/6.3.1/js/tabulator.min.js"></script>
<script src="../javascripts/literature.js"></script>
