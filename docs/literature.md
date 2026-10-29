---
hide:
  - navigation
  - toc
---

# Literature results

Every dataset and every reported number from the 55 Tsetlin-machine papers in
`source_documents/papers/`: the proposed models *and* the baselines each paper compared against.
The numbers are **as reported by the authors, not reproduced here**. The page loads the catalog
as a SQLite database (about 1.5 MB, once). All filtering and ranking then run as SQL in your
browser, and only one page of rows is drawn at a time.

<div id="lit-app" markdown="0">
  <nav class="lit-tabs" role="tablist">
    <button type="button" role="tab" data-tab="results" aria-selected="true">Results</button>
    <button type="button" role="tab" data-tab="best">Best per dataset</button>
    <button type="button" role="tab" data-tab="datasets">Datasets</button>
    <button type="button" role="tab" data-tab="papers">Papers</button>
    <button type="button" role="tab" data-tab="sql">SQL</button>
  </nav>
  <p id="lit-status" class="lit-status">Loading the database…</p>

  <section class="lit-panel" data-panel="results">
    <aside class="lit-sidebar" aria-label="Filters">
      <div class="lit-fields">
        <label>Rank best within
          <select id="r-group">
            <option value="dataset" selected>Dataset</option>
            <option value="paper">Paper</option>
            <option value="year">Year</option>
            <option value="category">Paper category</option>
            <option value="data_type">Data type</option>
            <option value="model">Model</option>
            <option value="">Nothing (whole selection)</option>
          </select>
        </label>
        <label>Dataset <input id="r-dataset" list="lit-datasets" placeholder="any"></label>
        <label>Metric <input id="r-metric" list="lit-metrics" placeholder="any"></label>
        <label>Data type <select id="r-dtype"><option value="">any</option></select></label>
        <label>Split <select id="r-split"><option value="">any</option></select></label>
        <label>Paper category <select id="r-category"><option value="">any</option></select></label>
        <label>Search <input id="r-search" type="search" placeholder="model, paper, config, notes…"></label>
      </div>
      <div class="lit-toggles">
        <label><input id="r-tm" type="checkbox"> TM models only</label>
        <label><input id="r-proposed" type="checkbox"> Proposed models only</label>
        <label><input id="r-flag" type="checkbox"> Hide ⚠ flagged papers</label>
        <label><input id="r-best" type="checkbox"> ★ Best per group only</label>
        <span class="lit-buttons">
          <button type="button" id="r-reset">Reset</button>
          <button type="button" id="r-csv">Download CSV</button>
        </span>
      </div>
    </aside>
    <div class="lit-main">
      <p id="r-summary" class="lit-status"></p>
      <div id="r-detail" class="lit-detail" hidden></div>
      <div id="r-table" class="lit-scroll"></div>
      <div id="r-pager" class="lit-pager"></div>
    </div>
  </section>

  <section class="lit-panel" data-panel="best" hidden>
    <aside class="lit-sidebar" aria-label="Filters">
      <p class="lit-hint">One row per dataset × metric × unit × split: the best reported value, and
      the best value from a Tsetlin-machine model.</p>
      <div class="lit-fields">
        <label>Dataset <input id="b-dataset" list="lit-datasets" placeholder="any"></label>
        <label>Data type <select id="b-dtype"><option value="">any</option></select></label>
        <label>Metric <input id="b-metric" list="lit-metrics" placeholder="any"></label>
        <label>Min. results per row <input id="b-min" type="number" min="1" value="2"></label>
      </div>
      <div class="lit-toggles">
        <label><input id="b-flag" type="checkbox" checked> Hide ⚠ flagged papers</label>
        <label><input id="b-tm" type="checkbox"> Rank TM models only</label>
        <span class="lit-buttons"><button type="button" id="b-csv">Download CSV</button></span>
      </div>
    </aside>
    <div class="lit-main">
      <p id="b-summary" class="lit-status"></p>
      <div id="b-table" class="lit-scroll"></div>
      <div id="b-pager" class="lit-pager"></div>
    </div>
  </section>

  <section class="lit-panel" data-panel="datasets" hidden>
    <aside class="lit-sidebar" aria-label="Filters">
      <div class="lit-fields">
        <label>Search <input id="d-search" type="search" placeholder="name, task, TM variant…"></label>
        <label>Data type <select id="d-dtype"><option value="">any</option></select></label>
      </div>
    </aside>
    <div class="lit-main">
      <p class="lit-hint">Click a dataset for how each paper used it (sizes, splits, Booleanization).</p>
      <div id="d-detail" class="lit-detail" hidden></div>
      <div id="d-table" class="lit-scroll"></div>
      <div id="d-pager" class="lit-pager"></div>
    </div>
  </section>

  <section class="lit-panel" data-panel="papers" hidden>
    <aside class="lit-sidebar" aria-label="Filters">
      <div class="lit-fields">
        <label>Search <input id="p-search" type="search" placeholder="title, author, TM variant, feature…"></label>
        <label>Category <select id="p-category"><option value="">any</option></select></label>
      </div>
    </aside>
    <div class="lit-main">
      <div id="p-detail" class="lit-detail" hidden></div>
      <div id="p-table" class="lit-scroll"></div>
      <div id="p-pager" class="lit-pager"></div>
    </div>
  </section>

  <section data-panel="sql" hidden>
    <p class="lit-hint">Any SQLite query against your own in-browser copy of the database (schema
    below). Ctrl/⌘ + Enter runs it.</p>
    <textarea id="s-sql" rows="8" spellcheck="false">-- Best reported test accuracy per TM paper on MNIST
SELECT p.year, p.title, r.model, MAX(r.value) AS accuracy
FROM results r JOIN papers p USING (paper_id)
WHERE r.dataset_id = 'mnist' AND r.metric = 'accuracy' AND r.is_tm = 1
  AND COALESCE(NULLIF(r.split, ''), 'test') = 'test'
GROUP BY r.paper_id
ORDER BY accuracy DESC;</textarea>
    <div class="lit-toggles"><span class="lit-buttons">
      <button type="button" id="s-run">Run</button>
      <button type="button" id="s-csv">Download CSV</button>
    </span></div>
    <p id="s-summary" class="lit-status"></p>
    <div id="s-table" class="lit-scroll"></div>
    <div id="s-pager" class="lit-pager"></div>
  </section>

  <datalist id="lit-datasets"></datalist>
  <datalist id="lit-metrics"></datalist>
</div>

## How the ranking works

<span class="lit-best">★ Highlighted</span> values are the best in their group. Rows are compared
only when they share the grouping you chose **and** the same metric, unit and split, and a group
needs at least two such rows. A train-set accuracy is never ranked against a test-set one, and nJ
never against µJ. Rows whose paper states no split (mostly hardware measurements) are ranked with
test rows. "Best" means highest for accuracy, F1, AUC, IoU, correlations, speed-ups, throughput and
efficiency, and lowest for errors (MAE, RMSE, ECE), time, latency, energy, power, area, memory and
model size; the ↑ / ↓ next to each value says which. A few count-type metrics, such as the number
of clusters, are never ranked.

!!! warning "Before comparing across papers"
    Papers use different splits, Booleanizations, clause budgets and epoch counts, and some report
    the *peak* test accuracy over training rather than a final one. Click a row to see the config,
    source table and notes behind a number. Papers marked **⚠** have a specific reliability
    problem; for example, the Pre-Sorted TM's top MNIST and Kuzushiji-MNIST accuracies are likely
    optimistic. Results from other implementations, or measured before `torchtsetlin` 0.3.0 made
    the clause budget a hard cap, are not directly comparable to `torchtsetlin` runs.

## Database schema

[Download `literature.db`](assets/literature/literature.db) to query it with any SQLite client.

| Table | Columns |
|---|---|
| `papers` | `paper_id`, `title`, `authors`, `year`, `venue`, `arxiv_id`, `doi`, `url`, `code_url`, `category`, `feature_type`, `tm_variants`, `new_feature`, `caution`, `extraction_notes`, `pdf` |
| `datasets` | `dataset_id`, `data_type`, `tasks`, `n_classes`, `input_shape_or_features`, `download_url`, `download_url_source`, `n_papers`, `tm_variants`, `names_in_papers` |
| `dataset_usages` | `paper_id`, `dataset_id`, `name_in_paper`, `task`, `train_size`, `val_size`, `test_size`, `split_as_reported`, `booleanization`, `notes` |
| `results` | `id`, `paper_id`, `dataset_id`, `model`, `is_tm`, `is_proposed`, `metric`, `value`, `unit`, `std_or_range`, `split`, `better` (`higher`/`lower`), `config`, `source_location`, `notes` |

The data is generated: `benchmarks/literature/papers/*.json` holds one record per paper (the source
of truth), and `python benchmarks/literature/build_catalog.py` rebuilds the database. Edit the
records, not the database. The same data is also served by a
[Hugging Face Space](https://huggingface.co/spaces/vlbthambawita/TsetlinMachines).

<script src="https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.12.0/sql-wasm.js"></script>
<script src="../javascripts/literature.js"></script>
