/* docs/literature.md: the literature catalog, queried in the browser with sql.js.
 *
 * docs/assets/literature/literature.db is written by benchmarks/literature/build_catalog.py.
 * Every filter, grouping and "best value" ranking is one SQL query; only the current page of
 * rows is turned into HTML, which keeps each interaction to a few milliseconds.
 */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  if (!$("lit-app") || typeof initSqlJs === "undefined") return;

  const SQLJS = "https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.12.0/";
  const DB_URL = "../assets/literature/literature.db";
  const PAGE_SIZES = [50, 100, 250, 1000];
  // Only these fixed strings are ever spliced into SQL; user input always goes through bind().
  const GROUPS = new Set(["dataset", "paper", "year", "category", "data_type", "model", ""]);
  const SPLIT_KEY = "COALESCE(NULLIF(r.split, ''), 'test')";
  const RANK = "CASE better WHEN 'lower' THEN value ELSE -value END";
  const RESULTS_ORDER = `group_size DESC, grp, metric_size DESC, metric, unit, split_key, ${RANK}, year`;

  let db, sqlDb;

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v) => (typeof v === "number"
    ? v.toLocaleString("en-US", { maximumFractionDigits: 6 }) : esc(v));
  const link = (url, text) => (url
    ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(text)}</a>` : "");

  function all(sql, params, target) {
    const st = (target || db).prepare(sql);
    try {
      if (params) st.bind(params);
      const out = [];
      while (st.step()) out.push(st.getAsObject());
      return out;
    } finally {
      st.free();
    }
  }
  const one = (sql, params) => all(sql, params)[0];

  function debounce(fn, ms) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  }

  function csvDownload(rows, name) {
    if (!rows.length) return;
    const cols = Object.keys(rows[0]);
    const cell = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const text = [cols.join(",")].concat(rows.map((r) => cols.map((c) => cell(r[c])).join(","))).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // "Exact pick from the datalist" -> equality; anything else -> substring (LIKE).
  function textCond(col, value, conds, params, exact) {
    const v = value.trim();
    if (!v) return;
    if (exact.has(v.toLowerCase())) {
      conds.push(`lower(${col}) = ?`);
      params.push(v.toLowerCase());
    } else {
      conds.push(`${col} LIKE ?`);
      params.push(`%${v}%`);
    }
  }

  // --------------------------------------------------------- paged table view
  /* A table whose rows come from a SQL source: `count()` and `page(limit, offset, order)`.
   * columns: [{key, title, render(row) -> html, sort: sql expression | undefined, cls}] */
  function pagedTable(prefix, columns, opts) {
    const state = { page: 0, size: opts.size || 100, sort: null, dir: "DESC", source: null };
    const tableEl = $(`${prefix}-table`), pagerEl = $(`${prefix}-pager`);

    function draw() {
      const src = state.source;
      if (!src) return;
      const total = src.count();
      const pages = Math.max(1, Math.ceil(total / state.size));
      state.page = Math.min(state.page, pages - 1);
      const order = state.sort ? `${state.sort} ${state.dir}` : null;
      const rows = src.page(state.size, state.page * state.size, order);
      let lastGroup;
      const head = columns.map((c) => {
        const mark = state.sort && state.sort === c.sort ? (state.dir === "ASC" ? " ▲" : " ▼") : "";
        return `<th${c.sort ? ` data-sort="${esc(c.sort)}" class="lit-sortable"` : ""}>${esc(c.title)}${mark}</th>`;
      }).join("");
      const body = rows.map((r, i) => {
        let pre = "";
        if (opts.groupOf && !state.sort) {
          const g = opts.groupOf(r);
          if (g !== undefined && g !== lastGroup) {
            pre = `<tr class="lit-group"><td colspan="${columns.length}">${opts.groupLabel(r)}</td></tr>`;
            lastGroup = g;
          }
        }
        const cls = opts.rowClass ? opts.rowClass(r) : "";
        return `${pre}<tr data-i="${i}" class="${cls}">${columns.map((c) =>
          `<td class="${c.cls || ""}">${c.render ? c.render(r) : esc(r[c.key])}</td>`).join("")}</tr>`;
      }).join("");
      tableEl.innerHTML = `<table class="lit-table"><thead><tr>${head}</tr></thead><tbody>${body
        || `<tr><td colspan="${columns.length}" class="lit-empty">No rows match.</td></tr>`}</tbody></table>`;
      tableEl._rows = rows;
      pagerEl.innerHTML = total <= PAGE_SIZES[0] ? "" :
        `<button type="button" data-go="prev" ${state.page ? "" : "disabled"}>‹ Prev</button>`
        + `<span>Page ${state.page + 1} of ${pages} · rows ${(state.page * state.size + 1).toLocaleString()}–`
        + `${Math.min(total, (state.page + 1) * state.size).toLocaleString()} of ${total.toLocaleString()}</span>`
        + `<button type="button" data-go="next" ${state.page < pages - 1 ? "" : "disabled"}>Next ›</button>`
        + `<label>Rows per page <select data-size>${PAGE_SIZES.map((s) =>
          `<option${s === state.size ? " selected" : ""}>${s}</option>`).join("")}</select></label>`;
    }

    tableEl.addEventListener("click", (e) => {
      const th = e.target.closest("th[data-sort]");
      if (th) {
        const key = th.dataset.sort;
        if (state.sort === key) {
          if (state.dir === "DESC") state.dir = "ASC"; else state.sort = null;  // 3rd click: default order
        } else {
          state.sort = key;
          state.dir = "DESC";
        }
        state.page = 0;
        draw();
        return;
      }
      const tr = e.target.closest("tr[data-i]");
      if (tr && opts.onRow && !e.target.closest("a")) opts.onRow(tableEl._rows[+tr.dataset.i]);
    });
    pagerEl.addEventListener("click", (e) => {
      const go = e.target.dataset && e.target.dataset.go;
      if (!go) return;
      state.page += go === "next" ? 1 : -1;
      draw();
      tableEl.scrollIntoView({ block: "nearest" });
    });
    pagerEl.addEventListener("change", (e) => {
      if (e.target.matches("[data-size]")) { state.size = +e.target.value; state.page = 0; draw(); }
    });

    return {
      set(source, newColumns) {
        if (newColumns) columns = newColumns;
        state.source = source;
        state.page = 0;
        state.sort = null;
        draw();
      },
      redraw: draw,
    };
  }

  // A narrow ranked TEMP table, paged in SQL, then joined to `base` for the drawn rows only.
  function joinedSource(table, defaultOrder, extra) {
    return {
      count: () => one(`SELECT COUNT(*) AS n FROM ${table}`).n,
      page: (limit, offset, order) => all(`
        WITH pg AS (SELECT *, ROW_NUMBER() OVER (ORDER BY ${order || defaultOrder}) AS rn
                    FROM ${table} ORDER BY ${order || defaultOrder} LIMIT ? OFFSET ?)
        SELECT ${extra}, b.* FROM pg c JOIN base b ON b.id = c.id ORDER BY c.rn`, [limit, offset]),
    };
  }

  // A source over a TEMP table that a filter step has just (re)built.
  function tempSource(table, defaultOrder) {
    return {
      count: () => one(`SELECT COUNT(*) AS n FROM ${table}`).n,
      page: (limit, offset, order) =>
        all(`SELECT * FROM ${table} ORDER BY ${order || defaultOrder} LIMIT ? OFFSET ?`, [limit, offset]),
    };
  }

  function detail(prefix, html) {
    const box = $(`${prefix}-detail`);
    box.innerHTML = `<button type="button" class="lit-close" aria-label="Close">×</button>${html}`;
    box.hidden = false;
    box.querySelector(".lit-close").onclick = () => { box.hidden = true; };
    box.scrollIntoView({ block: "nearest" });
  }
  const dl = (pairs) => `<dl>${pairs.filter(([, v]) => v != null && v !== "").map(([k, v]) =>
    `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;

  // ------------------------------------------------------------------ results
  const EXACT = { dataset: new Set(), metric: new Set() };

  const arrow = (b) => (b === "higher" ? '<span class="lit-dir" title="higher is better">↑</span>'
    : b === "lower" ? '<span class="lit-dir" title="lower is better">↓</span>' : "");
  const unit = (u) => (u === "%" ? "%" : u ? ` ${esc(u)}` : "");
  const valueCell = (r) => `${r.is_best ? "★ " : ""}${num(r.value)}${unit(r.unit)}`
    + `${r.std_or_range ? ` <span class="lit-muted">${esc(r.std_or_range)}</span>` : ""}${arrow(r.better)}`;
  const modelCell = (r) => esc(r.model)
    + (r.is_proposed ? ' <span class="lit-badge lit-proposed">proposed</span>' : "")
    + (r.is_tm ? "" : ' <span class="lit-badge">non-TM</span>');
  const paperCell = (r) => (r.caution ? `<span class="lit-caution" title="${esc(r.caution)}">⚠</span> ` : "")
    + (r.paper_url ? link(r.paper_url, r.paper) : esc(r.paper));

  const resultsTable = pagedTable("r", [
    { key: "dataset", title: "Dataset", sort: "dataset" },
    { key: "metric", title: "Metric", sort: "metric" },
    { key: "value", title: "Value", render: valueCell, sort: "value", cls: "lit-num" },
    { key: "split", title: "Split", sort: "split" },
    { key: "model", title: "Model", render: modelCell, sort: "model" },
    { key: "paper", title: "Paper", render: paperCell, sort: "paper" },
    { key: "year", title: "Year", sort: "year", cls: "lit-num" },
    { key: "source_location", title: "Source" },
  ], {
    size: 100,
    rowClass: (r) => (r.is_best ? "lit-best-row" : ""),
    groupOf: (r) => (r.grp === "" ? undefined : `${r.grp}\u0001${r.metric}\u0001${r.unit}\u0001${r.split_key}`),
    groupLabel: (r) => (r.grp === "" ? "" : `<b>${esc(r.grp)}</b> · ${esc(r.metric)}${r.unit ? ` (${esc(r.unit)})` : ""}`
      + ` · ${esc(r.split_key)} <span class="lit-muted">${r.n_cmp} result${r.n_cmp === 1 ? "" : "s"}</span>`),
    onRow: (r) => detail("r", `<strong>${esc(r.model)}</strong> on <code>${esc(r.dataset)}</code>: `
      + `${esc(r.metric)} = ${num(r.value)}${unit(r.unit)}` + dl([
      ["Paper", `${paperCell(r)} (${esc(r.year)})`], ["⚠ Caution", esc(r.caution)],
      ["TM variants in paper", esc(r.tm_variants)], ["Config", esc(r.config)],
      ["Source", esc(r.source_location)], ["Notes", esc(r.notes)],
      ["Extraction notes (paper)", esc(one("SELECT extraction_notes AS x FROM papers WHERE paper_id = ?", [r.paper_id]).x)],
      ["Record", `<code>benchmarks/literature/papers/${esc(r.paper_id)}.json</code>`]])),
  });

  function runResults() {
    const t0 = performance.now();
    const group = $("r-group").value;
    const g = GROUPS.has(group) && group ? group : "''";
    const conds = [], params = [];
    textCond("dataset", $("r-dataset").value, conds, params, EXACT.dataset);
    textCond("metric", $("r-metric").value, conds, params, EXACT.metric);
    for (const [col, id] of [["data_type", "r-dtype"], ["split_key", "r-split"], ["category", "r-category"]]) {
      if ($(id).value) { conds.push(`${col} = ?`); params.push($(id).value); }
    }
    if ($("r-tm").checked) conds.push("is_tm = 1");
    if ($("r-proposed").checked) conds.push("is_proposed = 1");
    if ($("r-flag").checked) conds.push("caution IS NULL");
    const q = $("r-search").value.trim();
    if (q) { conds.push("search LIKE ?"); params.push(`%${q}%`); }
    const where = conds.length ? `WHERE ${conds.join(" AND ")}` : "";
    // Rank over narrow rows only (ids, keys, values); the long text columns are joined in
    // for the one page that is drawn.
    db.run("DROP TABLE IF EXISTS temp.cur");
    db.run(`CREATE TEMP TABLE cur AS
      WITH f AS (SELECT id, ${g} AS grp, dataset, metric, unit, split_key, split, value, better,
                        model, paper, year FROM base ${where}),
      k AS (
        SELECT f.*, COUNT(value) OVER w AS n_cmp,
               CASE better WHEN 'higher' THEN MAX(value) OVER w WHEN 'lower' THEN MIN(value) OVER w END AS best_v,
               COUNT(*) OVER (PARTITION BY grp) AS group_size,
               COUNT(*) OVER (PARTITION BY grp, metric) AS metric_size
        FROM f WINDOW w AS (PARTITION BY grp, metric, unit, split_key)
      )
      SELECT id, grp, dataset, metric, unit, split_key, split, value, better, model, paper, year,
             n_cmp, group_size, metric_size,
             (better IS NOT NULL AND n_cmp > 1 AND value = best_v) AS is_best
      FROM k`, params);
    const stats = one("SELECT COUNT(*) AS n, COALESCE(SUM(is_best), 0) AS b FROM cur");
    if ($("r-best").checked) db.run("DELETE FROM cur WHERE NOT is_best");
    resultsTable.set(joinedSource("cur", RESULTS_ORDER, "c.is_best, c.grp, c.n_cmp"));
    const shown = $("r-best").checked ? stats.b : stats.n;
    $("r-summary").textContent = `${shown.toLocaleString()} rows · ${stats.b.toLocaleString()} best values (★)`
      + ` · ${(performance.now() - t0).toFixed(0)} ms · click a row for its config and notes`
      + " · click a column title to sort";
  }

  // --------------------------------------------------------- best per dataset
  const bestTable = pagedTable("b", [
    { key: "dataset", title: "Dataset", sort: "dataset" },
    { key: "data_type", title: "Type", sort: "data_type" },
    { key: "metric", title: "Metric", sort: "metric" },
    { key: "split_key", title: "Split" },
    { key: "n", title: "Results", sort: "n", cls: "lit-num" },
    { key: "value", title: "Best value", cls: "lit-num lit-best-cell",
      render: (r) => `${num(r.value)}${unit(r.unit)}${arrow(r.better)}` },
    { key: "model", title: "Best model", render: (r) => esc(r.model) + (r.is_tm ? "" : ' <span class="lit-badge">non-TM</span>') },
    { key: "paper", title: "Paper", render: (r) => (r.paper_url ? link(r.paper_url, r.paper) : esc(r.paper)) + ` (${esc(r.year)})` },
    { key: "tm_value", title: "Best TM value", cls: "lit-num",
      render: (r) => (r.tm_value == null ? "–" : `${num(r.tm_value)}${unit(r.unit)}`) },
    { key: "tm_model", title: "Best TM model", render: (r) => esc(r.tm_model || "") },
    { key: "tm_paper", title: "TM paper", render: (r) => (r.tm_paper ? `${esc(r.tm_paper)} (${esc(r.tm_year)})` : "") },
  ], {
    size: 100,
    onRow: (r) => {
      resetResults(false);
      $("r-flag").checked = $("b-flag").checked;
      $("r-dataset").value = r.dataset;
      $("r-metric").value = r.metric;
      showTab("results");
      runResults();
    },
  });

  function runBest() {
    const conds = ["better IS NOT NULL", "value IS NOT NULL"], params = [];
    textCond("dataset", $("b-dataset").value, conds, params, EXACT.dataset);
    textCond("metric", $("b-metric").value, conds, params, EXACT.metric);
    if ($("b-dtype").value) { conds.push("data_type = ?"); params.push($("b-dtype").value); }
    if ($("b-flag").checked) conds.push("caution IS NULL");
    const tmOnly = $("b-tm").checked;
    if (tmOnly) conds.push("is_tm = 1");
    const key = "dataset, metric, unit, split_key";
    db.run("DROP TABLE IF EXISTS temp.best");
    db.run(`CREATE TEMP TABLE best AS
      WITH f AS (SELECT id, dataset, data_type, metric, unit, split_key, better, value, year, is_tm
                 FROM base WHERE ${conds.join(" AND ")}),
      r AS (SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ${key} ORDER BY ${RANK}, year) AS rk,
                   ROW_NUMBER() OVER (PARTITION BY ${key}, is_tm ORDER BY ${RANK}, year) AS rk_cls,
                   COUNT(*) OVER (PARTITION BY ${key}) AS n FROM f),
      a AS (SELECT * FROM r WHERE rk = 1 AND n >= ?),
      t AS (SELECT * FROM r WHERE is_tm = 1 AND rk_cls = 1)
      SELECT a.dataset, a.data_type, a.metric, a.unit, a.split_key, a.better, a.n, a.value,
             ba.model, a.is_tm, ba.paper, a.year, ba.paper_url,
             t.value AS tm_value, bt.model AS tm_model, bt.paper AS tm_paper, t.year AS tm_year
      FROM a JOIN base ba ON ba.id = a.id
      LEFT JOIN t ON t.dataset = a.dataset AND t.metric = a.metric AND t.unit = a.unit
                 AND t.split_key = a.split_key
      LEFT JOIN base bt ON bt.id = t.id`, params.concat([Math.max(1, +$("b-min").value || 1)]));
    const n = one("SELECT COUNT(*) AS n FROM best").n;
    bestTable.set(tempSource("best", "n DESC, dataset, metric"));
    $("b-summary").textContent = `${n.toLocaleString()} dataset × metric × unit × split rows`
      + " · click a row to open its results";
  }

  // ----------------------------------------------------------------- datasets
  const datasetsTable = pagedTable("d", [
    { key: "dataset_id", title: "Dataset", sort: "dataset_id" },
    { key: "data_type", title: "Type", sort: "data_type" },
    { key: "tasks", title: "Tasks" },
    { key: "n_classes", title: "Classes", sort: "n_classes", cls: "lit-num" },
    { key: "input_shape_or_features", title: "Input" },
    { key: "sizes", title: "Train / val / test (as reported)" },
    { key: "n_papers", title: "Papers", sort: "n_papers", cls: "lit-num" },
    { key: "n_results", title: "Results", sort: "n_results", cls: "lit-num" },
    { key: "download_url", title: "Download",
      render: (r) => (r.download_url ? link(r.download_url, "download") : '<span class="lit-muted">not public / n.a.</span>') },
  ], {
    size: 50,
    onRow: (r) => {
      const uses = all(`SELECT p.title, p.year, p.url, u.name_in_paper, u.task, u.train_size, u.val_size,
                               u.test_size, u.split_as_reported, u.booleanization, u.notes
                        FROM dataset_usages u JOIN papers p ON p.paper_id = u.paper_id
                        WHERE u.dataset_id = ? ORDER BY p.year`, [r.dataset_id]);
      detail("d", `<strong><code>${esc(r.dataset_id)}</code></strong> ${link(r.download_url, "download")}`
        + ` · <button type="button" class="lit-linkbtn" id="d-show">show its ${r.n_results} results</button>`
        + `<table class="lit-table lit-inner"><thead><tr><th>Paper</th><th>Name in paper</th><th>Task</th>`
        + `<th>Train / val / test</th><th>Split as reported</th><th>Booleanization</th><th>Notes</th></tr></thead><tbody>`
        + uses.map((u) => `<tr><td>${u.url ? link(u.url, u.title) : esc(u.title)} (${esc(u.year)})</td>`
          + `<td>${esc(u.name_in_paper)}</td><td>${esc(u.task)}</td>`
          + `<td>${[u.train_size, u.val_size, u.test_size].map((v) => (v == null ? "–" : num(v))).join(" / ")}</td>`
          + `<td>${esc(u.split_as_reported)}</td><td>${esc(u.booleanization)}</td><td>${esc(u.notes)}</td></tr>`).join("")
        + "</tbody></table>");
      $("d-show").onclick = () => {
        resetResults(false);
        $("r-dataset").value = r.dataset_id;
        showTab("results");
        runResults();
      };
    },
  });

  function runDatasets() {
    const conds = [], params = [];
    const q = $("d-search").value.trim();
    if (q) {
      conds.push("(d.dataset_id || ' ' || COALESCE(d.tasks, '') || ' ' || COALESCE(d.names_in_papers, '') || ' ' || COALESCE(d.tm_variants, '')) LIKE ?");
      params.push(`%${q}%`);
    }
    if ($("d-dtype").value) { conds.push("d.data_type = ?"); params.push($("d-dtype").value); }
    db.run("DROP TABLE IF EXISTS temp.ds");
    db.run(`CREATE TEMP TABLE ds AS SELECT d.*,
        (SELECT COUNT(*) FROM results r WHERE r.dataset_id = d.dataset_id) AS n_results,
        (SELECT GROUP_CONCAT(s, ' | ') FROM (SELECT DISTINCT COALESCE(u.train_size, '–') || ' / '
            || COALESCE(u.val_size, '–') || ' / ' || COALESCE(u.test_size, '–') AS s
          FROM dataset_usages u WHERE u.dataset_id = d.dataset_id
            AND COALESCE(u.train_size, u.val_size, u.test_size) IS NOT NULL)) AS sizes
      FROM datasets d ${conds.length ? `WHERE ${conds.join(" AND ")}` : ""}`, params);
    datasetsTable.set(tempSource("ds", "n_papers DESC, n_results DESC, dataset_id"));
  }

  // ------------------------------------------------------------------- papers
  const papersTable = pagedTable("p", [
    { key: "year", title: "Year", sort: "year", cls: "lit-num" },
    { key: "title", title: "Title", sort: "title",
      render: (r) => (r.caution ? `<span class="lit-caution" title="${esc(r.caution)}">⚠</span> ` : "")
        + (r.url ? link(r.url, r.title) : esc(r.title)) },
    { key: "category", title: "Category", sort: "category" },
    { key: "tm_variants", title: "TM variants" },
    { key: "new_feature", title: "New feature" },
    { key: "n_results", title: "Results", sort: "n_results", cls: "lit-num" },
    { key: "code_url", title: "Code", render: (r) => link(r.code_url, "code") },
  ], {
    size: 100,
    onRow: (r) => {
      detail("p", `<strong>${r.url ? link(r.url, r.title) : esc(r.title)}</strong> (${esc(r.year)})`
        + ` · <button type="button" class="lit-linkbtn" id="p-show">show its ${r.n_results} results</button>`
        + dl([["Authors", esc(r.authors)], ["Venue", esc(r.venue)], ["Feature type", esc(r.feature_type)],
          ["New feature", esc(r.new_feature)], ["TM variants", esc(r.tm_variants)],
          ["Datasets", esc(r.datasets)], ["⚠ Caution", esc(r.caution)],
          ["Extraction notes", esc(r.extraction_notes)], ["Code", link(r.code_url, r.code_url)],
          ["Record", `<code>benchmarks/literature/papers/${esc(r.paper_id)}.json</code>`]]));
      $("p-show").onclick = () => {
        resetResults(false);
        $("r-search").value = r.title;
        showTab("results");
        runResults();
      };
    },
  });

  function runPapers() {
    const conds = [], params = [];
    const q = $("p-search").value.trim();
    if (q) {
      conds.push("(title || ' ' || COALESCE(authors, '') || ' ' || COALESCE(tm_variants, '') || ' ' || COALESCE(new_feature, '') || ' ' || COALESCE(venue, '')) LIKE ?");
      params.push(`%${q}%`);
    }
    if ($("p-category").value) { conds.push("category = ?"); params.push($("p-category").value); }
    db.run("DROP TABLE IF EXISTS temp.pp");
    db.run(`CREATE TEMP TABLE pp AS SELECT p.*,
        (SELECT COUNT(*) FROM results r WHERE r.paper_id = p.paper_id) AS n_results,
        (SELECT GROUP_CONCAT(dataset_id, ', ') FROM dataset_usages u WHERE u.paper_id = p.paper_id) AS datasets
      FROM papers p ${conds.length ? `WHERE ${conds.join(" AND ")}` : ""}`, params);
    papersTable.set(tempSource("pp", "year, title"));
  }

  // ---------------------------------------------------------------------- SQL
  let sqlRows = [];
  const sqlTable = pagedTable("s", [], { size: 100 });
  function runSql() {
    const t0 = performance.now();
    try {
      const res = sqlDb.exec($("s-sql").value);
      const last = res[res.length - 1];
      sqlRows = last ? last.values.map((v) => Object.fromEntries(last.columns.map((c, i) => [c, v[i]]))) : [];
      const cols = last ? last.columns : [];
      sqlTable.set({
        count: () => sqlRows.length,
        page: (limit, offset) => sqlRows.slice(offset, offset + limit),
      }, cols.map((c) => ({ key: c, title: c,
        render: (r) => (typeof r[c] === "number" ? num(r[c]) : esc(r[c])) })));
      $("s-summary").textContent = `${sqlRows.length.toLocaleString()} rows · ${(performance.now() - t0).toFixed(0)} ms`;
    } catch (err) {
      $("s-summary").textContent = `Error: ${err.message}`;
      $("s-table").innerHTML = "";
      $("s-pager").innerHTML = "";
    }
  }

  // --------------------------------------------------------------------- tabs
  const ran = new Set();
  const RUN = { results: runResults, best: runBest, datasets: runDatasets, papers: runPapers, sql: runSql };
  function showTab(name) {
    document.querySelectorAll("#lit-app [role=tab]").forEach((b) =>
      b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll("#lit-app [data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== name; });
    if (!ran.has(name)) { ran.add(name); RUN[name](); }
  }

  function resetResults(run = true) {
    for (const id of ["r-dataset", "r-metric", "r-dtype", "r-split", "r-category", "r-search"]) $(id).value = "";
    for (const id of ["r-tm", "r-proposed", "r-flag", "r-best"]) $(id).checked = false;
    $("r-group").value = "dataset";
    if (run) runResults();
  }

  function fill(id, rows) {
    const el = $(id);
    for (const { v, n } of rows) {
      if (v == null || v === "") continue;
      const o = document.createElement("option");
      o.value = v;
      if (el.tagName === "SELECT") o.textContent = `${v} (${n})`;
      el.appendChild(o);
    }
  }

  function wire() {
    const later = debounce(runResults, 150);
    for (const id of ["r-group", "r-dtype", "r-split", "r-category", "r-tm", "r-proposed", "r-flag", "r-best"]) {
      $(id).addEventListener("change", runResults);
    }
    for (const id of ["r-dataset", "r-metric", "r-search"]) $(id).addEventListener("input", later);
    $("r-reset").addEventListener("click", () => resetResults());
    $("r-csv").addEventListener("click", () => csvDownload(
      joinedSource("cur", RESULTS_ORDER, "c.is_best").page(-1, 0)
        .map(({ search, id, split_key, ...row }) => row),  // eslint-disable-line no-unused-vars
      "tm_literature_results.csv"));

    const laterBest = debounce(runBest, 150);
    for (const id of ["b-dtype", "b-flag", "b-tm", "b-min"]) $(id).addEventListener("change", runBest);
    for (const id of ["b-dataset", "b-metric"]) $(id).addEventListener("input", laterBest);
    $("b-csv").addEventListener("click", () => csvDownload(all("SELECT * FROM best ORDER BY n DESC, dataset, metric"),
      "tm_literature_best_per_dataset.csv"));

    $("d-search").addEventListener("input", debounce(runDatasets, 150));
    $("d-dtype").addEventListener("change", runDatasets);
    $("p-search").addEventListener("input", debounce(runPapers, 150));
    $("p-category").addEventListener("change", runPapers);

    $("s-run").addEventListener("click", runSql);
    $("s-sql").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runSql(); });
    $("s-csv").addEventListener("click", () => csvDownload(sqlRows, "tm_literature_query.csv"));

    document.querySelectorAll("#lit-app [role=tab]").forEach((b) =>
      b.addEventListener("click", () => showTab(b.dataset.tab)));
  }

  async function main() {
    const t0 = performance.now();
    const [SQL, buf] = await Promise.all([
      initSqlJs({ locateFile: (f) => SQLJS + f }),
      fetch(DB_URL).then((r) => {
        if (!r.ok) throw new Error(`${DB_URL}: HTTP ${r.status}`);
        return r.arrayBuffer();
      }),
    ]);
    const bytes = new Uint8Array(buf);
    db = new SQL.Database(bytes);
    sqlDb = new SQL.Database(bytes);  // separate copy: console queries cannot disturb the views
    // One denormalised, indexed table that every results query starts from.
    db.run(`CREATE TEMP TABLE base AS
      SELECT r.id AS id, COALESCE(r.dataset_id, '(no dataset)') AS dataset, r.metric, r.value, r.unit,
             r.std_or_range, r.split, ${SPLIT_KEY} AS split_key, r.better, r.model, r.is_tm,
             r.is_proposed, r.config, r.source_location, r.notes, r.paper_id, p.title AS paper,
             p.year, p.category, p.url AS paper_url, p.caution, p.tm_variants, d.data_type,
             r.model || ' ' || p.title || ' ' || COALESCE(r.config, '') || ' ' || COALESCE(r.notes, '')
               || ' ' || COALESCE(r.source_location, '') AS search
      FROM results r JOIN papers p ON p.paper_id = r.paper_id
      LEFT JOIN datasets d ON d.dataset_id = r.dataset_id`);
    db.run("CREATE UNIQUE INDEX temp.base_id ON base(id)");
    for (const c of ["dataset", "metric", "data_type", "split_key", "category"]) {
      db.run(`CREATE INDEX temp.base_${c} ON base(${c})`);
    }

    const counts = (sql) => all(sql);
    const ds = counts("SELECT dataset AS v, COUNT(*) AS n FROM base GROUP BY 1 ORDER BY n DESC, v");
    const metrics = counts("SELECT metric AS v, COUNT(*) AS n FROM base GROUP BY 1 ORDER BY n DESC, v");
    ds.forEach(({ v }) => EXACT.dataset.add(String(v).toLowerCase()));
    metrics.forEach(({ v }) => EXACT.metric.add(String(v).toLowerCase()));
    fill("lit-datasets", ds);
    fill("lit-metrics", metrics);
    const dtypes = counts("SELECT data_type AS v, COUNT(*) AS n FROM base WHERE data_type IS NOT NULL GROUP BY 1 ORDER BY n DESC");
    for (const id of ["r-dtype", "b-dtype"]) fill(id, dtypes);
    fill("d-dtype", counts("SELECT data_type AS v, COUNT(*) AS n FROM datasets WHERE data_type IS NOT NULL GROUP BY 1 ORDER BY n DESC"));
    fill("r-split", counts("SELECT split_key AS v, COUNT(*) AS n FROM base GROUP BY 1 ORDER BY n DESC"));
    const cats = counts("SELECT category AS v, COUNT(*) AS n FROM papers GROUP BY 1 ORDER BY n DESC");
    fill("r-category", counts("SELECT category AS v, COUNT(*) AS n FROM base GROUP BY 1 ORDER BY n DESC"));
    fill("p-category", cats);

    wire();
    const n = one("SELECT (SELECT COUNT(*) FROM papers) AS p, (SELECT COUNT(*) FROM datasets) AS d, (SELECT COUNT(*) FROM results) AS r");
    $("lit-status").textContent = `${n.p} papers · ${n.d} datasets · ${n.r.toLocaleString()} reported results`
      + ` · database loaded in ${(performance.now() - t0).toFixed(0)} ms`;
    showTab("results");
  }

  main().catch((err) => {
    $("lit-status").textContent = `Could not load the literature database (${err.message || err}).`;
  });
})();
