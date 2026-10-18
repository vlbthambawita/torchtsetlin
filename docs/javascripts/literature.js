/* Interactive tables for docs/literature.md.
 *
 * Data: docs/assets/literature/{results,datasets}.json, written by
 * benchmarks/literature/build_catalog.py. Filtering and the "best value" computation run here
 * in plain JS; Tabulator only renders, groups, sorts and exports what it is given.
 */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  if (!$("lit-results") || typeof Tabulator === "undefined") return;

  const DATA = "../assets/literature/";
  const LABEL = { dataset: "Dataset", metric: "Metric", paper: "Paper", category: "Category", year: "Year" };
  const SEP = "\u0001";
  let ALL = [];
  let best = new Set();
  let groupsOpen = true;
  let table;

  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v) => (typeof v === "number"
    ? v.toLocaleString("en-US", { maximumFractionDigits: 6 }) : esc(v));

  function groupFields() {
    const v = $("lit-group").value;
    return v ? v.split(",") : [];
  }

  // An exact pick from the datalist matches exactly; anything else is a substring match.
  function matcher(id, field, options) {
    const q = $(id).value.trim().toLowerCase();
    if (!q) return () => true;
    if (options.has(q)) return (r) => String(r[field]).toLowerCase() === q;
    return (r) => String(r[field]).toLowerCase().includes(q);
  }

  const OPTIONS = { dataset: new Set(), metric: new Set() };

  function filtered() {
    const q = $("lit-search").value.trim().toLowerCase();
    const byDataset = matcher("lit-dataset", "dataset", OPTIONS.dataset);
    const byMetric = matcher("lit-metric", "metric", OPTIONS.metric);
    const dtype = $("lit-dtype").value, split = $("lit-split").value, cat = $("lit-category").value;
    const tmOnly = $("lit-tm").checked, proposedOnly = $("lit-proposed").checked;
    const noCaution = $("lit-caution").checked;
    return ALL.filter((r) =>
      byDataset(r) && byMetric(r)
      && (!dtype || r.data_type === dtype)
      && (!split || r.split === split)
      && (!cat || r.category === cat)
      && (!tmOnly || r.is_tm)
      && (!proposedOnly || r.is_proposed)
      && (!noCaution || !r.caution)
      && (!q || r._text.includes(q)));
  }

  // Rows with no stated split are compared with test rows; train/val/cv stay separate.
  const splitKey = (r) => r.split || "test";

  // Best value per (grouping fields, metric, unit, split). A group needs at least two
  // comparable rows to have a "best"; ties are all marked.
  function computeBest(rows, fields) {
    const key = (r) => fields.map((f) => r[f]).concat([r.metric, r.unit, splitKey(r)]).join(SEP);
    const top = new Map();
    for (const r of rows) {
      if (!r.better || typeof r.value !== "number") continue;
      const k = key(r), cur = top.get(k);
      if (cur === undefined) top.set(k, { v: r.value, n: 1 });
      else {
        cur.n += 1;
        if (r.better === "higher" ? r.value > cur.v : r.value < cur.v) cur.v = r.value;
      }
    }
    const out = new Set();
    for (const r of rows) {
      const t = r.better && top.get(key(r));
      if (t && t.n > 1 && t.v === r.value) out.add(r.id);
    }
    return out;
  }

  // Default order: most-reported datasets and metrics first, best value first within a metric.
  function presort(rows) {
    const freq = (f) => rows.reduce((m, r) => m.set(r[f], (m.get(r[f]) || 0) + 1), new Map());
    const fd = freq("dataset"), fm = freq("metric");
    const v = (r) => (typeof r.value !== "number" ? Infinity : r.better === "lower" ? r.value : -r.value);
    rows.sort((a, b) => (fd.get(b.dataset) - fd.get(a.dataset)) || a.dataset.localeCompare(b.dataset)
      || (fm.get(b.metric) - fm.get(a.metric)) || a.metric.localeCompare(b.metric)
      || a.unit.localeCompare(b.unit) || v(a) - v(b));
    return rows;
  }

  function refresh() {
    const fields = groupFields();
    let rows = filtered();
    best = computeBest(rows, fields);
    const nBest = best.size;
    if ($("lit-best").checked) rows = rows.filter((r) => best.has(r.id));
    table.setGroupBy(fields.length ? fields : false);
    table.setGroupHeader(fields.map((f) => (value, count) =>
      `<span class="lit-group-label">${LABEL[f]}:</span> ${esc(value)} <span class="lit-group-count">(${count})</span>`));
    table.clearSort();
    table.replaceData(presort(rows));
    $("lit-status").textContent = `${rows.length.toLocaleString()} of ${ALL.length.toLocaleString()} results`
      + ` · ${nBest.toLocaleString()} best values highlighted · click a row for its config and notes`;
    groupsOpen = true;
    $("lit-collapse").textContent = "Collapse groups";
  }

  function valueFormatter(cell) {
    const r = cell.getData(), el = cell.getElement(), isBest = best.has(r.id);
    el.classList.toggle("lit-best-cell", isBest);
    const unit = r.unit === "%" ? "%" : (r.unit ? ` ${esc(r.unit)}` : "");
    const std = r.std ? ` <span class="lit-std">${esc(r.std)}</span>` : "";
    const arrow = r.better ? `<span class="lit-dir" title="${r.better} is better">${r.better === "higher" ? "↑" : "↓"}</span>` : "";
    return `${isBest ? "★ " : ""}${num(r.value)}${unit}${std}${arrow}`;
  }

  function modelFormatter(cell) {
    const r = cell.getData();
    return esc(r.model)
      + (r.is_proposed ? ' <span class="lit-badge lit-proposed">proposed</span>' : "")
      + (r.is_tm ? "" : ' <span class="lit-badge">non-TM</span>');
  }

  function paperFormatter(cell) {
    const r = cell.getData();
    const title = `${esc(r.paper)} (${esc(r.year)})`;
    const flag = r.caution ? `<span class="lit-caution" title="${esc(r.caution)}">⚠</span> ` : "";
    return flag + (r.paper_url ? `<a href="${esc(r.paper_url)}" target="_blank" rel="noopener">${title}</a>` : title);
  }

  function showDetail(r) {
    const box = $("lit-detail");
    const line = (k, v) => (v ? `<dt>${k}</dt><dd>${esc(v)}</dd>` : "");
    box.innerHTML = `<button type="button" class="lit-close" aria-label="Close">×</button>`
      + `<strong>${esc(r.model)}</strong> on <code>${esc(r.dataset)}</code>: ${esc(r.metric)} = ${num(r.value)} ${esc(r.unit)}`
      + `<dl>${line("Paper", `${r.paper} (${r.year})`)}${line("TM variants in paper", r.tm_variants)}`
      + `${line("⚠ Caution", r.caution)}${line("Config", r.config)}${line("Source", r.source)}${line("Notes", r.notes)}`
      + `${line("Extraction notes (paper)", r.extraction_notes)}${line("Record", `benchmarks/literature/papers/${r.paper_id}.json`)}</dl>`;
    box.hidden = false;
    box.querySelector(".lit-close").onclick = () => { box.hidden = true; };
  }

  function fillSelect(id, values) {
    const sel = $(id);
    for (const v of [...new Set(values)].filter(Boolean).sort()) sel.add(new Option(v, v));
  }

  function fillList(id, values, set) {
    const list = $(id);
    for (const v of [...new Set(values)].filter(Boolean).sort()) {
      set.add(String(v).toLowerCase());
      const o = document.createElement("option");
      o.value = v;
      list.appendChild(o);
    }
  }

  function buildResults() {
    const detail = document.createElement("div");
    detail.id = "lit-detail";
    detail.className = "lit-detail";
    detail.hidden = true;
    $("lit-results").before(detail);

    table = new Tabulator("#lit-results", {
      data: [],
      height: "70vh",
      layout: "fitDataStretch",
      groupStartOpen: true,
      groupToggleElement: "header",
      placeholder: "No results match these filters",
      columnDefaults: { tooltip: true, headerTooltip: true },
      columns: [
        { title: "Dataset", field: "dataset", width: 150 },
        { title: "Metric", field: "metric", width: 170 },
        { title: "Value", field: "value", sorter: "number", hozAlign: "right", formatter: valueFormatter,
          tooltip: false, width: 150 },
        { title: "Unit", field: "unit", visible: false, download: true },
        { title: "± / range", field: "std", visible: false, download: true },
        { title: "Split", field: "split", width: 70 },
        { title: "Model", field: "model", formatter: modelFormatter, width: 300 },
        { title: "Paper", field: "paper", formatter: paperFormatter, width: 280 },
        { title: "Year", field: "year", width: 65, hozAlign: "right" },
        { title: "Type", field: "data_type", width: 90 },
        { title: "Source", field: "source", width: 120 },
        { title: "Config", field: "config", width: 220 },
        { title: "Proposed", field: "is_proposed", visible: false, download: true },
        { title: "TM", field: "is_tm", visible: false, download: true },
        { title: "Direction", field: "better", visible: false, download: true },
        { title: "Notes", field: "notes", visible: false, download: true },
        { title: "Paper id", field: "paper_id", visible: false, download: true },
        { title: "Caution", field: "caution", visible: false, download: true },
      ],
    });
    table.on("rowClick", (e, row) => { if (!e.target.closest("a")) showDetail(row.getData()); });
    table.on("tableBuilt", refresh);

    for (const id of ["lit-group", "lit-dtype", "lit-split", "lit-category", "lit-tm", "lit-proposed", "lit-caution", "lit-best"]) {
      $(id).addEventListener("change", refresh);
    }
    let timer;
    for (const id of ["lit-search", "lit-dataset", "lit-metric"]) {
      $(id).addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(refresh, 200); });
    }
    $("lit-collapse").addEventListener("click", () => {
      groupsOpen = !groupsOpen;
      table.getGroups().forEach((g) => (groupsOpen ? g.show() : g.hide()));
      $("lit-collapse").textContent = groupsOpen ? "Collapse groups" : "Expand groups";
    });
    $("lit-reset").addEventListener("click", () => {
      for (const id of ["lit-search", "lit-dataset", "lit-metric", "lit-dtype", "lit-split", "lit-category"]) $(id).value = "";
      for (const id of ["lit-tm", "lit-proposed", "lit-caution", "lit-best"]) $(id).checked = false;
      $("lit-group").value = "dataset,metric";
      refresh();
    });
    $("lit-csv").addEventListener("click", () => table.download("csv", "tm-literature-results.csv"));
  }

  function linkFormatter(cell) {
    const url = cell.getValue();
    return url ? `<a href="${esc(url)}" target="_blank" rel="noopener">download</a>` : "not public";
  }

  function buildDatasets(rows) {
    const dt = new Tabulator("#lit-datasets", {
      data: rows,
      height: "55vh",
      layout: "fitDataStretch",
      columnDefaults: { tooltip: true, headerTooltip: true },
      initialSort: [{ column: "n_papers", dir: "desc" }],
      columns: [
        { title: "Dataset", field: "dataset", headerFilter: "input", width: 170 },
        { title: "Type", field: "data_type", headerFilter: "list",
          headerFilterParams: { valuesLookup: true, clearable: true }, width: 100 },
        { title: "Task", field: "tasks", headerFilter: "input", width: 200 },
        { title: "Classes", field: "n_classes", sorter: "number", hozAlign: "right", width: 80 },
        { title: "Input", field: "input", headerFilter: "input", width: 200 },
        { title: "Train / val / test (per paper)", field: "sizes", width: 230 },
        { title: "Papers", field: "n_papers", sorter: "number", hozAlign: "right", width: 75 },
        { title: "TM variants", field: "tm_variants", headerFilter: "input", width: 260 },
        { title: "Link", field: "download_url", formatter: linkFormatter, tooltip: false, width: 90 },
        { title: "Used in", field: "papers", headerFilter: "input", width: 320 },
      ],
    });
    dt.on("rowClick", (e, row) => {
      if (e.target.closest("a")) return;
      $("lit-dataset").value = row.getData().dataset;
      refresh();
      $("lit-status").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  Promise.all([
    fetch(DATA + "results.json").then((r) => r.json()),
    fetch(DATA + "datasets.json").then((r) => r.json()),
  ]).then(([results, datasets]) => {
    ALL = results;
    for (const r of ALL) {
      r._text = [r.model, r.paper, r.config, r.notes, r.source, r.tm_variants, r.dataset, r.metric]
        .join(" ").toLowerCase();
    }
    fillList("lit-dataset-list", ALL.map((r) => r.dataset), OPTIONS.dataset);
    fillList("lit-metric-list", ALL.map((r) => r.metric), OPTIONS.metric);
    fillSelect("lit-dtype", ALL.map((r) => r.data_type));
    fillSelect("lit-split", ALL.map((r) => r.split));
    fillSelect("lit-category", ALL.map((r) => r.category));
    buildResults();
    buildDatasets(datasets);
  }).catch((err) => {
    $("lit-status").textContent = `Could not load the literature data (${err}).`;
  });
})();
