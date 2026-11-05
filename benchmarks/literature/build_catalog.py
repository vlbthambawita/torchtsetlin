"""Merge per-paper literature records into a dataset catalog, a flat results table and a README.

Inputs:  benchmarks/literature/papers/*.json  (one record per paper, schema in README.md)
Outputs: benchmarks/literature/datasets.json  (one entry per dataset, with every paper that used it)
         benchmarks/literature/results.csv    (one row per reported number)
         benchmarks/literature/README.md      (generated; do not hand-edit)
         docs/assets/literature/literature.db  (SQLite, queried in the browser by the docs page
                                                docs/literature.md via sql.js)
         benchmarks/literature/space/literature.db  (copy for the Hugging Face Space app)

Run:  python benchmarks/literature/build_catalog.py
"""

import csv
import json
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE / "papers"
DOCS_DB = HERE.parents[1] / "docs" / "assets" / "literature" / "literature.db"

# Spelling variants produced during extraction -> canonical slug.
ALIASES = {
    "kmnist": "kuzushiji-mnist",
    "k-mnist": "kuzushiji-mnist",
    "fmnist": "fashion-mnist",
    "f-mnist": "fashion-mnist",
    "cifar10": "cifar-10",
    "cifar100": "cifar-100",
    "cifar2": "cifar-2",
    "20newsgroups": "20-newsgroups",
    "20-newsgroup": "20-newsgroups",
    "newsgroups-20": "20-newsgroups",
    "reuters": "reuters-21578",
    "noisy-xor-2d": "noisy-2d-xor",
    "2d-noisy-xor": "noisy-2d-xor",
    "uci-har": "ucihar",
    "uci-har-dataset": "ucihar",
    "speech-commands": "google-speech-commands",
    "google-speech-commands-v2": "google-speech-commands",
    "wisconsin-breast-cancer": "breast-cancer-wisconsin",
    "pima-indians-diabetes": "pima-diabetes",
    "mit-bih": "mit-bih-arrhythmia",
    "r8": "reuters-21578-r8",
    "r52": "reuters-21578-r52",
    "emg-data-for-gestures": "emg-gestures",
}

SIZE_KEYS = ("train_size", "val_size", "test_size")


def canon(slug):
    """Canonical dataset slug; ``None`` stays ``None`` (hardware metrics with no dataset)."""
    if not slug:
        return None
    slug = slug.strip().lower().replace("_", "-").replace(" ", "-")
    return ALIASES.get(slug, slug)


def first(values):
    for v in values:
        if v not in (None, "", []):
            return v
    return None


def load_papers():
    papers = []
    for path in sorted(PAPERS.glob("*.json")):
        rec = json.loads(path.read_text())
        for d in rec.get("datasets", []):
            d["dataset_id"] = canon(d.get("dataset_id"))
        for r in rec.get("results", []):
            r["dataset_id"] = canon(r.get("dataset_id"))
        papers.append(rec)
    papers.sort(key=lambda p: (p.get("year") or 0, p["paper_id"]))
    return papers


def build_datasets(papers):
    uses = defaultdict(list)
    for p in papers:
        for d in p.get("datasets", []):
            if d["dataset_id"]:
                uses[d["dataset_id"]].append((p, d))
    catalog = []
    for ds_id, items in sorted(uses.items()):
        ds = [d for _, d in items]
        paper_url = first(d.get("download_url") for d in ds if d.get("download_url_source") == "paper")
        entry = {
            "dataset_id": ds_id,
            "names_in_papers": sorted({d.get("name_in_paper") for d in ds if d.get("name_in_paper")}),
            "data_type": first(d.get("data_type") for d in ds),
            "tasks": sorted({d.get("task") for d in ds if d.get("task")}),
            # Papers use different class subsets (20 Newsgroups: 4 or 20), so keep the range.
            "n_classes": sorted({d["n_classes"] for d in ds if d.get("n_classes") is not None}) or None,
            "task_families": sorted({d["_v2"].values["task_family"] for d in ds if "task_family" in d["_v2"].values}),
            "modalities": sorted({d["_v2"].values["modality"] for d in ds if "modality" in d["_v2"].values}),
            "input_shape_or_features": first(d.get("input_shape_or_features") for d in ds),
            "download_url": paper_url or first(d.get("download_url") for d in ds),
            "download_url_source": "paper" if paper_url else first(d.get("download_url_source") for d in ds),
            "n_papers": len({p["paper_id"] for p, _ in items}),
            "tm_variants": sorted({v for p, _ in items for v in p.get("tm_variants_tested", [])}),
            "usages": [
                {
                    "paper_id": p["paper_id"],
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "tm_variants_tested": p.get("tm_variants_tested", []),
                    "name_in_paper": d.get("name_in_paper"),
                    "task": d.get("task"),
                    **{k: d.get(k) for k in SIZE_KEYS},
                    "split_as_reported": d.get("split_as_reported"),
                    "booleanization": d.get("booleanization"),
                    "notes": d.get("notes"),
                }
                for p, d in items
            ],
        }
        catalog.append(entry)
    for e in catalog:
        nc = e["n_classes"] or []
        e["n_classes_min"], e["n_classes_max"] = (nc[0], nc[-1]) if nc else (None, None)
        e["n_classes_text"] = None if not nc else str(nc[0]) if len(nc) == 1 else ", ".join(map(str, nc))
    catalog.sort(key=lambda e: (-e["n_papers"], e["dataset_id"]))
    return catalog


RESULT_COLS = ["paper_id", "year", "dataset_id", "model", "is_tm", "is_proposed", "metric", "value", "unit",
               "std_or_range", "split", "config", "source_location", "notes"]


def write_results(papers, path):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLS, extrasaction="ignore")
        w.writeheader()
        for p in papers:
            for r in p.get("results", []):
                w.writerow({"paper_id": p["paper_id"], "year": p.get("year"), **r})


def md(v):
    if v is None:
        return "–"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int):
        return f"{v:,}"
    return str(v).replace("|", "/").replace("\n", " ")


def fmt_result(r):
    unit = r.get("unit") or ""
    unit = "%" if unit == "%" else (f" {unit}" if unit else "")
    std = f" ({r['std_or_range']})" if r.get("std_or_range") else ""
    return f"{md(r.get('value'))}{unit}{std}"


def write_readme(papers, catalog, path):
    n_results = sum(len(p.get("results", [])) for p in papers)
    L = [
        "# Datasets and reported results in the Tsetlin-machine literature",
        "",
        "<!-- Generated by build_catalog.py from papers/*.json - edit those records, not this file. -->",
        "",
        f"Extracted from the {len(papers)} Tsetlin-machine papers in `source_documents/papers/`: "
        f"**{len(catalog)} datasets**, **{n_results} reported numbers**.",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `papers/<paper_id>.json` | One record per paper: bibliographic data, TM variant(s), the new feature, "
        "datasets (sizes, splits, Booleanization, download link) and every extracted result. **Source of truth.** |",
        "| `datasets.json` | One entry per dataset: metadata, download link, and every paper/TM variant that used it, "
        "with the split as that paper reported it. |",
        "| `results.csv` | Flat table of all reported numbers (proposed models and baselines). |",
        "",
        "Caveats: numbers are as *reported by the authors* (not reproduced), splits and Booleanization differ "
        "between papers, so cross-paper comparisons need care. `download_url_source = well_known` means the paper "
        "named the dataset but not a link, and the official public source was filled in. Sizes marked "
        "\"standard sizes, not stated in paper\" in `split_as_reported` were filled from the dataset's standard "
        "split. Papers that predate the hard clause-budget cap or use other implementations are not directly "
        "comparable to `torchtsetlin` numbers. Ten FHE / privacy background papers in "
        "`08_privacy_fhe_tm/` contain no Tsetlin-machine experiments and are not included.",
        "",
        "## Dataset catalog",
        "",
        "| Dataset | Type | Task | Classes | Input | Train / Val / Test | Papers | TM variants | Download |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in catalog:
        sizes = sorted({" / ".join(md(u.get(k)) for k in SIZE_KEYS) for u in e["usages"]} - {"– / – / –"})
        url = e.get("download_url")
        link = f"[link]({url})" if url else "not public / n.a."
        L.append(
            f"| `{e['dataset_id']}` | {md(e['data_type'])} | {md(', '.join(e['tasks']))} | {md(e['n_classes_text'])} "
            f"| {md(e['input_shape_or_features'])} | {'<br>'.join(sizes) or '–'} | {e['n_papers']} "
            f"| {md(', '.join(e['tm_variants']))} | {link} |"
        )
    L += ["", "## Papers", ""]
    for p in papers:
        ids = p.get("arxiv_id") and f"arXiv:{p['arxiv_id']}" or p.get("doi") or ""
        L += [
            f"### {p.get('title')} ({p.get('year')})",
            "",
            f"`{p['paper_id']}` · {md(p.get('venue'))} · {ids}"
            + (f" · [code]({p['code_url']})" if p.get("code_url") else ""),
            "",
            f"**TM variants:** {', '.join(p.get('tm_variants_tested', [])) or '–'}  ",
            f"**New feature ({p.get('feature_type')}):** {md(p.get('new_feature'))}  ",
            f"**Datasets:** {', '.join('`' + d['dataset_id'] + '`' for d in p.get('datasets', [])) or '–'}",
            "",
        ]
        res = p.get("results", [])
        if res:
            L += ["| Dataset | Model | Metric | Value | Split | Source |", "|---|---|---|---|---|---|"]
            for r in res:
                model = md(r.get("model")) + (" **(proposed)**" if r.get("is_proposed") else "")
                L.append(
                    f"| {'`' + r['dataset_id'] + '`' if r['dataset_id'] else '(no dataset)'} | {model} | {md(r.get('metric'))} | {fmt_result(r)} "
                    f"| {md(r.get('split'))} | {md(r.get('source_location'))} |"
                )
            L.append("")
        if p.get("caution"):
            L += [f"> **Caution:** {md(p['caution'])}", ""]
        if p.get("extraction_notes"):
            L += [f"> {md(p['extraction_notes'])}", ""]
    path.write_text("\n".join(L) + "\n")


# Which direction is "better" for a metric, used to highlight the best value on the docs page.
# Checked in order; the first matching pattern wins, and an unmatched metric is not highlighted.
_DIRECTION = [
    (r"reduction factor|reduction vs|speed-?up|throughput|efficiency|coverage", "higher"),
    (r"accuracy reduction|error|\bmae\b|\brmse\b|\bece\b|hamming|\bshd\b|ci tests|distance"
     r"|time|latency|delay|energy|power|area|memory|size|cost|usage|parameters|pulses"
     r"|\bluts?\b|\bffs\b|\bdsps?\b|bram|registers|buffers|operations per|increase", "lower"),
    (r"accura|f1|precision|recall|sensitivity|\bauc|auroc|auprc|\biou\b|\bnmi\b|\bari\b"
     r"|purity|spearman|kendall|\brho\b|\btau\b|similar|ndcg|hr@|robust|confidence"
     r"|interpretability|fraction of", "higher"),
]


def metric_direction(metric):
    m = (metric or "").lower()
    for pattern, direction in _DIRECTION:
        if re.search(pattern, m):
            return direction
    return None


def paper_url(p):
    if p.get("arxiv_id"):
        return f"https://arxiv.org/abs/{p['arxiv_id']}"
    if p.get("doi"):
        return f"https://doi.org/{p['doi']}"
    return None


# ------------------------------------------------------------------ schema v2 proposal layer
# benchmarks/literature/mappings/*.csv (written by propose_mappings.py) supply v2 values for the
# fields a record does not have yet. `review` column: blank = proposed, "ok" = reviewed,
# "reject" = dropped. A value in the record itself always wins and is marked "record".
MAPPINGS = HERE / "mappings"
_RANK = {"record": 0, "reviewed": 1, "proposed": 2}


def _csv(name, key):
    path = MAPPINGS / name
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        review = (r.get("review") or "").strip().lower()
        if review == "reject":
            continue
        r["_source"] = "reviewed" if review == "ok" else "proposed"
        out[key(r)] = r
    return out


def _num(v, kind=float):
    try:
        x = float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return int(x) if kind is int else x


class Layer:
    """Collects one row's v2 values and the weakest provenance among the sources used."""

    def __init__(self):
        self.values, self.source = {}, None

    def put(self, key, value, source):
        if value in (None, "", []):
            return
        self.values[key] = value
        if self.source is None or _RANK[source] > _RANK[self.source]:
            self.source = source


def enrich_v2(papers):
    """Attach `_v2` (values + provenance) to every dataset usage and result."""
    tasks = _csv("tasks.csv", lambda r: r["task"])
    dtypes = _csv("data_types.csv", lambda r: r["data_type"])
    uses = _csv("usages.csv", lambda r: (r["paper_id"], int(r["usage_index"])))
    metrics = _csv("metrics.csv", lambda r: r["metric"])
    configs = _csv("configs.csv", lambda r: (r["paper_id"], int(r["result_index"])))
    for p in papers:
        pid = p["paper_id"]
        for i, d in enumerate(p.get("datasets", [])):
            L = Layer()
            t, dt, u = tasks.get(d.get("task")), dtypes.get(d.get("data_type")), uses.get((pid, i))
            rec_bool = d.get("booleanization_v2") or {}
            for key, rec, prop, src in [
                ("task_family", d.get("task_family"), t and t["proposed_task_family"], t),
                ("setting_tags", d.get("setting_tags"),
                 t and [x for x in t["proposed_setting_tags"].split(";") if x], t),
                ("label_type", d.get("label_type"), u and u["proposed_label_type"], u),
                ("modality", d.get("modality"), dt and dt["proposed_modality"], dt),
                ("synthetic", d.get("synthetic"), dt and {"True": True, "False": False}.get(dt["proposed_synthetic"]), dt),
                ("booleanization_method", rec_bool.get("method"), u and u["proposed_booleanization_method"], u),
                ("booleanization_bits", rec_bool.get("bits_per_feature"), u and _num(u["proposed_bits_or_bins"], int), u),
            ]:
                if rec not in (None, "", []):
                    L.put(key, rec, "record")
                elif src is not None:
                    L.put(key, prop, src["_source"])
            if t and t.get("note"):
                L.values["task_note"] = t["note"]
            d["_v2"] = L
        for i, r in enumerate(p.get("results", [])):
            L = Layer()
            m, c = metrics.get(r.get("metric")), configs.get((pid, i))
            rec_m, rec_c = r.get("metric_canonical") or {}, r.get("tm_config") or {}
            for key, rec, prop, src in [
                ("metric_canonical", rec_m.get("name"), m and m["proposed_canonical"], m),
                ("metric_qualifier", rec_m.get("qualifier"), m and m["proposed_qualifier"], m),
                ("averaging", rec_m.get("averaging"), m and m["proposed_averaging"], m),
                ("metric_kind", rec_m.get("kind"), m and m["proposed_kind"], m),
                ("statistic", r.get("statistic"), (c and c["proposed_statistic"]) or (m and m["proposed_statistic"]),
                 c if c and c["proposed_statistic"] else m),
                ("n_runs", r.get("n_runs"), c and _num(c["proposed_n_runs"], int), c),
                ("n_clauses", rec_c.get("n_clauses"), c and _num(c["proposed_n_clauses"], int), c),
                ("clauses_scope", rec_c.get("clauses_scope"), c and c["proposed_clauses_scope"], c),
                ("T", rec_c.get("T"), c and _num(c["proposed_T"]), c),
                ("s", rec_c.get("s"), c and _num(c["proposed_s"]), c),
                ("epochs", rec_c.get("epochs"), c and _num(c["proposed_epochs"], int), c),
                ("n_states", rec_c.get("n_states"), c and _num(c["proposed_n_states"], int), c),
                ("state_bits", rec_c.get("state_bits"), c and _num(c["proposed_state_bits"], int), c),
                ("max_included_literals", rec_c.get("max_included_literals"),
                 c and _num(c["proposed_max_included_literals"], int), c),
                ("patch_size", rec_c.get("patch_size"), c and c["proposed_patch_size"], c),
                ("weighted", rec_c.get("weighted"), c and ({"true": True}.get(c["proposed_weighted"])), c),
                ("implementation", rec_c.get("implementation"), c and c["proposed_implementation"], c),
            ]:
                if rec not in (None, "", []):
                    L.put(key, rec, "record")
                elif src is not None:
                    L.put(key, prop, src["_source"])
            r["_v2"] = L


def task_details(papers):
    """Task-specific attributes as (paper, usage, family, key, value, source) rows."""
    out = []
    for p in papers:
        for i, d in enumerate(p.get("datasets", [])):
            if not d.get("dataset_id"):
                continue
            L = d["_v2"]
            fam, src = L.values.get("task_family"), L.source or "proposed"
            base = (p["paper_id"], i, d["dataset_id"], fam)
            detail = dict(d.get("task_detail") or {})  # record values first
            nc = d.get("n_classes")
            if fam in ("classification", "segmentation", "anomaly_novelty_detection") and nc is not None:
                detail.setdefault("n_classes", nc)
            if fam == "classification" and L.values.get("label_type"):
                detail.setdefault("label_type", L.values["label_type"])
            if fam == "clustering" and nc is not None:
                detail.setdefault("n_ground_truth_labels", nc)
            note = L.values.get("task_note", "")
            if "subtype = forecasting" in note:
                detail.setdefault("subtype", "forecasting")
                detail.setdefault("also_generates", "time_series")
            if "discretized_from" in note:
                detail.setdefault("discretized_from", "regression target")
            if fam == "regression":
                detail.setdefault("target", "not extracted yet")  # PDF pass; notes are not targets
            for k, v in detail.items():
                out.append(base + (k, str(v), "record" if k in (d.get("task_detail") or {}) else src))
    return out


SCHEMA = """
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, venue TEXT,
    arxiv_id TEXT, doi TEXT, url TEXT, code_url TEXT, category TEXT, feature_type TEXT,
    tm_variants TEXT, new_feature TEXT, caution TEXT, extraction_notes TEXT, pdf TEXT
);
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY, data_type TEXT, tasks TEXT, n_classes TEXT,
    n_classes_min INTEGER, n_classes_max INTEGER, task_families TEXT, modality TEXT,
    input_shape_or_features TEXT, download_url TEXT, download_url_source TEXT,
    n_papers INTEGER, tm_variants TEXT, names_in_papers TEXT
);
CREATE TABLE dataset_usages (
    paper_id TEXT REFERENCES papers(paper_id), dataset_id TEXT REFERENCES datasets(dataset_id),
    name_in_paper TEXT, task TEXT, train_size INTEGER, val_size INTEGER, test_size INTEGER,
    split_as_reported TEXT, booleanization TEXT, notes TEXT,
    usage_index INTEGER, n_classes INTEGER, task_family TEXT, setting_tags TEXT, label_type TEXT,
    modality TEXT, synthetic INTEGER, booleanization_method TEXT, booleanization_bits INTEGER,
    v2_source TEXT
);
CREATE TABLE results (
    id INTEGER PRIMARY KEY, paper_id TEXT REFERENCES papers(paper_id),
    dataset_id TEXT REFERENCES datasets(dataset_id), model TEXT, is_tm INTEGER,
    is_proposed INTEGER, metric TEXT, value REAL, unit TEXT, std_or_range TEXT, split TEXT,
    better TEXT, config TEXT, source_location TEXT, notes TEXT,
    metric_canonical TEXT, metric_qualifier TEXT, averaging TEXT, metric_kind TEXT, statistic TEXT,
    n_runs INTEGER, n_clauses INTEGER, clauses_scope TEXT, T REAL, s REAL, epochs INTEGER,
    n_states INTEGER, state_bits INTEGER, max_included_literals INTEGER, patch_size TEXT,
    weighted INTEGER, implementation TEXT, v2_source TEXT
);
CREATE TABLE task_details (
    paper_id TEXT REFERENCES papers(paper_id), usage_index INTEGER,
    dataset_id TEXT REFERENCES datasets(dataset_id), task_family TEXT, key TEXT, value TEXT,
    source TEXT
);
CREATE TABLE vocab (field TEXT, value TEXT, PRIMARY KEY (field, value));
CREATE INDEX results_dataset ON results(dataset_id);
CREATE INDEX results_metric ON results(metric);
CREATE INDEX results_paper ON results(paper_id);
CREATE INDEX usages_dataset ON dataset_usages(dataset_id);
CREATE INDEX results_canonical ON results(metric_canonical);
CREATE INDEX details_usage ON task_details(paper_id, usage_index);
"""


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


USAGE_V2 = ["task_family", "setting_tags", "label_type", "modality", "synthetic", "booleanization_method",
            "booleanization_bits"]
RESULT_V2 = ["metric_canonical", "metric_qualifier", "averaging", "metric_kind", "statistic", "n_runs", "n_clauses",
             "clauses_scope", "T", "s", "epochs", "n_states", "state_bits", "max_included_literals", "patch_size",
             "weighted", "implementation"]


def _v2row(layer, keys):
    def cell(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, list):
            return ", ".join(v)
        return v
    return [cell(layer.values.get(k)) for k in keys] + [layer.source]


def write_sqlite(papers, catalog, path):
    """The whole catalog as one SQLite file (rebuilt from scratch each run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(p["paper_id"], p.get("title"), "; ".join(p.get("authors") or []), p.get("year"),
          p.get("venue"), p.get("arxiv_id"), p.get("doi"), paper_url(p), p.get("code_url"),
          p.get("category"), p.get("feature_type"), ", ".join(p.get("tm_variants_tested", [])),
          p.get("new_feature"), p.get("caution"), p.get("extraction_notes"), p.get("pdf"))
         for p in papers])
    con.executemany(
        "INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(e["dataset_id"], e.get("data_type"), ", ".join(e["tasks"]), e["n_classes_text"],
          e["n_classes_min"], e["n_classes_max"], ", ".join(e["task_families"]), ", ".join(e["modalities"]),
          e.get("input_shape_or_features"), e.get("download_url"), e.get("download_url_source"),
          e["n_papers"], ", ".join(e["tm_variants"]), "; ".join(e["names_in_papers"]))
         for e in catalog])
    con.executemany(
        "INSERT INTO dataset_usages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(p["paper_id"], d["dataset_id"], d.get("name_in_paper"), d.get("task"),
          _int(d.get("train_size")), _int(d.get("val_size")), _int(d.get("test_size")),
          d.get("split_as_reported"), d.get("booleanization"), d.get("notes"),
          i, _int(d.get("n_classes")), *_v2row(d["_v2"], USAGE_V2))
         for p in papers for i, d in enumerate(p.get("datasets", [])) if d["dataset_id"]])
    rows = [r for p in papers for r in p.get("results", [])]
    pids = [p["paper_id"] for p in papers for _ in p.get("results", [])]
    con.executemany(
        f"INSERT INTO results VALUES ({','.join('?' * (15 + len(RESULT_V2) + 1))})",
        [(i, pid, r["dataset_id"], r.get("model"), int(bool(r.get("is_tm"))),
          int(bool(r.get("is_proposed"))), r.get("metric"), r.get("value"), r.get("unit") or "",
          r.get("std_or_range"), r.get("split") or "", metric_direction(r.get("metric")),
          r.get("config"), r.get("source_location"), r.get("notes"), *_v2row(r["_v2"], RESULT_V2))
         for i, (pid, r) in enumerate(zip(pids, rows))])
    con.executemany("INSERT INTO task_details VALUES (?,?,?,?,?,?,?)", task_details(papers))
    vocab = json.loads((HERE / "vocab.json").read_text())
    con.executemany("INSERT INTO vocab VALUES (?,?)",
                    [(k, v) for k, vs in vocab.items() if not k.startswith("_") for v in vs])
    con.commit()
    con.execute("VACUUM")
    con.close()


# Schema v2 (plans/literature-db-v2.md): field path -> vocabulary list in vocab.json. Only fields
# that are present are checked, so records can be migrated one at a time. "not_reported" (the
# paper does not say) is always allowed; null means not extracted yet.
VOCAB_FIELDS = {
    "usage": {"task_family": "task_family", "setting_tags": "setting_tags", "label_type": "label_type",
              "modality": "modality", "domain": "domain",
              "task_detail.subtype": "regression_subtype", "task_detail.output_modality": "generation_output",
              "task_detail.conditioning": "generation_conditioning",
              "booleanization_v2.method": "booleanization_method", "booleanization_v2.fit_on": "fit_on",
              "split.protocol": "split_protocol"},
    "result": {"metric_canonical.kind": "metric_kind", "metric_canonical.averaging": "averaging",
               "statistic": "statistic", "tm_config.clauses_scope": "clauses_scope",
               "tm_config.implementation": "implementation"},
}


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def validate_v2(papers):
    """Reject schema-v2 values outside vocab.json; return {level: {field: (filled, not_reported)}}."""
    vocab = json.loads((HERE / "vocab.json").read_text())
    errors, cover = [], {lvl: {f: [0, 0] for f in fields} for lvl, fields in VOCAB_FIELDS.items()}
    for p in papers:
        items = [("usage", d, d.get("dataset_id")) for d in p.get("datasets", [])]
        items += [("result", r, i) for i, r in enumerate(p.get("results", []))]
        for level, obj, where in items:
            for field, voc in VOCAB_FIELDS[level].items():
                v = _get(obj, field)
                if v is None:
                    continue
                cover[level][field][1 if v == "not_reported" else 0] += 1
                for x in (v if isinstance(v, list) else [v]):
                    if x != "not_reported" and x not in vocab[voc]:
                        errors.append(f"{p['paper_id']} {level} {where}: {field} = {x!r} not in vocab '{voc}'")
    if errors:
        raise SystemExit("schema v2 vocabulary errors:\n  " + "\n  ".join(errors))
    return cover


def main():
    papers = load_papers()
    cover = validate_v2(papers)
    enrich_v2(papers)
    if "--coverage" in sys.argv:
        totals = {"usage": sum(len(p.get("datasets", [])) for p in papers),
                  "result": sum(len(p.get("results", [])) for p in papers)}
        for level, fields in cover.items():
            for field, (filled, nr) in fields.items():
                print(f"{level:6} {field:32} filled {filled:5}  not_reported {nr:5}  "
                      f"missing {totals[level] - filled - nr:5} / {totals[level]}")
    catalog = build_datasets(papers)
    (HERE / "datasets.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    write_results(papers, HERE / "results.csv")
    write_readme(papers, catalog, HERE / "README.md")
    write_sqlite(papers, catalog, DOCS_DB)
    shutil.copyfile(DOCS_DB, HERE / "space" / "literature.db")
    n = sum(len(p.get("results", [])) for p in papers)
    print(f"{len(papers)} papers, {len(catalog)} datasets, {n} results")


if __name__ == "__main__":
    main()
