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


# ------------------------------------------------------------------ schema v2 layers
# Precedence per field: the paper record itself > the PDF pass (papers_v2/<paper_id>.json, see
# EXTRACTION_V2.md) > a reviewed proposal > an automatic proposal (mappings/*.csv, written by
# propose_mappings.py; `review` column: blank = proposed, "ok" = reviewed, "reject" = dropped).
# Each row's `v2_source` is the weakest source among its fields. "not_reported" (the paper does not
# say) is kept as text in text columns; numeric columns get NULL and the field name is listed in
# `v2_not_reported`, so numeric comparisons never see a string.
MAPPINGS = HERE / "mappings"
SIDECARS = HERE / "papers_v2"
_RANK = {"record": 0, "pdf": 1, "reviewed": 2, "proposed": 3}
NR = "not_reported"


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


def load_sidecars():
    return {p.stem: json.loads(p.read_text()) for p in sorted(SIDECARS.glob("*.json"))} if SIDECARS.exists() else {}


class Layer:
    """One row's v2 values, the weakest provenance used, and the fields reported as absent."""

    def __init__(self):
        self.values, self.source, self.not_reported, self.inferred = {}, None, [], []
        self.metric_source = None  # canonical-metric fields are tracked apart (they map per metric name)

    def _src(self, source, attr="source"):
        cur = getattr(self, attr)
        if cur is None or _RANK[source] > _RANK[cur]:
            setattr(self, attr, source)

    def offer(self, key, candidates, attr="source"):
        """candidates: (value, source) pairs, strongest first; the first usable one wins."""
        for value, source in candidates:
            if value is None or value == "" or value == []:
                continue
            if value == NR:
                self.not_reported.append(key)
            else:
                self.values[key] = value
            self._src(source, attr)
            return


def _prop(row, col, conv=None):
    """(value, source) of one proposal-CSV cell; a missing row yields nothing."""
    if row is None:
        return (None, "proposed")
    return (conv(row[col]) if conv else row[col], row["_source"])


def _sc_usage(sc, i):
    return next((u for u in (sc or {}).get("usages", []) if u.get("usage_index") == i), {})


def _sc_config(sc, i):
    return next((c for c in (sc or {}).get("configs", []) if i in (c.get("result_indices") or [])), {})


def enrich_v2(papers, sidecars):
    """Attach `_v2` (a Layer) to every paper, dataset usage and result."""
    tasks = _csv("tasks.csv", lambda r: r["task"])
    dtypes = _csv("data_types.csv", lambda r: r["data_type"])
    uses = _csv("usages.csv", lambda r: (r["paper_id"], int(r["usage_index"])))
    metrics = _csv("metrics.csv", lambda r: r["metric"])
    configs = _csv("configs.csv", lambda r: (r["paper_id"], int(r["result_index"])))
    for p in papers:
        pid, sc = p["paper_id"], sidecars.get(p["paper_id"])
        P = Layer()
        scp = (sc or {}).get("paper") or {}
        for key in ("primary_task_family", "code_available", "seeds_stated", "hyperparameters_complete",
                    "interpretability_evidence"):
            P.offer(key, [(p.get(key), "record"), (scp.get(key), "pdf")])
        p["_v2"] = P
        for i, d in enumerate(p.get("datasets", [])):
            L = Layer()
            t, dt, u = tasks.get(d.get("task")), dtypes.get(d.get("data_type")), uses.get((pid, i))
            su = _sc_usage(sc, i)
            sb, ss = su.get("booleanization") or {}, su.get("split") or {}
            rb = d.get("booleanization_v2") or {}

            L.offer("task_family", [(d.get("task_family"), "record"), (su.get("task_family"), "pdf"),
                                    _prop(t, "proposed_task_family")])
            L.offer("setting_tags", [(d.get("setting_tags"), "record"), (su.get("setting_tags"), "pdf"),
                                     _prop(t, "proposed_setting_tags", lambda v: [x for x in v.split(";") if x])])
            L.offer("label_type", [(d.get("label_type"), "record"), (su.get("label_type"), "pdf"),
                                   _prop(u, "proposed_label_type")])
            L.offer("modality", [(d.get("modality"), "record"), (su.get("modality"), "pdf"),
                                 _prop(dt, "proposed_modality")])
            L.offer("synthetic", [(d.get("synthetic"), "record"), (su.get("synthetic"), "pdf"),
                                  _prop(dt, "proposed_synthetic", {"True": True, "False": False}.get)])
            L.offer("domain", [(d.get("domain"), "record"), (su.get("domain"), "pdf")])
            L.offer("booleanization_method", [(rb.get("method"), "record"), (sb.get("method"), "pdf"),
                                              _prop(u, "proposed_booleanization_method")])
            L.offer("booleanization_bits", [(rb.get("bits_per_feature"), "record"), (sb.get("bits_per_feature"), "pdf"),
                                            _prop(u, "proposed_bits_or_bins", lambda v: _num(v, int))])
            for key in ("n_bins", "n_boolean_features", "n_literals", "fit_on", "threshold"):
                L.offer(f"booleanization_{key}", [(rb.get(key), "record"), (sb.get(key), "pdf")])
            L.offer("preprocessing", [(d.get("preprocessing"), "record"), (su.get("preprocessing"), "pdf")])
            L.offer("split_protocol", [((d.get("split") or {}).get("protocol"), "record"), (ss.get("protocol"), "pdf")])
            L.offer("k_folds", [(ss.get("k_folds"), "pdf")])
            L.offer("validation_used_for", [(ss.get("validation_used_for"), "pdf")])
            L.inferred = sorted((su.get("inferred") or {}).keys())
            L.task_detail = dict(su.get("task_detail") or {})
            L.task_detail_source = "pdf" if su.get("task_detail") else None
            if t and t.get("note"):
                L.values["task_note"] = t["note"]
            d["_v2"] = L
        for i, r in enumerate(p.get("results", [])):
            L = Layer()
            m, c = metrics.get(r.get("metric")), configs.get((pid, i))
            scc = _sc_config(sc, i)
            stc = scc.get("tm_config") if isinstance(scc.get("tm_config"), dict) else {}
            rec_m, rec_c = r.get("metric_canonical") or {}, r.get("tm_config") or {}

            L.offer("metric_canonical", [(rec_m.get("name"), "record"), _prop(m, "proposed_canonical")], attr="metric_source")
            L.offer("metric_qualifier", [(rec_m.get("qualifier"), "record"), _prop(m, "proposed_qualifier")], attr="metric_source")
            L.offer("averaging", [(rec_m.get("averaging"), "record"), _prop(m, "proposed_averaging")], attr="metric_source")
            L.offer("metric_kind", [(rec_m.get("kind"), "record"), _prop(m, "proposed_kind")], attr="metric_source")
            L.offer("statistic", [(r.get("statistic"), "record"), (scc.get("statistic"), "pdf"),
                                  _prop(c, "proposed_statistic"), _prop(m, "proposed_statistic")])
            L.offer("n_runs", [(r.get("n_runs"), "record"), (scc.get("n_runs"), "pdf"),
                               _prop(c, "proposed_n_runs", lambda v: _num(v, int))])
            L.offer("selection_on_test", [(r.get("selection_on_test"), "record"), (scc.get("selection_on_test"), "pdf")])
            L.inferred = sorted((scc.get("inferred") or {}).keys())
            if scc.get("tm_config") == NR:
                L.not_reported.append("tm_config")
            ints = {"n_clauses", "epochs", "n_states", "state_bits", "max_included_literals"}
            for key in ("n_clauses", "clauses_scope", "T", "s", "epochs", "n_states", "state_bits",
                        "max_included_literals", "patch_size", "weighted", "implementation", "hardware"):
                conv = (lambda v: _num(v, int)) if key in ints else _num if key in ("T", "s") else \
                    (lambda v: {"true": True}.get(v)) if key == "weighted" else None
                cand = [(rec_c.get(key), "record"), (stc.get(key), "pdf")]
                if key != "hardware":
                    cand.append(_prop(c, f"proposed_{key}", conv))
                L.offer(key, cand)
            r["_v2"] = L


def task_details(papers):
    """Task-specific attributes as (paper, usage, dataset, family, key, value, source) rows."""
    out = []
    for p in papers:
        for i, d in enumerate(p.get("datasets", [])):
            if not d.get("dataset_id"):
                continue
            L = d["_v2"]
            fam = L.values.get("task_family")
            detail = {k: (v, "record") for k, v in (d.get("task_detail") or {}).items()}
            for k, v in L.task_detail.items():
                detail.setdefault(k, (v, "pdf"))
            weak = L.source or "proposed"
            nc = d.get("n_classes")
            if fam in ("classification", "segmentation", "anomaly_novelty_detection") and nc is not None:
                detail.setdefault("n_classes", (nc, "record"))
            if fam == "classification" and L.values.get("label_type"):
                detail.setdefault("label_type", (L.values["label_type"], weak))
            if fam == "clustering" and nc is not None:
                detail.setdefault("n_ground_truth_labels", (nc, "record"))
            note = L.values.get("task_note", "")
            if "subtype = forecasting" in note:
                detail.setdefault("subtype", ("forecasting", weak))
                detail.setdefault("also_generates", ("time_series", weak))
            if "discretized_from" in note:
                detail.setdefault("discretized_from", ("regression target", weak))
            if fam == "regression":
                detail.setdefault("target", ("not extracted yet", weak))
            for k, (v, src) in detail.items():
                if v is None:
                    continue
                out.append((p["paper_id"], i, d["dataset_id"], fam, k,
                            v if isinstance(v, str) else json.dumps(v), src))
    return out


def corrections(sidecars):
    return [(pid, c.get("where"), json.dumps(c.get("record")), json.dumps(c.get("pdf")), c.get("location"),
             c.get("note"), "open") for pid, sc in sidecars.items() for c in sc.get("corrections", [])]


# Sidecar fields with a controlled vocabulary (EXTRACTION_V2.md).
SIDECAR_VOCAB = {
    "paper": {"primary_task_family": "task_family", "hyperparameters_complete": "hyperparameters_complete",
              "interpretability_evidence": "interpretability_evidence"},
    "usage": {"task_family": "task_family", "setting_tags": "setting_tags", "label_type": "label_type",
              "modality": "modality", "domain": "domain", "booleanization.method": "booleanization_method",
              "booleanization.fit_on": "fit_on", "split.protocol": "split_protocol",
              "split.validation_used_for": "validation_used_for", "task_detail.subtype": "regression_subtype",
              "task_detail.output_modality": "generation_output", "task_detail.conditioning": "generation_conditioning"},
    "config": {"statistic": "statistic", "tm_config.clauses_scope": "clauses_scope",
               "tm_config.implementation": "implementation"},
}
_NUMERIC = {"usage": ["n_classes", "booleanization.bits_per_feature", "booleanization.n_bins",
                      "booleanization.n_boolean_features", "booleanization.n_literals", "split.k_folds"],
            "config": ["n_runs", "tm_config.n_clauses", "tm_config.T", "tm_config.s", "tm_config.epochs",
                       "tm_config.n_states", "tm_config.state_bits", "tm_config.max_included_literals"]}


def check_sidecar(paper, sc, vocab):
    """Errors in one papers_v2 file against its record and vocab.json (empty list = valid)."""
    err, pid = [], paper["paper_id"]
    if sc.get("paper_id") != pid:
        err.append(f"paper_id is {sc.get('paper_id')!r}, expected {pid!r}")

    def check(obj, level, where):
        for field, voc in SIDECAR_VOCAB[level].items():
            v = _get(obj, field)
            for x in (v if isinstance(v, list) else [v]):
                if x not in (None, NR) and x not in vocab[voc]:
                    err.append(f"{where}: {field} = {x!r} not in vocab '{voc}'")
        for field in _NUMERIC.get(level, []):
            v = _get(obj, field)
            if v not in (None, NR) and (isinstance(v, bool) or not isinstance(v, (int, float))):
                err.append(f"{where}: {field} = {v!r} must be a number, {NR!r} or null")

    check(sc.get("paper") or {}, "paper", "paper")
    n_use = len(paper.get("datasets", []))
    seen = {}
    for u in sc.get("usages", []):
        i = u.get("usage_index")
        if not isinstance(i, int) or not 0 <= i < n_use:
            err.append(f"usage_index {i!r} out of range 0..{n_use - 1}")
            continue
        if i in seen:
            err.append(f"usage_index {i} appears twice")
        seen[i] = u
        if canon(u.get("dataset_id")) != paper["datasets"][i].get("dataset_id"):
            err.append(f"usages[{i}]: dataset_id {u.get('dataset_id')!r} != record "
                       f"{paper['datasets'][i].get('dataset_id')!r}")
        check(u, "usage", f"usages[{i}]")
    missing = sorted(set(range(n_use)) - set(seen))
    if missing:
        err.append(f"no usages entry for usage_index {missing}")
    n_res, owner = len(paper.get("results", [])), {}
    for g, c in enumerate(sc.get("configs", [])):
        tc = c.get("tm_config")
        if tc not in (None, NR) and not isinstance(tc, dict):
            err.append(f"configs[{g}]: tm_config must be an object, {NR!r} or null")
        for i in c.get("result_indices") or []:
            if not isinstance(i, int) or not 0 <= i < n_res:
                err.append(f"configs[{g}]: result_index {i!r} out of range 0..{n_res - 1}")
            elif i in owner:
                err.append(f"result_index {i} is in configs[{owner[i]}] and configs[{g}]")
            else:
                owner[i] = g
        check(c, "config", f"configs[{g}]")
    missing = sorted(set(range(n_res)) - set(owner))
    if missing:
        err.append(f"{len(missing)} result_index values in no configs group: {missing[:20]}"
                   f"{' ...' if len(missing) > 20 else ''}")
    return err


SCHEMA = """
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, venue TEXT,
    arxiv_id TEXT, doi TEXT, url TEXT, code_url TEXT, category TEXT, feature_type TEXT,
    tm_variants TEXT, new_feature TEXT, caution TEXT, extraction_notes TEXT, pdf TEXT,
    primary_task_family TEXT, code_available INTEGER, seeds_stated INTEGER,
    hyperparameters_complete TEXT, interpretability_evidence TEXT, v2_not_reported TEXT, v2_inferred TEXT,
    v2_source TEXT
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
    modality TEXT, synthetic INTEGER, domain TEXT, booleanization_method TEXT, booleanization_bits INTEGER,
    booleanization_n_bins INTEGER, booleanization_n_boolean_features INTEGER, booleanization_n_literals INTEGER,
    booleanization_fit_on TEXT, booleanization_threshold TEXT, preprocessing TEXT, split_protocol TEXT,
    k_folds INTEGER, validation_used_for TEXT, v2_not_reported TEXT, v2_inferred TEXT, v2_source TEXT
);
CREATE TABLE results (
    id INTEGER PRIMARY KEY, paper_id TEXT REFERENCES papers(paper_id),
    dataset_id TEXT REFERENCES datasets(dataset_id), model TEXT, is_tm INTEGER,
    is_proposed INTEGER, metric TEXT, value REAL, unit TEXT, std_or_range TEXT, split TEXT,
    better TEXT, config TEXT, source_location TEXT, notes TEXT,
    metric_canonical TEXT, metric_qualifier TEXT, averaging TEXT, metric_kind TEXT, statistic TEXT,
    n_runs INTEGER, selection_on_test INTEGER, n_clauses INTEGER, clauses_scope TEXT, T REAL, s REAL, epochs INTEGER,
    n_states INTEGER, state_bits INTEGER, max_included_literals INTEGER, patch_size TEXT,
    weighted INTEGER, implementation TEXT, hardware TEXT, v2_not_reported TEXT, v2_inferred TEXT,
    v2_source TEXT, metric_source TEXT
);
CREATE TABLE corrections (
    paper_id TEXT REFERENCES papers(paper_id), location_in_record TEXT, record_value TEXT,
    pdf_value TEXT, pdf_location TEXT, note TEXT, status TEXT
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


PAPER_V2 = ["primary_task_family", "code_available", "seeds_stated", "hyperparameters_complete",
            "interpretability_evidence"]
USAGE_V2 = ["task_family", "setting_tags", "label_type", "modality", "synthetic", "domain", "booleanization_method",
            "booleanization_bits", "booleanization_n_bins", "booleanization_n_boolean_features",
            "booleanization_n_literals", "booleanization_fit_on", "booleanization_threshold", "preprocessing",
            "split_protocol", "k_folds", "validation_used_for"]
RESULT_V2 = ["metric_canonical", "metric_qualifier", "averaging", "metric_kind", "statistic", "n_runs",
             "selection_on_test", "n_clauses", "clauses_scope", "T", "s", "epochs", "n_states", "state_bits",
             "max_included_literals", "patch_size", "weighted", "implementation", "hardware"]


def _v2row(layer, keys):
    def cell(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, list):
            return ", ".join(x if isinstance(x, str) else json.dumps(x) for x in v)
        if isinstance(v, dict):
            return json.dumps(v)
        return v
    return [cell(layer.values.get(k)) for k in keys] + [", ".join(layer.not_reported) or None,
                                                       ", ".join(layer.inferred) or None, layer.source]


def write_sqlite(papers, catalog, path, sidecars):
    """The whole catalog as one SQLite file (rebuilt from scratch each run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany(
        f"INSERT INTO papers VALUES ({','.join('?' * (16 + len(PAPER_V2) + 3))})",
        [(p["paper_id"], p.get("title"), "; ".join(p.get("authors") or []), p.get("year"),
          p.get("venue"), p.get("arxiv_id"), p.get("doi"), paper_url(p), p.get("code_url"),
          p.get("category"), p.get("feature_type"), ", ".join(p.get("tm_variants_tested", [])),
          p.get("new_feature"), p.get("caution"), p.get("extraction_notes"), p.get("pdf"),
          *_v2row(p["_v2"], PAPER_V2))
         for p in papers])
    con.executemany(
        "INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(e["dataset_id"], e.get("data_type"), ", ".join(e["tasks"]), e["n_classes_text"],
          e["n_classes_min"], e["n_classes_max"], ", ".join(e["task_families"]), ", ".join(e["modalities"]),
          e.get("input_shape_or_features"), e.get("download_url"), e.get("download_url_source"),
          e["n_papers"], ", ".join(e["tm_variants"]), "; ".join(e["names_in_papers"]))
         for e in catalog])
    con.executemany(
        f"INSERT INTO dataset_usages VALUES ({','.join('?' * (12 + len(USAGE_V2) + 3))})",
        [(p["paper_id"], d["dataset_id"], d.get("name_in_paper"), d.get("task"),
          _int(d.get("train_size")), _int(d.get("val_size")), _int(d.get("test_size")),
          d.get("split_as_reported"), d.get("booleanization"), d.get("notes"),
          i, _int(d.get("n_classes")), *_v2row(d["_v2"], USAGE_V2))
         for p in papers for i, d in enumerate(p.get("datasets", [])) if d["dataset_id"]])
    rows = [r for p in papers for r in p.get("results", [])]
    pids = [p["paper_id"] for p in papers for _ in p.get("results", [])]
    con.executemany(
        f"INSERT INTO results VALUES ({','.join('?' * (15 + len(RESULT_V2) + 4))})",
        [(i, pid, r["dataset_id"], r.get("model"), int(bool(r.get("is_tm"))),
          int(bool(r.get("is_proposed"))), r.get("metric"), r.get("value"), r.get("unit") or "",
          r.get("std_or_range"), r.get("split") or "", metric_direction(r.get("metric")),
          r.get("config"), r.get("source_location"), r.get("notes"), *_v2row(r["_v2"], RESULT_V2),
          r["_v2"].metric_source)
         for i, (pid, r) in enumerate(zip(pids, rows))])
    con.executemany("INSERT INTO task_details VALUES (?,?,?,?,?,?,?)", task_details(papers))
    con.executemany("INSERT INTO corrections VALUES (?,?,?,?,?,?,?)", corrections(sidecars))
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
    sidecars = load_sidecars()
    vocab = json.loads((HERE / "vocab.json").read_text())
    by_id = {p["paper_id"]: p for p in papers}
    if "--check-v2" in sys.argv:  # used by the PDF-pass agents: validate sidecars, write nothing
        ids = sys.argv[sys.argv.index("--check-v2") + 1:] or sorted(sidecars)
        bad = 0
        for pid in ids:
            if pid not in sidecars:
                print(f"{pid}: no file papers_v2/{pid}.json")
                bad += 1
                continue
            errs = check_sidecar(by_id[pid], sidecars[pid], vocab) if pid in by_id else ["no such paper record"]
            print(f"{pid}: {'OK' if not errs else f'{len(errs)} error(s)'}")
            for e in errs:
                print(f"  - {e}")
            bad += bool(errs)
        raise SystemExit(1 if bad else 0)
    errs = [f"{pid}: {e}" for pid, sc in sidecars.items()
            for e in (check_sidecar(by_id[pid], sc, vocab) if pid in by_id else ["no such paper record"])]
    if errs:
        raise SystemExit("papers_v2 errors (run --check-v2 <paper_id>):\n  " + "\n  ".join(errs))
    enrich_v2(papers, sidecars)
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
    write_sqlite(papers, catalog, DOCS_DB, sidecars)
    shutil.copyfile(DOCS_DB, HERE / "space" / "literature.db")
    n = sum(len(p.get("results", [])) for p in papers)
    print(f"{len(papers)} papers, {len(catalog)} datasets, {n} results")


if __name__ == "__main__":
    main()
