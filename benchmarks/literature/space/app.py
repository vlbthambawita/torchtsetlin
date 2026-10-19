"""Tsetlin-machine literature explorer (Hugging Face Space).

Serves ``literature.db``, the SQLite build of ``benchmarks/literature/papers/*.json`` in the
torchtsetlin repository (``python benchmarks/literature/build_catalog.py``). All filtering,
grouping and "best value" ranking run as SQL on the server, so the browser only receives the
rows it shows.
"""

import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

import gradio as gr
import pandas as pd

try:  # the Space runs on ZeroGPU hardware, which requires one @spaces.GPU function at startup
    import spaces
except ImportError:  # local runs
    spaces = None

DB = Path(__file__).with_name("literature.db")
REPO = "https://github.com/vlbthambawita/torchtsetlin"
MAX_ROWS = 1500        # rows sent to the browser; the CSV download always has all of them
SQL_MAX_ROWS = 5000
SQL_TIMEOUT_S = 3.0
BEST_STYLE = "background-color: #ffd54f; color: #111; font-weight: 700"

# Label -> SQL expression. Only these fixed strings are ever interpolated into a query.
GROUPS = {
    "Dataset": "dataset",
    "Paper": "paper",
    "Year": "year",
    "Paper category": "category",
    "Data type": "data_type",
    "Model": "model",
    "Nothing (best across the whole selection)": "''",
}
SPLIT_KEY = "COALESCE(NULLIF(split, ''), 'test')"  # papers that state no split are ranked with test
RANK = "CASE better WHEN 'lower' THEN value ELSE -value END"
BASE = f"""
SELECT r.id, COALESCE(r.dataset_id, '(no dataset)') AS dataset, r.metric, r.value, r.unit,
       r.std_or_range, r.split, {SPLIT_KEY} AS split_key, r.better, r.model, r.is_tm,
       r.is_proposed, r.config, r.source_location, r.notes, r.paper_id, p.title AS paper,
       p.year, p.category, p.url AS paper_url, p.caution, d.data_type
FROM results r JOIN papers p ON p.paper_id = r.paper_id
LEFT JOIN datasets d ON d.dataset_id = r.dataset_id
"""


def connect():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, check_same_thread=False)
    return con


def query(sql, params=()):
    with closing(connect()) as con:
        return pd.read_sql_query(sql, con, params=list(params))


def choices(sql):
    return [(f"{v} ({n})", v) for v, n in query(sql).itertuples(index=False)]


def where(datasets, metrics, dtypes, splits, categories, tm_only, proposed_only, hide_flagged,
          search):
    conds, params = [], []
    for col, values in (("dataset", datasets), ("metric", metrics), ("data_type", dtypes),
                        ("split_key", splits), ("category", categories)):
        if values:
            conds.append(f"{col} IN ({','.join('?' * len(values))})")
            params += list(values)
    if tm_only:
        conds.append("is_tm = 1")
    if proposed_only:
        conds.append("is_proposed = 1")
    if hide_flagged:
        conds.append("caution IS NULL")
    if search and search.strip():
        conds.append("(model || ' ' || paper || ' ' || COALESCE(config, '') || ' ' "
                     "|| COALESCE(notes, '') || ' ' || COALESCE(source_location, '')) LIKE ?")
        params.append(f"%{search.strip()}%")
    return (" WHERE " + " AND ".join(conds)) if conds else "", params


def ranked(group_label, filters):
    """Filtered rows plus `is_best`: best value per (group, metric, unit, split)."""
    group = GROUPS[group_label]
    cond, params = where(*filters)
    sql = f"""
    WITH f AS (SELECT * FROM ({BASE}) {cond}),
    k AS (
      SELECT f.*, COUNT(value) OVER w AS n_cmp,
             CASE better WHEN 'higher' THEN MAX(value) OVER w
                         WHEN 'lower' THEN MIN(value) OVER w END AS best_v,
             COUNT(*) OVER (PARTITION BY {group}) AS group_size,
             COUNT(*) OVER (PARTITION BY {group}, metric) AS metric_size
      FROM f WINDOW w AS (PARTITION BY {group}, metric, unit, split_key)
    )
    SELECT *, (better IS NOT NULL AND n_cmp > 1 AND value = best_v) AS is_best FROM k
    ORDER BY group_size DESC, {group}, metric_size DESC, metric, unit, split_key, {RANK}
    """
    return query(sql, params)


def arrow(better):
    return {"higher": "↑ higher", "lower": "↓ lower"}.get(better, "")


def link(url, text="link"):
    return f"[{text}]({url})" if url else ""


def to_csv(df, name):
    path = Path(tempfile.mkdtemp()) / name
    df.to_csv(path, index=False)
    return str(path)


def results_view(group_label, best_only, *filters):
    df = ranked(group_label, filters)
    n_best = int(df["is_best"].sum())
    if best_only:
        df = df[df["is_best"] == 1]
    out = pd.DataFrame({
        "★": df["is_best"].map({1: "★", 0: ""}),
        "Dataset": df["dataset"],
        "Metric": df["metric"],
        "Value": df["value"],
        "Unit": df["unit"],
        "± / range": df["std_or_range"].fillna(""),
        "Split": df["split"],
        "Better": df["better"].map(arrow),
        "Model": df["model"],
        "Proposed": df["is_proposed"].map({1: "yes", 0: ""}),
        "TM": df["is_tm"].map({1: "TM", 0: "non-TM"}),
        "Paper": df["paper"],
        "Year": df["year"],
        "Link": df["paper_url"].map(link),
        "⚠": df["caution"].fillna(""),
        "Source": df["source_location"].fillna(""),
        "Config": df["config"].fillna(""),
        "Notes": df["notes"].fillna(""),
    })
    csv = to_csv(out.drop(columns=["Link"]).assign(Paper_id=df["paper_id"].values),
                 "tm_literature_results.csv")
    shown = out.head(MAX_ROWS)
    styled = shown.style.apply(
        lambda row: [BEST_STYLE if row["★"] else ""] * len(row), axis=1
    ).format({"Value": lambda v: "" if pd.isna(v) else f"{v:,.6g}"})
    status = (f"**{len(out):,}** rows match · **{n_best:,}** best values (★) for grouping "
              f"*{group_label}*")
    if len(out) > MAX_ROWS:
        status += f" · showing the first {MAX_ROWS:,}; the CSV has all of them"
    return styled, status, csv


def metric_choices(datasets):
    cond, params = ("WHERE r.dataset_id IN (" + ",".join("?" * len(datasets)) + ")", datasets) \
        if datasets else ("", [])
    df = query(f"SELECT metric, COUNT(*) n FROM results r {cond} GROUP BY metric "
               "ORDER BY n DESC, metric", params)
    return gr.update(choices=[(f"{m} ({n})", m) for m, n in df.itertuples(index=False)])


def leaderboard(datasets, dtypes, hide_flagged, tm_only_best, min_n):
    cond, params = where(datasets, None, dtypes, None, None, False, False, hide_flagged, "")
    cond = (cond + " AND " if cond else " WHERE ") + "better IS NOT NULL AND value IS NOT NULL"
    key = "dataset, metric, unit, split_key"
    sql = f"""
    WITH f AS (SELECT * FROM ({BASE}) {cond}),
    r AS (
      SELECT f.*, ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY {RANK}) AS rk,
             ROW_NUMBER() OVER (PARTITION BY {key}, is_tm ORDER BY {RANK}) AS rk_cls,
             COUNT(*) OVER (PARTITION BY {key}) AS n
      FROM f
    )
    SELECT a.dataset, a.data_type, a.metric, a.unit, a.split_key, a.better, a.n,
           a.value, a.model, a.is_tm, a.paper, a.year, a.paper_url,
           t.value AS tm_value, t.model AS tm_model, t.paper AS tm_paper, t.year AS tm_year
    FROM r a LEFT JOIN r t ON t.dataset = a.dataset AND t.metric = a.metric
         AND t.unit = a.unit AND t.split_key = a.split_key AND t.is_tm = 1 AND t.rk_cls = 1
    WHERE a.rk = 1 AND a.n >= ?
    ORDER BY a.n DESC, a.dataset, a.metric
    """
    df = query(sql, params + [int(min_n or 1)])
    if tm_only_best:
        df = df.assign(value=df["tm_value"], model=df["tm_model"], paper=df["tm_paper"],
                       year=df["tm_year"], is_tm=1)
    out = pd.DataFrame({
        "Dataset": df["dataset"], "Type": df["data_type"].fillna(""), "Metric": df["metric"],
        "Unit": df["unit"], "Split": df["split_key"], "Better": df["better"].map(arrow),
        "Results": df["n"], "Best value": df["value"], "Best model": df["model"],
        "TM": df["is_tm"].map({1: "TM", 0: "non-TM"}), "Paper": df["paper"], "Year": df["year"],
        "Best TM value": df["tm_value"], "Best TM model": df["tm_model"].fillna(""),
        "TM paper": df["tm_paper"].fillna(""),
    })
    fmt = lambda v: "" if pd.isna(v) else f"{v:,.6g}"  # noqa: E731
    styled = out.style.format({"Best value": fmt, "Best TM value": fmt}).apply(
        lambda col: [BEST_STYLE] * len(col), subset=["Best value"])
    csv = to_csv(out, "tm_literature_best_per_dataset.csv")
    return styled, f"**{len(out):,}** dataset × metric × unit × split combinations", csv


def datasets_view(search, dtypes):
    conds, params = [], []
    if dtypes:
        conds.append(f"d.data_type IN ({','.join('?' * len(dtypes))})")
        params += list(dtypes)
    if search and search.strip():
        conds.append("(d.dataset_id || ' ' || COALESCE(d.tasks,'') || ' ' || "
                     "COALESCE(d.names_in_papers,'') || ' ' || COALESCE(d.tm_variants,'')) LIKE ?")
        params.append(f"%{search.strip()}%")
    cond = (" WHERE " + " AND ".join(conds)) if conds else ""
    df = query(f"""
      SELECT d.*, (SELECT COUNT(*) FROM results r WHERE r.dataset_id = d.dataset_id) AS n_results,
             (SELECT GROUP_CONCAT(s, ' | ') FROM (SELECT DISTINCT
                 COALESCE(u.train_size, '–') || ' / ' || COALESCE(u.val_size, '–') || ' / ' ||
                 COALESCE(u.test_size, '–') AS s FROM dataset_usages u
               WHERE u.dataset_id = d.dataset_id AND
                 COALESCE(u.train_size, u.val_size, u.test_size) IS NOT NULL)) AS sizes
      FROM datasets d {cond} ORDER BY d.n_papers DESC, n_results DESC, d.dataset_id""", params)
    return pd.DataFrame({
        "Dataset": df["dataset_id"], "Type": df["data_type"].fillna(""),
        "Tasks": df["tasks"].fillna(""), "Classes": df["n_classes"],
        "Input": df["input_shape_or_features"].fillna(""),
        "Train / val / test (as reported)": df["sizes"].fillna(""), "Papers": df["n_papers"],
        "Results": df["n_results"], "TM variants": df["tm_variants"].fillna(""),
        "Download": [link(u, "download") if u else "not public / n.a."
                     for u in df["download_url"]],
        "Names in papers": df["names_in_papers"].fillna(""),
    })


def usages_view(dataset):
    if not dataset:
        return pd.DataFrame()
    df = query("""
      SELECT p.title, p.year, p.tm_variants, u.name_in_paper, u.task, u.train_size, u.val_size,
             u.test_size, u.split_as_reported, u.booleanization, u.notes
      FROM dataset_usages u JOIN papers p ON p.paper_id = u.paper_id
      WHERE u.dataset_id = ? ORDER BY p.year""", [dataset])
    return df.rename(columns={
        "title": "Paper", "year": "Year", "tm_variants": "TM variants",
        "name_in_paper": "Name in paper", "task": "Task", "train_size": "Train",
        "val_size": "Val", "test_size": "Test", "split_as_reported": "Split as reported",
        "booleanization": "Booleanization", "notes": "Notes"})


def papers_view(search, categories):
    conds, params = [], []
    if categories:
        conds.append(f"category IN ({','.join('?' * len(categories))})")
        params += list(categories)
    if search and search.strip():
        conds.append("(title || ' ' || COALESCE(authors,'') || ' ' || COALESCE(tm_variants,'') "
                     "|| ' ' || COALESCE(new_feature,'') || ' ' || COALESCE(venue,'')) LIKE ?")
        params.append(f"%{search.strip()}%")
    cond = (" WHERE " + " AND ".join(conds)) if conds else ""
    df = query(f"""SELECT p.*, (SELECT COUNT(*) FROM results r WHERE r.paper_id = p.paper_id) n,
                   (SELECT GROUP_CONCAT(dataset_id, ', ') FROM dataset_usages u
                    WHERE u.paper_id = p.paper_id) ds
                   FROM papers p {cond} ORDER BY year, title""", params)
    return pd.DataFrame({
        "Year": df["year"], "Title": df["title"], "Venue": df["venue"].fillna(""),
        "Category": df["category"].fillna(""), "Feature type": df["feature_type"].fillna(""),
        "TM variants": df["tm_variants"].fillna(""), "New feature": df["new_feature"].fillna(""),
        "Datasets": df["ds"].fillna(""), "Results": df["n"], "⚠ Caution": df["caution"].fillna(""),
        "Paper": df["url"].map(link), "Code": df["code_url"].map(lambda u: link(u, "code")),
    })


def _deny_writes(action, *_):
    # The connection is already read-only; this also blocks ATTACH (other files) and PRAGMA.
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_PRAGMA):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def run_sql(sql):
    if not sql or not sql.strip():
        return pd.DataFrame(), "Enter a query."
    start = time.monotonic()
    try:
        with closing(connect()) as con:
            con.set_authorizer(_deny_writes)
            con.set_progress_handler(lambda: int(time.monotonic() - start > SQL_TIMEOUT_S), 10000)
            cur = con.execute(sql)
            rows = cur.fetchmany(SQL_MAX_ROWS + 1)
            cols = [c[0] for c in cur.description or []]
    except sqlite3.Error as err:
        return pd.DataFrame(), f"**Error:** `{err}`"
    more = len(rows) > SQL_MAX_ROWS
    df = pd.DataFrame(rows[:SQL_MAX_ROWS], columns=cols)
    msg = f"{len(df):,} rows in {time.monotonic() - start:.2f} s"
    return df, msg + (f" (truncated at {SQL_MAX_ROWS:,})" if more else "")


SQL_EXAMPLE = """-- Best reported test accuracy per TM paper on MNIST
SELECT p.year, p.title, r.model, MAX(r.value) AS accuracy
FROM results r JOIN papers p USING (paper_id)
WHERE r.dataset_id = 'mnist' AND r.metric = 'accuracy' AND r.is_tm = 1
  AND COALESCE(NULLIF(r.split, ''), 'test') = 'test'
GROUP BY r.paper_id
ORDER BY accuracy DESC;"""

ABOUT = f"""
## About this data

Every dataset and every reported number from **55 Tsetlin-machine papers**: the proposed models
*and* the baselines each paper compared against. Numbers are **as reported by the authors, not
reproduced**. Each was extracted with its source table or figure, and checked against the text or
the rendered PDF page.

**Ranking.** ★ marks the best value among rows that share the chosen grouping **and** the same
metric, unit and split. A group needs at least two such rows. Rows whose paper states no split
(mostly hardware measurements) are ranked with test rows. "Better" is highest for accuracy, F1,
AUC, IoU, correlations, speed-ups, throughput and efficiency, and lowest for errors (MAE, RMSE,
ECE), time, latency, energy, power, area, memory and model size. A few count-type metrics, such as
the number of clusters, are never ranked.

**Before comparing across papers.** Papers differ in splits, Booleanization, clause budgets and
epochs, and some report the *peak* test accuracy over training. The `Config`, `Source` and
`Notes` columns say how each number was obtained. Papers marked **⚠** have a specific
reliability problem. For example, the Pre-Sorted TM's MNIST/Kuzushiji-MNIST accuracies are likely
optimistic. Results from other implementations, or measured before torchtsetlin 0.3.0 made the
clause budget a hard cap, are not directly comparable to torchtsetlin runs.

**Source.** [`benchmarks/literature/`]({REPO}/tree/main/benchmarks/literature) in the torchtsetlin
repository: one JSON record per paper is the source of truth. `build_catalog.py` builds this
SQLite file (`literature.db`, downloadable from this Space's *Files* tab).

### Schema
- `papers(paper_id, title, authors, year, venue, arxiv_id, doi, url, code_url, category,
  feature_type, tm_variants, new_feature, caution, extraction_notes, pdf)`
- `datasets(dataset_id, data_type, tasks, n_classes, input_shape_or_features, download_url,
  download_url_source, n_papers, tm_variants, names_in_papers)`
- `dataset_usages(paper_id, dataset_id, name_in_paper, task, train_size, val_size, test_size,
  split_as_reported, booleanization, notes)`
- `results(id, paper_id, dataset_id, model, is_tm, is_proposed, metric, value, unit,
  std_or_range, split, better, config, source_location, notes)`
"""

RESULT_WIDTHS = ["40px", "150px", "170px", "90px", "60px", "90px", "60px", "85px", "300px",
                 "80px", "70px", "320px", "60px", "60px", "200px", "130px", "260px", "300px"]
BEST_WIDTHS = ["150px", "100px", "170px", "60px", "60px", "85px", "70px", "90px", "300px",
               "70px", "300px", "60px", "100px", "300px", "300px"]
DATASETS_SQL = "SELECT dataset_id, n_papers FROM datasets ORDER BY n_papers DESC, dataset_id"


def build():
    ds_choices = choices("SELECT COALESCE(dataset_id, '(no dataset)'), COUNT(*) n FROM results "
                         "GROUP BY 1 ORDER BY n DESC, 1")
    metric_all = choices("SELECT metric, COUNT(*) n FROM results GROUP BY metric "
                         "ORDER BY n DESC, metric")
    dtypes = choices("SELECT data_type, COUNT(*) FROM datasets WHERE data_type IS NOT NULL "
                     "GROUP BY 1 ORDER BY 2 DESC")
    splits = choices(f"SELECT {SPLIT_KEY}, COUNT(*) FROM results GROUP BY 1 ORDER BY 2 DESC")
    cats = choices("SELECT category, COUNT(*) FROM papers GROUP BY 1 ORDER BY 2 DESC")
    n = query("SELECT (SELECT COUNT(*) FROM papers) p, (SELECT COUNT(*) FROM datasets) d, "
              "(SELECT COUNT(*) FROM results) r").iloc[0]

    with gr.Blocks(title="Tsetlin Machine literature explorer") as demo:
        gr.Markdown(
            f"# Tsetlin Machine literature explorer\n"
            f"**{n.p}** papers · **{n.d}** datasets · **{n.r:,}** reported results, extracted "
            f"from the TM literature for [torchtsetlin]({REPO}). Numbers are as reported by the "
            f"authors. Read *About* before comparing across papers.")
        with gr.Tab("Results"):
            with gr.Row():
                f_ds = gr.Dropdown(ds_choices, multiselect=True, label="Dataset", scale=2)
                f_metric = gr.Dropdown(metric_all, multiselect=True, label="Metric", scale=2)
                f_group = gr.Dropdown(list(GROUPS), value="Dataset", label="Rank best within",
                                      scale=2)
            with gr.Row():
                f_dtype = gr.Dropdown(dtypes, multiselect=True, label="Data type")
                f_split = gr.Dropdown(splits, multiselect=True, label="Split")
                f_cat = gr.Dropdown(cats, multiselect=True, label="Paper category")
                f_search = gr.Textbox(label="Search model / paper / config / notes (Enter)",
                                      placeholder="e.g. convolutional, drop clause, 8000 clauses")
            with gr.Row():
                f_tm = gr.Checkbox(label="TM models only")
                f_prop = gr.Checkbox(label="Proposed models only")
                f_flag = gr.Checkbox(label="Hide ⚠ flagged papers")
                f_best = gr.Checkbox(label="★ Best per group only")
                dl = gr.DownloadButton("Download CSV", size="sm")
            status = gr.Markdown()
            table = gr.Dataframe(interactive=False, wrap=False, max_height=650,
                                 datatype=["str"] * 13 + ["markdown"] + ["str"] * 4,
                                 show_search="search", column_widths=RESULT_WIDTHS)
        with gr.Tab("Best per dataset"):
            gr.Markdown("One row per dataset × metric × unit × split: the best reported value, "
                        "and the best value from a Tsetlin-machine model.")
            with gr.Row():
                l_ds = gr.Dropdown(ds_choices, multiselect=True, label="Dataset", scale=3)
                l_dtype = gr.Dropdown(dtypes, multiselect=True, label="Data type", scale=2)
                l_min = gr.Number(value=2, minimum=1, precision=0,
                                  label="Min. results per row", scale=1)
            with gr.Row():
                l_flag = gr.Checkbox(label="Hide ⚠ flagged papers", value=True)
                l_tm = gr.Checkbox(label="Rank TM models only")
                l_dl = gr.DownloadButton("Download CSV", size="sm")
            l_status = gr.Markdown()
            l_table = gr.Dataframe(interactive=False, wrap=False, max_height=650,
                                   show_search="search", column_widths=BEST_WIDTHS)
        with gr.Tab("Datasets"):
            with gr.Row():
                d_search = gr.Textbox(label="Search (Enter)", placeholder="name, task, TM variant",
                                      scale=3)
                d_dtype = gr.Dropdown(dtypes, multiselect=True, label="Data type", scale=2)
            d_table = gr.Dataframe(interactive=False, wrap=True, max_height=520,
                                   datatype=["str"] * 9 + ["markdown", "str"])
            d_pick = gr.Dropdown(choices(DATASETS_SQL), label="How each paper used this dataset")
            d_usage = gr.Dataframe(interactive=False, wrap=True)
        with gr.Tab("Papers"):
            with gr.Row():
                p_search = gr.Textbox(label="Search (Enter)", placeholder="title, author, TM variant, "
                                      "feature", scale=3)
                p_cat = gr.Dropdown(cats, multiselect=True, label="Category", scale=2)
            p_table = gr.Dataframe(interactive=False, wrap=True, max_height=700,
                                   datatype=["str"] * 10 + ["markdown", "markdown"])
        with gr.Tab("SQL"):
            gr.Markdown(f"Read-only SQL over the database (schema under *About*). Queries stop "
                        f"after {SQL_TIMEOUT_S:.0f} s and at {SQL_MAX_ROWS:,} rows.")
            s_sql = gr.Code(SQL_EXAMPLE, language="sql", lines=8)
            s_run = gr.Button("Run", variant="primary")
            s_msg = gr.Markdown()
            s_out = gr.Dataframe(interactive=False, wrap=False, max_height=600)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)

        r_inputs = [f_group, f_best, f_ds, f_metric, f_dtype, f_split, f_cat, f_tm, f_prop,
                    f_flag, f_search]
        for comp in r_inputs:
            event = comp.submit if isinstance(comp, gr.Textbox) else comp.change
            event(results_view, r_inputs, [table, status, dl], show_progress="minimal")
        f_ds.change(metric_choices, f_ds, f_metric)
        l_inputs = [l_ds, l_dtype, l_flag, l_tm, l_min]
        for comp in l_inputs:
            comp.change(leaderboard, l_inputs, [l_table, l_status, l_dl], show_progress="minimal")
        for comp in (d_search, d_dtype):
            (comp.submit if isinstance(comp, gr.Textbox) else comp.change)(
                datasets_view, [d_search, d_dtype], d_table)
        d_pick.change(usages_view, d_pick, d_usage)
        for comp in (p_search, p_cat):
            (comp.submit if isinstance(comp, gr.Textbox) else comp.change)(
                papers_view, [p_search, p_cat], p_table)
        s_run.click(run_sql, s_sql, [s_out, s_msg])

        demo.load(results_view, r_inputs, [table, status, dl])
        demo.load(leaderboard, l_inputs, [l_table, l_status, l_dl])
        demo.load(datasets_view, [d_search, d_dtype], d_table)
        demo.load(papers_view, [p_search, p_cat], p_table)
    return demo


if spaces is not None:
    @spaces.GPU
    def _zerogpu_placeholder():
        """Never called: all queries are SQLite on the CPU, so no GPU is ever allocated."""


demo = build()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="amber"))
