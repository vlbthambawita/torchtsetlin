"""Propose schema-v2 values for the existing free-text fields (step 2 of plans/literature-db-v2.md).

Writes reviewable CSVs to benchmarks/literature/mappings/. Nothing here touches papers/*.json.
build_catalog.py reads the CSVs as a proposal layer under the records (see mappings/README.md
for the `review` column); a value in a paper record always wins over a proposal.

Re-running this script OVERWRITES the CSVs, review column included: regenerate only before
reviewing starts, or merge by hand. Each proposal row says which `rule` produced it, so a wrong
rule can be fixed here and the CSVs regenerated.

Run:  python benchmarks/literature/propose_mappings.py
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "mappings"
VOCAB = json.loads((HERE / "vocab.json").read_text())

# --------------------------------------------------------------------------- tasks
# Every task string in the records -> (task_family, setting_tags, note). Hand-written: 60 strings.
TASKS = {
    "classification": ("classification", "", ""),
    "binary classification": ("classification", "", ""),
    "text classification": ("classification", "", ""),
    "text categorization": ("classification", "", ""),
    "classification (text categorization)": ("classification", "", ""),
    "classification (text categorization, e.g. drug allergy detection)": ("classification", "", ""),
    "sentiment classification": ("classification", "", ""),
    "classification (sentiment)": ("classification", "", ""),
    "binary sentiment classification": ("classification", "", ""),
    "opinion polarity classification": ("classification", "", ""),
    "subjectivity classification": ("classification", "", ""),
    "question classification": ("classification", "", ""),
    "relation classification": ("classification", "", ""),
    "word sense disambiguation": ("classification", "", ""),
    "5-class review rating classification": ("classification", "", "ordinal labels? check paper"),
    "binary classification (pros vs cons)": ("classification", "", ""),
    "binary classification (vehicles vs animals)": ("classification", "", ""),
    "classification (benign vs malignant)": ("classification", "", ""),
    "classification (diabetic or not)": ("classification", "", ""),
    "classification (PD vs healthy)": ("classification", "", ""),
    "classification (human activity recognition)": ("classification", "", ""),
    "classification (winner prediction)": ("classification", "", ""),
    "classification (illustration of steady-state probabilities)": ("classification", "", "illustrative only"),
    "classification (regression target discretized into 5 bins)": ("classification", "",
                                                                    "discretized_from = regression target"),
    "sequence classification": ("classification", "", ""),
    "keyword spotting": ("classification", "", ""),
    "intrusion detection": ("classification", "", "security; check binary vs attack classes"),
    "malware detection": ("classification", "", "security"),
    "multi-label classification": ("classification", "", "label_type = multilabel"),
    "classification (federated)": ("classification", "federated", ""),
    "classification (vertically partitioned, decentralized)": ("classification", "decentralized", ""),
    "classification (counterfactual recourse)": ("classification", "counterfactual", ""),
    "classification (corruption robustness)": ("classification", "robustness", ""),
    "robustness evaluation (classification under corruption)": ("classification", "robustness", ""),
    "robustness evaluation (sentiment classification)": ("classification", "robustness", ""),
    "classification / robustness verification": ("classification", "robustness;verification", ""),
    "binary sentiment classification / robustness, equivalence, similarity verification":
        ("classification", "robustness;verification", ""),
    "uncertainty calibration (binary classification)": ("classification", "uncertainty", ""),
    "binary classification / uncertainty quantification": ("classification", "uncertainty", ""),
    "multiclass classification / uncertainty quantification": ("classification", "uncertainty", ""),
    "classification / uncertainty visualisation": ("classification", "uncertainty", ""),
    "regression": ("regression", "", "fill target / range from the paper"),
    "forecasting / sequence generation": ("regression", "",
                                          "subtype = forecasting; also generates sequences (task_families += generation)"),
    "novelty detection": ("anomaly_novelty_detection", "", ""),
    "document clustering": ("clustering", "", "embeddings evaluated by clustering"),
    "clustering/visualization": ("clustering", "interpretability", "qualitative only"),
    "self-supervised clustering": ("clustering", "", ""),
    "autoencoding/embedding": ("representation_learning", "", "TM autoencoder word embeddings"),
    "word similarity": ("representation_learning", "", "intrinsic embedding evaluation"),
    "word similarity (intrinsic)": ("representation_learning", "", "intrinsic embedding evaluation"),
    "word categorization (intrinsic)": ("representation_learning", "", "intrinsic embedding evaluation"),
    "structure learning": ("structure_learning", "", "Bayesian network DAG"),
    "top-n recommendation": ("recommendation_ranking", "", ""),
    "recommendation (customer-product-category rating classification)": ("recommendation_ranking", "",
                                                                          "rating as classification"),
    "rating prediction / recommendation (classification)": ("recommendation_ranking", "", "rating as classification"),
    "sequence understanding / action coreference": ("classification", "", "predict object at position x"),
    "segmentation (binary sky detection)": ("segmentation", "", "n_classes = 2"),
}

# --------------------------------------------------------------------------- data types
def modality_of(dt):
    d = dt.lower()
    synthetic = "synthetic" in d or "sampled from bn" in d
    for key, mod in [("time-series", "time_series"), ("sensor", "time_series"), ("audio", "audio"),
                     ("image", "image"), ("text", "text"), ("graph / tabular", "graph"), ("tabular/graph", "graph"),
                     ("sequence/graph", "graph"), ("graph", "graph"), ("sequence", "sequence"),
                     ("molecular", "tabular"), ("board game", "tabular"), ("tabular", "tabular"),
                     ("synthetic-boolean", "tabular"), ("voice", "tabular")]:
        if key in d:
            return mod, synthetic
    return "", synthetic

# --------------------------------------------------------------------------- metrics
CANON = [  # (pattern on lower-cased metric, canonical, kind, averaging)
    (r"^(macro[- ]f1)", "f1", "quality", "macro"),
    (r"^(macro[- ]precision)", "precision", "quality", "macro"),
    (r"^(macro[- ]recall)", "recall", "quality", "macro"),
    (r"^weighted[- ]f1", "f1", "quality", "weighted"),
    (r"^(edge )?f1\b", "f1", "quality", ""),
    (r"^(edge )?precision\b", "precision", "quality", ""),
    (r"^(edge )?recall\b", "recall", "quality", ""),
    (r"^spearman", "spearman_rho", "quality", ""),
    (r"^kendall", "kendall_tau", "quality", ""),
    (r"^balanced accuracy", "balanced_accuracy", "quality", ""),
    (r"^pixel accuracy", "pixel_accuracy", "quality", ""),
    (r"probe top-1", "probe_top1_accuracy", "quality", ""),
    (r"probe top-5", "probe_top5_accuracy", "quality", ""),
    (r"selective accuracy", "selective_accuracy", "quality", ""),
    (r"test accuracy reduction", "accuracy_drop", "quality", ""),
    (r"accuracy", "accuracy", "quality", ""),
    (r"^auc$|^auroc", "auroc", "quality", ""),
    (r"^auprc", "auprc", "quality", ""),
    (r"^iou", "iou", "quality", ""),
    (r"^mae$", "mae", "quality", ""),
    (r"^rmse$", "rmse", "quality", ""),
    (r"forecast error", "forecast_error", "quality", ""),
    (r"^ece$", "ece", "quality", ""),
    (r"^nmi$", "nmi", "quality", ""),
    (r"^ari$", "ari", "quality", ""),
    (r"^purity", "purity", "quality", ""),
    (r"cosine similarity", "cosine_similarity", "quality", ""),
    (r"^hr@", "hit_rate_at_k", "quality", ""),
    (r"^ndcg@", "ndcg_at_k", "quality", ""),
    (r"itemcoverage", "item_coverage", "quality", ""),
    (r"^sensitivity", "recall", "quality", ""),
    (r"^type i error", "type_i_error", "quality", ""),
    (r"^type ii error", "type_ii_error", "quality", ""),
    (r"structural hamming", "shd", "quality", ""),
    (r"graph-kernel similarity", "graph_kernel_similarity", "quality", ""),
    (r"epsilon-robust", "epsilon_robust", "quality", ""),
    (r"epsilon-similar", "epsilon_similar", "quality", ""),
    (r"counterfactual confidence", "counterfactual_confidence", "quality", ""),
    (r"counterfactual robustness", "counterfactual_robustness", "quality", ""),
    (r"l1 distance", "l1_distance", "quality", ""),
    (r"l2 distance", "l2_distance", "quality", ""),
    (r"fraction of datasets", "fraction_of_datasets", "quality", ""),
    (r"interpretability", "human_verified_clauses", "interpretability", ""),
    (r"number of ci tests", "ci_tests", "count", ""),
    (r"number of clusters", "n_clusters", "count", ""),
    (r"communication cost", "communication_cost", "cost_communication", ""),
    (r"energy reduction", "energy_reduction", "cost_energy", ""),
    (r"energy|pulse", "energy", "cost_energy", ""),
    (r"speed-?up", "speedup", "cost_time", ""),
    (r"reduction", "time_reduction", "cost_time", ""),
    (r"power", "power", "cost_energy", ""),
    (r"efficiency", "efficiency", "cost_energy", ""),
    (r"throughput|operations per", "throughput", "cost_time", ""),
    (r"latency|delay|t_v->s", "latency", "cost_time", ""),
    (r"time", "time", "cost_time", ""),
    (r"memory|model size|parameters|automata per class", "memory", "cost_memory", ""),
    (r"area|luts?\b|ffs\b|dsps?\b|bram|registers|buffers|cpu usage", "hardware_resource", "cost_hardware", ""),
]
STAT = [(r"\bfinal\b", "final_epoch"), (r"average of last (\d+) epochs", "average_last_epochs"), (r"best test accuracy over|max test accuracy",
         "peak_over_epochs"), (r"median peak", "median"), (r"mean over|average over", "mean")]

def canon_metric(m):
    lm = m.lower()
    canonical = kind = averaging = statistic = ""
    for pat, c, k, a in CANON:
        if re.search(pat, lm):
            canonical, kind, averaging = c, k, a
            break
    qualifier = ""
    q = re.search(r"\(([^)]*)\)", m)
    if q:
        qualifier = q.group(1)
        if kind == "quality" and canonical in {"f1", "precision", "recall"} and not averaging \
                and not re.search(r"final|epoch", qualifier):
            averaging = "per_class"
    if lm.startswith("edge "):
        qualifier = (qualifier + "; " if qualifier else "") + "edge-level"
    for pat, s in STAT:
        if re.search(pat, lm):
            statistic = s
    if "peak" in lm and not statistic:
        statistic = "peak_over_epochs"
    return canonical, qualifier, averaging, statistic, kind

# --------------------------------------------------------------------------- booleanization
BOOL = [
    (r"already (binary|boolean)|natively (binary|boolean)|native binary|inherently (binary|boolean)", "native_binary"),
    (r"adaptive (gaussian )?threshold", "adaptive_threshold"),
    (r"otsu", "otsu"),
    (r"quantile|percentile", "quantile_thermometer"),
    (r"thermometer", "thermometer"),
    (r"gray[- ]code", "gray_code"),
    (r"one-?hot", "one_hot"),
    (r"fingerprint|ecfp", "fingerprint"),
    (r"hypervector|hyperdimensional|\bhvs?\b", "hypervector"),
    (r"pixel\s*[<>]=?\s*\d", "fixed_threshold"),
    (r"glove|word2vec|embedding", "embedding_augmented"),
    (r"n-?gram|bigram", "n_gram_presence"),
    (r"bag[- ]of[- ]words|\bbow\b|presence|vocabulary|tokens?\b|words?\b", "bag_of_words_presence"),
    (r"threshold", "fixed_threshold"),
]

def bool_method(b):
    if not b:
        return "", ""
    lb = b.lower()
    if re.search(r"not (stated|specified|reported|described|detailed)", lb) and not re.search(
            r"thermometer|threshold|one-?hot|bag|quantile", lb):
        return "not_reported", ""
    for pat, m in BOOL:
        if re.search(pat, lb):
            bits = re.search(r"(\d+)\s*(uniform |quantile |equal-width )?(bins|bits|levels|thresholds)", lb)
            return m, bits.group(1) if bits else ""
    return "other", ""

# --------------------------------------------------------------------------- config
NUM = r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)"
CFG = {
    "n_clauses": [rf"\b{NUM}\s*(?:clauses|clause)\b", rf"\bclauses?\s*[=:]\s*{NUM}", rf"\bC\s*=\s*{NUM}"],
    "T": [rf"\bT\s*=\s*{NUM}", rf"\bthreshold\s*(?:T\s*)?[=:]\s*{NUM}"],
    "s": [rf"\b[sS]\s*=\s*{NUM}", rf"\bspecificity\s*(?:s\s*)?[=:]\s*{NUM}"],
    "epochs": [rf"\b{NUM}\s*epochs?\b", rf"\bepochs?\s*[=:]\s*{NUM}"],
    "max_included_literals": [rf"literal budget\s*(?:of\s*)?{NUM}", rf"max_included_literals\s*=\s*{NUM}",
                              rf"\bL\s*=\s*{NUM}"],
    "state_bits": [r"\b(\d+)[- ]?state[- ]bits?\b", r"state[_ ]bits\s*=\s*(\d+)", r"\b(\d+)[- ]bit (?:TA|automat)"],
    "n_states": [r"\b(\d+)\s*(?:TA )?states\b", r"\bN\s*=\s*(\d+)\s*states", r"number_of_states\s*=\s*(\d+)"],
    "n_runs": [r"(?:mean|average|median) (?:of|over) (\d+) (?:runs|seeds|trials|repetitions)",
               r"\b(\d+) (?:independent )?(?:runs|seeds|trials|repetitions)\b"],
    "patch_size": [r"\b(\d+\s*[x×]\s*\d+)\s*(?:patch|window|filter|kernel)", r"(?:patch|window)\s*(?:size\s*)?[=:]?\s*(\d+\s*[x×]\s*\d+)"],
}
IMPL = [(r"pycuda|tmu.*cuda|pytsetlinmachinecuda", "PyTsetlinMachineCUDA"), (r"\btmu\b", "tmu"),
        (r"pytsetlinmachineparallel", "pyTsetlinMachineParallel"), (r"pytsetlinmachine", "pyTsetlinMachine"),
        (r"green[ _]tsetlin", "green_tsetlin"), (r"\bfpga\b", "fpga"), (r"\basic\b|65 ?nm|28 ?nm", "asic")]

def parse_config(c):
    out = {k: "" for k in CFG}
    out["statistic"] = ""
    if not c:
        return out, "", "", ""
    for key, pats in CFG.items():
        for p in pats:
            m = re.search(p, c, flags=0 if key == "T" else re.I)
            if m:
                out[key] = m.group(1).replace(",", "")
                break
    lc = c.lower()
    scope = "per_class" if re.search(r"per[- ]class|clauses/class|each class", lc) else \
        "total" if re.search(r"\btotal\b", lc) else ""
    weighted = "true" if re.search(r"\bweighted\b", lc) else ""
    impl = next((v for p, v in IMPL if re.search(p, lc)), "")
    stat = "mean" if re.search(r"\b(mean|average) (of|over) \d+", lc) else \
        "median" if "median" in lc else \
        "final_epoch" if re.search(r"final epoch|last epoch", lc) else \
        "peak_over_epochs" if re.search(r"\b(peak|best|max(imum)?) (test )?(accuracy|epoch)", lc) else ""
    out["statistic"] = stat
    return out, scope, weighted, impl

# --------------------------------------------------------------------------- main
def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main():
    OUT.mkdir(exist_ok=True)
    papers = [json.loads(p.read_text()) for p in sorted((HERE / "papers").glob("*.json"))]

    task_n, task_ds = Counter(), defaultdict(set)
    dt_n = Counter()
    usages = []
    for p in papers:
        for d in p["datasets"]:
            task_n[d["task"]] += 1
            task_ds[d["task"]].add(d["dataset_id"])
            dt_n[d["data_type"]] += 1
            usages.append((p, d))

    unknown = [t for t in task_n if t not in TASKS]
    if unknown:
        raise SystemExit(f"task strings without a proposal, add them to TASKS: {unknown}")
    write("tasks.csv", ["task", "n_usages", "datasets", "proposed_task_family", "proposed_setting_tags", "note",
                        "review"],
          [[t, n, ", ".join(sorted(task_ds[t]))[:200], *TASKS[t], ""] for t, n in task_n.most_common()])

    write("data_types.csv", ["data_type", "n_usages", "proposed_modality", "proposed_synthetic", "review"],
          [[dt, n, *modality_of(dt), ""] for dt, n in dt_n.most_common()])

    # Per usage: only what depends on the row (label type, Booleanization). Task family and
    # modality come from tasks.csv / data_types.csv, so they are reviewed once per string.
    rows = []
    for p in papers:
        for i, d in enumerate(p["datasets"]):
            fam = TASKS[d["task"]][0]
            nc = d.get("n_classes")
            label = ""
            if fam in ("classification", "segmentation", "anomaly_novelty_detection"):
                label = "multilabel" if "multi-label" in d["task"] else "binary" if nc == 2 else \
                    "multiclass" if nc and nc > 2 else ""
            method, bits = bool_method(d.get("booleanization"))
            rows.append([p["paper_id"], i, d["dataset_id"], d["task"], nc if nc is not None else "", label,
                         d.get("booleanization") or "", method, bits, ""])
    write("usages.csv", ["paper_id", "usage_index", "dataset_id", "task", "n_classes", "proposed_label_type",
                         "booleanization", "proposed_booleanization_method", "proposed_bits_or_bins", "review"], rows)

    m_n, m_units = Counter(), defaultdict(set)
    rows = []
    for p in papers:
        for i, r in enumerate(p["results"]):
            m_n[r["metric"]] += 1
            m_units[r["metric"]].add(str(r["unit"]))
            cfg, scope, weighted, impl = parse_config(r.get("config"))
            rows.append([p["paper_id"], i, r["model"], int(bool(r["is_tm"])), r.get("config") or "",
                         *[cfg[k] for k in CFG], cfg["statistic"], scope, weighted, impl, ""])
    write("configs.csv", ["paper_id", "result_index", "model", "is_tm", "config", *[f"proposed_{k}" for k in CFG],
                          "proposed_statistic", "proposed_clauses_scope", "proposed_weighted", "proposed_implementation", "review"], rows)

    rows = []
    for m, n in sorted(m_n.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append([m, " | ".join(sorted(m_units[m])), n, *canon_metric(m), ""])
    write("metrics.csv", ["metric", "units", "n_results", "proposed_canonical", "proposed_qualifier",
                          "proposed_averaging", "proposed_statistic", "proposed_kind", "review"], rows)

    # Vocabulary check of the proposals themselves.
    bad = [m for m, *_ in rows if canon_metric(m)[4] and canon_metric(m)[4] not in VOCAB["metric_kind"]]
    assert not bad, bad


if __name__ == "__main__":
    main()
