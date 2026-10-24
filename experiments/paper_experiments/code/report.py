"""Phase 7: build REPORT.md and the paper's LaTeX tables from results/*.json only (rule P2).

    python code/report.py [--kind timing|smoke] [--paper DIR]

Accuracy statistics combine the timing run (seed 0) with the seed runs (seeds 1..n) of each
library's representative variant (launch.VARIANTS[lib][0]). Efficiency numbers come only from
the exclusive timing runs (rule P6).
"""
import argparse
import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
PAPER = Path("/work/vajira/DL2026/PAPER---torchtsetlin--Journal-of-Machine-Learning-Research/paper")

LIB_NAME = {"pytm": "pyTM", "pytmpar": "pyTM-Par", "pytmcuda": "pyTM-CUDA", "tmu": "tmu",
            "gt": "GT", "tt": "tt"}
ORDER = ["pytm", "pytmpar", "pytmcuda", "tmu", "gt", "tt"]
BENCH_TITLE = {
    "B1": "Noisy XOR", "B2": "2-D Noisy XOR (conv)", "B3": "Breast cancer",
    "B4": "MNIST flat", "B5": "MNIST flat weighted", "B6": "MNIST clause sweep",
    "B7": "MNIST conv", "B8": "Fashion-MNIST conv", "B9": "IMDb", "B10": "California housing",
}
# green-tsetlin paper Table I (seconds for 5 epochs incl. test after each), i9-12900K + RTX 3090
GT_TABLE_I = {
    "tmu-cpu1": [29.6, 33, 80, 128, 266], "tmu-cuda": [87.8, 62, 70, 75, 95],
    "pytm-cpu1": [4.5, 9, 47, 84, 197], "pytmpar-cpu4": [1.1, 2, 17, 47, 149],
    "pytmcuda-cuda": [14.8, 26, 30, 36, 88], "gt-cpu1": [0.3, 2, 18, 36, 90],
    "gt-cpu4": [4.0, 5, 11, 18, 31],
}
B6_C = [10, 100, 1000, 2000, 5000]
# Published reproduction targets (PLAN §3.1)
REPRO = {
    "R1": ("pyTM README, MNIST weighted, epoch 60", 0.9819, 8.80, 4.15),
    "R2": ("pyTM-Par README, MNIST weighted, epoch 100", 0.9814, 1.70, 1.08),
    "R3": ("pyTM-CUDA README, F-MNIST conv 8000 cl., epoch 30", 0.9094, 44.02, 6.12),
    "R4": ("tmu MNISTDemo (no published number)", None, None, None),
}


def load(kind):
    out = []
    for p in sorted((RES / kind).rglob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("status") == "done":
            r["_path"] = p
            r["_variant"] = f"{r['lib']}-{'cuda' if str(r['device']).startswith('cuda') else 'cpu' + str(r['threads'])}"
            out.append(r)
    return out


def fmt_ms(xs, scale=100.0, nd=2):
    xs = [x * scale for x in xs]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{st.mean(xs):.{nd}f} ± {st.stdev(xs):.{nd}f}"


def fmt_n(x, nd=2):
    if x is None:
        return "–"
    if x >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def pct(x):
    return "–" if x is None else f"{x * 100:.2f}"


def times(x):
    return "–" if x is None else f"{x:.1f}×"


def variant_label(v):
    lib, dev = v.split("-", 1)
    return f"{LIB_NAME[lib]} ({'GPU' if dev == 'cuda' else dev.replace('cpu', 'CPU×')})"


def ml_table(timing, seeds):
    """bench -> rows of (label, n, accuracy / f1 / precision / recall or rmse / mae)."""
    from launch import VARIANTS
    groups = defaultdict(list)
    for r in timing + seeds:
        if r["bench"] == "B6" or r["bench"].startswith("R"):
            continue
        rep = f"{r['lib']}-{'cuda' if VARIANTS[r['lib']][0][0] == 'cuda' else 'cpu' + str(VARIANTS[r['lib']][0][1])}"
        if r["_variant"] != rep:
            continue
        groups[(r["bench"], r["lib"], r["mode"])].append(r)
    table = defaultdict(list)
    for (bench, lib, mode), rs in sorted(groups.items(), key=lambda k: (int(k[0][0][1:]), ORDER.index(k[0][1]), k[0][2])):
        label = LIB_NAME[lib] + ("" if mode == "default" else f" [{mode}]")
        f = [r["final"] for r in rs]
        if rs[0]["config"]["task"] == "regression":
            row = dict(label=label, n=len(rs), rmse=fmt_ms([x["rmse"] for x in f], 1, 3),
                       mae=fmt_ms([x["mae"] for x in f], 1, 3))
        else:
            row = dict(label=label, n=len(rs), accuracy=fmt_ms([x["accuracy"] for x in f]),
                       f1=fmt_ms([x["f1"] for x in f]), precision=fmt_ms([x["precision"] for x in f]),
                       recall=fmt_ms([x["recall"] for x in f]),
                       best_logged=fmt_ms([r["best_epoch_logged_only"]["accuracy"] for r in rs]))
        table[bench].append(row)
    return table


def eff_table(timing):
    """bench -> rows of efficiency numbers from the seed-0 timing runs."""
    by = defaultdict(list)
    for r in timing:
        if r["bench"] == "B6" or r["bench"].startswith("R") or r["seed"] != 0:
            continue
        by[r["bench"]].append(r)
    table = defaultdict(list)
    for bench, rs in by.items():
        ref = [r for r in rs if r["_variant"] == "pytm-cpu1"]
        ref_t = ref[0]["timing"]["train_s_median_after1"] if ref else None
        for r in sorted(rs, key=lambda r: (ORDER.index(r["lib"]), r["mode"], r["_variant"])):
            t = r["timing"]
            tr = t["train_s_median_after1"] or t["train_s_epoch1"]
            m = r["memory"]
            table[bench].append(dict(
                label=variant_label(r["_variant"]) + ("" if r["mode"] == "default" else f" [{r['mode']}]"),
                train_s=tr, train_s_epoch1=t["train_s_epoch1"],
                test_s=t.get("test_s_median_after1") or t["test_s_median"],
                infer_eps=t["inference_examples_per_s"],
                speedup=(ref_t / tr) if ref_t and tr else None,
                state_kib=(r["model_size"].get("state_bytes") or 0) / 1024 or None,
                file_kib=(r["model_size"].get("saved_file_bytes") or 0) / 1024 or None,
                rss_mib=m["peak_rss_mib"],
                gpu_mib=m.get("peak_gpu_nvml_mib") or None,
            ))
    return table


def b6_table(timing):
    cells = defaultdict(lambda: defaultdict(list))
    accs = defaultdict(lambda: defaultdict(list))
    for r in timing:
        if r["bench"] != "B6":
            continue
        C = int(r["mode"].split("-")[0][1:])
        key = r["_variant"] + ("" if "-" not in r["mode"] else f" [{r['mode'].split('-')[1]}]")
        cells[key][C].append(r["protocol_total_s"])
        accs[key][C].append(r["final"]["accuracy"])
    return cells, accs


def repro_table(repro, timing):
    rows = []
    for r in repro:
        rid = r["bench"]
        desc, acc, tr, te = REPRO.get(rid, (rid, None, None, None))
        rows.append(dict(id=rid, variant=variant_label(r["_variant"]), desc=desc,
                         acc=r["final"]["accuracy"], acc_pub=acc,
                         train_s=r["timing"]["train_s_median_after1"], train_pub=tr,
                         test_s=r["timing"]["test_s_median"], test_pub=te,
                         passed=None if acc is None else abs(r["final"]["accuracy"] - acc) <= 0.01))
    for r in timing:       # R6: tt-seq on B1 vs Granmo (2018) Table 15, 99.3 ± 0.3
        if r["bench"] == "B1" and r["lib"] == "tt" and r["mode"] == "seq" and r["_variant"] == "tt-cuda":
            rows.append(dict(id="R6", variant="tt (GPU) [seq]", desc="Noisy XOR vs Granmo 2018 Tab. 15",
                             acc=r["final"]["accuracy"], acc_pub=0.993, train_s=None, train_pub=None,
                             test_s=None, test_pub=None,
                             passed=abs(r["final"]["accuracy"] - 0.993) <= 0.01))
    return rows


def write_report(kind, ml, eff, b6, b6acc, rep):
    L = [f"# paper_experiments: results ({kind} records)", "",
         f"Generated by `code/report.py` on {time.strftime('%Y-%m-%d %H:%M')} from `results/{kind}/` "
         "and `results/seeds/`. Do not edit by hand.", ""]
    L += ["## Reproduction checks", "",
          "| ID | Library | Check | Accuracy (ours / published) | Train s/epoch (ours / published) | Test s (ours / published) | Pass (±1 pp) |",
          "|---|---|---|---|---|---|---|"]
    for r in rep:
        L.append(f"| {r['id']} | {r['variant']} | {r['desc']} | {r['acc'] * 100:.2f} / "
                 f"{pct(r['acc_pub'])} | "
                 f"{fmt_n(r['train_s'])} / {fmt_n(r['train_pub'])} | {fmt_n(r['test_s'])} / {fmt_n(r['test_pub'])} | "
                 f"{'–' if r['passed'] is None else ('yes' if r['passed'] else 'NO')} |")
    L += ["", "## ML performance (final epoch, mean ± sd over runs, %)", ""]
    for bench in sorted(ml, key=lambda b: int(b[1:])):
        rows = ml[bench]
        L += [f"### {bench} {BENCH_TITLE.get(bench, '')}", ""]
        if "rmse" in rows[0]:
            L += ["| Library | runs | RMSE | MAE |", "|---|---|---|---|"]
            L += [f"| {r['label']} | {r['n']} | {r['rmse']} | {r['mae']} |" for r in rows]
        else:
            L += ["| Library | runs | Accuracy | F1 | Precision | Recall | Best epoch (logged only) |",
                  "|---|---|---|---|---|---|---|"]
            L += [f"| {r['label']} | {r['n']} | {r['accuracy']} | {r['f1']} | {r['precision']} | "
                  f"{r['recall']} | {r['best_logged']} |" for r in rows]
        L.append("")
    L += ["## Efficiency (seed-0 exclusive timing runs)", "",
          "Train = median seconds per epoch over epochs 2..E (epoch 1 separately); test = seconds to "
          "predict the test set; speed-up = pyTM (1 thread) train time / train time.", ""]
    for bench in sorted(eff, key=lambda b: int(b[1:])):
        L += [f"### {bench} {BENCH_TITLE.get(bench, '')}", "",
              "| Library | train s/epoch | epoch 1 s | test s | inference ex/s | speed-up | state KiB | file KiB | peak RSS MiB | peak GPU MiB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in eff[bench]:
            L.append(f"| {r['label']} | {fmt_n(r['train_s'], 3)} | {fmt_n(r['train_s_epoch1'], 3)} | "
                     f"{fmt_n(r['test_s'], 3)} | {fmt_n(r['infer_eps'])} | "
                     f"{times(r['speedup'])} | {fmt_n(r['state_kib'], 1)} | "
                     f"{fmt_n(r['file_kib'], 1)} | {fmt_n(r['rss_mib'], 0)} | {fmt_n(r['gpu_mib'], 0)} |")
        L.append("")
    if b6:
        L += ["## B6: MNIST clause sweep (green-tsetlin Table I protocol)", "",
              "Seconds for construction + 5 epochs with a test-set prediction after each, median of "
              "the repeats; in brackets the published value (i9-12900K + RTX 3090). Ours: Ryzen 9 3950X + RTX 3090.", "",
              "| Library | " + " | ".join(f"C={c}" for c in B6_C) + " | accuracy at C=5000 |",
              "|---|" + "---|" * (len(B6_C) + 1)]
        for key in sorted(b6, key=lambda k: (ORDER.index(k.split("-")[0]), k)):
            base = key.split(" ")[0]
            pub = GT_TABLE_I.get(base)
            cells = []
            for i, c in enumerate(B6_C):
                v = b6[key].get(c)
                s = fmt_n(st.median(v), 1) if v else "–"
                if pub:
                    s += f" ({pub[i]})"
                cells.append(s)
            a = b6acc[key].get(5000)
            L.append(f"| {variant_label(base)}{key[len(base):]} | " + " | ".join(cells) +
                     f" | {fmt_ms(a) if a else '–'} |")
        L.append("")
    (HERE / ("REPORT.md" if kind == "timing" else f"REPORT_{kind}.md")).write_text("\n".join(L) + "\n")


def tex_escape(s):
    return s.replace("×", r"$\times$").replace("±", r"$\pm$").replace("_", r"\_")


def write_paper_tables(ml, eff, b6):
    out = PAPER / "generated"
    out.mkdir(exist_ok=True)
    # ML table: accuracy (or RMSE) per benchmark x library
    lines = [r"% generated by experiments/paper_experiments/code/report.py; do not edit",
             r"\begin{tabular}{@{}l" + "l" * 1 + r"@{}}", r"\toprule", r"Benchmark & Library: accuracy (\%) / RMSE \\", r"\midrule"]
    for bench in sorted(ml, key=lambda b: int(b[1:])):
        cells = []
        for r in ml[bench]:
            v = r.get("accuracy") or r.get("rmse")
            cells.append(f"{tex_escape(r['label'])} {tex_escape(v)}")
        lines.append(f"{bench} & " + r"; ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "ml_table.tex").write_text("\n".join(lines) + "\n")
    lines = [r"% generated by experiments/paper_experiments/code/report.py; do not edit",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Bench & Library & train s/epoch & test s & speed-up & peak GPU MiB \\", r"\midrule"]
    for bench in sorted(eff, key=lambda b: int(b[1:])):
        for r in eff[bench]:
            sp = "--" if r["speedup"] is None else f"{r['speedup']:.1f}"
            lines.append(f"{bench} & {tex_escape(r['label'])} & {fmt_n(r['train_s'], 3)} & "
                         f"{fmt_n(r['test_s'], 3)} & {sp} & {fmt_n(r['gpu_mib'], 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "efficiency_table.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="timing")
    ap.add_argument("--no-paper", action="store_true")
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    timing = load(args.kind)
    seeds = load("seeds") if args.kind == "timing" else []
    repro = load("repro") if args.kind == "timing" else []
    ml = ml_table(timing, seeds)
    eff = eff_table(timing)
    b6, b6acc = b6_table(timing)
    rep = repro_table(repro, timing)
    write_report(args.kind, ml, eff, b6, b6acc, rep)
    if args.kind == "timing" and not args.no_paper:
        write_paper_tables(ml, eff, b6)
    print(f"{len(timing)} {args.kind} records, {len(seeds)} seed records, {len(repro)} repro records")


if __name__ == "__main__":
    main()
