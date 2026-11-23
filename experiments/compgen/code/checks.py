"""Phase-0 checks (plan §5):

  M0  ``data.generate`` gives the arrays the reference script's own data block gives (the block is
      executed from the pinned clone; skipped when the clone is absent)
  M1  the GPU metrics equal the reference script's formulas in plain NumPy, on trained models
  M2  with ``literals="negated"`` the benchmark model never includes a positive literal, in
      sequential and batched mode, and it does include negated ones
  M3  a run interrupted and resumed equals the uninterrupted run

Writes results/checks/checks.json. Usage: python code/checks.py [--device cuda:0]
"""

import argparse
import math  # noqa: F401 - used by the executed reference block
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import torch
from common import CACHE, HERE, REF_DIR, REF_SCRIPT, RESULTS, new_record, read_json, write_json
from data import count_inversions, generate, load
from metrics import evaluate, naive
from train import build

import torchtsetlin as tt


def m0(dev):
    script = REF_DIR / REF_SCRIPT
    if not script.exists():
        return True, {"skipped": f"{script} not found (run setup_ref.sh)"}
    src = script.read_text().splitlines()
    a = next(i for i, ln in enumerate(src) if ln.startswith("if args.generate_data:"))
    b = next(i for i, ln in enumerate(src) if i > a and "if args.negate_input:" in ln)
    block = textwrap.dedent("\n".join(src[a + 1:b]).expandtabs(4))
    out = {}
    for seed, n in ((0, 1500), (3, 800)):
        import itertools

        from scipy.stats import zipfian
        args = SimpleNamespace(sequence_length=5, number_of_non_informative_elements=0, number_of_examples=n,
                               number_of_synonyms=5, sequence_skewedness=0.0, skewed_test_data_only=False,
                               synonym_skewedness=0.0, noise=0.01, sequence_distribution_drift=False,
                               synonym_distribution_drift=False)
        ns = {"args": args, "np": np, "itertools": itertools, "math": math, "zipfian": zipfian,
              "count_inversions": count_inversions}
        np.random.seed(seed)
        exec(compile(block, str(script), "exec"), ns)                # the reference's own data block
        mine = generate(seed, number_of_examples=n)
        same = all(np.array_equal(ns[k], m) for k, m in zip(("X_train", "Y_train", "X_test", "Y_test"), mine))
        out[f"seed{seed}_n{n}"] = {"identical": bool(same), "ones_per_row": int(mine[0].sum(1).max())}
    return all(v["identical"] for v in out.values()), out


def _trained(arm, feedback, dev, x, y, epochs=3):
    tt.seed_everything(0)
    m = build(arm, 100, 10, feedback, dev)
    bs = 1000 if feedback == "seq" else int(feedback[1:])
    for _ in range(epochs):
        for i in range(0, x.shape[0], bs):
            m.update(x[i:i + bs], y[i:i + bs])
    return m


def _data(dev, n=6000):
    d, _ = load(0)
    return (torch.from_numpy(d["x_train"][:n]).bool().to(dev), torch.from_numpy(d["y_train"][:n]).to(dev),
            torch.from_numpy(d["x_test"][:n]).bool().to(dev), torch.from_numpy(d["y_test"][:n]).to(dev))


def m1(dev):
    xtr, ytr, xte, yte = _data(dev)
    out = {}
    for arm, fb in (("neg", "b32"), ("both", "b32"), ("neg", "seq")):
        m = _trained(arm, fb, dev, xtr, ytr)
        fast = evaluate(m, xte, yte, chunk=700)
        ref, arrays = naive(m, xte, yte)
        diff = max(float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())
                   for a, b in zip(fast["_arrays"], arrays))
        scal = max(abs(fast[k] - v) for k, v in ref.items() if k != "votes_hist")
        acc = 100.0 * int((m.eval()(xte).argmax(1) == yte).sum()) / xte.shape[0]
        out[f"{arm}_{fb}"] = {"max_array_diff": diff, "max_scalar_diff": scal, "acc_diff": abs(acc - fast["acc"]),
                              "hist_equal": fast["votes_hist"] == ref["votes_hist"], "coverage": fast["coverage"]}
    ok = all(v["max_array_diff"] < 1e-9 and v["max_scalar_diff"] < 1e-9 and v["acc_diff"] < 1e-9 and v["hist_equal"]
             for v in out.values())
    return ok, out


def m2(dev):
    xtr, ytr, _, _ = _data(dev)
    out = {}
    for fb in ("seq", "b8", "b512"):
        m = _trained("neg", fb, dev, xtr, ytr)
        inc, Fn = m.included_mask(), m.n_features
        out[fb] = {"positive_literals_included": int(inc[:, :Fn].sum()), "negated_literals_included": int(inc[:, Fn:].sum())}
    ok = all(v["positive_literals_included"] == 0 and v["negated_literals_included"] > 0 for v in out.values())
    return ok, out


def m3(dev):
    base = [sys.executable, "train.py", "--phase", "resume_check", "--arm", "neg", "--feedback", "b32", "--device", dev,
            "--subset", "4000", "--max-epochs", "4", "--ckpt-every", "1", "--no-wandb"]
    tag = "neg_b32_s0"
    for t in ("_full", "_resumed"):
        (RESULTS / "resume_check" / f"{tag}{t}.json").unlink(missing_ok=True)
        (CACHE / "ckpt" / "resume_check" / f"{tag}{t}.pt").unlink(missing_ok=True)
    quiet = dict(cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rc = [subprocess.call(base + ["--tag-suffix", "_full"], **quiet),
          subprocess.call(base + ["--tag-suffix", "_resumed", "--stop-after", "2"], **quiet),
          subprocess.call(base + ["--tag-suffix", "_resumed"], **quiet)]
    a, b = (read_json(RESULTS / "resume_check" / f"{tag}{t}.json") for t in ("_full", "_resumed"))

    def curve(r):
        return [[c["test_acc"], c["train_acc"], c["mean_clause_len"], c["coverage"], c["precision_mean"]] for c in r["curve"]]
    same = bool(a and b and curve(a) == curve(b))
    return rc == [0, 3, 0] and same, {"exit_codes": rc, "identical": same, "curve": a and curve(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        torch.cuda.set_device(torch.device(dev))
    rec = new_record("checks", device=dev if dev != "cpu" else None, checks={})
    allok = True
    for name, fn in (("M0", m0), ("M1", m1), ("M2", m2), ("M3", m3)):
        t0 = time.time()
        ok, detail = fn(dev)
        rec["checks"][name] = {"ok": bool(ok), "detail": detail}
        allok = allok and ok
        note = " (skipped: no reference clone)" if isinstance(detail, dict) and "skipped" in detail else ""
        print(f"{name}: {'PASS' if ok else 'FAIL'}{note} ({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            print("   ", detail, flush=True)
    rec["ok"] = allok
    write_json(rec, RESULTS / "checks" / "checks.json")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
