"""One run of the reference library (arm ``ref``): cair/PyHierarchicalTsetlinMachineCUDA with the
command of issue #15, on the same data files and with the same record format as ``train.py``.

    --sequence-length 5 --s 25.0 --T 10 --noise 0.01 --number-of-examples 40000
    --number-of-state-bits 8 --q 1.0 --clause-weight-update-ratio 1.0 --literal-budget 100
    --number-of-or-alternatives 1 --number-of-clauses 200 --vanilla --number-of-synonyms 5
    --and-group-normalization --negate-input

The model is built exactly as the script's ``--vanilla`` branch. As in the script, the input is
negated, negated literals are not appended, and ``fit`` sees the examples in file order every epoch
(no shuffling). Metrics come from one ``transform`` per data set through ``metrics_np`` (the
script's formulas, vectorised; the script itself recomputes ``transform`` four times per epoch and
loops over examples in Python).

Runs in the reference environment (``setup_ref.sh``; no torch). A resumed run restores automata
and weights but not the library's random state.

Usage: .cache/ref/venv/bin/python code/run_ref.py --phase p1 --seed 0
"""

import argparse
import pickle
import signal
import sys
import time

import numpy as np
from common import (
    CACHE,
    CKPT_EVERY,
    MODEL,
    N_FEATURES,
    PHASES,
    PRICE_EPOCHS,
    REF_COMMIT,
    REF_DIR,
    RESULTS,
    STOP,
    finish,
    metrics_due,
    new_record,
    read_json,
    ref_line,
    should_stop,
    tag_of,
    write_json,
)
from data import load
from metrics_np import clause_metrics, summarise

sys.path.insert(0, str(REF_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=PHASES + ["resume_check"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=STOP["max_epochs"])
    ap.add_argument("--time-cap-h", type=float, default=STOP["time_cap_h"])
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    import PyHierarchicalTsetlinMachineCUDA.tm as tm  # imports pycuda: needs the GPU
    tag = tag_of("ref", "seq", args.seed) + args.tag_suffix
    d, digest = load(args.seed)
    n = args.subset or len(d["y_train"])
    # --negate-input; uint32 as the script's arrays
    x = {k: (1 - d[f"x_{k}"][:n].astype(np.uint32)) for k in ("train", "test")}
    y = {k: d[f"y_{k}"][:n].astype(np.uint32) for k in ("train", "test")}
    machine = tm.TsetlinMachine(
        MODEL["clauses"], MODEL["T"], MODEL["s"], q=1.0, clause_weight_update_ratio=1.0,
        literal_budget=MODEL["literal_budget"], weighted_clauses=False, binary_inference=False,
        constant_update_p=False, and_group_normalization=True, seed=args.seed,
        number_of_state_bits=MODEL["state_bits"], no_clipping=False, append_negated=False,
        hierarchy_structure=((tm.AND_GROUP, N_FEATURES), (tm.AND_GROUP, 1)))
    hp = dict(arm="ref", library="PyHierarchicalTsetlinMachineCUDA", library_commit=REF_COMMIT, model="TsetlinMachine (--vanilla)",
              clauses_total=MODEL["clauses"], T=MODEL["T"], s=MODEL["s"], state_bits=MODEL["state_bits"], weighted=False,
              literal_budget=MODEL["literal_budget"], literals="negated input, negated literals not appended",
              feedback="seq", batch_size=1, updates_per_epoch=n, seed=args.seed, shuffle=False,
              data=dict(sha256=digest, n_train=n, n_test=n, subset=args.subset,
                        stats=read_json(RESULTS / "data" / f"seed{args.seed}.json")["stats"]),
              stop=dict(STOP, max_epochs=args.max_epochs, time_cap_h=args.time_cap_h))
    print(f"[{tag}] reference library {REF_COMMIT}: {MODEL['clauses']} clauses, T={MODEL['T']}, s={MODEL['s']:g}, seed {args.seed}",
          flush=True)

    out = RESULTS / args.phase / f"{tag}.json"
    ckpt = CACHE / "ckpt" / args.phase / f"{tag}.pkl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rec = new_record("run", phase=args.phase, tag=tag, hp=hp, curve=[], status="running", train_s_total=0.0, elapsed_s=0.0)
    start = 0
    if ckpt.exists():
        with open(ckpt, "rb") as fh:
            st = pickle.load(fh)
        machine.load(st["state"])
        rec, start = st["rec"], st["epoch"]
        rec["curve"] = [c for c in rec["curve"] if c["epoch"] <= start]
        rec.setdefault("resumed_at", []).append({"epoch": start, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                 "note": "the library's random state is not restored"})
        print(f"[{tag}] resumed after epoch {start}", flush=True)
    elif (read_json(out) or {}).get("status") == "done":
        print(f"[{tag}] already done", flush=True)
        return
    rec["status"] = "running"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    lines = RESULTS / args.phase / f"{tag}.stats.txt"
    if start == 0:
        lines.parent.mkdir(parents=True, exist_ok=True)
        lines.write_text("")
    price = args.phase == "price"
    for epoch in range(start + 1, args.max_epochs + 1):
        t0 = time.time()
        machine.fit(x["train"], y["train"])
        train_s = time.time() - t0
        t0 = time.time()
        test_acc = 100.0 * float((machine.predict(x["test"]) == y["test"]).mean())
        train_acc = 100.0 * float((machine.predict(x["train"]) == y["train"]).mean())
        state = machine.save()
        weights = np.asarray(state["clause_weights"]).reshape(-1)[:MODEL["clauses"]]
        included = np.array([[machine.ta_action(c, 0, k) for k in range(N_FEATURES)] for c in range(MODEL["clauses"])]) \
            if epoch == 1 or epoch % 50 == 0 else None
        row = {"epoch": epoch, "test_acc": test_acc, "train_acc": train_acc, "updates": epoch * n}
        if metrics_due(epoch):
            co = machine.transform(x["test"].reshape(n, N_FEATURES, 1)) > 0
            row.update(summarise(*clause_metrics(co, y["test"], (weights >= 0).astype(np.int64))))
        eval_s = time.time() - t0
        rec["train_s_total"] += train_s
        rec["elapsed_s"] += train_s + eval_s
        row.update(train_s=train_s, eval_s=eval_s, train_s_total=rec["train_s_total"], elapsed_s=rec["elapsed_s"])
        if included is not None:
            row["mean_clause_len"] = float(included.sum(1).mean())
        if "coverage" in row:
            with open(lines, "a") as fh:
                fh.write(ref_line(args.seed, row) + "\n")
        rec["curve"].append(row)
        if epoch <= 5 or epoch % 10 == 0:
            print(f"[{tag}] epoch {epoch:4d} test {test_acc:6.2f} train {train_acc:6.2f} | "
                  + (f"coverage {row['coverage']:6.2f} precision p50 {row['precision_p50']:5.1f} mean "
                     f"{row['precision_mean']:5.1f} votes {row['votes_mean']:.1f} | " if "coverage" in row else "")
                  + f"{train_s:.2f}s + {eval_s:.2f}s", flush=True)
        reason = should_stop(rec["curve"], args.max_epochs, rec["elapsed_s"], args.time_cap_h)
        if price and epoch >= PRICE_EPOCHS:
            reason = f"pricing ({PRICE_EPOCHS} epochs)"
        if reason:
            finish(rec, reason)
            if price:
                steady = rec["curve"][1:] or rec["curve"]
                rec["s_per_epoch"] = sum(r["train_s"] + r["eval_s"] for r in steady) / len(steady)
                rec["train_s_per_epoch"] = sum(r["train_s"] for r in steady) / len(steady)
        write_json(rec, out)
        if reason or epoch % args.ckpt_every == 0 or epoch == args.stop_after:
            tmp = ckpt.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump({"state": state, "epoch": epoch, "rec": rec}, fh)
            tmp.replace(ckpt)
        if reason:
            ckpt.unlink(missing_ok=True)
            print(f"[{tag}] done after {epoch} epochs ({reason}): test {rec['test_acc_last']:.2f} %, epochs to "
                  + ", ".join(f"{k} %: {v['epoch'] if v else 'not reached'}" for k, v in rec["marks"].items()), flush=True)
            break
        if epoch == args.stop_after:
            print(f"[{tag}] --stop-after {epoch}: exiting with a checkpoint", flush=True)
            raise SystemExit(3)


if __name__ == "__main__":
    main()
