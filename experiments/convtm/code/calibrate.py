"""Throughput and peak-memory calibration (PLAN Section 8, P0.4).

Replaces every cost *estimate* in PLAN Section 10 with a measurement, and answers the one
scheduling question the programme cannot guess: **what is the largest arm that fits on the
10 GB card?**

Two modes:

``--mode grid``  full-scale training: ``--epochs`` (default 2) real epochs over the whole
                 45000-image training split, per (clauses x patch) configuration. Epoch 1 is
                 reported separately from the steady-state epochs because it carries CUDA
                 context creation, kernel autotuning and the first allocator growth.
``--mode probe`` a fixed number of mini-batches only, for configurations too large to spend
                 two epochs on. Gives peak memory exactly and s/epoch by extrapolation
                 (flagged as such in the record).

Both modes record ``torch.cuda.max_memory_allocated`` twice: ``peak_total_mb`` includes the
resident Boolean dataset (the number that decides whether an arm fits), and ``peak_model_mb``
is measured from a counter reset after the data is resident (the number that scales with the
arm). ``reserved`` is reported too, because the caching allocator, not ``allocated``, is what
actually OOMs.

An OOM is a result, not a crash: the cell is recorded with ``"oom": true`` and the sweep
continues.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import List, Optional

import arms as arms_mod
import data as data_mod
import record as rec_mod
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
MB = 1024.0 ** 2


def _sync() -> float:
    torch.cuda.synchronize()
    return time.time()


def _mem() -> dict:
    return {"alloc_mb": torch.cuda.max_memory_allocated() / MB,
            "reserved_mb": torch.cuda.max_memory_reserved() / MB}


def measure(n_clauses: int, patch: int, xtr, ytr, xva, yva, booleanization: str,
            device: str, batch_size: int, epochs: int, max_batches: Optional[int],
            base_mb: float, seed: int = 0, chunk_elements: Optional[int] = None) -> dict:
    """One calibration cell.

    ``chunk_elements`` overrides ``max_chunk_elements`` (LG-004). It is a pure
    throughput/memory dial: chunking partitions the mini-batch for the accumulation phase
    and the feedback is still committed once per batch, so the learning algorithm is
    unchanged -- only the random-number stream and the peak memory differ.
    """
    cfg = {**arms_mod.get("ctm-vanilla").defaults, "n_clauses": n_clauses, "patch": patch,
           "booleanization": booleanization, "batch_size": batch_size, "seed": seed,
           "device": device, "input_shape": tuple(xtr.shape[1:])}
    out = {"n_clauses": n_clauses, "patch": patch, "booleanization": booleanization,
           "batch_size": batch_size, "n_train": int(xtr.shape[0]), "oom": False,
           "extrapolated": max_batches is not None,
           "max_chunk_elements": int(chunk_elements or 2 ** 27)}
    torch.cuda.reset_peak_memory_stats()
    try:
        arm = cfg_build = arms_mod.get("ctm-vanilla").build(cfg)
        if chunk_elements:
            arm.model.max_chunk_elements = int(chunk_elements)
        out["chunk_size"] = int(arm.model._chunk_size(xtr[:batch_size]))
        out.update({k: v for k, v in arm.capacity().items()})
        out["T"] = cfg["_resolved"]["T"]
        n = xtr.shape[0]
        n_batches = (n + batch_size - 1) // batch_size
        use_batches = min(n_batches, max_batches) if max_batches else n_batches
        ep_times: List[float] = []
        for ep in range(epochs):
            t0 = _sync()
            if max_batches:
                arm.model.train()
                perm = torch.randperm(n, device=xtr.device)
                for i in range(use_batches):
                    idx = perm[i * batch_size : (i + 1) * batch_size]
                    arm.model.update(xtr[idx], ytr[idx])
            else:
                arm.fit_epoch(xtr, ytr, batch_size)
            ep_times.append(_sync() - t0)
        # inference throughput on the validation split (eval mode, the reported path)
        t0 = _sync()
        arm.predict(xva)
        infer_s = _sync() - t0
        m = _mem()
        out.update({
            "epoch_s_first": ep_times[0],
            "epoch_s": (sum(ep_times[1:]) / (len(ep_times) - 1)) if len(ep_times) > 1
                       else ep_times[0],
            "epoch_s_all": ep_times,
            "batches_measured": use_batches,
            "epoch_s_full_scale": (sum(ep_times[1:]) / (len(ep_times) - 1) if len(ep_times) > 1
                                   else ep_times[0]) * (n_batches / use_batches),
            "infer_img_per_s": float(xva.shape[0]) / max(infer_s, 1e-9),
            "peak_total_mb": m["alloc_mb"], "peak_reserved_mb": m["reserved_mb"],
            "peak_model_mb": m["alloc_mb"] - base_mb,
        })
        del arm, cfg_build
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        out["oom"] = True
        out["error"] = str(exc)[:200]
    finally:
        torch.cuda.empty_cache()
    return out


GRID_CLAUSES = [640, 2000, 4000, 8000]
GRID_PATCHES = [4, 8, 10]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grid", choices=("grid", "probe"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--bool", dest="booleanization", default="therm4")
    ap.add_argument("--clauses", type=int, nargs="+", default=None)
    ap.add_argument("--patches", type=int, nargs="+", default=None)
    ap.add_argument("--cells", default=None,
                    help="explicit 'clauses:patch[:bool]' cells, comma separated (probe mode)")
    ap.add_argument("--chunk-elements", type=int, nargs="+", default=[None],
                    help="max_chunk_elements values to sweep (LG-004); default: library default")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="probe mode: mini-batches per 'epoch' instead of the whole split")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory / MB
    print(f"# {gpu} ({total_mb:.0f} MB), torch {torch.__version__}, "
          f"batch {a.batch_size}, mode {a.mode}", flush=True)

    cells = []
    if a.cells:
        for c in a.cells.split(","):
            parts = c.split(":")
            cells.append((int(parts[0]), int(parts[1]),
                          parts[2] if len(parts) > 2 else a.booleanization))
    else:
        for nc in (a.clauses or GRID_CLAUSES):
            for p in (a.patches or GRID_PATCHES):
                cells.append((nc, p, a.booleanization))

    cells = [(nc, p, bl, ce) for (nc, p, bl) in cells for ce in a.chunk_elements]
    records, loaded = [], {}
    t_start = time.time()
    for k, (nc, p, bl, ce) in enumerate(cells, 1):
        if bl not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ds = data_mod.load_boolean(bl, device=dev, splits=("train", "val"))
            loaded[bl] = (ds["xtr"], ds["ytr"], ds["xva"], ds["yva"],
                          torch.cuda.max_memory_allocated() / MB)
        xtr, ytr, xva, yva, base_mb = loaded[bl]
        el = time.time() - t_start
        print(f"[{k}/{len(cells)}] {nc} clauses, {p}x{p}, {bl}"
              f"{'' if ce is None else f', chunk 2**{ce.bit_length()-1}'}  "
              f"(elapsed {el/60:.1f}m)", flush=True)
        r = measure(nc, p, xtr, ytr, xva, yva, bl, dev, a.batch_size, a.epochs,
                    a.max_batches, base_mb, chunk_elements=ce)
        r.update({"gpu": gpu, "gpu_total_mb": total_mb, "host": platform.node(),
                  "torch": torch.__version__, "git_sha": rec_mod.git_sha(),
                  "mode": a.mode, "epochs_measured": a.epochs})
        records.append(r)
        if r["oom"]:
            print("      OOM", flush=True)
        else:
            print(f"      {r['epoch_s_full_scale']:.1f} s/epoch  chunk {r['chunk_size']}/"
                  f"{a.batch_size}  "
                  f"peak {r['peak_total_mb']:.0f} MB alloc / {r['peak_reserved_mb']:.0f} MB "
                  f"reserved  infer {r['infer_img_per_s']:.0f} img/s  "
                  f"automata {r['n_automata']/1e6:.1f}M", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:                      # merged after every cell, so a
            json.dump(records, f, indent=1)              # crash costs only the running cell
    print(f"# done: {len(records)} cells in {(time.time()-t_start)/60:.1f} min -> {a.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
