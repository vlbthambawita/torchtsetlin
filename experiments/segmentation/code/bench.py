"""Feasibility / latency / memory profile on an otherwise idle GPU (PLAN sections 6, 7).

    python code/bench.py profile --device cuda:1          # E0: B1/B2 TM + U-Net at 32..256 px
    python code/bench.py latency RUN_ID [RUN_ID ...]       # finalists: inference latency/throughput

Timed GPU work is bracketed by ``torch.cuda.synchronize``; warmup first; repeated measures.
Includes Boolean encoding (and the coarse stage for stacks). Results -> results/BENCH_*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines  # noqa: E402
import features  # noqa: E402
import record  # noqa: E402
import run as R  # noqa: E402


def _sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)


def timed(fn, dev, reps=20, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        _sync(dev)
        t = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t)
    return {"median_s": float(np.median(ts)), "p95_s": float(np.percentile(ts, 95)), "reps": reps}


def tm_model(cfg, K, dev, S, C_in):
    m = R.build_tm(cfg, K, dev, None)
    x = torch.randint(0, 256, (1, C_in, S, S), dtype=torch.uint8)
    m.update(features.encode(x.to(dev), cfg["spec"]), torch.zeros(1, S, S, dtype=torch.long, device=dev))
    return m


def profile(dev):
    torch.zeros(1, device=dev)  # initialise the CUDA context before resetting peak stats
    out = []
    K, C_in = 2, 3
    for S in (32, 64, 128, 256):
        for name, cfg in (
            ("B1", {**R.DEFAULTS, "family": "tm", "spec": {"bits": 4}, "patch": 3}),
            ("B2", {**R.DEFAULTS, "family": "tm", "spec": {"bits": 4, "pyr": {"scales": [1, 2, 4], "mode": "or", "patch": 3}}, "patch": 1}),
            ("B4_unet16", {**R.DEFAULTS, "family": "unet"}),
        ):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            x8 = torch.randint(0, 256, (8, C_in, S, S), dtype=torch.uint8)
            y8 = torch.randint(0, K, (8, S, S))
            rec = {"name": name, "size": S}
            try:
                if cfg["family"] == "tm":
                    m = tm_model(cfg, K, dev, S, C_in)
                    m.train()
                    rec["train_8img"] = timed(lambda: m.update(features.encode(x8.to(dev), cfg["spec"]), y8.to(dev)), dev, reps=3, warm=1)
                    m.eval()
                    rec["infer_1img"] = timed(lambda: m.vote_map(features.encode(x8[:1].to(dev), cfg["spec"])), dev)
                    rec["infer_8img"] = timed(lambda: m.vote_map(features.encode(x8.to(dev), cfg["spec"])), dev)
                    rec["n_features"] = int(m.n_features)
                    rec["automaton_states"] = int(m.ta_state.numel())
                else:
                    net = baselines.UNet(C_in, K, 16).to(dev)
                    opt = torch.optim.AdamW(net.parameters(), 1e-3)

                    def step():
                        loss = baselines.masked_loss(net(x8.to(dev).float() / 255), y8.to(dev), K)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    rec["train_8img"] = timed(step, dev, reps=5, warm=2)
                    net.eval()
                    with torch.no_grad():
                        rec["infer_1img"] = timed(lambda: net(x8[:1].to(dev).float() / 255), dev)
                        rec["infer_8img"] = timed(lambda: net(x8.to(dev).float() / 255), dev)
                    rec["params"] = sum(p.numel() for p in net.parameters())
                rec["status"] = "completed"
            except torch.cuda.OutOfMemoryError as e:
                rec["status"], rec["failure"] = "oom", str(e)[:300]
            rec["gpu_peak_alloc_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
            print(json.dumps(rec), flush=True)
            out.append(rec)
    record.write_record({"run_id": f"BENCH_profile_{time.strftime('%Y%m%d%H%M')}", "status": "completed",
                         "env": record.environment(dev), "rows": out})


def latency(run_ids, dev, reps=30):
    rows = []
    for rid in run_ids:
        with open(os.path.join(record.RESULTS, rid + ".json")) as f:
            r = json.load(f)
        cfg = r["config"]
        K = R.K_OF[cfg["dataset"]]
        va = R.load_data(cfg, "val")
        state = torch.load(os.path.join(record.HERE, r["checkpoint"]["path"]))
        if cfg["family"] in ("tm", "coalesced"):
            m = R.build_tm(cfg, K, dev, None)
            m._prepare(R.model_input(cfg, va["x"][:1], dev, va.get("extra", [None])[0:1] if va.get("extra") is not None else None))
            m.load_state_dict(state)
            m.eval()

            def f(b):
                return lambda: m.vote_map(R.model_input(cfg, va["x"][:b], dev))
        else:
            in_ch = R.model_input(cfg, va["x"][:1], "cpu").shape[1]
            net = (baselines.UNet(in_ch, K, cfg["width"]) if cfg["family"] == "unet" else baselines.LocalCNN(in_ch, K)).to(dev)
            net.load_state_dict(state)
            net.eval()

            def f(b):
                def g():
                    with torch.no_grad():
                        net(R.model_input(cfg, va["x"][:b], dev))
                return g
        torch.cuda.reset_peak_memory_stats(dev)
        row = {"run_id": rid, "b1": timed(f(1), dev, reps), "b16": timed(f(16), dev, reps)}
        S = cfg["size"]
        row["images_per_s_b16"] = 16 / row["b16"]["median_s"]
        row["pixels_per_s_b16"] = 16 * S * S / row["b16"]["median_s"]
        row["gpu_peak_alloc_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
        row["checkpoint_bytes"] = r["checkpoint"]["bytes"]
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.mode == "profile":
        profile(a.device)
    else:
        rows = latency(a.runs, a.device)
        record.write_record({"run_id": f"BENCH_latency_{time.strftime('%Y%m%d%H%M%S')}", "status": "completed",
                             "env": record.environment(a.device), "rows": rows})
