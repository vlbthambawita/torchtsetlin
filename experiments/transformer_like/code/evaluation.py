"""Complete-pipeline timing (spec §16 timing protocol). Run serially, nothing else on the GPU.

batch-1: warm up on 100 validation images, then time the first 1,000 test ids one at a time
(median, p95). Throughput: warm up on 5 validation batches of 256, then 5 full passes over the
10k test images in batches of 256 (including the short last batch); median images/s.
Raw uint8 in, digits out: preprocessing, Q/K, attention, quantisation and head all included;
no feature or Q/K caches.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from . import data as D
from .common import ART, FROZEN, RESULTS, provenance, read_json, sync, write_json
from .models import TMQKAttention


def time_model(model: TMQKAttention, val: np.ndarray, test: np.ndarray, device: str) -> Dict:
    torch.set_grad_enabled(False)
    for i in range(100):
        model.predict_images(torch.as_tensor(val[i:i + 1]).to(device))
    sync(device)
    lat = []
    for i in range(1000):
        x = torch.as_tensor(test[i:i + 1])
        t0 = time.perf_counter()
        model.predict_images(x.to(device)).cpu()
        sync(device)
        lat.append(time.perf_counter() - t0)
    for i in range(5):
        model.predict_images(torch.as_tensor(val[i * 256:(i + 1) * 256]).to(device)).cpu()
    sync(device)
    rates = []
    for _ in range(5):
        n = 0
        t0 = time.perf_counter()
        for i in range(0, len(test), 256):
            n += len(model.predict_images(torch.as_tensor(test[i:i + 256]).to(device)).cpu())
        sync(device)
        rates.append(n / (time.perf_counter() - t0))
    lat = np.asarray(lat) * 1e3
    return {"batch1_median_ms": float(np.median(lat)), "batch1_p95_ms": float(np.percentile(lat, 95)),
            "batch256_images_per_s": float(np.median(rates)), "batch256_passes": rates}


def run_timing(device: str = "cuda", cpu_seed: int = 23, cpu_threads: int = 8) -> None:
    fr = read_json(FROZEN)
    va = D.load_split(D.MAIN, "valid")
    te = D.load_split(D.MAIN, "test", unseal=True, purpose="timing (images only, labels unused)")
    recs = sorted((RESULTS / "final").glob("*.json"))
    for p in recs:
        r = read_json(p)
        if r.get("status") != "completed":
            continue
        targets = [device]
        if r["spec"]["seed"] == cpu_seed:
            targets.append("cpu")
        for dev in targets:
            out = RESULTS / "timing" / f"{r['run_id']}__{dev}.json"
            if out.exists():
                continue
            if dev == "cpu":
                torch.set_num_threads(cpu_threads)
            model = TMQKAttention.load(ART.parent / r["checkpoint"], dev)
            t = time_model(model, va.images, te.images, dev)
            rec = {"run_id": r["run_id"], "arm": r["arm"], "seed": r["spec"]["seed"], "device": dev,
                   "threads": torch.get_num_threads(), **t, **provenance(dev)}
            write_json(out, rec)
            print(f"{r['run_id']:45s} {dev:5s} b1={t['batch1_median_ms']:.2f}ms "
                  f"thr={t['batch256_images_per_s']:.0f}/s", flush=True)
            del model
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
