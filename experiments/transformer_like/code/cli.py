"""Command-line entry points (spec §14, D10).

    python -m experiments.transformer_like.code.cli prepare-data
    python -m experiments.transformer_like.code.cli smoke --device cuda
    python -m experiments.transformer_like.code.cli probe --device cuda
    python -m experiments.transformer_like.code.cli search [--gpus 0,1 --per-gpu 2]
    python -m experiments.transformer_like.code.cli freeze-experiment
    python -m experiments.transformer_like.code.cli benchmark --manifest results/frozen_experiment.json
    python -m experiments.transformer_like.code.cli timing
    python -m experiments.transformer_like.code.cli report
"""

from __future__ import annotations

import argparse
import json

from . import data as D
from .common import RESULTS, write_json


def smoke(device: str) -> None:
    """1,000/200 images, small even clause counts, 1 encoder epoch, 2 head epochs. Never loads test."""
    from .run import run_encoders, run_trial
    enc = run_encoders(17, device, D.SMOKE, tag=f"_smoke_{device.split(':')[0]}",
                       cfg={"M": 10, "T": 5, "epochs": 1})
    epath = enc["checkpoint"]
    out = {}
    for arm in ("B0", "B1", "B1s7", "N0", "A0", "A1", "A2", "A3"):
        spec = {"stage": f"smoke_{device.split(':')[0]}", "arm": arm, "seed": 17, "manifest": D.SMOKE,
                "head": {"M": 20, "T_ratio": 0.5, "s": 5.0}, "temperature": 1.0, "encoders": epath,
                "max_epochs": 2, "patience": 10, "interventions": arm == "N0"}
        r = run_trial(spec, device)
        if r["status"] != "completed":
            raise SystemExit(r["error"])
        out[arm] = r["valid"]["accuracy"]
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--manifest", default=str(RESULTS / "frozen_experiment.json"))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--stage", default=None)
    a = ap.parse_args()
    if a.cmd == "prepare-data":
        print(D.write_manifests())
    elif a.cmd == "smoke":
        smoke(a.device)
    elif a.cmd == "probe":
        from .search import probe
        probe(a.device)
    elif a.cmd == "search":
        from .search import run_search
        run_search([int(g) for g in a.gpus.split(",")], a.per_gpu)
    elif a.cmd == "freeze-experiment":
        from .search import freeze
        freeze()
    elif a.cmd == "benchmark":
        from .search import run_benchmark
        run_benchmark(a.manifest, [int(g) for g in a.gpus.split(",")], a.per_gpu, a.stage)
    elif a.cmd == "timing":
        from .evaluation import run_timing
        run_timing(a.device)
    elif a.cmd == "report":
        from .report import main as report_main
        report_main()
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
