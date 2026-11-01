"""Diagnostic (INVESTIGATION_B2.md, problem 2): B2 tt-cuda batched feedback at several batch
sizes, full protocol (5000 epochs, seeds 0-4). Records go to results/diag_b2_batch/, never into
results/seeds/. Usage: python code/diag_b2_batch.py [--jobs 4]"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "diag_b2_batch"
BATCHES = (16, 32, 50, 100, 200, 500)
SEEDS = range(5)

if __name__ == "__main__" and "--one" in sys.argv:  # child: run.py with the batch size patched
    i = sys.argv.index("--one")
    batch = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    sys.path.insert(0, str(Path(__file__).parent))
    import configs
    import run

    def cell_config(bench, lib, mode="default", _orig=configs.cell_config):
        cfg = _orig(bench, lib, mode)
        cfg["tt"]["batch_size"] = batch
        cfg["mode"] = f"batch{batch}"
        return cfg

    run.cell_config = cell_config
    run.main()
    sys.exit(0)


def one(job):
    batch, seed = job
    out = OUT / f"B2_tt-cuda_batch{batch}_s{seed}.json"
    if out.exists() and '"status": "done"' in out.read_text():
        return f"{out.stem}: skip"
    log = OUT / f"{out.stem}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, __file__, "--one", str(batch), "--bench", "B2",
                              "--lib", "tt", "--mode", "batch", "--seed", str(seed),
                              "--device", "cuda:0", "--out", str(out)], stdout=fh, stderr=subprocess.STDOUT)
    msg = f"[{time.strftime('%H:%M:%S')}] {out.stem}: {'done' if rc == 0 else f'FAILED rc={rc}'} ({time.time() - t0:.0f}s)"
    print(msg, flush=True)
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, s) for b in BATCHES for s in SEEDS]
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(one, jobs))
