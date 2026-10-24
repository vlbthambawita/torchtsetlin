"""Launch cells in the right virtualenv with the right threads, CPU pinning and GPU.

    python code/launch.py smoke [--bench B1 ...] [--lib tt ...]   # Phase 2: 1 epoch, seed 0
    python code/launch.py timing [--bench ...] [--lib ...]        # Phase 4: seed 0, exclusive
    python code/launch.py seeds  [--bench ...] [--lib ...] [--jobs 6]   # Phase 5: other seeds
    python code/launch.py repro                                   # Phase 3: R1-R6
    python code/launch.py list                                    # print the cell matrix

Re-running skips cells whose record already has status "done". Timing runs run strictly one at
a time (rule P6); seed runs may run concurrently and their timings are never reported.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from configs import BENCH, LIBS, TT_CPU_TIMING_EPOCHS, all_cells, modes  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / ".cache"
RES = HERE / "results"
MAIN_PY = "/home/vajira/anaconda3/envs/pytorch2/bin/python"
CUDA_ROOT = "/usr/local/cuda-12.6"
SYS_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
TIMING_GPU = "0"          # RTX 3090 (nvidia-smi index 0)
N_CORES = 16              # physical cores 0-15; 16-31 are their SMT siblings

# (lib, device, threads) variants measured per benchmark. The first variant of each library is
# its accuracy representative for seeds 1..n (PLAN §6, Phase 5).
VARIANTS = {
    "pytm": [("cpu", 1)],
    "pytmpar": [("cpu", 16)],
    "pytmcuda": [("cuda", 1)],
    "tmu": [("cuda", 1), ("cpu", 1)],
    "gt": [("cpu", 16)],
    "tt": [("cuda", 1), ("cpu", 16)],
}
B6_EXTRA = {"pytmpar": [("cpu", 4)], "gt": [("cpu", 1), ("cpu", 4)], "tt": [("cpu", 1), ("cpu", 4)]}


def variants(bench, lib):
    v = list(VARIANTS[lib])
    if bench == "B6":
        v += B6_EXTRA.get(lib, [])
    return v


def python_for(lib):
    return MAIN_PY if lib == "tt" else str(CACHE / f"venv-{lib}" / "bin" / "python")


def env_for(lib, device, threads, gpu):
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu if device == "cuda" else ""
    if lib in ("pytmcuda", "tmu"):
        env["PATH"] = f"{CUDA_ROOT}/bin:" + env["PATH"]
        env["CUDA_ROOT"] = CUDA_ROOT
        env["LD_PRELOAD"] = SYS_LIBSTDCXX      # pycuda is built against the system libstdc++
    return env


def out_path(kind, bench, lib, mode, device, threads, seed):
    tag = f"{lib}-{device}{threads if device == 'cpu' else ''}"
    return RES / kind / bench / f"{bench}_{tag}_{mode}_s{seed}.json"


def is_done(p):
    try:
        return json.loads(p.read_text()).get("status") == "done"
    except Exception:
        return False


def run_cell(kind, bench, lib, mode, device, threads, seed, epochs=0, cores=None, gpu=TIMING_GPU,
             timing=False):
    out = out_path(kind, bench, lib, mode, device, threads, seed)
    if is_done(out):
        return out, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_for(lib), str(HERE / "code" / "run.py"), "--bench", bench, "--lib", lib,
           "--mode", mode, "--seed", str(seed), "--threads", str(threads),
           "--device", "cuda:0" if device == "cuda" else "cpu", "--out", str(out)]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    if timing:
        cmd += ["--timing"]
    if cores is None:
        cores = list(range(max(threads, 1)))
    cmd = ["taskset", "-c", ",".join(map(str, cores))] + cmd
    log = CACHE / "logs" / kind / (out.stem + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env_for(lib, device, threads, gpu), stdout=f,
                            stderr=subprocess.STDOUT).returncode
    status = "done" if is_done(out) else f"FAILED rc={rc} (see {log.relative_to(HERE)})"
    print(f"[{time.strftime('%H:%M:%S')}] {kind} {out.stem}: {status} ({time.time() - t0:.0f}s)",
          flush=True)
    return out, status


def selected(args):
    for bench, lib, mode in all_cells():
        if args.bench and bench not in args.bench:
            continue
        if args.lib and lib not in args.lib:
            continue
        if args.mode and not any(m in mode for m in args.mode):
            continue
        for device, threads in variants(bench, lib):
            if args.device and device not in args.device:
                continue
            # green-tsetlin splits the clause pool into n_jobs blocks and refuses an empty block
            # ("Cannot have a clause block size under 1"), so C < n_jobs is not a valid cell.
            if bench == "B6" and lib == "gt" and int(mode[1:].split("-")[0]) < threads:
                continue
            yield bench, lib, mode, device, threads


SMOKE_EPOCHS = {"B2": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["smoke", "timing", "seeds", "repro", "list"])
    ap.add_argument("--bench", nargs="*")
    ap.add_argument("--lib", nargs="*")
    ap.add_argument("--mode", nargs="*")
    ap.add_argument("--device", nargs="*")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=0)
    args = ap.parse_args()

    if args.what == "list":
        for c in selected(args):
            print(*c)
        return
    if args.what == "smoke":
        for bench, lib, mode, device, threads in selected(args):
            run_cell("smoke", bench, lib, mode, device, threads, 0,
                     epochs=args.epochs or SMOKE_EPOCHS.get(bench, 1))
        return
    if args.what == "timing":
        for bench, lib, mode, device, threads in selected(args):
            seeds = range(BENCH[bench]["seeds"]) if bench == "B6" else [0]   # B6: 3 repeats
            ep = args.epochs
            if lib == "tt" and device == "cpu" and bench != "B6":
                ep = ep or min(TT_CPU_TIMING_EPOCHS, BENCH[bench]["epochs"])
            for seed in seeds:
                run_cell("timing", bench, lib, mode, device, threads, seed, epochs=ep, timing=True)
        return
    if args.what == "repro":
        for r, lib in (("R1", "pytm"), ("R2", "pytmpar"), ("R3", "pytmcuda"), ("R4", "tmu")):
            if args.bench and r not in args.bench:
                continue
            device, threads = {"R2": ("cpu", 10), "R3": ("cuda", 1)}.get(r, ("cpu", 1))
            run_cell("repro", r, lib, "default", device, threads, 0, timing=True)
            if r == "R4":
                run_cell("repro", r, lib, "default", "cuda", 1, 0, timing=True)
        return
    if args.what == "seeds":
        # accuracy seeds 1..n with each library's representative variant, run concurrently.
        # CPU jobs get disjoint cores; GPU jobs alternate between the two GPUs.
        jobs = []
        for bench, lib, mode in all_cells():
            if bench == "B6" or (args.bench and bench not in args.bench) or \
                    (args.lib and lib not in args.lib):
                continue
            device, threads = VARIANTS[lib][0]
            for seed in range(1, BENCH[bench]["seeds"]):
                jobs.append((bench, lib, mode, device, threads, seed))
        free_cores = list(range(N_CORES))
        gpu_turn = [0]

        def work(j):
            bench, lib, mode, device, threads, seed = j
            n = 1 if device == "cuda" else threads
            while True:
                if len(free_cores) >= n:
                    cores = [free_cores.pop() for _ in range(n)]
                    break
                time.sleep(5)
            gpu = str(gpu_turn[0] % 2)
            gpu_turn[0] += 1
            try:
                return run_cell("seeds", bench, lib, mode, device, threads, seed, cores=cores, gpu=gpu)
            finally:
                free_cores.extend(cores)

        with ThreadPoolExecutor(args.jobs) as ex:
            list(ex.map(work, jobs))


if __name__ == "__main__":
    main()
