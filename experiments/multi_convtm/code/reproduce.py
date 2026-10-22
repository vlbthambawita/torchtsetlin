"""One-command reproduction of the multi_convtm programme (see ../reproduce.sh).

Stages, in order (``--stages`` picks a subset):
  data          CIFAR-10 (official tarball, md5-checked) and the authors' released class sums
  reference     Table 2 recomputed from the authors' class sums (instant)
  tmu-env       isolated tmu 0.8.3 + PyCUDA environment in .cache/venv-tmu (built once)
  torchtsetlin  the four specialists with torchtsetlin, for every --variants entry and seed
  tmu           the four specialists with the authors' own scripts on tmu
  report        composites, REPORT.md and PROPOSAL.md from results/*.json

Jobs run on every visible GPU (or --gpus). A worker takes the next job its card can hold (HoG
needs about 20 GiB), running at most one torchtsetlin job or two tmu jobs at a time per GPU.
Finished records are skipped, so re-running the same command resumes. ``--quick`` runs 1 epoch
of seed 0 into .cache/quick/, so it can never overwrite the real results.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROG = HERE.parent
REPO = PROG.parents[1]
DATA = REPO / "data"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
REF_REPO = "cair/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines"
REF_COMMIT = "91524f06b7185f9ff81e0bb66c053daf55de5cde"
REF_FILES = [f"{n}_{e}_{rest}" for e in (9, 99) for n, rest in [
    ("CIFAR10AdaptiveThresholding", "2000_500_10.0_10_32_1.txt"),
    ("CIFAR10ColorThermometers", "2000_1500_2.5_3_8_32_1.txt"),
    ("CIFAR10ColorThermometers", "2000_1500_2.5_4_8_32_1.txt"),
    ("CIFAR10HistogramOfGradients", "2000_50_10.0_0_32_0.txt")]]
SPECS = ["HoG", "C4", "C3", "T10"]                   # longest first
VARIANTS = {"hard": [], "soft": ["--budget-mode", "soft"]}
VARIANT_NAME = {"hard": "", "soft": "softbudget"}
NEEDS_GIB = {"HoG": 20.0}
STAGES = ["data", "reference", "tmu-env", "torchtsetlin", "tmu", "report"]


def log(msg):
    print(f"[reproduce {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_data():
    DATA.mkdir(exist_ok=True)
    tgz = DATA / "cifar-10-python.tar.gz"
    if not (DATA / "cifar-10-batches-py" / "test_batch").exists():
        if not tgz.exists():
            log(f"downloading {CIFAR_URL}")
            urllib.request.urlretrieve(CIFAR_URL, tgz)
        if md5(tgz) != CIFAR_MD5:
            sys.exit(f"{tgz}: md5 mismatch, delete it and re-run")
        with tarfile.open(tgz) as tf:
            tf.extractall(DATA, filter="data") if sys.version_info >= (3, 12) else tf.extractall(DATA)
    log("CIFAR-10 ok")
    ref = DATA / "tmcomposites_reference"
    ref.mkdir(exist_ok=True)
    for name in REF_FILES:
        if not (ref / name).exists():
            url = f"https://raw.githubusercontent.com/{REF_REPO}/{REF_COMMIT}/class_sums/{name}"
            log(f"downloading {name}")
            urllib.request.urlretrieve(url, ref / name)
    (ref / "SOURCE").write_text(f"https://github.com/{REF_REPO} (class_sums/, MIT licence)\n")
    (ref / "SOURCE_COMMIT").write_text(REF_COMMIT + "\n")
    log("authors' class sums ok")


def cuda_root():
    if os.environ.get("CUDA_ROOT"):
        return os.environ["CUDA_ROOT"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parents[1])
    for cand in sorted(Path("/usr/local").glob("cuda-12*"), reverse=True):
        if (cand / "bin" / "nvcc").exists():
            return str(cand)
    sys.exit("no CUDA toolkit (nvcc) found; set CUDA_ROOT")


def stage_tmu_env():
    venv = PROG / ".cache" / "venv-tmu"
    py = venv / "bin" / "python"
    if not py.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    root = cuda_root()
    env = dict(os.environ, CUDA_ROOT=root, PATH=f"{root}/bin:" + os.environ["PATH"])
    pins = ["tmu==0.8.3", "numpy==1.26.4", "opencv-python-headless==4.10.0.84", "pycuda"]
    subprocess.check_call([str(py), "-m", "pip", "install", "-q", *pins], env=env)
    log(f"tmu environment ok ({venv}, CUDA {root})")


def gpu_list(arg):
    import torch
    ids = [int(g) for g in arg.split(",")] if arg else list(range(torch.cuda.device_count()))
    if not ids:
        sys.exit("no GPU visible")
    return {g: torch.cuda.get_device_properties(g).total_memory / 2**30 for g in ids}


def record_done(results, tag):
    p = results / f"{tag}.json"
    return p.exists() and json.loads(p.read_text()).get("status") == "done"


def run_jobs(jobs, gpus, results, logs):
    """jobs: dicts with tag, kind ('ttl'|'tmu'), spec, cmd(gpu) -> argv."""
    lock = threading.Lock()
    pending = [j for j in jobs if not record_done(results, j["tag"])]
    log(f"{len(pending)} job(s) to run, {len(jobs) - len(pending)} already done")
    running = {g: [] for g in gpus}

    def take(g):
        with lock:
            kinds = [j["kind"] for j in running[g]]
            for j in pending:
                if NEEDS_GIB.get(j["spec"], 0) > gpus[g]:
                    continue
                if (j["kind"] == "ttl" and kinds) or (j["kind"] == "tmu" and ("ttl" in kinds or len(kinds) >= 2)):
                    continue
                pending.remove(j)
                running[g].append(j)
                return j
        return None

    def worker(g):
        threads = []
        while True:
            j = take(g)
            if j is None:
                with lock:
                    idle = not pending or all(NEEDS_GIB.get(p["spec"], 0) > gpus[g] for p in pending)
                if idle and not running[g]:
                    return
                time.sleep(5)
                continue

            def go(j=j):
                log(f"GPU {g}: start {j['tag']}")
                with open(logs / f"{j['tag']}.log", "a") as fh:
                    rc = subprocess.call(j["cmd"](g), cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
                log(f"GPU {g}: {j['tag']} exit {rc}")
                with lock:
                    running[g].remove(j)
            t = threading.Thread(target=go, daemon=True)
            t.start()
            threads.append(t)

    ws = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    left = [j["tag"] for j in jobs if not record_done(results, j["tag"])]
    if left:
        log(f"not finished: {left}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list of {STAGES}")
    ap.add_argument("--variants", default="hard,soft", help="torchtsetlin variants: hard, soft")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tmu-runs", type=int, default=1)
    ap.add_argument("--gpus", default="", help="e.g. 0,1 (default: all visible)")
    ap.add_argument("--quick", action="store_true", help="1 epoch, seed 0, into .cache/quick/")
    args = ap.parse_args()
    stages = args.stages.split(",")
    bad = set(stages) - set(STAGES)
    if bad:
        sys.exit(f"unknown stage(s) {bad}")
    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    epochs = ["--epochs", "1"] if args.quick else []
    if args.quick:
        os.environ["MC_RESULTS"] = str(PROG / ".cache" / "quick" / "results")
        os.environ["MC_CACHE"] = str(PROG / ".cache" / "quick")
        os.environ["MC_EPOCHS"] = "1"
    results = Path(os.environ.get("MC_RESULTS", PROG / "results"))
    logs = Path(os.environ.get("MC_CACHE", PROG / ".cache")) / "logs" if args.quick else PROG / "logs"
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    if "data" in stages:
        stage_data()
    if "reference" in stages or "report" in stages:
        subprocess.check_call([sys.executable, "composite.py"], cwd=HERE)
    if "tmu-env" in stages:
        stage_tmu_env()
    jobs = []
    if "torchtsetlin" in stages:
        for v in args.variants.split(","):
            name = VARIANT_NAME[v]
            for seed in seeds:
                for spec in SPECS:
                    tag = f"{spec}_seed{seed}" + (f"_{name}" if name else "")
                    extra = (["--variant", name] if name else []) + VARIANTS[v] + epochs
                    jobs.append({"tag": tag, "kind": "ttl", "spec": spec,
                                 "cmd": lambda g, spec=spec, seed=seed, extra=extra: [
                                     sys.executable, "train_specialist.py", "--spec", spec, "--seed", str(seed),
                                     "--device", f"cuda:{g}", *extra]})
    if "tmu" in stages:
        for run in range(args.tmu_runs):
            for spec in SPECS:
                tag = f"tmu_{spec}_run{run}" + ("_cal1" if args.quick else "")
                jobs.append({"tag": tag, "kind": "tmu", "spec": spec,
                             "cmd": lambda g, spec=spec, run=run: [
                                 sys.executable, "run_tmu_reference.py", "--spec", spec, "--run", str(run),
                                 "--gpu", str(g), *(["--epochs", "1"] if args.quick else [])]})
    if jobs:
        run_jobs(jobs, gpu_list(args.gpus), results, logs)
    if "report" in stages:
        for script in ("composite.py", "report.py", "proposal.py"):
            if (HERE / script).exists():
                subprocess.check_call([sys.executable, script], cwd=HERE)
        out = results.parent if args.quick else PROG
        log(f"wrote {out / 'REPORT.md'} and {out / 'PROPOSAL.md'}")


if __name__ == "__main__":
    main()
