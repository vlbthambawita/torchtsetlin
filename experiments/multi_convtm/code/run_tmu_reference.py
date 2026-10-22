"""Run the authors' TM Composites scripts (reference_code/, tmu) and record their output.

Each specialist runs the authors' script almost verbatim (patches are marked
"# PATCH (multi_convtm)") inside its own working directory, with the keras shim on PYTHONPATH,
in the isolated tmu environment .cache/venv-tmu. The script itself writes the per-epoch test-set
class sums (its own np.savetxt); this wrapper parses its "#<epoch> Accuracy" lines into
results/tmu_<spec>_run<k>.json.

Usage: python code/run_tmu_reference.py --spec C3 --run 0 --gpu 0 [--epochs 1]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import CACHE, DATA, PROG, RESULTS, VENV_TMU, new_record, write_json

REF = PROG / "reference_code"
VENV_PY = VENV_TMU / "bin" / "python"
SCRIPTS = {   # spec -> (script, extra args); hyperparameters are the scripts' own defaults = Table 1
    "T10": ("CIFAR10AdaptiveThresholding.py", []),
    "C3": ("CIFAR10ColorThermometerScoring.py", ["--patch_size", "3"]),
    "C4": ("CIFAR10ColorThermometerScoring.py", ["--patch_size", "4"]),
    "HoG": ("CIFAR10HistogramOfGradients.py", []),
}
SYS_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
LINE = re.compile(r"#(\d+) Accuracy: ([\d.]+)% Training: ([\d.]+)s Testing: ([\d.]+)s")


def cuda_root():
    if os.environ.get("CUDA_ROOT"):
        return os.environ["CUDA_ROOT"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parents[1])
    cands = sorted(Path("/usr/local").glob("cuda-12*"), reverse=True)
    for c in cands:
        if (c / "bin" / "nvcc").exists():
            return str(c)
    sys.exit("no CUDA toolkit (nvcc) found; set CUDA_ROOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SCRIPTS))
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--epochs", type=int, default=0, help="override (calibration only)")
    args = ap.parse_args()

    script, extra = SCRIPTS[args.spec]
    tag = f"tmu_{args.spec}_run{args.run}" + (f"_cal{args.epochs}" if args.epochs else "")
    wd = CACHE / "tmu_runs" / tag
    wd.mkdir(parents=True, exist_ok=True)
    root = cuda_root()
    env = dict(os.environ,
               PYTHONPATH=str(REF / "shim"), CIFAR10_DIR=str(DATA / "cifar-10-batches-py"),
               CUDA_VISIBLE_DEVICES=args.gpu, CUDA_ROOT=root,
               PATH=f"{root}/bin:" + os.environ.get("PATH", ""), PYTHONUNBUFFERED="1")
    if Path(SYS_LIBSTDCXX).exists():
        # PyCUDA builds against the system libstdc++; a conda python would otherwise load its
        # own older copy first (GLIBCXX_3.4.32 not found).
        env["LD_PRELOAD"] = SYS_LIBSTDCXX
    if args.epochs:
        env["MC_EPOCHS"] = str(args.epochs)
    cmd = [str(VENV_PY), str(REF / script), *extra]
    ver = subprocess.run([str(VENV_PY), "-I", "-c", "import importlib.metadata as m;print(m.version('tmu'))"],
                         capture_output=True, text=True, env=env).stdout.strip()
    rec = new_record("tmu_specialist", spec=args.spec, run=args.run, script=script, script_args=extra,
                     tmu_version=ver, cmd=cmd, workdir=str(wd), curve=[], status="running",
                     note="tmu does not expose a seed in these scripts; runs differ by tmu's own RNG.")
    out = RESULTS / f"{tag}.json"
    t0 = time.time()
    with open(wd / "stdout.log", "w") as log:
        p = subprocess.Popen(cmd, cwd=wd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            m = LINE.search(line)
            if m:
                e, acc, tr, te = int(m[1]), float(m[2]), float(m[3]), float(m[4])
                rec["curve"].append({"epoch": e, "test_acc": acc, "train_s": tr, "test_s": te})
                rec["wall_s"] = time.time() - t0
                write_json(rec, out)
                print(f"[{tag}] epoch {e} {acc:.2f}% train {tr:.0f}s test {te:.0f}s", flush=True)
        rc = p.wait()
    rec["returncode"] = rc
    rec["status"] = "done" if rc == 0 and rec["curve"] else "failed"
    c = {e["epoch"]: e["test_acc"] for e in rec["curve"]}
    rec["test_acc_epoch100"], rec["test_acc_epoch99"] = c.get(100), c.get(99)
    write_json(rec, out)
    if rc:
        print((wd / "stdout.log").read_text()[-3000:], file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
