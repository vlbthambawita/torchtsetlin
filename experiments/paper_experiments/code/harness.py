"""Measurement harness shared by every library adapter (PLAN.md §5).

Runs inside each library's own virtualenv, so it depends only on the standard library, numpy,
psutil and nvidia-ml-py (all installed by code/setup_envs.sh). torchtsetlin runs in the main env.
"""
import json
import os
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / ".cache" / "data"
REPO = HERE.parents[1]                       # the torchtsetlin repository

# --------------------------------------------------------------------------------------- data


def load(bench, split=0):
    """Return x_train, y_train, x_test, y_test for a benchmark (split k for B3 / B10)."""
    f = {"B5": "B4"}.get(bench, bench)
    d = np.load(DATA / f"{f}.npz")
    if "train_mask" in d:
        m = d["train_mask"][split]
        x, y = d["x"], d["y"]
        return x[m], y[m], x[~m], y[~m]
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


# ------------------------------------------------------------------------------------ metrics


def classification_metrics(y, p, n_classes):
    """Accuracy and precision / recall / F1 (positive class if binary, else macro)."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.int64)
    out = {"accuracy": float((y == p).mean())}
    prec, rec, f1 = [], [], []
    for c in range(n_classes):
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        prec.append(pr)
        rec.append(rc)
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    if n_classes == 2:
        out.update(precision=prec[1], recall=rec[1], f1=f1[1], averaging="positive class 1")
    else:
        out.update(precision=float(np.mean(prec)), recall=float(np.mean(rec)),
                   f1=float(np.mean(f1)), averaging="macro")
    return out


def regression_metrics(y, p):
    e = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean())}


# ------------------------------------------------------------------------------- memory probe


class MemoryProbe:
    """Peak host RSS of this process and peak GPU memory it holds (NVML), sampled every 50 ms.

    NVML is used for every GPU library alike (PyCUDA and PyTorch), so the numbers are comparable;
    torchtsetlin additionally reports torch.cuda.max_memory_allocated in its own record.
    """

    def __init__(self, period=0.05):
        import psutil
        self.proc = psutil.Process()
        self.period = period
        self.peak_rss = 0
        self.peak_gpu = 0
        self.base_rss = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._nv = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self._nv = None
        self._t = threading.Thread(target=self._run, daemon=True)

    def _gpu_bytes(self):
        if self._nv is None:
            return 0
        pid, tot = os.getpid(), 0
        for h in self._handles:
            try:
                for p in self._nv.nvmlDeviceGetComputeRunningProcesses(h):
                    if p.pid == pid and p.usedGpuMemory:
                        tot += p.usedGpuMemory
            except Exception:
                pass
        return tot

    def sample(self):
        self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
        self.peak_gpu = max(self.peak_gpu, self._gpu_bytes())

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            time.sleep(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.sample()

    def result(self):
        return {"peak_rss_mib": self.peak_rss / 2 ** 20,
                "rss_at_start_mib": self.base_rss / 2 ** 20,
                "peak_gpu_nvml_mib": self.peak_gpu / 2 ** 20}


# -------------------------------------------------------------------------------- environment


def environment():
    env = {"host": socket.gethostname(), "python": platform.python_version(),
           "numpy": np.__version__, "pid": os.getpid(),
           "threads_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
           "affinity": sorted(os.sched_getaffinity(0)),
           "loadavg_start": os.getloadavg()}
    try:
        env["cpu"] = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                      if ln.startswith("model name")][0]
        env["governor"] = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except Exception:
        pass
    try:
        env["gpus"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception:
        pass
    return env


def git_sha():
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "src"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty-src" if dirty else "")
    except Exception:
        return None


def write_json(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    tmp.replace(path)
