"""Per-update anatomy of a tt update on CUDA: python deep.py <bench> <mode> [n]
Prints wall ms/update, GPU-busy ms/update, kernel launches/update, host syncs/update, and the
GPU time grouped by kernel family."""
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/work/vajira/DL2026/torchtsetlin/experiments/paper_experiments/code")
from configs import cell_config  # noqa: E402
import torchtsetlin as tt  # noqa: E402

bench, mode = sys.argv[1], sys.argv[2]
n_up = int(sys.argv[3]) if len(sys.argv) > 3 else 50
dev = torch.device("cuda")
cfg = cell_config(bench, "tt", mode)
d = np.load(f"/work/vajira/DL2026/torchtsetlin/experiments/paper_experiments/.cache/data/{cfg['data']}.npz")
if "x_train" in d:
    x, y = d["x_train"], d["y_train"]
else:
    m_ = d["train_mask"][0]
    x, y = d["x"][m_], d["y"][m_]
x = torch.as_tensor(x, dtype=torch.bool)
y = torch.as_tensor(y)
kw = dict(cfg["lib_cfg"]["kwargs"], feedback_mode=cfg["tt"]["feedback_mode"])
cls = cfg["lib_cfg"]["cls"]
if cls == "ConvTsetlinMachine":
    K = int(y.max()) + 1
    m = tt.ConvTsetlinMachine(K, **kw)
    x = x.reshape(len(x), 1, *x.shape[1:])
elif cls == "CoalescedTsetlinMachine":
    m = tt.CoalescedTsetlinMachine(int(np.prod(x.shape[1:])), int(y.max()) + 1, **kw)
elif cls == "RegressionTsetlinMachine":
    y = y.float()
    m = tt.RegressionTsetlinMachine(int(np.prod(x.shape[1:])), y_range=(float(y.min()), float(y.max())), **kw)
else:
    m = tt.TsetlinMachine(int(np.prod(x.shape[1:])), int(y.max()) + 1, **kw)
m = m.to(dev)
x, y = x.to(dev), y.to(dev)
seq = cfg["tt"]["feedback_mode"] == "sequential"
bs = 1000 if seq else cfg["tt"]["batch_size"]
per = bs  # examples per update() call
torch.manual_seed(0)
perm = torch.randperm(len(x), device=dev)


def run(n, off=0):
    for i in range(n):
        s = ((off + i) * bs) % max(1, len(x) - bs)
        idx = perm[s:s + bs]
        m.update(x[idx], y[idx])


run(20 if not seq else 2)
torch.cuda.synchronize()
t = time.perf_counter()
run(n_up, 20)
torch.cuda.synchronize()
wall = (time.perf_counter() - t) / n_up

# host syncs
syncs = [0]
torch.cuda.set_sync_debug_mode("warn")
import warnings  # noqa: E402
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    run(3, 200)
    torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode(0)
n_sync = sum("synchroniz" in str(x.message) for x in w) / 3

from torch.profiler import ProfilerActivity, profile  # noqa: E402
k = min(n_up, 10)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    run(k, 300)
    torch.cuda.synchronize()
fam = defaultdict(float)
n_k = 0
gpu = 0.0
for e in p.events():
    if e.device_type.name == "CUDA":
        n_k += 1
        dt = e.device_time if hasattr(e, "device_time") else e.cuda_time
        gpu += dt
        nm = e.name
        if "gemm" in nm or "sgemm" in nm or "cutlass" in nm:
            key = "GEMM"
        elif nm.startswith("triton"):
            key = "triton:" + re.sub(r"_\d+$", "", nm)[:60]
        elif "Memcpy" in nm or "Memset" in nm:
            key = nm.split(" (")[0]
        else:
            key = re.sub(r"<.*", "", nm)[:60]
        fam[key] += dt
unit = 1e3 * k  # us -> ms per update
ex = per if not seq else per
print(f"{bench} {mode} bs={bs}: wall {wall*1e3:.2f} ms/update, GPU busy {gpu/unit:.2f} ms/update, "
      f"kernels {n_k/k:.0f}/update, host syncs {n_sync:.0f}/update, examples/update {ex}, "
      f"automata {m.ta_state.numel()/1e6:.1f}M")
for key, v in sorted(fam.items(), key=lambda kv: -kv[1])[:12]:
    print(f"   {v/unit:7.3f} ms  {key}")
