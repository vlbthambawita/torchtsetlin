"""G9 check: B4's tt model in sequential mode, one epoch from initialisation, on CUDA.

B4 has no tt-seq cell since D10, so the harness refuses this configuration; the script mirrors
the tt adapter (data on the device, update() calls of 1000 examples, eval() before predicting).

    python code/gap_timing_seq.py compiled|eager OUT.json
"""
import json
import subprocess
import sys
import time

import numpy as np
import torch

import torchtsetlin as tt
from torchtsetlin import functional as F

mode, out = sys.argv[1], sys.argv[2]
if mode == "eager":
    F.set_compile(False)
d = np.load(__file__.rsplit("/code/", 1)[0] + "/.cache/data/B4.npz")
dev = torch.device("cuda")
x, y = torch.as_tensor(d["x_train"] > 0).to(dev), torch.as_tensor(d["y_train"]).to(dev)
xt, yt = torch.as_tensor(d["x_test"] > 0).to(dev), torch.as_tensor(d["y_test"]).to(dev)
torch.manual_seed(0)
m = tt.TsetlinMachine(784, 10, n_clauses=2000, T=50, s=10.0, boost_true_positive=True,
                      weighted=False, feedback_mode="sequential").to(dev)
perm = torch.randperm(60000, generator=torch.Generator().manual_seed(0)).to(dev)
torch.cuda.synchronize()
t = time.perf_counter()
for i in range(0, 60000, 1000):
    m.update(x[perm[i:i + 1000]], y[perm[i:i + 1000]])
torch.cuda.synchronize()
train_s = time.perf_counter() - t
m.eval()
acc = float((m.predict(xt) == yt).float().mean())
root = __file__.rsplit("/experiments/", 1)[0]
git = ["git", "-C", root]
sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(git + ["status", "--porcelain", "src"], capture_output=True, text=True).stdout
rec = dict(kind="gap_check", gap="G9", bench="B4", lib="tt", mode="seq", compile=mode, seed=0,
           device=torch.cuda.get_device_name(dev), torch=torch.__version__, lib_version=tt.__version__,
           git_sha=sha + ("-dirty-src" if dirty else ""), train_s_epoch1_incl_compile=train_s,
           accuracy=acc)
json.dump(rec, open(out, "w"), indent=1)
print(rec)
