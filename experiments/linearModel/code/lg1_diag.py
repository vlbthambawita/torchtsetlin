"""LG-1 diagnostic: unweighted ConvTsetlinMachine on MNIST (records in results/lg1/).

    python experiments/linearModel/code/lg1_diag.py NAME key=val ...
"""
import json, sys, time
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from experiments.linearModel.code.data import load_split
import torchtsetlin as tt

name = sys.argv[1]
cfg = dict(M=200, T=100.0, s=10.0, n_train=50000, epochs=3, batch=32, seq=0, boost=1, budget=0,
           weighted=0, pos=1, seed=0)
for a in sys.argv[2:]:
    k, v = a.split("="); cfg[k] = type(cfg[k])(v)
torch.manual_seed(cfg["seed"])
dev = "cuda"
tr, va = load_split("split_seed0", "train"), load_split("split_seed0", "valid")
Xtr = (torch.as_tensor(tr.X[:cfg["n_train"]]) > 0.5).reshape(-1, 1, 28, 28).to(dev)
ytr = torch.as_tensor(tr.y[:cfg["n_train"]]).to(dev)
Xva = (torch.as_tensor(va.X) > 0.5).reshape(-1, 1, 28, 28).to(dev)
yva = torch.as_tensor(va.y).to(dev)
kw = {}
if cfg["budget"]:
    kw["max_included_literals"] = cfg["budget"]
m = tt.ConvTsetlinMachine(10, cfg["M"], T=cfg["T"], s=cfg["s"], patch_size=10, stride=1,
                          position_encoding=bool(cfg["pos"]), input_shape=(1, 28, 28),
                          weighted=bool(cfg["weighted"]), boost_true_positive=bool(cfg["boost"]),
                          feedback_mode="sequential" if cfg["seq"] else "batch", **kw).to(dev)
hist = []
for ep in range(1, cfg["epochs"] + 1):
    t0 = time.time(); m.train()
    perm = torch.randperm(len(ytr), device=dev)
    for i in range(0, len(ytr), cfg["batch"]):
        idx = perm[i:i + cfg["batch"]]
        m.update(Xtr[idx], ytr[idx])
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m.predict(Xva[i:i + 1000]) for i in range(0, len(yva), 1000)])
    acc = float((pred == yva).float().mean())
    lit = m.include_count.float()
    h = dict(epoch=ep, val=round(acc, 4), mean_lits=round(float(lit.mean()), 1),
             sec=round(time.time() - t0, 1))
    hist.append(h); print(name, h, flush=True)
json.dump(dict(name=name, cfg=cfg, hist=hist), open(__import__("pathlib").Path(__file__).resolve().parents[1] / "results" / "lg1" / f"{name}.json", "w"))
