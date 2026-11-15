"""Library CPU vs CUDA batch feedback on the same data: included literals and accuracy (5 seeds).
Probe for a possible library gap seen in E-H1a (D20). Writes results/htm/lib_device_check.json."""
import json
from statistics import mean, stdev

import torch

import torchtsetlin as tt
from experiments.transformer_concepts.code.htm import exp_h1 as E

x, y, K, _ = E.dataset("T5", "train")
x, y = x[:4000], y[:4000]
xv, yv, _, _ = E.dataset("T5", "valid")
xv, yv = xv[:2000], yv[:2000]
out = {}
for dev in ("cpu", "cuda"):
    lits, accs = [], []
    for seed in range(5):
        torch.manual_seed(seed)
        m = tt.Conv1dTsetlinMachine(2, 100, 25, 5.0, kernel_size=1, position_encoding=False, weighted=True).to(dev)
        g = torch.Generator().manual_seed(seed)
        xd, yd = x.to(dev), y.to(dev)
        for ep in range(2):
            p = torch.randperm(len(x), generator=g).to(dev)
            for k in range(0, len(x), 32):
                m.update(xd[p[k:k + 32]], yd[p[k:k + 32]])
        lits.append(float((m.ta_state >= m.n_states).sum(1).float().mean()))
        m.eval()
        accs.append(float((m.predict(xv.to(dev)).cpu() == yv).float().mean()))
    out[dev] = dict(included_per_clause=lits, mean=mean(lits), sd=stdev(lits), val_acc=accs)
    print(dev, out[dev], flush=True)
E.R.mkdir(parents=True, exist_ok=True)
(E.R / "lib_device_check.json").write_text(json.dumps(out, indent=1))
