import sys, traceback, warnings, torch, numpy as np
import torchtsetlin as tt
dev = torch.device("cuda")
def sites(m, x, y, n=3):
    for i in range(3): m.update(x[i*32:(i+1)*32], y[i*32:(i+1)*32])
    torch.cuda.synchronize()
    found = {}
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        orig = warnings.showwarning
        for i in range(3, 3 + n):
            m.update(x[i*32:(i+1)*32], y[i*32:(i+1)*32])
    torch.cuda.set_sync_debug_mode(0)
    for r in w:
        if "synchroniz" in str(r.message):
            key = f"{r.filename.split('torchtsetlin/')[-1]}:{r.lineno}"
            found[key] = found.get(key, 0) + 1
    return {k: v / n for k, v in found.items()}
d = np.load("/work/vajira/DL2026/torchtsetlin/experiments/paper_experiments/.cache/data/B4.npz")
x = torch.as_tensor(d["x_train"][:500], dtype=torch.bool, device=dev); y = torch.as_tensor(d["y_train"][:500], device=dev)
print("flat weighted:", sites(tt.TsetlinMachine(784, 10, 2000, 5000, 10.0, weighted=True).to(dev), x, y))
print("coalesced FNS budget:", sites(tt.CoalescedTsetlinMachine(784, 10, 2000, 500, 3.0, max_included_literals=20, focused_negative_sampling=True).to(dev), x, y))
print("conv:", sites(tt.ConvTsetlinMachine(10, 2000, 5000, 5.0, patch_size=10, input_shape=(1, 28, 28), weighted=True).to(dev), x.view(-1, 1, 28, 28), y))
