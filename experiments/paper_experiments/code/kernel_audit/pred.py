import sys, time, torch, numpy as np
import torchtsetlin as tt
dev = torch.device("cuda")
for name in ("B4", "B7"):
    d = np.load(f"/work/vajira/DL2026/torchtsetlin/experiments/paper_experiments/.cache/data/{name}.npz")
    x = torch.as_tensor(d["x_test"], dtype=torch.bool, device=dev)
    if name == "B4":
        m = tt.TsetlinMachine(784, 10, 2000, 50, 10.0)
    else:
        x = x.unsqueeze(1); m = tt.ConvTsetlinMachine(10, 2000, 5000, 5.0, patch_size=10, input_shape=(1, 28, 28), weighted=True)
    m = m.to(dev); xt = torch.as_tensor(d["x_train"][:2000], dtype=torch.bool, device=dev); yt = torch.as_tensor(d["y_train"][:2000], device=dev)
    if name == "B7": xt = xt.unsqueeze(1)
    for i in range(0, 2000, 100): m.update(xt[i:i+100], yt[i:i+100])
    m.eval()
    for _ in range(2): m.predict(x[:4096])
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(0, len(x), 4096): m.predict(x[i:i+4096])
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    print(f"{name} predict 10k: {dt*1e3:.0f} ms; chunk={m._chunk_size(x[:4096])}")
