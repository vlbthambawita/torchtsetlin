import time, torch, numpy as np
import torchtsetlin as tt
from torchtsetlin import functional as F
dev = torch.device("cuda")
d = np.load("/work/vajira/DL2026/torchtsetlin/experiments/paper_experiments/.cache/data/B1.npz")
x = torch.as_tensor(d["x_train"], dtype=torch.bool, device=dev); y = torch.as_tensor(d["y_train"], device=dev)
xt = torch.as_tensor(d["x_test"], dtype=torch.bool, device=dev); yt = torch.as_tensor(d["y_test"], device=dev)
F.COMPILE_MIN_ELEMENTS = 0
m = tt.TsetlinMachine(12, 2, 10, 15, 3.9, boost_true_positive=False, feedback_mode="sequential").to(dev)
m._coerce_targets = lambda yy, n, dv: yy.view(-1)
m.update(x[:200], y[:200])                      # warm up / compile the fused step
torch.cuda.synchronize(); t = time.perf_counter(); m.update(x[:2000], y[:2000]); torch.cuda.synchronize()
eager = (time.perf_counter() - t) / 2000
sx, sy = x[:1].clone(), y[:1].clone()
K = 64                                          # examples per captured graph
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        for _ in range(K): m.update(sx, sy)
torch.cuda.current_stream().wait_stream(s)
gx, gy = x[:K].clone(), y[:K].clone()
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        for i in range(K):
            m.update(gx[i:i + 1], gy[i:i + 1])
except Exception as e:
    print("capture failed:", type(e).__name__, str(e)[:300]); raise SystemExit
m.ta_state.fill_(m.n_states - 1); m.include.zero_(); m.include_count.zero_()
torch.cuda.synchronize(); t = time.perf_counter()
for ep in range(20):
    perm = torch.randperm(5000, device=dev)
    for i in range(0, 5000 - K + 1, K):
        gx.copy_(x[perm[i:i + K]]); gy.copy_(y[perm[i:i + K]]); g.replay()
torch.cuda.synchronize(); graphed = (time.perf_counter() - t) / (20 * (5000 // K) * K)
m.eval(); acc = float((m.predict(xt) == yt).float().mean())
print(f"seq per example: eager-fused {eager*1e6:.0f} us, CUDA graph {graphed*1e6:.0f} us ({eager/graphed:.1f}x); "
      f"B1 acc after 20 graphed epochs {acc:.3f}")
