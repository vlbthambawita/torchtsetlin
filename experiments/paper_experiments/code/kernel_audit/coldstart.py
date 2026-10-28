import time, torch
t0 = time.perf_counter()
import torchtsetlin as tt
from torch._inductor import config as ic
dev = torch.device("cuda")
x = torch.randint(0, 2, (3000, 784), device=dev).bool(); y = torch.randint(0, 10, (3000,), device=dev)
m = tt.TsetlinMachine(784, 10, 2000, 50, 10.0).to(dev)
t = time.perf_counter(); m.update(x[:32], y[:32]); torch.cuda.synchronize(); first = time.perf_counter() - t
t = time.perf_counter()
for i in range(1, 50): m.update(x[i*32:(i+1)*32], y[i*32:(i+1)*32])
torch.cuda.synchronize(); steady = (time.perf_counter() - t) / 49
print(f"torch {torch.__version__} fx_graph_cache={ic.fx_graph_cache}: first update {first:.2f} s, steady {steady*1e3:.1f} ms")
