"""Smoke test on synthetic data: exercises every MCTM code path before the CIFAR run."""
import torch, torchtsetlin as tt
from mctm import MCTM, conv_clause_maps, effective_receptive_field, firing_stats, or_pool, randomize_clauses

dev = "cuda"
tt.seed_everything(0)
N, Z, H, W = 400, 12, 32, 32
# signal: class = which quadrant carries a dense blob
x = (torch.rand(N, Z, H, W, device=dev) < 0.15)
y = torch.randint(0, 10, (N,), device=dev)
for i in range(N):
    q = int(y[i]) % 4
    r, c = (q // 2) * 16, (q % 2) * 16
    x[i, :, r:r+8, c:c+8] = True

l1 = tt.ConvCoalescedTsetlinMachine(10, 32, 64.0, 10.0, patch_size=4, stride=1,
                                    input_shape=(Z, H, W)).to(dev)
l1.train()
for ep in range(15):
    for i in range(0, N, 50):
        l1.update(x[i:i+50], y[i:i+50])
l1.eval()
print("l1 include_count min/median/max:", int(l1.include_count.min()),
      int(l1.include_count.median()), int(l1.include_count.max()))

maps = conv_clause_maps(l1, x[:64])
print("maps", tuple(maps.shape), maps.dtype, "expect (64, 32, 29, 29) bool")
assert maps.shape == (64, 32, 29, 29) and maps.dtype == torch.bool

# empty_value must not leak all-ones channels
fresh = tt.ConvCoalescedTsetlinMachine(10, 8, 32.0, 10.0, patch_size=4, stride=1,
                                       input_shape=(Z, H, W)).to(dev)
fresh.train()  # training mode: empty clauses would be True if not forced
m0 = conv_clause_maps(fresh, x[:8])
print("fresh-model map density (must be 0.0):", float(m0.float().mean()))
assert float(m0.float().mean()) == 0.0, "empty clauses leaked an all-ones channel"

pooled = or_pool(maps, 2)
print("pooled", tuple(pooled.shape), "density", f"{maps.float().mean():.4f} -> {pooled.float().mean():.4f}")
assert pooled.shape == (64, 32, 14, 14) and pooled.float().mean() >= maps.float().mean()
assert float(maps.float().mean()) > 0.0, "trained layer 1 produced entirely dead maps"

st = firing_stats(pooled)
print("firing median", round(st["median"], 4), "dead", round(st["frac_dead"], 3))

rnd = tt.ConvCoalescedTsetlinMachine(10, 32, 64.0, 10.0, patch_size=4, stride=1,
                                     input_shape=(Z, H, W)).to(dev)
randomize_clauses(rnd, n_include=3, seed=1)
inc = rnd.include_count
print("random clause include counts:", inc.min().item(), inc.max().item(), "expect 3 3")
assert int(inc.min()) == 3 and int(inc.max()) == 3
rnd.eval()
rmaps = conv_clause_maps(rnd, x[:64])
print("random-layer1 map density", f"{rmaps.float().mean():.4f}")
assert 0.0 < float(rmaps.float().mean()) < 1.0

# stack: layer 2 over the pooled maps
l2 = tt.ConvCoalescedTsetlinMachine(10, 64, 128.0, 10.0, patch_size=3, stride=1,
                                    input_shape=tuple(pooled.shape[1:])).to(dev)
# pooled is (32,14,14); 3x3 stride 1 -> Py=Px=12 -> (12-1)+(12-1)=22 position bits
print("layer2 n_features", l2.n_features, "= 32*9 + 22 =", 32*9+22)
assert l2.n_features == 32*9 + 22
htr = or_pool(conv_clause_maps(l1, x), 2)
l2.train(); l2.update(htr[:100], y[:100])
stack = MCTM([l1], [2], l2)
out = stack.forward(x[:16])
print("stack out", tuple(out.shape), "clauses", stack.n_clauses_total(), "automata", stack.n_automata())
assert out.shape == (16, 10)
print("RF", effective_receptive_field([4, 3], [1, 1], [2, 1]))
print("SMOKE OK")
