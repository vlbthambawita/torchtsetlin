"""B2 tt-seq on CUDA: does predict() use a stale include_bits after graph replays?"""
import sys, torch, numpy as np
sys.path.insert(0, "code")
import harness, torchtsetlin as tt
xtr, ytr, xte, yte = harness.load("B2")[:4]
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
dev = torch.device("cuda")
m = tt.ConvTsetlinMachine(2, n_clauses=40, T=60, s=3.9, boost_true_positive=False, patch_size=2,
                          stride=1, position_encoding=True, input_shape=[1, 4, 4], weighted=False,
                          feedback_mode="sequential").to(dev)
X = torch.as_tensor(xtr, dtype=torch.bool).reshape(len(xtr), 1, 4, 4).to(dev); Y = torch.as_tensor(ytr).long().to(dev)
Xt = torch.as_tensor(xte, dtype=torch.bool).reshape(len(xte), 1, 4, 4).to(dev); Yt = torch.as_tensor(yte).long().to(dev)
g = torch.Generator().manual_seed(0)
for ep in range(1, 13):
    m.train(); perm = torch.randperm(len(X), generator=g).to(dev)
    for i in range(0, len(X), 1000):
        m.update(X[perm[i:i+1000]], Y[perm[i:i+1000]])
    m.eval()
    a = (m.predict(Xt) == Yt).float().mean().item()
    stale = not torch.equal(m.include_bits(), tt._triton.pack_bits(m.include)) if hasattr(tt, "_triton") else None
    m.__dict__["_bits_fresh"] = False   # force a refresh
    b = (m.predict(Xt) == Yt).float().mean().item()
    print(f"epoch {ep:2d}: predict {a*100:5.1f}  bits stale={stale}  after refresh {b*100:5.1f}  "
          f"graphs={[v if isinstance(v,str) else 'graph' for v in m.__dict__.get('_graphs',{}).values()]}")
