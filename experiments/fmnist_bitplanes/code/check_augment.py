"""Phase 0 checks for the augmentation runs (plans/fmnist-bitplanes-augmentation.md §4).

1. encode_gpu == common.encode, bit for bit, on all 60 000 training images (bit8, gray8).
2. Zero shift and no flip is the identity; a shift moves the pixels where expected; a flip mirrors.
3. Every (dy, dx) in [-2, 2]^2 and both flips occur, and the draws repeat with the same seed.
Writes results/smoke/augment_checks.json. Tiny training checks (resume, reload) are in aug_smoke.sh.
"""

import sys

import numpy as np
import torch

from augment import Augmenter, encode_gpu
from common import RESULTS, encode, load_raw, write_json

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
raw = load_raw()["x_train"]
out = {}
xt = torch.from_numpy(raw).to(dev)
for enc in ("bit8", "gray8"):
    ref = torch.from_numpy(encode(enc, raw)).to(dev)
    got = torch.cat([encode_gpu(enc, xt[i:i + 10000]) for i in range(0, len(xt), 10000)])
    out[f"encode_gpu_equals_cpu_{enc}"] = bool(torch.equal(ref.bool(), got))

g = torch.Generator().manual_seed(0)
a = Augmenter(xt, 2, 0.5, g)
idx = torch.arange(64, device=dev)
a.off = torch.full((len(xt), 2), 2, device=dev)          # offset 2 = no shift
a.flip = torch.zeros(len(xt), dtype=torch.bool, device=dev)
out["identity"] = bool(torch.equal(a.batch(idx), xt[:64]))
a.off[:, 0], a.off[:, 1] = 0, 4                          # rows shifted down 2 (content moves down), cols left 2
b = a.batch(idx)
exp = torch.zeros_like(xt[:64])
exp[:, 2:, :-2] = xt[:64, :-2, 2:]
out["shift"] = bool(torch.equal(b, exp))
a.off[:] = 2
a.flip[:] = True
out["flip"] = bool(torch.equal(a.batch(idx), xt[:64].flip(-1)))

def draws(seed):
    a = Augmenter(xt, 2, 0.5, torch.Generator().manual_seed(seed))
    a.draw_epoch()
    return a.off.cpu(), a.flip.cpu()
o1, f1 = draws(2000)
o2, f2 = draws(2000)
out["reproducible"] = bool(torch.equal(o1, o2) and torch.equal(f1, f2))
pairs = {(int(y), int(x)) for y, x in o1.tolist()}
out["all_25_offsets"] = len(pairs) == 25 and min(min(p) for p in pairs) == 0 and max(max(p) for p in pairs) == 4
out["flip_rate"] = float(f1.float().mean())
print(out)
write_json(out, RESULTS / "smoke" / "augment_checks.json")
assert all(v for k, v in out.items() if k != "flip_rate") and 0.49 < out["flip_rate"] < 0.51, out
print("augment checks OK")
