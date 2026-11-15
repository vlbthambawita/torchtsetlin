"""Smoke-test checks of the encodings and of the plane bookkeeping (plan §2.1, §3).

1. bit8 reconstructs every pixel exactly (sum 2^k B_k == v), MSB first.
2. therm8/therm4 are monotone (plane k+1 implies plane k), their thresholds are 255*k/(n+1), and
   the number of planes on equals the number of thresholds <= v.
3. gray8 decodes back to v; bit4 equals the top 4 planes of bit8; agt equals the paper's B8.npz.
4. feature_groups(): an input with only plane p on lights only group p's features (flat, conv2,
   conv3 at depth position 0) — the layout that plane_usage relies on.
Writes results/smoke/encoding_checks.json (with the per-plane bit density of every encoding).
"""

import numpy as np
import torch

from common import B8, ENCODINGS, RESULTS, encode, load_raw, write_json
from train import build_model, feature_groups


def main():
    raw = load_raw()
    x = np.concatenate([raw["x_train"], raw["x_test"]])            # (70000, 28, 28) uint8
    v = x.astype(np.int64)
    out = {"n_images": int(len(x)), "density": {}}
    with np.load(B8) as f:
        agt_all = np.concatenate([f["x_train"], f["x_test"]])
    enc = {e: encode(e, x, agt_all) for e in ENCODINGS}
    for e, a in enc.items():
        assert a.dtype == np.bool_ and a.shape == (len(x), ENCODINGS[e], 28, 28), (e, a.shape, a.dtype)
        out["density"][e] = [round(float(a[:, p].mean()), 4) for p in range(a.shape[1])]

    w = (1 << np.arange(7, -1, -1))[None, :, None, None]
    out["bit8_reconstructs"] = bool(np.array_equal((enc["bit8"] * w).sum(1), v))
    for e, n in (("therm8", 8), ("therm4", 4)):
        a = enc[e]
        thr = 255 * np.arange(1, n + 1) / (n + 1)
        mono = bool(np.all(a[:, 1:] <= a[:, :-1]))
        count = bool(np.array_equal(a.sum(1), (v[:, None] >= thr[None, :, None, None]).sum(1)))
        out[f"{e}_monotone"], out[f"{e}_count_matches_thresholds"] = mono, count
        out[f"{e}_thresholds"] = thr.round(2).tolist()
    g = (enc["gray8"] * w).sum(1)
    b = g.copy()
    for k in (1, 2, 4):
        b ^= b >> k
    out["gray8_roundtrip"] = bool(np.array_equal(b, v))
    out["bit4_is_top4_of_bit8"] = bool(np.array_equal(enc["bit4"], enc["bit8"][:, :4]))
    out["agt_equals_B8"] = bool(np.array_equal(enc["agt"][:, 0], agt_all != 0))

    lay = {}
    for arch, planes, kd in (("flat", 8, None), ("conv2", 8, None), ("conv3", 8, 2), ("conv3", 8, 4)):
        m = build_model(arch, planes, 10, 10, 5.0, kd)
        groups, names, pos_id = feature_groups(arch, planes, kd, int(m.n_features))
        ok = True
        for p in range(planes if arch != "conv3" else kd):
            xb = torch.zeros(1, planes, 28, 28, dtype=torch.bool)
            xb[:, p] = True
            if arch == "flat":
                feats = xb.reshape(1, -1)[0].numpy()
            else:
                xin = xb if arch == "conv2" else xb.unsqueeze(1)
                lit = m._encode(m._coerce_input(xin))[0, 0]           # first patch, (2F,)
                feats = lit[: m.n_features].numpy() > 0.5
                if pos_id is not None:
                    feats = feats & (groups != pos_id)
            ok &= bool(np.all(groups[feats] == p)) and bool(feats.any())
        lay[f"{arch}{'' if kd is None else f'_kd{kd}'}"] = ok
    out["feature_groups_layout"] = lay

    checks = [k for k, val in out.items() if isinstance(val, bool)] + \
             [f"feature_groups_layout.{k}" for k in lay]
    out["all_pass"] = all(out[k] for k in out if isinstance(out[k], bool)) and all(lay.values())
    write_json(out, RESULTS / "smoke" / "encoding_checks.json")
    print({k: out[k] for k in out if k != "density"})
    print("density per plane:", out["density"])
    if not out["all_pass"]:
        raise SystemExit(f"encoding checks failed: {checks}")


if __name__ == "__main__":
    main()
