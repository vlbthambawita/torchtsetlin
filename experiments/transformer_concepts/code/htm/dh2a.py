"""D-H2a (torchtsetlin_plans/experiments/transformer_concepts/PLAN_HTM.md, tests THEORY_HTM H2.2): is the trained `qkv` inbox saturated?

Reads the saved final `qkv` model (results/cifar/qkv_*_seed1_e30.pt) and 1,000 validation
images. Reports, per patch node: the share f of the 2,000 layer-0 parts that match (split into
empty parts, which match everywhere by the library's convention, and non-empty ones), the inbox
density, the density H2.2 predicts from f, and which share of the included literals of the
clauses' layer-2 parts are inbox literals. Validation data only.

    python -m experiments.transformer_concepts.code.htm.dh2a
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import torch

from experiments.transformer_concepts.code import cifar

OUT = Path(cifar.ROOT) / "results" / "htm" / "dh2a.json"


@torch.no_grad()
def main(device="cuda"):
    paths = sorted(glob.glob(str(cifar.RESULTS / "qkv_s*_seed1_e30.pt")))
    if not paths:
        raise SystemExit("no saved qkv final model yet")
    path = Path(paths[0])
    rec = json.loads(path.with_suffix(".json").read_text())
    D = cifar.load(device)
    model, feats = cifar.build("qkv", rec["s"], device)
    x = D["xva"][:1000]
    model._prepare(feats(x[:2]))  # lazy shape init
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    inc = (model.ta_state >= model.n_states)
    Fn = int(model.n_features)
    F0 = Fn - model.message_size
    n_empty0 = int(((inc[:, :F0].sum(1) + inc[:, Fn:Fn + F0].sum(1)) == 0).sum())
    lit_node = int(inc[:, :F0].sum() + inc[:, Fn:Fn + F0].sum())
    lit_inbox = int(inc[:, F0:Fn].sum() + inc[:, Fn + F0:].sum())
    fs, fs_nonempty, dens = [], [], []
    nonempty = ((inc[:, :F0].sum(1) + inc[:, Fn:Fn + F0].sum(1)) > 0)
    for k in range(0, len(x), 50):
        xb = model._prepare(feats(x[k:k + 50]))
        enc = model._encode(xb)
        running, xs = model._evaluate_layers(enc)
        m0 = running[0]  # (N, C) layer-0 parts matching at each node
        fs.append(m0.float().mean(1))
        fs_nonempty.append(m0[:, nonempty].float().mean(1))
        dens.append(xs[1].float().mean(1))
    f = torch.cat(fs)
    f_ne = torch.cat(fs_nonempty)
    q = torch.cat(dens)
    C, b, M, n = model.n_clauses_total, model.message_bits, model.message_size, 63
    fm = f.mean().item()
    res = dict(model=path.name, s=rec["s"], test_acc=rec.get("test_acc"), n_images=len(x),
               n_clauses=C, message_bits=b, message_size=M, senders=n,
               empty_layer0_parts=n_empty0, f_mean=fm, f_nonempty_mean=f_ne.mean().item(),
               inbox_density_mean=q.mean().item(), inbox_density_min=q.min().item(),
               inbox_density_predicted_H2_2=1 - math.exp(-b * C * fm * n / M),
               f_threshold_for_q_099=4.605 * M / (b * C * n),
               included_literals_layer0=lit_node, included_literals_inbox=lit_inbox,
               inbox_literal_share=lit_inbox / max(1, lit_node + lit_inbox),
               prediction="f > 0.02 and inbox density > 0.95 (THEORY_HTM P-H2a)",
               prediction_holds=bool(fm > 0.02 and q.mean().item() > 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
