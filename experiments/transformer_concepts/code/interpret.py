"""Interventions and clause readout on the trained A2 pair TMs (PLAN 3.5 and 7; H4).

For every saved final A2 model (``results/final/*A2_pair_w2*.pt``), on its test set:

- accuracy with all pairs (must equal the record), with self pairs only (i = j), with the
  window |j - i| <= 1, and with the relative block shuffled across pairs;
- clause composition: the share of non-empty clauses (weighted by clause weight) that include
  literals on both the receiver block and a sender block ("cross-token"), and on the relative block;
- where the attention goes: for clauses of the predicted class that fire, the distribution of the
  offsets j - i of their matching pairs; for T4 the share at the planted offset d, for T2 the
  share of matching pairs whose sender is the query's key slot.

Writes results/interpret/<key>.json. Run after the finals:
    python -m experiments.transformer_concepts.code.interpret
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from experiments.transformer_concepts.code.pairs import pair_feature_names, pair_index, pair_tokens
from experiments.transformer_concepts.code.runner import ARMS, RESULTS, log_test_access, make_data

from torchtsetlin import functional as F

OUT = RESULTS / "interpret"


def blocks(names):
    out = {}
    for k, n in enumerate(names):
        if n.startswith("q="):
            out[k] = "q"
        elif n.startswith("k"):
            out[k] = "k"
        elif n.startswith("rel"):
            out[k] = "rel"
        else:
            out[k] = "extra"
    return out


@torch.no_grad()
def pair_matches(model, cols):
    xb = model._prepare(cols)
    lit = model._encode(xb)  # (B, P, 2F)
    B, P, L2 = lit.shape
    m = F.clause_outputs(lit.reshape(B * P, L2), model.include, model.include_count, False)
    return m.view(B, P, -1)


def analyse(rec_path: Path, device="cuda", split="test", out_dir=None):
    rec = json.loads(rec_path.read_text())
    key = rec["key"]
    out_dir = out_dir or OUT
    out_path = out_dir / f"{key}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    task, L = rec["task"], rec["L"]
    x, y, spec = make_data(task, L, split)
    if split == "test":
        log_test_access(f"interpret/{key}", len(x))
    arm = ARMS["A2_pair_w2"]
    model = arm.build(spec, L, rec["M"], rec["T"], rec["s"]).to(device)
    model._prepare(pair_tokens(x[:1].to(device), spec.V, window=2, last_bit=True))
    model.load_state_dict(torch.load(rec_path.with_suffix(".pt"), map_location=device))
    model.eval()
    ii, jj = pair_index(L, device=device)
    masks = {"all": torch.ones_like(ii, dtype=torch.bool), "self": ii == jj,
             "window1": (jj - ii).abs() <= 1}
    correct = {k: 0 for k in list(masks) + ["shuffled_rel"]}
    offs_hist = np.zeros(2 * L - 1)
    planted, total_pairs = 0, 0
    gen = torch.Generator(device="cpu").manual_seed(0)
    for k in range(0, len(x), 128):
        xb = x[k:k + 128].to(device)
        yb = y[k:k + 128].to(device)
        cols = pair_tokens(xb, spec.V, window=2, last_bit=True)
        mt = pair_matches(model, cols)  # (B, P, C)
        for name, mk in masks.items():
            out = (mt & mk.view(1, -1, 1)).any(1)
            pred = model._votes(out, clamp=False).argmax(1)
            correct[name] += int((pred == yb).sum())
        perm = torch.argsort(torch.rand(xb.shape[0], L * L, generator=gen), 1).to(device)
        cols_s = pair_tokens(xb, spec.V, window=2, last_bit=True, rel_perm=perm)
        pred_s = model._votes(pair_matches(model, cols_s).any(1), clamp=False).argmax(1)
        correct["shuffled_rel"] += int((pred_s == yb).sum())
        # where the attention goes: positive clauses of the predicted class that fire
        pred = model._votes(mt.any(1), clamp=False).argmax(1)
        C = model.n_clauses
        cls_idx = (pred.view(-1, 1) * C + torch.arange((C + 1) // 2, device=device).view(1, -1))  # positive slots
        sel = torch.gather(mt, 2, cls_idx.unsqueeze(1).expand(-1, mt.shape[1], -1))  # (B, P, C/2)
        pairs_used = sel.any(2)  # (B, P): a pair matched by some voting positive clause
        offs = (jj - ii).view(1, -1).expand_as(pairs_used)[pairs_used]
        offs_hist += np.bincount((offs + L - 1).cpu().numpy(), minlength=2 * L - 1)
        total_pairs += int(pairs_used.sum())
        if task in ("T4", "T4n"):
            planted += int((offs == spec.d).sum())
        elif task == "T2":
            q = xb[:, -1].view(-1, 1)
            key_slot = (xb == q) & (torch.arange(L, device=device).view(1, -1) < L - 1)  # (B, L)
            hit = key_slot[:, jj]  # sender is the query's key slot
            planted += int((hit & pairs_used).sum())
    names = pair_feature_names(L, spec.V, 2, False, True)
    Fn = len(names)
    bl = blocks(names)
    inc = (model.ta_state >= model.n_states).cpu()
    weights = model.weights.cpu().float() if hasattr(model, "weights") else torch.ones(inc.shape[0])
    comp = {"nonempty": 0.0, "cross": 0.0, "rel": 0.0, "q_only": 0.0, "k_only": 0.0}
    for r in range(inc.shape[0]):
        lits = torch.nonzero(inc[r]).flatten().tolist()
        if not lits:
            continue
        kinds = {bl[k % Fn] for k in lits}
        wt = float(weights[r])
        comp["nonempty"] += wt
        comp["cross"] += wt * ("q" in kinds and "k" in kinds)
        comp["rel"] += wt * ("rel" in kinds)
        comp["q_only"] += wt * ("q" in kinds and "k" not in kinds)
        comp["k_only"] += wt * ("k" in kinds and "q" not in kinds)
    n = len(x)
    res = dict(key=key, task=task, L=L, seed=rec["seed"], test_acc_record=rec.get("test_acc"),
               acc={k: v / n for k, v in correct.items()},
               composition={k: (v / comp["nonempty"] if k != "nonempty" else v) for k, v in comp.items()},
               offset_hist=offs_hist.tolist(), offsets=list(range(-(L - 1), L)),
               planted_share=(planted / total_pairs) if total_pairs and task in ("T2", "T4") else None,
               planted_definition={"T4": "offset == d", "T2": "sender is the query's key slot"}.get(task))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1))
    return res


def main():
    for p in sorted((RESULTS / "final").glob("*A2_pair_w2*.json")):
        if not p.with_suffix(".pt").exists():
            continue
        r = analyse(p)
        print(r["key"], {k: round(v, 4) for k, v in r["acc"].items()}, "cross", round(r["composition"]["cross"], 3),
              "planted", r["planted_share"], flush=True)


if __name__ == "__main__":
    main()
