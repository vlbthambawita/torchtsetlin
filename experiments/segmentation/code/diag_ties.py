"""Diagnostic: vote clipping ties at prediction time (PLAN section 2: 'measure clipping/ties').

    python code/diag_ties.py RUN_ID [--device cuda:1]

For a frozen TM checkpoint, compare argmax of clipped votes (library ``forward``) against
argmax of unclipped votes on validation; report tie rates and metrics for both.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalio  # noqa: E402
import record  # noqa: E402
import run as R  # noqa: E402


@torch.no_grad()
def votes_both(cfg, m, d, dev, bs=2):
    clipped, raw = [], []
    for i in range(0, len(d["x"]), bs):
        ex = d["extra"][i : i + bs] if d.get("extra") is not None else None
        xb = m._coerce_input(R.model_input(cfg, d["x"][i : i + bs], dev, ex))
        B, _, H, W = xb.shape
        out, _ = m._evaluate(m._encode(xb), empty_value=False)
        v = m._votes(out, clamp=False)
        raw.append(v.view(B, H, W, -1).permute(0, 3, 1, 2).float().cpu())
        clipped.append(raw[-1].clamp(-float(m.T), float(m.T)))
    return torch.cat(clipped), torch.cat(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    for rid in a.runs:
        rec = evalio.load_run(rid)
        cfg = {**R.DEFAULTS, **rec["config"]}
        d = dict(R.load_data(cfg, "val"))
        if cfg.get("stack"):
            import hierarchy
            d["extra"] = hierarchy.deploy_features(cfg, rec, d, R.K_OF[cfg["dataset"]], a.device)
        cfg, m = evalio.load_model(rec, a.device, d)
        vc, vr = votes_both(cfg, m, d, a.device)
        valid = (d["y"] != 255).unsqueeze(1)
        T = float(m.T)
        top = vc.max(1, keepdim=True).values
        n_top = (vc == top).sum(1)
        v = valid.squeeze(1)
        out = {"T": T,
               "tie_rate": float((n_top[v] > 1).float().mean()),
               "tie_at_plusT_rate": float(((n_top > 1) & (top.squeeze(1) >= T))[v].float().mean()),
               "any_class_clipped_rate": float(((vr.abs() >= T).any(1))[v].float().mean()),
               "pred_changed_rate": float((vc.argmax(1) != vr.argmax(1))[v].float().mean())}
        K = R.K_OF[cfg["dataset"]]
        sc, _ = R.eval_screen(vc.half(), d, K, boundaries=False)
        sr, st = R.eval_screen(vr, d, K, boundaries=False)
        out["clipped_miou"], out["unclipped_miou"] = sc["miou"], sr["miou"]
        out["clipped_iou"], out["unclipped_iou"] = sc["iou"], sr["iou"]
        print(rid, out, flush=True)
        record.write_record({"run_id": f"DIAG_ties_{rid}", "status": "completed", "source_run": rid,
                             "env": record.environment(a.device), "result": out, "unclipped_stats": st})


if __name__ == "__main__":
    main()
