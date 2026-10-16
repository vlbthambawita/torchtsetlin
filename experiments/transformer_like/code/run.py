"""One spec -> one record. Encoders: ``--encoders``; head/baseline trials: ``--spec file.json``.

    python -m experiments.transformer_like.code.run --encoders --seed 17 --device cuda
    python -m experiments.transformer_like.code.run --spec results/specs/<id>.json --device cuda
"""

from __future__ import annotations

import argparse
import gzip
import json
import resource
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from . import data as D
from .common import ART, RESULTS, provenance, read_json, resolve_T, seed_torch, sha256_file, sync, write_json
from .features import (ATTENTION_MODES, N_FEATURES, NEEDS_ENCODERS, attention_diagnostics, cached_transform,
                       constant_column_fraction)
from .models import TMQKAttention
from .patch_training import encoder_diagnostics, fit_encoders
from .tm_adapter import TMAdapter
from .training import fit_classifier, metrics_from_votes

ARMS = {
    "B0": {"mode": "image_flat"}, "B0M": {"mode": "image_flat"},
    "B1": {"mode": "image_conv", "stride": 1}, "B1s7": {"mode": "image_conv", "stride": 7},
    "N0": {"mode": "qk"}, "A0": {"mode": "uniform"}, "A1": {"mode": "self_only"},
    "A2": {"mode": "direct"}, "A2M": {"mode": "direct"}, "A3": {"mode": "shuffled_keys"},
}
INTERVENTIONS = ("uniform", "self_only", "shuffled_keys")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def encoder_path(seed: int, tag: str = "") -> Path:
    return ART / "encoders" / f"seed{seed}{tag}.pt"


def load_encoders(path: Path, device: str):
    o = torch.load(path, map_location="cpu", weights_only=False)
    return TMAdapter.from_state(o["q"], device), TMAdapter.from_state(o["k"], device), o


def run_encoders(seed: int, device: str, manifest: str = D.MAIN, tag: str = "", cfg: Optional[Dict] = None) -> Dict:
    path = encoder_path(seed, tag)
    rec_path = RESULTS / "encoders" / f"seed{seed}{tag}.json"
    tr = D.load_split(manifest, "train")
    va = D.load_split(manifest, "valid")
    t0 = time.time()
    q, k, info = fit_encoders(tr.images, tr.y, seed, device, cfg)
    diag = encoder_diagnostics(q, k, va.images, va.y, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"q": q.state_dict(), "k": k.state_dict(), "info": info, "seed": seed, "manifest": manifest}, path)
    rec = {"kind": "encoders", "seed": seed, "tag": tag, "manifest": manifest,
           "split_sha": D.read_manifest(manifest)["sha256"], "info": info, "diagnostics": diag,
           "checkpoint": str(path.relative_to(ART.parent)), "checkpoint_sha256": sha256_file(path),
           "q_sha256": q.state_hash(), "k_sha256": k.state_hash(), "wall_s": time.time() - t0,
           "peak_rss_gb": peak_rss_gb(), "status": "completed", **provenance(device)}
    if str(device).startswith("cuda"):
        rec["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
    write_json(rec_path, rec)
    return rec


def head_adapter(arm: str, head: Dict, device: str) -> TMAdapter:
    a = ARMS[arm]
    M = int(head["M"])
    T = resolve_T(M, head["T_ratio"]) if "T" not in head else int(head["T"])
    kw = dict(weighted=bool(head.get("weighted", False)), feedback_mode=head.get("feedback_mode", "batch"),
              n_states=int(head.get("n_states", 128)), device=device)
    if a["mode"] == "image_conv":
        return TMAdapter(None, M, T, head["s"], kind="conv", patch_size=7, stride=a["stride"], **kw)
    return TMAdapter(N_FEATURES[a["mode"]], M, T, head["s"], **kw)


def run_id_of(spec: Dict) -> str:
    h = spec["head"]
    rid = f"{spec['arm']}_M{h['M']}_r{h['T_ratio']}_s{h['s']}"
    if ARMS[spec["arm"]]["mode"] in ("qk", "shuffled_keys"):
        rid += f"_t{spec.get('temperature', 1.0)}"
    if h.get("weighted"):
        rid += "_w"
    if h.get("feedback_mode", "batch") != "batch":
        rid += "_seq"
    return rid + f"_seed{spec['seed']}" + spec.get("tag", "")


def _write_predictions(path: Path, ids, y, votes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = votes.cpu().numpy()
    with gzip.open(path, "wt") as f:
        f.write("image_id,label,pred," + ",".join(f"score_{c}" for c in range(10)) + "\n")
        for i, yy, row in zip(ids, y, v):
            f.write(f"{int(i)},{int(yy)},{int(row.argmax())}," + ",".join(f"{x:g}" for x in row) + "\n")


def run_trial(spec: Dict, device: str) -> Dict:
    stage, arm = spec["stage"], spec["arm"]
    mode = ARMS[arm]["mode"]
    rid = spec.get("run_id") or run_id_of(spec)
    out_path = RESULTS / stage / f"{rid}.json"
    rec: Dict = {"run_id": rid, "stage": stage, "arm": arm, "mode": mode, "spec": spec, "status": "running",
                 **provenance(device)}
    try:
        manifest = spec.get("manifest", D.MAIN)
        tr, va = D.load_split(manifest, "train"), D.load_split(manifest, "valid")
        rec["manifest"], rec["split_sha"] = manifest, D.read_manifest(manifest)["sha256"]
        q = k = None
        enc_time = 0.0
        if mode in NEEDS_ENCODERS:
            epath = ART.parent / spec["encoders"]
            q, k, eo = load_encoders(epath, device)
            rec["encoders"] = {"path": spec["encoders"], "sha256": sha256_file(epath),
                               "q_sha256": q.state_hash(), "k_sha256": k.state_hash()}
            enc_time = float(sum(eo["info"]["time_s"].values()))
        temp = float(spec.get("temperature", 1.0))
        sync(device)
        t0 = time.time()
        Xtr, ktr, hit_tr = cached_transform(tr.images, tr.ids, mode, q, k, temp, device)
        Xva, kva, hit_va = cached_transform(va.images, va.ids, mode, q, k, temp, device)
        sync(device)
        feat_time = time.time() - t0
        rec["features"] = {"train_key": ktr, "valid_key": kva, "cache_hit": [hit_tr, hit_va],
                           "seconds": feat_time, "n_features": int(Xtr[0].numel())}
        ytr = torch.as_tensor(tr.y).to(device)
        yva = torch.as_tensor(va.y).to(device)
        rec["head_torch_seed"] = seed_torch(spec["seed"], "head")
        head = head_adapter(arm, spec["head"], device)
        if mode in ("qk", "shuffled_keys", "direct"):
            # the head never changes the frozen encoders (spec test 27)
            qh, kh = q.state_hash(), k.state_hash()
        fit = fit_classifier(head, Xtr, ytr, Xva, yva, max_epochs=spec.get("max_epochs", 50),
                             patience=spec.get("patience", 10), batch_size=spec.get("batch_size", 32),
                             log=lambda s: print(s, flush=True))
        if q is not None and (q.state_hash() != qh or k.state_hash() != kh):
            raise RuntimeError("encoder state changed during head training")
        model = TMQKAttention(head, mode, q, k, temp, meta={"run_id": rid, "arm": arm})
        ck = ART / "runs" / stage / f"{rid}.pt"
        ck_bytes = model.save(ck)
        va_votes = head.votes(Xva)
        rec["fit"] = fit
        rec["valid"] = metrics_from_votes(va_votes, yva)
        assert abs(rec["valid"]["accuracy"] - fit["best_val_accuracy"]) < 1e-12
        rec["sizes"] = {"automata": model.n_automata(), "stored_bytes": model.stored_bytes(),
                        "logical_bytes_8bit": model.n_automata()["total"], "checkpoint_bytes": ck_bytes,
                        "head_state": head.state_manifest(),
                        "feature_cache_bytes_train_packed": int(np.ceil(Xtr[0].numel() / 8) * len(tr))}
        P = 1
        if mode == "image_conv":
            P = int(np.prod(head.model._grid(28, 28)))
        lit = head.n_automata() * P
        if q is not None:
            lit += 16 * (q.n_automata() + k.n_automata())
        rec["sizes"]["literals_per_image"] = int(lit)
        rec["timings"] = {"encoders_s": enc_time, "features_s": feat_time, "head_train_s": fit["train_time_s"],
                          "total_train_s": enc_time + feat_time + fit["train_time_s"]}
        rec["checkpoint"] = str(ck.relative_to(ART.parent))
        rec["checkpoint_sha256"] = sha256_file(ck)
        _write_predictions(RESULTS / stage / "predictions" / f"{rid}_valid.csv.gz", va.ids, va.y, va_votes)
        if mode in ATTENTION_MODES:
            rec["constant_context_bit_fraction_train"] = constant_column_fraction(Xtr)
            rec["attention_valid"] = attention_diagnostics(va.images, mode, q, k, temp, device)
        if spec.get("interventions") and mode == "qk":
            rec["interventions_valid"] = {m: metrics_from_votes(model.votes_array(va.images, m), yva)["accuracy"]
                                          for m in INTERVENTIONS}
        del Xtr
        if spec.get("test"):
            te = D.load_split(manifest, "test", unseal=True, purpose=f"{stage}/{rid}")
            yte = torch.as_tensor(te.y).to(device)
            te_votes = model.votes_array(te.images)
            rec["test"] = metrics_from_votes(te_votes, yte)
            _write_predictions(RESULTS / stage / "predictions" / f"{rid}_test.csv.gz", te.ids, te.y, te_votes)
            if mode in ATTENTION_MODES:
                rec["attention_test"] = attention_diagnostics(te.images, mode, q, k, temp, device)
            if spec.get("interventions") and mode == "qk":
                rec["interventions_test"] = {m: metrics_from_votes(model.votes_array(te.images, m), yte)["accuracy"]
                                             for m in INTERVENTIONS}
        rec["peak_rss_gb"] = peak_rss_gb()
        if str(device).startswith("cuda"):
            rec["peak_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec["status"] = "completed"
    except torch.cuda.OutOfMemoryError:
        rec["status"], rec["error"] = "failed", "OOM\n" + traceback.format_exc()
    except Exception:
        rec["status"], rec["error"] = "failed", traceback.format_exc()
    write_json(out_path, rec)
    print(f"[{rec['status']}] {rid} -> {out_path}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--encoders", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--manifest", default=D.MAIN)
    ap.add_argument("--encoder-cfg", default=None, help="JSON overrides for the encoder config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.encoders:
        cfg = json.loads(a.encoder_cfg) if a.encoder_cfg else None
        rec = run_encoders(a.seed, a.device, a.manifest, a.tag, cfg)
        print(json.dumps({k: rec["diagnostics"][k]["patch_accuracy"] for k in ("q", "k")}))
        return
    rec = run_trial(read_json(a.spec), a.device)
    if rec["status"] != "completed":
        print(rec.get("error"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
