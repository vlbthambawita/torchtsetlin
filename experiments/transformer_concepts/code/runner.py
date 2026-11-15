"""Shared harness for Part II/III experiments: data, arms, training loop, JSON records.

Rules: every run writes one JSON record (``results/<stage>/<key>.json``); a run whose record
exists is skipped (resumable). Pair columns are gathered per mini-batch (DECISIONS D8). Test sets
are read only when ``test=True``, and every read is appended to ``results/test_access_log.jsonl``.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from experiments.transformer_concepts.code import tasks
from experiments.transformer_concepts.code.pairs import (
    n_buckets,
    offset_bucket,
    pair_tokens,
    token_columns,
)

import torchtsetlin as tt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLIT_SIZES = {"train": 20000, "valid": 5000, "test": 10000}


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


# ----------------------------------------------------------------------------- tasks
@dataclass
class TaskSpec:
    name: str
    V: int
    n_classes: int
    d: Optional[int] = None


def task_spec(task: str, L: int, V: Optional[int] = None) -> TaskSpec:
    if task == "T1":
        return TaskSpec(task, V or 16, 2)
    if task == "T2":
        return TaskSpec(task, 16 + 4, 4)
    if task == "T3":
        return TaskSpec(task, V or L + 16, 2)
    if task in ("T4", "T4n"):
        return TaskSpec(task, V or 16, 4, d=max(1, L // 4))
    if task == "T5":
        return TaskSpec(task, V or 16, 2)
    if task == "T6":
        return TaskSpec(task, V or 16, 4)
    if task == "T7":
        return TaskSpec(task, V or 16, 2)
    raise ValueError(task)


def make_data(task: str, L: int, split: str, n: Optional[int] = None, V: Optional[int] = None,
              base_seed: int = 314159):
    spec = task_spec(task, L, V)
    n = n or SPLIT_SIZES[split]
    g = torch.Generator().manual_seed(stable_seed(base_seed, task, L, spec.V, split, n))
    if task == "T1":
        x, y = tasks.ordered_pair(n, L, spec.V, g)
    elif task == "T2":
        x, y = tasks.associative_recall(n, L, 16, 4, g)
    elif task == "T3":
        x, y = tasks.repeat_detect(n, L, spec.V, g)
    elif task == "T4":
        x, y = tasks.needle(n, L, spec.V, spec.d, spec.n_classes, g, distractors=True)
    elif task == "T4n":
        x, y = tasks.needle(n, L, spec.V, spec.d, spec.n_classes, g, distractors=False)
    elif task == "T5":
        x, y = tasks.bag_control(n, L, spec.V, g)
    elif task == "T6":
        x, y = tasks.absolute_position(n, L, spec.V, 3, spec.n_classes, g)
    elif task == "T7":
        x, y = tasks.presence(n, L, spec.V, g)
    else:
        raise ValueError(task)
    return x, y, spec


# ----------------------------------------------------------------------------- arms
def onehot_channels(x, V):
    return torch.nn.functional.one_hot(x, V).bool().transpose(1, 2).contiguous()


def build_graphs(x: torch.Tensor, V: int, kind: str):
    """GraphBatch with one node per token; ``chain`` (left/right edges) or ``complete`` (edge
    j -> i typed by the offset bucket of j - i)."""
    N, L = x.shape
    feats = torch.nn.functional.one_hot(x, V).bool()
    if kind == "chain":
        k = torch.arange(L)
        e = torch.cat([torch.stack([k[1:], k[:-1]], 1), torch.stack([k[:-1], k[1:]], 1)])
        t = torch.cat([torch.zeros(L - 1, dtype=torch.long), torch.ones(L - 1, dtype=torch.long)])
        n_types = 2
    else:
        j, i = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        keep = j != i
        src, dst = j[keep], i[keep]
        e = torch.stack([src, dst], 1)
        bm = n_buckets(L)
        t = offset_bucket(src - dst) + bm  # 0..2bm
        t = torch.where(t > bm, t - 1, t)  # remove the unused zero offset -> 0..2bm-1
        n_types = 2 * bm
    gb = tt.data.GraphBatch.from_graphs([feats[b] for b in range(N)], [e] * N, [t] * N, n_edge_types=n_types)
    return gb.pad(), n_types


@dataclass
class Arm:
    name: str
    build: Callable  # (spec, L, M, T, s) -> model
    feats: Callable  # (x_batch, idx_batch, ctx) -> model input
    kind: str = "tensor"  # or "graph"


def _conv(kernel, pe):
    def b(spec, L, M, T, s, **kw):
        return tt.Conv1dTsetlinMachine(spec.n_classes, M, T, s, kernel_size=kernel, position_encoding=pe,
                                       weighted=True, **kw)
    return b


def _pairs(**pk):
    def f(x, idx, ctx):
        perm = ctx["perm"][idx] if pk.get("rel_perm") else None
        kw = {k: v for k, v in pk.items() if k != "rel_perm"}
        return pair_tokens(x, ctx["V"], rel_perm=perm, **kw)
    return f


def _graph_build(kind):
    def b(spec, L, M, T, s, **kw):
        n_types = 2 if kind == "chain" else 2 * n_buckets(L)
        return tt.GraphTsetlinMachine(None, spec.n_classes, M * spec.n_classes // 2, T, s, depth=2,
                                      message_size=256, message_bits=2, n_edge_types=n_types, **kw)
    return b


ARMS: Dict[str, Arm] = {
    "B0_flat": Arm("B0_flat", lambda spec, L, M, T, s, **kw: tt.TsetlinMachine(None, spec.n_classes, M, T, s, weighted=True, **kw),
                   lambda x, idx, ctx: torch.nn.functional.one_hot(x, ctx["V"]).bool().reshape(x.shape[0], -1)),
    "B1_conv_k1": Arm("B1_conv_k1", _conv(1, False), lambda x, idx, ctx: onehot_channels(x, ctx["V"])),
    "B2_conv_k3_pe": Arm("B2_conv_k3_pe", _conv(3, True), lambda x, idx, ctx: onehot_channels(x, ctx["V"])),
    "B3_conv_kd1": Arm("B3_conv_kd1", lambda spec, L, M, T, s, **kw: _conv(spec.d + 1, False)(spec, L, M, T, s, **kw),
                       lambda x, idx, ctx: onehot_channels(x, ctx["V"])),
    "P_onehot": Arm("P_onehot", _conv(1, False), lambda x, idx, ctx: token_columns(x, ctx["V"], "onehot")),
    "P_therm": Arm("P_therm", _conv(1, False), lambda x, idx, ctx: token_columns(x, ctx["V"], "thermometer")),
    "P_binary": Arm("P_binary", _conv(1, False), lambda x, idx, ctx: token_columns(x, ctx["V"], "binary")),
    "P_gray": Arm("P_gray", _conv(1, False), lambda x, idx, ctx: token_columns(x, ctx["V"], "gray")),
    "B4_graph_chain": Arm("B4_graph_chain", _graph_build("chain"), None, kind="graph"),
    "A1_pair_w1": Arm("A1_pair_w1", _conv(1, False), _pairs(window=1, last_bit=True)),
    "A2_pair_w2": Arm("A2_pair_w2", _conv(1, False), _pairs(window=2, last_bit=True)),
    "A3_pair_bucket": Arm("A3_pair_bucket", _conv(1, False), _pairs(window=2, last_bit=True, rel="bucket")),
    "A4_graph_complete": Arm("A4_graph_complete", _graph_build("complete"), None, kind="graph"),
    "A2_self": Arm("A2_self", _conv(1, False), _pairs(window=2, last_bit=True, self_only=True)),
    "A2_norel": Arm("A2_norel", _conv(1, False), _pairs(window=2, last_bit=True, rel="none")),
    "A2_abs": Arm("A2_abs", _conv(1, False), _pairs(window=2, last_bit=True, rel="abs")),
    "A2_shuf": Arm("A2_shuf", _conv(1, False), _pairs(window=2, last_bit=True, rel_perm=True)),
}


def make_ctx(arm: Arm, x: torch.Tensor, spec: TaskSpec, device, seed_tag) -> dict:
    ctx = {"V": spec.V}
    L = x.shape[1]
    if arm.name == "A2_shuf":
        g = torch.Generator().manual_seed(stable_seed("perm", seed_tag))
        ctx["perm"] = torch.argsort(torch.rand(x.shape[0], L * L, generator=g), dim=1).to(torch.int32).to(device)
    if arm.kind == "graph":
        gb, _ = build_graphs(x, spec.V, "chain" if arm.name == "B4_graph_chain" else "complete")
        ctx["graphs"] = gb.to(device)
    return ctx


def batch_input(arm: Arm, x_dev, idx, ctx):
    if arm.kind == "graph":
        return ctx["graphs"][idx]
    return arm.feats(x_dev[idx], idx, ctx)


# ----------------------------------------------------------------------------- training
def predict_all(model, arm, x_dev, ctx, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, x_dev.shape[0], bs):
            idx = torch.arange(k, min(k + bs, x_dev.shape[0]), device=x_dev.device)
            out.append(model.predict(batch_input(arm, x_dev, idx, ctx)))
    return torch.cat(out)


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def log_test_access(record_key: str, n: int):
    with open(RESULTS / "test_access_log.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "record": record_key, "n": n}) + "\n")


def run(stage: str, key: str, task: str, L: int, arm_name: str, s: float, seed: int, *,
        M: int = 100, T: int = 25, epochs: int = 40, patience: int = 8, batch: int = 32,
        n_train: Optional[int] = None, n_valid: Optional[int] = None, test: bool = False,
        V: Optional[int] = None, device: str = "cuda", model_kw: Optional[dict] = None,
        save_model: bool = False, extra: Optional[dict] = None) -> dict:
    out_dir = RESULTS / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    arm = ARMS[arm_name]
    dev = torch.device(device)
    xtr, ytr, spec = make_data(task, L, "train", n_train, V)
    xva, yva, _ = make_data(task, L, "valid", n_valid, V)
    ctx_tr = make_ctx(arm, xtr, spec, dev, (task, L, "train", len(xtr)))
    ctx_va = make_ctx(arm, xva, spec, dev, (task, L, "valid", len(xva)))
    xtr_d, ytr_d, xva_d, yva_d = xtr.to(dev), ytr.to(dev), xva.to(dev), yva.to(dev)
    torch.manual_seed(seed)
    model = arm.build(spec, L, M, T, s, **(model_kw or {})).to(dev)
    hist, best, best_ep, best_state, t_train, t0_all = [], -1.0, -1, None, 0.0, time.time()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k in range(0, len(xtr), batch):
            idx = perm[k:k + batch]
            model.update(batch_input(arm, xtr_d, idx, ctx_tr), ytr_d[idx])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        acc = (predict_all(model, arm, xva_d, ctx_va) == yva_d).float().mean().item()
        hist.append(round(acc, 5))
        if acc > best:
            best, best_ep, best_state = acc, ep, snapshot(model)
        if ep - best_ep >= patience or best >= 1.0:
            break
    model.load_state_dict(best_state)
    rec = dict(stage=stage, key=key, task=task, L=L, V=spec.V, n_classes=spec.n_classes, d=spec.d,
               arm=arm_name, s=s, seed=seed, M=M, T=T, batch=batch, epochs_max=epochs, patience=patience,
               n_train=len(xtr), n_valid=len(xva), val_curve=hist, best_val=best, best_epoch=best_ep,
               epochs_run=len(hist), train_seconds=round(t_train, 2), wall_seconds=round(time.time() - t0_all, 2),
               device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
               torchtsetlin=tt.__version__, model_kw=model_kw or {}, **(extra or {}))
    if test:
        xte, yte, _ = make_data(task, L, "test", None, V)
        ctx_te = make_ctx(arm, xte, spec, dev, (task, L, "test", len(xte)))
        pred = predict_all(model, arm, xte.to(dev), ctx_te).cpu()
        log_test_access(key, len(xte))
        rec["test_acc"] = (pred == yte).float().mean().item()
        np.save(out_dir / f"{key}.pred.npy", pred.numpy().astype(np.int8))
    if save_model:
        torch.save(model.state_dict(), out_dir / f"{key}.pt")
    path.write_text(json.dumps(rec, indent=1))
    return rec
