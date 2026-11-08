"""Graph Tsetlin machine vs the published numbers of Granmo et al. (arXiv:2507.14874).

Plan: ``plans/graph-tm-reproduction.md``. Only experiments with published numbers we can
rebuild are run:

* G0: three consecutive A's (Sec. 3.1): 100% test accuracy, depth 2, 4 clauses, 4 epochs.
* G1: MNIST, disconnected nodes (Table 3 / 16): GraphTM 98.42 +- 0.05, CoTM 98.93 +- 0.02.
* G2: Fashion-MNIST, disconnected nodes: GraphTM 89.49 +- 0.09, CoTM 91.05 +- 0.09.

Skipped, with notes in the plan: CIFAR-10 (preprocessing cannot be rebuilt: 2,500 symbols) and
the multivalue noisy XOR (curves only, no numbers).

Usage::

    python run_graphtm.py prepare                       # Booleanise once (.cache/data/graphtm_*.npz)
    python run_graphtm.py run G1 --model graph --seed 0 # one record in results/graphtm/
    python run_graphtm.py run G1 --model cotm --seed 0 --limit 1000 --epochs 1 --smoke

The paper's statistic is the mean test accuracy over the final 5 of 30 epochs; each record
keeps the per-epoch test accuracy, so that statistic, the final epoch and seed means are all
derived from the records (report.py), never typed.
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import DATA, environment, git_sha, write_json  # noqa: E402

import torchtsetlin as tt  # noqa: E402
from torchtsetlin.data import GraphBatch, SymbolTable  # noqa: E402

RESULTS = HERE.parents[1] / "results"
RAW = DATA / "raw"

# Table 16 (G1, G2) and Sec. 3.1 (G0). Open choices are fixed here, before any run (plan §3).
CONFIGS = {
    "G0": dict(C=4, T=40, s=2.0, depth=2, epochs=4, batch=25, hv=None,
               reported={"graph": {"accuracy": 100.0}}),
    "G1": dict(C=2500, T=3125, s=10.0, depth=1, epochs=30, batch=16, patch=10, hv=128, hv_bits=2,
               budget=None, reported={"graph": {"accuracy": 98.42, "pm": 0.05},
                                      "cotm": {"accuracy": 98.93, "pm": 0.02}}),
    "G2": dict(C=40000, T=15000, s=10.0, depth=1, epochs=30, batch=16, patch=3, hv=128, hv_bits=2,
               budget=None, reported={"graph": {"accuracy": 89.49, "pm": 0.09},
                                      "cotm": {"accuracy": 91.05, "pm": 0.09}}),
}
THERMO_LEVELS = [k * 256 / 9 for k in range(1, 9)]  # G2: 8 uniform thermometer levels (assumed)


# ------------------------------------------------------------------------------------ data
def _idx(path: Path) -> np.ndarray:
    with gzip.open(path) as f:
        b = f.read()
    if int.from_bytes(b[2:3], "big") != 8:
        raise ValueError(path)
    dims = b[3]
    shape = [int.from_bytes(b[4 + 4 * i: 8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def prepare() -> None:
    """Booleanise MNIST (pixel > 75, the reference demo's stationary threshold) and
    Fashion-MNIST (8-level thermometer) once, as (N, Z, 28, 28) bool."""
    d = np.load(RAW / "mnist.npz")
    np.savez_compressed(DATA / "graphtm_G1.npz", x_train=(d["x_train"] > 75)[:, None],
                        y_train=d["y_train"].astype(np.int64), x_test=(d["x_test"] > 75)[:, None],
                        y_test=d["y_test"].astype(np.int64))
    out = {}
    for split, stem in (("train", "train"), ("test", "t10k")):
        x = _idx(RAW / f"fashion-{stem}-images-idx3-ubyte.gz").astype(np.float32)
        out[f"x_{split}"] = np.stack([x >= t for t in THERMO_LEVELS], axis=1)  # (N, 8, 28, 28)
        out[f"y_{split}"] = _idx(RAW / f"fashion-{stem}-labels-idx1-ubyte.gz").astype(np.int64)
    np.savez_compressed(DATA / "graphtm_G2.npz", **out)
    for k in ("G1", "G2"):
        z = np.load(DATA / f"graphtm_{k}.npz")
        print(k, {n: (z[n].shape, round(float(z[n].mean()), 4)) for n in z.files})


def image_symbols(Z: int, k: int, grid: int):
    """The reference demos' node symbols: one per patch pixel (and level), plus one row and one
    column symbol per window position (``MNISTConvolutionDemo.py``)."""
    pix = [f"p{z}_{i}" for z in range(Z) for i in range(k * k)]
    pos = [f"C:{i}" for i in range(grid)] + [f"R:{i}" for i in range(grid)]
    return pix, pos


def image_graphs(x: torch.Tensor, patch: int, table: SymbolTable, chunk: int = 2000) -> GraphBatch:
    """(N, Z, H, W) bool images -> disconnected graphs, one node per patch at stride 1, node
    features = OR of the hypervectors of the node's symbols."""
    N, Z, H, W = x.shape
    grid = H - patch + 1
    P = grid * grid
    pix, _ = image_symbols(Z, patch, grid)
    dev = x.device
    hpix = torch.stack([table.vector([s]) for s in pix]).to(dev, torch.float16)  # (Z*k*k, dim)
    q, r = torch.div(torch.arange(P), grid, rounding_mode="floor"), torch.arange(P) % grid
    pos = torch.stack([table.vector([f"C:{int(a)}", f"R:{int(b)}"]) for a, b in zip(q, r)]).to(dev)
    feats = torch.empty(N, P, table.dim, dtype=torch.bool, device=dev)
    for i in range(0, N, chunk):
        u = torch.nn.functional.unfold(x[i:i + chunk].half(), patch)  # (n, Z*k*k, P), channel-major
        feats[i:i + chunk] = (torch.matmul(u.transpose(1, 2), hpix) > 0) | pos
    g = GraphBatch(feats, torch.full((N,), P, dtype=torch.long, device=dev), None, None,
                   torch.zeros(N, dtype=torch.long, device=dev), None, 1, 1).pad()
    g.full_nodes = True
    return g


def aaa_task(n: int, rng: np.random.Generator, noise: float):
    """G0: five-letter sequences over A-E, one third with 'AAA' planted (the paper's balance,
    13,330 of 40,000); the label says whether 'AAA' occurs. Training labels get ``noise``."""
    seqs, ys = [], []
    for _ in range(n):
        s = list(rng.choice(list("ABCDE"), 5))
        if rng.random() < 1 / 3:
            k = int(rng.integers(0, 3))
            s[k:k + 3] = ["A"] * 3
        y = int("AAA" in "".join(s))
        if rng.random() < noise:
            y = 1 - y
        seqs.append(s)
        ys.append(y)
    return seqs, np.array(ys)


# ------------------------------------------------------------------------------------ runs
def run(exp: str, model_kind: str, seed: int, epochs=None, limit=None, smoke=False) -> Path:
    cfg = dict(CONFIGS[exp])
    epochs = int(epochs or cfg["epochs"])
    dev = torch.device("cuda")
    tt.seed_everything(seed)
    env = environment()
    t_data = time.time()
    if exp == "G0":
        rng = np.random.default_rng(seed)
        s_tr, y_tr = aaa_task(32000, rng, 0.01)
        s_te, y_te = aaa_task(8000, rng, 0.0)
        sym = SymbolTable(list("ABCDE"), encoding="one_hot")
        g_tr = GraphBatch.from_sequences(s_tr, sym).pad(5, 8).to(dev)
        g_te = GraphBatch.from_sequences(s_te, sym).pad(5, 8).to(dev)
        y_tr, y_te = torch.as_tensor(y_tr, device=dev), torch.as_tensor(y_te, device=dev)
        n_classes, data_note = 2, "five letters A-E, 1/3 with AAA planted, 32k/8k, 1% train noise"
        m = tt.GraphTsetlinMachine(5, 2, cfg["C"], T=cfg["T"], s=cfg["s"], depth=2, message_encoding="one_hot",
                                   n_edge_types=2, edge_type_names=["left", "right"])
        x_tr, x_te = g_tr, g_te
    else:
        z = np.load(DATA / f"graphtm_{exp}.npz")
        xi_tr = torch.as_tensor(z["x_train"], device=dev)
        xi_te = torch.as_tensor(z["x_test"], device=dev)
        y_tr = torch.as_tensor(z["y_train"], device=dev)
        y_te = torch.as_tensor(z["y_test"], device=dev)
        if limit:
            xi_tr, y_tr = xi_tr[:limit], y_tr[:limit]
        n_classes, Z = 10, xi_tr.shape[1]
        if model_kind == "graph":
            grid = 28 - cfg["patch"] + 1
            pix, pos = image_symbols(Z, cfg["patch"], grid)
            table = SymbolTable(pos + pix, dim=cfg["hv"], bits=cfg["hv_bits"], encoding="random", seed=seed)
            x_tr, x_te = image_graphs(xi_tr, cfg["patch"], table), image_graphs(xi_te, cfg["patch"], table)
            data_note = (f"{len(pix) + len(pos)} symbols in a {cfg['hv']}-bit hypervector with "
                         f"{cfg['hv_bits']} bits each (random positions, seed {seed})")
            m = tt.GraphTsetlinMachine(cfg["hv"], 10, cfg["C"], T=cfg["T"], s=cfg["s"], depth=1,
                                       max_included_literals=cfg["budget"])
        else:
            x_tr, x_te = xi_tr, xi_te
            data_note = "raw Boolean images, thermometer-coded patch coordinates (position_encoding)"
            m = tt.ConvCoalescedTsetlinMachine(10, cfg["C"], cfg["T"], cfg["s"], patch_size=cfg["patch"],
                                               input_shape=tuple(xi_tr.shape[1:]),
                                               max_included_literals=cfg["budget"])
    m = m.to(dev)
    t_data = time.time() - t_data
    n_tr, bs = len(y_tr), cfg["batch"]
    log = []
    for ep in range(epochs):
        m.train()
        torch.cuda.synchronize()
        t0 = time.time()
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            m.update(x_tr[idx], y_tr[idx])
        torch.cuda.synchronize()
        t_train = time.time() - t0
        m.eval()
        t0 = time.time()
        pred = m.predict(x_te)
        torch.cuda.synchronize()
        t_test = time.time() - t0
        acc = float((pred == y_te).float().mean()) * 100
        log.append({"epoch": ep + 1, "test_accuracy": acc, "train_s": t_train, "test_s": t_test})
        print(f"{exp} {model_kind} seed {seed} epoch {ep + 1}/{epochs}: test {acc:.2f}%  "
              f"train {t_train:.1f}s  test {t_test:.1f}s", flush=True)
    accs = [r["test_accuracy"] for r in log]
    last5 = accs[-5:]
    rec = {
        "experiment": exp, "model": model_kind, "seed": seed, "smoke": bool(smoke),
        "argv": sys.argv, "git_sha": git_sha(), "torchtsetlin": tt.__version__,
        "torch": torch.__version__, "env": env, "env_end_loadavg": list(__import__("os").getloadavg()),
        "config": {k: v for k, v in cfg.items() if k != "reported"} | {"epochs": epochs, "limit": limit},
        "data": data_note, "data_build_s": t_data, "n_train": n_tr, "n_test": len(y_te),
        "model_repr": repr(m).splitlines()[0][:400],
        "reported": cfg["reported"].get(model_kind),
        "per_epoch": log,
        "summary": {"final_epoch": accs[-1], "mean_last5": float(np.mean(last5)),
                    "sd_last5": float(np.std(last5)), "best_epoch_logged_only": max(accs)},
    }
    sub = "graphtm_smoke" if smoke else "graphtm"
    path = RESULTS / sub / f"{exp}_{model_kind}_seed{seed}.json"
    write_json(path, rec)
    print("wrote", path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    r = sp.add_parser("run")
    r.add_argument("exp", choices=sorted(CONFIGS))
    r.add_argument("--model", choices=["graph", "cotm"], default="graph")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--epochs", type=int)
    r.add_argument("--limit", type=int, help="training examples (smoke)")
    r.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    else:
        run(a.exp, a.model, a.seed, a.epochs, a.limit, a.smoke)


if __name__ == "__main__":
    main()
