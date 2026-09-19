"""E2: compositional synthetic tasks where the right answer is known in advance.

CIFAR-10 classes may simply not be concepts that reward a second layer, in which case a null
result there says nothing about the architecture. These two tasks are constructed so that
the expressivity of each arm can be worked out on paper, which makes them a test of the
implementation as much as of the idea.

Setup: a 32x32 Boolean image containing a SQUARE and/or a CROSS at random non-overlapping
positions, plus salt noise. Layer 1 uses 4x4 patches, so no single patch can ever contain
both shapes.

TASK "xor" -- label = (square present) XOR (cross present).
  One layer computes threshold( sum_j w_j * OR_p conj_j(patch_p) ): its features are
  patch-local presence detectors, and "cross absent everywhere" is an AND over patches that
  no single clause can form. Writing z_A, z_B for the two presences, the score is a linear
  threshold in (z_A, z_B), and XOR is not linearly separable -- w_A > t and w_B > t force
  w_A + w_B > t. Adding position literals does not help: the same argument runs through with
  z_{A,top}, z_{A,bot}, ... Best achievable for one layer is the best linear separator, 75%.
  Both stacked variants CAN express it -- including flat-stack, because negating a globally
  OR-pooled bit does mean "absent everywhere".
  Expected: single ~75%, flat-stack ~100%, mctm ~100%.

TASK "near" -- both shapes always present; label = are they close together.
  Shapes sit on 16 equally spaced slots around a circle, and by rotational symmetry every
  slot has the same number of near and far partners -- so the marginal position of each
  shape is EXACTLY the same in both classes and only the JOINT separation carries the label. One layer is out: proximity is not of the form
  f(pos_A) + g(pos_B) > t that its additive vote gives, and no 4x4 patch sees both shapes.
  flat-stack is out: it globally OR-pools before the second stage and keeps no position at
  all. The spatial MCTM can: a layer-2 clause "channel A fires here AND channel B fires here"
  over a local window, OR-pooled over positions, is exactly "somewhere the two are close".
  Expected: single ~50%, flat-stack ~50%, mctm well above.

"xor" asks whether depth helps at all; "near" asks whether keeping the SPATIAL map is what
helps, rather than the trivial global-OR stack.
"""
from __future__ import annotations

import argparse, json, math, os
from typing import Tuple

import torch
from torch import Tensor

import torchtsetlin as tt
from mctm import MCTM, clause_composition, conv_clause_maps, firing_stats, or_pool
from run import RESULTS, accuracy, now

SQUARE = torch.tensor([[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]], dtype=torch.bool)
CROSS = torch.tensor([[0, 1, 1, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 1, 1, 0]], dtype=torch.bool)

SIZE, SH = 32, 4
N_SLOTS, RING_R = 16, 11.0
NEAR_GAP, FAR_GAP = (2, 3), (7, 8)   # slot separations, measured around the ring

# Shapes are placed on N_SLOTS equally spaced positions around a circle. The ring is what
# makes "near" clean: by rotational symmetry every slot has exactly the same number of near
# partners and far partners, so the MARGINAL position of each shape is identical in both
# classes and only the JOINT separation carries the label. Sampling positions freely and
# rejecting on distance does not have this property -- it pulls near-pairs towards the centre,
# and an additive f(pos_A)+g(pos_B) predictor (precisely what one conv layer can realise)
# then reaches 66% without any joint reasoning. Verified by scripts/check_leakage.py.
SLOTS = []
for _k in range(N_SLOTS):
    _a = 2.0 * math.pi * _k / N_SLOTS
    SLOTS.append((int(round(SIZE / 2 + RING_R * math.sin(_a) - SH / 2)),
                  int(round(SIZE / 2 + RING_R * math.cos(_a) - SH / 2))))


def _ring_gap(i: int, j: int) -> int:
    d = abs(i - j) % N_SLOTS
    return min(d, N_SLOTS - d)


def _place(img: Tensor, shape: Tensor, r: int, c: int) -> None:
    img[r : r + SH, c : c + SH] |= shape


def make_task(n: int, task: str, noise: float = 0.01, seed: int = 0,
              device="cuda", return_presence: bool = False):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, SIZE, SIZE, generator=g) < noise
    y = torch.zeros(n, dtype=torch.long)
    presence = torch.zeros(n, 2, dtype=torch.bool)

    def rslot() -> int:
        return int(torch.randint(0, N_SLOTS, (1,), generator=g))

    for i in range(n):
        if task == "xor":
            has_a = bool(torch.randint(0, 2, (1,), generator=g))
            has_b = bool(torch.randint(0, 2, (1,), generator=g))
            y[i] = int(has_a != has_b)
            sa = rslot()
            sb = rslot()
            while sb == sa:
                sb = rslot()
            presence[i, 0], presence[i, 1] = has_a, has_b
            if has_a:
                _place(x[i, 0], SQUARE, *SLOTS[sa])
            if has_b:
                _place(x[i, 0], CROSS, *SLOTS[sb])
        else:  # "near": label = the two shapes are close around the ring
            want_near = bool(torch.randint(0, 2, (1,), generator=g))
            y[i] = int(want_near)
            gaps = NEAR_GAP if want_near else FAR_GAP
            sa = rslot()
            gap = gaps[int(torch.randint(0, len(gaps), (1,), generator=g))]
            sign = 1 if bool(torch.randint(0, 2, (1,), generator=g)) else -1
            sb = (sa + sign * gap) % N_SLOTS
            presence[i, 0], presence[i, 1] = True, True
            _place(x[i, 0], SQUARE, *SLOTS[sa])
            _place(x[i, 0], CROSS, *SLOTS[sb])
    if return_presence:
        return x.to(device), y.to(device), presence.to(device)
    return x.to(device), y.to(device)


# ---- architecture ----------------------------------------------------------------------
P1, S1, C1, T1 = 4, 1, 64, 64.0
POOL = 2
C2, T2 = 128, 128.0
# "xor" needs a GLOBAL view at layer 2 ("B nowhere"), so its patch spans the whole pooled
# map. "near" needs a LOCAL one, so a 3x3 window is OR-pooled over positions as usual.
L2_PATCH = {"xor": None, "near": 5}


def build_l1(in_shape, s, seed, device, n_clauses=C1, pos=False):
    tt.seed_everything(seed)
    return tt.ConvCoalescedTsetlinMachine(2, n_clauses, T1, s, patch_size=P1, stride=S1,
                                          position_encoding=pos, input_shape=in_shape).to(device)


@torch.no_grad()
def build_oracle_l1(in_shape, device, n_clauses: int = 2, seed: int = 0):
    """Layer 1 with two hand-built clauses: an exact SQUARE detector and an exact CROSS
    detector, written straight into the automata.

    This separates the two questions the greedy scheme confounds. Trained on a task it cannot
    represent, layer 1 never learns clean detectors, so a null result for the stack says only
    that greedy training failed. With the oracle the features are perfect by construction, so
    whatever the stack then achieves is a property of the ARCHITECTURE alone.
    """
    tt.seed_everything(seed)
    m = tt.ConvCoalescedTsetlinMachine(2, n_clauses, T1, 5.0, patch_size=P1, stride=S1,
                                       position_encoding=False, input_shape=in_shape).to(device)
    F = int(m.n_features)                     # 1 * P1 * P1
    assert F == P1 * P1, f"oracle assumes a single input plane, got n_features={F}"
    N = m.n_states
    m.ta_state.fill_(N - 1)                   # everything excluded, one step below boundary
    sq = SQUARE.reshape(-1).to(device)
    cr = CROSS.reshape(-1).to(device)
    for clause, patt in ((0, sq), (1, cr)):
        for k in range(F):
            col = k if bool(patt[k]) else F + k     # literal or its negation
            m.ta_state[clause, col] = N
    if n_clauses > 2:
        # Distractor channels, so that layer 2 faces the SAME channel count (and therefore
        # the same literal-space size) as it does with a greedily trained layer 1. Without
        # this control, the oracle's advantage could be read as "fewer channels" rather than
        # "better features".
        from mctm import randomize_clauses
        rest = tt.ConvCoalescedTsetlinMachine(2, n_clauses - 2, T1, 5.0, patch_size=P1,
                                              stride=S1, position_encoding=False,
                                              input_shape=in_shape).to(device)
        randomize_clauses(rest, n_include=4, seed=seed + 31)
        m.ta_state[2:] = rest.ta_state
    m._refresh_include()
    m.eval()
    return m


def train(model, xtr, ytr, xte, yte, epochs, bs, tag, log):
    model.train()
    for ep in range(epochs):
        t0 = now()
        perm = torch.randperm(xtr.shape[0], device=xtr.device)
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i : i + bs]
            model.update(xtr[idx], ytr[idx])
        te = accuracy(model, xte, yte)
        log.append({"epoch": ep + 1, "test_acc": te, "epoch_s": now() - t0})
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  [{tag}] epoch {ep+1}/{epochs} test {te:.4f}", flush=True)
    return log


def run(task, arm, seed, epochs, n_train, n_test, s, bs, device, l1_mode="task") -> dict:
    xtr, ytr = make_task(n_train, task, seed=seed, device=device)
    xte, yte = make_task(n_test, task, seed=seed + 500, device=device)
    in_shape = tuple(xtr.shape[1:])
    rec = {"task": task, "arm": arm, "l1_mode": l1_mode, "seed": seed, "epochs": epochs, "s": s,
           "n_train": n_train, "n_test": n_test, "input_shape": list(in_shape),
           "label_balance": float(ytr.float().mean()), "log": [], "log_l1": []}

    if arm == "single":
        m = build_l1(in_shape, s, seed, device)
        train(m, xtr, ytr, xte, yte, epochs, bs, f"{task}:single", rec["log"])
        rec["n_clauses_total"] = int(m.n_clauses_total)
    else:
        if l1_mode.startswith("oracle"):
            n_cl = int(l1_mode[6:]) if len(l1_mode) > 6 else 2
            l1 = build_oracle_l1(in_shape, device, n_clauses=n_cl, seed=seed)
            rec["l1_test_acc"] = None
        else:
            l1 = build_l1(in_shape, s, seed, device)
            train(l1, xtr, ytr, xte, yte, epochs, bs, f"{task}:L1", rec["log_l1"])
            l1.eval()
            rec["l1_test_acc"] = rec["log_l1"][-1]["test_acc"]
        rec["l1_clauses"] = int(l1.n_clauses_total)
        if arm == "flat-stack":
            htr = torch.cat([l1.evaluate_clauses(xtr[i:i+512]) for i in range(0, n_train, 512)])
            hte = torch.cat([l1.evaluate_clauses(xte[i:i+512]) for i in range(0, n_test, 512)])
            tt.seed_everything(seed + 7)
            head = tt.CoalescedTsetlinMachine(None, 2, C2, T2, s).to(device)
        else:
            htr = or_pool(conv_clause_maps(l1, xtr), POOL)
            hte = or_pool(conv_clause_maps(l1, xte), POOL)
            rec["l1_firing"] = {k: v for k, v in firing_stats(htr).items() if k != "per_channel"}
            k = L2_PATCH[task] or int(htr.shape[2])
            rec["l2_patch"] = k
            tt.seed_everything(seed + 7)
            head = tt.ConvCoalescedTsetlinMachine(2, C2, T2, s, patch_size=k, stride=1,
                                                  position_encoding=False,
                                                  input_shape=tuple(htr.shape[1:])).to(device)
        rec["feature_shape"] = list(htr.shape[1:])
        train(head, htr, ytr, hte, yte, epochs, bs, f"{task}:{arm}", rec["log"])
        rec["composition"] = clause_composition(head)
        rec["n_clauses_total"] = int(l1.n_clauses_total) + int(head.n_clauses_total)

    rec["final_test_acc"] = rec["log"][-1]["test_acc"]
    rec["best_test_acc"] = max(r["test_acc"] for r in rec["log"])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["xor", "near"])
    ap.add_argument("--arms", nargs="+", default=["single", "flat-stack", "mctm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--l1-modes", nargs="+", default=["task", "oracle"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    out = []
    for task in a.tasks:
        for arm in a.arms:
            # the single-layer arm has no layer 1 to vary
            modes = ["task"] if arm == "single" else a.l1_modes
            for l1_mode in modes:
                for seed in a.seeds:
                    print(f"=== {task} / {arm} / L1={l1_mode} / seed {seed} ===", flush=True)
                    rec = run(task, arm, seed, a.epochs, a.n_train, a.n_test, a.s,
                              a.batch_size, a.device, l1_mode)
                    out.append(rec)
                    print(f"  -> best {rec['best_test_acc']:.4f}", flush=True)
    path = a.out or os.path.join(RESULTS, "synthetic.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {path}")
