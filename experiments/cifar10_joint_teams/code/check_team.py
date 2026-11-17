"""Phase-0 checks of what is new here (plan §9; the joint update itself is checked by
joint_wide_tm/code/check_joint.py, T0-T8):

  C2  JointConv3d with the offset off learns bit for bit like Conv3dTsetlinMachine
  C3  mixed 2D + 3D team: the votes a member's update sees = own votes + the others' (exact),
      on every path (CPU, chunked, CUDA events, CUDA graphs)
  C4  GPU encoders = cifar10_batch_clauses' encoders bit for bit; the real members take the fused
      CUDA path (FB-1)
  C5  augmentation: an epoch is the same for any worker count / chunking and on repetition,
      differs between epochs, and telemetry is off

Writes results/checks/check_team.json. Usage: python code/check_team.py [--device cuda:0]
"""

import argparse
import sys
import time

import numpy as np
import torch
from augment import POLICIES, EpochSource, augment_epoch, telemetry_off
from common import CLAUSES, MEMBERS, RESULTS, cbc, load_cifar, new_record, write_json
from encoders import encode_fixed, gpu_encode
from team import JointConv, JointConv3d, JointTeam, build_member, learning_votes

import torchtsetlin as tt

K, CL, IMG, BATCH = 10, 40, 12, 16
KW = dict(weighted=True, max_included_literals=8, boost_true_positive=True, position_encoding=True,
          feedback_mode="batch")
TINY = 1 << 15


def modes(dev):
    out = [("cpu", "cpu", None, False), ("cpu-chunks", "cpu", TINY, False)]
    if torch.device(dev).type == "cuda":
        out += [("cuda-eager-chunks", dev, TINY, False), ("cuda-events", dev, None, False),
                ("cuda-graphs", dev, None, True)]
    return out


def n_graphs(m):
    return sum(isinstance(v, tuple) for v in m.__dict__.get("_graphs", {}).values())


def make(cls, three_d, patch, T, dev, seed, chunk, graphs, joint=True):
    tt.seed_everything(seed)
    kw = dict(KW, **({"max_chunk_elements": chunk} if chunk else {}))
    if three_d:
        m = cls(K, CL, T, 4.0, patch_size=(4, patch, patch), input_shape=(2, 4, IMG, IMG), **kw)
    else:
        m = cls(K, CL, T, 4.0, patch_size=patch, input_shape=(8, IMG, IMG), **kw)
    if joint:
        m.joint_setup(BATCH)
    m = m.to(dev)
    m.cuda_graphs = graphs
    return m


def data(n, seed, dev):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, BATCH, 2, 4, IMG, IMG, generator=g) < 0.4
    y = torch.randint(0, K, (n, BATCH), generator=g)
    return x.to(dev), y.to(dev)          # 3D input; the 2D members see it as 8 planes


def c2(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev):
        x, y = data(8, 1, d)
        plain = make(tt.Conv3dTsetlinMachine, True, 3, 60, d, 7, chunk, graphs, joint=False)
        jm = make(JointConv3d, True, 3, 60, d, 7, chunk, graphs)
        for m in (plain, jm):
            tt.seed_everything(11)
            for i in range(8):
                m.update(x[i], y[i])
        fresh = make(tt.Conv3dTsetlinMachine, True, 3, 60, d, 7, chunk, graphs, joint=False)
        out[label] = {"identical": bool(torch.equal(plain.ta_state, jm.ta_state)
                                        and torch.equal(plain.weights, jm.weights)),
                      "automata_moved": int((plain.ta_state != fresh.ta_state).sum()),
                      "graphs_replayed": n_graphs(jm)}
    ok = all(v["identical"] and v["automata_moved"] > 0 for v in out.values())
    ok = ok and all(v["graphs_replayed"] > 0 for k, v in out.items() if k == "cuda-graphs")
    return ok, out


def c3(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev):
        x, y = data(10, 2, d)
        ms = [make(JointConv, False, 3, 2, d, 3, chunk, graphs), make(JointConv3d, True, 3, 3, d, 4, chunk, graphs),
              make(JointConv, False, 5, 4, d, 5, chunk, graphs)]
        team = JointTeam(["a2d", "b3d", "c2d"], ms, "sum")
        xin = lambda xb: [xb.reshape(BATCH, 8, IMG, IMG), xb, xb.reshape(BATCH, 8, IMG, IMG)]  # noqa: E731
        tt.seed_everything(5)
        worst, clamped = 0.0, 0
        for i in range(10):
            xs = xin(x[i])
            vs = [learning_votes(m, xm) for m, xm in zip(team.members, xs)]
            offs = team.offsets(vs)
            expect = [(v + o).clamp(-tj, tj) for v, o, tj in zip(vs, offs, team.T_joint)]
            assert torch.equal(team.combine(vs), vs[0] + vs[1] + vs[2])
            neg = (y[i] + torch.randint(1, K, (BATCH,), device=y.device)) % K
            got = []
            for m, xm, o in zip(team.members, xs, offs):
                m._joint_offset[:BATCH].copy_(o)
                m._joint_neg[:BATCH].copy_(neg)
                m._joint_on, m._joint_cur = True, 0
                got.append(m.update(xm, y[i]))
                m._joint_on = False
            worst = max(worst, max(float((g - e).abs().max()) for g, e in zip(got, expect)))
            clamped += sum(int((e.abs() == tj).sum()) for e, tj in zip(expect, team.T_joint))
        out[label] = {"max_abs_diff": worst, "clamped_entries": clamped,
                      "graphs_replayed": [n_graphs(m) for m in team.members]}
    ok = all(v["max_abs_diff"] == 0.0 for v in out.values())
    return ok, out


def c4(dev):
    x = load_cifar(600)["train"][0][:512]
    out = {}
    for enc in ("therm8", "bit8"):
        ref = cbc.encode(enc, x)
        got = gpu_encode(enc, torch.from_numpy(x).to(dev)).cpu().numpy()
        out[enc] = {"identical": bool(np.array_equal(ref, got)), "shape": list(got.shape)}
    for enc in ("agauss", "amean"):
        got = encode_fixed(enc, x, dev)
        out[enc] = {"shape": list(got.shape), "density": float(got.float().mean())}
    ok = all(v.get("identical", True) for v in out.values())
    if torch.device(dev).type == "cuda":
        for n, c in MEMBERS.items():
            m = build_member(dict(c, clauses=CLAUSES), 32, dev)
            xb = encode_fixed(c["enc"], x[:32], dev)
            P = m._patches_per_example(m._prepare(xb))
            out[f"fused/{n}"] = {"patches": int(P), "fused": bool(m._fused_eval_ok(P)),
                                 "max_chunk_elements": int(m.max_chunk_elements)}
            ok = ok and out[f"fused/{n}"]["fused"]
            del m
    return ok, out


def c5(dev):
    x = load_cifar(600)["train"][0][:300]
    out = {"telemetry_off": telemetry_off()}
    ok = out["telemetry_off"]
    for pol in POLICIES:
        _, ref, ref_enc = augment_epoch(x, pol, 0, 3, ("agauss", "amean"))
        res = {}
        for workers, chunk in ((2, 64), (3, 100)):
            src = EpochSource(x, pol, 0, ("agauss", "amean"), workers=workers, chunk=chunk)
            img, enc = src.get(3)
            img2, _ = src.get(4, last=True)
            src.close()
            res[f"w{workers}c{chunk}"] = bool(np.array_equal(img, ref) and all(np.array_equal(enc[e], ref_enc[e])
                                                                             for e in enc))
            res["epochs_differ"] = bool(not np.array_equal(img, img2))
        _, again, _ = augment_epoch(x, pol, 0, 3)
        _, other, _ = augment_epoch(x, pol, 1, 3)
        res.update(repeatable=bool(np.array_equal(again, ref)), seeds_differ=bool(not np.array_equal(other, ref)),
                   changed_images=int((img.reshape(len(x), -1) != x.reshape(len(x), -1)).any(1).sum()),
                   dtype=str(img.dtype), shape=list(img.shape))
        out[pol] = res
        ok = ok and all(v for k, v in res.items() if isinstance(v, bool)) and res["changed_images"] > len(x) // 2
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        # The Triton kernels and CUDA graphs run on the *current* device: without this a model
        # on cuda:1 silently falls back to the eager torch path (LIBRARY_GAPS CJT-1).
        torch.cuda.set_device(torch.device(dev))
    rec = new_record("checks", device=dev if dev != "cpu" else None, checks={})
    allok = True
    for name, fn in (("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5)):
        t0 = time.time()
        ok, detail = fn(dev)
        rec["checks"][name] = {"ok": bool(ok), "detail": detail}
        allok = allok and ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            print("   ", detail, flush=True)
    rec["ok"] = allok
    write_json(rec, RESULTS / "checks" / "check_team.json")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
