"""Phase-0 checks on one GPU (plan §5). Writes results/checks.json; exit code 1 if one fails.

1. **Depth 1 is the convolutional coalesced TM.** A ``GraphTsetlinMachine`` of depth 1 on this
   programme's batches (4 x 4 windows, stride 2) and a ``ConvCoalescedTsetlinMachine`` with the
   same windows, started from the same automata, weights and random stream, have identical vote
   sums, automata and weights after every update, and identical predictions.
2. **Fast path.** Updates of depth 1, 2 and 3 on the per-batch graphs are committed by the Triton
   kernel (events mode), and a warmed-up depth-2 update never synchronises with the host.
3. **Rewired control.** The rewired graph has the same number of in-edges of every type at every
   node as the grid, no self-loops, and different senders.

Usage: CUDA_VISIBLE_DEVICES=0 python code/checks.py
"""

import sys

import numpy as np
import torch
from common import BUDGET, EDGE_TYPES, RESULTS, load_cifar, new_record, write_json
from graphs import GridInput, grid_edges, thermometer

import torchtsetlin as tt
from torchtsetlin import _triton

C, T, S = 400, 200, 5.0


def graph_model(dev, depth=1, **kw):
    return tt.GraphTsetlinMachine(412, 10, C, T, S, depth=depth, message_size=64, n_edge_types=len(EDGE_TYPES),
                                  max_included_literals=BUDGET, **kw).to(dev)


def depth1_equals_conv(x, y, inp, dev):
    conv = tt.ConvCoalescedTsetlinMachine(10, C, T, S, patch_size=4, stride=2, input_shape=(24, 32, 32),
                                          max_included_literals=BUDGET).to(dev)
    graph = graph_model(dev)
    conv.update(x[:32], y[:32])                       # lazy buffers exist; then copy the whole state
    graph.update(inp(x[:32]), y[:32])
    graph.ta_state.copy_(conv.ta_state)
    graph.weights.copy_(conv.weights)
    graph._refresh_include()
    steps = 0
    for i in range(0, 640, 32):
        torch.manual_seed(100 + i)
        v1 = conv.update(x[i:i + 32], y[i:i + 32])
        torch.manual_seed(100 + i)
        v2 = graph.update(inp(x[i:i + 32]), y[i:i + 32])
        if not (torch.equal(v1, v2) and torch.equal(conv.ta_state, graph.ta_state)
                and torch.equal(conv.weights, graph.weights)):
            return {"ok": False, "first_difference_at_update": steps}
        steps += 1
    conv.eval(), graph.eval()
    same = torch.equal(conv(x[:500]), graph(inp(x[:500])))
    learned = int((conv.ta_state != conv.n_states - 1).sum())
    return {"ok": bool(same and learned > 1000), "updates": steps, "predictions_equal": bool(same),
            "automata_moved": learned}


def fast_path(x, y, inp, dev):
    out = {"triton": bool(_triton.available(dev))}
    if not out["triton"]:
        return dict(out, ok=False, note="Triton kernels are not available on this GPU: training runs on "
                                        "the slower eager path")
    calls = {"n": 0}
    real = _triton.fused_commit

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    _triton.fused_commit = counted
    try:
        for depth in (1, 2, 3):
            m = graph_model(dev, depth)
            m.cuda_graphs = False                     # count the Python-level calls
            calls["n"] = 0
            for i in range(0, 320, 32):
                m.update(inp(x[i:i + 32]), y[i:i + 32])
            out[f"depth{depth}_kernel_commits_of_10"] = calls["n"]
            if depth == 2:
                g = inp(x[320:352])
                torch.cuda.synchronize(dev)
                torch.cuda.set_sync_debug_mode("error")
                try:
                    m.update(g, y[320:352])
                    out["depth2_update_sync_free"] = True
                except RuntimeError as e:
                    out["depth2_update_sync_free"] = False
                    out["sync_error"] = str(e)[:300]
                finally:
                    torch.cuda.set_sync_debug_mode(0)
    finally:
        _triton.fused_commit = real
    out["ok"] = all(out[f"depth{d}_kernel_commits_of_10"] == 10 for d in (1, 2, 3)) and out["depth2_update_sync_free"]
    return out


def rewired():
    (e, t), (r, rt) = grid_edges("grid"), grid_edges("rewired", 0)
    same_in = np.array_equal(e[:, 1], r[:, 1]) and np.array_equal(t, rt)
    return {"ok": bool(same_in and (r[:, 0] != r[:, 1]).all() and (e[:, 0] != r[:, 0]).mean() > 0.9),
            "edges": int(len(e)), "senders_changed": float((e[:, 0] != r[:, 0]).mean())}


def main():
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    tt.seed_everything(0)
    raw, lab = load_cifar(1000)["train"]
    x, y = thermometer(raw, dev), torch.from_numpy(lab).to(dev)
    inp = GridInput(dev)
    rec = new_record("checks", device=dev, status="done", depth1_equals_conv=depth1_equals_conv(x, y, inp, dev),
                     fast_path=fast_path(x, y, inp, dev), rewired=rewired())
    rec["ok"] = all(rec[k]["ok"] for k in ("depth1_equals_conv", "fast_path", "rewired"))
    write_json(rec, RESULTS / "checks.json")
    for k in ("depth1_equals_conv", "fast_path", "rewired"):
        print(f"  {k}: {'ok' if rec[k]['ok'] else 'FAILED'}  {rec[k]}")
    sys.exit(0 if rec["ok"] else 1)


if __name__ == "__main__":
    main()
