"""Fixed HTM-style registers as TM features (PLAN_HTM.md E-H3b; THEORY_HTM H3.3-H3.5).

All registers are computed per mini-batch on the device from the tokens ``x`` (B, L):

- seen set (strictly before t):    s_{t-1}[a] = OR_{u < t} [x_u = a]
- comparator (HTM predicted-vs-actual, a Boolean dot product):  r_t = <e(x_t), s_{t-1}>
- transition memory (bigrams before the query slot):  b[k, v] = OR_{t <= L-2} [x_{t-1} = k][x_t = v]

Arms registered into ``runner.ARMS`` on import (the runner file is not modified):

- ``H_T3_seen``  conv TM over time steps, kernel 1, columns [e(x_t) | s_{t-1}]       (H3.5c)
- ``H_T3_comp``  flat TM on [OR_t r_t | presence(x)]                                  (H3.4a)
- ``H_T2_mem``   flat TM on [e(x_{L-1}) | b]                                          (H3.5a)
- ``*_shuf``     the same with the register rolled by one example within the batch (the
                 register comes from another, independent example: must be at chance).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fnn

import torchtsetlin as tt
from experiments.transformer_concepts.code import runner


def onehot(x, V):
    return Fnn.one_hot(x, V).bool()  # (B, L, V)


def seen_before(e):
    c = torch.cumsum(e.to(torch.int16), 1) - e.to(torch.int16)
    return c > 0  # (B, L, V)


def comparator(e):
    return (e & seen_before(e)).any(-1)  # (B, L)


def transitions(e):
    """b[k, v] over the bigrams (x_{t-1}, x_t), t = 1 .. L-2 (the query slot L-1 excluded)."""
    a, b = e[:, :-2].float(), e[:, 1:-1].float()
    return torch.einsum("blk,blv->bkv", a, b) > 0  # (B, V, V)


def _roll(t, shuf):
    return torch.roll(t, 1, 0) if shuf else t


def f_t3_seen(shuf):
    def f(x, idx, ctx):
        e = onehot(x, ctx["V"])
        s = _roll(seen_before(e), shuf)
        return torch.cat([e, s], 2).transpose(1, 2).contiguous()  # (B, 2V, L)
    return f


def f_t3_comp(shuf):
    def f(x, idx, ctx):
        e = onehot(x, ctx["V"])
        r = _roll(comparator(e).any(1, keepdim=True), shuf)
        return torch.cat([r, e.any(1)], 1)  # (B, 1 + V)
    return f


def f_t2_mem(shuf):
    def f(x, idx, ctx):
        e = onehot(x, ctx["V"])
        b = _roll(transitions(e).reshape(x.shape[0], -1), shuf)
        return torch.cat([e[:, -1], b], 1)  # (B, V + V*V)
    return f


def _flat(spec, L, M, T, s, **kw):
    return tt.TsetlinMachine(None, spec.n_classes, M, T, s, weighted=True, **kw)


def _conv1(spec, L, M, T, s, **kw):
    return tt.Conv1dTsetlinMachine(spec.n_classes, M, T, s, kernel_size=1, position_encoding=False,
                                   weighted=True, **kw)


ARMS = {}
for shuf in (False, True):
    sfx = "_shuf" if shuf else ""
    ARMS[f"H_T3_seen{sfx}"] = runner.Arm(f"H_T3_seen{sfx}", _conv1, f_t3_seen(shuf))
    ARMS[f"H_T3_comp{sfx}"] = runner.Arm(f"H_T3_comp{sfx}", _flat, f_t3_comp(shuf))
    ARMS[f"H_T2_mem{sfx}"] = runner.Arm(f"H_T2_mem{sfx}", _flat, f_t2_mem(shuf))
runner.ARMS.update(ARMS)
