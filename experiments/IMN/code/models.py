"""Neural arms: TabResNet, IMN (hypernetwork -> linear model) and its Tsetlin-machine variants.

Every interpretable arm is one :class:`Mesomorphic` module: a *generator* produces, per
example, the parameters of an interpretable *local model*, and the logits are that local
model evaluated at the example.

=================  ======================  ================================================
arm                generator               local model (per output k)
=================  ======================  ================================================
``imn``            TabResNet MLP of x      ``b_k(x) + sum_m w_km(x) x_m``  (the paper)
``imn_tm``         TabResNet MLP of x      ``b_k(x) + sum_j W_kj(x) c_j(x)``  (TM clauses)
``imn_hybrid``     TabResNet MLP of x      both sums
``tm_gated``       TM clauses (linear)     ``b_k(x) + sum_m w_km(x) x_m`` with
                                           ``w_k(x) = w0_k + sum_j c_j(x) V_jk``
``clause_lr``      none (static)           ``b_k + sum_j W_kj c_j(x)``  (re-weighted TM)
=================  ======================  ================================================

Ablation arms (Phase 5): ``imn_tm_xc`` (generator input ``[x, c(x)]``), ``imn_tm_randfr``
(random clauses matched to the learned clauses' training firing rates).

``c_j(x)`` are the outputs of a *frozen* clause basis (a trained TM's clauses), so no gradient
has to pass through the TM: the clauses are fixed Boolean features of ``x`` and everything
that is learned is differentiable. Binary tasks use one output (BCE), as in the official code.

Architecture details follow the official IMN code (``models/hypernetwork.py`` and
``utils.BasicBlock``): input Linear -> BN -> GELU -> Dropout, then residual blocks
(Linear-BN-GELU-Dropout-Linear-BN + dropout(residual), GELU), last BN of each block zero-init.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.residual_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.linear2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.gelu = nn.GELU()
        nn.init.constant_(self.bn2.weight, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        r = self.residual_dropout(x)
        out = self.hidden_dropout(self.gelu(self.bn1(self.linear1(x))))
        out = self.bn2(self.linear2(out)) + r
        return self.gelu(out)


class TabResNetBackbone(nn.Module):
    def __init__(self, d_in: int, width: int = 128, n_blocks: int = 2, dropout: float = 0.25):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.GELU(),
                                   nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[BasicBlock(width, dropout) for _ in range(n_blocks)])
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.input(x))


class TabResNet(nn.Module):
    """Black-box counterpart with the same backbone (``tabresnet`` arm)."""

    interpretable = False

    def __init__(self, d_in: int, n_out: int, width: int = 128, n_blocks: int = 2, dropout: float = 0.25):
        super().__init__()
        self.backbone = TabResNetBackbone(d_in, width, n_blocks, dropout)
        self.head = nn.Linear(width, n_out)

    def forward(self, x: Tensor, c: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self.head(self.backbone(x)), {}


class Mesomorphic(nn.Module):
    """Generator of per-example interpretable local models.

    Args:
        d_in: number of (preprocessed) input features ``x``.
        n_out: 1 for binary tasks, else the number of classes.
        use_linear: the local model has a linear term in ``x``.
        n_local_clauses: ``K`` clause features in the local model (0 = none).
        generator: ``"mlp"`` (TabResNet hypernetwork of ``x``), ``"clauses"`` (a linear map of
            the clause outputs, i.e. a TM acting as the hypernetwork) or ``"static"`` (no
            dependence on ``x``: an ordinary global model on the local features).
        n_gen_clauses: clause count feeding the ``"clauses"`` generator.
        clause_init: optional ``(K, n_out)`` initial clause weights (e.g. the source TM's
            signed votes scaled by ``1/T``). With a zero-initialised head the model then starts
            out *exactly* as that weighted clause vote, for every ``x``.
    """

    interpretable = True

    def __init__(self, d_in: int, n_out: int, *, use_linear: bool = True, n_local_clauses: int = 0,
                 generator: str = "mlp", n_gen_clauses: int = 0, width: int = 128,
                 n_blocks: int = 2, dropout: float = 0.25, clause_init: Optional[Tensor] = None,
                 gen_input_clauses: bool = False):
        super().__init__()
        if generator not in ("mlp", "clauses", "static"):
            raise ValueError(generator)
        if not use_linear and n_local_clauses == 0:
            raise ValueError("the local model needs a linear and/or a clause term")
        self.d_in, self.n_out = d_in, n_out
        self.use_linear, self.K = use_linear, n_local_clauses
        self.generator = generator
        self.gen_input_clauses = bool(gen_input_clauses) and generator == "mlp"
        self.P = (d_in if use_linear else 0) + n_local_clauses + 1  # +1 = bias
        n_params = self.P * n_out
        if generator == "mlp":
            g_in = d_in + (n_local_clauses if self.gen_input_clauses else 0)
            self.backbone = TabResNetBackbone(g_in, width, n_blocks, dropout)
            self.head = nn.Linear(width, n_params)
        elif generator == "clauses":
            if n_gen_clauses <= 0:
                raise ValueError("generator='clauses' needs n_gen_clauses > 0")
            self.head = nn.Linear(n_gen_clauses, n_params)  # bias = w0, weight = V
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            self.static = nn.Parameter(torch.zeros(n_params))
        # prior of the generated parameters: 0, except the clause block when initialised from
        # a TM; train_nn's l1_mode="prior" penalises |w(x) - prior| (shrink toward the TM)
        self.register_buffer("w_prior", torch.zeros(self.P, n_out))
        if clause_init is not None:
            if n_local_clauses != clause_init.shape[0]:
                raise ValueError("clause_init must have n_local_clauses rows")
            self.init_clause_weights(clause_init)

    # layout of the generated vector: [linear (d) | clauses (K) | bias] x n_out
    def _bias_view(self) -> Tensor:
        if self.generator == "static":
            return self.static.view(self.P, self.n_out)
        return self.head.bias.view(self.P, self.n_out)

    @torch.no_grad()
    def init_clause_weights(self, w: Tensor) -> None:
        if self.generator == "mlp":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        lo = self.d_in if self.use_linear else 0
        self._bias_view()[lo:lo + self.K] = w.to(self._bias_view())
        self.w_prior[lo:lo + self.K] = w.to(self.w_prior)

    def generate(self, x: Tensor, c_gen: Optional[Tensor] = None) -> Tensor:
        """Local-model parameters ``(B, P, n_out)``."""
        B = x.shape[0]
        if self.generator == "mlp":
            g_in = torch.cat([x, c_gen], 1) if self.gen_input_clauses else x
            w = self.head(self.backbone(g_in))
        elif self.generator == "clauses":
            w = self.head(c_gen)
        else:
            w = self.static.unsqueeze(0).expand(B, -1)
        return w.view(B, self.P, self.n_out)

    def local_features(self, x: Tensor, c: Optional[Tensor]) -> Tensor:
        parts = []
        if self.use_linear:
            parts.append(x)
        if self.K:
            parts.append(c)
        parts.append(torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
        return torch.cat(parts, 1)

    def forward(self, x: Tensor, c: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """``c``: clause outputs ``(B, K)`` (local features and/or generator input)."""
        w = self.generate(x, c if (self.generator == "clauses" or self.gen_input_clauses) else None)
        z = torch.einsum("bp,bpk->bk", self.local_features(x, c if self.K else None), w)
        return z, {"w": w}

    def split(self, w: Tensor) -> Dict[str, Tensor]:
        """Split generated ``(B, P, n_out)`` into ``linear (B,d,n_out)``, ``clauses (B,K,n_out)``,
        ``bias (B,n_out)``."""
        out, lo = {}, 0
        if self.use_linear:
            out["linear"] = w[:, :self.d_in]
            lo = self.d_in
        if self.K:
            out["clauses"] = w[:, lo:lo + self.K]
        out["bias"] = w[:, -1]
        return out


def build(arm: str, d_in: int, n_out: int, K: int = 0, cfg: Optional[Dict] = None,
          clause_init: Optional[Tensor] = None) -> nn.Module:
    cfg = cfg or {}
    kw = dict(width=cfg.get("width", 128), n_blocks=cfg.get("n_blocks", 2),
              dropout=cfg.get("dropout", 0.25))
    if arm == "tabresnet":
        return TabResNet(d_in, n_out, **kw)
    if arm == "imn":
        return Mesomorphic(d_in, n_out, use_linear=True, generator="mlp", **kw)
    if arm in ("imn_tm", "imn_tm_rand", "imn_tm_randfr", "imn_tm_xc"):
        return Mesomorphic(d_in, n_out, use_linear=False, n_local_clauses=K, generator="mlp",
                           clause_init=clause_init, gen_input_clauses=arm == "imn_tm_xc", **kw)
    if arm == "imn_hybrid":
        return Mesomorphic(d_in, n_out, use_linear=True, n_local_clauses=K, generator="mlp",
                           clause_init=clause_init, **kw)
    if arm == "tm_gated":
        return Mesomorphic(d_in, n_out, use_linear=True, generator="clauses", n_gen_clauses=K, **kw)
    if arm == "clause_lr":
        return Mesomorphic(d_in, n_out, use_linear=False, n_local_clauses=K, generator="static",
                           clause_init=clause_init, **kw)
    raise ValueError(f"unknown neural arm {arm!r}")


def needs_clauses(arm: str) -> bool:
    return arm in ("imn_tm", "imn_tm_rand", "imn_tm_randfr", "imn_tm_xc", "imn_hybrid", "tm_gated",
                   "clause_lr")
