"""The three repairs proposed in section 7.3 of the first report, implemented.

The first report's conclusion was that the MCTM *architecture* is sound (E2: given oracle
features the stack clears a ceiling one layer provably cannot) but that **greedy supervised
layer-wise training** is not: training layer 1 to classify rewards specific clauses, those
clauses fire on 0.14% of positions, and layer 2 degenerates into 97% negated literals that
assert only absences. Its section 7.3 named three ways out, in increasing order of how much
they change the proposal. This module implements all three.

A. ``pretrain_autoencoder_l1`` -- **a different objective for layer 1.** Clauses predict a
   held-out part of a patch from the rest, with no labels anywhere. Nothing in that objective
   rewards extreme specificity, and nothing discards the residual information a classifier
   would throw away.

B. ``calibrate_density`` -- **explicit density calibration.** A target per-channel firing
   rate (and optionally a clause-size cap), and a greedy per-clause walk -- removing literals
   from clauses that fire too rarely, adding them to clauses that fire too often -- that moves
   each clause towards it. The Boolean analogue of normalisation: it changes the
   representation's *statistics* without consulting labels.

C. ``CreditStack`` -- **credit propagation instead of freezing.** When a layer-2 clause
   receives Type I feedback and includes the literal "channel k active at offset (dy,dx)",
   layer-1 clause k receives Type I feedback at the corresponding position. A discrete,
   gradient-free credit path that makes layer 1 trainable through layer 2.

Everything here is built from the documented update pipeline
(``_encode -> _evaluate -> _votes -> _select_feedback -> _feedback_counts -> _commit``) and
``functional.apply_feedback``; no library file is modified.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from mctm import conv_clause_maps, or_pool
from torch import Tensor
from torch.nn import functional as TF

import torchtsetlin as tt
from torchtsetlin import functional as TTF
from torchtsetlin.utils import chunk_indices


# =====================================================================================
#  A. A different objective for layer 1: a patch autoencoder
# =====================================================================================
def _centre_mask(Z: int, kh: int, kw: int) -> Tensor:
    """Boolean ``(Z*kh*kw,)`` mask selecting the centre pixels of a patch, all planes.

    The conv layer's feature order is ``p[z,i,j]`` at index ``z*kh*kw + i*kw + j``
    (``_ConvMixin.default_feature_names``), so the same index arithmetic addresses both the
    flat autoencoder's features and the conv layer's automata columns.
    """
    m = torch.zeros(Z, kh, kw, dtype=torch.bool)
    i0, i1 = (kh - 1) // 2, kh // 2 + 1
    j0, j1 = (kw - 1) // 2, kw // 2 + 1
    m[:, i0:i1, j0:j1] = True
    return m.reshape(-1)


@torch.no_grad()
def sample_patches(x: Tensor, patch: Tuple[int, int], stride: Tuple[int, int],
                   n_patches: int, seed: int = 0, img_chunk: int = 256) -> Tensor:
    """``(n_patches, Z*kh*kw)`` bool: patches drawn uniformly from random images.

    Unfolding all of CIFAR-10 at 4x4/stride-1 gives 42M patches; the autoencoder only needs a
    sample, so patches are drawn per image chunk and never materialised in full.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    N, Z, H, W = x.shape
    kh, kw = patch
    per_img = max(1, n_patches // N + 1)
    out: List[Tensor] = []
    got = 0
    order = torch.randperm(N, generator=g)
    for sl in chunk_indices(N, img_chunk):
        idx = order[sl].to(x.device)
        u = TF.unfold(x[idx].to(torch.float32), kernel_size=patch, stride=stride)  # (b, F, P)
        b, Fn, P = u.shape
        pick = torch.randint(0, P, (b, per_img), generator=g).to(x.device)
        sel = torch.gather(u, 2, pick.unsqueeze(1).expand(b, Fn, per_img))  # (b, F, per_img)
        out.append(sel.permute(0, 2, 1).reshape(-1, Fn) > 0.5)
        got += out[-1].shape[0]
        if got >= n_patches:
            break
    return torch.cat(out, dim=0)[:n_patches]


@torch.no_grad()
def embed_flat_into_conv(conv, flat, keep: Tensor) -> None:
    """Write a flat model's automata into a conv layer's automata.

    ``keep`` is a ``(Z*kh*kw,)`` bool mask over the conv layer's *pixel* features naming the
    ones the flat model was trained on, in order. Everything else -- the masked-out pixels and
    the thermometer position bits -- stays excluded, so the conv layer spans exactly the same
    literal space as a greedily trained layer 1 and simply never uses those columns.
    """
    N = conv.n_states
    Fc, Ff = int(conv.n_features), int(flat.n_features)
    cols = torch.nonzero(keep, as_tuple=False).flatten().to(conv.ta_state.device)
    assert cols.numel() == Ff, f"keep selects {cols.numel()} features, flat model has {Ff}"
    conv.ta_state.fill_(N - 1)
    conv.ta_state[:, cols] = flat.ta_state[:, :Ff]
    conv.ta_state[:, Fc + cols] = flat.ta_state[:, Ff:]
    conv._refresh_include()


@torch.no_grad()
def _ae_accuracy(flat, xc: Tensor, yt: Tensor, batch: int = 4096) -> float:
    """Mean per-bit reconstruction accuracy of the held-out target bits."""
    was = flat.training
    flat.eval()
    correct = 0
    for sl in chunk_indices(xc.shape[0], batch):
        correct += int(((flat(xc[sl]) > 0) == yt[sl]).sum())
    flat.train(was)
    return correct / (xc.shape[0] * yt.shape[1])


def pretrain_autoencoder_l1(conv, x: Tensor, *, n_patches: int = 300_000, epochs: int = 6,
                            s: float = 10.0, T: Optional[float] = None, batch_size: int = 200,
                            seed: int = 0, log: Optional[list] = None) -> dict:
    """Idea A: train ``conv``'s clauses with an unsupervised patch-reconstruction objective.

    Each patch is split into a **target** (its centre pixels, all bit planes) and a
    **context** (the surrounding ring). A flat coalesced Tsetlin machine with one output per
    target bit is trained multi-label to predict the target from the context; its clauses are
    then written into the convolutional layer by :func:`embed_flat_into_conv`.

    No label is used anywhere, which is the point: the greedy scheme's failure is that
    classification rewards specific, rarely-firing clauses and discards whatever did not help
    the local classifier. Reconstruction has no such pressure -- a clause earns its keep by
    predicting *any* part of the input.

    The split is spatial rather than a random subset of bits because thermometer planes of the
    same pixel are nested (``v > t_1`` implies ``v > t_0``): holding out one plane of a pixel
    while showing another leaks the answer, and the clause learns the encoder instead of the
    image.
    """
    Z, H, W = conv.input_shape
    kh, kw = conv.patch_size
    tgt = _centre_mask(Z, kh, kw).to(x.device)
    ctx = ~tgt
    patches = sample_patches(x, conv.patch_size, conv.stride, n_patches, seed=seed)
    xc, yt = patches[:, ctx], patches[:, tgt]
    C = int(conv.n_clauses_total)
    tt.seed_everything(seed)
    flat = tt.CoalescedTsetlinMachine(int(ctx.sum()), int(tgt.sum()), C,
                                      float(T if T is not None else C), s,
                                      multi_label=True).to(x.device)
    flat.train()
    log = [] if log is None else log
    n = xc.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=xc.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            flat.update(xc[idx], yt[idx])
        rec = {"epoch": ep + 1,
               "recon_acc": _ae_accuracy(flat, xc[:20000], yt[:20000]),
               "clause_size": float(flat.include_count.float().median())}
        log.append(rec)
        print(f"  [ae:L1] epoch {ep+1}/{epochs}  recon {rec['recon_acc']:.4f}  "
              f"size {rec['clause_size']:.0f}", flush=True)
    embed_flat_into_conv(conv, flat, ctx)
    base = torch.tensor([yt.float().mean(), 1 - yt.float().mean()]).max()
    return {"log": log, "n_patches": int(n), "n_context": int(ctx.sum()),
            "n_target": int(tgt.sum()), "s": s, "epochs": epochs,
            "recon_acc": log[-1]["recon_acc"], "recon_baseline": float(base)}


# =====================================================================================
#  B. Explicit density calibration
# =====================================================================================
@torch.no_grad()
def _probe_literals(model, x: Tensor, n_patches: int, seed: int = 0,
                    img_chunk: int = 64) -> Tensor:
    """``(n_patches, 2F)`` float literal rows sampled from ``x``'s patches.

    Calibration needs patch literals, not images: the firing rate being calibrated is per
    *patch position*, and the gain of removing a literal is a matmul over patch rows.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = x.shape[0]
    per_img = max(1, n_patches // N + 1)
    out: List[Tensor] = []
    got = 0
    for sl in chunk_indices(N, img_chunk):
        lits = model._encode(model._prepare(x[sl]))  # (b, P, 2F)
        b, P, L = lits.shape
        pick = torch.randint(0, P, (b, per_img), generator=g).to(x.device)
        sel = torch.gather(lits, 1, pick.unsqueeze(2).expand(b, per_img, L))
        out.append(sel.reshape(-1, L))
        got += out[-1].shape[0]
        if got >= n_patches:
            break
    return torch.cat(out, dim=0)[:n_patches]


def pixel_features(model) -> int:
    """How many of a conv layer's features are patch pixels rather than position bits.

    ``_ConvMixin`` lays a patch out as ``Z*kh*kw`` pixels followed by thermometer-encoded
    patch coordinates. The two are not interchangeable for a *feature* layer: a position
    literal makes a clause fire on a strip of the image regardless of what is in it, which
    is a way to hit any firing rate while carrying no information at all.
    """
    if getattr(model, "input_shape", None) is None:
        return int(model.n_features)
    Z = model.input_shape[0]
    kh, kw = model.patch_size
    return Z * kh * kw


@torch.no_grad()
def calibrate_density(model, x: Tensor, target_rate: float, *, band: float = 2.0,
                      n_patches: int = 60_000, max_rounds: int = 200, min_size: int = 1,
                      max_size: Optional[int] = None, seed: int = 0,
                      lits: Optional[Tensor] = None, pixels_only: bool = True) -> dict:
    """Idea B: move every clause towards a target per-patch firing rate.

    A clause fires on a patch iff none of its included literals is false there, so its firing
    rate is monotone in its literal set: removing a literal can only raise it, adding one can
    only lower it. That makes the target rate reachable by a greedy walk in both directions,
    which is what "adjust each clause towards it" asks for.

    Per round, for the clauses **below** the band:

        ``gain[j,k] = #{patches where clause j has exactly one violated literal, and it is k}``

    the literal whose removal buys the most, computed as ``(violations == 1)^T @ (1 - literals)``
    masked to included literals -- the same shape of computation as the Type Ia count in
    :mod:`torchtsetlin.functional`. Where no single removal helps (every blocked patch violates
    two or more literals) the most-often-violated literal goes instead, so a round always makes
    progress. For the clauses **above** the band, adding literal ``k`` drops the rate by the
    fraction of currently-matching patches in which ``k`` is false, so the literal that lands
    closest to the target without undershooting it is included.

    The two directions are what let this run as a *controller* rather than a one-off prune: an
    over-general clause (an empty one included -- it violates nothing, so its control rate is
    1) is specialised, an over-specific one is generalised, and a clause that has once been
    inside the band is frozen so the two rules cannot chase each other.

    Adjustments are written as ``n_states`` / ``n_states - 1``: included or excluded, one step
    from the boundary, exactly where a fresh automaton sits. Calibration therefore stays inside
    the automata state and survives continued training, which is what lets it be interleaved
    with idea C.

    Args:
        target_rate: the per-patch firing rate to aim for.
        band: multiplicative tolerance; clauses within ``[target/band, target*band]`` are left
            alone. ``band=1`` would make every clause a permanent construction site.
        lits: pre-sampled probe literals, to reuse one probe across repeated calls.
        max_size: optional hard cap on clause size, applied before the rate rules and
            respected by them. Firing rate and clause size are different properties -- a
            long conjunction that is rare-but-common and a short general one can sit at the
            same rate -- and only the second is a feature a second layer can build on, so the
            controller can be asked for both.
        pixels_only: restrict *added* literals to patch pixels (see :func:`pixel_features`).
            Left off, the cheapest way to hit any target rate is a position literal --
            ``y > 27`` fires on exactly 1/29 of patch rows whatever the image contains -- and
            the controller takes it every time.
    """
    if lits is None:
        lits = _probe_literals(model, x, n_patches, seed=seed)
    neg = 1.0 - lits
    Np = lits.shape[0]
    N = model.n_states
    C = model.n_clauses_total
    lo, hi = target_rate / band, target_rate * band
    dev = lits.device
    addable = torch.ones(model.n_literals, dtype=torch.bool, device=dev)
    if pixels_only:
        n_pix = pixel_features(model)
        Fn = int(model.n_features)
        addable[:] = False
        addable[:n_pix] = True
        addable[Fn : Fn + n_pix] = True
    frozen = torch.zeros(C, dtype=torch.bool, device=dev)
    last = torch.zeros(C, dtype=torch.int8, device=dev)   # +1 = added, -1 = removed
    hist: List[dict] = []
    for rnd in range(max_rounds):
        viol = torch.matmul(neg, model.include.transpose(0, 1))       # (Npatch, C)
        fires = (viol == 0).to(model.compute_dtype)
        rate = fires.mean(dim=0)
        size = model.include_count
        oversize = (size > max_size) if max_size is not None else torch.zeros_like(frozen)
        frozen |= (rate >= lo) & (rate <= hi) & ~oversize
        need_drop = ((rate < lo) & (size > min_size) & ~frozen) | oversize
        need_add = (rate > hi) & ~frozen & ~oversize
        if max_size is not None:
            need_add &= size < max_size
        hist.append({"round": rnd, "median_rate": float(rate.median()),
                     "n_below": int(need_drop.sum()), "n_above": int(need_add.sum()),
                     "median_size": float(size.float().median())})
        if not bool((need_drop | need_add).any()):
            break
        if bool(need_drop.any()):
            sole = (viol == 1).to(model.compute_dtype)
            gain = torch.matmul(sole.transpose(0, 1), neg) * model.include
            stuck = (gain.sum(dim=1) == 0) & need_drop
            if bool(stuck.any()):
                gain[stuck] = neg.sum(dim=0).unsqueeze(0) * model.include[stuck]
            # Removing literal k lets through exactly the patches it alone was blocking, so
            # the post-removal rate is known in closed form.
            pick_drop = gain.argmax(dim=1)
            if max_size is not None and bool(oversize.any()):
                # Under a size cap the choice is not "which removal buys the most" but "which
                # removal leaves the rate nearest the target": dropping the most load-bearing
                # literal keeps the most permissive ones and sends the rate to ~1, dropping the
                # least keeps the most selective and sends it to ~0. Neither is the clause
                # being asked for.
                new_rate = rate.unsqueeze(1) + gain / Np
                cost = torch.where(model.include > 0, (new_rate - target_rate).abs(),
                                   torch.full_like(new_rate, float("inf")))
                pick_drop = torch.where(oversize, cost.argmin(dim=1), pick_drop)
            flip = need_drop & (last > 0) & ~oversize
            frozen |= flip                                    # add-then-drop: settle here
            rows = torch.nonzero(need_drop & ~flip, as_tuple=False).flatten()
            if rows.numel():
                model.ta_state[rows, pick_drop[rows]] = N - 1
                last[rows] = -1
        if bool(need_add.any()):
            # rate after including literal k = rate - (matching patches where k is false)/Np
            cnt = torch.matmul(fires.transpose(0, 1), neg) / Np       # (C, 2F)
            new_rate = rate.unsqueeze(1) - cnt
            free = (model.include == 0) & addable.unsqueeze(0)
            valid = (new_rate >= lo) & free
            score = torch.where(valid, new_rate, torch.full_like(new_rate, float("inf")))
            pick = score.argmin(dim=1)
            none_valid = ~valid.any(dim=1)
            gentle = torch.where(free, new_rate,
                                 torch.full_like(new_rate, -float("inf"))).argmax(dim=1)
            pick = torch.where(none_valid, gentle, pick)
            flip = need_add & (last < 0)
            frozen |= flip                                    # drop-then-add: settle here
            rows = torch.nonzero(need_add & ~flip, as_tuple=False).flatten()
            if rows.numel():
                model.ta_state[rows, pick[rows]] = N
                last[rows] = 1
        model._refresh_include()
    viol = torch.matmul(neg, model.include.transpose(0, 1))
    rate = (viol == 0).to(torch.float32).mean(dim=0)
    return {"target_rate": target_rate, "band": band, "max_size": max_size,
            "rounds": len(hist),
            "rate_median": float(rate.median()), "rate_mean": float(rate.mean()),
            "rate_min": float(rate.min()), "rate_max": float(rate.max()),
            "frac_in_band": float(((rate >= lo) & (rate <= hi)).float().mean()),
            "size_median": float(model.include_count.float().median()),
            "size_mean": float(model.include_count.float().mean()),
            "n_empty": int((model.include_count == 0).sum()),
            "history": hist[:: max(1, len(hist) // 40)]}


# =====================================================================================
#  C. Credit propagation
# =====================================================================================
class CreditStack:
    """Idea C: a two-layer stack in which layer-2 feedback reaches layer 1.

    The pseudocode froze layer 1 because Tsetlin automata have no gradient to propagate.
    But clause *inclusion* is already a credit path. If layer-2 clause ``j`` matched patch
    ``(py,px)`` and includes the positive literal "channel ``k`` fires at offset ``(dy,dx)``",
    then the reason it matched is that layer-1 clause ``k`` fired somewhere inside the pooled
    cell ``(py+dy, px+dx)``. Whatever feedback layer 2 decided clause ``j`` deserves is
    therefore also a verdict on layer-1 clause ``k`` at that position:

      * layer-2 **Type Ia** (matched, should keep matching) -> layer-1 clause ``k`` receives
        **Type Ia** at a position inside that cell where it actually fired: reinforce the
        pixel pattern that produced the activation;
      * layer-2 **Type Ib** (should have matched, did not) -> layer-1 clause ``k`` receives
        **Type Ib**: forget literals, i.e. become more general, because a channel the layer-2
        clause needs did not fire. Type Ib is position-free in the library too, which is what
        makes it propagate cleanly with no patch to point at;
      * layer-2 **Type II** (matched, votes for the wrong class) -> layer-1 clause ``k``
        receives **Type II** there: include a literal that is false in that patch, so it stops
        firing at it.

    Propagating **Type Ib matters more than it looks**. Type Ia and Type II both *add*
    literals; only Type Ib removes them. A credit rule that forwards Type Ia alone -- the
    literal reading of the proposal, available here as ``type_ib=False`` -- gives layer 1 a
    memorization term with no forgetting term, and a single boosted Type Ia event includes
    every literal true in the drawn patch. Layer 1 saturates within a few dozen batches, stops
    firing anywhere, and the credit path then extinguishes itself: with no channel active,
    no layer-2 clause can include a positive channel literal, and there is nothing left to
    propagate. \\S\ref{sec:credit} reports that run.

    Two further choices, recorded because they are the parts a reader would want to vary:

    * **One credited literal per event.** A layer-2 clause can include hundreds of literals;
      crediting all of them would make a single event a bulk update of layer 1 and would cost
      ``n_events x n_literals`` gathered patch rows. One included positive literal is drawn
      uniformly per event, exactly as the convolutional machine draws one matching patch per
      event.
    * **Negated literals do not propagate.** "Channel ``k`` is absent here" is satisfied by a
      layer-1 clause that already does not fire there; there is no event to reinforce.
    """

    def __init__(self, l1, l2, pool: int = 2, n_credit: int = 1,
                 credit_type_ii: bool = True, type_ib: bool = True, ib_ratio: float = 0.0,
                 credit_rate: float = 1.0, max_events: int = 20000, dedup: bool = True):
        self.l1, self.l2, self.pool = l1, l2, int(pool)
        self.n_credit = int(n_credit)
        self.credit_type_ii = bool(credit_type_ii)
        self.type_ib = bool(type_ib)
        self.ib_ratio = float(ib_ratio)
        self.credit_rate = float(credit_rate)
        self.max_events = int(max_events)
        self.dedup = bool(dedup)
        self.stats = {k: 0 for k in ("events_i", "events_ib", "events_ii",
                                     "kept_i", "kept_ib", "kept_ii", "batches")}

    # ---- forward ------------------------------------------------------------------
    @torch.no_grad()
    def maps(self, xb: Tensor, raw: bool = False):
        m = conv_clause_maps(self.l1, xb)                 # (B, C1, Py1, Px1)
        return (m, or_pool(m, self.pool)) if raw else or_pool(m, self.pool)

    @torch.no_grad()
    def forward(self, x, batch_size: int = 256) -> Tensor:
        was1, was2 = self.l1.training, self.l2.training
        self.l1.eval()
        outs = []
        for sl in chunk_indices(x.shape[0], batch_size):
            outs.append(self.l2(self.maps(x[sl])))
        self.l1.train(was1), self.l2.train(was2)
        return torch.cat(outs, dim=0)

    def __call__(self, x) -> Tensor:
        return self.forward(x)

    @property
    def training(self) -> bool:
        return self.l2.training

    def train(self, mode: bool = True):
        self.l1.train(mode), self.l2.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ---- the credit path ----------------------------------------------------------
    @torch.no_grad()
    def _events(self, kind: str, sel: Tensor, matches2: Tensor, raw_maps: Tensor,
                Py2: int, Px2: int) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Layer-2 feedback events ``(example, layer-1 clause, layer-1 patch index)``.

        One event is one (image, layer-2 clause) pair that received feedback while matching.
        It is resolved to a single layer-1 position in three draws: a matching layer-2 patch,
        one included positive channel literal of that clause, and one layer-1 position inside
        the pooled cell that literal names at which the channel actually fired.
        """
        l2 = self.l2
        C1, H1, W1 = raw_maps.shape[1:]
        kh2, kw2 = l2.patch_size
        sh2, sw2 = l2.stride
        n_pix2 = C1 * kh2 * kw2
        pos_mask = l2.included_mask()[:, :n_pix2].to(l2.compute_dtype)
        maps_flat = raw_maps.reshape(-1)
        pool = self.pool
        uu = torch.arange(pool, device=sel.device).repeat_interleave(pool)
        vv = torch.arange(pool, device=sel.device).repeat(pool)
        offs = (uu * W1 + vv).unsqueeze(0)

        nz = torch.nonzero(sel, as_tuple=True)
        if nz[0].numel() == 0:
            return None
        b_idx, j_idx = nz
        if b_idx.numel() > self.max_events:
            keep = torch.randperm(b_idx.numel(), device=sel.device)[: self.max_events]
            b_idx, j_idx = b_idx[keep], j_idx[keep]
        outs: List[Tuple[Tensor, Tensor, Tensor]] = []
        for sl in chunk_indices(b_idx.numel(), 8192):
            bi, ji = b_idx[sl], j_idx[sl]
            cand = matches2[bi, :, ji].to(l2.compute_dtype)             # (n, P2)
            p = (torch.rand(cand.shape, device=cand.device) * cand).argmax(dim=1)
            py = torch.div(p, Px2, rounding_mode="floor")
            px = p % Px2
            for _ in range(self.n_credit):
                pm = pos_mask[ji]                                        # (n, n_pix2)
                m = (torch.rand(pm.shape, device=pm.device) * pm).argmax(dim=1)
                k = torch.div(m, kh2 * kw2, rounding_mode="floor")
                rem = m % (kh2 * kw2)
                dy = torch.div(rem, kw2, rounding_mode="floor")
                dx = rem % kw2
                r, c = py * sh2 + dy, px * sw2 + dx
                base = ((bi * C1 + k) * H1 + r * pool) * W1 + c * pool
                blk = maps_flat[base.unsqueeze(1) + offs].to(l2.compute_dtype)
                pick = (torch.rand(blk.shape, device=blk.device) * blk).argmax(dim=1)
                ok = (pm.sum(dim=1) > 0) & (blk.sum(dim=1) > 0)
                if not bool(ok.any()):
                    continue
                row1 = r[ok] * pool + torch.div(pick[ok], pool, rounding_mode="floor")
                col1 = c[ok] * pool + pick[ok] % pool
                outs.append((bi[ok], k[ok], row1 * W1 + col1))
        if not outs:
            return None
        bi = torch.cat([o[0] for o in outs])
        kk = torch.cat([o[1] for o in outs])
        pp = torch.cat([o[2] for o in outs])
        self.stats["events_" + kind] += int(bi.numel())
        if self.dedup:
            key = bi * C1 + kk
            order = torch.rand(key.numel(), device=key.device).argsort()
            win = torch.full((int(key.max()) + 1,), -1, dtype=torch.long, device=key.device)
            win[key[order]] = order
            keep = win[key] == torch.arange(key.numel(), device=key.device)
            bi, kk, pp = bi[keep], kk[keep], pp[keep]
        bi, kk, pp = self._thin(bi, kk, pp)
        self.stats["kept_" + kind] += int(bi.numel())
        return bi, kk, pp

    @torch.no_grad()
    def _events_ib(self, sel: Tensor, viol2: Tensor, lits2: Tensor, C1: int,
                   budget: Optional[int] = None) -> Optional[Tensor]:
        """Layer-1 clauses to generalise, from layer-2 Type Ib events.

        A Type Ib event says the layer-2 clause *should* have matched and did not, so the
        credit question is which literal blocked it. The clause is taken at the patch it came
        closest to matching (fewest violated literals), and the blame falls on an included
        positive channel literal that was **false** there: channel ``k`` was needed at that
        cell and did not fire. Layer-1 clause ``k`` is then the one that should become more
        general, and it receives Type Ib.

        Crediting a *random* included positive literal instead -- including ones that were
        satisfied -- blames layer-1 clauses that did their job, and erases layer 1 within an
        epoch. Type Ib is position-free in the library, so no patch has to be carried down.
        """
        l2 = self.l2
        kh2, kw2 = l2.patch_size
        n_pix2 = C1 * kh2 * kw2
        pos_mask = l2.included_mask()[:, :n_pix2].to(l2.compute_dtype)
        B, P2 = viol2.shape[0], viol2.shape[1]
        flat2 = lits2.reshape(B * P2, -1)
        nz = torch.nonzero(sel, as_tuple=True)
        if nz[0].numel() == 0:
            return None
        b_idx, j_idx = nz
        if b_idx.numel() > self.max_events:
            keep = torch.randperm(b_idx.numel(), device=sel.device)[: self.max_events]
            b_idx, j_idx = b_idx[keep], j_idx[keep]
        bs, ks = [], []
        for sl in chunk_indices(b_idx.numel(), 4096):
            bi, ji = b_idx[sl], j_idx[sl]
            pstar = viol2[bi, :, ji].argmin(dim=1)                     # closest patch
            row = flat2[bi * P2 + pstar][:, :n_pix2]                   # positive pixel literals
            blame = pos_mask[ji] * (1.0 - row)                         # included AND false
            m = (torch.rand(blame.shape, device=blame.device) * blame).argmax(dim=1)
            ok = blame.sum(dim=1) > 0
            if not bool(ok.any()):
                continue
            bs.append(bi[ok])
            ks.append(torch.div(m[ok], kh2 * kw2, rounding_mode="floor"))
        if not bs:
            return None
        bi, kk = torch.cat(bs), torch.cat(ks)
        self.stats["events_ib"] += int(bi.numel())
        if self.dedup:
            bi, kk = self._dedup(bi, kk, C1)
        (kk,) = self._thin(kk)
        if budget is not None and kk.numel() > budget:
            keep = torch.randperm(kk.numel(), device=kk.device)[:budget]
            kk = kk[keep]
        self.stats["kept_ib"] += int(kk.numel())
        return kk

    @torch.no_grad()
    def _thin(self, *tensors: Tensor) -> Tuple[Tensor, ...]:
        """Keep a ``credit_rate`` fraction of the events -- the step size of the credit path.

        Layer-2 clauses are conjunctions over *named channels*. When layer-1 clause ``k``
        changes, every layer-2 clause that includes channel ``k`` silently becomes a different
        rule, and one boosted Type Ia event rewrites a layer-1 clause wholesale. Thinning is
        the only knob that trades that representation drift against learning speed, so it is
        swept rather than assumed.
        """
        if self.credit_rate >= 1.0:
            return tensors
        n = tensors[0].numel()
        keep = torch.rand(n, device=tensors[0].device) < self.credit_rate
        return tuple(t[keep] for t in tensors)

    @torch.no_grad()
    def _dedup(self, bi: Tensor, kk: Tensor, C1: int) -> Tuple[Tensor, Tensor]:
        """Keep one event per ``(example, layer-1 clause)``, as in the classical algorithm
        where an example gives a clause at most one feedback event. Without it 512 layer-2
        clauses credit 128 layer-1 clauses on every image, and layer 1 sees orders of
        magnitude more events than it was designed for."""
        key = bi * C1 + kk
        order = torch.rand(key.numel(), device=key.device).argsort()
        win = torch.full((int(key.max()) + 1,), -1, dtype=torch.long, device=key.device)
        win[key[order]] = order
        keep = win[key] == torch.arange(key.numel(), device=key.device)
        return bi[keep], kk[keep]

    @torch.no_grad()
    def _credit(self, xb: Tensor, raw_maps: Tensor, matches2: Tensor, viol2: Tensor,
                lits2: Tensor, sel_i: Tensor, sel_ib: Tensor, sel_ii: Tensor) -> None:
        """Turn layer-2 feedback events into layer-1 feedback and apply it."""
        l1, l2 = self.l1, self.l2
        B = xb.shape[0]
        C1 = raw_maps.shape[1]
        Py2, Px2 = l2._grid(*self.maps_shape[1:])
        lits1 = l1._encode(l1._prepare(xb))            # (B, P1, 2F1)
        P1 = lits1.shape[1]
        flat1 = lits1.reshape(B * P1, -1)
        cd = l1.compute_dtype
        acc_true = torch.zeros(C1, l1.n_literals, device=xb.device, dtype=cd)
        acc_false = torch.zeros_like(acc_true)
        acc_n2 = torch.zeros_like(acc_true)
        acc_ib = torch.zeros(C1, device=xb.device, dtype=cd)
        n_ia = 0
        for sel, kind in ((sel_i, "i"), (sel_ii, "ii")):
            if kind == "ii" and not self.credit_type_ii:
                continue
            ev = self._events(kind, sel, matches2, raw_maps, Py2, Px2)
            if ev is None:
                continue
            bi, kk, pp = ev
            if kind == "i":
                n_ia = int(kk.numel())
            for sl in chunk_indices(bi.numel(), 8192):
                rows = flat1[bi[sl] * P1 + pp[sl]]
                if kind == "i":
                    acc_true.index_add_(0, kk[sl], rows)
                    acc_false.index_add_(0, kk[sl], 1.0 - rows)
                else:
                    acc_n2.index_add_(0, kk[sl], 1.0 - rows)
        if self.type_ib:
            # Type Ia and Type II both only ever ADD literals; Type Ib is the sole term that
            # removes them, so their ratio sets layer 1's equilibrium clause size. Layer-2
            # clauses match rarely, which makes raw Ib events outnumber Ia by ~100:1 and
            # erases layer 1; ``ib_ratio`` caps the Ib budget at a multiple of the Ia count
            # actually delivered this batch. 0 leaves the raw rule in place.
            budget = int(self.ib_ratio * n_ia) if self.ib_ratio else None
            kk = self._events_ib(sel_ib, viol2, lits2, C1, budget=budget)
            if kk is not None and kk.numel():
                acc_ib.index_add_(0, kk, torch.ones_like(kk, dtype=cd))
        TTF.apply_feedback(
            l1.ta_state, acc_true, acc_false, acc_ib, acc_n2,
            n_states=l1.n_states, s=l1.s, boost_true_positive=l1.boost_true_positive,
            max_included_literals=l1.max_included_literals, include_count=l1.include_count)
        l1._refresh_include()

    # ---- one learning step --------------------------------------------------------
    @torch.no_grad()
    def update(self, xb: Tensor, y: Tensor) -> Tensor:
        """One joint step: layer 2 learns from its own feedback, layer 1 from layer 2's."""
        l2 = self.l2
        raw, pooled = self.maps(xb, raw=True)
        self.maps_shape = tuple(pooled.shape[1:])
        mb = l2._prepare(pooled)
        y_t = l2._coerce_targets(y, mb.shape[0], mb.device)
        lits2 = l2._encode(mb)
        # clause_violations serves twice: matches2 == (viol2 == 0) is exactly what
        # _evaluate(empty_value=True) returns, and the counts are what the Type Ib blame rule
        # needs to find the patch a clause came closest to matching.
        Bm, P2, L2n = lits2.shape
        viol2 = TTF.clause_violations(lits2.reshape(Bm * P2, L2n), l2.include).view(Bm, P2, -1)
        matches2 = viol2 == 0
        out2 = matches2.any(dim=1)
        votes = l2._votes(out2)
        type_i, type_ii, aux = l2._select_feedback(votes, y_t, out2)
        cd = l2.compute_dtype
        sel_fire_i = (type_i & out2).to(cd)
        sel_nofire_i = (type_i & ~out2).to(cd)
        sel_fire_ii = (type_ii & out2).to(cd)
        acc = l2._new_accumulator()
        l2._feedback_counts(lits2, matches2, sel_fire_i, sel_nofire_i, sel_fire_ii, acc)
        l2._accumulate_weights(acc, aux, out2, sel_fire_i, sel_fire_ii)
        # Layer 1 is credited from the SAME events, before layer 2 commits: both layers see
        # the state that produced the events, as one simultaneous step.
        self._credit(xb, raw, matches2, viol2, lits2,
                     type_i & out2, type_i & ~out2, type_ii & out2)
        l2._commit(acc)
        self.stats["batches"] += 1
        return votes

    def n_clauses_total(self) -> int:
        return int(self.l1.n_clauses_total) + int(self.l2.n_clauses_total)

    def n_automata(self) -> int:
        return int(self.l1.ta_state.numel()) + int(self.l2.ta_state.numel())
