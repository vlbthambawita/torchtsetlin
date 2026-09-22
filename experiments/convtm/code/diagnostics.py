"""Cheap per-epoch diagnostics for clause-based arms.

PLAN Section 7.6: *"collecting them after the fact means re-running everything"*. Everything
here is designed to cost a small constant fraction of an epoch so that it can be switched on
for every run of the programme without a budget argument:

* **clause length** (median / mean / p95) and **negation fraction** are read straight off the
  automata -- no data, no forward pass, microseconds.
* **firing rate**, **dead / saturated fraction** and **clauses-used fraction** need one
  ``eval()``-mode forward pass over a small fixed probe set (default 2000 validation images),
  i.e. ~4% of a 45000-image epoch.
* **M2, the per-clause match count** |M_j(x)| -- how many patches a *firing* clause actually
  matches -- is a reduction over the ``(B, P, C)`` tensor the convolutional forward pass
  already computes, on a smaller probe (default 512 images).

Why these four, specifically: they are the measured signature of the failure the previous
programme hit. ``experiments/mctm`` diagnosed density collapse by exactly this pair --
trained layer-1 clauses firing on 0.14% of positions while the negation fraction rose to
97.4% against a healthy 57-60% [MEASURED: experiments/mctm/report2]. A run whose clauses are
degenerating says so in these numbers several epochs before it says so in the accuracy.

Two subtleties that make the numbers mean what they say:

1. **``model.eval()`` is mandatory.** A clause with no included literals evaluates True while
   ``training`` and False while predicting (repo ``CLAUDE.md``). Probed in train mode, every
   clause of a freshly initialised model reads as firing on 100% of images. ``firing_stats``
   forces eval mode and restores the previous mode.
2. **Why M2 exists.** The CTM aggregates patches with an OR, and a P5 candidate replaces that
   OR with a *count* threshold ("clause fires if at least k patches match"). That candidate is
   dead on arrival if a firing clause almost never matches more than one patch:
   [MEASURED: experiments/mctm] trained conv clauses fire on ~0.14% of patch positions, about
   1.2 of 841. But that is an *unconditional* mean, and a mean near 1 is equally consistent
   with "everything matches one patch" (count buys nothing) and with "a few match many, the
   rest match none" (count buys a lot). The decisive statistic is therefore
   ``P(|M_j| >= 5 | |M_j| >= 1)``, reported as ``frac_ge5_given_firing``. Collected per epoch
   because collecting it afterwards means re-running P3.

   M2 uses the **RNG-free** path: ``_evaluate`` returns the match tensor and draws nothing
   (the random matching patch is drawn in ``_feedback_counts``), which is the property
   ``tests/test_models.py::test_conv_prediction_draws_no_random_patches`` pins. The
   distribution is accumulated as a histogram over patch counts, so memory is bounded by the
   number of patch positions rather than by the probe size.

3. **Image-level vs patch-level firing.** For a convolutional model the default here is the
   *image-level* rate (the clause fires on an image iff it matches at least one patch) --
   this is what the vote actually sees, and it is one cheap forward pass. The patch-level
   rate (the quantity the mctm density controller targeted) is strictly smaller and costs the
   full ``(B, P, C)`` map; ``patch_firing_stats`` computes it on demand, and it is *not* run
   per epoch.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

DEAD = 1e-4      # a clause firing on < 0.01% of probe images is dead
SATURATED = 0.99  # ... on > 99% is saturated (a constant, carrying no information)


@torch.no_grad()
def clause_composition(model) -> Dict[str, float]:
    """Clause length and how much of it is negation. Reads the automata; no data needed."""
    inc = model.included_mask()               # (C, 2F) bool
    Fn = int(model.n_features)
    pos = inc[:, :Fn].sum(dim=1).float()
    neg = inc[:, Fn:].sum(dim=1).float()
    total = pos + neg
    nonempty = total > 0
    q = torch.tensor([0.5, 0.95], device=total.device)
    med, p95 = torch.quantile(total, q).tolist()
    frac_neg = torch.where(nonempty, neg / total.clamp(min=1), torch.zeros_like(total))
    return {
        "clause_len": {"median": med, "mean": float(total.mean()), "p95": p95,
                       "max": float(total.max()), "min": float(total.min())},
        "negation_fraction": (float(frac_neg[nonempty].mean()) if bool(nonempty.any()) else 0.0),
        "negation_fraction_median": (float(frac_neg[nonempty].median())
                                     if bool(nonempty.any()) else 0.0),
        "empty_clause_frac": float((~nonempty).float().mean()),
    }


@torch.no_grad()
def firing_stats(model, x: Tensor, batch_size: int = 500,
                 y: Optional[Tensor] = None) -> Dict[str, object]:
    """Per-clause image-level firing rate over the probe set ``x``.

    Returns the rate summary plus the Boolean ``ever`` vector (which clauses fired at least
    once), which the caller accumulates across epochs into ``clauses_ever_fired_frac``.

    **D-FIRE (THEORY.md 5.5.9).** When the probe labels ``y`` are supplied and the model is
    class-owned, this also returns ``firing_rate_eligible``: the firing rate of each clause
    restricted to the examples on which that clause is *eligible for Type I feedback*. For a
    positive-polarity clause of class ``c`` that is ``y == c``; for a negative-polarity one it
    is ``y != c`` -- the multi-class rule gives Type I to the positive clauses of the target
    class and to the negative clauses of the sampled contrast class
    ``[FACT: src/torchtsetlin/models/classifier.py:24-30]``.

    This is the quantity THEORY.md C5.2 predicts, and the *global* rate cannot test it: the
    global rate averages positive clauses (eligible on 1/10 of images) with negative ones
    (eligible on 9/10), so it is a polarity-weighted mixture of two very different numbers
    and it cannot be compared with ``1/(1+s)`` at all. The `s` sweep moves ``1/(1+s)`` by 7x
    between ``s = 2`` and ``s = 20``; this field is what that sweep is read against.

    Implementation note: the per-class firing sums are accumulated as a ``(n_classes, C)``
    matrix by one ``index_add_`` per batch, so the eligible rate is **exact** over the probe
    and costs no extra forward pass and no memory that scales with the probe size.
    """
    was_training = model.training
    model.eval()                                   # see module docstring, subtlety 1
    per_class = None
    n_per_class = None
    K = int(getattr(model, "n_classes", 0) or 0)
    use_y = y is not None and K >= 2 and hasattr(model, "clause_class")
    try:
        tot = None
        if use_y:
            per_class = torch.zeros((K, int(model.n_clauses_total)), dtype=torch.float32,
                                    device=model.ta_state.device)
            n_per_class = torch.zeros(K, dtype=torch.float64, device=per_class.device)
        for i in range(0, x.shape[0], batch_size):
            out = model.evaluate_clauses(x[i : i + batch_size]).to(torch.float32)  # (b, C)
            s = out.sum(dim=0)
            tot = s if tot is None else tot + s
            if use_y:
                yb = y[i : i + batch_size].to(per_class.device).long()
                per_class.index_add_(0, yb, out)
                n_per_class += torch.bincount(yb, minlength=K).to(torch.float64)
    finally:
        model.train(was_training)
    n = float(x.shape[0])
    rate = tot / n
    q = torch.tensor([0.05, 0.5, 0.95], device=rate.device)
    q05, med, q95 = torch.quantile(rate, q).tolist()
    ever = rate > 0
    out: Dict[str, object] = {
        "firing_rate": {"median": med, "mean": float(rate.mean()), "q05": q05, "q95": q95,
                        "dead_frac": float((rate < DEAD).float().mean()),
                        "sat_frac": float((rate > SATURATED).float().mean())},
        "clauses_used_frac": float(ever.float().mean()),
        "_ever": ever,
    }
    # D-FIRE is additive, so it is the part allowed to fail soft -- see DiagnosticProbe.epoch.
    out["firing_rate_eligible"] = (_safe("D-FIRE", _eligible_firing, model, tot, per_class,
                                         n_per_class, n)
                                   if use_y else None)
    return out


def _eligible_firing(model, tot: Tensor, per_class: Tensor, n_per_class: Tensor,
                     n: float) -> Dict[str, object]:
    """D-FIRE's arithmetic: turn the ``(K, C)`` per-class firing sums into ``P(fire | eligible)``."""
    cls = model.clause_class.to(per_class.device).long()          # (C,)
    pol = model.clause_polarity.to(per_class.device)              # (C,) +1 / -1
    own = per_class.gather(0, cls.unsqueeze(0)).squeeze(0).to(torch.float64)   # fires with y == c
    n_own = n_per_class.gather(0, cls)                                          # count of y == c
    pos = pol > 0
    num = torch.where(pos, own, tot.to(torch.float64) - own)
    den = torch.where(pos, n_own, n - n_own).clamp_(min=1.0)
    rate = (num / den).to(torch.float32)
    # the complement, for contrast: how often the clause fires where it is NOT Type-I-eligible
    inum = torch.where(pos, tot.to(torch.float64) - own, own)
    iden = torch.where(pos, n - n_own, n_own).clamp_(min=1.0)
    irate = (inum / iden).to(torch.float32)
    s = float(getattr(model, "s", float("nan")))
    pred = 1.0 / (1.0 + s) if s == s and s > 0 else float("nan")
    qq = torch.tensor([0.05, 0.5, 0.95], device=rate.device)
    q05, med, q95 = torch.quantile(rate, qq).tolist()
    mean = float(rate.mean())
    return {
        "definition": "P(clause fires on an image | the image is Type-I-eligible for that "
                      "clause): y == c for a positive-polarity clause of class c, y != c for "
                      "a negative-polarity one [FACT: models/classifier.py:24-30]. This is "
                      "the f of THEORY.md C5.2; the global firing_rate is a mixture over "
                      "polarities and cannot be compared with 1/(1+s).",
        "mean": mean, "median": med, "q05": q05, "q95": q95,
        "pos_mean": (float(rate[pos].mean()) if bool(pos.any()) else float("nan")),
        "neg_mean": (float(rate[~pos].mean()) if bool((~pos).any()) else float("nan")),
        "ineligible_mean": float(irate.mean()),
        "s": s,
        "predicted_1_over_1_plus_s": pred,
        "ratio_to_prediction": (mean / pred if pred and pred == pred else float("nan")),
        "n_probe": int(n),
    }


@torch.no_grad()
def match_count_stats(model, x: Tensor, max_elements: int = 64_000_000) -> Dict[str, object]:
    """M2: the distribution of |M_j(x)|, the number of patches a clause matches.

    **The statistic that matters is conditional.** A mean near 1 is compatible with two
    opposite distributions: almost every clause matching about one patch, in which case a
    count carries nothing an indicator does not and the counting-pool branch of P5 is dead;
    or a small fraction of clauses matching many patches while the rest match none, in which
    case it carries a great deal. The mean cannot tell them apart, so the decisive number is
    ``p_ge5_given_fires = P(|M_j| >= 5 | |M_j| >= 1)``. The unconditional mean is reported
    too, because that is what is comparable to the 0.14%-firing figure in PLAN Section 3.1.

    Field names mirror ``code/cnn/arms_cnn.py::CnnArm.match_count_stats`` so that the TM-side
    and CNN-side numbers sit in one schema. **The two "match" definitions are not the same
    object** and that is recorded in ``definition``, not hidden: here a match is a satisfied
    conjunction of included literals; there it is a post-BatchNorm post-activation value > 0,
    i.e. a thresholded real number.

    Exact, not sampled: a full ``0..P`` histogram, accumulated in float64, so quantiles are
    exact over the probed images and memory does not grow with the probe size. The inner
    batch is sized so that ``B*P*C`` stays under ``max_elements``.

    Reduces the ``(B, P, C)`` tensor ``_ConvMixin._evaluate`` already computes; draws no
    random patches (the draw lives in ``_feedback_counts``), so this is the RNG-free
    prediction path. Returns ``{}`` for a non-convolutional model, where the count is 1 by
    construction.
    """
    if not hasattr(model, "_grid") or getattr(model, "input_shape", None) is None:
        return {}
    Py, Px = model._grid(model.input_shape[1], model.input_shape[2])
    P, C = Py * Px, int(model.n_clauses_total)
    bs = max(1, min(int(x.shape[0]), max_elements // max(1, P * C)))
    was_training = model.training
    model.eval()                                   # prediction semantics for empty clauses
    try:
        hist = torch.zeros(P + 1, dtype=torch.float64, device=x.device)
        for i in range(0, x.shape[0], bs):
            xb = model._prepare(x[i : i + bs])
            _, matches = model._evaluate(model._encode(xb), empty_value=False)  # (b, P, C)
            counts = matches.sum(dim=1).reshape(-1)                             # (b*C,)
            hist += torch.bincount(counts, minlength=P + 1).to(torch.float64)
            del matches, counts
    finally:
        model.train(was_training)

    h = hist.cpu()
    total = float(h.sum())
    firing = h.clone()
    firing[0] = 0.0                                # condition on the clause firing at all
    n_fire = float(firing.sum())
    idx = torch.arange(P + 1, dtype=torch.float64)
    nan = float("nan")

    def quantile(counts: Tensor, q: float) -> float:
        n = float(counts.sum())
        if n <= 0:
            return nan
        c = torch.cumsum(counts, 0)
        return float(torch.searchsorted(c, torch.tensor(q * n)).item())

    return {
        "definition": "count of patch positions at which the clause's conjunction of "
                      "included literals is satisfied (exact logical match)",
        "split": "validation",
        "n_images": int(x.shape[0]),
        "n_patches": P,
        "n_pairs": total,
        "fire_frac": (n_fire / total) if total else nan,
        # --- conditional on the clause firing at least once (the decisive distribution) ---
        "median_given_firing": quantile(firing, 0.5),
        "mean_given_firing": (float((idx * firing).sum()) / n_fire) if n_fire else nan,
        "q25_given_firing": quantile(firing, 0.25),
        "q75_given_firing": quantile(firing, 0.75),
        "p95_given_firing": quantile(firing, 0.95),
        "max_given_firing": (float(idx[int((firing > 0).nonzero()[-1].item())])
                             if n_fire else nan),
        "frac_ge5_given_firing": (float(firing[5:].sum()) / n_fire) if n_fire else nan,
        "frac_eq1_given_firing": (float(firing[1]) / n_fire) if n_fire else nan,
        # --- unconditional, comparable to the 0.14%-of-positions figure in PLAN 3.1 --------
        "mean_all_pairs": (float((idx * h).sum()) / total) if total else nan,
        "median_all_pairs": quantile(h, 0.5),
    }


# ======================================================================================
# The three mechanism diagnostics of THEORY.md 5.5.9 (D-TA / D-FIRE / D-ORDER).
#
# Added 2026-09-21, BEFORE jobs/p3_budget_predictions.txt ran, so that the clause-size-budget
# sweep tests the proposed mechanism directly instead of only its accuracy consequence.
# Everything below is additive: no existing field changes meaning, and no existing number
# moves. In particular **nothing here draws from a CUDA RNG stream** -- the only random draw
# is a clause subsample taken from an explicit CPU generator -- so adding them cannot perturb
# training. That is checked, not assumed: `ctm-small-T80-s10-{unc,b32}` re-run the pre-existing
# `ctm-small-T80-{unc,b32}` cells at the same seeds on the same card and must reproduce them
# (AUDIT A-D1).
#
# Each is wrapped by `_safe` at the call site: a diagnostic that raises records
# `{"error": ...}` and prints a loud WARNING rather than killing a multi-hour training run it
# is only observing. The error is IN the record, so a failed diagnostic is visible to the
# aggregator and to the audit; it is never silent.
# ======================================================================================


@torch.no_grad()
def ta_state_stats(model, max_elements: int = 4_000_000) -> Dict[str, object]:
    """**D-TA** -- the TA-state histogram over *included* literals.

    Tests THEORY.md C5.3. Type Ia re-memorises an included literal on every Type Ia event
    (once included it is True in every drawn matching patch), so it drives the automaton all
    the way to ``2N-1``; Type II is clipped at the include boundary
    (``room = N - state``, ``functional.py:256-257``) and can therefore only ever place a
    literal at **exactly** ``N``, one Type Ib decrement from eviction. The predicted shape is
    bimodal: a deep core at ``2N-1`` and a shallow Type-II fringe at ``N``. The clause-size
    budget is predicted to strip the fringe first, because eviction time grows with depth.

    ``frac_at_boundary`` is ``P(state == N | included)``. **It is an upper bound on the
    Type-II fringe, not an estimate of it**: a literal climbing under Type Ia also passes
    through ``N``, and with ``init="constant"`` every automaton starts at ``N-1``, one step
    below. The bound is the honest reading and it is the one C5.3 needs, since the claim is
    that the fringe is *at most* this large and the deep core *at least* ``frac_deep``.

    Exact: a full ``0 .. 2N-1`` bincount, accumulated over row chunks so the int64 cast is
    bounded by ``max_elements`` regardless of clause count. No data, no forward pass.
    """
    st = model.ta_state
    if st.numel() == 0:
        return {}
    N = int(model.n_states)
    rows, cols = st.shape
    chunk = max(1, min(rows, int(max_elements) // max(1, cols)))
    hist = torch.zeros(2 * N, dtype=torch.float64, device=st.device)
    for i in range(0, rows, chunk):
        flat = st[i : i + chunk].reshape(-1).to(torch.long).clamp_(0, 2 * N - 1)
        hist += torch.bincount(flat, minlength=2 * N).to(torch.float64)
    h = hist.cpu()
    inc = h[N:]                                   # included: state >= N
    n_inc = float(inc.sum())
    n_all = float(h.sum())
    if n_inc <= 0:
        return {"n_states": N, "n_included": 0, "n_automata": int(n_all),
                "definition": "no included literals"}
    idx = torch.arange(N, dtype=torch.float64)    # depth 0 .. N-1 above the boundary
    deep_from = (N + N // 2) - N                  # state >= 1.5N
    csum = torch.cumsum(inc, 0)
    med = int(torch.searchsorted(csum, torch.tensor(0.5 * n_inc)).item())
    # 16 equal-width bins over [N, 2N-1], reported as fractions of the included mass.
    nb = 16
    per = max(1, N // nb)
    binned = [float(inc[j * per : (j + 1) * per].sum()) / n_inc for j in range(nb)]
    return {
        "definition": "histogram of ta_state over INCLUDED literals (state >= n_states). "
                      "depth = (state - n_states) / (n_states - 1) in [0,1]: 0 = parked at "
                      "the include boundary, 1 = saturated at 2N-1",
        "n_states": N,
        "n_included": int(n_inc),
        "n_automata": int(n_all),
        "include_frac": n_inc / n_all,
        # --- C5.3's two populations -----------------------------------------------------
        "frac_at_boundary": float(inc[0]) / n_inc,          # state == N  (upper bound)
        "frac_deep": float(inc[deep_from:].sum()) / n_inc,  # state >= 1.5N
        "frac_saturated": float(inc[-1]) / n_inc,           # state == 2N-1
        "mean_depth": float((idx * inc).sum()) / n_inc / max(1, N - 1),
        "median_depth": med / max(1, N - 1),
        "hist_included_16": binned,
        # --- the other side of the boundary, free from the same histogram ---------------
        "frac_excluded_at_boundary": (float(h[N - 1]) / (n_all - n_inc)
                                      if n_all > n_inc else float("nan")),
    }


@torch.no_grad()
def order_stats(model, x: Tensor, ks: Sequence[int] = (8, 16, 32),
                n_clauses_sample: int = 2000, max_elements: int = 32_000_000,
                generator: Optional[torch.Generator] = None) -> Dict[str, object]:
    """**D-ORDER** -- marginal selectivity by inclusion depth.

    THEORY.md 5.5.3 shows *in aggregate over epochs* that the literals an unconstrained clause
    admits after epoch 5 do not shrink its match set. This shows the same thing **inside a
    single trained clause**: sort the clause's included literals by TA state (deepest first,
    i.e. earliest-and-most-often memorised) and evaluate the top-``k`` prefix as a clause in
    its own right. If the late arrivals are logically redundant, the prefix matches at
    essentially the same rate as the whole clause.

    **The caveat, stated because it is load-bearing.** The ordering is by TA state, and C5.3
    predicts a large mass at exactly ``2N-1``; within that saturated core the order is
    arbitrary (``topk`` breaks ties by index). ``frac_tied_at_k`` reports the fraction of
    clauses whose k-th and (k+1)-th states are *equal*, i.e. for which the prefix is an
    arbitrary rather than a meaningful subset. Read a level with ``frac_tied_at_k`` near 1 as
    "a random k-subset of the deep core", which is still informative but is a different
    statement.

    Sampling, and it is in the record: ``n_images`` images and ``n_clauses_sampled`` clauses.
    The clause subsample is drawn from an explicit **CPU** generator, so this never touches the
    CUDA RNG stream the feedback rule draws its matching patches from.
    """
    if model.ta_state.numel() == 0 or x.shape[0] == 0:
        return {}
    from torchtsetlin import functional as TF  # noqa: N813  read-only library use (C1)

    N = int(model.n_states)
    C, L = model.ta_state.shape
    dev = model.ta_state.device
    if n_clauses_sample and n_clauses_sample < C:
        g = generator if generator is not None else torch.Generator().manual_seed(20260921)
        sel = torch.randperm(C, generator=g)[:n_clauses_sample].sort().values.to(dev)
        sampled = True
    else:
        sel = torch.arange(C, device=dev)
        sampled = False
    st = model.ta_state[sel].to(torch.int64)           # (Cs, 2F)
    Cs = int(st.shape[0])
    included = st >= N
    inc_count = included.sum(dim=1)
    masked = torch.where(included, st, torch.full_like(st, -1))

    levels = []
    was_training = model.training
    model.eval()                                       # prediction semantics for empty clauses
    try:
        for k in list(ks) + [None]:
            if k is None:
                inc_mask = included.to(model.compute_dtype)
                count = inc_count.clone()
                tied = float("nan")
            else:
                kk = min(int(k) + 1, int(L))
                vals, idx = masked.topk(kk, dim=1)
                keep = vals[:, : int(k)] >= N
                inc_mask = torch.zeros((Cs, L), dtype=model.compute_dtype, device=dev)
                inc_mask.scatter_(1, idx[:, : int(k)], keep.to(model.compute_dtype))
                count = keep.sum(dim=1)
                over = inc_count > int(k)
                tied = (float((vals[over, int(k) - 1] == vals[over, int(k)]).float().mean())
                        if bool(over.any()) and kk > int(k) else float("nan"))
            n_patch = n_img = 0
            fired_patch = fired_img = 0.0
            bs = max(1, min(int(x.shape[0]),
                            int(max_elements) // max(1, Cs * _n_patches(model))))
            for i in range(0, x.shape[0], bs):
                lits = model._encode(model._prepare(x[i : i + bs]))
                if lits.dim() == 2:                    # flat model: one "patch" per image
                    lits = lits.unsqueeze(1)
                b, P, Lx = lits.shape
                m = TF.clause_outputs(lits.reshape(b * P, Lx), inc_mask, count,
                                      empty_value=False).view(b, P, Cs)
                fired_patch += float(m.sum())
                fired_img += float(m.any(dim=1).sum())
                n_patch += b * P * Cs
                n_img += b * Cs
                del lits, m
            p_patch = fired_patch / max(1, n_patch)
            mean_len = float(count.to(torch.float64).mean())
            levels.append({
                "k": (None if k is None else int(k)),
                "label": ("all" if k is None else str(int(k))),
                "mean_len": mean_len,
                "p_patch": p_patch,
                "p_image": fired_img / max(1, n_img),
                # per-literal selectivity, the -ln q of THEORY.md 5.5.3
                "neg_log_q": ((-torch.log(torch.tensor(p_patch)).item() / mean_len)
                              if p_patch > 0 and mean_len > 0 else float("nan")),
                "frac_tied_at_k": tied,
            })
    finally:
        model.train(was_training)

    allv = levels[-1]
    return {
        "definition": "match probability of the top-k prefix of each clause's included "
                      "literals, ordered by TA state (deepest first). k=null is the whole "
                      "clause. p_patch = per-patch match probability, p_image = the clause "
                      "fires on the image (matches >= 1 patch). A prefix is a WEAKER "
                      "condition than the whole clause, so p_image(k) >= p_image(all) always "
                      "and p_image_ratio_32_over_all >= 1: it equals 1 exactly when the "
                      "literals below rank 32 are logically redundant on this probe, which "
                      "is the claim THEORY.md 5.5.3 makes about late arrivals.",
        "n_images": int(x.shape[0]),
        "n_clauses_sampled": Cs,
        "sampled": sampled,
        "levels": levels,
        # The headline: how much of the whole clause's selectivity the top 32 already buys.
        "p_image_ratio_32_over_all": (
            next((lv["p_image"] for lv in levels if lv["k"] == 32), float("nan"))
            / allv["p_image"] if allv["p_image"] > 0 else float("nan")),
    }


def _n_patches(model) -> int:
    if hasattr(model, "_grid") and getattr(model, "input_shape", None) is not None:
        Py, Px = model._grid(model.input_shape[1], model.input_shape[2])
        return Py * Px
    return 1


def _safe(name: str, fn, *args, **kw):
    """Run a diagnostic; on failure record the error and warn loudly, never kill the run.

    These are observers bolted onto training runs that cost hours. A diagnostic that OOMs on
    the largest arm must not destroy the arm -- but it must also not vanish, so the error text
    goes into the record under its own key and a WARNING line goes into the job log, where
    crash accounting and AUDIT.md will find it.
    """
    try:
        return fn(*args, **kw)
    except Exception as exc:                       # noqa: BLE001 -- deliberate, see docstring
        print(f"  !! diagnostic {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
        # The realistic failure on the largest arms is OOM. Release the diagnostic's caching
        # allocator blocks so that the *training* step after this one is not starved by the
        # fragments of an observer that already gave up.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"error": f"{type(exc).__name__}: {exc}"}


@torch.no_grad()
def patch_firing_stats(model, x: Tensor, batch_size: int = 128) -> Dict[str, float]:
    """Per-clause *patch-level* firing rate for a convolutional model (expensive; on demand).

    Uses the model's private encode/evaluate hooks to reach the ``(B, P, C)`` match tensor
    the public API discards -- the same route ``experiments/mctm/mctm.py`` takes. Not called
    per epoch: it materialises ``B*P*C`` bits.
    """
    was_training = model.training
    model.eval()
    try:
        tot, n = None, 0
        for i in range(0, x.shape[0], batch_size):
            xb = model._prepare(x[i : i + batch_size])
            _, matches = model._evaluate(model._encode(xb), empty_value=False)  # (b, P, C)
            s = matches.to(torch.float32).sum(dim=(0, 1))
            tot = s if tot is None else tot + s
            n += matches.shape[0] * matches.shape[1]
    finally:
        model.train(was_training)
    rate = tot / float(n)
    q = torch.tensor([0.5], device=rate.device)
    return {"patch_firing_median": float(torch.quantile(rate, q).item()),
            "patch_firing_mean": float(rate.mean()),
            "patch_dead_frac": float((rate < DEAD).float().mean())}


class DiagnosticProbe:
    """Per-epoch diagnostics with a fixed probe set and a running 'ever fired' accumulator.

    ``epoch(model)`` returns a flat dict ready to hand to ``record.add_epoch``; ``summary()``
    returns the last epoch's numbers plus ``clauses_ever_fired_frac`` -- the fraction of
    clauses that contributed at *any* point in training, which separates "this clause is
    currently silent" from "this clause was never used at all".
    """

    def __init__(self, x_probe: Tensor, batch_size: int = 500, n_match: int = 2000,
                 x_match: Optional[Tensor] = None, y_probe: Optional[Tensor] = None,
                 n_order: int = 256, order_clauses: int = 2000, seed: int = 0):
        self.x = x_probe
        # D-FIRE needs the probe LABELS to condition on Type-I eligibility. They come from
        # the same split and the same slice as `x_probe`; None disables D-FIRE rather than
        # guessing.
        self.y = y_probe
        self.batch_size = batch_size
        # M2 reduces a (B, P, C) tensor, so it gets its own probe, and it is taken from the
        # *validation* split so that it is directly comparable to the CNN-side measurement.
        xm = x_probe if x_match is None else x_match
        self.x_match = xm[: min(n_match, xm.shape[0])]
        # D-ORDER evaluates one masked clause set per k, so it gets a SMALL probe and a
        # clause subsample. Both are in the record (`n_images`, `n_clauses_sampled`).
        self.x_order = x_probe[: min(int(n_order), x_probe.shape[0])] if n_order else None
        self.order_clauses = int(order_clauses)
        # An explicit CPU generator: the clause subsample must never draw from the CUDA RNG
        # stream that `_feedback_counts` draws its matching patches from, or the diagnostic
        # would change the training run it is observing.
        self.gen = torch.Generator().manual_seed(int(seed) + 911_382)
        self.ever: Optional[Tensor] = None
        self.last: Dict[str, object] = {}

    def epoch(self, model) -> Dict[str, object]:
        d: Dict[str, object] = dict(clause_composition(model))
        # NOT wrapped by `_safe`: `firing_rate`, `clauses_used_frac` and `match_count` are
        # REQUIRED by record.validate(). Swallowing a failure there would convert an
        # immediate crash into an invalid record written after hours of training -- strictly
        # worse. Only the additive fields are made fail-soft.
        f = firing_stats(model, self.x, self.batch_size, self.y)
        ever = f.pop("_ever")
        self.ever = ever if self.ever is None else (self.ever | ever)
        d.update(f)
        d["clauses_ever_fired_frac"] = float(self.ever.float().mean())
        d["match_count"] = match_count_stats(model, self.x_match) or None
        d["ta_state"] = _safe("D-TA", ta_state_stats, model) or None
        d["order"] = (_safe("D-ORDER", order_stats, model, self.x_order,
                            n_clauses_sample=self.order_clauses, generator=self.gen) or None
                      if self.x_order is not None else None)
        self.last = d
        return d

    def summary(self, model=None) -> Dict[str, object]:
        d = dict(self.last)
        if model is not None:                      # refresh composition from the final state
            d.update(clause_composition(model))
            d["ta_state"] = _safe("D-TA", ta_state_stats, model) or None
        d["n_probe"] = int(self.x.shape[0])
        return d


def one_line(d: Dict[str, object]) -> str:
    """Compact per-epoch log line; the shape a degenerating run shows up in."""
    cl, fr = d.get("clause_len", {}), d.get("firing_rate", {})
    mc = d.get("match_count") or {}
    fe = d.get("firing_rate_eligible") or {}
    ta = d.get("ta_state") or {}
    od = d.get("order") or {}
    fe = {} if "error" in fe else fe          # a failed diagnostic is reported, not logged
    ta = {} if "error" in ta else ta
    od = {} if "error" in od else od
    return (f"len {cl.get('median', float('nan')):.0f}/{cl.get('p95', float('nan')):.0f} "
            f"neg {d.get('negation_fraction', float('nan')):.2f} "
            f"fire {fr.get('median', float('nan')):.3f} "
            f"dead {fr.get('dead_frac', float('nan')):.2f} "
            f"sat {fr.get('sat_frac', float('nan')):.2f} "
            f"used {d.get('clauses_used_frac', float('nan')):.2f}"
            + (f" |M|fire {mc.get('median_given_firing', 0):.0f}/"
               f"{mc.get('p95_given_firing', 0):.0f} "
               f"P(>=5|fire) {mc.get('frac_ge5_given_firing', 0):.2f}" if mc else "")
            + (f" fireE {fe.get('mean', float('nan')):.3f}"
               f"(x{fe.get('ratio_to_prediction', float('nan')):.2f})" if fe else "")
            + (f" ta {ta.get('frac_at_boundary', float('nan')):.2f}/"
               f"{ta.get('frac_deep', float('nan')):.2f}" if ta else "")
            + (f" ord32/all {od.get('p_image_ratio_32_over_all', float('nan')):.2f}"
               if od else ""))
