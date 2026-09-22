"""The arm registry.

Nothing runs that is not registered here (PLAN Section 6.3). An arm is an ``ArmSpec``: a
family, the paper it re-implements, a builder, its default hyperparameters, and one sentence
for ``ARMS.md``. The object the builder returns implements the thin protocol of
``code/CONTRACT.md`` -- ``fit_epoch`` / ``predict`` / ``capacity`` / ``diagnostics`` -- plus
``snapshot`` / ``restore``, which the runner needs to evaluate the test set *once*, at the
epoch validation selected, without ever looking at test accuracy during training.

Two harness-wide conventions, both chosen so that PLAN Section 7.3 budget matching is
possible at all, and both recorded in ``hp`` of every record:

**``n_clauses`` is always the TOTAL clause count.** A class-owned model (Granmo's
multi-class TM: every class owns its own clauses, half positive, half negative polarity)
divides it by ``n_classes``; a coalesced model (one shared pool, per-class weights) uses it
directly. Without this, "8000 clauses" means 8000 or 80000 depending on the arm and every
clause-budget comparison in the report is wrong.

**Every arm with a paper must carry that paper's own ``T`` and ``s``.** There is no safe
shared default: measured across the bibliography, ``T / (total clauses)`` spans **0.025 to
1.5** -- Drop Clause 0.80, GraphTM/CoTM 0.19, the Toolbox colour thermometers 1.50, Composites
HOG 0.025 [FACT: LIBRARY_GAPS LG-010 A3]. An arm whose ``T`` came from the harness fallback is
measuring our guess, so its record carries ``hp.config_status = "provisional"`` and
``hp.T_source = "ratio-default"``, and ``record.admissible_as_reproduction`` refuses it as
evidence for a reproduction or a ``GAP`` verdict.

The fallback that remains, for arms with **no** paper (our own controls and candidates), is
``T = T_ratio x`` (clauses that vote for one class) -- ``n_clauses / n_classes`` for a
class-owned model, ``n_clauses`` for a coalesced one. ``T`` must be commensurate with the
achievable vote sum; above it, the feedback probability is pinned near 0.5 for every example
and the margin stops doing anything.

**``max_chunk_elements`` is a throughput dial, and arms are compared only at equal values.**
Chunking is a pure partition of the mini-batch -- feedback accumulates across chunks and is
committed once [FACT: ``models/base.py:305-322``] -- but the library default serialises large
convolutional models (LG-004), so it is a hyperparameter of the *measurement*, recorded in
``hp`` like the batch size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
from diagnostics import DiagnosticProbe, clause_composition
from torch import Tensor

import torchtsetlin as tt

N_CLASSES = 10
"""Default number of output classes. An arm may override it with ``cfg['n_classes']`` --
``ctm-small-cifar2-*`` does, via the ``cifar2`` label map in ``data.py``. It is *not* a
constant of the harness: a class-owned model's clause-per-class count, its vote capacity and
therefore its ``T`` all divide by it, so an arm that remaps labels and leaves this at 10
silently trains 10 clause banks of which 8 never receive a positive example."""


# --------------------------------------------------------------------------- spec
@dataclass
class ArmSpec:
    family: str                       # baseline | existing | candidate | control
    paper: Optional[str]
    build: Callable[[dict], object]   # cfg -> arm object (the protocol below)
    defaults: Dict[str, object]       # hyperparameters, overridable from the CLI
    doc: str                          # one sentence for ARMS.md
    notes: str = ""                   # ambiguities / deviations, copied into the record
    config_status: str = "provisional"  # "paper" once T, s and the clause budget are the
                                        # paper's own; "own" for arms with no paper to
                                        # reproduce; see LG-010 and ARMS.md A1-A4
    config_gaps: Tuple[str, ...] = ()
    """What is still *not* the paper's, in words. An arm can have the paper's `T` and `s` and
    still be provisional for another reason — an ambiguous clause convention, an unstated
    thermometer bit depth. Printed by the runner and recorded in `hp.config_gaps`, so a reader
    never has to guess which part of a provisional configuration is ours."""
    smoke: Dict[str, str] = field(default_factory=dict)
    """Extra CLI flags used by `code/smoke.py` only. A smoke test proves the *path* works, so an
    arm whose per-epoch probes dominate at its real size caps them here rather than being left
    untested or smoke-tested at a size it never runs at."""


ARMS: Dict[str, ArmSpec] = {}


def register(name: str, spec: ArmSpec) -> None:
    if name in ARMS:
        raise ValueError(f"arm {name!r} is already registered")
    ARMS[name] = spec


# --------------------------------------------------------------------------- arm protocol
class Arm:
    """Base class implementing the runner-facing protocol. Subclass it or duplicate it;
    the runner only ever calls these six methods."""

    def fit_epoch(self, xtr: Tensor, ytr: Tensor, batch_size: int) -> None: ...
    def predict(self, x: Tensor) -> Tensor: ...
    def capacity(self) -> dict: ...
    def diagnostics(self) -> dict: ...
    def snapshot(self) -> dict: ...
    def restore(self, snap: dict) -> None: ...


class TMArm(Arm):
    """A Tsetlin-machine arm: one ``torchtsetlin`` model plus the protocol.

    ``fit_epoch`` walks a fresh permutation of the training set in mini-batches of
    ``batch_size`` and calls ``model.update``. Batch size is load-bearing, not an
    implementation detail: batched feedback tracks the sequential algorithm at 10-50 and
    degrades noticeably at 200 (repo ``CLAUDE.md``), so arms are only ever compared at equal
    batch size and the value is in every record.
    """

    def __init__(self, model, seed: int, probe_x: Optional[Tensor] = None,
                 eval_batch: int = 500, probe_x_match: Optional[Tensor] = None,
                 probe_y: Optional[Tensor] = None, n_order: int = 256,
                 order_clauses: int = 2000):
        self.model = model
        self.seed = int(seed)
        self.eval_batch = int(eval_batch)
        # `probe_y` are the labels of `probe_x`, needed by D-FIRE to condition the firing rate
        # on Type-I eligibility (THEORY.md 5.5.9). Passing them changes nothing else: the
        # diagnostics never touch a CUDA RNG stream, so the training run is bit-identical with
        # and without them (AUDIT A-D1).
        self.probe = (DiagnosticProbe(probe_x, x_match=probe_x_match, y_probe=probe_y,
                                      n_order=n_order, order_clauses=order_clauses,
                                      seed=int(seed))
                      if probe_x is not None else None)
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)

    # -- training -------------------------------------------------------------------
    def fit_epoch(self, xtr: Tensor, ytr: Tensor, batch_size: int) -> None:
        self.model.train()
        n = xtr.shape[0]
        perm = torch.randperm(n, generator=self._gen).to(xtr.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            self.model.update(xtr[idx], ytr[idx])

    # -- inference ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        was = self.model.training
        self.model.eval()          # empty clauses flip meaning between train and eval
        try:
            out = [self.model(x[i : i + self.eval_batch]).argmax(dim=1)
                   for i in range(0, x.shape[0], self.eval_batch)]
        finally:
            self.model.train(was)
        return torch.cat(out)

    def accuracy(self, x: Tensor, y: Tensor) -> float:
        return float((self.predict(x) == y).float().mean())

    # -- reporting ------------------------------------------------------------------
    def capacity(self) -> dict:
        m = self.model
        C, L = int(m.n_clauses_total), int(m.n_literals)
        P = self._n_patches()
        cap = {
            "n_clauses_total": C,
            "n_features": int(m.n_features),
            "n_literals": L,
            "n_patches": P,
            "n_automata": int(m.ta_state.numel()),
            "state_bytes": int(m.ta_state.numel() * m.ta_state.element_size()),
            # Dense cost: every clause is evaluated against every literal of every patch,
            # which is what the library actually computes (a matmul, not a sparse walk).
            "literals_evaluated_per_image": int(P) * C * L,
            # ... and the logical cost, the one an ASIC would pay: only included literals.
            "included_literals_per_image": int(P) * int(m.include_count.sum()),
            "included_literals_total": int(m.include_count.sum()),
        }
        if hasattr(m, "weights"):
            cap["weight_bytes"] = int(m.weights.numel() * m.weights.element_size())
        return cap

    def _n_patches(self) -> int:
        m = self.model
        if hasattr(m, "_grid") and getattr(m, "input_shape", None) is not None:
            Py, Px = m._grid(m.input_shape[1], m.input_shape[2])
            return Py * Px
        return 1

    def diagnostics(self) -> dict:
        if self.probe is not None:
            return self.probe.epoch(self.model)
        return dict(clause_composition(self.model))

    def final_diagnostics(self) -> dict:
        if self.probe is not None:
            return self.probe.summary(self.model)
        return dict(clause_composition(self.model))

    # -- val-selected checkpointing --------------------------------------------------
    def snapshot(self) -> dict:
        """CPU copy of the learnable state. ``ta_state`` is int32 of shape
        ``(C, 2F)``: 80 MB for the largest arm planned, so a CPU copy per improving epoch is
        cheap next to an epoch, and it is what lets the test set stay untouched until the
        training loop has ended."""
        return {k: v.detach().to("cpu", copy=True)          # never non_blocking to the host
                for k, v in self.model.state_dict().items()}

    def restore(self, snap: dict) -> None:
        self.model.load_state_dict({k: v.to(self.model.device) for k, v in snap.items()})
        self.model._refresh_include()   # the load hook does this too; explicit is cheaper to audit


# --------------------------------------------------------------------------- builders
def _vote_capacity(cfg: dict, n_clauses_per_class: int, coalesced: bool) -> int:
    """The largest class sum the model can actually produce — the only sane denominator for ``T``.

    This is **not** the clause count, and getting it wrong silently disables the margin.
    A class-owned unweighted model gives each class ``n_clauses_per_class`` clauses of which
    only the **positive half** can push the sum up, and each contributes at most 1: capacity is
    ``n_clauses_per_class / 2``. `ctm-small` was registered with ``T = 160`` against a capacity
    of **100**, so ``(T - v) / 2T`` never fell below ~0.48 and ``T`` did nothing at all — the
    exact failure `ARMS.md` convention 2 warns about, sitting undetected in the arm that
    defined the programme's seed band.

    With integer weights the sum is unbounded above at init, so the capacity is reported at its
    initial value (weights start at 1) and flagged as a lower bound.
    """
    if coalesced:
        return max(1, int(cfg["n_clauses"]))          # every clause votes for every class
    return max(1, n_clauses_per_class // 2)           # positive polarity half only


def _resolve_T(cfg: dict, capacity: int) -> Tuple[float, float, str]:
    """``(T, T/capacity, source)``. ``source="ratio-default"`` marks a guessed ``T``.

    The ratio is against the **achievable vote sum** (:func:`_vote_capacity`), so a ratio above
    1.0 means the margin is unreachable and ``T`` is inert.
    """
    if cfg.get("T") is not None:
        T = float(cfg["T"])
        return T, T / max(1, capacity), "explicit"
    ratio = float(cfg.get("T_ratio", 0.8))
    return max(1.0, ratio * capacity), ratio, "ratio-default"


def _build_conv_classwise(cfg: dict) -> TMArm:
    """``ConvTsetlinMachine``: class-owned clauses of positive/negative polarity.

    **Why this class and not the coalesced one, for ``ctm-vanilla``.** Granmo et al. 2019,
    *The Convolutional Tsetlin Machine*, convolves the multi-class Tsetlin machine of Granmo
    2018: each class owns its own clause set, half voting for the class and half against,
    and there are no clause weights. The shared clause pool with per-class weights is a
    later and different contribution (*Coalesced Multi-Output Tsetlin Machines*, 2021) and
    is registered separately as ``ctm-coalesced`` in P3. Using the coalesced model for the
    arm labelled "vanilla CTM" would make the T0 reproduction a reproduction of the wrong
    paper -- and, because the coalesced model is cheaper per clause, would also quietly
    break every budget-matched comparison against it.

    [HYPOTHESIS] Whether the paper's "number of clauses" is per class or in total is the one
    ambiguity that matters here; this harness fixes it to *total* by convention (module
    docstring) and P1 checks it against the paper text.
    """
    n_total = int(cfg["n_clauses"])
    n_classes = int(cfg.get("n_classes") or N_CLASSES)
    per_class = max(2, n_total // n_classes)
    cap = _vote_capacity(cfg, per_class, coalesced=False)
    T, ratio, src = _resolve_T(cfg, cap)
    tt.seed_everything(int(cfg["seed"]))
    model = tt.ConvTsetlinMachine(
        n_classes, per_class, T, float(cfg["s"]),
        patch_size=cfg["patch"], stride=cfg["stride"],
        position_encoding=bool(cfg["position_encoding"]),
        input_shape=tuple(cfg["input_shape"]),
        weighted=bool(cfg.get("weighted", False)),
        max_included_literals=cfg.get("max_included_literals"),
        drop_clause_p=float(cfg.get("drop_clause_p", 0.0)),
        feedback_mode=str(cfg.get("feedback_mode", "batch")),
        **({"max_chunk_elements": int(cfg["max_chunk_elements"])}
           if cfg.get("max_chunk_elements") else {}),
    ).to(cfg["device"])
    cfg["_resolved"] = {"T": T, "T_ratio": ratio, "T_source": src, "vote_capacity": cap,
                        "n_clauses_per_class": per_class, "n_classes": n_classes,
                        "n_clauses_total": int(model.n_clauses_total)}
    return TMArm(model, cfg["seed"], cfg.get("probe_x"),
                 probe_x_match=cfg.get("probe_x_match"), probe_y=cfg.get("probe_y"),
                 n_order=int(cfg.get("n_order", 256)),
                 order_clauses=int(cfg.get("order_clauses", 2000)))


def _build_conv_coalesced(cfg: dict) -> TMArm:
    """``ConvCoalescedTsetlinMachine``: one shared clause pool, an integer weight per
    (clause, class). Registered for P3's ``ctm-coalesced``; used here only as the control
    that shows what the shared pool buys at an identical total clause budget."""
    n_total = int(cfg["n_clauses"])
    n_classes = int(cfg.get("n_classes") or N_CLASSES)
    cap = _vote_capacity(cfg, n_total, coalesced=True)
    T, ratio, src = _resolve_T(cfg, cap)
    tt.seed_everything(int(cfg["seed"]))
    model = tt.ConvCoalescedTsetlinMachine(
        n_classes, n_total, T, float(cfg["s"]),
        patch_size=cfg["patch"], stride=cfg["stride"],
        position_encoding=bool(cfg["position_encoding"]),
        input_shape=tuple(cfg["input_shape"]),
        max_included_literals=cfg.get("max_included_literals"),
        drop_clause_p=float(cfg.get("drop_clause_p", 0.0)),
        feedback_mode=str(cfg.get("feedback_mode", "batch")),
        **({"max_chunk_elements": int(cfg["max_chunk_elements"])}
           if cfg.get("max_chunk_elements") else {}),
    ).to(cfg["device"])
    cfg["_resolved"] = {"T": T, "T_ratio": ratio, "T_source": src, "vote_capacity": cap,
                        "n_clauses_per_class": n_total, "n_classes": n_classes,
                        "n_clauses_total": int(model.n_clauses_total)}
    return TMArm(model, cfg["seed"], cfg.get("probe_x"),
                 probe_x_match=cfg.get("probe_x_match"), probe_y=cfg.get("probe_y"),
                 n_order=int(cfg.get("n_order", 256)),
                 order_clauses=int(cfg.get("order_clauses", 2000)))


# --------------------------------------------------------------------------- registry
_COMMON = {"s": 10.0, "stride": 1, "position_encoding": True, "batch_size": 50,
           "feedback_mode": "batch", "max_included_literals": None, "drop_clause_p": 0.0,
           "weighted": False, "booleanization": "therm4", "T": None, "T_ratio": 0.8,
           "max_chunk_elements": None, "task": "cifar10"}

register("ctm-vanilla", ArmSpec(
    family="existing",
    paper="Granmo et al. 2019, The Convolutional Tsetlin Machine, arXiv:1905.09688",
    build=_build_conv_classwise,
    # 8000 clauses PER CLASS -> 80000 total. [FACT: LG-010 A1] the paper's Table 1 column is
    # literally "#Class Clauses", so the faithful reading is 10x the unqualified one. This is
    # a ~13 GPU-hour arm per seed at the default chunk budget; see AUDIT A7 before queueing it.
    defaults={**_COMMON, "n_clauses": 80000, "patch": 10, "epochs": 60},
    doc="The 2019 convolutional TM: class-owned clauses, a clause fires on an image iff it "
        "matches at least one patch, feedback is applied at one randomly chosen matching patch.",
    # At 78 img/s the default per-epoch probes dominate this arm (AUDIT A7); cap them for the
    # smoke test so the path is still exercised at its real clause count and window.
    smoke={"--eval-n": "500", "--m2-n": "200"},
    notes="n_clauses is the TOTAL clause budget; 80000 = 8000 per class, the paper's own "
          "reading [FACT: LG-010 A1]. T and s are NOT the paper's yet -- the record will say "
          "config_status=provisional until P1 supplies them. Patch coordinates are "
          "thermometer-encoded [FACT: LG-010 A4].",
))

register("ctm-vanilla-8k", ArmSpec(
    family="existing",
    paper="Granmo et al. 2019, The Convolutional Tsetlin Machine, arXiv:1905.09688",
    build=_build_conv_classwise,
    defaults={**_COMMON, "n_clauses": 8000, "patch": 10, "epochs": 60},
    doc="The 2019 convolutional TM at 8000 clauses in TOTAL -- the unqualified reading of "
        "'8000 clauses' that several later papers use, at a tenth of ctm-vanilla's budget.",
    notes="Exists so that LG-010's A1 ambiguity is settled by measurement rather than by "
          "reading: same code path as ctm-vanilla, 8000 total instead of 80000. This is the "
          "theorist's measurement M0.",
))

register("ctm-small", ArmSpec(
    family="control",
    paper=None,
    build=_build_conv_classwise,
    # T = 0.8 x the achievable class sum. With 2000 total clauses that is 200 per class, of
    # which only the 100 POSITIVE ones can raise the sum, so the capacity is 100 and T is 80.
    # This arm was originally registered with T = 160 -- above the capacity -- which made the
    # margin unreachable and T inert. The P0 seed band was measured under that fault; see
    # ARMS.md and AUDIT.md A21.
    defaults={**_COMMON, "n_clauses": 2000, "patch": 4, "epochs": 30, "T": 80.0},
    doc="A cheap scaled-down ctm-vanilla (same model class and code path, 2000 clauses, 4x4 "
        "patches) used for the P0 seed-noise band, screens and calibration.",
    notes="Deliberately the same builder as ctm-vanilla so the measured seed-noise band "
          "applies to the family the T0 reproduction uses. T=160 (0.8 x 200 clauses per "
          "class) is our own choice -- this arm has no paper to be faithful to.",
    config_status="own",
))

register("ctm-hog", ArmSpec(
    family="existing",
    paper="Granmo et al. 2023, TM Composites, arXiv:2309.04801 (Table 1) / "
          "Gronningsaeter et al. 2024, Optimized Toolbox, arXiv:2406.00704 (Table IV)",
    build=_build_conv_classwise,
    # T=50, s=10.0, budget 32, unweighted, 100 epochs are the paper's own [FACT: Composites
    # Table 1, via LITERATURE_TM.md A8]. The window is "32x32 = whole image", which on HOG's
    # 7x7 block grid means one window covering the whole map -> patch 7, P = 1 (ARMS.md A5).
    # 2 000 clauses PER CLASS = 20 000 total [FACT: arXiv:2406.00704 SII-B "a group of n
    # conjunctive clauses for each class"; cair/tmu vanilla_classifier.py populates one clause
    # bank per class]. patch=1 over the (N,5832,1,1) descriptor is the FLAT specialist.
    defaults={**_COMMON, "n_clauses": 20000, "patch": 1, "T": 50.0, "s": 10.0,
              "weighted": False, "max_included_literals": 32, "booleanization": "hog",
              "position_encoding": False, "epochs": 100},
    doc="The TM Composites / Toolbox HOG specialist: HOG Booleanization (8 orientations, 4x4 "
        "cells, 2x2 L2-Hys blocks, thermometer-4) with a whole-map window, so the clause is a "
        "conjunction over the entire descriptor rather than a sliding patch.",
    notes="HOG encoder is OURS (code/hog.py) -- the library has none (LG-008) -- so the report "
          "must say so. Published curve 64.2 (2k) -> 67.5 (64k), FLAT from 32k, which is half "
          "of the C-2 discriminator. position_encoding is vacuous at P=1 and set False to say "
          "so. The clause convention (total vs per class) is ambiguous in the source (A1), but "
          "the C-2 ladder sweeps it, so it does not bind there. **64 000 clauses OOMs a 24 GB "
          "card** at 12 544 literals/clause -- the ladder tops out at 32 000, which is where "
          "the published curve is already flat.",
    config_gaps=("colour channel order fed to cv2 (RGB here; the reference scripts' loader may "
                 "give BGR). Affects only which channel wins the max-gradient rule.",),
))

register("ctm-therm5", ArmSpec(
    family="existing",
    paper="Gronningsaeter et al. 2024, Optimized Toolbox, arXiv:2406.00704 (Table IV)",
    build=_build_conv_classwise,
    # The best published SINGLE model: 75.4%, 64 000 clauses, 250 epochs, 5x5 colour
    # thermometers, T=3000, s=5.0, weighted [FACT: LITERATURE_TM.md A9].
    # The paper's "64 000 clauses" is PER CLASS = 640 000 total, which does not fit this
    # machine (see ARMS.md). The registered default is therefore the paper's CHEAPEST measured
    # cell -- "2 000 clauses" = 20 000 total, 64.5% -- which is the pre-flight target and the
    # bottom of the capacity ladder. Epochs 100: 250 is an unexamined default worth 0.3-0.5 pp
    # [FACT: theorist's digitisation of Fig. 3].
    defaults={**_COMMON, "n_clauses": 20000, "patch": 5, "T": 3000.0, "s": 5.0,
              "weighted": True, "max_included_literals": 32, "booleanization": "therm8",
              "epochs": 100},
    doc="The Toolbox's best single specialist: 5x5 colour thermometers, weighted clauses -- "
        "the published curve that is STILL CLIMBING at 64 000 clauses, and the other half of "
        "the C-2 discriminator.",
    notes="FULLY SPECIFIED as of ROUND-1: T=3000, s=5.0, weighted, 5x5 window [FACT: Table IV]; "
          "clauses are PER CLASS [FACT: arXiv:2406.00704 SII-B and cair/tmu "
          "vanilla_classifier.py], so 20000 total = the paper's 2 000 and its 64.5% cell; "
          "class-owned [FACT: same]; ColorThermometerEncoder(n_bits=8) is bit-identical to the "
          "reference [FACT: theorist]; budget 32 appears in all 22 reference scripts. "
          "ONE DELIBERATE, QUANTIFIED DEVIATION: we run 50-100 epochs against the paper's 250, "
          "worth 1.1-1.2 pp at 50 and 0.3-0.5 pp at 100 [FACT: theorist's digitisation of "
          "Fig. 3, recovered finals within 0.3 pp of Table IV]. Targets are corrected for it "
          "rather than the deviation being hidden: the 50-epoch pre-flight target is X=63.3, "
          "not 64.5. T is NOT rescaled with clause count -- the Toolbox reuses its 2 000-clause "
          "Optuna values at every budget, and so do we.",
    config_status="paper",
))

register("ctm-therm5c", ArmSpec(
    family="existing",
    paper="Gronningsaeter et al. 2024, Optimized Toolbox, arXiv:2406.00704 (Table IV)",
    build=_build_conv_coalesced,
    defaults={**_COMMON, "n_clauses": 20000, "patch": 5, "T": 3000.0, "s": 5.0,
              "max_included_literals": 32, "booleanization": "therm8", "epochs": 100},
    doc="`ctm-therm5` on the COALESCED builder -- identical configuration, one shared weighted "
        "clause pool instead of class-owned clause sets. Exists to settle ARMS.md A7 by "
        "measurement rather than by reading.",
    notes="ARMS.md A7. The paper's T=3000 at 2 000 clauses is **incoherent for a class-owned "
          "model**: with 200 clauses per class the vote sum cannot exceed ~200, so (T-v)/2T is "
          "pinned near 0.5 for every example and the margin stops doing anything "
          "[MEASURED: the runner reports T_ratio=15.0 for the class-owned reading]. For a "
          "coalesced pool every clause votes for every class and integer weights let the sum "
          "exceed 2 000, so T=3000 is sensible. Whichever reading reproduces the published "
          "64.5% at 2 000 clauses is the right one, and that is a 2 GPU-h experiment.",
    config_gaps=("thermometer bit depth not stated; therm8 assumed (A6)",
                 "clause convention (A1)"),
))

# --- the CIFAR-2 discriminator (THEORY.md 5.5.6) --------------------------------------
# Two arms, identical to the `ctm-small-T80-{unc,b32}` pair in every respect except the label
# map, which merges CIFAR-10's ten classes into vehicle vs animal on the IDENTICAL split
# (data.TASKS['cifar2']). They exist to separate the two live explanations of why we measure
# +9.46 pp for a clause-size budget where CSC-TM measured +0.06:
#   BLOAT    -- their unconstrained clauses ran to 60.4 literals, ours to 170.4 of 248;
#   HEADROOM -- their CIFAR-2 baseline was already at 94.18% on two classes, ours at 38.31%
#               on ten, so the available gain differs by an order of magnitude before any
#               mechanism is invoked.
# Falsifier for the bloat law [FACT: THEORY.md 5.5.6]: the CIFAR-2 arm shows beta >= 4 and
# Delta <= +1 pp. Read them with `diagnostics.clause_len`, not only with accuracy.
#
# TWO DELIBERATE CHOICES, both recorded in ARMS.md A9 because both could have gone otherwise:
#  1. `T` is NOT 80. At two classes the same 2 000 total clauses give 1 000 per class, so the
#     achievable class sum is 500, not 100. Holding T = 80 would have held the *absolute*
#     margin while multiplying the reachable vote by 5, i.e. changed the one hyperparameter
#     DR-002/DR-003 proved is load-bearing (an unreachable T made the effect 2 pp smaller).
#     `T` is therefore set by the same rule as its CIFAR-10 sibling -- T_ratio 0.8 against the
#     achievable sum, giving T = 400 -- so that the *protocol* is identical rather than the
#     numeral.
#  2. The split is the ten-class stratified one, not re-stratified on two classes, so
#     `split_hash` and the images themselves are shared with every other arm.
_CIFAR2 = {**_COMMON, "n_clauses": 2000, "patch": 4, "epochs": 30, "task": "cifar2",
           "T": None, "T_ratio": 0.8}

register("ctm-small-cifar2-unc", ArmSpec(
    family="control",
    paper=None,
    build=_build_conv_classwise,
    defaults={**_CIFAR2, "max_included_literals": None},
    doc="ctm-small on the vehicle/animal CIFAR-2 merge with NO clause-size budget -- the "
        "unconstrained half of the discriminator between the bloat and headroom accounts of "
        "the +9.46 pp clause-budget effect.",
    notes="Identical to ctm-small-T80-unc except for the label map and the resulting "
          "n_classes=2 (T = 0.8 x the achievable class sum = 400, by the same rule). The "
          "CIFAR-2 merge is an ASSUMPTION: CSC-TM reports a CIFAR-2 row without defining it "
          "[FACT: arXiv:2301.08190 Table 3 + fn. 6]. See data.py CIFAR2_MAP and ARMS.md A9.",
    config_status="own",
))

register("ctm-small-cifar2-b32", ArmSpec(
    family="control",
    paper=None,
    build=_build_conv_classwise,
    defaults={**_CIFAR2, "max_included_literals": 32},
    doc="ctm-small on the vehicle/animal CIFAR-2 merge at clause-size budget 32 -- the "
        "budgeted half of the same discriminator.",
    notes="The pair (ctm-small-cifar2-unc, ctm-small-cifar2-b32) is the only experiment that "
          "separates bloat from headroom, and it costs 0.62 GPU-h at 3 seeds. Budget 32 is "
          "the Toolbox's value, present in all 22 reference scripts.",
    config_status="own",
))


# Available to P3 as a matched-budget control; registered now so the builder is exercised.
register("ctm-coalesced-small", ArmSpec(
    family="control",
    paper="Glimsdal & Granmo 2021, Coalesced Multi-Output Tsetlin Machines, arXiv:2108.07594",
    build=_build_conv_coalesced,
    defaults={**_COMMON, "n_clauses": 2000, "patch": 4, "epochs": 30, "T": 1600.0},
    doc="ctm-small's clause budget in a single shared, weighted clause pool -- the "
        "matched-budget control for what coalescing buys.",
    notes="Same total clause budget as ctm-small; T scales with the whole pool because every "
          "clause votes for every class here. A budget-matched control, not a reproduction of "
          "the coalesced paper's CIFAR-10 configuration -- hence config_status='own'.",
    config_status="own",
))


def get(name: str) -> ArmSpec:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; registered: {sorted(ARMS)}")
    return ARMS[name]


if __name__ == "__main__":
    for n, sp in ARMS.items():
        print(f"{n:22s} {sp.family:9s} {sp.config_status:12s} "
              f"clauses={sp.defaults['n_clauses']:6d} patch={sp.defaults['patch']:2d} "
              f"T={str(sp.defaults['T']):>8s} epochs={sp.defaults['epochs']:3d}  "
              f"{sp.doc[:56]}")
