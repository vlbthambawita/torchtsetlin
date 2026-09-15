#!/usr/bin/env python
"""Systematic CPU vs GPU performance benchmarks for torchtsetlin.

Each *suite* varies one axis and measures training (``model.update``) and inference
(``model.predict``) throughput on every requested device:

==============  ==========================================================================
``batch``       throughput vs mini-batch size (flat MNIST-shaped classifier)
``small``       same, for a Noisy-XOR scale machine (12 features, 20 clauses)
``clauses``     throughput and memory vs clause budget
``features``    throughput vs number of Boolean features
``models``      the model zoo (flat / coalesced / conv2d / conv-coalesced / regression)
``feedback``    ``feedback_mode="batch"`` vs ``"sequential"``
``threads``     CPU thread scaling (GPU shown as a reference line)
``transfer``    dataset resident on the device vs copied from the host per batch
``phases``      where the time of one update goes (encode / evaluate / counts / commit)
``mnist``       end-to-end MNIST epochs (wall clock + test accuracy)
==============  ==========================================================================

Results are written as one JSON document (environment + flat list of records) that
``benchmarks/plot_benchmarks.py`` turns into the plots used in ``docs/benchmarks.md``.

Examples::

    python benchmarks/bench_device.py --suites batch --devices cpu cuda:0
    python benchmarks/bench_device.py --all --out benchmarks/results/rtx3090.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

import torchtsetlin as tt

# Fraction of True bits in the synthetic Boolean data (MNIST at threshold 0.3 is ~0.19).
DENSITY = 0.2
SEED = 0

SUITES = ["batch", "small", "clauses", "features", "models", "feedback", "threads",
          "transfer", "phases", "mnist"]


# ---------------------------------------------------------------------------- utilities
def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_bool(n: int, shape: Sequence[int], device: torch.device, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, *shape, generator=g) < DENSITY
    return x.to(device)


def make_labels(n: int, n_classes: int, device: torch.device, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randint(0, n_classes, (n,), generator=g).to(device)


def make_floats(n: int, device: torch.device, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 2)
    return torch.rand(n, generator=g).to(device)


def state_mb(model: tt.TsetlinMachineBase) -> float:
    """Megabytes held by the automata state plus the float include cache."""
    n = model.ta_state.numel()
    return (n * model.ta_state.element_size() + n * 4) / 1e6


def batch_views(x: torch.Tensor, y: torch.Tensor, batch_size: int, n_batches: int) -> List[Tuple]:
    """``n_batches`` distinct (x, y) slices — views, so indexing costs nothing at run time."""
    out = []
    for i in range(n_batches):
        lo = (i * batch_size) % (x.shape[0] - batch_size + 1)
        out.append((x[lo : lo + batch_size], y[lo : lo + batch_size]))
    return out


def hms(seconds: float) -> str:
    m, s = divmod(int(seconds + 0.5), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


class Progress:
    """Counts the configurations of one suite on one device and reports them as they run.

    Each line is flushed immediately, so progress is visible even when stdout is a pipe
    (``run_benchmarks.sh`` tees it into a log file, which block-buffers otherwise).
    """

    def __init__(self, suite: str = "", device: str = "") -> None:
        self.suite, self.device = suite, device
        self.total = 0
        self.done = 0
        self.t0 = time.perf_counter()

    def plan(self, total: int) -> None:
        """Declare how many configurations this suite will measure."""
        self.total = total

    def item(self, label: str) -> None:
        """Announce the configuration that is about to be measured."""
        self.done += 1
        elapsed = time.perf_counter() - self.t0
        eta = ""
        if self.done > 1 and self.total:
            per_item = elapsed / (self.done - 1)
            eta = f", eta {hms(per_item * (self.total - self.done + 1))}"
        counter = f"{self.done}/{self.total}" if self.total else f"{self.done}"
        print(f"  [{counter:>5s}] {label:<44s} ({hms(elapsed)} elapsed{eta})", flush=True)


PROG = Progress()


@dataclass
class Record:
    suite: str
    phase: str  # "train" | "infer" | "epoch"
    device: str
    device_name: str
    model: str
    batch_size: int
    n_features: int
    n_clauses_per_class: int
    n_clauses_total: int
    n_classes: int
    threads: int
    sec_per_batch: float
    examples_per_s: float
    iters: int
    reps_sec_per_batch: List[float] = field(default_factory=list)
    peak_mem_mb: Optional[float] = None
    state_mb: Optional[float] = None
    extra: Dict = field(default_factory=dict)


# ------------------------------------------------------------------------------- timing
def timed(
    step: Callable[[int], None],
    n_batches: int,
    device: torch.device,
    *,
    budget: float = 2.0,
    max_iters: int = 100,
    min_iters: int = 3,
    warmup: int = 3,
    reps: int = 3,
) -> Tuple[float, List[float], int]:
    """Run ``step`` in timed blocks and return (median sec/batch, per-block sec/batch, iters).

    Every block runs at least ``min_iters`` iterations and then stops at ``max_iters`` or
    when ``budget`` seconds have elapsed; the device is synchronised once per block, so the
    result is steady-state throughput rather than single-call latency.
    """
    for i in range(warmup):
        step(i % n_batches)
    sync(device)
    per_block: List[float] = []
    iters = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        n = 0
        while n < max_iters:
            step(n % n_batches)
            n += 1
            if n >= min_iters and (time.perf_counter() - t0) > budget:
                break
        sync(device)
        per_block.append((time.perf_counter() - t0) / n)
        iters = n
    return median(per_block), per_block, iters


def measure(
    model: tt.TsetlinMachineBase,
    batches: List[Tuple],
    device: torch.device,
    phase: str,
    **kw,
) -> Tuple[float, List[float], int, Optional[float]]:
    """Time ``update`` (phase="train") or ``predict`` (phase="infer") over ``batches``."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if phase == "train":
        model.train()

        def step(i: int) -> None:
            xb, yb = batches[i]
            model.update(xb, yb)
    else:
        model.eval()

        def step(i: int) -> None:
            model(batches[i][0])

    sec, reps, iters = timed(step, len(batches), device, **kw)
    peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    return sec, reps, iters, peak


def record(
    suite: str,
    phase: str,
    device: torch.device,
    model: tt.TsetlinMachineBase,
    label: str,
    batch_size: int,
    sec: float,
    reps: List[float],
    iters: int,
    peak: Optional[float],
    *,
    n_clauses_per_class: int = 0,
    n_classes: int = 0,
    extra: Optional[Dict] = None,
) -> Record:
    return Record(
        suite=suite,
        phase=phase,
        device=str(device),
        device_name=device_name(device),
        model=label,
        batch_size=batch_size,
        n_features=int(model.n_features or 0),
        n_clauses_per_class=n_clauses_per_class,
        n_clauses_total=model.n_clauses_total,
        n_classes=n_classes,
        threads=torch.get_num_threads(),
        sec_per_batch=sec,
        examples_per_s=batch_size / sec,
        iters=iters,
        reps_sec_per_batch=reps,
        peak_mem_mb=peak,
        state_mb=state_mb(model),
        extra=extra or {},
    )


# --------------------------------------------------------------------------------- model
def build(
    kind: str,
    device: torch.device,
    *,
    n_features: Optional[int],
    n_classes: int,
    clauses: int,
    T: float = 25.0,
    s: float = 10.0,
    **kw,
) -> tt.TsetlinMachineBase:
    tt.seed_everything(SEED)
    if kind == "flat":
        model = tt.TsetlinMachine(n_features, n_classes, clauses, T=T, s=s, **kw)
    elif kind == "coalesced":
        model = tt.CoalescedTsetlinMachine(n_features, n_classes, clauses * n_classes, T=T, s=s, **kw)
    elif kind == "regression":
        kw.setdefault("y_range", (0.0, 1.0))
        model = tt.RegressionTsetlinMachine(n_features, clauses * n_classes, T=T, s=s, **kw)
    elif kind == "conv":
        model = tt.ConvTsetlinMachine(n_classes, clauses, T=T, s=s, **kw)
    elif kind == "conv_coalesced":
        model = tt.ConvCoalescedTsetlinMachine(n_classes, clauses * n_classes, T=T, s=s, **kw)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return model.to(device)


def bench_config(
    suite: str,
    device: torch.device,
    kind: str,
    *,
    n_features: int,
    n_classes: int,
    clauses: int,
    batch_size: int,
    n_examples: Optional[int] = None,
    n_batches: int = 8,
    phases: Sequence[str] = ("train", "infer"),
    input_shape: Optional[Sequence[int]] = None,
    label: Optional[str] = None,
    extra: Optional[Dict] = None,
    model_kw: Optional[Dict] = None,
    timing: Optional[Dict] = None,
) -> List[Record]:
    """Build one model + dataset and time the requested phases."""
    model_kw = dict(model_kw or {})
    timing = dict(timing or {})
    shape = list(input_shape or [n_features])
    n_examples = n_examples or max(batch_size * n_batches, 2048)
    x = make_bool(n_examples, shape, device)
    if kind == "regression":
        y = make_floats(n_examples, device)
    else:
        y = make_labels(n_examples, n_classes, device)
    if kind in ("conv", "conv_coalesced"):
        model_kw.setdefault("input_shape", tuple(shape))
        feats = None
    else:
        feats = n_features
    model = build(kind, device, n_features=feats, n_classes=n_classes, clauses=clauses, **model_kw)
    batches = batch_views(x, y, batch_size, min(n_batches, max(1, n_examples // batch_size)))
    out = []
    for phase in phases:
        sec, reps, iters, peak = measure(model, batches, device, phase, **timing)
        out.append(
            record(
                suite, phase, device, model, label or kind, batch_size, sec, reps, iters, peak,
                n_clauses_per_class=clauses, n_classes=n_classes, extra=extra,
            )
        )
        print(f"    {phase:5s} {out[-1].examples_per_s:12,.0f} ex/s  "
              f"({out[-1].sec_per_batch * 1e3:8.2f} ms/batch, {iters} iters)", flush=True)
    del model, x, y, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# -------------------------------------------------------------------------------- suites
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
CLAUSE_COUNTS = [50, 100, 250, 500, 1000, 2000, 4000]
FEATURE_COUNTS = [128, 256, 512, 1024, 2048, 4096, 8192]

# Reference point: an MNIST-shaped flat classifier.
REF = dict(kind="flat", n_features=784, n_classes=10, clauses=500)
REF_BATCH = 256


def suite_batch(device: torch.device) -> List[Record]:
    out = []
    PROG.plan(len(BATCH_SIZES))
    for bs in BATCH_SIZES:
        PROG.item(f"batch={bs}")
        out += bench_config("batch", device, batch_size=bs, n_batches=8,
                            n_examples=max(8 * bs, 4096), **REF)
    return out


def suite_clauses(device: torch.device) -> List[Record]:
    out = []
    PROG.plan(len(CLAUSE_COUNTS))
    for c in CLAUSE_COUNTS:
        PROG.item(f"clauses/class={c}")
        cfg = dict(REF, clauses=c)
        out += bench_config("clauses", device, batch_size=REF_BATCH, **cfg)
    return out


def suite_features(device: torch.device) -> List[Record]:
    out = []
    PROG.plan(len(FEATURE_COUNTS))
    for f in FEATURE_COUNTS:
        PROG.item(f"features={f}")
        cfg = dict(REF, n_features=f)
        out += bench_config("features", device, batch_size=REF_BATCH, **cfg)
    return out


def suite_models(device: torch.device) -> List[Record]:
    """One representative configuration per model class, at a common batch size."""
    bs = 128
    configs = [
        ("TsetlinMachine", dict(kind="flat", n_features=784, n_classes=10, clauses=500)),
        ("TsetlinMachine (weighted)", dict(kind="flat", n_features=784, n_classes=10, clauses=500,
                                           model_kw=dict(weighted=True))),
        ("CoalescedTsetlinMachine", dict(kind="coalesced", n_features=784, n_classes=10, clauses=500)),
        ("RegressionTsetlinMachine", dict(kind="regression", n_features=784, n_classes=1, clauses=5000)),
        ("ConvTsetlinMachine", dict(kind="conv", n_features=0, n_classes=10, clauses=200,
                                    input_shape=(1, 28, 28),
                                    model_kw=dict(patch_size=10, position_encoding=True, T=100.0))),
        ("ConvCoalescedTsetlinMachine", dict(kind="conv_coalesced", n_features=0, n_classes=10,
                                             clauses=200, input_shape=(1, 28, 28),
                                             model_kw=dict(patch_size=10, position_encoding=True,
                                                           T=100.0))),
    ]
    out = []
    PROG.plan(len(configs))
    for label, cfg in configs:
        PROG.item(label)
        kw = dict(cfg)
        mk = kw.pop("model_kw", {})
        T = mk.pop("T", 25.0)
        mk_all = dict(mk)
        mk_all["T"] = T
        out += bench_config("models", device, batch_size=bs, label=label,
                            model_kw=mk_all, n_examples=2048, **kw)
    return out


# Noisy-XOR scale: the smallest configuration anyone actually trains.
SMALL = dict(kind="flat", n_features=12, n_classes=2, clauses=10)


def suite_small(device: torch.device) -> List[Record]:
    """A tiny (Noisy-XOR scale) machine across batch sizes — where launch overhead rules."""
    batches = [1, 4, 16, 64, 256, 1024]
    out = []
    PROG.plan(len(batches))
    for bs in batches:
        PROG.item(f"batch={bs}")
        out += bench_config("small", device, batch_size=bs, n_examples=max(8 * bs, 4096),
                            label="flat (12 features, 20 clauses)",
                            model_kw=dict(T=15.0, s=3.9), **SMALL)
    return out


def suite_feedback(device: torch.device) -> List[Record]:
    """Exact sequential feedback vs batched feedback, at two model scales."""
    scales = [
        ("xor", dict(SMALL), dict(T=15.0, s=3.9), [1, 32, 256]),
        ("mnist-1k", dict(kind="flat", n_features=784, n_classes=10, clauses=100),
         dict(T=25.0), [1, 32, 256]),
    ]
    jobs = [
        (scale, cfg, mk, mode, bs)
        for scale, cfg, mk, batches in scales
        for mode in ("batch", "sequential")
        for bs in batches
        # sequential commits once per example: batch 32 is already enough to extrapolate
        if not (mode == "sequential" and bs > 32)
    ]
    out = []
    PROG.plan(len(jobs))
    for scale, cfg, mk, mode, bs in jobs:
        PROG.item(f"{scale} feedback_mode={mode} batch={bs}")
        out += bench_config(
            "feedback", device, batch_size=bs, phases=("train",),
            label=f"{scale}/{mode}",
            extra=dict(feedback_mode=mode, scale=scale),
            model_kw=dict(feedback_mode=mode, **mk),
            timing=dict(budget=2.0, max_iters=60 if mode == "batch" else 10),
            n_examples=max(8 * bs, 2048), **cfg,
        )
    return out


def suite_threads(device: torch.device) -> List[Record]:
    """CPU thread scaling. On CUDA a single point is measured as a reference line."""
    out = []
    if device.type != "cpu":
        PROG.plan(1)
        PROG.item("cuda reference point")
        return bench_config("threads", device, batch_size=REF_BATCH, extra=dict(threads_axis=True), **REF)
    default = torch.get_num_threads()
    counts = [n for n in (1, 2, 4, 8, 16, 32) if n <= os.cpu_count()]
    PROG.plan(len(counts))
    try:
        for n in counts:
            torch.set_num_threads(n)
            PROG.item(f"threads={n}")
            out += bench_config("threads", device, batch_size=REF_BATCH,
                                extra=dict(threads_axis=True), **REF)
    finally:
        torch.set_num_threads(default)
    return out


def suite_transfer(device: torch.device) -> List[Record]:
    """Dataset resident on the device vs a host-resident dataset copied in per batch."""
    out = []
    PROG.plan(3 * 2)
    for bs in (32, 256, 2048):
        for resident in (True, False):
            PROG.item(f"batch={bs} data_on={'device' if resident else 'host'}")
            data_device = device if resident else torch.device("cpu")
            n = max(8 * bs, 4096)
            x = make_bool(n, [REF["n_features"]], data_device)
            y = make_labels(n, REF["n_classes"], data_device)
            model = build("flat", device, n_features=REF["n_features"],
                          n_classes=REF["n_classes"], clauses=REF["clauses"])
            batches = batch_views(x, y, bs, 8)
            sec, reps, iters, peak = measure(model, batches, device, "train")
            out.append(record("transfer", "train", device, model, "flat", bs, sec, reps, iters,
                              peak, n_clauses_per_class=REF["clauses"],
                              n_classes=REF["n_classes"],
                              extra=dict(data_resident_on_device=resident)))
            print(f"    train {out[-1].examples_per_s:12,.0f} ex/s", flush=True)
            del model, x, y, batches
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out


def suite_mnist(device: torch.device, root: str = "./data", epochs: int = 3) -> List[Record]:
    """End-to-end MNIST: wall clock per epoch and test accuracy, per device."""
    out = []
    x_tr, y_tr = tt.data.load_mnist_boolean(root, train=True, threshold=0.3, device=device)
    x_te, y_te = tt.data.load_mnist_boolean(root, train=False, threshold=0.3, device=device)
    configs = [
        ("TsetlinMachine 500/class", dict(kind="flat", clauses=500, batch_size=100, T=25.0,
                                          flatten=True, subset=None, epochs=epochs)),
        ("ConvTsetlinMachine 200/class 10x10", dict(kind="conv", clauses=200, batch_size=100,
                                                    T=100.0, flatten=False, subset=10000,
                                                    epochs=1)),
    ]
    PROG.plan(len(configs))
    for label, cfg in configs:
        PROG.item(label)
        n = cfg["subset"]
        xt = x_tr[:n] if n else x_tr
        yt = y_tr[:n] if n else y_tr
        xv, yv = (x_te[:2000], y_te[:2000]) if n else (x_te, y_te)
        if cfg["flatten"]:
            xt = xt.reshape(xt.shape[0], -1)
            xv = xv.reshape(xv.shape[0], -1)
            model = build("flat", device, n_features=784, n_classes=10, clauses=cfg["clauses"],
                          T=cfg["T"], weighted=False)
        else:
            model = build("conv", device, n_features=None, n_classes=10, clauses=cfg["clauses"],
                          T=cfg["T"], patch_size=10, position_encoding=True, input_shape=(1, 28, 28))
        trainer = tt.Trainer(model, device=device, batch_size=cfg["batch_size"], verbose=True)
        print(f"    {xt.shape[0]:,} train examples, {cfg['epochs']} epoch(s)", flush=True)
        sync(device)
        t0 = time.perf_counter()
        hist = trainer.fit((xt, yt), epochs=cfg["epochs"], val_data=(xv, yv))
        sync(device)
        total = time.perf_counter() - t0
        epoch_times = [float(e["epoch_time"]) for e in hist.epochs]
        acc = float(hist.epochs[-1].get("val_accuracy", float("nan")))
        peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
        med = median(epoch_times)
        out.append(Record(
            suite="mnist", phase="epoch", device=str(device), device_name=device_name(device),
            model=label, batch_size=cfg["batch_size"], n_features=int(model.n_features or 0),
            n_clauses_per_class=cfg["clauses"], n_clauses_total=model.n_clauses_total,
            n_classes=10, threads=torch.get_num_threads(), sec_per_batch=med,
            examples_per_s=xt.shape[0] / med, iters=cfg["epochs"], reps_sec_per_batch=epoch_times,
            peak_mem_mb=peak, state_mb=state_mb(model),
            extra=dict(train_examples=int(xt.shape[0]), val_examples=int(xv.shape[0]),
                       epochs=cfg["epochs"], val_accuracy=acc, total_sec=total,
                       epoch_times=epoch_times,
                       batch_accuracy=float(hist.epochs[-1].get("batch_accuracy", float("nan")))),
        ))
        print(f"    {med:.2f} s/epoch, val accuracy {acc:.4f}", flush=True)
        del model, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out



def suite_phases(device: torch.device) -> List[Record]:
    """Cost breakdown of a single ``update`` call, plus the raw ``torch.binomial`` cost.

    The stages are the ones named in the update pipeline: ``_encode`` -> ``_evaluate`` ->
    ``_votes`` / ``_select_feedback`` -> ``_feedback_counts`` -> ``_commit``. ``commit`` is
    measured as ``update`` minus the accumulation stages, so it includes ``apply_feedback``
    and ``_refresh_include``.
    """
    PROG.plan(1)
    PROG.item("update() stage breakdown")
    bs = REF_BATCH
    n = max(8 * bs, 4096)
    x = make_bool(n, [REF["n_features"]], device)
    y = make_labels(n, REF["n_classes"], device)
    model = build("flat", device, n_features=REF["n_features"], n_classes=REF["n_classes"],
                  clauses=REF["clauses"])
    model.train()
    xb, yb = x[:bs], y[:bs]
    literals = model._encode(xb)
    clause_out, ctx = model._evaluate(literals, True)
    votes = model._votes(clause_out)
    type_i, type_ii, _ = model._select_feedback(votes, yb, clause_out)
    cd = model.compute_dtype
    sel_fire_i = (type_i & clause_out).to(cd)
    sel_nofire_i = (type_i & ~clause_out).to(cd)
    sel_fire_ii = (type_ii & clause_out).to(cd)
    acc = model._new_accumulator()

    def step_update(_: int) -> None:
        model.update(xb, yb)

    def step_accumulate(_: int) -> None:
        a = model._new_accumulator()
        model._accumulate_chunk(xb, yb, a)

    stages = [
        ("encode", lambda _: model._encode(xb)),
        ("evaluate", lambda _: model._evaluate(literals, True)),
        ("votes+select", lambda _: model._select_feedback(model._votes(clause_out), yb, clause_out)),
        ("feedback_counts", lambda _: model._feedback_counts(
            literals, ctx, sel_fire_i, sel_nofire_i, sel_fire_ii, acc)),
        ("accumulator_alloc", lambda _: model._new_accumulator()),
        ("coerce_targets", lambda _: model._coerce_targets(yb, bs, device)),
        ("refresh_include", lambda _: model._refresh_include()),
    ]
    timings: Dict[str, float] = {}
    for name, fn in stages:
        timings[name], _, _ = timed(fn, 1, device, budget=1.0, max_iters=50)
    t_update, reps_update, iters = timed(step_update, 1, device, budget=2.0, max_iters=50)
    t_accumulate, _, _ = timed(step_accumulate, 1, device, budget=2.0, max_iters=50)
    timings["commit"] = max(t_update - t_accumulate, 0.0)
    # apply_feedback = everything in commit that is not the include refresh, minus the
    # target coercion that update() does before the loop (two .any() host syncs).
    timings["apply_feedback"] = max(
        timings["commit"] - timings["refresh_include"] - timings["coerce_targets"], 0.0)

    # Raw cost of the (C, 2F) binomial draw inside apply_feedback, for reference.
    shape = (model.n_clauses_total, model.n_literals)
    count = torch.randint(0, 6, shape, dtype=cd, device=device)
    prob = torch.full(shape, 1.0 / float(model.s), dtype=cd, device=device)
    timings["torch.binomial (C x 2F)"], _, _ = timed(
        lambda _: torch.binomial(count, prob), 1, device, budget=1.0, max_iters=30)

    out = [
        record("phases", "update_total", device, model, "flat", bs, t_update, reps_update, iters,
               torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None,
               n_clauses_per_class=REF["clauses"], n_classes=REF["n_classes"],
               extra=dict(stage="update_total")),
    ]
    for name, sec in timings.items():
        out.append(record("phases", "stage", device, model, "flat", bs, sec, [sec], 0, None,
                          n_clauses_per_class=REF["clauses"], n_classes=REF["n_classes"],
                          extra=dict(stage=name)))
        print(f"    {name:26s} {sec * 1e3:9.3f} ms", flush=True)
    print(f"    {'update (total)':26s} {t_update * 1e3:9.3f} ms", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


SUITE_FN: Dict[str, Callable[[torch.device], List[Record]]] = {
    "batch": suite_batch,
    "small": suite_small,
    "clauses": suite_clauses,
    "features": suite_features,
    "models": suite_models,
    "feedback": suite_feedback,
    "threads": suite_threads,
    "transfer": suite_transfer,
    "phases": suite_phases,
    "mnist": suite_mnist,
}


# ----------------------------------------------------------------------------- reporting
_NAMES: Dict[str, str] = {}


def device_name(device: torch.device) -> str:
    key = str(device)
    if key not in _NAMES:
        if device.type == "cuda":
            _NAMES[key] = torch.cuda.get_device_name(device)
        else:
            _NAMES[key] = cpu_name()
    return _NAMES[key]


def cpu_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover - platform dependent
        pass
    return platform.processor() or platform.machine()


def environment() -> Dict:
    env = {
        "torchtsetlin": tt.__version__,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", default=["batch"], choices=SUITES + ["all"])
    ap.add_argument("--all", action="store_true", help="run every suite")
    ap.add_argument("--devices", nargs="+", default=None,
                    help="devices to benchmark (default: cpu plus cuda:0 if available)")
    ap.add_argument("--out", default="benchmarks/results/results.json")
    ap.add_argument("--root", default="./data", help="dataset root for the mnist suite")
    ap.add_argument("--epochs", type=int, default=3, help="epochs for the mnist suite")
    args = ap.parse_args()

    suites = SUITES if (args.all or "all" in args.suites) else args.suites
    devices = args.devices or (["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
    devices = [torch.device(d) for d in devices]

    global PROG
    records: List[Record] = []
    t_start = time.perf_counter()
    for suite in suites:
        for device in devices:
            print(f"\n=== {suite} on {device} ({device_name(device)}) ===", flush=True)
            PROG = Progress(suite, str(device))
            fn = SUITE_FN[suite]
            if suite == "mnist":
                records += fn(device, root=args.root, epochs=args.epochs)  # type: ignore[call-arg]
            else:
                records += fn(device)
    elapsed = time.perf_counter() - t_start

    payload = {
        "environment": environment(),
        "suites": suites,
        "devices": [str(d) for d in devices],
        "wall_clock_sec": elapsed,
        "records": [asdict(r) for r in records],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        with open(args.out) as fh:
            old = json.load(fh)
        keep = [r for r in old.get("records", [])
                if not (r["suite"] in suites and r["device"] in payload["devices"])]
        payload["records"] = keep + payload["records"]
        payload["suites"] = sorted({*old.get("suites", []), *suites})
        payload["devices"] = sorted({*old.get("devices", []), *payload["devices"]})
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nWrote {len(payload['records'])} records to {args.out} in {hms(elapsed)}",
          flush=True)


if __name__ == "__main__":
    main()
