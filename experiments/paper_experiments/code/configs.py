"""Frozen per-cell configurations (config_map.md §(b); decisions in PLAN.md §8).

`cell_config(bench, lib, mode)` returns everything an adapter needs: the task, the epoch count,
the library-neutral model description and the library-specific constructor arguments. Nothing
here is tuned; every value is the published one (rule P3).

Libraries: pytm, pytmpar, pytmcuda, tmu, gt, tt.
Modes: "default" for the external libraries. torchtsetlin has three (decisions D3, Q2):
    seq    feedback_mode="sequential" (the exact per-example algorithm)
    batch  feedback_mode="batch", batch 32 (flat / regression) or 16 (conv)
    b100   feedback_mode="batch", batch 100 (PyTsetlinMachineCUDA's fit batch_size; same number,
           not the same algorithm: CUDA's batch is a synchronisation window)
B6 modes are "C<clauses>" (external) and "C<clauses>-batch" / "C<clauses>-b100" (tt).
"""
import copy

B6_CLAUSES = (10, 100, 1000, 2000, 5000)

# Library-neutral description: C = clauses per class (pyTM semantics), T, s, boost = boosted
# true-positive feedback, weighted, patch = conv window. GT is coalesced: it gets C x K clauses
# (same automata count, config_map a.1) and feedback_type="uniform" outside B6.
BENCH = {
    "B1": dict(task="classification", model="flat", epochs=200, seeds=5, C=10, T=15, s=3.9,
               boost=False, weighted=False, n_classes=2),
    "B2": dict(task="classification", model="conv", epochs=5000, seeds=5, C=40, T=60, s=3.9,
               boost=False, weighted=False, patch=2, shape=(4, 4), n_classes=2),
    "B3": dict(task="classification", model="flat", epochs=25, seeds=100, splits=True, C=800,
               T=40, s=5.0, boost=True, weighted=False, n_classes=2),
    "B4": dict(task="classification", model="flat", epochs=250, seeds=5, C=2000, T=50, s=10.0,
               boost=True, weighted=False, n_classes=10),
    "B5": dict(task="classification", model="flat", epochs=100, seeds=5, C=2000, T=5000, s=10.0,
               boost=True, weighted=True, n_classes=10),
    "B6": dict(task="classification", model="b6", epochs=5, seeds=3, s=3.0, n_classes=10),
    "B7": dict(task="classification", model="conv", epochs=30, seeds=5, C=2000, T=5000, s=5.0,
               boost=True, weighted=True, patch=10, shape=(28, 28), n_classes=10),
    "B8": dict(task="classification", model="conv", epochs=30, seeds=5, C=2000, T=5000, s=5.0,
               boost=True, weighted=True, patch=10, shape=(28, 28), n_classes=10),
    "B9": dict(task="classification", model="flat", epochs=30, seeds=5, C=10000, T=8000, s=10.0,
               boost=True, weighted=True, n_classes=2),
    "B10": dict(task="regression", model="regression", epochs=30, seeds=25, splits=True, C=1000,
                T=5000, s=2.75, boost=True, weighted=True),
    # Reproduction checks: each library's own published configuration (PLAN §3.1).
    "R1": dict(task="classification", model="flat", epochs=60, seeds=1, data="B4", C=2000, T=5000,
               s=10.0, boost=True, weighted=True, n_classes=10),
    "R2": dict(task="classification", model="flat", epochs=100, seeds=1, data="B4", C=2000, T=5000,
               s=10.0, boost=True, weighted=True, n_classes=10),
    "R3": dict(task="classification", model="conv", epochs=30, seeds=1, data="B8", C=8000, T=10000,
               s=10.0, boost=True, weighted=True, patch=10, shape=(28, 28), n_classes=10),
    "R4": dict(task="classification", model="flat", epochs=60, seeds=1, data="B4", C=2000, T=5000,
               s=10.0, boost=True, weighted=True, budget=32, n_classes=10),
}

# Which libraries implement each benchmark's model (config_map §(b)); tt-seq only where the
# sequential path is affordable (PLAN §4.3).
LIBS = {
    "B1": ["pytm", "pytmpar", "pytmcuda", "tmu", "gt", "tt"],
    "B2": ["pytm", "pytmpar", "pytmcuda", "tmu", "tt"],
    "B3": ["pytm", "pytmpar", "pytmcuda", "tmu", "gt", "tt"],
    "B4": ["pytm", "pytmpar", "pytmcuda", "tmu", "gt", "tt"],
    "B5": ["pytm", "pytmpar", "pytmcuda", "tmu", "gt", "tt"],
    "B6": ["pytm", "pytmpar", "pytmcuda", "tmu", "gt", "tt"],
    "B7": ["pytm", "pytmpar", "pytmcuda", "tmu", "tt"],
    "B8": ["pytm", "pytmpar", "pytmcuda", "tmu", "tt"],
    "B9": ["pytm", "pytmpar", "pytmcuda", "tmu", "gt", "tt"],
    "B10": ["pytm", "pytmpar", "pytmcuda", "tmu", "tt"],
    "R1": ["pytm"], "R2": ["pytmpar"], "R3": ["pytmcuda"], "R4": ["tmu"],
}
# D10 (user, 2026-10-23): tt-seq only on B1, B2, B3, B10 (406 s/epoch on MNIST even on the GPU).
TT_MODES = {b: ["seq", "batch", "b100"] for b in ("B1", "B2", "B3", "B10")}
TT_MODES.update({b: ["batch", "b100"] for b in ("B4", "B5", "B7", "B8", "B9")})
# D10: tt on the CPU is timed over this many epochs (per-epoch rate; its accuracy comes from the
# identical algorithm on the GPU), except B6, which keeps its 5-epoch protocol.
TT_CPU_TIMING_EPOCHS = 3


def modes(bench, lib):
    if bench == "B6":
        cs = [f"C{c}" for c in B6_CLAUSES]
        return [f"{c}-{m}" for c in cs for m in ("batch", "b100")] if lib == "tt" else cs
    return TT_MODES[bench] if lib == "tt" else ["default"]


def _b6(lib, mode):
    """green_tsetlin performance_tests/tmu_mnist_dense.py, argument for argument."""
    C = int(mode.split("-")[0][1:])
    T = C // 4
    s = 3.0
    if lib == "pytm":
        return dict(cls="MultiClassTsetlinMachine",
                    kwargs=dict(number_of_clauses=C, T=T, s=s, boost_true_positive_feedback=1,
                                weighted_clauses=True, max_included_literals=20))
    if lib == "pytmpar":
        return dict(cls="MultiClassTsetlinMachine",
                    kwargs=dict(number_of_clauses=C, T=T, s=s, boost_true_positive_feedback=1,
                                weighted_clauses=True))
    if lib == "pytmcuda":
        return dict(cls="MultiClassTsetlinMachine",
                    kwargs=dict(number_of_clauses=C, T=T, s=s, boost_true_positive_feedback=1),
                    batch_size=100)
    if lib == "tmu":
        return dict(cls="TMCoalescedClassifier",
                    kwargs=dict(number_of_clauses=C, T=T, s=s, weighted_clauses=True,
                                focused_negative_sampling=True, max_included_literals=20))
    if lib == "gt":
        return dict(kwargs=dict(n_clauses=C, s=s, threshold=T, boost_true_positives=True,
                                literal_budget=20),
                    trainer=dict(feedback_type="focused_negative_sampling", fixed_seed=42))
    if lib == "tt":
        return dict(cls="CoalescedTsetlinMachine",
                    kwargs=dict(n_clauses=C, T=T, s=s, boost_true_positive=True,
                                max_included_literals=20, focused_negative_sampling=True))
    raise KeyError(lib)


def _common(b, lib, mode):
    """Shared configuration of B1-B5, B7-B10 and R1-R4 for one library."""
    C, T, s, w, boost = b["C"], b["T"], b["s"], b["weighted"], b["boost"]
    model = b["model"]
    if lib in ("pytm", "pytmpar"):
        cls = {"flat": "MultiClassTsetlinMachine", "conv": "MultiClassConvolutionalTsetlinMachine2D",
               "regression": "RegressionTsetlinMachine"}[model]
        kw = dict(number_of_clauses=C, T=T, s=s, boost_true_positive_feedback=int(boost))
        if model == "conv":
            kw["patch_dim"] = (b["patch"], b["patch"])
        if w:
            kw["weighted_clauses"] = True
        return dict(cls=cls, kwargs=kw)
    if lib == "pytmcuda":
        cls = {"flat": "MultiClassTsetlinMachine", "conv": "MultiClassConvolutionalTsetlinMachine2D",
               "regression": "RegressionTsetlinMachine"}[model]
        kw = dict(number_of_clauses=C, T=T, s=s, boost_true_positive_feedback=int(boost))
        if model == "conv":
            kw["patch_dim"] = (b["patch"], b["patch"])
        if w:
            kw["max_weight"] = 255          # the only weighting the CUDA package has (a.2)
        return dict(cls=cls, kwargs=kw, batch_size=100)
    if lib == "tmu":
        cls = "TMRegressor" if model == "regression" else "TMClassifier"
        kw = dict(number_of_clauses=C, T=T, s=s, boost_true_positive_feedback=int(boost))
        if model == "conv":
            kw["patch_dim"] = (b["patch"], b["patch"])
        if w:
            kw["weighted_clauses"] = True
        if b.get("budget"):
            kw["max_included_literals"] = b["budget"]
            kw["type_iii_feedback"] = False
        return dict(cls=cls, kwargs=kw)
    if lib == "gt":
        assert model == "flat", "green-tsetlin: no regression; conv not usable via its API (a.8)"
        return dict(kwargs=dict(n_clauses=C * b["n_classes"], s=s, threshold=T,
                                boost_true_positives=boost),
                    trainer=dict(feedback_type="uniform"))
    if lib == "tt":
        kw = dict(n_clauses=C, T=T, s=s, boost_true_positive=boost)
        if model == "regression":
            cls = "RegressionTsetlinMachine"
            kw["weighted"] = w
        elif model == "conv":
            cls = "ConvTsetlinMachine"
            kw.update(patch_size=b["patch"], stride=1, position_encoding=True,
                      input_shape=(1,) + tuple(b["shape"]), weighted=w)
        else:
            cls = "TsetlinMachine"
            kw["weighted"] = w
        if b.get("budget"):
            kw["max_included_literals"] = b["budget"]
        return dict(cls=cls, kwargs=kw)
    raise KeyError(lib)


def cell_config(bench, lib, mode="default"):
    b = copy.deepcopy(BENCH[bench])
    if lib not in LIBS[bench]:
        raise ValueError(f"{lib} does not implement {bench}")
    if mode not in modes(bench, lib):
        raise ValueError(f"mode {mode!r} not defined for {bench}/{lib}: {modes(bench, lib)}")
    lib_cfg = _b6(lib, mode) if bench == "B6" else _common(b, lib, mode)
    cfg = dict(bench=bench, data=b.pop("data", bench), lib=lib, mode=mode, **b, lib_cfg=lib_cfg)
    if lib == "tt":
        tt_mode = mode.split("-")[-1]
        cfg["tt"] = dict(
            feedback_mode="sequential" if tt_mode == "seq" else "batch",
            batch_size={"seq": 1000, "batch": 16 if b["model"] == "conv" else 32, "b100": 100}[tt_mode],
        )
    return cfg


def all_cells():
    """Every (bench, lib, mode) cell of the comparison (reproduction checks excluded)."""
    for bench in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"):
        for lib in LIBS[bench]:
            for mode in modes(bench, lib):
                yield bench, lib, mode
