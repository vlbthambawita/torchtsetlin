# Harness interface contract

Frozen by the orchestrator so that the engineer, the DL expert and the arm implementers can write
code concurrently. Changes to this file go through the orchestrator; breaking it silently is the
fastest way to make three people's code disagree.

## `code/data.py`

```python
SPLIT_SEED = 1234                 # never changes
N_VAL      = 5000

def raw_cifar10(root=None) -> dict
    # {'xtr': uint8 (50000,3,32,32), 'ytr': int64 (50000,), 'xte': ..., 'yte': ...}
    # Loads from ../.data (symlink to experiments/mctm/.data). Never re-downloads.

def split_indices(seed=SPLIT_SEED, n_val=N_VAL) -> (LongTensor train_idx, LongTensor val_idx)
    # Stratified, deterministic, 45000/5000. IDENTICAL for every arm, TM and CNN.

def split_hash() -> str           # sha1 of the two index tensors; goes into every result record

def subset_indices(train_idx, n, seed=SPLIT_SEED) -> LongTensor
    # Stratified nested subsets for the sample-efficiency curve: n in {1000, 5000, 10000}.
    # Nested means subset(1000) is a subset of subset(5000).

BOOLEANIZATIONS: dict[str, callable]
    # at least: 'therm4', 'therm8', 'adaptive', later 'hog' and whatever P1 turns up.

def load_boolean(name='therm4', device='cuda', splits=('train','val','test')) -> dict
    # {'xtr': bool (45000,Z,32,32), 'ytr', 'xva', 'yva', 'xte', 'yte', 'name': name, 'Z': Z}
    # Cached per name under ../.cache/. therm4/therm8 reuse the existing caches
    # (experiments/mctm/.cache/cifar10_therm{4,8}.pt) by re-splitting them with split_indices();
    # do NOT re-encode those two.

def load_float(device='cuda', augment=False) -> dict
    # float32 CIFAR-10 normalised with the standard CIFAR mean/std, same splits, for the CNNs.
    # augment=True -> random crop 32 pad 4 + horizontal flip, applied on-device per batch.
```

**Hard rule**: every loader keeps encoder thresholds and any CPU-side work on CPU, and never issues
`.to(cpu, non_blocking=True)` (it silently Booleanizes garbage — see CHARTER.md).

## `code/record.py`

```python
def new_record(arm, family, seed, argv, hp, data, paper=None) -> dict
def add_epoch(rec, epoch, train_acc, val_acc, epoch_s, **diagnostics) -> None
def finalize(rec, test_acc, test_preds, selected_epoch, capacity, diagnostics) -> None
def write(rec, path) -> None      # validates against the schema; refuses to write an invalid record
def validate(rec) -> list[str]    # [] == valid
```

Schema: Section 6.2 of PLAN.md. Mandatory keys — `arm, family, paper, seed, argv, git_sha, env,
started, wall_s, data, hp, capacity, curve, selected_epoch, val_acc, test_acc, test_acc_per_class,
test_predictions_sha, diagnostics, notes`.

**Amendments agreed during P0** (orchestrator-approved; `record.validate()` enforces all three):

* `data.split_hash` is mandatory — Section 7.4.3 requires the split hash in every record.
* `hp.batch_size` is mandatory — arms are comparable only at equal batch size.
* `hp.max_chunk_elements` / `hp.chunk_size_at_batch`: the library's chunk budget is a
  throughput dial that silently serialises large conv models (LIBRARY_GAPS LG-004), so it is
  recorded like a hyperparameter and **arms are compared only at equal values**.
* `hp.config_status` ∈ `paper | provisional | own`, and `hp.T_source` ∈ `explicit |
  ratio-default`. `T` and `s` span two orders of magnitude across the bibliography
  (LG-010 A3), so an arm running on the harness fallback is measuring our guess.
  `record.admissible_as_reproduction(rec)` — not `validate` — is the gate for a
  "reproduced" / `GAP` verdict.
* **`diagnostics.match_count` (M2)** — the distribution of |M_j(x)|, the number of patches a
  *firing* clause matches: `{n_patches, probe_pairs, firing_pairs, firing_pair_frac, mean,
  mean_over_all, q0, q25, q50, q75, q95, q100, frac_ge_5}`. **Mandatory for any arm with
  `capacity.n_patches > 1`**; `null` is correct for flat and CNN arms, where the count is 1
  by construction. It is a reduction over the `(B, P, C)` tensor the conv forward pass
  already computes, draws no random patches, and is collected per epoch because collecting it
  afterwards means re-running P3.
* `hp.vote_capacity` — the largest class sum the model can actually produce, and the only sane
  denominator for `T` (`n_clauses_per_class / 2` for an unweighted class-owned model, the whole
  pool for a coalesced one). `hp.T_ratio` is `T / vote_capacity`. The runner **hard-warns when
  an unweighted arm's `T` exceeds it**, because `T` is then inert — that fault sat undetected in
  the arm that defined the seed band (AUDIT A21).
* `hp.config_gaps` — a list, in words, of what is still *not* the paper's. An arm can carry the
  paper's `T` and `s` and still be provisional for another reason; this says which.
* `hp.test_curve_last` — see `--test-curve-last` in `run_arm.py`.
* **Amendment, 2026-09-21 (research-engineer): the three mechanism diagnostics of
  THEORY.md 5.5.9.** All three are **additive** — no existing field changes meaning, no
  existing number moves — and all three sit under the existing `diagnostics` key, per epoch in
  the curve and once more in the final summary. They exist so that the clause-size-budget
  sweep tests the proposed *mechanism* directly rather than only its accuracy consequence;
  they landed **before** `jobs/p3_block2_s_sweep.txt` ran, which is why that sweep does not
  have to be re-run to get them.
  - `diagnostics.ta_state` (**D-TA**) — `{n_states, n_included, include_frac,
    frac_at_boundary, frac_deep, frac_saturated, mean_depth, median_depth,
    hist_included_16, frac_excluded_at_boundary}`. A full `0..2N-1` bincount over `ta_state`,
    chunked so the int64 cast is bounded regardless of clause count; no data, no forward pass.
    `frac_at_boundary` is `P(state == N | included)` and is an **upper bound** on the Type-II
    fringe of C5.3, not an estimate of it: a literal climbing under Type Ia also passes
    through `N`.
  - `diagnostics.firing_rate_eligible` (**D-FIRE**) — `P(fire | the example is Type-I-eligible
    for that clause)`: `y == c` for a positive-polarity clause of class `c`, `y != c` for a
    negative one. Carries `s`, `predicted_1_over_1_plus_s` and `ratio_to_prediction`, so
    C5.2's prediction is *in the record* rather than in a downstream script. Exact over the
    probe, from one `index_add_` on a forward pass that already happens. `null` for a
    coalesced model (no per-clause class or polarity) and when the probe labels are absent.
    **The pre-existing global `firing_rate` cannot substitute for it**: it mixes positive
    clauses (eligible on 1/10 of images) with negative ones (eligible on 9/10).
  - `diagnostics.order` (**D-ORDER**) — `{n_images, n_clauses_sampled, sampled, levels[],
    p_image_ratio_32_over_all}`, one `level` per `k ∈ {8, 16, 32, all}` with
    `{mean_len, p_patch, p_image, neg_log_q, frac_tied_at_k}`. **Sampled, and it says so**:
    `--order-n` images (default 256) and `--order-clauses` clauses (default 2000), both in
    `hp`. `p_image_ratio_32_over_all >= 1` always and equals 1 exactly when the literals past
    rank 32 are logically redundant on the probe. Read it with `frac_tied_at_k`, which is near
    1 when the deep core is saturated and the prefix is therefore an arbitrary subset of it.
  - **Failure policy.** Each is wrapped by `diagnostics._safe`: a diagnostic that raises
    records `{"error": "..."}` under its own key and prints `!! diagnostic <name> FAILED` to
    the job log. These are observers bolted onto runs that cost hours; one must not destroy an
    arm it is only watching, and it must not vanish either. **A record carrying an `error`
    inside `diagnostics` is a record whose mechanism fields are missing, and the audit treats
    it as such.**
  - **None of the three consumes a CUDA RNG draw** (the only random draw is a clause subsample
    from an explicit CPU generator), so switching them on cannot perturb training. Verified on
    both devices by training the same arm with and without them and comparing the `ta_state`
    hash: identical (AUDIT A-D1).
* **Amendment, 2026-09-21: the task/label-map keys.** `data.task` ∈ `cifar10 | cifar2` and
  `data.n_classes` are now in every record, and `data.dataset` carries the task name.
  `record.validate()` checks `len(test_acc_per_class) == data.n_classes`, defaulting to 10 when
  the key is absent, so every record written before this date is unaffected. `hp.task` is an
  overridable hyperparameter like any other. The split is **not** re-derived per task: it stays
  stratified on the original ten labels, so `split_hash` is shared and a CIFAR-2 arm is
  comparable image for image with a CIFAR-10 one (ARMS.md A9).
* A curve entry **may not contain `test_acc`**; `selected_epoch` must equal
  `argmax(val_acc)`. Both are refused by `validate()`.

`selected_epoch` is chosen by `argmax(val_acc)`. `test_acc` is evaluated **once**, at that epoch.
Test predictions are saved to `results/preds/<arm>_seed<k>.npy` (int8, 10000) for paired tests.

## `code/run_arm.py`

```
python code/run_arm.py --arm <name> --seed <k> [--epochs N] [--subset N] [--bool therm4]
                       [--device cuda] [--out results/<arm>_seed<k>.json] [arm-specific flags]
```

Resolves `<name>` through `code/arms.py::ARMS`, trains, writes exactly one record. Idempotent:
refuses to overwrite an existing record unless `--force`.

## `code/arms.py`

```python
ARMS: dict[str, ArmSpec]
@dataclass ArmSpec:
    family: str            # baseline | existing | candidate | control
    paper: str | None
    build: Callable[[cfg], object]    # returns a trained-model-shaped object with the protocol below
    defaults: dict         # hyperparameters, overridable from the CLI
    doc: str               # one sentence for ARMS.md
```

An arm object implements: `fit_epoch(xtr, ytr, batch_size) -> None`,
`predict(x) -> LongTensor`, `capacity() -> dict`, `diagnostics() -> dict`.

**AMENDED 2026-09-21 — the two drivers are separate, and that is deliberate.** The original intent
that one runner serve both families did not survive contact with the data: `run_arm.py` calls
`data.load_boolean(cfg['booleanization'])` unconditionally, reads `model._chunk_size(...)`, and
calls `arm.accuracy()` / `arm.final_diagnostics()` — none of which apply to a float CNN arm, whose
`booleanization` is `None`. `code/arms.py` therefore does **not** call `register_cnn_arms()`.

- **TM arms** -> `code/run_arm.py`, driven by `code/queue_runner.py`.
- **CNN arms** -> `code/cnn/train.py`, driven by `code/cnn/queue_cnn.py`.

The two drivers **share one per-GPU lock**: `queue_cnn.py` imports `queue_runner._acquire`, so a TM
queue and a CNN queue still exclude each other on a card. One lock, two drivers. Records from both
validate against the same `record.py` schema and carry the same `split_hash`, which is what actually
makes the families comparable — not a shared runner.

**Amendments agreed during P0** (orchestrator-approved):

* Two more methods, `snapshot() -> dict` and `restore(snap) -> None`. The runner needs them to
  evaluate the test set **once**, at the epoch validation selected: it snapshots the model on
  every validation improvement and restores the best one *after* the training loop, so no test
  accuracy exists while any decision is being made. `arms.Arm` / `arms.TMArm` provide working
  implementations to inherit.
* `ArmSpec` gains `notes: str` (ambiguities and deviations, copied into the record) and
  `config_status: str` ∈ `paper | provisional | own` (see the `record.py` amendments).
* `capacity()` should return `n_patches` as well; `record.validate()` keys the M2 requirement
  off it, and Section 7.3 budget matching needs it.

## `code/queue_runner.py`

Adapted from `experiments/mctm/queue_runner.py`; keep the per-GPU lock and skip-if-exists. Job list
format `name|args`, one per line, `#` comments ignored. `PYTHONUNBUFFERED=1` always.

## Device policy

GPU 0 = RTX 3090 (24 GB): large-clause / large-patch TM arms.
GPU 1 = RTX 3080 (10 GB): screens, diagnostics, CNN baselines, small TM arms.
One queue per GPU, enforced by the lock.
