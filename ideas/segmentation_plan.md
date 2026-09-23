# Plan: first-class semantic segmentation in `torchtsetlin`

Status: **implemented in 0.2.0** (all three phases). This document is kept as the design
record; see `docs/concepts/segmentation.md` and `docs/guides/segmentation.md` for the
user-facing version, and the *Deviations* section at the end for where the build differed
from the plan.

The original framing follows.

## 1. What exists today

| thing | where | importable? |
|---|---|---|
| `DenseTsetlinSegmenter` — per-pixel head over a flat `TsetlinMachine` | `examples/notebooks/05_ctm_unet_segmentation.ipynb`, cell 5 | **no**, notebook-local |
| `segmentation_metrics` (pixel acc, per-class IoU, mIoU) | same notebook, cell 5 | **no** |
| CamVid-like scene generator | same notebook, cell 7 | **no** |
| OR-pooled multi-scale context (`boolean_downsample`, `thermometer_votes`, `pyramid`) | same notebook, cell 17 | **no** |
| `ConvTsetlinMachine` / `ConvCoalescedTsetlinMachine` | `src/torchtsetlin/models/conv.py` | yes — but **classify whole images, cannot segment** |

The notebook reaches 0.996 pixel accuracy / 0.968 mIoU on synthetic scenes, so the
mechanism is validated; it is only the packaging that is missing.

## 2. Why `ConvTsetlinMachine` cannot be used as-is

`models/conv.py::_ConvMixin._evaluate` ends with

```python
return matches.any(dim=1), matches       # (B, P, C) -> (B, C)
```

That disjunction over patch positions — `c_j = ⋁_b c_j^b`, Granmo et al. 2019 — is what
gives a CTM translation tolerance, and it is exactly what destroys a dense output: after
it, the model knows a pattern occurred *somewhere*, not *where*. Every published TM
aggregates patches this way (see `experiments/convtm/LITERATURE_TM.md`, rows A11/A12), and
CTM-UNet (ISTM 2025) is the only paper that works around it — by keeping `V = CTM2D(B)` as a
vote *map* between blocks.

Segmentation therefore needs a model that **keeps the per-patch axis**, which is a different
`_evaluate` / `_votes` contract, not a hyper-parameter of the existing one.

## 3. Design

### 3.1 The key simplification

A dense segmenter *is* a convolutional TM with `padding="same"`, `stride=1`, no OR-pooling,
and a per-patch label taken from the patch's centre pixel. If the patch axis is **folded into
the batch axis** (`B' = B*P`), then `_votes`, `_select_feedback`, `_feedback_counts` and
`functional.apply_feedback` all work unchanged — the base learning loop never has to know.

So the new model is small: reuse `_ConvMixin`'s unfold-and-position-encode `_encode`, drop
the `any(dim=1)`, reshape, and teach `_coerce_targets` to accept a label map.

### 3.2 New file `src/torchtsetlin/models/segmentation.py`

```python
class SegmentationTsetlinMachine(TsetlinMachine):     # dense per-pixel classifier
class CoalescedSegmentationTsetlinMachine(CoalescedTsetlinMachine)   # shared clause bank
```

built on a shared `_DenseMixin`. Methods to override (all named against `models/base.py`):

| hook | behaviour |
|---|---|
| `_coerce_input` | accept `(B, Z, H, W)`, `(Z, H, W)`, `(B, H, W)`; record `input_shape` |
| `_infer_n_features` | `Z*kh*kw` (+ `(Py-1)+(Px-1)` position bits when `position_encoding=True`) |
| `_encode` | zero-pad by `k//2`, `unfold`, → `(B*P, 2F)` literals. `P == H*W` under same-padding |
| `_evaluate` | plain `F.clause_outputs` on the folded batch — **no** `any(dim=1)` |
| `_coerce_targets` | `(B, H, W)` long label map → `(B, P)`; `ignore_index` rows masked out |
| `_select_feedback` | `y_t.reshape(-1)`, then delegate to the parent policy |
| `_feedback_counts` | inherited flat (dense-matmul) path — no random patch draw, no `(B,P,C)` gather |
| `_chunk_elements_per_example` | `P * (C + 2F)` — per *image*, since chunking stays image-granular |

Public API on top:

```python
model.vote_map(x)      # (B, K, H, W)  float vote sums
model.predict(x)       # (B, H, W)     long label map
model.forward(x)       # (B*P, K)      folded votes — the nn.Module contract
```

`forward` keeps the folded shape so `Trainer`, `metrics` and `interpret` keep working
unmodified; `vote_map`/`predict` are the segmentation-facing views.

### 3.3 The one genuinely new design decision: commit granularity

`update()` today accumulates over every chunk and commits **once**. For segmentation, one
image batch of 8 at 32×32 is 8192 patch-examples in a single commit — far past the point
where batched feedback degrades (CLAUDE.md: "200 degrades noticeably" on Noisy XOR).

**Recommendation:** add `patches_per_commit: Optional[int] = 4096` and override `update()`
to call `_commit(acc)` per patch-chunk rather than once per batch. This makes fidelity a
function of a stated hyper-parameter instead of an accident of image size. Verify with an
ablation (`patches_per_commit` ∈ {256, 1024, 4096, ∞}) before fixing the default — the
notebook used an effective batch of 512 and still scored 0.996, so the sensitivity may be
mild for per-pixel tasks and the default can be loosened if so.

### 3.4 Class imbalance

Background dominates every real segmentation set (CamVid void, cardiac MRI myocardium).
TMs have no loss to weight, so the lever is **which patches reach `update()`**. Add
`data.BalancedPatchSampler(labels, per_class, ignore_index)` yielding class-balanced patch
indices, and document that this — not `T` or `s` — is the imbalance knob.

## 4. Supporting work

### 4.1 `metrics.py` — new functions
```python
pixel_accuracy(pred, target, ignore_index=None) -> float
iou(pred, target, n_classes, ignore_index=None) -> Tensor       # (K,), NaN for absent classes
mean_iou(...) -> float                                          # nanmean
dice(...) -> Tensor                                             # medical-imaging convention
boundary_f1(pred, target, tolerance=2) -> float                 # optional, phase 3
segmentation_report(pred, target, class_names) -> str           # mirrors classification_report
```
`confusion_matrix` already works on flattened label maps; add an `ignore_index` argument
rather than a second implementation.

### 4.2 `train/` — `Trainer` integration
- `Trainer.task` returns `"segmentation"` for `_DenseMixin` instances.
- `_batches` coerces `y` to a long `(B, H, W)` map.
- default eval metrics become `{pixel_acc, mIoU}`.
- `evaluate`/`predict` accumulate the confusion matrix across batches instead of
  concatenating `(B, K, H, W)` vote tensors (a 512×512×20 float map per image is too large
  to keep).

### 4.3 `data/` — datasets and encoders
- `make_scenes(n, H, W, seed)` — the notebook's CamVid-like generator (wavy horizon, sky /
  building / tree / road, per-class texture, illumination jitter), promoted and tested.
- `make_segmentation_shapes(n, size)` — a tiny 3-class toy for fast unit tests.
- `load_torchvision_segmentation_boolean(...)` for `VOCSegmentation` / `Cityscapes`,
  mirroring `load_mnist_boolean`, import-guarded on torchvision.
- `encoders.PyramidEncoder(scales=(1,2,4), patch=3)` — the notebook's `pyramid`, which is
  the single largest accuracy lever measured (building IoU 0.41 → 0.95).

### 4.4 `functional.py` — TM-native context primitives
```python
boolean_pool(x, factor)            # OR-pool = max_pool2d on Boolean planes
thermometer_cast(v, levels)        # integer votes -> ordered Boolean planes
```
Both are currently notebook-local and both are what make a stacked/multi-scale TM possible
without a single float parameter. Also closes `LG-005` from `LITERATURE_TM.md` by adding
`_ConvMixin.patch_clause_outputs(x) -> (B, Py, Px, C)`, a public accessor for the per-patch
match tensor that today is reachable only through the private `_encode`.

### 4.5 `viz.py`
```python
plot_segmentation(image, target, pred, palette=None, ax=None)
plot_vote_map(votes, class_index, ax=None)
```

### 4.6 `interpret.py` — the differentiating feature
```python
explain_pixel(model, x, y, x_pos, feature_names=None) -> Explanation
```
Returns the clauses that voted for a *named pixel's* class, each decoded as a statement
about neighbouring pixels. A CNN cannot do this. For the cardiac-MRI use case this is the
main reason to use a TM at all, so it should ship in phase 1, not as an afterthought.

## 5. Tests (`tests/test_segmentation.py`)

1. `test_dense_learns_solid_regions` — 3-colour blocks, mIoU > 0.95, device-parametrised.
2. `test_vote_map_shape_and_argmax_matches_predict`.
3. `test_padding_preserves_resolution` — `vote_map(x).shape[-2:] == x.shape[-2:]`.
4. `test_ignore_index_excluded_from_feedback` — `ta_state` unchanged when all labels are ignored.
5. `test_train_eval_mode_empty_clause_semantics` — the `self.training` trap, per CLAUDE.md.
6. `test_no_random_patch_draw_in_prediction` — mirrors the conv test; dense prediction must
   be RNG-free (patch `torch.rand` and assert it is never called).
7. `test_save_load_roundtrip` — lazy-init shape reconstruction through `state_dict`.
8. `test_chunking_matches_unchunked` — tiny `max_chunk_elements` gives identical results.
9. `test_iou_matches_sklearn_jaccard` (skipped without scikit-learn).
10. `test_metrics_ignore_index`.

## 6. Docs

- `docs/concepts/segmentation.md` — why OR-pooling blocks dense output, what the dense head
  does instead, the CTM-UNet result and its caveats (the notebook's honest reading: the
  paper's 96.35% is binary sky-vs-rest, a row-position-only baseline scores 0.895).
- `docs/guides/segmentation.md` — how to use it: data layout, `ignore_index`, multi-scale
  context, `patches_per_commit`, imbalance, per-pixel explanations, memory budgeting.
- `docs/examples/segmentation.md` — walkthrough of the new example script.
- `docs/api/models.md`, `api/metrics.md`, `api/data.md` — `:::` stubs for the new symbols.
- `mkdocs.yml` — three new nav entries.
- `README.md` feature list, `CHANGELOG.md` under a new `0.2.0` heading.

## 7. Example script

`examples/segmentation_scenes.py` — CPU-runnable in ~1 min, mirroring `shapes_conv.py`:
generate scenes → thermometer-encode → `SegmentationTsetlinMachine` → `Trainer` →
pixel acc / mIoU → save a `segmentation.png` triptych → print the clauses explaining one
misclassified pixel. A `--multiscale` flag turns on `PyramidEncoder` to reproduce the
building-IoU jump.

## 8. Phasing

| phase | contents | est. |
|---|---|---|
| **1 — usable** | `models/segmentation.py`, metrics, `Trainer` hookup, `make_scenes`, tests 1–8, example script, guide + API stubs | 2–3 d |
| **2 — accurate** | `PyramidEncoder`, `boolean_pool`, `thermometer_cast`, `patch_clause_outputs`, `BalancedPatchSampler`, `patches_per_commit` ablation, concepts page | 2–3 d |
| **3 — real data** | torchvision loader, `explain_pixel` + `plot_segmentation`, boundary F1, a benchmark suite entry in `bench_device.py` | 3–4 d |

Phase 1 alone makes segmentation a supported task. Phases 2–3 are what make it competitive.

## 9. Risks

- **Memory.** 512×512 same-padding = 262 144 patches/image. `_chunk_elements_per_example`
  must be right or chunking collapses (CLAUDE.md records a 4–20× slowdown from exactly this).
  Budget check belongs in test 8.
- **Throughput.** Dense evaluation is `H*W` clause evaluations per image, ~1000× a flat
  classifier. GPU-only for anything above 64×64; say so in the guide and size the example
  accordingly.
- **Accuracy ceiling.** A k×k patch cannot resolve classes that need global context — the
  notebook measures building IoU stuck at 0.41 without multi-scale. Ship phase 2 before
  claiming segmentation works on natural images.
- **Scope discipline.** Do *not* reimplement CTM-UNet's `Conv2D` front-end: the notebook
  measured it costing 5.5 points of pixel accuracy, because no gradient can cross a CTM
  block and it is therefore a frozen random projection.


---

## Deviations from this plan

Four things changed once the work was measured.

1. **`patches_per_commit` defaults to 128, not 4096.** The plan hedged and asked for an
   ablation before fixing the default; the ablation (3 seeds, `make_segmentation_shapes`)
   showed quality flat to ~128 and collapsing above it — mIoU 0.749 at 128 against 0.527 at
   4096, with the standard deviation growing 0.009 → 0.073. The variance matters as much as
   the mean: a large-commit run can look fine once and collapse the next time. 128 is the
   knee, and it is 2× faster than 64 for the same accuracy.

2. **No `BalancedPatchSampler`.** The plan wanted a sampler yielding class-balanced patch
   indices, but the model takes *images*, not patches, so a sampler would have forced callers
   out of the normal API. `class_feedback_p` (a per-class probability that a pixel takes part
   in feedback) does the same job inside `_select_feedback`, works with `Trainer` unchanged,
   and composes with `ignore_index`. `data.balanced_class_probabilities` computes the
   equalising vector. Measured, it is a rescue rather than a default — it lifts a
   pyramid-only run 0.874 → 0.927 mIoU but costs 0.05 once position literals are on.

3. **Multi-scale context shipped as an encoder, not a model.** `data.PyramidEncoder` composes
   with any dense model via `patch_size=1`, so there is no
   `MultiScaleSegmentationTsetlinMachine` to keep in sync. `functional.boolean_pool` and
   `functional.thermometer_cast` are the primitives under it, and they are also what a
   stacked (CTM-UNet-style) construction would need.

4. **`boundary_f1` landed in phase 1**, not phase 3 — it is 30 lines of max-pool and it is the
   metric that catches what a per-pixel model with no smoothing gets wrong.

Also worth recording: **position literals turned out to be the strongest single lever** on the
example scenes (building IoU 0.505 → 0.894), ahead of the pyramid (→ 0.699). The plan expected
the multi-scale pyramid to carry the result, on the strength of the notebook's measurement.
Both help and they compose (→ 0.921), but the ranking is a property of *this* dataset's fixed
vertical layout — the same property that lets a row-prior baseline reach 0.876 pixel accuracy.
On data where the structure moves, expect the pyramid to matter more and position literals
less.
