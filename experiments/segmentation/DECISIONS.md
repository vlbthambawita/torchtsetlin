# Decisions (dated; link supporting run IDs)

Initial state, 2026-10-13: repo HEAD `f5168bdf656d597461a5ef14f463afd1eee90eeb`, `git status src` clean,
`src/torchtsetlin` tree hash `453d2e8bb264fb6b9d0bdca1c52f248ad6a663351cdbc04878bbc240e318e928`
(`code/record.py::src_tree_hash`, recorded in every run's `env.src_tree_sha256`).

## D-001 (2026-10-13) — compressed execution of PLAN.md

The user asked for the plan to be executed and reported in one working session. PLAN §8
budgets 112 device-hours over 4–6 weeks. This run keeps the plan's structure (gates G0–G4,
baselines B0–B5, H1–H5, sealed final split, paired bootstrap) but compresses it:

* Seeds: screening 1 seed; promoted arms 3 seeds; finalists and their direct controls 5 seeds (as planned).
* TM epochs fixed per dataset after the E0 learning curves (D-004), not tuned per arm.
* D1 is reduced to one generator (`longrange`, the H3 diagnostic). Thin-curve / ring /
  rare-object / layout-shift generators are **not built**; H7 (topology) and H6 (relational
  messages) are therefore **not run** — their triggers could not be evaluated. Recorded as
  not-done, not as negative results.
* Kvasir 256×256 finalists are **not run**: all real-data numbers are from 128×128 models
  restored to the native grid. The plan's 256 tier is left for later.
* No augmentation for any family on either dataset (shared, therefore fair; differs from typical U-Net practice).
* Timing comparisons use concurrent jobs during training (5–6 processes per GPU, measured
  2.4–2.6× aggregate throughput); per-run `train_time_s` is therefore inflated by contention.
  Inference latency for the finalists is measured separately on an idle GPU (`code/bench.py`).

## D-002 (2026-10-13) — Kvasir duplicate grouping, frozen before any model fit

Rule: exact SHA-256 duplicates plus 256-bit difference-hash Hamming distance ≤ 12, transitive.
The nearest-neighbour distance histogram has a gap between 12 and 31; the four pairs ≤ 12 were
inspected visually and are all the same frame (re-encoded/cropped). Result: 996 groups, four
2-member groups; split 700/150/150 (`manifests/kvasir_split.json`). No patient identifiers exist,
so this is an image/group-held-out split, not patient-independent.

## D-003 (2026-10-13) — ACDC split

16 train / 4 val patients per group (`split_seed=1234`): 80 train, 20 val patients
(1554 / 348 slices at ED+ES). The 50 testing patients are the sealed final set
(`manifests/acdc_split.json`). `data.load_split(..., "final")` refuses unless `SEG_UNSEAL=1`.

## D-004 (2026-10-13) — E0 observations and settings carried forward

* Training cost is per-commit launch overhead (~4.4 ms/commit idle, independent of F), so
  wall time ∝ pixels / `patches_per_commit`. Kvasir 128 px, q=128: ~190 s/epoch alone.
* E0 TM runs used `order="perm"`. From E2 on, the *baselines* B1/B2/B3 use `order="raster"`
  (library raster order of valid pixels; bit-identical to the library when there is no
  padding, `tests/test_adapter.py`) so B-arms represent the current library behaviour.
* TM epochs: 10 on Kvasir (curve still rising at epoch 3 in E0), checkpoint = best validation
  epoch (earliest on ties). CNN: up to 60 epochs, same rule.

## D-005 (2026-10-13) — invalid records archived

Eight `E2_kvasir_B5_*` records failed on a harness bug (Boolean tensor passed to a float conv;
fixed in `run.model_input`). They are moved to `results/invalid/` (kept, not summarised) and
the configs re-run. Any `E1_*_H2_linear_*` failure of the same cause is handled identically.

## D-006 (2026-10-13 ~13:55) — Kvasir re-prioritisation after the B5 receptive-field control

Evidence (validation, 1 seed): local CNN with B1's 3×3 receptive field on B1's Boolean planes
`E2_kvasir_B5_local_bool_t0_s0` = 0.426 mIoU (= all-background floor 0.425); dense TM B1/B2
≈ 0.48–0.50 (`E0_kvasir_B1_s0`, `E0_kvasir_B2_pyr_or_s0`, `E2_kvasir_B2_t*`); U-Net on the B2
Boolean pyramid planes `E2_kvasir_B5_unet_bool_t0_s0` = 0.69; U-Net on RGB up to 0.789 native.
Vote clipping/ties explain ≤ 1.6% of pixel decisions (`DIAG_ties_*`).
Conclusion: on Kvasir the binding limitation is spatial context, not TM learning dynamics.

Actions:
* The three `E1_kvasir_H1_*_q32` runs were **stopped by hand** after ~1 h (≈ 2 of 10 epochs,
  ~4 h remaining each under contention); no record is written for them. H1 order × commit
  size is answered on D0 scenes (3 seeds) plus Kvasir q ∈ {128, 512}.
* Remaining E1 Kvasir H1/H2 arms are not run at B2 settings (local-context regime where all
  arms are near the floor). Instead E2 adds wide-context arms: OR vs occupancy pyramids at
  scales (1,2,4,8,16,32) — the regime where H2's OR-saturation argument applies — the H3
  coarse-to-fine stack, and matched linear / U-Net-on-planes controls.

## D-007 (2026-10-13 ~14:50) — pixel-level feature diagnostic and corrected H1 wave

`DIAG_pixels_or_wide` (60k sampled train pixels, 20k val pixels, same wide-OR Boolean vectors):
logistic regression val mIoU 0.53, gradient boosting 0.56 (train 0.75), flat TM 0.53–0.56
with large epoch-to-epoch swings and full collapse to one class at batch 1024. So the fixed
features cap per-pixel classifiers near 0.55 regardless of learner, and the TM's dense
pipeline (≈0.45–0.50) is below its own sampled-pixel level.

Chunk check (LG-001) — `cfg = or_wide, 600 clauses/class`: `_chunk_size` = 1 image of 8
(`max_chunk_elements = 2**27`); B1 settings also 1. The E1 Kvasir `perm`/`disperse` arms therefore
only reordered pixels *within* one image. Wave `configs/e1x.json` (`H1x_*`) repeats H1 with
`cross_image=True` (whole image batch in one chunk) and pixel sampling (512 or 2048 pixels per
image, with replacement), at q ∈ {32, 128}; sampled arms run 20 epochs (still ≤ 1/16 of the
all-pixel exposure). One single-image-chunk sampled arm isolates the pool effect.

## D-008 (2026-10-13 ~15:20) — OOM outcomes and the gathered literal path

Seven `E1_kvasir_H1x_*` runs ended `oom` (records kept as outcomes): the cross-image pool held the
whole batch's literals while ~20 jobs shared the GPUs. `ExpSegTM.gather` (patch-1 models) builds
literals only for selected pixels; `tests/test_adapter.py::test_gathered_path_equals_full_encoding`
shows identical automata. Re-run as `H1x_*_g` (`configs/e1x_g.json`).

## D-009 (2026-10-13 ~15:35) — B3 search trimmed

GPU throughput is saturated (~27 concurrent jobs). The coalesced B3 search is cut to the one
config already running (`B3_t00`, 1200 shared clauses); `B3_t01..t05` are not run. B2 keeps its 12 configs.

## D-010 (2026-10-13 ~15:45) — contention OOMs

All seven `H1x_*_g` reruns also ended `oom` (GPU memory exhausted by ~27 jobs; the failing
allocation was the 16-image validation forward pass). TM validation now runs 4 images per batch,
`AcceleratorError` OOMs are classified as `oom`, and the arms run again as `H1x_*_r2` on cuda:0
only. For each arm the analysis uses the latest *completed* attempt; OOM records stay as outcomes.

## D-011 (2026-10-13 ~17:00) — E4 Kvasir comparison frozen (pre-registration, before any final-split access)

Selection evidence (validation, seed 0; records named):
* Tuned B2 = best of the 12-config B2 search by validation screen mIoU: `B2_t08` 0.505 and
  `B2_t07` 0.504 are tied within 0.001; **`B2_t07` hyper-parameters (600 clauses/class,
  T = 0.2·600, s = 20, no literal cap)** are used for every TM arm so the factorial shares
  hyper-parameters (the capped config is also ~2× slower per epoch).
* H1: sampled pixels (512/image, mixture a = b = 0.25, q = 128, cross-image pool of 16
  images, 20 epochs) — `E1_kvasir_H1x_samp512_a0.25_b0.25_q128_r2_s0` 0.568 native vs all-pixel
  0.50–0.52. Mixtures differ by < 0.01 at one seed; a = b = 0.25 fixed (plan's joint mixture).
* H2: occupancy did not beat OR at equal feature count (`S_occmaj_wide` 0.540, `S_occ3_wide`,
  `CTX_occmaj_wide` 0.519 vs `CTX_or_wide` 0.516); contrast literals no better. The wide **OR**
  pyramid (1,2,4,8,16,32) is the context component.
* H3: no gain over its shuffled control (all-pixel: oof 0.499, shuffled 0.500); not in finalist.
* H4/H5: rejected for the finalist (≤ +0.007 mIoU with a boundary-F1 drop > 0.01, `E3_refine_*`).
* U-Net: best of the 12-config search `B4_t11` (lr 1e-3, wd 0, width 32, Dice weight 0).

Frozen E4 (`configs/e4kv.json`): factorial base / A (+sampling) / B (+wide OR) / A+B, and U-Net,
5 seeds (0–4), same seeds and checkpoint rule (best validation screen mIoU, earliest on ties).
Deviation: the plan's "best two on three seeds" re-screen is skipped; the single best goes to 5 seeds.

**Primary acceptance route (one, fixed now): quality.** A+B vs base on the sealed Kvasir final split:
mean native mIoU difference ≥ +0.02 with paired hierarchical-bootstrap 95% lower bound > 0
(2,000 draws, seed 20261013, resampling seeds then duplicate-groups); no mean class-aware
boundary-F1 (normalised tolerance, foreground) decline > 0.01. ACDC must confirm the sign with
its own frozen comparison and the foreground-Dice guardrail (mean decline ≤ 0.01).
CNN comparison: gap and cost reported; no parity claim is pre-registered.

## D-012 (2026-10-13 ~17:50) — E4 ACDC comparison frozen (pre-registration, before final-split access)

ACDC screening (validation, seed 0, native mIoU): `E4_acdc_B2_allpix_raster` 0.292,
`E4_acdc_S_pyr124_a0.25_b0.25` 0.304, `E4_acdc_CTX_or_wide_allpix` 0.352,
`E4_acdc_S_or_wide_a0.25_b0.25` 0.396 (a = b = 0 also 0.396; class/boundary-only mixtures 0.384–0.388);
U-Net search (6 of the 12 Kvasir-grid configs, budget) best `E4_acdc_B4_t04` 0.863
(lr 1e-3, wd 1e-4, width 32, Dice 0). B5: local CNN 0.241 (floor), U-Net on wide-OR Boolean planes 0.810.

Frozen (`configs/e4ac.json`): the same factorial as D-011 with 8 thermometer levels on the
normalised MRI intensity, TM hyper-parameters of D-011, mixture a = b = 0.25, 5 seeds.
Deviation (cost): all-pixel arms (`F_base_B2`, `F_B_wide`) train **4 epochs** (≈ 1.3 h each
under contention) instead of 10; sampled arms 20 epochs (in total ≈ 0.62 of the pixel exposure of one
all-pixel epoch's pixels in total). Acceptance on ACDC as in D-011: sign of A+B − base
confirmed, foreground 3D Dice guardrail.

## D-013 (2026-10-13 ~18:55) — reproducibility is per GPU model

Same config and seed reproduce bit-identically on the same GPU (`E2_kvasir_CTX_or_wide_s0` and
`E4_kvasir_F_B_wide_s0`, both RTX 3090: identical per-epoch curves), but not across GPU models
(`E2_kvasir_B2_t07_s0` on the RTX 3080 vs `E4_kvasir_F_base_B2_s0` on the 3090: same config,
val native mIoU 0.512 vs 0.498): the CUDA Philox stream used by `torch.rand` depends on the launch
geometry of the device. Every run records `env.gpu`. All E4 Kvasir runs ran on the 3090 and all E4
ACDC runs on the 3080, so paired seeds are device-matched within each comparison.

## D-014 (2026-10-13 ~20:00) — ACDC E4 OOMs re-run on the same GPU model

A second queue on the 10 GB RTX 3080 overloaded it: ten ACDC E4 runs ended `oom` (records kept).
They are re-run with the same configs as `<run_id>_r2` on the same GPU model (device-matched, D-013),
three at a time (`configs/e4ac_r2.json`). The analysis takes, per arm and seed, the completed attempt.
