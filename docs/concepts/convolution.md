# Convolution

A convolutional Tsetlin machine (Granmo et al., 2019; book chapter 4) lets clauses look for
patterns *anywhere* in an image or sequence instead of memorising absolute pixel positions.

## Patches and position literals

An input image `(Z, H, W)` of Boolean planes is cut into every `kh × kw` window (with a
stride). Each window becomes a patch with

* `Z * kh * kw` pixel features, followed by
* thermometer-encoded coordinates: `P_y - 1` bits `[y > k]` and `P_x - 1` bits `[x > k]`,
  where `P_y`, `P_x` are the number of window positions along each axis
  (`position_encoding=True`).

A clause that includes `y > 2 AND NOT y > 5` therefore only matches windows in rows 3–5,
i.e. it can learn *where* a pattern is allowed to appear — or ignore position entirely.

## Evaluation and learning

A clause is True for an image if it matches **at least one** patch (OR over patches). Vote
sums, `T` and the Type I / Type II selection are unchanged. When a matching clause receives
Type Ia or Type II feedback, **one matching patch is drawn uniformly at random** and its
literals are used for the update; Type Ib (no match) needs no patch.

`torchtsetlin` implements this with `torch.nn.functional.unfold` for the patches, a single
matmul for all (patch, clause) matches and a gather of the randomly chosen patch rows for the
feedback counts. Memory is bounded by chunking the batch (see
[Batched vs sequential feedback](batching.md)).

```python
model = tt.ConvTsetlinMachine(
    n_classes=10, n_clauses=200, T=100, s=10, patch_size=10, weighted=True,
).to("cuda")
model.update(images, labels)          # images: (B, 1, 28, 28) Boolean
model.clause_patch(0)                 # (Z, kh, kw) map: 1 = must be on, -1 = off, 0 = free
model.clause_region(0)                # allowed window rows/columns of clause 0
tt.viz.plot_conv_clauses(model, range(16))
```

Variants: `ConvCoalescedTsetlinMachine` (shared clauses, multi-label capable),
`ConvRegressionTsetlinMachine`, and `Conv1dTsetlinMachine` for sequences `(B, Z, L)`.

## Runnable examples

| script | model | what it demonstrates |
|---|---|---|
| `examples/shapes_conv.py` | `ConvTsetlinMachine` | one clause beats a flat machine that has to memorise every position (1.000 on every seed, against 0.55–0.91), and what `position_encoding` costs on a translation-invariant task |
| `examples/conv1d_ramps.py` | `Conv1dTsetlinMachine` | a motif anywhere in a thermometer-encoded signal; the clauses print as the ramp they detect |
| `examples/conv_regression_blobs.py` | `ConvRegressionTsetlinMachine` | reading a continuous quantity (a square's side length) off an image |
| `examples/mnist_conv.py` | `ConvTsetlinMachine`, `ConvCoalescedTsetlinMachine` | the real thing: 10×10 windows, adaptive thresholding, `--stride`, `--coalesced` |

## Tips

* Booleanize images with `AdaptiveThresholdEncoder` (MNIST-like) or
  `ColorThermometerEncoder` (RGB) — see [Booleanization](booleanization.md).
* Typical MNIST settings from the literature: 10 × 10 windows, 2000–8000 clauses per class,
  `T` around 1.25–12.5 × clauses-per-class, `s` 5–10, weighted clauses.
* Set `position_encoding=False` for tasks that are fully translation invariant; position
  literals otherwise make clauses more specific and slower to generalise on small data.
