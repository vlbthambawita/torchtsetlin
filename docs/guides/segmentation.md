# Semantic segmentation

Dense, per-pixel prediction with `SegmentationTsetlinMachine`. For *why* it is built the way
it is — and why a `ConvTsetlinMachine` cannot do this — see
[Segmentation](../concepts/segmentation.md).

## Data layout

| | shape | dtype |
|---|---|---|
| input | `(B, Z, H, W)` | Boolean (`(B, H, W)` is treated as one plane) |
| targets | `(B, H, W)` | `long`, values in `[0, K-1]` or `ignore_index` |
| `predict(x)` | `(B, H, W)` | `long` label map |
| `vote_map(x)` | `(B, K, H, W)` | float vote sums |
| `forward(x)` | `(B·H·W, K)` | float, patches folded into the batch |

Booleanize first. A plain threshold throws away the ordering between intensities, which is
usually the information a segmenter needs; use a thermometer:

```python
import torch, torchtsetlin as tt

rgb, labels = tt.data.make_scenes(300, 32, seed=0)          # (N, 3, H, W) float, (N, H, W) long
enc = tt.data.ColorThermometerEncoder(n_bits=4, value_range=(0.0, 1.0))
x = enc(rgb).float()                                        # (N, 12, H, W) Boolean planes
```

## Training

`Trainer` detects a dense model and switches to segmentation metrics — pixel accuracy, mean
IoU, mean Dice and per-class IoU — accumulating a confusion matrix batch by batch rather than
keeping `(B, K, H, W)` vote tensors for a whole test set.

```python
model = tt.SegmentationTsetlinMachine(
    n_classes=4, n_clauses=300, T=60, s=10.0,
    patch_size=3,               # the neighbourhood each pixel is classified from
    position_encoding=8,        # thermometer bits per axis of patch position (0/False = off)
    patches_per_commit=128,     # pixels per feedback commit — see below
    weighted=True, n_states=100,
).to("cuda")
model.class_names = list(tt.data.SCENE_CLASSES)

trainer = tt.Trainer(model, device="cuda", batch_size=8)     # batch_size is in IMAGES
trainer.fit((x[:220], labels[:220]), epochs=8, val_data=(x[220:], labels[220:]))
trainer.test((x[220:], labels[220:]))
# {'test_pixel_accuracy': 0.991, 'test_mean_iou': 0.946, 'test_mean_dice': 0.972,
#  'test_iou_sky': 0.998, 'test_iou_building': 0.894, ...}
```

Or without the `Trainer`:

```python
model.update(x_batch, label_batch)     # returns pre-update folded votes (B·P, K)
model.eval()
pred = model.predict(x_test)           # (B, H, W)
```

!!! danger "`model.eval()` is not optional"
    A clause with no included literals is True while learning and False when predicting.
    Forgetting `eval()` silently changes every pixel of the output.

## The knobs that matter

### `patches_per_commit`

One batch of 8 images at 32×32 is 8 192 pixel-examples; committing their feedback all at once
is a mini-batch of 8 192, far past where batched feedback stops matching the classical
algorithm. This bounds it. The default `128` was chosen by
[ablation](../concepts/segmentation.md#commit-granularity-is-a-real-hyper-parameter): quality
is flat below it, and both the mean and the *variance* degrade above it. Lower it if learning
looks unstable; raise it — or set `None` — only when you want throughput and have checked the
cost. `feedback_mode="sequential"` is `patches_per_commit=1`.

### `patch_size` and context

A `k × k` patch is all the model sees of a pixel. When a class is not decidable from it —
building interior vs. road, in the example scenes — no amount of clause budget fixes it. Add
context instead of enlarging the patch (which grows the feature count quadratically):

```python
# (a) position literals: a clause can say "only in rows 12-18"
model = tt.SegmentationTsetlinMachine(4, 300, T=60, patch_size=3, position_encoding=8)

# (b) an OR-pooled multi-scale pyramid: neighbourhood + coarse structure, patch_size=1
pyr = tt.data.PyramidEncoder(scales=(1, 2, 4), patch=3)
x_ctx = pyr(x).float()                       # (B, Z*9*3, H, W)
model = tt.SegmentationTsetlinMachine(4, 300, T=60, patch_size=1, position_encoding=8)
```

Both work and they compose; on the example scenes building IoU goes 0.505 → 0.699 (pyramid)
→ 0.894 (position) → 0.921 (both). Pass `position_encoding` an **integer** on anything bigger
than about 64 px: `True` adds one bit per row and column, which is 1 022 extra features at
512 px.

### `ignore_index`

The void class of CamVid / Cityscapes / Pascal VOC. Pixels with this label produce no
feedback and are excluded from every metric:

```python
model = tt.SegmentationTsetlinMachine(21, 400, T=80, patch_size=3, ignore_index=255)
tt.metrics.mean_iou(pred, target, n_classes=21, ignore_index=255)
```

### Class imbalance

A Tsetlin machine has no loss to reweight, so the lever is which pixels produce feedback:

```python
p = tt.data.balanced_class_probabilities(labels, n_classes=4, floor=0.05)
model.set_class_feedback_p(p)          # per-class probability of contributing feedback
```

Use it when a rare class is being lost, and check the per-class IoU afterwards — it *costs*
accuracy once the model can already separate the classes.

### `stride` and `padding`

`padding="same"` with `stride=1` (the defaults) keep the input resolution. A larger stride
produces a coarser output grid; full-resolution label maps are subsampled to it
automatically, so you can pass `(B, H, W)` labels either way. `model.output_shape()` reports
the grid.

## Metrics

```python
tt.metrics.pixel_accuracy(pred, target, ignore_index=255)
tt.metrics.iou(pred, target, n_classes=4)        # (K,) per class, NaN where absent
tt.metrics.mean_iou(pred, target, n_classes=4)
tt.metrics.dice(pred, target, n_classes=4)       # medical-imaging convention
tt.metrics.boundary_f1(pred, target, tolerance=2)
print(tt.metrics.segmentation_report(pred, target, class_names=CLASSES))
```

All of them accept vote maps `(B, K, H, W)`, label maps `(B, H, W)` or folded votes `(N, K)`.
An absent class gets `NaN` IoU rather than `0`, and `mean_iou` skips it — averaging a zero
there would punish the model for a class the image does not contain.

!!! warning "Pixel accuracy is a weak claim"
    On any dataset with a stable layout, a baseline that knows only a pixel's *row* scores
    well. On the example scenes it reaches 0.876 pixel accuracy — and 0.410 mIoU, with zero
    on two of four classes. Compute that floor and report it next to your numbers.

## Explaining a pixel

This is the part a convolutional network cannot do. Each matching clause decodes into a
conjunction over **absolute image coordinates**:

```python
print(tt.interpret.explain_pixel(model, x_test[0], row=20, col=12, target=int(y[0, 20, 12])))
```

```
pixel (20, 12) -> 3  (true: 3)
prediction: 3  votes: [-60.0, -60.0, -60.0, 60.0]
  [+109 -> output 3] clause 1029: IF p[0,19,12] AND p[0,20,11] AND p[1,21,13] AND ...
                                     AND y>2 AND y>3 AND NOT p[3,19,11] AND ...
```

`p[1,21,13]` reads "plane 1 of pixel (21, 13) is set". Clauses get long at high `s`; cap them
with `max_included_literals` if you want rules you can read at a glance. Related tools:

```python
model.clause_map(x, clauses=[0, 5])    # (B, 2, H, W): where those clauses fire
model.clause_patch(0)                  # (Z, kh, kw): 1 = must be on, -1 = off, 0 = free
model.clause_region(0)                 # the output-grid rows/columns a clause is pinned to
tt.viz.plot_segmentation(rgb[0], target, pred, class_names=CLASSES)
tt.viz.plot_vote_map(model.vote_map(x[:1]), class_index=1, class_name="building")
```

## Real datasets

```python
x, y = tt.data.load_voc_segmentation_boolean("./data", image_set="train", size=96)
model = tt.SegmentationTsetlinMachine(21, 400, T=80, patch_size=3, ignore_index=255)
```

`load_segmentation_boolean` takes any instantiated torchvision segmentation dataset
(`VOCSegmentation`, `Cityscapes`, …), resizes masks with **nearest** interpolation — bilinear
would invent class indices — and thermometer-encodes the images. Both need `torchvision`.

## Cost and memory

Dense evaluation is one clause evaluation per pixel: ~1000× a flat classifier on a 32×32
input. Measured on a 16-core CPU (4 planes, 4 classes, 100 clauses/class, batch of 8 images):
~35 000 px/s training, ~1 090 000 px/s predicting. Above ~64 px, use a GPU; see
[CPU vs GPU](../cpu-vs-gpu.md).

`max_chunk_elements` bounds the largest intermediate; the dense model budgets
`P · (8C + 4F)` elements per image, so chunking adapts to the image size on its own. Reduce
`batch_size` (in images) before touching it.

## Multi-label masks

For overlapping structures, where a pixel may belong to several classes at once:

```python
model = tt.CoalescedSegmentationTsetlinMachine(
    n_outputs=2, n_clauses=300, T=25, patch_size=5, multi_label=True,
)
model.update(x, masks)          # masks: (B, K, H, W) Boolean
model.eval()
model.predict(x)                # (B, K, H, W) Boolean
```

`CoalescedSegmentationTsetlinMachine` also works in ordinary multi-class mode, sharing one
clause pool across classes — a better fit when classes share structure, since edges and
textures are the same features whichever class they belong to.
