# Segmentation on generated scenes

Full script: `examples/segmentation_scenes.py`. Runs in about a minute on a GPU, a few on a
CPU, and needs no download. See [Segmentation](../concepts/segmentation.md) for the algorithm
and the [segmentation guide](../guides/segmentation.md) for the API.

## The task

`tt.data.make_scenes` generates CamVid-like street scenes: a wavy horizon, a large contiguous
sky, buildings standing on the horizon, trees and a road — with per-class texture (window
grids, foliage noise, lane markings) and a global illumination jitter, so the classes are not
separable by colour alone. Class frequencies are heavily imbalanced, as in CamVid
(road 0.60, sky 0.30, building 0.07, tree 0.02).

```python
import torch, torchtsetlin as tt

tt.seed_everything(0)
rgb, labels = tt.data.make_scenes(300, 32, seed=0)          # (300, 3, 32, 32), (300, 32, 32)
enc = tt.data.ColorThermometerEncoder(n_bits=4, value_range=(0.0, 1.0))
x = enc(rgb).float()                                        # (300, 12, 32, 32) Boolean

model = tt.SegmentationTsetlinMachine(
    4, n_clauses=300, T=60, s=10.0, patch_size=3,
    position_encoding=8, weighted=True, n_states=100,
).to("cuda")
model.class_names = list(tt.data.SCENE_CLASSES)

trainer = tt.Trainer(model, device="cuda", batch_size=8)
trainer.fit((x[:220], labels[:220]), epochs=8)
trainer.test((x[220:], labels[220:]))
```

```
test_pixel_accuracy  0.991
test_mean_iou        0.946
test_iou_sky         0.998   test_iou_building  0.894
test_iou_tree        0.903   test_iou_road      0.989
```

## What the script measures

`--all` runs the whole table (300 clauses/class, 8 epochs, 220 train / 80 test):

| | pixel acc. | mIoU | sky | building | tree | road |
|---|---|---|---|---|---|---|
| *row prior, no image content* | *0.876* | *0.410* | *0.715* | *0.000* | *0.000* | *0.923* |
| single scale 3×3 | 0.934 | 0.798 | 0.980 | 0.505 | 0.808 | 0.898 |
| pyramid 1×/2×/4× | 0.967 | 0.874 | 0.982 | 0.699 | 0.867 | 0.950 |
| single scale + position | 0.991 | 0.946 | 0.998 | 0.894 | 0.903 | 0.989 |
| pyramid + position | **0.993** | **0.959** | 0.996 | **0.921** | 0.926 | 0.992 |

**Building is the interesting column.** A 3×3 patch of building interior and a 3×3 patch of
road look alike once illumination varies; separating them needs to know where the horizon is,
which is not inside the window. Both context mechanisms supply that, and they compose.

**The first row is the point of the `--baseline` flag.** A classifier that has never seen a
pixel — it labels each pixel by the most common class in its row — reaches 0.876 pixel
accuracy, because these scenes have a fixed vertical layout. Its mIoU is 0.410, with zero on
two classes. Report that floor next to your own numbers; it is the same objection that
applies to *CTM-UNet*'s headline 96.35%, which is a binary sky-vs-background task.

## Other flags

```bash
python examples/segmentation_scenes.py                      # single scale, with the baseline
python examples/segmentation_scenes.py --pyramid --position 8
python examples/segmentation_scenes.py --all --plot         # the table + segmentation_scenes.png
python examples/segmentation_scenes.py --commit-ablation    # patches_per_commit sweep
python examples/segmentation_scenes.py --explain 20 12      # why one pixel got its label
python examples/segmentation_scenes.py --balanced           # class-balanced feedback
```

`--explain` prints the clauses that decided a single pixel, decoded into statements about
named pixels:

```
pixel (20, 12) -> 3  (true: 3)
prediction: 3  votes: [-60.0, -60.0, -60.0, 60.0]
  [+109 -> output 3] clause 1029: IF p[0,19,12] AND p[0,20,11] AND p[1,21,13] AND ...
                                     AND y>2 AND y>3 AND NOT p[3,19,11] AND ...
```

`--commit-ablation` sweeps `patches_per_commit`, the fidelity/throughput knob described in
[Commit granularity](../concepts/segmentation.md#commit-granularity-is-a-real-hyper-parameter).

## A shape task for tests

`tt.data.make_segmentation_shapes` is the fast toy used by the test suite: solid discs
(class 1) and rings (class 2) on a Boolean background. A ring's *interior* is background
pixels labelled class 2, so the classes are only separable from the neighbourhood — a
per-pixel lookup cannot solve it.

```python
x, y = tt.data.make_segmentation_shapes(120, size=16, seed=0)   # (120, 1, 16, 16), (120, 16, 16)
```
