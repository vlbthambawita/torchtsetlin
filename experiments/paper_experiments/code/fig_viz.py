"""Paper figure: what torchtsetlin's interpretation and plotting tools show.

Trains three small models with the library's own loaders, then draws, left to right:
  (a) the global literal importance of a flat MNIST machine per class (interpret.global_feature_importance),
  (b) the heaviest clauses of a convolutional MNIST machine (viz.plot_conv_clause),
  (c) a dense segmentation machine on a synthetic street scene: image, ground truth,
      prediction and the vote map of one class (viz.segmentation_palette, viz.plot_vote_map).
This is an illustration, not a benchmark: one seed, a few epochs, nothing reported from it.

    python code/fig_viz.py [--device cuda]
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patches
import torch

import torchtsetlin as tt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
PAPER = Path("/work/vajira/DL2026/PAPER---torchtsetlin--Journal-of-Machine-Learning-Research/paper")
MNIST_ROOT = HERE.parents[1] / "data"


def flat_mnist(dev):
    tt.seed_everything(0)
    x, y = tt.data.load_mnist_boolean(str(MNIST_ROOT), train=True, flatten=True, device=dev)
    xt, yt = tt.data.load_mnist_boolean(str(MNIST_ROOT), train=False, flatten=True, device=dev)
    m = tt.TsetlinMachine(None, 10, n_clauses=2000, T=50, s=10.0).to(dev)
    tt.Trainer(m, device=dev, batch_size=32, verbose=False).fit((x, y), epochs=5)
    m.eval()
    return m, float((m.predict(xt) == yt).float().mean())


def conv_mnist(dev):
    tt.seed_everything(0)
    enc = tt.data.AdaptiveThresholdEncoder(block_size=11, C=2, method="gaussian").to(dev)
    x, y = tt.data.load_mnist_boolean(str(MNIST_ROOT), train=True, encoder=lambda v: enc(v * 255.0), device=dev)
    xt, yt = tt.data.load_mnist_boolean(str(MNIST_ROOT), train=False, encoder=lambda v: enc(v * 255.0), device=dev)
    m = tt.ConvTsetlinMachine(10, n_clauses=500, T=1000, s=5.0, patch_size=10, weighted=True,
                              input_shape=tuple(x.shape[1:])).to(dev)
    tt.Trainer(m, device=dev, batch_size=16, verbose=False).fit((x, y), epochs=5)
    m.eval()
    return m, float((m.predict(xt) == yt).float().mean())


def segmentation(dev):
    tt.seed_everything(0)
    classes = list(tt.data.SCENE_CLASSES)
    rgb, lab = tt.data.make_scenes(300, 32, seed=0, device=dev)
    x = tt.data.ColorThermometerEncoder(n_bits=4, value_range=(0.0, 1.0)).to(dev)(rgb).float()
    x = tt.data.PyramidEncoder(scales=(1, 2, 4), patch=3)(x).float()
    m = tt.SegmentationTsetlinMachine(len(classes), n_clauses=300, T=60, s=10.0, patch_size=1,
                                      position_encoding=8, patches_per_commit=128, weighted=True,
                                      n_states=100).to(dev)
    tt.Trainer(m, device=dev, batch_size=8, verbose=False).fit((x[:220], lab[:220]), epochs=8)
    m.eval()
    k = 220 + 3
    votes = m.vote_map(x[k:k + 1])
    pred = m.predict(x[k:k + 1])[0]
    miou = tt.metrics.segmentation_metrics(m.predict(x[220:]), lab[220:], len(classes), classes)["mean_iou"]
    return classes, rgb[k], lab[k], pred, votes, float(miou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    flat, acc_flat = flat_mnist(dev)
    conv, acc_conv = conv_mnist(dev)
    classes, img, gt, pred, votes, miou = segmentation(dev)

    plt.rcParams.update({"font.family": "serif", "font.size": 6.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 2.0))
    gs = fig.add_gridspec(2, 15, wspace=0.12, hspace=0.35,
                          width_ratios=[1] * 5 + [0.25] + [1] * 4 + [0.25] + [1] * 4)

    # (a) global importance of the positive literals (pixel on) per class, as 28x28 maps
    g = tt.interpret.global_feature_importance(flat).cpu()[:, :784].reshape(10, 28, 28)
    for d in range(10):
        ax = fig.add_subplot(gs[d // 5, d % 5])
        ax.imshow(g[d], cmap="magma", interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(str(d), fontsize=6, pad=1)

    # (b) among the 100 heaviest clauses of the conv machine, the eight that require the most
    # pixels to be on: these read as stroke templates
    w = conv.weights.abs()
    heavy = w.argsort(descending=True)[:100].tolist()
    strokes = sorted(heavy, key=lambda c: -int((conv.clause_patch(c)[0] == 1).sum()))
    for i, c in enumerate(strokes[:8]):
        ax = fig.add_subplot(gs[i // 4, 6 + i % 4])
        tt.viz.plot_conv_clause(conv, c, ax=ax, show_region=False)
        ax.set_title(f"w={int(w[c])}", fontsize=6, pad=1)

    # (c) segmentation: image, truth, prediction, vote map of "building"
    pal = tt.viz.segmentation_palette(len(classes))
    panels = [("image", img.permute(1, 2, 0).cpu().numpy()), ("truth", pal[gt.cpu().numpy()]),
              ("prediction", pal[pred.cpu().numpy()])]
    for i, (name, im) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, 11 + (i % 2) * 2:13 + (i % 2) * 2])
        ax.imshow(im, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(name, fontsize=6, pad=1)
    ax = fig.add_subplot(gs[1, 13:15])
    b = classes.index("building")
    v = votes[0, b].cpu().numpy()
    lim = abs(v).max() or 1.0
    ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("votes: building", fontsize=6, pad=1)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.80, bottom=0.12)
    groups = ((0, 5, "(a) flat TM: importance per class"),
              (6, 10, "(b) conv. TM: stroke clauses"),
              (11, 15, "(c) dense TM: segmentation"))
    for c0, c1, label in groups:
        x0 = gs[0, c0].get_position(fig).x0
        x1 = gs[0, c1 - 1].get_position(fig).x1
        fig.text((x0 + x1) / 2, 0.985, label, fontsize=6.8, weight="bold", va="top", ha="center")
    fig.text((gs[0, 6].get_position(fig).x0 + gs[0, 9].get_position(fig).x1) / 2, 0.05,
             "black: pixel on, white: off, grey: either", fontsize=6, ha="center")
    handles = [matplotlib.patches.Patch(color=pal[i], label=c) for i, c in enumerate(classes)]
    fig.legend(handles=handles, loc="lower center", ncol=len(classes), fontsize=5.5, frameon=False,
               handlelength=0.8, handletextpad=0.3, columnspacing=0.6,
               bbox_to_anchor=((gs[0, 11].get_position(fig).x0 + gs[0, 14].get_position(fig).x1) / 2, -0.03))
    out = PAPER / "figures" / "viz.pdf"
    fig.savefig(out)
    fig.savefig(HERE / ".cache" / "viz.png", dpi=200)
    meta = dict(flat_accuracy=acc_flat, conv_accuracy=acc_conv, seg_mean_iou=miou,
                version=tt.__version__, device=dev)
    (HERE / "results" / "fig_viz.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(meta)


if __name__ == "__main__":
    main()
