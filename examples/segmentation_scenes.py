"""Semantic segmentation with a dense Tsetlin machine (synthetic street scenes).

A convolutional Tsetlin machine takes the disjunction over patch positions, so it can tell
you *that* a pattern occurred but not *where* — which is fatal for a dense output.
``SegmentationTsetlinMachine`` keeps the patch axis instead: every pixel is one example,
classified from its ``k x k`` neighbourhood, and the prediction is a label map.

The scenes have a wavy horizon, sky, buildings, trees and a road, with per-class texture and
an illumination jitter, so classes are not separable by colour alone. The interesting class
is **building**: a 3x3 patch of building interior and a 3x3 patch of road look alike, and
telling them apart needs context that does not exist inside the window. Two TM-native ways to
supply it are measured here — position literals and an OR-pooled multi-scale pyramid.

Watch out for the ``--baseline`` number: predicting each pixel from its *row alone*, with no
image content at all, already reaches ~0.88 pixel accuracy on these scenes. Pixel accuracy on
a layout-driven dataset is a weak claim; mIoU is the one to read.

Run:  python examples/segmentation_scenes.py
      python examples/segmentation_scenes.py --pyramid --position 8
      python examples/segmentation_scenes.py --all --plot
      python examples/segmentation_scenes.py --commit-ablation
"""

import argparse
import time

import torch

import torchtsetlin as tt

CLASSES = list(tt.data.SCENE_CLASSES)


def booleanize(rgb, n_bits, device):
    """Thermometer-encode each colour channel — keeps the ordering a threshold would lose."""
    enc = tt.data.ColorThermometerEncoder(n_bits=n_bits, value_range=(0.0, 1.0)).to(device)
    return enc(rgb).float()


def row_prior_baseline(l_train, l_test):
    """The honest floor: label every pixel by the most common class in its row."""
    H = l_train.shape[1]
    mode = torch.stack(
        [torch.bincount(l_train[:, r, :].reshape(-1), minlength=len(CLASSES)).argmax()
         for r in range(H)]
    ).to(l_test.device)
    pred = mode.view(1, H, 1).expand_as(l_test)
    return tt.metrics.segmentation_metrics(pred, l_test, len(CLASSES), CLASSES)


def train_one(label, x_tr, y_tr, x_te, y_te, args, patch, position, class_p=None):
    tt.seed_everything(args.seed)
    model = tt.SegmentationTsetlinMachine(
        len(CLASSES),
        n_clauses=args.clauses,
        T=args.T,
        s=args.s,
        patch_size=patch,
        position_encoding=position,
        patches_per_commit=args.patches_per_commit,
        weighted=True,
        n_states=100,
    ).to(args.device)
    model.class_names = CLASSES
    if class_p is not None:
        model.set_class_feedback_p(class_p)
    t0 = time.time()
    trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size, verbose=False)
    trainer.fit((x_tr, y_tr), epochs=args.epochs)
    res = trainer.test((x_te, y_te))
    model.eval()
    bf = tt.metrics.boundary_f1(model.predict(x_te), y_te, tolerance=2)
    print(
        f"  {label:<32} acc={res['test_pixel_accuracy']:.4f}  mIoU={res['test_mean_iou']:.4f}  "
        + "  ".join(f"{c}={res['test_iou_' + c]:.3f}" for c in CLASSES)
        + f"  BF={bf:.3f}  [{time.time() - t0:.0f}s]"
    )
    return model, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32, help="scene side length")
    ap.add_argument("--n-train", type=int, default=220)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--clauses", type=int, default=300, help="clauses per class")
    ap.add_argument("--T", type=float, default=60)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--bits", type=int, default=4, help="thermometer levels per colour channel")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="images per update")
    ap.add_argument("--patches-per-commit", type=int, default=128,
                    help="pixels per feedback commit (the fidelity knob)")
    ap.add_argument("--position", type=int, default=0,
                    help="thermometer bits per axis of patch-position literals (0 = off)")
    ap.add_argument("--pyramid", action="store_true",
                    help="OR-pooled multi-scale planes instead of a single 3x3 patch")
    ap.add_argument("--balanced", action="store_true",
                    help="equalise how often each class produces feedback")
    ap.add_argument("--all", action="store_true", help="run every combination as a table")
    ap.add_argument("--commit-ablation", action="store_true",
                    help="sweep --patches-per-commit and stop")
    ap.add_argument("--baseline", action="store_true", default=True,
                    help="also report the row-prior floor")
    ap.add_argument("--explain", type=int, nargs=2, metavar=("ROW", "COL"), default=None,
                    help="print the clauses that decided this pixel of the first test image")
    ap.add_argument("--plot", action="store_true", help="save segmentation_scenes.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tt.seed_everything(args.seed)
    n = args.n_train + args.n_test
    rgb, lab = tt.data.make_scenes(n, args.size, seed=args.seed, device=args.device)
    x = booleanize(rgb, args.bits, args.device)
    x_tr, y_tr = x[: args.n_train], lab[: args.n_train]
    x_te, y_te = x[args.n_train :], lab[args.n_train :]
    freq = torch.bincount(lab.reshape(-1), minlength=len(CLASSES)).float()
    print(f"scenes {tuple(rgb.shape)} -> Boolean planes {tuple(x.shape)}")
    print("class frequency: " + "  ".join(
        f"{c}={p:.3f}" for c, p in zip(CLASSES, (freq / freq.sum()).tolist())))

    if args.baseline:
        b = row_prior_baseline(y_tr, y_te)
        print(f"\n  {'row prior (no image content)':<32} acc={b['pixel_accuracy']:.4f}  "
              f"mIoU={b['mean_iou']:.4f}  "
              + "  ".join(f"{c}={b['iou_' + c]:.3f}" for c in CLASSES))
        print("  ^ pixel accuracy is nearly free on this layout; read mIoU instead\n")

    if args.commit_ablation:
        print("patches_per_commit — fidelity to the classical algorithm vs throughput")
        for ppc in (32, 128, 512, 2048, None):
            args.patches_per_commit = ppc
            train_one(f"patches_per_commit={ppc}", x_tr, y_tr, x_te, y_te, args, 3, args.position)
        return

    pyr = tt.data.PyramidEncoder(scales=(1, 2, 4), patch=3)
    class_p = tt.data.balanced_class_probabilities(y_tr, len(CLASSES), floor=0.05)

    runs = []
    if args.all:
        px_tr, px_te = pyr(x_tr).float(), pyr(x_te).float()
        runs = [
            ("single scale 3x3", x_tr, x_te, 3, False, None),
            ("+ position literals", x_tr, x_te, 3, 8, None),
            ("+ balanced feedback", x_tr, x_te, 3, 8, class_p),
            ("pyramid 1x/2x/4x", px_tr, px_te, 1, False, None),
            ("pyramid + position", px_tr, px_te, 1, 8, None),
        ]
    else:
        if args.pyramid:
            xt, xv, patch = pyr(x_tr).float(), pyr(x_te).float(), 1
            name = "pyramid 1x/2x/4x"
        else:
            xt, xv, patch = x_tr, x_te, 3
            name = "single scale 3x3"
        pos = args.position or False
        if pos:
            name += f" + position({pos})"
        if args.balanced:
            name += " + balanced"
        runs = [(name, xt, xv, patch, pos, class_p if args.balanced else None)]

    model = None
    for label, xt, xv, patch, pos, cp in runs:
        model, _ = train_one(label, xt, y_tr, xv, y_te, args, patch, pos, cp)

    print()
    print(tt.metrics.segmentation_report(model.predict(runs[-1][2]), y_te, class_names=CLASSES))

    if args.explain is not None:
        r, c = args.explain
        print()
        print("Why this pixel got its label — clauses decode to statements about named pixels:")
        print(tt.interpret.explain_pixel(
            model, runs[-1][2][0], r, c, target=int(y_te[0, r, c]), max_clauses=5
        ))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        pred = model.predict(runs[-1][2][:1])[0]
        fig = tt.viz.plot_segmentation(
            rgb[args.n_train], y_te[0], pred, class_names=CLASSES, n_classes=len(CLASSES)
        )
        fig.savefig("segmentation_scenes.png", dpi=130)
        print("\nsaved segmentation_scenes.png")


if __name__ == "__main__":
    main()
