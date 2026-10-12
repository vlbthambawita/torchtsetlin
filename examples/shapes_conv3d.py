"""3D convolution: shapes that only differ along depth (toy volumes).

A 3x3x3 shape sits at a random position in a ``(D, H, W)`` volume. Class 0 (*spindle*) stacks
the slices ``plus, dot, plus``; class 1 (*3D cross*) stacks ``dot, plus, dot``. Any single
slice is a plus or a dot in both classes, so only the order along depth separates them.

Three models get the same clause budget:

* ``Conv3dTsetlinMachine`` with a 3x3x3 window: one clause covers the shape at every
  position, in depth too.
* "depth as channels": the 2D ``ConvTsetlinMachine`` on ``(Z*D, H, W)``. Its window spans
  the full depth, so it has to learn a separate copy of each shape for every depth offset.
* a flat ``TsetlinMachine`` that has to learn every position separately.

The ``--position-encoding`` flag adds depth/row/column thermometer literals. On a fully
translation-invariant task they only make clauses more specific (see
``docs/concepts/convolution.md``).

Run:  python examples/shapes_conv3d.py --baselines
      python examples/shapes_conv3d.py --depth 12 --size 16 --plot
"""

import argparse

import torch

import torchtsetlin as tt


def ascii_volume(patch: torch.Tensor) -> str:
    """Render a ``(kd, kh, kw)`` clause patch slice by slice, side by side:
    ``#`` = must be on, ``.`` = must be off, blank = free."""
    glyph = {1: "#", -1: ".", 0: " "}
    kd, kh, _ = patch.shape
    header = "   ".join(f"d={i}".ljust(patch.shape[2] + 2) for i in range(kd))
    rows = ["  " + header]
    for r in range(kh):
        rows.append("  " + "   ".join("|" + "".join(glyph[int(v)] for v in patch[i, r]) + "|" for i in range(kd)))
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8, help="volume depth D")
    ap.add_argument("--size", type=int, default=12, help="volume height and width")
    ap.add_argument("--clauses", type=int, default=20, help="clauses per class")
    ap.add_argument("--T", type=float, default=15)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--position-encoding", action="store_true", help="add patch-coordinate literals")
    ap.add_argument("--baselines", action="store_true", help="also train depth-as-channels and flat TMs")
    ap.add_argument("--plot", action="store_true", help="save clause patches to shapes3d_clauses.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    size = (args.depth, args.size, args.size)
    x_train, y_train = tt.data.make_3d_shapes(2000, size=size, seed=0)
    x_test, y_test = tt.data.make_3d_shapes(1000, size=size, seed=1)
    print(f"volumes {tuple(x_train.shape)} -> 3x3x3 windows")
    common = dict(T=args.T, s=args.s, n_states=100)

    def fit(model, xtr, xte, verbose=True):
        tt.seed_everything(args.seed)
        trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size, verbose=verbose)
        trainer.fit((xtr, y_train), epochs=args.epochs, val_data=(xte, y_test) if verbose else None)
        return trainer.test((xte, y_test))["test_accuracy"]

    model = tt.Conv3dTsetlinMachine(
        2, args.clauses, patch_size=3, position_encoding=args.position_encoding, **common
    )
    model.class_names = ["spindle", "3D cross"]
    acc = fit(model, x_train, x_test)
    print(f"\n3D convolutional: {acc:.3f}  (position_encoding={args.position_encoding})")

    if args.baselines:
        dac = tt.ConvTsetlinMachine(
            2, args.clauses, patch_size=3, position_encoding=args.position_encoding, **common
        )
        dac_acc = fit(dac, x_train.flatten(1, 2), x_test.flatten(1, 2), verbose=False)
        print(f"depth as channels (2D, {args.depth} planes, same clause budget): {dac_acc:.3f}")
        flat = tt.TsetlinMachine(x_train[0].numel(), 2, args.clauses, **common)
        flat_acc = fit(flat, x_train.flatten(1), x_test.flatten(1), verbose=False)
        print(f"flat ({x_train[0].numel()} features, same clause budget): {flat_acc:.3f}")

    model.eval()
    score = tt.interpret.clause_precision(model, x_test, y_test)
    print("\nThe clause each class relies on most (one 3x3 slice per depth offset)")
    print("  # = voxel must be on, . = must be off, blank = don't care")
    for k, name in enumerate(model.class_names):
        own = (model.clause_class == k) & (model.clause_polarity > 0) & (score["support"] > 0.05)
        if not bool(own.any()):
            print(f"\nno clause of '{name}' fires on more than 5% of the test volumes")
            continue
        clause = int(torch.where(own, score["precision"], torch.zeros_like(score["precision"])).argmax())
        print(
            f"\nclause {clause} of '{name}': fires on {score['support'][clause]:.0%} of volumes, "
            f"{score['precision'][clause]:.0%} of them '{name}'  region={model.clause_region(clause)}"
        )
        print(ascii_volume(model.clause_patch(clause)[0]))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        fig = tt.viz.plot_conv_clauses(model, range(min(8, model.n_clauses_total)), ncols=4)
        fig.savefig("shapes3d_clauses.png", dpi=120)
        print("\nsaved shapes3d_clauses.png")


if __name__ == "__main__":
    main()
