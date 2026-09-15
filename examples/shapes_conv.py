"""Translation invariance with a convolutional Tsetlin machine (toy shapes).

A hollow 3x3 circle (class 0) or a 3x3 cross (class 1) sits at a random position in a
12x12 image. A flat Tsetlin machine has to memorise the shape at all 100 positions; a
convolutional one learns a single 3x3 clause that matches anywhere. With the same clause
budget (``--flat`` runs both) the convolutional model lands on 1.000 every time, while the
flat one scores anywhere between 0.55 and 0.91 depending on ``--seed``.

The ``--position-encoding`` flag switches on the thermometer-encoded patch coordinates. On a
task that is fully translation invariant they only make the clauses more specific, and the
accuracy drops — see ``docs/concepts/convolution.md``.

Run:  python examples/shapes_conv.py --flat
      python examples/shapes_conv.py --position-encoding    # 1.00 -> ~0.5
      python examples/shapes_conv.py --size 16 --seed 2 --plot
"""

import argparse

import torch

import torchtsetlin as tt


def ascii_patch(patch: torch.Tensor) -> str:
    """Render a ``(kh, kw)`` clause patch: ``#`` = must be on, ``.`` = must be off, ` ` = free."""
    glyph = {1: "#", -1: ".", 0: " "}
    return "\n".join("|" + "".join(glyph[int(v)] for v in row) + "|" for row in patch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=12, help="image side length")
    ap.add_argument("--clauses", type=int, default=40, help="clauses per class")
    ap.add_argument("--T", type=float, default=15)
    ap.add_argument("--s", type=float, default=3.9)
    ap.add_argument("--patch", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--position-encoding", action="store_true", help="add patch-coordinate literals")
    ap.add_argument("--flat", action="store_true", help="also train a flat TM on the same images")
    ap.add_argument("--plot", action="store_true", help="save clause patches to shapes_clauses.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tt.seed_everything(args.seed)
    x_train, y_train = tt.data.make_shapes(2000, size=args.size, seed=0)
    x_test, y_test = tt.data.make_shapes(1000, size=args.size, seed=1)
    print(f"images {tuple(x_train.shape)} -> patches of {args.patch}x{args.patch}")

    model = tt.ConvTsetlinMachine(
        2,
        n_clauses=args.clauses,
        T=args.T,
        s=args.s,
        patch_size=args.patch,
        stride=args.stride,
        position_encoding=args.position_encoding,
        n_states=100,
    )
    model.class_names = ["circle", "cross"]
    trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size)
    trainer.fit((x_train, y_train), epochs=args.epochs, val_data=(x_test, y_test))
    conv_acc = trainer.test((x_test, y_test))["test_accuracy"]
    print(f"\nconvolutional: {conv_acc:.3f}  (position_encoding={args.position_encoding})")
    if args.position_encoding:
        print("  position literals pin clauses to patch coordinates, and the shape here can sit\n"
              "  anywhere — run without the flag to see the difference")

    if args.flat:
        tt.seed_everything(args.seed)
        flat = tt.TsetlinMachine(args.size**2, 2, n_clauses=args.clauses, T=args.T, s=args.s, n_states=100)
        flat_trainer = tt.Trainer(flat, device=args.device, batch_size=args.batch_size, verbose=False)
        flat_trainer.fit((x_train.flatten(1), y_train), epochs=args.epochs)
        flat_acc = flat_trainer.test((x_test.flatten(1), y_test))["test_accuracy"]
        print(f"flat ({args.size**2} features, same clause budget): {flat_acc:.3f}")

    # Which clause is the model actually relying on? Score every clause by how often it
    # fires and how often it is right when it does, then show the winner per class.
    model.eval()
    score = tt.interpret.clause_precision(model, x_test, y_test)
    print("\nThe clause each class relies on most")
    print("  # = pixel must be on, . = must be off, blank = don't care")
    for k, name in enumerate(model.class_names):
        # Positive clauses of this class that fire often enough to matter, best precision first.
        own = (model.clause_class == k) & (model.clause_polarity > 0) & (score["support"] > 0.05)
        if not bool(own.any()):
            print(f"\nno clause of '{name}' fires on more than 5% of the test images")
            continue
        clause = int(torch.where(own, score["precision"], torch.zeros_like(score["precision"])).argmax())
        region = model.clause_region(clause)
        print(
            f"\nclause {clause} of '{name}': fires on {score['support'][clause]:.0%} of images, "
            f"{score['precision'][clause]:.0%} of them class '{name}'"
        )
        print(f"  allowed window rows y={region['y']}, columns x={region['x']}")
        print(ascii_patch(model.clause_patch(clause)[0]))
        print("  " + model.clause_expression(clause).replace(" AND ", "\n  AND "))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        fig = tt.viz.plot_conv_clauses(model, range(min(16, model.n_clauses_total)))
        fig.savefig("shapes_clauses.png", dpi=120)
        print("\nsaved shapes_clauses.png")


if __name__ == "__main__":
    main()
