"""MNIST with a convolutional Tsetlin machine (10x10 windows, adaptive thresholding).

Clauses look at one window at a time and fire if *any* window of the image matches, so a
stroke pattern is learned once instead of once per position. ``--coalesced`` swaps the
per-class clause pool for a single shared pool with per-class weights
(:class:`~torchtsetlin.models.ConvCoalescedTsetlinMachine`), which usually needs far fewer
clauses in total.

Run:  python examples/mnist_conv.py --clauses 2000 --T 2500 --epochs 30
      python examples/mnist_conv.py --coalesced --clauses 1000 --T 500     # 0.985 in 10 epochs
      python examples/mnist_conv.py --stride 2 --no-position-encoding
"""

import argparse

import torch

import torchtsetlin as tt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--clauses", type=int, default=200, help="clauses per class (total if --coalesced)")
    ap.add_argument("--T", type=float, default=100)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--patch", type=int, default=10, help="convolution window side")
    ap.add_argument("--stride", type=int, default=1, help="window stride (2 = a quarter of the patches)")
    ap.add_argument("--no-position-encoding", dest="position_encoding", action="store_false",
                    help="drop the patch-coordinate literals (fully translation invariant)")
    ap.add_argument("--coalesced", action="store_true", help="one shared clause pool with class weights")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=None, help="subsample the training set")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot", action="store_true", help="save clause patches to mnist_clauses.png")
    args = ap.parse_args()

    tt.seed_everything(0)
    # The encoder lives on the compute device so the images never travel back to the host.
    enc = tt.data.AdaptiveThresholdEncoder(block_size=11, C=2, method="gaussian").to(args.device)

    def booleanize(x: torch.Tensor) -> torch.Tensor:
        return enc(x * 255.0)

    x_train, y_train = tt.data.load_mnist_boolean(
        args.root, train=True, encoder=booleanize, device=args.device, max_samples=args.max_samples
    )
    x_test, y_test = tt.data.load_mnist_boolean(args.root, train=False, encoder=booleanize, device=args.device)

    common = dict(
        n_clauses=args.clauses,
        T=args.T,
        s=args.s,
        patch_size=args.patch,
        stride=args.stride,
        position_encoding=args.position_encoding,
        input_shape=tuple(x_train.shape[1:]),
    )
    model = (
        # Coalesced clauses carry one integer weight per class already, hence no `weighted`.
        tt.ConvCoalescedTsetlinMachine(10, **common)
        if args.coalesced
        else tt.ConvTsetlinMachine(10, weighted=True, **common)
    )
    h, w = x_train.shape[2], x_train.shape[3]
    py, px = (h - args.patch) // args.stride + 1, (w - args.patch) // args.stride + 1
    print(f"{type(model).__name__}: {py}x{px} = {py * px} windows per image, "
          f"{model.n_features} features per window ({model.n_literals} literals), "
          f"{model.n_clauses_total} clauses in total")

    trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size, callbacks=[tt.StateSummaryLogger()])
    trainer.fit((x_train, y_train), epochs=args.epochs, val_data=(x_test, y_test))
    votes = trainer.predict((x_test, y_test), return_votes=True)
    print(tt.metrics.classification_report(votes, y_test))

    # What a clause looks for, and where it is allowed to look. Clause weights say how much
    # a clause counts towards a class, so the heaviest ones are the ones worth reading.
    weight = model.weights.abs()
    if weight.dim() > 1:  # coalesced: one weight per (clause, class)
        weight = weight.max(dim=1).values
    print("Heaviest clauses (rows = patch pixels, # = on, . = off, blank = don't care)")
    for clause in weight.argsort(descending=True)[:3].tolist():
        region = model.clause_region(clause)
        patch = model.clause_patch(clause)[0]
        print(f"\nclause {clause}: weight {int(weight[clause])}, "
              f"{int(model.include_count[clause])} literals, "
              f"windows y={region['y']} x={region['x']}")
        print("\n".join("  " + "".join({1: "#", -1: ".", 0: " "}[int(v)] for v in row) for row in patch))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        fig = tt.viz.plot_conv_clauses(model, range(min(32, model.n_clauses_total)))
        fig.savefig("mnist_clauses.png", dpi=120)
        print("\nsaved mnist_clauses.png")


if __name__ == "__main__":
    main()
