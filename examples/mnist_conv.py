"""MNIST with a convolutional Tsetlin machine (10x10 windows, adaptive thresholding).

Run:  python examples/mnist_conv.py --clauses 2000 --T 2500 --epochs 30
"""

import argparse

import torch

import torchtsetlin as tt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--clauses", type=int, default=200, help="clauses per class")
    ap.add_argument("--T", type=float, default=100)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--patch", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot", action="store_true", help="save clause patches to mnist_clauses.png")
    args = ap.parse_args()

    tt.seed_everything(0)
    enc = tt.data.AdaptiveThresholdEncoder(block_size=11, C=2, method="gaussian")
    booleanize = lambda x: enc(x * 255.0)  # noqa: E731
    x_train, y_train = tt.data.load_mnist_boolean(args.root, train=True, encoder=booleanize, device=args.device)
    x_test, y_test = tt.data.load_mnist_boolean(args.root, train=False, encoder=booleanize, device=args.device)

    model = tt.ConvTsetlinMachine(
        10, n_clauses=args.clauses, T=args.T, s=args.s, patch_size=args.patch, weighted=True,
    )
    trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size, callbacks=[tt.StateSummaryLogger()])
    trainer.fit((x_train, y_train), epochs=args.epochs, val_data=(x_test, y_test))
    print(trainer.test((x_test, y_test)))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        fig = tt.viz.plot_conv_clauses(model, range(32))
        fig.savefig("mnist_clauses.png", dpi=120)
        print("saved mnist_clauses.png")


if __name__ == "__main__":
    main()
