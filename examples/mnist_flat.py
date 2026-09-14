"""MNIST with a flat (non-convolutional) Tsetlin machine.

Run:  python examples/mnist_flat.py --clauses 2000 --epochs 30
"""

import argparse

import torch

import torchtsetlin as tt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--clauses", type=int, default=500, help="clauses per class")
    ap.add_argument("--T", type=float, default=25)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tt.seed_everything(0)
    x_train, y_train = tt.data.load_mnist_boolean(args.root, train=True, threshold=0.3, device=args.device)
    x_test, y_test = tt.data.load_mnist_boolean(args.root, train=False, threshold=0.3, device=args.device)

    model = tt.TsetlinMachine(784, 10, n_clauses=args.clauses, T=args.T, s=args.s, weighted=args.weighted)
    trainer = tt.Trainer(
        model, device=args.device, batch_size=args.batch_size,
        callbacks=[tt.ModelCheckpoint("mnist_flat_best.pt"), tt.StateSummaryLogger()],
    )
    trainer.fit((x_train, y_train), epochs=args.epochs, val_data=(x_test, y_test))
    votes = trainer.predict((x_test, y_test), return_votes=True)
    print(tt.metrics.classification_report(votes, y_test))


if __name__ == "__main__":
    main()
