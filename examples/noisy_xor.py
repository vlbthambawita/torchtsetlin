"""Noisy XOR — the benchmark of the original Tsetlin machine paper.

Run:  python examples/noisy_xor.py [--sequential]
"""

import argparse

import torch

import torchtsetlin as tt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--sequential", action="store_true", help="exact one-example-at-a-time feedback")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tt.seed_everything(0)
    x_train, y_train = tt.data.make_noisy_xor(5000, noise=0.4, seed=0)
    x_test, y_test = tt.data.make_noisy_xor(5000, noise=0.0, seed=1)

    model = tt.TsetlinMachine(
        n_features=12, n_classes=2, n_clauses=20, T=15, s=3.9, n_states=50,
        feedback_mode="sequential" if args.sequential else "batch",
    )
    trainer = tt.Trainer(
        model, device=args.device, batch_size=args.batch_size,
        callbacks=[tt.EarlyStopping(monitor="val_accuracy", patience=30), tt.StateSummaryLogger()],
    )
    trainer.fit((x_train, y_train), epochs=args.epochs, val_data=(x_test, y_test))
    print(trainer.test((x_test, y_test)))
    print("\nLearned rules:")
    for rule in model.rules():
        print(" ", rule)
    print("\nExplanation of the first test example:")
    print(tt.interpret.explain(model, x_test[0]))


if __name__ == "__main__":
    main()
