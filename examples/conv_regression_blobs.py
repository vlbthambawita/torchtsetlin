"""Convolutional regression: read a *number* off an image (Abeyrathna et al., ICMLT 2021).

Every image holds one solid square of side 2-6 pixels at a random position and the target is
that side length. Nothing about the answer depends on where the square sits, so the model is
convolutional with ``position_encoding=False``: a clause that matches a ``k x k`` block of on
pixels surrounded by off pixels adds its integer weight to the vote sum wherever in the image
it finds one, and the vote sum is mapped linearly onto ``y_range``.

Run:  python examples/conv_regression_blobs.py
      python examples/conv_regression_blobs.py --clauses 800 --epochs 50
"""

import argparse

import torch

import torchtsetlin as tt

SIZE, MIN_SIDE, MAX_SIDE = 12, 2, 6


def make_squares(n: int, seed: int = 0):
    """``(n, 1, SIZE, SIZE)`` Boolean images with one square each, and its side length."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, 1, SIZE, SIZE, dtype=torch.bool)
    side = torch.randint(MIN_SIDE, MAX_SIDE + 1, (n,), generator=g)
    free = (SIZE - side + 1).float()                     # positions that fit a square of this side
    row = (torch.rand(n, generator=g) * free).long()
    col = (torch.rand(n, generator=g) * free).long()
    for i in range(n):
        k = int(side[i])
        x[i, 0, row[i] : row[i] + k, col[i] : col[i] + k] = True
    return x, side.float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", type=int, default=400)
    ap.add_argument("--T", type=float, default=400)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--patch", type=int, default=7, help="window side; must exceed the largest square")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tt.seed_everything(0)
    x_train, y_train = make_squares(args.train_size, seed=0)
    x_test, y_test = make_squares(1000, seed=1)

    model = tt.ConvRegressionTsetlinMachine(
        n_clauses=args.clauses,
        T=args.T,
        s=args.s,
        patch_size=args.patch,
        position_encoding=False,
        weighted=True,                      # integer clause weights: one clause can vote k times
        y_range=(float(MIN_SIDE), float(MAX_SIDE)),
        n_states=100,
    )
    trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size)
    trainer.fit((x_train, y_train), epochs=args.epochs, val_data=(x_test, y_test))
    print("\n" + str(trainer.test((x_test, y_test))))

    # Error per true side length — the ends of the range are the hard part.
    model.eval()
    pred = model.predict(x_test).cpu()
    print("\nside  n     mean prediction   MAE")
    for k in range(MIN_SIDE, MAX_SIDE + 1):
        m = y_test == k
        print(f"{k:4d}  {int(m.sum()):4d}  {pred[m].mean():15.2f}   {(pred[m] - k).abs().mean():.2f}")

    # A clause's patch is the block it looks for; its weight is how much it adds to the vote.
    step = (MAX_SIDE - MIN_SIDE) / args.T
    order = model.weights.argsort(descending=True)[:3]
    print(f"\nHighest-weighted clauses (each adds weight x {step:.4g} to the prediction)")
    for clause in order.tolist():
        block = model.clause_patch(clause)[0]
        on = int((block > 0).sum())
        print(f"\nclause {clause}: weight {int(model.weights[clause])}, "
              f"{on} pixels required on, {int((block < 0).sum())} required off")
        print("\n".join("  " + "".join({1: "#", -1: ".", 0: " "}[int(v)] for v in row) for row in block))


if __name__ == "__main__":
    main()
