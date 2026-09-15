"""1-D convolution on a signal: find a motif anywhere in a noisy time series.

Each example is a 48-sample signal of uniform noise with one 8-sample ramp hidden in it —
rising (class 1) or falling (class 0) — at a random offset. The signal is thermometer
encoded, which turns every timestep into ``n_bits`` Boolean channels, so the input is
``(B, n_bits, L)`` — exactly the layout ``Conv1dTsetlinMachine`` expects. A clause then reads
a ``kernel_size``-wide window across all bit planes and fires if *any* window matches, which
is what makes the detector shift invariant.

Run:  python examples/conv1d_ramps.py
      python examples/conv1d_ramps.py --bits 12 --kernel 10 --position-encoding
"""

import argparse

import torch

import torchtsetlin as tt

LENGTH, MOTIF = 48, 8


def make_ramps(n: int, seed: int = 0):
    """``(n, LENGTH)`` float signals in ``[0, 1]`` and labels (1 = rising ramp)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, LENGTH, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    ramp = torch.linspace(0.05, 0.95, MOTIF)
    pos = torch.randint(0, LENGTH - MOTIF + 1, (n,), generator=g)
    for i in range(n):
        x[i, pos[i] : pos[i] + MOTIF] = ramp if y[i] == 1 else ramp.flip(0)
    return x, y.long()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8, help="thermometer bits = Boolean channels")
    ap.add_argument("--kernel", type=int, default=8, help="window width in timesteps")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--clauses", type=int, default=50, help="clauses per class")
    ap.add_argument("--T", type=float, default=30)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--position-encoding", action="store_true", help="pin clauses to offsets")
    ap.add_argument("--plot", action="store_true", help="save the learning curves to ramps_history.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tt.seed_everything(0)
    x_train, y_train = make_ramps(3000, seed=0)
    x_test, y_test = make_ramps(1000, seed=1)

    # (N, L) floats -> (N, L, bits) -> (N, bits, L): bit planes become the channel dimension.
    enc = tt.data.ThermometerEncoder(
        n_bits=args.bits, strategy="uniform", value_range=(0.0, 1.0), flatten=False
    )
    b_train = enc(x_train).permute(0, 2, 1).contiguous()
    b_test = enc(x_test).permute(0, 2, 1).contiguous()
    print(f"signals {tuple(x_train.shape)} -> Boolean {tuple(b_train.shape)} (B, Z, L)")

    model = tt.Conv1dTsetlinMachine(
        2,
        n_clauses=args.clauses,
        T=args.T,
        s=args.s,
        kernel_size=args.kernel,
        stride=args.stride,
        position_encoding=args.position_encoding,
        n_states=100,
    )
    model.class_names = ["falling", "rising"]
    trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size)
    trainer.fit((b_train, y_train), epochs=args.epochs, val_data=(b_test, y_test))
    print("\n" + str(trainer.test((b_test, y_test))))
    print(f"{args.kernel * args.bits} signal features per window, "
          f"{model.n_features} with position bits, {model.n_literals} literals")

    # Each clause is a (bits, 1, kernel) pattern: read a column as "level at this timestep".
    model.eval()
    score = tt.interpret.clause_precision(model, b_test, y_test)
    print("\nThe clause each class relies on most (rows = thermometer bits, high bit on top)")
    print("  # = bit must be 1, . = must be 0, blank = don't care")
    for k, name in enumerate(model.class_names):
        own = (model.clause_class == k) & (model.clause_polarity > 0) & (score["support"] > 0.05)
        if not bool(own.any()):
            print(f"\nno clause of '{name}' fires on more than 5% of the test signals")
            continue
        clause = int(torch.where(own, score["precision"], torch.zeros_like(score["precision"])).argmax())
        patch = model.clause_patch(clause)[:, 0]  # (bits, kernel)
        print(f"\nclause {clause} of '{name}': fires on {score['support'][clause]:.0%} of signals, "
              f"{score['precision'][clause]:.0%} of them class '{name}'")
        for bit in range(args.bits - 1, -1, -1):
            print("  " + "".join({1: "#", -1: ".", 0: " "}[int(v)] for v in patch[bit]))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        fig = tt.viz.plot_history(trainer.history).figure
        fig.savefig("ramps_history.png", dpi=120)
        print("\nsaved ramps_history.png")


if __name__ == "__main__":
    main()
