"""Graph Tsetlin machine on sequences: how many consecutive A's does a sequence hold?

The reference implementation's ``SequenceClassificationDemo`` (cair/GraphTsetlinMachine):
sequences of 3-10 positions with a single run of 1, 2 or 3 consecutive ``A``s, and the class
is the run length. Each position is a node; node ``k`` sends to ``k - 1`` over a "left" edge
and to ``k + 1`` over a "right" edge.

Two layers are enough. A node sees its own symbol and which clauses matched at its two
neighbours, so one clause can say "an A with an A on both sides" (three A's), another "an A with
an A on its left" (at least two), another "an A with no A beside it" (one); the signed class
weights combine them. With ``--depth 1`` the machine is a convolutional machine with a
one-position window and stays at chance.

The literal budget matters here. Early in training every clause matches everywhere, so a
boosted Type Ia event includes every message literal at once; without ``--budget`` the clauses
can collapse into "no A here, and everything matched around me", true in every sequence.

Run:  python examples/graph_sequence.py [--depth 2] [--sequential]
"""

import argparse
import re
import time

import torch

import torchtsetlin as tt
from torchtsetlin.data import GraphBatch, SymbolTable


def make_sequences(n: int, n_classes: int = 3, max_len: int = 10, noise: float = 0.0, seed: int = 0):
    """``n`` sequences of length ``n_classes..max_len`` holding one run of ``y + 1`` A's."""
    g = torch.Generator().manual_seed(seed)
    seqs, ys = [], []
    for _ in range(n):
        L = int(torch.randint(n_classes, max_len + 1, (1,), generator=g))
        y = int(torch.randint(0, n_classes, (1,), generator=g))
        end = int(torch.randint(y, L, (1,), generator=g))
        s = [[] for _ in range(L)]  # positions without a symbol
        for k in range(end - y, end + 1):
            s[k] = ["A"]
        if float(torch.rand(1, generator=g)) < noise:  # label noise
            y = int((y + torch.randint(1, n_classes, (1,), generator=g)) % n_classes)
        seqs.append(s)
        ys.append(y)
    return seqs, torch.tensor(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--clauses", type=int, default=40)
    ap.add_argument("--T", type=float, default=200)
    ap.add_argument("--s", type=float, default=3.0)
    ap.add_argument("--budget", type=int, default=8, help="max_included_literals (0 = none)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--message-encoding", default="one_hot", choices=["one_hot", "random", "double_hashing"])
    ap.add_argument("--sequential", action="store_true", help="one graph at a time, as the reference code")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tt.seed_everything(args.seed)
    symbols = SymbolTable(["A"], encoding="one_hot")
    x_train, y_train = make_sequences(args.n_train, noise=0.01, seed=0)
    x_test, y_test = make_sequences(2000, seed=1)
    g_train = GraphBatch.from_sequences(x_train, symbols)
    g_test = GraphBatch.from_sequences(x_test, symbols)

    model = tt.GraphTsetlinMachine(
        n_features=symbols.dim, n_outputs=3, n_clauses=args.clauses, T=args.T, s=args.s,
        depth=args.depth, message_encoding=args.message_encoding, message_size=256,
        n_edge_types=2, edge_type_names=["left", "right"], max_included_literals=args.budget or None,
        feedback_mode="sequential" if args.sequential else "batch",
    )
    trainer = tt.Trainer(model, device=args.device, batch_size=args.batch_size)
    t0 = time.perf_counter()
    trainer.fit((g_train, y_train), epochs=args.epochs, val_data=(g_test, y_test))
    print(f"trained in {time.perf_counter() - t0:.1f} s")
    print(trainer.test((g_test, y_test)))

    # The strongest clauses. A message literal such as right1:7 reads "clause 7's first
    # component (here: its node literals) matched at the node a right-going edge came from",
    # i.e. at the left neighbour.
    print("\nStrongest clauses (class weights for one, two, three A's):")
    W = model.weights
    order = W.abs().max(dim=1).values.argsort(descending=True)[:8].tolist()
    cited = set()
    for j in order:
        expr = model.clause_expression(j, symbols=symbols)
        cited |= {int(k) for k in re.findall(r"(?:left|right)1:(\d+)", expr)}
        print(f"  clause {j:2d}  {str(W[j].tolist()):18s} {expr}")
    print("\nWhat the cited messages vouch for (the node part of each clause they name):")
    for k in sorted(cited):
        print(f"  clause {k:2d}: {model.clause_expression(k, symbols=symbols, layers=[0])}")


if __name__ == "__main__":
    main()
