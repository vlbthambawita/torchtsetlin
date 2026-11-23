"""The benchmark's data, generated exactly as the reference script does.

``generate`` follows the data block of ``examples/CompositionalGeneralizationBenchmark.py`` of
cair/PyHierarchicalTsetlinMachineCUDA line by line (same calls on NumPy's global random state in
the same order), so that with the same ``np.random.seed`` it yields the arrays the script would
write. ``checks.py`` (M0) executes the original block and compares. The script itself does not
seed NumPy; here the data seed is the run seed.

The generating code is derived from that script:

    Copyright (c) 2026 Ole-Christoffer Granmo and the University of Agder

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Usage: python code/data.py [--seeds 0 1 2]        # writes .cache/data/seed<k>.npz, prints statistics
"""

import argparse
import hashlib
import itertools
import math

import numpy as np
from common import CACHE, DATA, ISSUE, RESULTS, env_seeds, read_json, write_json
from scipy.stats import zipfian


def count_inversions(permutation):
    """Counts the number of inversions to determine parity."""
    inv = 0
    n = len(permutation)
    for i in range(n):
        for j in range(i + 1, n):
            if permutation[i] > permutation[j]:
                inv += 1
    return inv


def generate(seed, sequence_length=5, number_of_synonyms=5, number_of_examples=40000, noise=0.01):
    """(X_train, Y_train, X_test, Y_test): one-hot position x concept x synonym, not negated.
    The reference command adds ``--negate-input``; here the model option ``literals="negated"``
    plays that role, so the stored input is the plain one-hot."""
    np.random.seed(seed)
    L, S, n = sequence_length, number_of_synonyms, number_of_examples
    Y_unique = np.zeros(math.factorial(L), dtype=np.uint32)
    permutations = list(itertools.permutations(range(L)))
    for i in range(math.factorial(L)):
        Y_unique[i] = count_inversions(list(permutations[i])) % 2

    X_train = np.zeros((n, L * L * S), dtype=np.uint32)
    Y_train = np.empty(n, dtype=np.uint32)
    train_indices = np.random.choice(math.factorial(L), size=n, replace=True)
    for i in range(n):
        for j in range(L):
            X_train[i, j * L * S + permutations[train_indices[i]][j] * S
                    + zipfian.rvs(a=0.0, n=S, loc=-1, size=1)[0]] = 1
        Y_train[i] = Y_unique[train_indices[i]]
    Y_train = np.where(np.random.rand(n) <= noise, 1 - Y_train, Y_train)  # Adds noise

    X_test = np.zeros((n, L * L * S), dtype=np.uint32)
    Y_test = np.empty(n, dtype=np.uint32)
    test_indices = np.random.choice(math.factorial(L), size=n, replace=True)
    for i in range(n):
        for j in range(L):
            X_test[i, j * L * S + permutations[test_indices[i]][j] * S + zipfian.rvs(0.0, S, loc=-1, size=1)[0]] = 1
        Y_test[i] = Y_unique[test_indices[i]]
    return X_train, Y_train, X_test, Y_test


def pattern_stats(x_train, x_test, sequence_length, number_of_synonyms):
    """The "unique patterns" block of the reference script."""
    tr = {r.tobytes() for r in np.unique(x_train, axis=0)}
    te = {r.tobytes() for r in np.unique(x_test, axis=0)}
    return {"unique_total": math.factorial(sequence_length) * number_of_synonyms ** sequence_length,
            "unique_train": len(tr), "unique_test": len(te), "unique_both": len(tr & te),
            "unseen_test": len(te - tr)}


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load(seed):
    """uint8 X (N, 125) and int64 Y of training and test set; generated once and cached. The
    record results/data/seed<k>.json holds the hash and the pattern statistics."""
    f = CACHE / "data" / f"seed{seed}.npz"
    meta = RESULTS / "data" / f"seed{seed}.json"
    if not f.exists():
        xtr, ytr, xte, yte = generate(seed, **DATA)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, x_train=xtr.astype(np.uint8), y_train=ytr.astype(np.int64),
                            x_test=xte.astype(np.uint8), y_test=yte.astype(np.int64))
        tmp.replace(f)
    d = dict(np.load(f))
    digest = sha(d["x_train"], d["y_train"], d["x_test"], d["y_test"])
    old = read_json(meta)
    if old is None:
        stats = pattern_stats(d["x_train"], d["x_test"], DATA["sequence_length"], DATA["number_of_synonyms"])
        write_json({"seed": seed, "sha256": digest, "data": DATA, "stats": stats, "issue": ISSUE,
                    "label_1_share": {"train": float(d["y_train"].mean()), "test": float(d["y_test"].mean())},
                    "generator": "code/data.py generate(), np.random.seed(seed)"}, meta)
    elif old["sha256"] != digest:
        raise RuntimeError(f"data of seed {seed} differs from results/data/seed{seed}.json "
                           f"(another NumPy/SciPy version?): delete one of them deliberately")
    return d, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    for s in args.seeds if args.seeds is not None else env_seeds():
        _, digest = load(s)
        st = read_json(RESULTS / "data" / f"seed{s}.json")["stats"]
        print(f"seed {s}: sha256 {digest[:12]}  unique train {st['unique_train']} test {st['unique_test']} "
              f"both {st['unique_both']} unseen test {st['unseen_test']}  "
              f"(issue: {ISSUE['unique_train']} / {ISSUE['unique_test']} / {ISSUE['unique_both']} / {ISSUE['unseen_test']})")


if __name__ == "__main__":
    main()
