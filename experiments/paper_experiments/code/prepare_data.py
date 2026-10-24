"""Phase 1: Booleanize every benchmark dataset once (PLAN.md rule P5).

    python code/prepare_data.py [B1 B2 ...]      # default: all

Writes .cache/data/<bench>.npz (uint8 bits, int64 labels or float64 targets) and
results/data_manifest.json (shapes, bit density, sha256). Each pipeline follows the upstream
demo script named in its docstring line for line; only the loader is replaced by a direct
download of the same file Keras / scikit-learn would fetch (no TensorFlow needed).
"""
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / ".cache" / "data"
RAW = DATA / "raw"
UP = HERE / ".cache" / "upstream" / "pyTsetlinMachine"
KERAS = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"


def fetch(url, name):
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / name
    if not p.exists():
        print(f"  download {url}")
        urllib.request.urlretrieve(url, p)
    return p


def keras_mnist():
    with np.load(fetch(KERAS + "mnist.npz", "mnist.npz")) as f:
        return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])


def keras_fashion_mnist():
    def idx(name, images):
        with gzip.open(fetch(KERAS + name, "fashion-" + name), "rb") as f:
            buf = f.read()
        return (np.frombuffer(buf, np.uint8, offset=16).reshape(-1, 28, 28) if images
                else np.frombuffer(buf, np.uint8, offset=8))
    return ((idx("train-images-idx3-ubyte.gz", 1), idx("train-labels-idx1-ubyte.gz", 0)),
            (idx("t10k-images-idx3-ubyte.gz", 1), idx("t10k-labels-idx1-ubyte.gz", 0)))


def keras_imdb(num_words, index_from, start_char=1, oov_char=2, seed=113):
    """keras.datasets.imdb.load_data(num_words=..., index_from=...) re-implemented."""
    with np.load(fetch(KERAS + "imdb.npz", "imdb.npz"), allow_pickle=True) as f:
        x_train, y_train, x_test, y_test = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
    rng = np.random.RandomState(seed)
    i = np.arange(len(x_train)); rng.shuffle(i); x_train, y_train = x_train[i], y_train[i]
    i = np.arange(len(x_test)); rng.shuffle(i); x_test, y_test = x_test[i], y_test[i]
    x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
    x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
    xs = [[w if w < num_words else oov_char for w in x] for x in x_train + x_test]
    n = len(x_train)
    word_index = json.loads(fetch(KERAS + "imdb_word_index.json", "imdb_word_index.json").read_text())
    return (xs[:n], y_train), (xs[n:], y_test), word_index


def b1():
    """pyTsetlinMachine examples/NoisyXORDemo.py"""
    tr, te = np.loadtxt(UP / "NoisyXORTrainingData.txt"), np.loadtxt(UP / "NoisyXORTestData.txt")
    return dict(x_train=tr[:, :-1], y_train=tr[:, -1], x_test=te[:, :-1], y_test=te[:, -1])


def b2():
    """pyTsetlinMachine examples/2DNoisyXORDemo.py"""
    tr, te = np.loadtxt(UP / "2DNoisyXORTrainingData.txt"), np.loadtxt(UP / "2DNoisyXORTestData.txt")
    return dict(x_train=tr[:, :-1].reshape(-1, 4, 4), y_train=tr[:, -1],
                x_test=te[:, :-1].reshape(-1, 4, 4), y_test=te[:, -1])


def _binarized_splits(X, Y, n_splits, test_size):
    from sklearn.model_selection import train_test_split
    from vendor.pytm_binarizer import Binarizer
    b = Binarizer(max_bits_per_feature=10)
    b.fit(X)                     # the demos fit on all of X, before splitting
    Xb = b.transform(X)
    idx = np.arange(len(Y))
    tr = []
    for k in range(n_splits):    # the demos draw unseeded splits; split k uses random_state=k
        a, _ = train_test_split(idx, test_size=test_size, random_state=k)
        m = np.zeros(len(Y), bool); m[a] = True
        tr.append(m)
    return dict(x=Xb, y=Y, train_mask=np.stack(tr))


def b3():
    """pyTsetlinMachine examples/BreastCancerDemo.py (100 splits, test_size=0.2)"""
    from sklearn import datasets
    d = datasets.load_breast_cancer()
    return _binarized_splits(d.data, d.target, 100, 0.2)


def b4():
    """pyTsetlinMachine examples/MNISTDemo.py and MNISTDemoWeightedClauses.py (B4, B5)"""
    (xtr, ytr), (xte, yte) = keras_mnist()
    return dict(x_train=np.where(xtr.reshape(-1, 784) > 75, 1, 0), y_train=ytr,
                x_test=np.where(xte.reshape(-1, 784) > 75, 1, 0), y_test=yte)


def b6():
    """green_tsetlin performance_tests/tmu_mnist_dense.py get_mnist()"""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
                        data_home=str(RAW / "openml"))
    xtr, xte, ytr, yte = train_test_split(X, y, random_state=42, test_size=10_000)
    return dict(x_train=np.where(xtr.reshape(-1, 784) > 75, 1, 0), y_train=ytr.astype(np.uint32),
                x_test=np.where(xte.reshape(-1, 784) > 75, 1, 0), y_test=yte.astype(np.uint32))


def b7():
    """pyTsetlinMachine examples/MNISTDemo2DConvolutionWeightedClauses.py"""
    (xtr, ytr), (xte, yte) = keras_mnist()
    return dict(x_train=np.where(xtr >= 75, 1, 0), y_train=ytr,
                x_test=np.where(xte >= 75, 1, 0), y_test=yte)


def b8():
    """pyTsetlinMachine examples/FashionMNISTDemo2DConvolutionWeightedClauses.py"""
    import cv2
    (xtr, ytr), (xte, yte) = keras_fashion_mnist()
    out = {}
    for name, x in (("x_train", xtr), ("x_test", xte)):
        x = np.copy(x)
        for i in range(x.shape[0]):
            x[i, :] = cv2.adaptiveThreshold(x[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        out[name] = x
    return dict(x_train=out["x_train"], y_train=ytr, x_test=out["x_test"], y_test=yte)


def b9():
    """pyTsetlinMachine examples/IMDbTextCategorizationDemo.py (MAX_NGRAM 2, 5000 words, chi2 5000)"""
    from sklearn.feature_selection import SelectKBest, chi2
    MAX_NGRAM, NUM_WORDS, INDEX_FROM, FEATURES = 2, 5000, 2, 5000
    (train_x, train_y), (test_x, test_y), word_to_id = keras_imdb(NUM_WORDS, INDEX_FROM)
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {v: k for k, v in word_to_id.items()}

    def phrases(doc):
        terms = [id_to_word[w] for w in doc]
        for N in range(1, MAX_NGRAM + 1):
            for j in range(len(terms) - N + 1):
                yield " ".join(terms[j:j + N])

    vocabulary = {}
    for doc in train_x:
        for p in phrases(doc):
            vocabulary[p] = vocabulary.get(p, 0) + 1
    bit = {}
    for p, c in vocabulary.items():          # dict order = first occurrence, as in the demo
        if c >= 10:
            bit[p] = len(bit)

    def encode(docs):
        X = np.zeros((len(docs), len(bit)), dtype=np.uint8)
        for i, doc in enumerate(docs):
            for p in phrases(doc):
                j = bit.get(p)
                if j is not None:
                    X[i, j] = 1
        return X

    X_train, X_test = encode(train_x), encode(test_x)
    skb = SelectKBest(chi2, k=FEATURES).fit(X_train, train_y)
    return dict(x_train=skb.transform(X_train), y_train=train_y,
                x_test=skb.transform(X_test), y_test=test_y,
                n_phrases=np.array(len(bit)))


def b10():
    """pyTsetlinMachine examples/RegressionDemo.py (25 splits, default test_size 0.25)"""
    from sklearn import datasets
    d = datasets.fetch_california_housing(data_home=str(RAW / "sklearn"))
    return _binarized_splits(d.data, d.target, 25, 0.25)


BUILDERS = {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B6": b6, "B7": b7, "B8": b8, "B9": b9, "B10": b10}
# B5 uses B4's data file.


def main():
    names = sys.argv[1:] or list(BUILDERS)
    DATA.mkdir(parents=True, exist_ok=True)
    man_path = HERE / "results" / "data_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for name in names:
        print(name, BUILDERS[name].__doc__.strip())
        d = BUILDERS[name]()
        out = {}
        for k, v in d.items():
            v = np.asarray(v)
            if k.startswith("x"):
                assert set(np.unique(v)) <= {0, 1}, (name, k)
                v = v.astype(np.uint8)
            elif k.startswith("y") and name != "B10":
                v = v.astype(np.int64)
            out[k] = v
        path = DATA / f"{name}.npz"
        np.savez_compressed(path, **out)
        manifest[name] = {
            "source": BUILDERS[name].__doc__.strip(),
            "file": str(path.relative_to(HERE)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                           **({"bit_density": round(float(v.mean()), 4)} if k.startswith("x") else {})}
                       for k, v in out.items()},
        }
        print("  ", {k: v.shape for k, v in out.items()})
        man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
