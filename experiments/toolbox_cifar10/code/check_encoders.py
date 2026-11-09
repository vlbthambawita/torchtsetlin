"""Check every encoder in ``encoders.py`` bit for bit against the authors' code on the first
N training and test images (and their flips). The reference side is the scripts' own loops, copied
verbatim from ``reference_code/`` (their encoding happens inline in ``__main__``), and the authors'
``mlt_temp`` imported from ``reference_code/MultilevelThresholdingThermometers.py``.

Writes results/encoder_check.json.  Usage: python code/check_encoders.py [--n 500]
"""

import argparse
import sys
import warnings

import cv2
import numpy as np

import encoders as E
from common import PROG, RESULTS, load_cifar10, new_record, write_json

sys.path.insert(0, str(PROG / "reference_code"))


def ref_canny(X_train_org):                      # CIFAR10Canny.py
    X_train = np.copy(X_train_org)
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
            blur = cv2.GaussianBlur(X_train_org[i, :, :, j], (3, 3), 0)
            X_train[i, :, :, j] = np.where(cv2.Canny(blur, 100, 200) != 0, 1, 0)
    return X_train


def ref_adaptive(X_train_org, method):           # CIFAR10AdaptiveThresholding{Mean,Gaussian}.py
    X_train = np.copy(X_train_org)
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
            X_train[i, :, :, j] = cv2.adaptiveThreshold(
                X_train_org[i, :, :, j], 1, method, cv2.THRESH_BINARY, 11, 2)
    return X_train


def ref_otsu(X_train_org):                       # CIFAR10OtsuThresholding.py
    X_train = np.copy(X_train_org)
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
            ret, X_train[i, :, :, j] = cv2.threshold(
                X_train_org[i, :, :, j], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return X_train


def ref_therm(X_train_org, resolution=8):        # CIFAR105x5ColorThermometerScoring.py
    X_train = np.empty(X_train_org.shape + (resolution,), dtype=np.uint8)
    for z in range(resolution):
        X_train[:, :, :, :, z] = X_train_org[:, :, :, :] >= (z + 1) * 255 / (resolution + 1)
    return X_train.reshape(X_train_org.shape[:3] + (3 * resolution,))


def ref_hog(X_train_org):                        # CIFAR10HistogramOfGradients.py
    hog = cv2.HOGDescriptor((32, 32), (12, 12), (4, 4), (4, 4), 18, 1, -1.0, 0, 0.2, True, 64, True)
    fd = hog.compute(X_train_org[0])
    X_train = np.empty((X_train_org.shape[0], fd.shape[0]), dtype=np.uint32)
    for i in range(X_train_org.shape[0]):
        fd = hog.compute(X_train_org[i])
        X_train[i] = fd >= 0.1
    return X_train


def ref_augment(X_train_org, Y_train):           # the Augmented* scripts
    imgs, labels = [], []
    for i in range(len(X_train_org)):
        imgs.append(X_train_org[i])
        labels.append(Y_train[i])
        imgs.append(cv2.flip(X_train_org[i], 1))
        labels.append(Y_train[i])
    return np.array(imgs), np.array(labels)


def channel_first(a):
    return a.transpose(0, 3, 1, 2).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    xtr, ytr, xte, _ = load_cifar10()
    x = np.concatenate([xtr[:args.n], xte[:args.n]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from MultilevelThresholdingThermometers import mlt_temp
        ref = {
            "canny": channel_first(ref_canny(x)),
            "agauss": channel_first(ref_adaptive(x, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)),
            "amean": channel_first(ref_adaptive(x, cv2.ADAPTIVE_THRESH_MEAN_C)),
            "otsu": channel_first(ref_otsu(x)),
            "therm": channel_first(ref_therm(x)),
            "atherm": channel_first(mlt_temp(x)),
            "hog": ref_hog(x).astype(bool),
        }
    out = {}
    for name, r in ref.items():
        ours = E.ENCODERS[name](x)
        out[name] = {"shape": list(ours.shape), "identical": bool(ours.shape == r.shape and np.array_equal(ours, r)),
                     "mismatched_bits": int((ours != r).sum()) if ours.shape == r.shape else None,
                     "density": float(ours.mean())}
        print(f"{name:7s} {out[name]}", flush=True)
    xa, ya = ref_augment(xtr[:args.n], ytr[:args.n])
    oa, oy = E.augment(xtr[:args.n], ytr[:args.n])
    out["augment"] = {"identical": bool(np.array_equal(xa, oa) and np.array_equal(ya, oy))}
    print("augment", out["augment"])
    ok = all(v["identical"] for v in out.values())
    write_json(new_record("encoder_check", n_images=2 * args.n, results=out, all_identical=ok),
               RESULTS / "encoder_check.json")
    print("ALL IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
