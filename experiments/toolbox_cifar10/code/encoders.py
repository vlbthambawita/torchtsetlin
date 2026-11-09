"""The seven Booleanizations of the Optimized Toolbox (Grønningsæter et al. 2024, arXiv:2406.00704),
transcribed from the authors' scripts (``reference_code/``, cair repo @ a4f6093).

Each encoder takes uint8 ``(N, 32, 32, 3)`` RGB images and returns a NumPy Boolean array:
convolutional specialists ``(N, Z, 32, 32)`` (channel-first, as ``ConvTsetlinMachine`` expects),
HOG a flat ``(N, 5832)``. Plane order follows the scripts: channel ``j`` for the 3-plane
encodings, ``channel * 8 + z`` for the thermometers (their ``reshape(..., 3 * resolution)``).
A TM treats its features as a set, so the order only matters for reproducibility.

``code/check_encoders.py`` compares every encoder bit for bit with the authors' own code.
"""

import math

import cv2
import numpy as np

from common import ENC_CACHE, load_cifar10


def _per_channel(x, fn):
    out = np.empty((x.shape[0], x.shape[3], x.shape[1], x.shape[2]), dtype=bool)
    for i in range(x.shape[0]):
        for j in range(x.shape[3]):
            out[i, j] = fn(np.ascontiguousarray(x[i, :, :, j])) != 0
    return out


def canny(x):
    """``GaussianBlur(ch, (3, 3), 0)`` then ``Canny(blur, 100, 200) != 0`` per channel."""
    return _per_channel(x, lambda ch: cv2.Canny(cv2.GaussianBlur(ch, (3, 3), 0), 100, 200))


def adaptive_gaussian(x):
    """``adaptiveThreshold(ch, 1, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2)`` per channel."""
    return _per_channel(x, lambda ch: cv2.adaptiveThreshold(
        ch, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))


def adaptive_mean(x):
    """``adaptiveThreshold(ch, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2)`` per channel."""
    return _per_channel(x, lambda ch: cv2.adaptiveThreshold(
        ch, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2))


def otsu(x):
    """``threshold(ch, 0, 1, THRESH_BINARY + THRESH_OTSU)`` per channel (no blur, as in the script)."""
    return _per_channel(x, lambda ch: cv2.threshold(ch, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])


def color_thermometer(x, resolution=8):
    """``x >= (z+1)*255/(resolution+1)`` for z = 0..resolution-1, per colour channel."""
    levels = np.arange(1, resolution + 1) * 255 / (resolution + 1)
    bits = x[..., None] >= levels                     # (N, 32, 32, 3, res)
    n, h, w = x.shape[:3]
    return np.ascontiguousarray(bits.reshape(n, h, w, 3 * resolution).transpose(0, 3, 1, 2))


def _multilevel_thresholds(img, resolution=8):
    """``multi_level_tresholding`` from ``MultilevelThresholdingThermometers.py``, kept line for
    line, quirks included (``k`` grows as ``k * (i + 1)``; an empty interval yields a NaN
    threshold, which ``>=`` turns into an all-False plane; ``a > b`` yields ``-1``, all-True)."""
    a, b = 0, 255
    n = resolution
    k = 0.7
    T = []

    def multi_thresh(img, a, b):
        if a > b:
            return -1, -1
        X = np.multiply(img >= a, img <= b)
        Y = np.multiply(img, X)
        s = np.sum(X)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.sum(Y) / s
        return m, s

    for i in range(int(n / 2 - 1)):
        X = np.multiply(img >= a, img <= b)
        Y = np.multiply(img, X)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.sum(Y) / np.sum(X)
            Z = np.multiply(Y - mu, X)
            sigma = math.sqrt(np.sum(np.multiply(Z, Z)) / np.sum(X))
        T1 = mu - k * sigma
        T2 = mu + k * sigma
        x, _ = multi_thresh(img, a, T1)
        w, _ = multi_thresh(img, T2, b)
        T.append(x)
        T.append(w)
        a = T1 + 1
        b = T2 - 1
        k = k * (i + 1)
    T1 = mu
    T2 = mu + 1
    x, _ = multi_thresh(img, a, T1)
    w, _ = multi_thresh(img, T2, b)
    T.append(x)
    T.append(w)
    T.sort()
    return T


def adaptive_color_thermometer(x, resolution=8):
    """``mlt_temp``: per image and channel, ``ch >= t`` for the 8 multilevel thresholds."""
    out = np.empty((x.shape[0], x.shape[1], x.shape[2], 3, resolution), dtype=bool)
    for i in range(x.shape[0]):
        for c in range(3):
            ch = x[i, :, :, c]
            for z, t in enumerate(_multilevel_thresholds(ch, resolution)):
                out[i, :, :, c, z] = ch >= t
    n, h, w = x.shape[:3]
    return np.ascontiguousarray(out.reshape(n, h, w, 3 * resolution).transpose(0, 3, 1, 2))


def hog_descriptor():
    # The scripts' parameters; nbins=18 and signedGradient=True are non-default.
    return cv2.HOGDescriptor((32, 32), (12, 12), (4, 4), (4, 4), 18, 1, -1.0, 0, 0.2, True, 64, True)


def hog(x, threshold=0.1):
    """``hog.compute(img_uint8_rgb) >= 0.1`` -> 5 832 bits per image (flat specialist)."""
    h = hog_descriptor()
    first = h.compute(np.ascontiguousarray(x[0]))
    out = np.empty((x.shape[0], first.shape[0]), dtype=bool)
    for i in range(x.shape[0]):
        out[i] = h.compute(np.ascontiguousarray(x[i])).reshape(-1) >= threshold
    return out


ENCODERS = {
    "canny": canny,
    "agauss": adaptive_gaussian,
    "amean": adaptive_mean,
    "otsu": otsu,
    "therm": color_thermometer,
    "atherm": adaptive_color_thermometer,
    "hog": hog,
}


def augment(x, y):
    """The scripts' augmentation: every training image followed by ``cv2.flip(img, 1)``
    (horizontal flip), interleaved, 50 000 -> 100 000 images. The test set is not augmented."""
    out = np.empty((2 * x.shape[0],) + x.shape[1:], dtype=x.dtype)
    out[0::2] = x
    out[1::2] = x[:, :, ::-1, :]                       # == cv2.flip(img, 1) for (H, W, C)
    return out, np.repeat(y, 2)


def encoded(encoding, augmented):
    """(train bits, train labels, test bits, test labels), cached in ``.cache/enc`` as packed bits."""
    ENC_CACHE.mkdir(parents=True, exist_ok=True)
    path = ENC_CACHE / f"{encoding}{'_aug' if augmented else ''}.npz"
    if path.exists():
        z = np.load(path)
        unpack = lambda k: np.unpackbits(z[k], count=int(np.prod(z[k + "_shape"]))).reshape(z[k + "_shape"]).astype(bool)
        return unpack("xtr"), z["ytr"], unpack("xte"), z["yte"]
    xtr, ytr, xte, yte = load_cifar10()
    if augmented:
        xtr, ytr = augment(xtr, ytr)
    enc = ENCODERS[encoding]
    btr, bte = enc(xtr), enc(xte)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, xtr=np.packbits(btr), xtr_shape=np.array(btr.shape), ytr=ytr,
             xte=np.packbits(bte), xte_shape=np.array(bte.shape), yte=yte)
    tmp.replace(path)
    return btr, ytr, bte, yte
