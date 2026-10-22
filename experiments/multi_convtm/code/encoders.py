"""The three Booleanizations of TM Composites (Granmo 2023), transcribed from the authors' scripts
(github.com/cair/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines @ 91524f0:
CIFAR10AdaptiveThresholding.py, CIFAR10ColorThermometerScoring.py, CIFAR10HistogramOfGradients.py).

Each takes uint8 ``(N, 32, 32, 3)`` images and returns a NumPy Boolean array:
conv specialists ``(N, Z, 32, 32)`` (channel-first, as ``ConvTsetlinMachine`` expects), the
HOG specialist a flat ``(N, 5832)``.
"""

import cv2
import numpy as np


def adaptive_gaussian_threshold(x):
    """``cv2.adaptiveThreshold(channel, 1, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2)``
    per colour channel -> 3 bit-planes."""
    out = np.empty((x.shape[0], x.shape[3], x.shape[1], x.shape[2]), dtype=bool)
    for i in range(x.shape[0]):
        for j in range(x.shape[3]):
            out[i, j] = cv2.adaptiveThreshold(np.ascontiguousarray(x[i, :, :, j]), 1,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              11, 2).astype(bool)
    return out


def color_thermometer(x, resolution=8):
    """``x >= (z+1)*255/(resolution+1)`` for z = 0..resolution-1, per colour channel. The
    reference reshapes ``(N,32,32,3,res)`` to ``(N,32,32,3*res)``, i.e. plane index
    ``channel*res + z``; kept here."""
    levels = (np.arange(1, resolution + 1) * 255 / (resolution + 1))
    bits = x[..., None] >= levels                     # (N, 32, 32, 3, res)
    n, h, w = x.shape[:3]
    return np.ascontiguousarray(bits.reshape(n, h, w, 3 * resolution).transpose(0, 3, 1, 2))


def hog_descriptor():
    # Exactly the reference parameters; nbins=18 and signedGradient=True are non-default.
    return cv2.HOGDescriptor((32, 32), (12, 12), (4, 4), (4, 4), 18, 1, -1.0, 0, 0.2, True, 64, True)


def histogram_of_gradients(x, threshold=0.1):
    """``hog.compute(img_uint8_rgb) >= 0.1`` -> 5 832 bits per image (flat specialist)."""
    hog = hog_descriptor()
    first = hog.compute(np.ascontiguousarray(x[0]))
    out = np.empty((x.shape[0], first.shape[0]), dtype=bool)
    for i in range(x.shape[0]):
        out[i] = hog.compute(np.ascontiguousarray(x[i])).reshape(-1) >= threshold
    return out


ENCODERS = {
    "thresh": adaptive_gaussian_threshold,
    "therm8": color_thermometer,
    "hog": histogram_of_gradients,
}
