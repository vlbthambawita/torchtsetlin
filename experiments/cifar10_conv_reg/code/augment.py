"""Training-time augmentation with AlbumentationsX on the uint8 RGB images, before the
thermometer encoding (plan §4). The policies and the random streams are those of
``cifar10_joint_teams/code/augment.py`` (this file is its copy without the OpenCV
binarizations), so a run here sees exactly the augmented epochs a joint-team run sees.

Rules:
  1. One augmented image per training image and epoch.
  2. The random stream of image ``i`` in epoch ``e`` depends only on ``(seed, e, i)``: a resumed
     run trains on exactly the same augmented epochs.
  3. ``EpochSource`` prepares the next epoch in worker processes while the current one trains.
  4. Validation and test images never pass through here.

Library: ``pip install albumentationsx`` (import name ``albumentations``; AGPL-3.0, Python >=
3.11). It sends usage telemetry unless ``ALBUMENTATIONS_NO_TELEMETRY=1`` is set *before* the
import; this module sets it (and ``ALBUMENTATIONS_OFFLINE`` / ``NO_ALBUMENTATIONS_UPDATE``) itself.
``albumentations==2.0.8`` (MIT, the last release before the rename) has the same API for the
transforms used here and is the fallback on Python < 3.11.
"""

import os

os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"
os.environ["ALBUMENTATIONS_OFFLINE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
POLICIES = ("geo", "geo+col", "full")


def telemetry_off():
    return all(os.environ.get(k) == "1" for k in ("ALBUMENTATIONS_NO_TELEMETRY", "ALBUMENTATIONS_OFFLINE"))


def build(policy):
    """The transform pipeline of a policy (cifar10_joint_teams plan §5)."""
    import inspect
    import warnings

    import albumentations as A
    if policy not in POLICIES:
        raise KeyError(policy)
    kw = {}
    if "telemetry" in inspect.signature(A.Compose.__init__).parameters:
        kw["telemetry"] = False              # AlbumentationsX: also off per pipeline
    # ColorJitter: AlbumentationsX takes ranges, albumentations 2.0.8 takes amounts (same meaning).
    if "brightness_range" in inspect.signature(A.ColorJitter.__init__).parameters:
        jitter = dict(brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2),
                      hue_range=(-0.02, 0.02))
    else:
        jitter = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
    with warnings.catch_warnings():          # a mis-named argument must stop the run, not be ignored
        warnings.filterwarnings("error", message=".*not valid for transform.*")
        return _build(A, policy, jitter, kw)


def _build(A, policy, jitter, kw):
    reflect = cv2.BORDER_REFLECT_101
    t = [A.PadIfNeeded(min_height=40, min_width=40, border_mode=reflect), A.RandomCrop(height=32, width=32),
         A.HorizontalFlip(p=0.5)]
    if policy in ("geo+col", "full"):
        t.append(A.ColorJitter(p=0.5, **jitter))
    if policy == "full":
        t.append(A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), border_mode=reflect, p=0.5))
        t.append(A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(8, 8), hole_width_range=(8, 8),
                                 fill=0, p=0.5))
    return A.Compose(t, **kw)


def image_seed(seed, epoch, i):
    """A 32-bit seed that depends only on (run seed, epoch, image index)."""
    return int(np.random.SeedSequence([int(seed), int(epoch), int(i)]).generate_state(1)[0])


_W = {}


def _init(x, policy, seed):
    cv2.setNumThreads(1)
    _W.update(x=x, t=build(policy), seed=seed)


def _work(args):
    epoch, lo, hi = args
    x, t = _W["x"], _W["t"]
    out = np.empty((hi - lo, 32, 32, 3), dtype=np.uint8)
    for i in range(lo, hi):
        t.set_random_seed(image_seed(_W["seed"], epoch, i))
        out[i - lo] = t(image=x[i])["image"]
    return lo, out


def augment_epoch(x, policy, seed, epoch, lo=0, hi=None):
    """In-process reference (checks): the augmented images ``lo:hi`` of an epoch."""
    _init(x, policy, seed)
    return _work((epoch, lo, len(x) if hi is None else hi))[1]


class EpochSource:
    """Augmented epochs from a pool of worker processes, one epoch ahead.

    ``get(e)`` returns the images of epoch ``e`` (uint8 (N, 32, 32, 3)) and starts epoch
    ``e + 1`` (unless ``last``)."""

    def __init__(self, x, policy, seed, workers=None, chunk=1000):
        self.n, self.chunk = len(x), chunk
        workers = workers or int(os.environ.get("CCR_WORKERS", 0)) or max(2, min(8, (os.cpu_count() or 8) // 4 - 1))
        self.workers = workers
        self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                        initargs=(x, policy, seed))
        self.pending = {}

    def _submit(self, epoch):
        if epoch not in self.pending:
            self.pending[epoch] = [self.pool.submit(_work, (epoch, lo, min(lo + self.chunk, self.n)))
                                   for lo in range(0, self.n, self.chunk)]

    def get(self, epoch, last=False):
        self._submit(epoch)
        if not last:
            self._submit(epoch + 1)
        img = np.empty((self.n, 32, 32, 3), dtype=np.uint8)
        for f in self.pending.pop(epoch):
            lo, out = f.result()
            img[lo:lo + len(out)] = out
        for old in [k for k in self.pending if k < epoch]:
            for f in self.pending.pop(old):
                f.cancel()
        return img

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
