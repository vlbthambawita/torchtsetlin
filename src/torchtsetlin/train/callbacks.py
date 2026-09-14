"""Callbacks for :class:`torchtsetlin.train.Trainer` (Keras / Lightning flavoured)."""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Callable, Dict

import torch

__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "CSVLogger",
    "LambdaCallback",
    "HyperparameterSchedule",
    "ProgressLogger",
    "StateSummaryLogger",
]


class Callback:
    """Base callback. Override any of the hooks; ``trainer`` and ``model`` are attached by
    the trainer before training starts."""

    trainer = None
    model = None

    def set_trainer(self, trainer) -> None:
        self.trainer = trainer
        self.model = trainer.model

    def on_train_begin(self, logs: Dict) -> None: ...

    def on_train_end(self, logs: Dict) -> None: ...

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None: ...

    def on_epoch_end(self, epoch: int, logs: Dict) -> None: ...

    def on_batch_begin(self, batch: int, logs: Dict) -> None: ...

    def on_batch_end(self, batch: int, logs: Dict) -> None: ...


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: key in the epoch logs (e.g. ``"val_accuracy"``, ``"val_mae"``).
        patience: epochs without improvement before stopping.
        mode: ``"max"`` or ``"min"``.
        min_delta: minimum change to count as an improvement.
        restore_best: reload the best ``state_dict`` when training stops.
    """

    def __init__(self, monitor: str = "val_accuracy", patience: int = 10, mode: str = "max", min_delta: float = 0.0, restore_best: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best: float = -math.inf if mode == "max" else math.inf
        self.best_epoch = -1
        self.wait = 0
        self._best_state = None

    def _improved(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        if self.monitor not in logs:
            return
        value = float(logs[self.monitor])
        if self._improved(value):
            self.best, self.best_epoch, self.wait = value, epoch, 0
            if self.restore_best:
                self._best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.trainer.stop_training = True
                logs["early_stopped"] = True

    def on_train_end(self, logs: Dict) -> None:
        if self.restore_best and self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logs["restored_epoch"] = self.best_epoch


class ModelCheckpoint(Callback):
    """Save ``model.state_dict()`` to ``path`` whenever the monitored metric improves
    (``save_best_only=True``) or after every epoch."""

    def __init__(self, path: str, monitor: str = "val_accuracy", mode: str = "max", save_best_only: bool = True, verbose: bool = False):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best: float = -math.inf if mode == "max" else math.inf

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        value = logs.get(self.monitor)
        improved = value is not None and (
            (self.mode == "max" and float(value) > self.best) or (self.mode == "min" and float(value) < self.best)
        )
        if improved:
            self.best = float(value)
        if improved or not self.save_best_only:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            torch.save({"state_dict": self.model.state_dict(), "epoch": epoch, "logs": dict(logs)}, self.path)
            if self.verbose:
                print(f"[ModelCheckpoint] saved to {self.path} ({self.monitor}={value})")


class CSVLogger(Callback):
    """Append the epoch logs to a CSV file."""

    def __init__(self, path: str, append: bool = False):
        self.path = path
        self.append = append
        self._keys = None

    def on_train_begin(self, logs: Dict) -> None:
        if not self.append and os.path.exists(self.path):
            os.remove(self.path)

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        row = {"epoch": epoch, **{k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool))}}
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            if self._keys is None:
                self._keys = list(row.keys())
            w = csv.DictWriter(f, fieldnames=self._keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


class LambdaCallback(Callback):
    """Build a callback from plain functions, e.g. ``LambdaCallback(on_epoch_end=lambda e, logs: ...)``."""

    def __init__(self, **hooks: Callable):
        for name, fn in hooks.items():
            if not hasattr(Callback, name):
                raise ValueError(f"Unknown hook {name}")
            setattr(self, name, fn)


class HyperparameterSchedule(Callback):
    """Change a model hyper-parameter (``"s"``, ``"T"``, ``"drop_clause_p"``, ...) at the start
    of each epoch: ``schedule(epoch) -> value``. Handy for annealing the specificity."""

    def __init__(self, name: str, schedule: Callable[[int], float]):
        self.name = name
        self.schedule = schedule

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None:
        value = self.schedule(epoch)
        setattr(self.model, self.name, value)
        logs[self.name] = value


class ProgressLogger(Callback):
    """Print one line per epoch with the logged metrics."""

    def __init__(self, print_fn: Callable[[str], None] = print):
        self.print_fn = print_fn
        self._t0 = 0.0

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None:
        self._t0 = time.time()

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        items = []
        for k, v in logs.items():
            if k == "epoch_time":
                continue
            if isinstance(v, float):
                items.append(f"{k}={v:.4f}")
            elif isinstance(v, (int, str)):
                items.append(f"{k}={v}")
        self.print_fn(f"epoch {epoch + 1:3d} | {time.time() - self._t0:6.1f}s | " + " ".join(items))


class StateSummaryLogger(Callback):
    """Add automata statistics (mean/max included literals, empty clauses) to the epoch logs."""

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        for k, v in self.model.state_summary().items():
            if k in ("included_literals_mean", "empty_clauses"):
                logs[k] = v
