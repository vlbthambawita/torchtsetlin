"""A PyTorch-style training loop for Tsetlin machines."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .. import metrics as M
from ..models import CoalescedTsetlinMachine, RegressionTsetlinMachine, TsetlinMachineBase
from ..utils import _to_device, as_bool_tensor
from .callbacks import Callback, ProgressLogger
from .history import History

__all__ = ["Trainer", "evaluate", "predict"]

DataLike = Union[DataLoader, Dataset, Tuple, List]


class _TensorBatcher:
    """Fast in-memory batching for ``(x, y)`` tensors that already live on the device."""

    def __init__(self, x: Tensor, y: Tensor, batch_size: int, shuffle: bool):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __iter__(self):
        n = self.x.shape[0]
        idx = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(0, n, self.batch_size):
            j = idx[i : i + self.batch_size]
            yield self.x[j], self.y[j]


class Trainer:
    """Train, validate and test Tsetlin machines with the familiar ``fit`` / ``evaluate`` /
    ``predict`` trio, PyTorch ``DataLoader`` support and callbacks.

    Args:
        model: any :class:`~torchtsetlin.models.TsetlinMachineBase`.
        device (str | torch.device | None): where to run (defaults to the model's device;
            the model is moved if given).
        callbacks: list of :class:`~torchtsetlin.train.Callback` objects.
        metrics: extra metrics ``{name: fn(votes_or_pred, y) -> float}`` computed on
            validation / test data in addition to the defaults (accuracy for classification,
            MAE/RMSE/R² for regression, Hamming/subset accuracy for multi-label).
        batch_size: default batch size when ``fit`` receives tensors or a ``Dataset``.
        eval_batch_size: batch size for evaluation (defaults to ``4 * batch_size``).
        verbose: print one line per epoch.

    Example:
        >>> trainer = Trainer(model, device="cuda", callbacks=[EarlyStopping(patience=5)])
        >>> history = trainer.fit((x_train, y_train), epochs=30, val_data=(x_val, y_val))
        >>> trainer.evaluate((x_test, y_test))
        {'accuracy': 0.98, ...}
    """

    def __init__(
        self,
        model: TsetlinMachineBase,
        device=None,
        callbacks: Optional[Sequence[Callback]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        batch_size: int = 32,
        eval_batch_size: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        if device is not None:
            self.model.to(torch.device(device))
        self.callbacks: List[Callback] = list(callbacks or [])
        if verbose and not any(isinstance(c, ProgressLogger) for c in self.callbacks):
            self.callbacks.append(ProgressLogger())
        for c in self.callbacks:
            c.set_trainer(self)
        self.extra_metrics = dict(metrics or {})
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size or 4 * batch_size)
        self.history = History()
        self.stop_training = False
        self.epoch = 0
        self.global_step = 0

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        return self.model.ta_state.device

    @property
    def task(self) -> str:
        if isinstance(self.model, RegressionTsetlinMachine):
            return "regression"
        if isinstance(self.model, CoalescedTsetlinMachine) and self.model.multi_label:
            return "multilabel"
        return "classification"

    def _batches(self, data: DataLike, batch_size: int, shuffle: bool) -> Iterable:
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, Dataset):
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        if isinstance(data, (tuple, list)) and len(data) == 2:
            x, y = data
            x = as_bool_tensor(x, device=self.device)
            if self.task == "regression":
                y = torch.as_tensor(y, dtype=torch.float32, device=self.device) if not isinstance(y, Tensor) else y.to(self.device, torch.float32)
            elif self.task == "multilabel":
                y = as_bool_tensor(y, device=self.device)
            else:
                y = torch.as_tensor(y, dtype=torch.long, device=self.device) if not isinstance(y, Tensor) else y.to(self.device, torch.long)
            return _TensorBatcher(x, y, batch_size, shuffle)
        raise TypeError("data must be a DataLoader, a Dataset or an (x, y) pair")

    def _move(self, x, y):
        if isinstance(x, Tensor) and x.device != self.device:
            x = _to_device(x, self.device)
        if isinstance(y, Tensor) and y.device != self.device:
            y = _to_device(y, self.device)
        return x, y

    def _call(self, hook: str, *args) -> None:
        for c in self.callbacks:
            getattr(c, hook)(*args)

    # ------------------------------------------------------------------ API
    def fit(
        self,
        train_data: DataLike,
        epochs: int = 10,
        val_data: Optional[DataLike] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        eval_train: bool = False,
    ) -> History:
        """Train for ``epochs`` epochs.

        Args:
            train_data: ``(x, y)`` tensors/arrays, a ``Dataset`` or a ``DataLoader``.
            epochs: number of passes over the data.
            val_data: optional validation data evaluated after each epoch (``val_*`` logs).
            batch_size: overrides the trainer default for tensor / ``Dataset`` inputs.
            shuffle: shuffle tensor / ``Dataset`` inputs each epoch.
            eval_train: also evaluate on the training data each epoch (``train_*`` logs,
                slower). The running batch accuracy (``batch_accuracy``) is always logged.
        """
        bs = int(batch_size or self.batch_size)
        batches = self._batches(train_data, bs, shuffle)
        self.stop_training = False
        logs: Dict = {}
        self._call("on_train_begin", logs)
        for epoch in range(epochs):
            self.epoch = epoch
            logs = {}
            self._call("on_epoch_begin", epoch, logs)
            self.model.train()
            if self.model.drop_granularity == "epoch":
                self.model.resample_dropout()
            t0 = time.time()
            correct, total, err_sum = 0.0, 0, 0.0
            for b, (x, y) in enumerate(batches):
                x, y = self._move(x, y)
                self._call("on_batch_begin", b, logs)
                votes = self.model.update(x, y)
                total += votes.shape[0]
                if self.task == "classification":
                    correct += float((votes.argmax(dim=1) == y.view(-1)).sum())
                elif self.task == "multilabel":
                    correct += float(((votes > 0) == y.bool()).float().mean(dim=1).sum())
                else:
                    err_sum += float((self.model.votes_to_targets(votes) - y.view(-1)).abs().sum())
                self.global_step += 1
                self._call("on_batch_end", b, logs)
            logs["epoch_time"] = time.time() - t0
            if total:
                if self.task == "regression":
                    logs["batch_mae"] = err_sum / total
                else:
                    logs["batch_accuracy"] = correct / total
            if eval_train:
                logs.update({f"train_{k}": v for k, v in self.evaluate(train_data, prefix="").items()})
            if val_data is not None:
                logs.update(self.evaluate(val_data, prefix="val_"))
            self._call("on_epoch_end", epoch, logs)
            self.history.append(logs)
            if self.stop_training:
                break
        self._call("on_train_end", logs)
        return self.history

    @torch.no_grad()
    def predict(self, data: DataLike, batch_size: Optional[int] = None, return_votes: bool = False) -> Tensor:
        """Predictions (or vote sums with ``return_votes=True``) for a dataset / tensors."""
        self.model.eval()
        bs = int(batch_size or self.eval_batch_size)
        if isinstance(data, Tensor) or (isinstance(data, (tuple, list)) and len(data) == 2 and not isinstance(data[0], (Dataset, DataLoader))):
            x = data[0] if isinstance(data, (tuple, list)) else data
            x = as_bool_tensor(x, device=self.device)
            outs = [self.model(x[i : i + bs]) for i in range(0, x.shape[0], bs)]
        else:
            outs = []
            for batch in self._batches(data, bs, shuffle=False):
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x, _ = self._move(x, None)
                outs.append(self.model(x))
        votes = torch.cat(outs, dim=0)
        if return_votes:
            return votes
        return self._votes_to_pred(votes)

    def _votes_to_pred(self, votes: Tensor) -> Tensor:
        if self.task == "regression":
            return self.model.votes_to_targets(votes)
        if self.task == "multilabel":
            return votes > 0
        return votes.argmax(dim=1)

    @torch.no_grad()
    def evaluate(self, data: DataLike, batch_size: Optional[int] = None, prefix: str = "") -> Dict[str, float]:
        """Compute metrics over ``data`` (model in ``eval()`` mode)."""
        self.model.eval()
        bs = int(batch_size or self.eval_batch_size)
        votes_l, y_l = [], []
        for x, y in self._batches(data, bs, shuffle=False):
            x, y = self._move(x, y)
            votes_l.append(self.model(x))
            y_l.append(y)
        votes = torch.cat(votes_l, dim=0)
        y = torch.cat(y_l, dim=0)
        out: Dict[str, float] = {}
        if self.task == "classification":
            out["accuracy"] = M.accuracy(votes, y.view(-1))
            arg = votes
        elif self.task == "multilabel":
            pred = votes > 0
            out.update(M.multilabel_metrics(pred, y))
            arg = pred
        else:
            pred = self.model.votes_to_targets(votes)
            out.update(M.regression_metrics(pred, y.view(-1)))
            arg = pred
        for name, fn in self.extra_metrics.items():
            out[name] = float(fn(arg, y))
        self.model.train()
        return {prefix + k: v for k, v in out.items()}

    def test(self, data: DataLike, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Alias of :meth:`evaluate` with a ``test_`` prefix."""
        return self.evaluate(data, batch_size, prefix="test_")


def evaluate(model: TsetlinMachineBase, data: DataLike, batch_size: int = 512, **kwargs) -> Dict[str, float]:
    """Functional shortcut: ``Trainer(model, verbose=False).evaluate(data)``."""
    return Trainer(model, verbose=False, eval_batch_size=batch_size, **kwargs).evaluate(data)


def predict(model: TsetlinMachineBase, data, batch_size: int = 512) -> Tensor:
    """Functional shortcut for :meth:`Trainer.predict`."""
    return Trainer(model, verbose=False, eval_batch_size=batch_size).predict(data)
