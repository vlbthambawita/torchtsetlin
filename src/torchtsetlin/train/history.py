"""Training history container."""

from __future__ import annotations

from typing import Dict, List, Optional


class History:
    """Per-epoch metrics collected by :class:`torchtsetlin.train.Trainer`.

    ``history.epochs`` is a list of dicts; ``history["val_accuracy"]`` returns the series of a
    metric (``None`` where missing); ``history.best("val_accuracy")`` the best value/epoch.
    """

    def __init__(self) -> None:
        self.epochs: List[Dict] = []

    def append(self, logs: Dict) -> None:
        self.epochs.append(dict(logs))

    def __getitem__(self, key: str) -> List[Optional[float]]:
        return [e.get(key) for e in self.epochs]

    def __len__(self) -> int:
        return len(self.epochs)

    def keys(self) -> List[str]:
        keys: List[str] = []
        for e in self.epochs:
            for k in e:
                if k not in keys:
                    keys.append(k)
        return keys

    def best(self, key: str, mode: str = "max"):
        vals = [(v, i) for i, v in enumerate(self[key]) if v is not None]
        if not vals:
            return None, None
        v, i = max(vals) if mode == "max" else min(vals)
        return v, i

    def to_dataframe(self):
        """Return a ``pandas.DataFrame`` (pandas required)."""
        import pandas as pd

        return pd.DataFrame(self.epochs)

    def plot(self, metrics=None, ax=None):
        """Plot metric curves (delegates to :func:`torchtsetlin.viz.plot_history`)."""
        from ..viz import plot_history

        return plot_history(self, metrics=metrics, ax=ax)

    def __repr__(self) -> str:
        return f"History(epochs={len(self)}, keys={self.keys()})"
