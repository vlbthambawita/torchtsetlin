"""Visualisation helpers (matplotlib is an optional dependency).

Every function returns the matplotlib ``Axes`` it drew on, so figures can be composed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from torch import Tensor

__all__ = [
    "plot_history",
    "plot_confusion_matrix",
    "plot_clause_memory",
    "plot_ta_states",
    "plot_literal_frequency",
    "plot_feature_importance",
    "plot_conv_clause",
    "plot_conv_clauses",
    "plot_votes",
    "plot_trustworthiness",
    "plot_clause_weights",
]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("matplotlib is required for torchtsetlin.viz (pip install torchtsetlin[viz])") from e
    return plt


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy() if isinstance(t, Tensor) else np.asarray(t)


def plot_history(history, metrics: Optional[Sequence[str]] = None, ax=None):
    """Plot metric curves of a :class:`~torchtsetlin.train.History`."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    keys = metrics or [k for k in history.keys() if k.endswith("accuracy") or k.endswith("mae") or k.endswith("f1")]
    for k in keys:
        ys = history[k]
        xs = [i + 1 for i, v in enumerate(ys) if v is not None]
        ax.plot(xs, [v for v in ys if v is not None], marker=".", label=k)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_confusion_matrix(cm, class_names: Optional[Sequence[str]] = None, normalize: bool = False, ax=None, cmap="Blues"):
    """Heat-map of a confusion matrix (rows = true, columns = predicted)."""
    plt = _plt()
    cm = _np(cm).astype(float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap=cmap)
    K = cm.shape[0]
    names = list(class_names) if class_names is not None else [str(i) for i in range(K)]
    ax.set_xticks(range(K), names, rotation=45, ha="right")
    ax.set_yticks(range(K), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(K):
        for j in range(K):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center", color="white" if v > thresh else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_clause_memory(model, clause: int, feature_names: Optional[Sequence[str]] = None, ax=None):
    """Book-style memory plot of one clause: the automaton state of every literal with the
    include boundary drawn as a dashed line (states above it are memorized)."""
    plt = _plt()
    names = model.literal_names(feature_names)
    states = _np(model.ta_state[clause]).astype(int) + 1  # 1-based like the book
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(names) * 0.35), 3.5))
    colors = ["tab:blue" if s > model.n_states else "lightgray" for s in states]
    ax.bar(range(len(names)), states, color=colors)
    ax.axhline(model.n_states + 0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.set_ylim(0, 2 * model.n_states + 1)
    ax.set_ylabel("memory position")
    ax.set_title(f"clause {clause}: IF {model.clause_expression(clause, feature_names)}")
    return ax


def plot_ta_states(model, clauses: Optional[Sequence[int]] = None, ax=None, cmap="RdBu_r"):
    """Heat-map of all automata states (clauses x literals), centred on the include boundary."""
    plt = _plt()
    st = model.ta_state
    if clauses is not None:
        st = st[list(clauses)]
    st = _np(st).astype(float) - (model.n_states - 0.5)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(st, aspect="auto", cmap=cmap, vmin=-model.n_states, vmax=model.n_states, interpolation="nearest")
    ax.set_xlabel("literal (x_0..x_F-1, NOT x_0..NOT x_F-1)")
    ax.set_ylabel("clause")
    plt.colorbar(im, ax=ax, label="state − boundary")
    return ax


def plot_literal_frequency(model, feature_names: Optional[Sequence[str]] = None, top_k: Optional[int] = 30, ax=None):
    """Bar plot of how many clauses include each literal."""
    plt = _plt()
    freq = _np(model.literal_frequency())
    names = model.literal_names(feature_names)
    order = np.argsort(-freq)
    if top_k:
        order = order[:top_k]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(order) * 0.3), 3.5))
    ax.bar(range(len(order)), freq[order], color=["tab:blue" if i < len(names) // 2 else "tab:orange" for i in order])
    ax.set_xticks(range(len(order)), [names[i] for i in order], rotation=90, fontsize=8)
    ax.set_ylabel("# clauses")
    return ax


def plot_feature_importance(importance, names: Optional[Sequence[str]] = None, top_k: Optional[int] = 20, ax=None, title: str = "feature importance"):
    """Horizontal bar plot of a ``(n,)`` importance vector."""
    plt = _plt()
    imp = _np(importance).reshape(-1)
    names = list(names) if names is not None else [str(i) for i in range(len(imp))]
    order = np.argsort(-imp)
    if top_k:
        order = order[:top_k]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2.5, 0.3 * len(order))))
    ax.barh(range(len(order))[::-1], imp[order])
    ax.set_yticks(range(len(order))[::-1], [names[i] for i in order])
    ax.set_title(title)
    return ax


def plot_conv_clause(model, clause: int, plane: int = 0, ax=None, show_region: bool = True):
    """Draw the patch pattern of a convolutional clause: black = pixel must be on, white =
    must be off, gray = don't care. The title reports the allowed patch-position range."""
    plt = _plt()
    patch = _np(model.clause_patch(clause))[plane].astype(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(-patch, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"clause {clause}"
    if show_region and model.position_encoding:
        r = model.clause_region(clause)
        title += f"\ny∈[{r['y'][0]},{r['y'][1]}] x∈[{r['x'][0]},{r['x'][1]}]"
    ax.set_title(title, fontsize=8)
    return ax


def plot_conv_clauses(model, clauses: Sequence[int], ncols: int = 8, plane: int = 0, figsize_per: float = 1.6):
    """Grid of :func:`plot_conv_clause` panels. Returns the figure."""
    plt = _plt()
    n = len(clauses)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per * ncols, figsize_per * 1.25 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, j in zip(axes.flat, clauses):
        ax.axis("on")
        plot_conv_clause(model, int(j), plane=plane, ax=ax)
    fig.tight_layout()
    return fig


def plot_votes(votes, y=None, class_names: Optional[Sequence[str]] = None, ax=None, bins: int = 30):
    """Histogram of the winning-class vote sums, split by correct / incorrect if ``y`` given."""
    plt = _plt()
    v = _np(votes)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    top = v.max(axis=1)
    if y is not None:
        yy = _np(y).reshape(-1)
        correct = v.argmax(axis=1) == yy
        ax.hist(top[correct], bins=bins, alpha=0.6, label="correct")
        ax.hist(top[~correct], bins=bins, alpha=0.6, label="incorrect")
        ax.legend()
    else:
        ax.hist(top, bins=bins)
    ax.set_xlabel("winning vote sum")
    ax.set_ylabel("count")
    return ax


def plot_trustworthiness(curve, ax=None):
    """Plot accuracy vs confidence level (see :func:`torchtsetlin.metrics.trustworthiness_curve`)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(_np(curve["confidence"]), _np(curve["accuracy"]), marker="o", label="accuracy")
    ax.plot(_np(curve["confidence"]), _np(curve["coverage"]), marker="s", label="coverage")
    ax.set_xlabel("confidence ≥")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_clause_weights(model, ax=None):
    """Weights of a weighted / coalesced model as an image (clauses x outputs) or bars."""
    plt = _plt()
    w = _np(model.weights)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if w.ndim == 2:
        im = ax.imshow(w.T, aspect="auto", cmap="RdBu_r", vmin=-np.abs(w).max(), vmax=np.abs(w).max(), interpolation="nearest")
        ax.set_xlabel("clause")
        ax.set_ylabel("output")
        plt.colorbar(im, ax=ax)
    else:
        ax.bar(range(len(w)), w)
        ax.set_xlabel("clause")
        ax.set_ylabel("weight")
    return ax
