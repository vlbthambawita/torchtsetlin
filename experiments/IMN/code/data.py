"""OpenML tabular data and preprocessing, mirroring the official IMN code.

Reference: ``utils.preprocess_dataset`` in https://github.com/ArlindKadra/IMN (cloned and read
on 2026-10-14). What is reproduced, and where we deviate (see DECISIONS.md D2-D4):

- columns with >90% missing values, a single unique value, or (categorical) >90% unique values
  are dropped;
- numerical columns: ``StandardScaler`` fitted on the training split, then NaN -> 0;
- categorical columns: ``TargetEncoder`` for binary tasks, ``OrdinalEncoder`` for multi-class
  tasks (the repo default ``encoding_type='ordinal'``; the paper text says one-hot);
- labels via ``LabelEncoder``;
- split: stratified train/valid/test = 60/20/20 (the repo's ``hpo_tuning=True`` protocol), so
  every arm, including the default-hyperparameter ones, sees exactly the same three splits.

Data is fetched with :func:`sklearn.datasets.fetch_openml` (no ``openml`` dependency) and cached
under ``experiments/IMN/.cache/openml`` (git-ignored).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

EXP_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = EXP_DIR / ".cache" / "openml"

# Table 6 of the paper (OpenML dataset ids). Tier 1 = the development/main benchmark: small to
# medium datasets that every arm (including a TM with a thermometer encoding) can run in
# minutes. Tier 2 = the rest of the AutoML benchmark, run only if Tier 1 warrants it.
TIER1: List[int] = [31, 1067, 1464, 1489, 40981, 54, 40984, 3, 41027, 41143, 41146, 1590]
TIER2: List[int] = [
    12, 1461, 1468, 1486, 4135, 23512, 23517, 40685, 40996, 41138, 41142, 41150, 41159,
    41161, 41163, 41164, 41165, 41166, 41168, 41169, 1111, 1169, 41147,
]
NAMES: Dict[int, str] = {
    3: "kr-vs-kp", 12: "mfeat-factors", 31: "credit-g", 54: "vehicle", 1067: "kc1",
    1111: "KDDCup09_appetency", 1169: "airlines", 1461: "bank-marketing",
    1464: "blood-transfusion", 1468: "cnae-9", 1486: "nomao", 1489: "phoneme", 1590: "adult",
    4135: "Amazon_employee_access", 23512: "higgs", 23517: "numerai28.6", 40685: "shuttle",
    40981: "Australian", 40984: "segment", 40996: "Fashion-MNIST", 41027: "jungle_chess",
    41138: "APSFailure", 41142: "christine", 41143: "jasmine", 41146: "sylvine",
    41147: "albert", 41150: "MiniBooNE", 41159: "guillermo", 41161: "riccardo",
    41163: "dilbert", 41164: "fabert", 41165: "robert", 41166: "volkert", 41168: "jannis",
    41169: "helena",
}


@dataclass
class TabularSplits:
    """Preprocessed float splits plus the metadata every arm needs."""

    dataset_id: int
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    categorical: List[bool]
    n_classes: int
    split_seed: int
    info: Dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    def fingerprint(self) -> str:
        """SHA-256 over the three splits (records which exact data a run used)."""
        h = hashlib.sha256()
        for a in (self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def fetch(dataset_id: int):
    """Raw ``(X, y)`` dataframe/series for an OpenML dataset id (cached)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(data_id=dataset_id, as_frame=True, data_home=str(CACHE_DIR), parser="auto")
    return bunch.data, bunch.target


def _is_categorical(s: pd.Series) -> bool:
    return s.dtype == "object" or str(s.dtype) == "category" or s.dtype == "string" or s.dtype == bool


def preprocess(X: pd.DataFrame, y: pd.Series, dataset_id: int, seed: int,
               test_size: float = 0.2, valid_size: float = 0.2) -> TabularSplits:
    """Split and preprocess following the official IMN ``preprocess_dataset`` (hpo protocol)."""
    X = X.copy()
    n = len(X)
    drop = []
    for c in X.columns:
        s = X[c]
        if s.isnull().sum() > 0.9 * n or s.nunique() <= 1:
            drop.append(c)
        elif _is_categorical(s) and s.nunique() / n > 0.9:
            drop.append(c)
    X = X.drop(columns=drop)
    cat_cols = [c for c in X.columns if _is_categorical(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:  # missing categories become their own category (paper Sec. 4.1)
        X[c] = X[c].astype("object").where(X[c].notna(), "__missing__").astype(str)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float64)

    y = y.astype(str) if not pd.api.types.is_numeric_dtype(y) else y
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=valid_size / (1 - test_size), random_state=seed, stratify=y_tr)

    le = LabelEncoder().fit(y_tr)
    y_tr, y_va, y_te = (le.transform(v).astype(np.int64) for v in (y_tr, y_va, y_te))
    n_classes = int(len(le.classes_))

    parts = []
    if num_cols:
        parts.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        if n_classes == 2:
            enc = TargetEncoder(random_state=seed)
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(("cat", enc, cat_cols))
    ct = ColumnTransformer(parts, remainder="drop")
    A_tr = ct.fit_transform(X_tr, y_tr)
    A_va, A_te = ct.transform(X_va), ct.transform(X_te)
    A_tr, A_va, A_te = (np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0).astype(np.float32)
                        for a in (A_tr, A_va, A_te))
    names = list(num_cols) + list(cat_cols)
    num_stats = {}
    if num_cols:
        sc = ct.named_transformers_["num"]
        num_stats = {"num_mean": sc.mean_.tolist(), "num_scale": sc.scale_.tolist()}
    categorical = [False] * len(num_cols) + [True] * len(cat_cols)
    return TabularSplits(
        dataset_id=dataset_id, name=NAMES.get(dataset_id, str(dataset_id)),
        X_train=A_tr, y_train=y_tr, X_valid=A_va, y_valid=y_va, X_test=A_te, y_test=y_te,
        feature_names=[str(c) for c in names], categorical=categorical, n_classes=n_classes,
        split_seed=seed, info={"dropped_columns": [str(c) for c in drop], "classes": [str(c) for c in le.classes_],
              "n_numeric": len(num_cols), **num_stats},
    )


_MEMO: Dict = {}


def load(dataset_id: int, seed: int) -> TabularSplits:
    key = (dataset_id, seed)
    if key not in _MEMO:
        X, y = fetch(dataset_id)
        _MEMO[key] = preprocess(X, y, dataset_id, seed)
    return _MEMO[key]


def synthetic_splits(X: np.ndarray, y: np.ndarray, name: str, seed: int,
                     feature_names: Optional[List[str]] = None) -> TabularSplits:
    """Wrap an already-numeric synthetic dataset (e.g. the XAI tasks) with the same split
    protocol; features are standardized on the training split."""
    idx = np.arange(len(X))
    strat = y if np.issubdtype(y.dtype, np.integer) else None
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    i_tr, i_va = train_test_split(i_tr, test_size=0.25, random_state=seed,
                                  stratify=None if strat is None else y[i_tr])
    sc = StandardScaler().fit(X[i_tr])
    f = lambda a: sc.transform(a).astype(np.float32)  # noqa: E731
    names = feature_names or [f"x{i}" for i in range(X.shape[1])]
    return TabularSplits(
        dataset_id=-1, name=name, X_train=f(X[i_tr]), y_train=y[i_tr], X_valid=f(X[i_va]),
        y_valid=y[i_va], X_test=f(X[i_te]), y_test=y[i_te], feature_names=names,
        categorical=[False] * X.shape[1], n_classes=int(y.max()) + 1, split_seed=seed,
        info={"scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
              "test_index": i_te.tolist()},
    )
