"""torchtsetlin — GPU-enabled, PyTorch-native Tsetlin machines.

Quick start::

    import torchtsetlin as tt

    model = tt.TsetlinMachine(n_features=12, n_classes=2, n_clauses=20, T=15, s=3.9).to("cuda")
    for x, y in loader:                 # Boolean features, integer labels
        model.update(x.cuda(), y.cuda())  # learn (Type I / Type II feedback)
    pred = model.predict(x_test.cuda())  # argmax of the vote sums
"""

from . import data, functional, interpret, metrics, viz
from .models import (
    CoalescedSegmentationTsetlinMachine,
    CoalescedTsetlinMachine,
    Conv1dTsetlinMachine,
    ConvCoalescedTsetlinMachine,
    ConvRegressionTsetlinMachine,
    ConvTsetlinMachine,
    RegressionTsetlinMachine,
    SegmentationTsetlinMachine,
    TsetlinMachine,
    TsetlinMachineBase,
)
from .train import (
    Callback,
    CSVLogger,
    EarlyStopping,
    History,
    HyperparameterSchedule,
    LambdaCallback,
    ModelCheckpoint,
    ProgressLogger,
    StateSummaryLogger,
    Trainer,
    evaluate,
    predict,
)
from .utils import seed_everything

__version__ = "0.2.0"
__all__ = [
    "__version__",
    # sub-packages
    "data",
    "functional",
    "interpret",
    "metrics",
    "viz",
    # models
    "TsetlinMachineBase",
    "TsetlinMachine",
    "CoalescedTsetlinMachine",
    "RegressionTsetlinMachine",
    "ConvTsetlinMachine",
    "ConvCoalescedTsetlinMachine",
    "ConvRegressionTsetlinMachine",
    "Conv1dTsetlinMachine",
    "SegmentationTsetlinMachine",
    "CoalescedSegmentationTsetlinMachine",
    # training
    "Trainer",
    "evaluate",
    "predict",
    "History",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "CSVLogger",
    "LambdaCallback",
    "HyperparameterSchedule",
    "ProgressLogger",
    "StateSummaryLogger",
    # utils
    "seed_everything",
]
