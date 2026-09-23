"""Tsetlin machine model zoo."""

from .base import FeedbackAccumulator, TsetlinMachineBase
from .classifier import TsetlinMachine
from .coalesced import CoalescedTsetlinMachine
from .conv import (
    Conv1dTsetlinMachine,
    ConvCoalescedTsetlinMachine,
    ConvRegressionTsetlinMachine,
    ConvTsetlinMachine,
)
from .regression import RegressionTsetlinMachine
from .segmentation import (
    CoalescedSegmentationTsetlinMachine,
    SegmentationTsetlinMachine,
)

__all__ = [
    "TsetlinMachineBase",
    "FeedbackAccumulator",
    "TsetlinMachine",
    "CoalescedTsetlinMachine",
    "RegressionTsetlinMachine",
    "ConvTsetlinMachine",
    "ConvCoalescedTsetlinMachine",
    "ConvRegressionTsetlinMachine",
    "Conv1dTsetlinMachine",
    "SegmentationTsetlinMachine",
    "CoalescedSegmentationTsetlinMachine",
]
