"""pyTsetlinMachineParallel 0.2.1 (C + OpenMP; threads from OMP_NUM_THREADS)."""
from .pytm import Adapter as _PyTM


class Adapter(_PyTM):
    module = "pyTsetlinMachineParallel.tm"
