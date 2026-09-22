"""CNN baselines for the convtm programme (dl-expert).

`models.py`    architectures (simple CNN, VGG-16, CIFAR ResNet, ResNet-18, CTM-shaped, BNN-VGG)
`binary.py`    BinaryConnect / BNN / XNOR-Net layers and the sign straight-through estimator
`arms_cnn.py`  the arm registry + `CnnArm`, which satisfies `code/CONTRACT.md`'s arm protocol
`train.py`     the driver: trains one arm, writes one `results/*.json` via `code/record.py`

Nothing here imports or modifies `src/torchtsetlin` (constraint C1).
"""

from .arms_cnn import CNN_ARMS, CnnArm, build_cnn_arm, register_cnn_arms  # noqa: F401

__all__ = ["CNN_ARMS", "CnnArm", "build_cnn_arm", "register_cnn_arms"]
