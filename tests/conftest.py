import pytest
import torch


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _probe(x):
    return x + 1


@pytest.fixture
def compiled(monkeypatch):
    """Run the ``torch.compile`` kernels even on test-sized inputs (skipped where
    ``torch.compile`` does not work, e.g. without a C++ compiler)."""
    from torchtsetlin import functional as F

    monkeypatch.setattr(F, "COMPILE_MIN_ELEMENTS", 0)
    monkeypatch.setitem(F._COMPILE, "enabled", True)
    if F._run_compiled(_probe, torch.ones(2)) is None:
        pytest.skip("torch.compile is unavailable")
    return F
