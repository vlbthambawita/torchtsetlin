"""Packaging sanity checks (guards against subpackages missing from the wheel)."""

import importlib
import pkgutil

import torchtsetlin


def test_all_subpackages_present():
    found = {m.name for m in pkgutil.iter_modules(torchtsetlin.__path__)}
    assert {"data", "models", "train", "functional", "metrics", "interpret", "viz", "utils"} <= found


def test_version_matches_metadata():
    from importlib.metadata import version

    assert torchtsetlin.__version__ == version("torchtsetlin")


def test_public_names_importable():
    for name in torchtsetlin.__all__:
        assert getattr(torchtsetlin, name, None) is not None, name
    for mod in ("torchtsetlin.data", "torchtsetlin.train", "torchtsetlin.models"):
        importlib.import_module(mod)
