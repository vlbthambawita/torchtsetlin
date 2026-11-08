"""The example notebooks are shipped without outputs and open in Google Colab.

The docs render them as they are stored (``mkdocs-jupyter``, ``execute: false``), so stale outputs
would be shown as results of the current library. Every notebook needs an Open-in-Colab badge
pointing at its own file, a setup cell that installs torchtsetlin when it runs in Colab, a nav entry
in ``mkdocs.yml`` and a row on the notebooks page.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples" / "notebooks").glob("*.ipynb"))
COLAB = "https://colab.research.google.com/github/vlbthambawita/torchtsetlin/blob/main/examples/notebooks/"

pytestmark = pytest.mark.skipif(not NOTEBOOKS, reason="no example notebooks in this checkout")


def _cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return [(c["cell_type"], "".join(c["source"]), c) for c in nb["cells"]]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_stored_without_outputs(path):
    for kind, _, cell in _cells(path):
        if kind == "code":
            assert cell.get("outputs") == [], f"{path.name}: strip the outputs before committing"
            assert cell.get("execution_count") is None, path.name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_opens_in_colab(path):
    cells = _cells(path)
    markdown = "\n".join(src for kind, src, _ in cells[:3] if kind == "markdown")
    assert COLAB + path.name in markdown, f"{path.name}: Open-in-Colab badge missing or wrong"
    code = [src for kind, src, _ in cells if kind == "code"]
    assert 'IN_COLAB = "google.colab" in sys.modules' in code[0], path.name
    assert "%pip install" in code[0] and "torchtsetlin" in code[0], path.name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_listed_in_the_docs(path):
    assert f"examples/notebooks/{path.name}" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "examples" / "notebooks.md").read_text(encoding="utf-8")
    assert f"notebooks/{path.name}" in page and COLAB + path.name in page
