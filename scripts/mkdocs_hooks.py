"""MkDocs hooks: expose the package version to the docs, so it is never typed by hand.

``config.extra.version`` feeds the announcement bar (``overrides/main.html``), and
``%%version%%`` in any page is replaced with it. The version is parsed from
``src/torchtsetlin/__init__.py`` (no import, so the docs build does not need torch).
"""

import re
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "src" / "torchtsetlin" / "__init__.py"
TOKEN = "%%version%%"


def _version():
    m = re.search(r'^__version__ = "([^"]+)"', INIT.read_text(), re.M)
    if not m:
        raise RuntimeError(f"no __version__ in {INIT}")
    return m.group(1)


def on_config(config, **kwargs):
    config.extra["version"] = _version()
    return config


def on_page_markdown(markdown, page, config, **kwargs):
    return markdown.replace(TOKEN, config.extra["version"])
