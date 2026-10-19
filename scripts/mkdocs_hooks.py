"""MkDocs hooks: the package version in the docs, and the white paper PDFs in the site.

``config.extra.version`` feeds the announcement bar (``overrides/main.html``), and
``%%version%%`` in any page is replaced with it. The version is parsed from
``src/torchtsetlin/__init__.py`` (no import, so the docs build does not need torch).

After the build, the committed PDFs in ``whitepaper/`` are copied to
``assets/papers/`` (docs/cite.md links them), so the site never carries a stale duplicate.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "torchtsetlin" / "__init__.py"
TOKEN = "%%version%%"
PAPERS = {"torchtsetlin-report.pdf": ROOT / "whitepaper" / "summary.pdf",
          "torchtsetlin-whitepaper.pdf": ROOT / "whitepaper" / "main.pdf"}


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


def on_post_build(config, **kwargs):
    out = Path(config["site_dir"]) / "assets" / "papers"
    out.mkdir(parents=True, exist_ok=True)
    for name, src in PAPERS.items():
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing; build it with make in whitepaper/")
        shutil.copyfile(src, out / name)
