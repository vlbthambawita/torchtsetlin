#!/usr/bin/env python
"""Extract text from the local PDF bibliography, with optional grep.

    python tools/pdf_text.py <pdf>                       # whole document
    python tools/pdf_text.py <pdf> --pages 1-4           # a page range
    python tools/pdf_text.py <pdf> --grep "CIFAR-10"     # matching lines + context
    python tools/pdf_text.py --all --grep "CIFAR-10"     # across the whole bibliography

Reading a 20-page paper into an agent's context costs a lot for very little; --grep first,
then --pages on what matters.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

PAPERS = "/work/vajira/DL2026/torchtsetlin/source_documents"


def text_of(path: str, pages: str | None = None) -> str:
    import pymupdf

    doc = pymupdf.open(path)
    if pages:
        a, _, b = pages.partition("-")
        lo, hi = int(a) - 1, int(b or a)
        rng = range(max(0, lo), min(len(doc), hi))
    else:
        rng = range(len(doc))
    return "\n".join(f"[p{i+1}] " + doc[i].get_text() for i in rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?")
    ap.add_argument("--all", action="store_true", help="every PDF under source_documents/")
    ap.add_argument("--pages", help="e.g. 1-4")
    ap.add_argument("--grep", help="regex; prints matching lines with 1 line of context")
    ap.add_argument("--context", type=int, default=1)
    a = ap.parse_args()

    files = (sorted(glob.glob(os.path.join(PAPERS, "**", "*.pdf"), recursive=True))
             if a.all else [a.pdf])
    if not files or files == [None]:
        sys.exit("give a pdf path or --all")

    for f in files:
        try:
            txt = text_of(f, a.pages)
        except Exception as e:                      # a corrupt or image-only PDF must not stop a sweep
            print(f"!! {os.path.basename(f)}: {e}", file=sys.stderr)
            continue
        if not a.grep:
            print(f"===== {f}\n{txt}")
            continue
        lines = txt.split("\n")
        rx = re.compile(a.grep, re.I)
        hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
        if not hits:
            continue
        print(f"===== {os.path.basename(f)}  ({len(hits)} hits)")
        shown: set[int] = set()
        for i in hits:
            for j in range(max(0, i - a.context), min(len(lines), i + a.context + 1)):
                if j not in shown:
                    shown.add(j)
                    print(f"  {lines[j].strip()}")
            print("  --")


if __name__ == "__main__":
    main()
