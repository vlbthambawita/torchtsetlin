"""Report which generated macros the text uses but the data has not filled in.

``report_data_ideas.py`` defines a placeholder ``n/a`` for every macro an arm *would* have, so
that a partial run builds instead of dropping the number silently. That is the right default
while runs are in flight and the wrong one for a finished report, where ``n/a%`` in the middle
of a sentence is worse than a build failure. This is the check that catches it.

    python audit_macros.py            # list them, exit 0
    STRICT=1 python audit_macros.py   # exit 1 if any are still unfilled
"""
from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    with open(os.path.join(HERE, "macros.tex")) as f:
        defs = dict(re.findall(r"\\newcommand\\([A-Za-z]+)\{(.*)\}", f.read()))
    used = set()
    for path in glob.glob(os.path.join(HERE, "sections", "*.tex")) + [
        os.path.join(HERE, "main.tex")
    ]:
        with open(path) as f:
            used |= {m for m in re.findall(r"\\([A-Za-z]+)", f.read()) if m in defs}
    pending = sorted(m for m in used if defs[m] == "n/a")
    if not pending:
        print(f"OK: all {len(used)} generated macros used in the text are filled")
        return 0
    print(f"PENDING: {len(pending)} of {len(used)} used macros are still n/a:")
    for m in pending:
        print(f"    \\{m}")
    return 1 if os.environ.get("STRICT") else 0


if __name__ == "__main__":
    sys.exit(main())
