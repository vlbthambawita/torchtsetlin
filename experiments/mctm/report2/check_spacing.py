"""Find generated macros followed by a bare space, whose space TeX then swallows.

``\\accFoo bar`` typesets as ``12.3bar``: the space terminates the control word and is not
printed. The fix is ``\\accFoo{} bar`` or ``\\accFoo\\ bar``, and the failure is invisible in
the log, so it has to be checked for.
"""
from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "macros.tex")) as f:
    defined = set(re.findall(r"\\newcommand\\([A-Za-z]+)\{", f.read()))
bad = []
for path in sorted(glob.glob(os.path.join(HERE, "sections", "*.tex"))):
    with open(path) as f:
        for i, line in enumerate(f, 1):
            for m in re.finditer(r"\\([A-Za-z]+) ", line):
                if m.group(1) in defined:
                    bad.append(f"{os.path.basename(path)}:{i}  \\{m.group(1)}")
if bad:
    print(f"FAIL: {len(bad)} macro(s) followed by a bare space (it will be swallowed):")
    for b in bad:
        print(f"    {b}")
sys.exit(1 if bad else 0)
