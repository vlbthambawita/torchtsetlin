"""Build torchtsetlin-arxiv.zip: the summary report as two flat files, main.tex + ttwhitepaper.sty.

arXiv (and any upload that loses sub-directories) cannot be relied on to keep ``figures/`` and
``data/``, so the package is flattened: every ``\\input{figures/...}`` and ``\\input{data/...}`` in
summary.tex is replaced by the file's content, and every ``\\pgfplotstableread{data/x.csv}\\name``
becomes an inline table. The numbers still come from the generated CSVs; nothing is retyped.

Run:  make arxiv     (or: python arxiv_pack.py)
"""

import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "torchtsetlin-arxiv.zip"
TABLE = re.compile(r"\\pgfplotstableread\[col sep=comma\]\{(data/[^}]+\.csv)\}(\\[A-Za-z]+)")
INPUT = re.compile(r"\\input\{((?:figures|data)/[^}]+)\}")


def inline_table(match):
    rows = [r.strip() for r in (HERE / match.group(1)).read_text().splitlines() if r.strip()]
    body = "\\\\\n".join(rows) + "\\\\"
    return (f"% inlined from {match.group(1)}\n"
            f"\\pgfplotstableread[col sep=comma,row sep=\\\\]{{\n{body}\n}}{match.group(2)}")


def expand(text):
    def inp(match):
        name = match.group(1)
        path = HERE / (name if name.endswith(".tex") else name + ".tex")
        return f"% ---- begin {path.name}\n{expand(path.read_text()).rstrip()}\n% ---- end {path.name}\n"
    return INPUT.sub(inp, TABLE.sub(inline_table, text))


def main():
    tex = expand((HERE / "summary.tex").read_text())
    left = INPUT.findall(tex) + TABLE.findall(tex)
    if left:
        raise SystemExit(f"unexpanded inputs remain: {left}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("main.tex", tex)
        z.write(HERE / "ttwhitepaper.sty", "ttwhitepaper.sty")
    print(f"wrote {OUT.name}: main.tex ({len(tex):,} chars), ttwhitepaper.sty")


if __name__ == "__main__":
    main()
