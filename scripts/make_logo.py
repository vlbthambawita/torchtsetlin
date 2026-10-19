"""Generate the docs logo and favicon from the white paper's automata-grid motif.

The cells follow ``whitepaper/sections/00-titlepage.tex`` exactly:
``v = mod(r*7 + c*5 + int(r*c/2), 9)``; ``v > 5.5`` is a solid cell, ``v > 3`` a 40%-opacity
cell, anything else a 28%-opacity outline. The icon is its top-left 4x4 corner. Colours are the
white paper's ``ttorange`` (#E8590C) from ``ttwhitepaper.sty``.

Run:  python scripts/make_logo.py   (writes docs/assets/logo.svg and docs/assets/favicon.svg)
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
ORANGE = "#E8590C"
N = 4
CELL, GAP, RX = 8.0, 2.0, 1.4


def cells(colour):
    out = []
    for r in range(N):
        for c in range(N):
            v = (r * 7 + c * 5 + int(r * c / 2)) % 9
            x, y = c * (CELL + GAP), r * (CELL + GAP)
            box = f'x="{x:g}" y="{y:g}" width="{CELL:g}" height="{CELL:g}" rx="{RX:g}"'
            if v > 5.5:
                out.append(f'<rect {box} fill="{colour}"/>')
            elif v > 3.0:
                out.append(f'<rect {box} fill="{colour}" fill-opacity="0.4"/>')
            else:
                # outline drawn inside the cell so every cell has the same footprint
                out.append(f'<rect x="{x + 0.6:g}" y="{y + 0.6:g}" width="{CELL - 1.2:g}" '
                           f'height="{CELL - 1.2:g}" rx="{RX - 0.4:g}" fill="none" stroke="{colour}" '
                           f'stroke-opacity="0.55" stroke-width="1.2"/>')
    return "\n  ".join(out)


def main():
    side = N * CELL + (N - 1) * GAP                      # 38
    logo = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side:g} {side:g}" '
            f'role="img" aria-label="torchtsetlin">\n  {cells("#FFFFFF")}\n</svg>\n')
    pad = 7.0
    full = side + 2 * pad
    favicon = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {full:g} {full:g}" '
               f'role="img" aria-label="torchtsetlin">\n'
               f'  <rect width="{full:g}" height="{full:g}" rx="9" fill="{ORANGE}"/>\n'
               f'  <g transform="translate({pad:g} {pad:g})">\n  {cells("#FFFFFF")}\n  </g>\n</svg>\n')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logo.svg").write_text(logo)
    (OUT / "favicon.svg").write_text(favicon)
    print("wrote", OUT / "logo.svg", OUT / "favicon.svg")


if __name__ == "__main__":
    main()
