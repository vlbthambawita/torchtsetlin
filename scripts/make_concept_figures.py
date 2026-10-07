#!/usr/bin/env python
"""Draw the explanatory figures used on the ``docs/concepts/`` pages.

Every figure is a self-contained SVG, written twice — ``<name>.svg`` for the light and
``<name>_dark.svg`` for the dark documentation theme, which MkDocs Material switches between
with the ``#only-light`` / ``#only-dark`` image suffixes (the same convention as the
benchmark figures).

Animated figures use CSS keyframes inside the SVG, so they play in an ordinary ``<img>``
without JavaScript. The un-animated state of every element is the *finished* picture, so
the figure still reads correctly when a viewer has ``prefers-reduced-motion`` set (which
switches the animation off) or when it is rendered by a tool that ignores CSS animation.

Usage::

    python scripts/make_concept_figures.py            # writes docs/assets/concepts/*.svg
    python scripts/make_concept_figures.py --out DIR
"""

from __future__ import annotations

import argparse
import os
from html import escape
from typing import Callable, Dict, List, Optional, Sequence, Tuple

THEMES: Dict[str, Dict[str, str]] = {
    "light": dict(
        fg="#1f2328",
        muted="#646b78",
        line="#cfd3db",
        panel="#f6f4fb",
        panel2="#eceef2",
        accent="#5e35b1",
        accent_soft="#e6def6",
        pos="#2e7d32",
        pos_soft="#d9eed9",
        neg="#c62828",
        neg_soft="#f7dada",
        amber="#b26a00",
        amber_soft="#ffecc7",
        blue="#1d63c4",
        blue_soft="#d8e6fa",
        on_accent="#ffffff",
    ),
    "dark": dict(
        fg="#e4e6eb",
        muted="#a0a7b4",
        line="#4a505e",
        panel="#272b37",
        panel2="#2f3442",
        accent="#b39ddb",
        accent_soft="#3b3253",
        pos="#74c378",
        pos_soft="#20402a",
        neg="#f07878",
        neg_soft="#4c2528",
        amber="#ffc552",
        amber_soft="#4a3a16",
        blue="#86b7f7",
        blue_soft="#1e3352",
        on_accent="#1b1530",
    ),
}

# Natural-looking "photo" colours and flat label colours for the segmentation scenes; they
# carry meaning on their own, so they are the same in both themes.
PHOTO = {"S": "#bfdcf4", "B": "#9b9ca3", "W": "#61666e", "T": "#4f9a45", "R": "#4a4e56", "L": "#d9d9d9"}
LABEL = {"S": "#4ea1e8", "B": "#e2774c", "T": "#2f9e44", "R": "#8f949d"}
LABEL_NAME = {"S": "sky", "B": "building", "T": "tree", "R": "road"}

HIDDEN = 'opacity="0"'  # static state of an element that only appears mid-animation

FONT = "Roboto, 'Helvetica Neue', Helvetica, Arial, sans-serif"

Frames = List[Tuple[float, str]]


# --------------------------------------------------------------------------------- canvas


class Fig:
    """A tiny SVG builder with CSS-keyframe animation helpers."""

    def __init__(self, w: int, h: int, c: Dict[str, str], title: str, desc: str,
                 duration: Optional[float] = None):
        self.w, self.h, self.c = w, h, c
        self.title, self.desc, self.duration = title, desc, duration
        self.body: List[str] = []
        self.css: List[str] = []
        self.n_anim = 0

    # -- primitives ------------------------------------------------------------------
    def add(self, s: str) -> None:
        self.body.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", rx=6, sw=1.5, extra="") -> str:
        return (f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" {extra}/>')

    def text(self, x, y, s, size=13, fill=None, anchor="middle", weight="normal",
             extra="", italic=False) -> str:
        fill = fill or self.c["fg"]
        style = ' font-style="italic"' if italic else ""
        return (f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" fill="{fill}" '
                f'text-anchor="{anchor}" font-weight="{weight}"{style} {extra}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke, sw=1.5, extra="") -> str:
        return (f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{stroke}" '
                f'stroke-width="{sw:g}" stroke-linecap="round" {extra}/>')

    def arrow(self, x1, y1, x2, y2, stroke, sw=1.8, head=7, extra="") -> str:
        import math

        a = math.atan2(y2 - y1, x2 - x1)
        hx1 = x2 - head * math.cos(a - 0.45)
        hy1 = y2 - head * math.sin(a - 0.45)
        hx2 = x2 - head * math.cos(a + 0.45)
        hy2 = y2 - head * math.sin(a + 0.45)
        sx2, sy2 = x2 - 0.6 * head * math.cos(a), y2 - 0.6 * head * math.sin(a)
        return (f'<g {extra}>{self.line(x1, y1, sx2, sy2, stroke, sw)}'
                f'<polygon points="{x2:g},{y2:g} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" '
                f'fill="{stroke}"/></g>')

    def check(self, x, y, size, color, sw=2.4) -> str:
        """A check mark centred on (x, y)."""
        s = size / 2
        return (f'<polyline points="{x - s:.1f},{y:.1f} {x - s * 0.25:.1f},{y + s * 0.7:.1f} '
                f'{x + s:.1f},{y - s * 0.7:.1f}" fill="none" stroke="{color}" '
                f'stroke-width="{sw:g}" stroke-linecap="round" stroke-linejoin="round"/>')

    def cross(self, x, y, size, color, sw=2.4) -> str:
        s = size / 2 * 0.8
        return (self.line(x - s, y - s, x + s, y + s, color, sw)
                + self.line(x - s, y + s, x + s, y - s, color, sw))

    def chip(self, x, y, w, h, label, kind="neutral", size=12.5, icon=True) -> str:
        """A literal / fact pill. ``kind``: true, false, included, excluded, neutral."""
        c = self.c
        if kind == "true":
            fill, stroke, tc = c["pos_soft"], c["pos"], c["fg"]
        elif kind == "false":
            fill, stroke, tc = c["panel2"], c["line"], c["muted"]
        elif kind == "included":
            fill, stroke, tc = c["accent_soft"], c["accent"], c["fg"]
        elif kind == "excluded":
            fill, stroke, tc = "none", c["line"], c["muted"]
        else:
            fill, stroke, tc = c["panel2"], c["line"], c["fg"]
        dash = ' stroke-dasharray="4 3"' if kind == "excluded" else ""
        out = self.rect(x, y, w, h, fill, stroke, rx=h / 2, extra=dash)
        cx = x + w / 2
        if icon and kind in ("true", "false"):
            ix = x + 13
            out += (self.check(ix, y + h / 2, 9, c["pos"], 2)
                    if kind == "true" else self.cross(ix, y + h / 2, 9, c["muted"], 1.8))
            cx = x + 8 + w / 2
        out += self.text(cx, y + h / 2 + size * 0.36, label, size, tc)
        return out

    # -- animation -------------------------------------------------------------------
    def anim(self, frames: Frames, timing: str = "linear") -> str:
        """Register keyframes; return ``class="..."`` for the element to animate."""
        assert self.duration, "figure has no animation duration"
        self.n_anim += 1
        name = f"k{self.n_anim}"
        frames = sorted(frames, key=lambda f: f[0])
        kf = " ".join(f"{p:.3f}%{{{css}}}" for p, css in frames)
        self.css.append(f"@keyframes {name}{{{kf}}}")
        self.css.append(f".{name}{{animation:{name} {self.duration:g}s {timing} infinite}}")
        return f'class="anim {name}"'

    # -- output ----------------------------------------------------------------------
    def svg(self) -> str:
        style = ""
        if self.css:
            style = ("<style>" + "\n".join(self.css)
                     + "\n@media (prefers-reduced-motion: reduce){.anim{animation:none!important}}"
                     + "</style>")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img" font-family="{FONT}">'
                f"<title>{escape(self.title)}</title><desc>{escape(self.desc)}</desc>"
                f"{style}" + "\n".join(self.body) + "</svg>\n")


def steps(segments: Sequence[Tuple[float, str]]) -> Frames:
    """Piecewise-constant keyframes: ``[(start_pct, css), ...]`` with the first at 0."""
    starts = [t for t, _ in segments]
    assert starts[0] == 0 and all(a < b for a, b in zip(starts, starts[1:])), starts
    out: Frames = []
    for i, (t, css) in enumerate(segments):
        end = segments[i + 1][0] - 0.01 if i + 1 < len(segments) else 100.0
        out += [(t, css), (end, css)]
    return out


def moves(start: str, changes: Sequence[Tuple[float, str]], d: float) -> Frames:
    """Hold ``start``; at each ``(t, css)`` glide to the new value over ``d`` percent."""
    out: Frames = [(0.0, start)]
    prev = start
    for t, css in changes:
        out += [(t, prev), (min(t + d, 100.0), css)]
        prev = css
    if out[-1][0] < 100.0:
        out.append((100.0, prev))
    return out


def shown(windows: Sequence[Tuple[float, float]], on: str = "opacity:1",
          off: str = "opacity:0") -> Frames:
    """Visible during each ``(start, end)`` window, hidden otherwise."""
    merged: List[List[float]] = []
    for a, b in sorted(windows):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    segs: List[Tuple[float, str]] = [(0.0, off)]
    for a, b in merged:
        segs += [(a, on), (b, off)]
    if segs[1][0] == 0.0:
        segs = segs[1:]
    if segs[-1][0] >= 100.0:
        segs = segs[:-1]
    return steps(segs)


def tx(dx: float, dy: float = 0.0) -> str:
    return f"transform:translate({dx:.1f}px,{dy:.1f}px)"


# --------------------------------------------------------------------------------- figures


def fig_tm_clauses(c: Dict[str, str]) -> Fig:
    f = Fig(790, 430, c, "Clauses vote on an example",
            "A day that is sunny, windy and a weekend is turned into literals, four clauses check "
            "whether all their literals are true, and their votes are added up.", duration=13)
    lab_x, x0 = 20, 146

    def section(y, title, sub):
        f.add(f.text(lab_x, y, title, 13.5, weight="bold", anchor="start"))
        f.add(f.text(lab_x, y + 16, sub, 11, c["muted"], anchor="start"))

    # 1. facts
    section(40, "1. The day", "the input")
    for i, (name, val) in enumerate([("Sunny", True), ("Windy", True), ("Weekend", True)]):
        f.add(f.chip(x0 + i * 140, 26, 128, 30, f"{name}: {'yes' if val else 'no'}",
                     "true" if val else "false"))
    # 2. literals
    section(106, "2. Literals", "facts and opposites")
    lits = [("Sunny", True), ("NOT Sunny", False), ("Windy", True), ("NOT Windy", False),
            ("Weekend", True), ("NOT Weekend", False)]
    for i, (name, val) in enumerate(lits):
        f.add(f.chip(x0 + i * 107, 92, 104, 30, name, "true" if val else "false", 11))
    # 3. clauses
    section(172, "3. Clauses", "ANDs of literals")
    clauses = [
        (+1, [("Sunny", True), ("Weekend", True)], True),
        (+1, [("Weekend", True)], True),
        (-1, [("Windy", True)], True),
        (-1, [("NOT Sunny", False)], False),
    ]
    total_after: List[int] = []
    running = 0
    t0, dt = 10.0, 16.0
    for k, (pol, lits_k, match) in enumerate(clauses):
        y = 160 + k * 46
        t = t0 + k * dt
        f.add(f.rect(x0 - 8, y - 5, 640, 42, c["accent_soft"], rx=8,
                     extra=f'opacity="0" {f.anim(shown([(t, t + dt - 2)]))}'))
        badge_fill, badge_c = (c["pos_soft"], c["pos"]) if pol > 0 else (c["neg_soft"], c["neg"])
        f.add(f.rect(x0, y + 2, 112, 26, badge_fill, badge_c, rx=5))
        f.add(f.text(x0 + 56, y + 19.5, "votes FOR" if pol > 0 else "votes AGAINST", 11.5,
                     badge_c, weight="bold"))
        x = x0 + 128
        for j, (name, val) in enumerate(lits_k):
            if j:
                f.add(f.text(x + 16, y + 20, "AND", 11, c["muted"], weight="bold"))
                x += 34
            f.add(f.chip(x, y, 98, 30, name, "true" if val else "false", 11.5))
            x += 98
        # result + vote, revealed when the clause is evaluated
        appear = f.anim(steps([(0, "opacity:0"), (t + 5, "opacity:1"), (97, "opacity:0")]))
        rx_ = x0 + 470
        res = (f.check(rx_, y + 15, 14, c["pos"]) + f.text(rx_ + 14, y + 20, "all true", 12,
                                                           c["pos"], anchor="start")
               if match else
               f.cross(rx_, y + 15, 13, c["muted"]) + f.text(rx_ + 14, y + 20, "not all true", 12,
                                                             c["muted"], anchor="start"))
        vote = pol if match else 0
        running += vote
        total_after.append(running)
        vtxt = {1: "+1", -1: "−1", 0: "0"}[vote]
        vcol = {1: c["pos"], -1: c["neg"], 0: c["muted"]}[vote]
        f.add(f"<g {appear}>{res}{f.text(x0 + 610, y + 21, vtxt, 16, vcol, weight='bold')}</g>")
    # 4. votes
    y = 372
    section(y + 8, "4. Votes", "for − against")
    f.add(f.rect(x0, y - 12, 636, 56, c["panel"], c["line"], rx=10))
    f.add(f.text(x0 + 18, y + 22, "Total:", 15, anchor="start", weight="bold"))
    values = [0] + total_after
    times = [0.0] + [t0 + k * dt + 5 for k in range(len(clauses))]
    for i, v in enumerate(values):
        last = i == len(values) - 1
        segs = [(0.0, "opacity:0"), (times[i], "opacity:1")]
        if not last:
            segs.append((times[i + 1], "opacity:0"))
        else:
            segs.append((97.0, "opacity:0"))
        if i == 0:
            segs = [(0.0, "opacity:1"), (times[1], "opacity:0")]
        s = f"{v:+d}" if v else "0"
        f.add(f.text(x0 + 100, y + 23, s, 22, c["accent"], weight="bold",
                     extra=f'{"" if last else HIDDEN} {f.anim(steps(segs))}'))
    verdict = f.anim(steps([(0, "opacity:0"), (78, "opacity:1"), (97, "opacity:0")]))
    f.add(f"<g {verdict}>{f.text(x0 + 150, y + 22, '→  more votes for than against:', 13.5, anchor='start')}"
          f"{f.rect(x0 + 420, y + 1, 190, 30, c['pos_soft'], c['pos'], rx=15)}"
          f"{f.text(x0 + 515, y + 21, 'a beach day!', 14, c['pos'], weight='bold')}</g>")
    return f


def fig_tm_automaton(c: Dict[str, str]) -> Fig:
    f = Fig(760, 280, c, "A Tsetlin automaton",
            "A token on a track of eight positions decides whether the literal Sunny is in the "
            "clause. Rewards push it right towards remembering, penalties push it left towards "
            "forgetting; it only changes the clause when it crosses the middle.", duration=11)
    x0, cw, y0, h = 60, 80, 62, 58
    n = 8
    for i in range(n):
        fill = c["neg_soft"] if i < n // 2 else c["pos_soft"]
        f.add(f.rect(x0 + i * cw + 2, y0, cw - 4, h, fill, rx=8))
    mid = x0 + n // 2 * cw
    f.add(f.line(mid, y0 - 12, mid, y0 + h + 12, c["accent"], 2.5, 'stroke-dasharray="6 4"'))
    f.add(f.text(x0 + 2 * cw, y0 - 18, "FORGET  ·  literal left out of the clause", 13,
                 c["neg"], weight="bold"))
    f.add(f.text(x0 + 6 * cw, y0 - 18, "REMEMBER  ·  literal included", 13, c["pos"],
                 weight="bold"))
    f.add(f.text(x0 + cw / 2, y0 + h + 20, "firmly forgotten", 11.5, c["muted"]))
    f.add(f.text(mid, y0 + h + 30, "on the fence", 11.5, c["muted"]))
    f.add(f.text(x0 + (n - 0.5) * cw, y0 + h + 20, "firmly remembered", 11.5, c["muted"]))

    seq = [(7, 4), (19, 5), (31, 6), (46, 5), (58, 4), (70, 3), (82, 2), (93, 3)]
    start, base = 3, 5
    d = 3.5
    pos = moves(tx((start - base) * cw), [(t, tx((s - base) * cw)) for t, s in seq], d)
    cx, cy = x0 + (base + 0.5) * cw, y0 + h / 2
    f.add(f'<g transform="translate({cx},{cy})"><g {f.anim(pos, "ease-in-out")}>'
          f'<circle r="25" fill="{c["accent"]}"/>'
          f'{f.text(0, 4.5, "Sunny", 12.5, c["on_accent"], weight="bold")}</g></g>')
    # reward / penalty legend, lit while used
    ups = [t for (t, s), prev in zip(seq, [start] + [s for _, s in seq]) if s > prev]
    downs = [t for (t, s), prev in zip(seq, [start] + [s for _, s in seq]) if s < prev]
    ly = 196
    lit = lambda ts: f.anim(shown([(t, t + 7) for t in ts], "opacity:1", "opacity:.3"))  # noqa: E731
    f.add(f'<g {lit(downs)}>{f.arrow(330, ly, 110, ly, c["neg"], 2.5, 10)}'
          f'{f.text(220, ly - 9, "penalty: one step towards forgetting", 12.5, c["neg"])}</g>')
    f.add(f'<g {lit(ups)}>{f.arrow(430, ly, 650, ly, c["pos"], 2.5, 10)}'
          f'{f.text(540, ly - 9, "reward: one step towards remembering", 12.5, c["pos"])}</g>')
    # the clause the automaton controls
    y = 228
    f.add(f.text(200, y + 20, "The clause right now:", 13, anchor="end", weight="bold"))
    included_at = [(0, False)] + [(t + d / 2, s >= n // 2) for t, s in seq]
    on_segs = [(t, "opacity:1" if inc else "opacity:0") for t, inc in included_at]
    off_segs = [(t, "opacity:0" if inc else "opacity:1") for t, inc in included_at]
    f.add(f'<g {f.anim(steps(on_segs))}>{f.chip(212, y, 86, 30, "Sunny", "included")}</g>')
    f.add(f'<g opacity="0" {f.anim(steps(off_segs))}>'
          f'{f.chip(212, y, 86, 30, "Sunny", "excluded")}</g>')
    f.add(f.text(318, y + 20, "AND", 11.5, c["muted"], weight="bold"))
    f.add(f.chip(340, y, 100, 30, "Weekend", "included"))
    f.add(f.text(456, y + 20, "← Sunny is in only while the token is on the right",
                 11.5, c["muted"], anchor="start"))
    return f


def fig_tm_feedback(c: Dict[str, str]) -> Fig:
    f = Fig(760, 420, c, "Type I and Type II feedback",
            "Two panels. Type I: an example of the clause's own class pushes its true literals "
            "towards remembering, so the clause learns Weekend. Type II: a look-alike example "
            "from another class pushes the false literals towards remembering, so the clause "
            "stops matching it.", duration=9)
    cw, nst = 24, 6
    t_move, d = 22.0, 8.0
    panels = [
        dict(title="Type I  ·  learn the pattern",
             who="An example of the clause's own class:",
             ex="a sunny weekend — a beach day",
             rows=[("Sunny", True, 3, 4, "true → +1", "remember"),
                   ("NOT Sunny", False, 1, 0, "false → −1", "forget (rarely)"),
                   ("Weekend", True, 2, 3, "true → +1", "now included!"),
                   ("NOT Weekend", False, 1, 0, "false → −1", "forget (rarely)")],
             res="still matches", res_sub="it has learned “weekend”"),
        dict(title="Type II  ·  reject a look-alike",
             who="An example of a different class:",
             ex="a sunny weekday — not a beach day",
             rows=[("Sunny", True, 3, 3, "true", "no change"),
                   ("NOT Sunny", False, 0, 1, "false → +1", "still left out"),
                   ("Weekend", False, 2, 3, "false → +1", "now included!"),
                   ("NOT Weekend", True, 1, 1, "true", "no change")],
             res="no longer matches", res_sub="the weekday is rejected"),
    ]
    for p, P in enumerate(panels):
        px = 10 + p * 375
        f.add(f.rect(px, 8, 365, 404, c["panel"], c["line"], rx=12))
        f.add(f.text(px + 16, 34, P["title"], 15, anchor="start", weight="bold"))
        f.add(f.text(px + 16, 56, P["who"], 12, c["muted"], anchor="start"))
        f.add(f.text(px + 16, 74, P["ex"], 13, anchor="start", italic=True))
        f.add(f.text(px + 16, 102, "Before:", 12.5, anchor="start", weight="bold"))
        f.add(f.chip(px + 72, 86, 76, 26, "Sunny", "included", 12))
        f.add(f.check(px + 166, 99, 12, c["pos"]))
        f.add(f.text(px + 176, 103.5, "matches this example", 12, c["pos"], anchor="start"))
        tx0 = px + 120
        for r, (name, val, s0, s1, n1, n2) in enumerate(P["rows"]):
            y = 128 + r * 52
            f.add(f.text(px + 16, y + 15, name, 12.5, anchor="start", weight="bold"))
            f.add(f.text(px + 16, y + 31, "true here" if val else "false here", 11,
                         c["pos"] if val else c["muted"], anchor="start"))
            for i in range(nst):
                fill = c["neg_soft"] if i < nst // 2 else c["pos_soft"]
                f.add(f.rect(tx0 + i * cw + 1, y + 4, cw - 2, 26, fill, rx=4))
            f.add(f.line(tx0 + 3 * cw, y, tx0 + 3 * cw, y + 34, c["accent"], 1.8,
                         'stroke-dasharray="3 3"'))
            cy = y + 17
            if s0 != s1:
                gx = tx0 + (s0 + 0.5) * cw
                f.add(f'<circle cx="{gx}" cy="{cy}" r="8" fill="none" stroke="{c["muted"]}" '
                      f'stroke-width="1.3" stroke-dasharray="2.5 2"/>')
                ex = tx0 + (s1 + 0.5) * cw
                f.add(f.arrow(gx, y - 3, ex, y - 3, c["pos"] if s1 > s0 else c["neg"], 1.6, 5))
            fx = tx0 + (s1 + 0.5) * cw
            anim = f.anim(moves(tx((s0 - s1) * cw), [(t_move, tx(0)), (96, tx((s0 - s1) * cw))], d),
                          "ease-in-out")
            f.add(f'<g transform="translate({fx},{cy})"><g {anim}>'
                  f'<circle r="8.5" fill="{c["accent"]}"/></g></g>')
            hot = s1 >= 3 > s0
            ncol = c["accent"] if hot else (c["pos"] if s1 > s0 else c["neg"] if s1 < s0 else c["muted"])
            f.add(f.text(px + 272, y + 14, n1, 11, ncol, anchor="start",
                         weight="bold" if hot else "normal"))
            f.add(f.text(px + 272, y + 28, n2, 11, ncol, anchor="start",
                         weight="bold" if hot else "normal"))
        # after
        after = f.anim(steps([(0, "opacity:0"), (42, "opacity:1"), (96, "opacity:0")]))
        y = 345
        f.add(f'<g {after}>'
              f'{f.text(px + 16, y + 17, "After:", 12.5, anchor="start", weight="bold")}'
              f'{f.chip(px + 72, y, 76, 26, "Sunny", "included", 12)}'
              f'{f.text(px + 166, y + 17.5, "AND", 11, c["muted"], weight="bold")}'
              f'{f.chip(px + 184, y, 92, 26, "Weekend", "included", 12)}'
              f'{f.check(px + 26, y + 46, 12, c["pos"])}'
              f'{f.text(px + 38, y + 50.5, P["res"] + " — " + P["res_sub"], 12, c["pos"], anchor="start")}'
              f'</g>')
    return f


def fig_tm_vote_margin(c: Dict[str, str]) -> Fig:
    f = Fig(760, 310, c, "The vote margin T",
            "Chance that an example trains a class's clauses, against that class's vote sum. "
            "For the correct class it falls from always at minus T to never at plus T; for a "
            "wrong class it rises.", duration=10)
    X0, X1, Y0, Y1 = 110, 470, 236, 46
    f.add(f.rect(X0, Y1, X1 - X0, Y0 - Y1, c["panel"], rx=0))
    for yy, lab in [(Y0, "never"), ((Y0 + Y1) / 2, "half"), (Y1, "always")]:
        f.add(f.line(X0, yy, X1, yy, c["line"], 1))
        f.add(f.text(X0 - 8, yy + 4, lab, 11.5, c["muted"], anchor="end"))
    for xx, lab, sub in [(X0, "−T", "clearly says no"), ((X0 + X1) / 2, "0", "undecided"),
                         (X1, "+T", "clearly says yes")]:
        f.add(f.line(xx, Y0, xx, Y0 + 5, c["muted"], 1.2))
        f.add(f.text(xx, Y0 + 20, lab, 13, weight="bold"))
        f.add(f.text(xx, Y0 + 35, sub, 11, c["muted"]))
    f.add(f.text(44, (Y0 + Y1) / 2, "chance of feedback", 12, c["muted"],
                 extra=f'transform="rotate(-90 44 {(Y0 + Y1) / 2})"'))
    f.add(f.text((X0 + X1) / 2, Y0 + 58, "the class's vote sum (clipped to −T … +T)", 12,
                 c["muted"]))
    f.add(f.line(X0, Y1, X1, Y0, c["pos"], 3.5))
    f.add(f.line(X0, Y0, X1, Y1, c["neg"], 3, 'stroke-dasharray="8 5"'))
    # moving dot along the correct-class line
    frac0 = 0.75
    bx, by = X0 + frac0 * (X1 - X0), Y1 + frac0 * (Y0 - Y1)
    pts = [(0.0, 0.08), (80.0, 1.0), (92.0, 1.0), (100.0, 0.08)]
    frames = [(t, tx((fr - frac0) * (X1 - X0), (fr - frac0) * (Y0 - Y1))) for t, fr in pts]
    f.add(f'<g transform="translate({bx:.1f},{by:.1f})"><g {f.anim(frames, "ease-in-out")}>'
          f'<circle r="8" fill="{c["pos"]}" stroke="{c["panel"]}" stroke-width="2.5"/></g></g>')
    # explanations
    tx0 = 500
    f.add(f.line(tx0, 62, tx0 + 26, 62, c["pos"], 3.5))
    lines_pos = ["The example's own class", "The fewer votes it has, the more", "often its clauses are taught.",
                 "Once it wins by T votes, teaching", "stops — those clauses are free to", "learn other examples."]
    for i, s in enumerate(lines_pos):
        f.add(f.text(tx0 + (34 if i == 0 else 0), 66 + i * 17 + (6 if i else 0), s,
                     12.5 if i == 0 else 12, c["pos"] if i == 0 else c["fg"], anchor="start",
                     weight="bold" if i == 0 else "normal"))
    f.add(f.line(tx0, 196, tx0 + 26, 196, c["neg"], 3, 'stroke-dasharray="8 5"'))
    lines_neg = ["A random wrong class", "The more it (wrongly) votes yes,",
                 "the more often its clauses are", "corrected."]
    for i, s in enumerate(lines_neg):
        f.add(f.text(tx0 + (34 if i == 0 else 0), 200 + i * 17 + (6 if i else 0), s,
                     12.5 if i == 0 else 12, c["neg"] if i == 0 else c["fg"], anchor="start",
                     weight="bold" if i == 0 else "normal"))
    return f


def fig_batching(c: Dict[str, str]) -> Fig:
    f = Fig(760, 350, c, "Sequential versus batched feedback",
            "Top: four examples, each followed by its own memory update. Bottom: the same four "
            "examples evaluated together against the same memory, their feedback added up and "
            "applied in one update, finishing much sooner.", duration=10)
    x0, bw, gap = 24, 72, 14
    cur0, cur1 = x0 - 6, 740
    t0, t1 = 4.0, 84.0

    def t_at(x):
        return t0 + (x - cur0) / (cur1 - cur0) * (t1 - t0)

    def reveal(x):
        return f.anim(steps([(0, "opacity:.25"), (t_at(x), "opacity:1"), (97, "opacity:.25")]))

    # sequential lane
    f.add(f.text(x0, 30, "One example at a time", 14, anchor="start", weight="bold"))
    f.add(f.text(x0, 46, "the classical algorithm  ·  feedback_mode=\"sequential\"", 11.5,
                 c["muted"], anchor="start"))
    y = 58
    x = x0
    for k in range(4):
        for kind in ("ex", "up"):
            if kind == "ex":
                body = (f.rect(x, y, bw, 44, c["blue_soft"], c["blue"], rx=7)
                        + f.text(x + bw / 2, y + 27, f"example {k + 1}", 11.5, c["blue"]))
            else:
                body = (f.rect(x, y, bw, 44, c["amber_soft"], c["amber"], rx=7)
                        + f.text(x + bw / 2, y + 19, "update", 11.5, c["amber"], weight="bold")
                        + f.text(x + bw / 2, y + 34, "memory", 11.5, c["amber"], weight="bold"))
            f.add(f"<g {reveal(x)}>{body}</g>")
            if not (k == 3 and kind == "up"):
                f.add(f.arrow(x + bw + 2, y + 22, x + bw + gap - 2, y + 22, c["muted"], 1.4, 5))
            x += bw + gap
    seq_end = x - gap
    # batched lane
    f.add(f.text(x0, 136, "A mini-batch", 14, anchor="start", weight="bold"))
    f.add(f.text(x0, 152, "the default  ·  feedback_mode=\"batch\"", 11.5, c["muted"],
                 anchor="start"))
    y = 164
    for k in range(4):
        f.add(f"<g {reveal(x0)}>{f.rect(x0, y + k * 27, bw, 23, c['blue_soft'], c['blue'], rx=6)}"
              f"{f.text(x0 + bw / 2, y + k * 27 + 16, f'example {k + 1}', 11, c['blue'])}</g>")
    bx = x0 + bw + gap
    f.add(f.arrow(x0 + bw + 2, y + 52, bx - 2, y + 52, c["muted"], 1.4, 5))
    f.add(f"<g {reveal(bx)}>{f.rect(bx, y + 18, 120, 70, c['panel'], c['line'], rx=8)}"
          f"{f.text(bx + 60, y + 42, 'add up all', 12, weight='bold')}"
          f"{f.text(bx + 60, y + 58, 'their feedback', 12, weight='bold')}"
          f"{f.text(bx + 60, y + 76, '(counts per literal)', 10.5, c['muted'])}</g>")
    ux = bx + 120 + gap
    f.add(f.arrow(bx + 122, y + 52, ux - 2, y + 52, c["muted"], 1.4, 5))
    f.add(f"<g {reveal(ux)}>{f.rect(ux, y + 30, bw + 10, 44, c['amber_soft'], c['amber'], rx=7)}"
          f"{f.text(ux + bw / 2 + 5, y + 49, 'one', 11.5, c['amber'], weight='bold')}"
          f"{f.text(ux + bw / 2 + 5, y + 64, 'update', 11.5, c['amber'], weight='bold')}</g>")
    done_x = ux + bw + 10
    f.add(f"<g {reveal(done_x + 6)}>{f.check(done_x + 22, y + 52, 16, c['pos'], 3)}"
          f"{f.text(done_x + 36, y + 57, 'done — on to the next batch', 12.5, c['pos'], anchor='start')}</g>")
    f.add(f.text(x0 + bw + gap, y + 124,
                 "all four see the same memory; they do not see each other's updates",
                 11.5, c["muted"], anchor="start", italic=True))
    # time axis and sweeping cursor
    ay = 318
    f.add(f.arrow(x0, ay, 740, ay, c["muted"], 1.4, 7))
    f.add(f.text(740, ay + 18, "time", 11.5, c["muted"], anchor="end"))
    f.add(f.line(seq_end, 110, seq_end, ay, c["line"], 1, 'stroke-dasharray="3 4"'))
    f.add(f.line(done_x, 300, done_x, ay, c["line"], 1, 'stroke-dasharray="3 4"'))
    frames = [(0, f"opacity:0;{tx(0)}"), (t0, f"opacity:1;{tx(0)}"),
              (t1, f"opacity:1;{tx(cur1 - cur0)}"), (t1 + 0.1, f"opacity:0;{tx(cur1 - cur0)}"),
              (100, f"opacity:0;{tx(0)}")]
    f.add(f'<g opacity="0" {f.anim(frames)}>{f.line(cur0, 40, cur0, ay, c["accent"], 2.2)}</g>')
    return f


def _plus_image() -> List[List[int]]:
    img = [[0] * 8 for _ in range(8)]
    for r, cc in [(3, 5), (4, 4), (4, 5), (4, 6), (5, 5),       # the plus
                  (0, 1), (1, 1), (1, 2), (6, 1), (7, 6), (1, 6), (6, 7)]:  # noise
        img[r][cc] = 1
    return img


PLUS = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]


def fig_conv_sliding(c: Dict[str, str]) -> Fig:
    f = Fig(760, 330, c, "A convolutional clause slides over the image",
            "A three by three clause describing a plus shape is checked at every window "
            "position of an eight by eight image. Positions are marked in a match map; since "
            "the plus is found at one position, the clause is true for the whole image.",
            duration=16)
    img = _plus_image()
    ix, iy, cs = 24, 56, 24
    f.add(f.text(ix + 4 * cs, 36, "The image (8 × 8 pixels)", 13.5, weight="bold"))
    for r in range(8):
        for q in range(8):
            fill = c["fg"] if img[r][q] else c["panel2"]
            f.add(f.rect(ix + q * cs + 1, iy + r * cs + 1, cs - 2, cs - 2, fill, rx=3))
    # clause pattern
    px, py, pc = 268, 90, 30
    f.add(f.text(px + 1.5 * pc, 36, "The clause", 13.5, weight="bold"))
    f.add(f.text(px + 1.5 * pc, 54, "“a plus shape”", 12, c["muted"], italic=True))
    for r in range(3):
        for q in range(3):
            if (r, q) in PLUS:
                f.add(f.rect(px + q * pc + 1, py + r * pc + 1, pc - 2, pc - 2, c["accent"], rx=4))
            else:
                f.add(f.rect(px + q * pc + 2, py + r * pc + 2, pc - 4, pc - 4, "none", c["muted"],
                             rx=4, sw=1.2, extra='stroke-dasharray="3 3"'))
    ly = py + 3 * pc + 24
    f.add(f.rect(px - 18, ly - 10, 12, 12, c["accent"], rx=2))
    f.add(f.text(px - 1, ly, "must be on", 11.5, anchor="start"))
    f.add(f.rect(px - 18, ly + 9, 12, 12, "none", c["muted"], rx=2, sw=1.2,
                 extra='stroke-dasharray="3 3"'))
    f.add(f.text(px - 1, ly + 19, "doesn't matter", 11.5, anchor="start"))
    # match map
    mx, my = 424, 56
    f.add(f.text(mx + 3 * cs, 36, "Where it matches", 13.5, weight="bold"))
    n = 6
    order = [(r, q) for r in range(n) for q in range(n)]
    s0, s1 = 2.0, 80.0
    dt = (s1 - s0) / len(order)
    match_pos = None
    for k, (r, q) in enumerate(order):
        hit = all(img[r + a][q + b] for a, b in PLUS)
        if hit:
            match_pos = (r, q)
        t = s0 + k * dt
        f.add(f.rect(mx + q * cs + 1, my + r * cs + 1, cs - 2, cs - 2, c["panel2"], c["line"],
                     rx=3, sw=1))
        fill = c["accent"] if hit else c["line"]
        inner = f.rect(mx + q * cs + 1, my + r * cs + 1, cs - 2, cs - 2, fill, rx=3)
        if hit:
            inner += f.check(mx + (q + 0.5) * cs, my + (r + 0.5) * cs, 12, c["on_accent"])
        f.add(f"<g {f.anim(steps([(0, 'opacity:0'), (t, 'opacity:1'), (97, 'opacity:0')]))}>{inner}</g>")
    assert match_pos is not None
    # sliding window
    br, bq = match_pos
    wx, wy = ix + bq * cs, iy + br * cs
    segs = [(0.0, tx(0))] + [(s0 + k * dt, tx((q - bq) * cs, (r - br) * cs))
                             for k, (r, q) in enumerate(order)]
    segs += [(s1, tx(0))]
    f.add(f'<g {f.anim(steps(segs))}>{f.rect(wx - 2, wy - 2, 3 * cs + 4, 3 * cs + 4, "none", c["amber"], rx=5, sw=3.5)}</g>')
    f.add(f.text(ix + 4 * cs, iy + 8 * cs + 24, "the window visits every position", 12,
                 c["muted"], italic=True))
    f.add(f.text(mx + 3 * cs, my + 6 * cs + 24, "one square per window position", 12, c["muted"],
                 italic=True))
    # OR -> result
    ox = mx + 6 * cs + 18
    res = f.anim(steps([(0, "opacity:0"), (s1 + 1, "opacity:1"), (97, "opacity:0")]))
    f.add(f"<g {res}>{f.arrow(ox, my + 3 * cs, ox + 30, my + 3 * cs, c['muted'], 1.6, 6)}"
          f"{f.text(ox + 15, my + 3 * cs - 10, 'OR', 12, c['muted'], weight='bold')}"
          f"{f.rect(ox + 34, my + 3 * cs - 44, 118, 88, c['pos_soft'], c['pos'], rx=10)}"
          f"{f.text(ox + 93, my + 3 * cs - 18, 'found', 13, c['pos'], weight='bold')}"
          f"{f.text(ox + 93, my + 3 * cs - 2, 'somewhere', 13, c['pos'], weight='bold')}"
          f"{f.text(ox + 93, my + 3 * cs + 22, 'clause = TRUE', 12, c['pos'])}</g>")
    return f


def fig_conv_position(c: Dict[str, str]) -> Fig:
    f = Fig(760, 290, c, "Position literals",
            "Each window row is described by thermometer bits y greater than k. A clause "
            "that includes y > 2 and NOT y > 5 only accepts windows in rows 3 to 5.")
    gx, gy, cs, n = 40, 58, 24, 8
    lo, hi = 2, 5
    f.add(f.text(gx + n * cs / 2, 36, "Window positions", 13.5, weight="bold"))
    for r in range(n):
        ok = lo < r <= hi
        for q in range(n):
            f.add(f.rect(gx + q * cs + 1, gy + r * cs + 1, cs - 2, cs - 2,
                         c["accent_soft"] if ok else c["panel2"], c["accent"] if ok else "none",
                         rx=3, sw=1.2))
        f.add(f.text(gx - 10, gy + r * cs + 16, f"y={r}", 11, c["muted"], anchor="end"))
    tx0 = 290
    cols = [("y > 2", tx0), ("y > 5", tx0 + 80), ("clause looks here?", tx0 + 200)]
    for lab, x in cols:
        f.add(f.text(x, 36, lab, 13, weight="bold"))
    for r in range(n):
        y = gy + r * cs
        for bit, x in [(r > lo, tx0), (r > hi, tx0 + 80)]:
            f.add(f.rect(x - 16, y + 2, 32, cs - 4, c["blue_soft"] if bit else c["panel2"],
                         c["blue"] if bit else c["line"], rx=4, sw=1))
            f.add(f.text(x, y + 16, "1" if bit else "0", 12, c["blue"] if bit else c["muted"],
                         weight="bold"))
        ok = lo < r <= hi
        f.add(f.check(tx0 + 200, y + 12, 12, c["pos"]) if ok else f.cross(tx0 + 200, y + 12, 10,
                                                                          c["muted"], 1.6))
    f.add(f.text(tx0 + 40, gy + n * cs + 30, "needs  y > 2 = 1  AND  y > 5 = 0", 12.5,
                 c["accent"], weight="bold"))
    x = 560
    for i, s in enumerate(["Clause:", "plus shape", "AND  y > 2", "AND  NOT  y > 5", "",
                           "→ only looks for the", "plus in rows 3–5"]):
        f.add(f.text(x, 80 + i * 21, s, 13 if i else 13.5,
                     c["accent"] if i in (2, 3) else c["fg"], anchor="start",
                     weight="bold" if i in (0, 2, 3) else "normal"))
    return f


def _scene8() -> List[str]:
    return ["SSSSSSSS",
            "SSSSSSSS",
            "SBBBSSSS",
            "SBBBSTTT",
            "SBBBSTTT",
            "SBBBSSTS",
            "RRRRRRRR",
            "RRRRRRRR"]


def _photo_colour(scene: List[str], r: int, q: int) -> str:
    k = scene[r][q]
    if k == "B" and (r + q) % 2 == 0 and 0 < r:
        return PHOTO["W"]
    if k == "R" and r == len(scene) - 2 and q % 3 == 1:
        return PHOTO["L"]
    return PHOTO[k]


def _draw_photo(f: Fig, scene: List[str], x: float, y: float, cs: float) -> None:
    n = len(scene)
    f.add(f.rect(x - 1, y - 1, n * cs + 2, n * cs + 2, "none", f.c["line"], rx=2, sw=1))
    for r in range(n):
        for q in range(n):
            f.add(f'<rect x="{x + q * cs:g}" y="{y + r * cs:g}" width="{cs:g}" height="{cs:g}" '
                  f'fill="{_photo_colour(scene, r, q)}" shape-rendering="crispEdges"/>')


def fig_segmentation_dense(c: Dict[str, str]) -> Fig:
    f = Fig(760, 330, c, "From one answer per image to one answer per pixel",
            "Left: a convolutional Tsetlin machine ORs its windows into a single answer for the "
            "whole picture. Right: the dense head gives every pixel its own window and its own "
            "vote, painting a label map.", duration=16)
    scene = _scene8()
    n = 8
    # left: convolution
    f.add(f.rect(8, 8, 262, 314, c["panel"], c["line"], rx=12))
    f.add(f.text(139, 32, "Convolution", 14, weight="bold"))
    f.add(f.text(139, 50, "one answer for the whole image", 11.5, c["muted"]))
    lx, ly, lc = 26, 70, 16
    _draw_photo(f, scene, lx, ly, lc)
    for r, q, col in [(2, 1, c["amber"]), (0, 5, c["amber"]), (5, 4, c["amber"])]:
        f.add(f.rect(lx + q * lc, ly + r * lc, 3 * lc, 3 * lc, "none", col, rx=3, sw=2))
    ox, oy = 214, 134
    for r, q in [(2, 1), (0, 5), (5, 4)]:
        f.add(f.line(lx + (q + 3) * lc, ly + (r + 1.5) * lc, ox - 17, oy, c["muted"], 1.1))
    f.add(f'<circle cx="{ox}" cy="{oy}" r="17" fill="{c["panel2"]}" stroke="{c["muted"]}" '
          f'stroke-width="1.5"/>')
    f.add(f.text(ox, oy + 4.5, "OR", 12.5, weight="bold"))
    f.add(f.arrow(ox, oy + 19, ox, 232, c["muted"], 1.5, 6))
    f.add(f.rect(28, 238, 222, 54, c["pos_soft"], c["pos"], rx=10))
    f.add(f.text(139, 260, "“there is a building", 13, c["pos"], weight="bold"))
    f.add(f.text(139, 278, "somewhere in this picture”", 13, c["pos"], weight="bold"))
    # right: dense head
    f.add(f.rect(282, 8, 470, 314, c["panel"], c["line"], rx=12))
    f.add(f.text(517, 30, "Dense head (segmentation)", 14, weight="bold"))
    f.add(f.text(517, 47, "every pixel gets its own window and its own vote", 11.5, c["muted"]))
    sx, sy, sc = 304, 76, 20
    mx, my = 552, 76
    _draw_photo(f, scene, sx, sy, sc)
    f.add(f.arrow(sx + n * sc + 14, sy + n * sc / 2, mx - 14, my + n * sc / 2, c["muted"], 1.8, 8))
    f.add(f.text((sx + n * sc + mx) / 2, sy + n * sc / 2 - 10, "vote", 11.5, c["muted"]))
    f.add(f.rect(mx - 1, my - 1, n * sc + 2, n * sc + 2, c["panel2"], c["line"], rx=2, sw=1))
    order = [(r, q) for r in range(n) for q in range(n)]
    s0, s1 = 2.0, 82.0
    dt = (s1 - s0) / len(order)
    br, bq = 3, 2
    for k, (r, q) in enumerate(order):
        t = s0 + k * dt
        f.add(f'<rect x="{mx + q * sc:g}" y="{my + r * sc:g}" width="{sc:g}" height="{sc:g}" '
              f'fill="{LABEL[scene[r][q]]}" shape-rendering="crispEdges" '
              f'{f.anim(steps([(0, "opacity:0"), (t, "opacity:1"), (97, "opacity:0")]))}/>')
    segs = [(0.0, tx(0))] + [(s0 + k * dt, tx((q - bq) * sc, (r - br) * sc))
                             for k, (r, q) in enumerate(order)] + [(s1, tx(0))]
    win = f.anim(steps(segs))
    f.add(f'<g {win}>'
          f'{f.rect(sx + (bq - 1) * sc - 1, sy + (br - 1) * sc - 1, 3 * sc + 2, 3 * sc + 2, "none", c["amber"], rx=4, sw=3)}'
          f'{f.rect(sx + bq * sc + 6, sy + br * sc + 6, sc - 12, sc - 12, c["amber"], rx=2)}'
          f'{f.rect(mx + bq * sc + 1, my + br * sc + 1, sc - 2, sc - 2, "none", c["fg"], rx=2, sw=2.5)}'
          f'</g>')
    f.add(f.text(sx + n * sc / 2, sy + n * sc + 20, "the input picture", 11.5, c["muted"],
                 italic=True))
    f.add(f.text(mx + n * sc / 2, my + n * sc + 20, "the predicted label map", 11.5, c["muted"],
                 italic=True))
    lx = 330
    for i, k in enumerate("SBTR"):
        x = lx + i * 100
        f.add(f.rect(x, 280, 16, 16, LABEL[k], rx=3))
        f.add(f.text(x + 22, 293, LABEL_NAME[k], 12.5, anchor="start"))
    return f


def _scene16() -> List[str]:
    return ["SSSSSSSSSSSSSSSS",
            "SSSSSSSSSSSSSSSS",
            "SSSSSSSSSSSSSSSS",
            "SSSSSSSSSSSSSSSS",
            "SSBBBBBBSSSSSSSS",
            "SSBBBBBBSSSSSSSS",
            "SSBBBBBBSSSSSSSS",
            "SSBBBBBBSSSTTTSS",
            "SSBBBBBBSSTTTTTS",
            "SSBBBBBBSSTTTTTS",
            "SSBBBBBBSSSTTTSS",
            "SSBBBBBBSSSSTSSS",
            "RRRRRRRRRRRRRRRR",
            "RRRRRRRRRRRRRRRR",
            "RRRRRRRRRRRRRRRR",
            "RRRRRRRRRRRRRRRR"]


def _pool(scene: List[str], k: int) -> List[str]:
    """Majority class of each k x k block (ties broken in favour of the first seen)."""
    n = len(scene) // k
    out = []
    for r in range(n):
        row = ""
        for q in range(n):
            block = [scene[r * k + a][q * k + b] for a in range(k) for b in range(k)]
            row += max(dict.fromkeys(block), key=block.count)
        out.append(row)
    return out


def fig_segmentation_pyramid(c: Dict[str, str]) -> Fig:
    f = Fig(760, 280, c, "A multi-scale pyramid",
            "The same three by three window around one pixel, taken on the full-resolution "
            "image and on two and four times coarser copies, covers more and more of the "
            "scene.")
    scene = _scene16()
    pr, pq = 8, 6                       # the pixel we classify: inside the building
    size = 168
    for i, k in enumerate([1, 2, 4]):
        x, y = 30 + i * 250, 50
        lvl = _pool(scene, k) if k > 1 else scene
        n = len(lvl)
        cs = size / n
        f.add(f.text(x + size / 2, 32, ["full resolution", "2× coarser", "4× coarser"][i], 13.5,
                     weight="bold"))
        for r in range(n):
            for q in range(n):
                col = PHOTO[lvl[r][q]]
                f.add(f'<rect x="{x + q * cs:g}" y="{y + r * cs:g}" width="{cs:g}" '
                      f'height="{cs:g}" fill="{col}" shape-rendering="crispEdges"/>')
        f.add(f.rect(x - 1, y - 1, size + 2, size + 2, "none", c["line"], rx=2, sw=1))
        cr, cq = pr // k, pq // k
        f.add(f.rect(x + (cq - 1) * cs, y + (cr - 1) * cs, 3 * cs, 3 * cs, "none", c["amber"],
                     rx=3, sw=3))
        f.add(f'<circle cx="{x + (pq + 0.5) * size / 16:g}" cy="{y + (pr + 0.5) * size / 16:g}" '
              f'r="4" fill="{c["amber"]}" stroke="#ffffff" stroke-width="1.5"/>')
        seen = sorted({lvl[r][q] for r in range(max(cr - 1, 0), min(cr + 2, n))
                       for q in range(max(cq - 1, 0), min(cq + 2, n))}, key="SBTR".index)
        f.add(f.text(x + size / 2, y + size + 26, "the window sees:", 11.5, c["muted"]))
        f.add(f.text(x + size / 2, y + size + 44, ", ".join(LABEL_NAME[s] for s in seen), 13,
                     weight="bold"))
        if i < 2:
            f.add(f.arrow(x + size + 18, y + size / 2, x + 232, y + size / 2, c["muted"], 1.6, 7))
            f.add(f.text(x + size + 41, y + size / 2 - 9, "pool", 11.5, c["muted"]))
    return f


def fig_thermometer(c: Dict[str, str]) -> Fig:
    f = Fig(760, 310, c, "Thermometer encoding",
            "A number such as age is turned into bits age at least 20, 30, 40, 50 and 60. The "
            "bits fill up from the left as the value grows, and a range such as 30 to 49 needs "
            "only two literals.", duration=12)
    X0, X1, ly = 70, 690, 92
    vmax = 80

    def vx(v):
        return X0 + v / vmax * (X1 - X0)

    th = [20, 30, 40, 50, 60]
    f.add(f.text(X0 - 20, ly + 5, "age", 13, weight="bold", anchor="end"))
    f.add(f.rect(vx(30), ly - 14, vx(50) - vx(30), 28, c["accent_soft"], rx=4))
    f.add(f.line(X0, ly, X1, ly, c["muted"], 2))
    for v in range(0, vmax + 1, 10):
        f.add(f.line(vx(v), ly - 5, vx(v), ly + 5, c["muted"], 1.3))
        f.add(f.text(vx(v), ly + 22, str(v), 11.5, c["fg"] if v in th else c["muted"],
                     weight="bold" if v in th else "normal"))
    # value sequence (static picture shows 45)
    seq = [(0.0, 25), (25.0, 45), (50.0, 65), (75.0, 35)]
    base = 45
    d = 5.0
    pos = moves(tx(vx(seq[0][1]) - vx(base)),
                [(t, tx(vx(v) - vx(base))) for t, v in seq[1:]] + [(97, tx(vx(seq[0][1]) - vx(base)))],
                d / 2)
    g = (f'<polygon points="0,-14 -8,-28 8,-28" fill="{c["accent"]}"/>')
    labels = ""
    for i, (t, v) in enumerate(seq):
        end = (seq[i + 1][0] if i + 1 < len(seq) else 97.0) + d / 2
        win = [(t + (d / 2 if t else 0), end)]
        if i == 0:
            win.append((97.0 + d / 2, 100.0))
        if v == base:
            labels += f.text(0, -34, f"age = {v}", 13, c["accent"], weight="bold",
                             extra=f.anim(shown(win)))
        else:
            labels += f.text(0, -34, f"age = {v}", 13, c["accent"], weight="bold",
                             extra=f'opacity="0" {f.anim(shown(win))}')
    f.add(f'<g transform="translate({vx(base):.1f},{ly})"><g {f.anim(pos, "ease-in-out")}>'
          f'{g}{labels}</g></g>')

    def value_at_windows(pred):
        """Time windows (in %) during which ``pred(value)`` holds."""
        wins = []
        for i, (t, v) in enumerate(seq):
            a = t + (d / 2 if t else 0)
            b = seq[i + 1][0] + d / 2 if i + 1 < len(seq) else 97.0 + d / 2
            if pred(v):
                wins.append((a, b))
        if pred(seq[0][1]):
            wins.append((97.0 + d / 2, 100.0))
        return wins

    # bits
    by, bw = 150, 112
    bx0 = (760 - 5 * bw - 4 * 10) / 2
    f.add(f.text(bx0 - 12, by + 30, "bits", 13, weight="bold", anchor="end"))
    for i, t in enumerate(th):
        x = bx0 + i * (bw + 10)
        f.add(f.rect(x, by, bw, 50, c["panel2"], c["line"], rx=8))
        f.add(f.text(x + bw / 2, by + 20, f"age ≥ {t}", 12, c["muted"]))
        f.add(f.text(x + bw / 2, by + 41, "0", 16, c["muted"], weight="bold"))
        on = value_at_windows(lambda v, t=t: v >= t)
        static_on = base >= t
        f.add(f'<g {"" if static_on else HIDDEN} {f.anim(shown(on))}>'
              f'{f.rect(x, by, bw, 50, c["blue_soft"], c["blue"], rx=8, sw=2)}'
              f'{f.text(x + bw / 2, by + 20, f"age ≥ {t}", 12, c["blue"])}'
              f'{f.text(x + bw / 2, by + 41, "1", 16, c["blue"], weight="bold")}</g>')
    f.add(f.text(380, by + 72, "the 1s fill up from the left as the value rises, like a thermometer",
                 12, c["muted"], italic=True))
    # the range rule
    ry = 262
    f.add(f.rect(40, ry - 22, 680, 44, c["panel"], c["line"], rx=10))
    f.add(f.text(60, ry + 5, "Rule:", 13, anchor="start", weight="bold"))
    f.add(f.chip(108, ry - 14, 94, 28, "age ≥ 30", "included", 12))
    f.add(f.text(220, ry + 4.5, "AND", 11.5, c["muted"], weight="bold"))
    f.add(f.chip(240, ry - 14, 128, 28, "NOT age ≥ 50", "included", 12))
    f.add(f.text(384, ry + 5, "=  ages 30 to 49", 13, anchor="start"))
    inside = value_at_windows(lambda v: 30 <= v < 50)
    outside = value_at_windows(lambda v: not 30 <= v < 50)
    f.add(f'<g {f.anim(shown(inside))}>{f.check(530, ry, 14, c["pos"])}'
          f'{f.text(544, ry + 5, "matches this age", 12.5, c["pos"], anchor="start")}</g>')
    f.add(f'<g opacity="0" {f.anim(shown(outside))}>{f.cross(530, ry, 12, c["muted"])}'
          f'{f.text(544, ry + 5, "does not match", 12.5, c["muted"], anchor="start")}</g>')
    return f


def fig_specificity(c: Dict[str, str]) -> Fig:
    f = Fig(760, 268, c, "Specificity s",
            "With a small s literals are forgotten often, clauses stay short and match many "
            "days. With a large s they are forgotten rarely, clauses grow long and match few "
            "days.")
    days = 100
    panels = [
        ("Small s  ·  forget often", ["Sunny"], 42, "a short rule",
         "general — but may be too broad"),
        ("Large s  ·  forget rarely", ["Sunny", "Weekend", "NOT Rainy"], 7,
         "a long rule", "specific — but may memorise noise"),
    ]
    for p, (title, lits, hits, l1, l2) in enumerate(panels):
        px = 10 + p * 375
        f.add(f.rect(px, 8, 365, 252, c["panel"], c["line"], rx=12))
        f.add(f.text(px + 182, 34, title, 15, weight="bold"))
        x = px + 18
        for j, name in enumerate(lits):
            if j:
                f.add(f.text(x + 15, 70, "AND", 10.5, c["muted"], weight="bold"))
                x += 30
            w = 18 + 7.6 * len(name)
            f.add(f.chip(x, 52, w, 28, name, "included", 12))
            x += w
        gx, gy, cs = px + 18, 100, 14
        rng = [(i * 37 + 11) % days for i in range(days)]
        hit_set = set(rng[:hits])
        for i in range(days):
            r, q = divmod(i, 10)
            on = i in hit_set
            f.add(f'<circle cx="{gx + q * cs + 7}" cy="{gy + r * cs + 7}" r="5" '
                  f'fill="{c["accent"] if on else c["panel2"]}" '
                  f'stroke="{c["accent"] if on else c["line"]}" stroke-width="1"/>')
        tx0 = gx + 10 * cs + 18
        f.add(f.text(tx0, 125, f"{hits} of {days}", 22, c["accent"], anchor="start", weight="bold"))
        f.add(f.text(tx0, 145, "days match", 12.5, c["muted"], anchor="start"))
        f.add(f.text(tx0, 190, l1, 12, anchor="start"))
        f.add(f.text(tx0, 208, l2.split(" — ")[0] + " —", 12, anchor="start", weight="bold"))
        f.add(f.text(tx0, 226, l2.split(" — ")[1], 12, anchor="start"))
    return f


FIGURES: Dict[str, Callable[[Dict[str, str]], Fig]] = {
    "tm-clauses": fig_tm_clauses,
    "tm-automaton": fig_tm_automaton,
    "tm-feedback": fig_tm_feedback,
    "tm-vote-margin": fig_tm_vote_margin,
    "batching": fig_batching,
    "conv-sliding": fig_conv_sliding,
    "conv-position": fig_conv_position,
    "segmentation-dense": fig_segmentation_dense,
    "segmentation-pyramid": fig_segmentation_pyramid,
    "thermometer": fig_thermometer,
    "specificity": fig_specificity,
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "docs", "assets", "concepts"))
    ap.add_argument("--only", nargs="*", help="draw only these figures")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        for theme, suffix in (("light", ""), ("dark", "_dark")):
            path = os.path.join(args.out, f"{name}{suffix}.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(fn(THEMES[theme]).svg())
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
