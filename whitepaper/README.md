# torchtsetlin white paper

LaTeX source for **_torchtsetlin: A GPU-native, PyTorch-idiomatic framework for Tsetlin
machines_** — an orange-themed white paper with TikZ diagrams and pgfplots charts drawn from
the repository's own benchmark records.

Built PDF: [`main.pdf`](main.pdf) (~32 pages, A4).

## Build

```bash
make              # two pdflatex passes (needed for the ToC and cross-references)
make watch        # rebuild on save, via latexmk -pvc
make clean        # remove .aux/.log/.toc, keep the PDF
```

Or by hand:

```bash
pdflatex main && pdflatex main
```

There is no bibtex/biber step: the reference list is typed out in
`sections/12-references.tex` so the document builds with `pdflatex` alone.
`references.bib` carries the same entries in BibTeX form for reuse elsewhere.

### Requirements

A TeX Live installation with `pgf`/`tikz`, `pgfplots`, `booktabs`, `colortbl`, `listings`,
`titlesec`, `fancyhdr`, `caption`, `enumitem`, `microtype`, `needspace`, `upquote`,
`hyperref`, plus the fonts `xcharter`, `fira`, `inconsolata` and `newtx`. On a minimal
TeX Live:

```bash
tlmgr install pgf pgfplots booktabs colortbl listings titlesec fancyhdr caption \
              enumitem microtype needspace upquote hyperref geometry xcolor tools \
              multirow xcharter fira inconsolata newtx
```

If the font packages are unavailable, comment out the four font lines near the top of
`ttwhitepaper.sty`; everything else falls back to Computer Modern.

## Layout

```
main.tex              document skeleton: geometry, hyperref, section order
ttwhitepaper.sty      the theme — palette, headings, callouts, listing style,
                      table rules, shared TikZ/pgfplots styles
sections/             one file per section, 00 (title page) to 12 (references)
figures/              TikZ diagrams and pgfplots charts, \input by the sections
data/                 benchmark CSVs the charts read (extracted from
                      ../benchmarks/results/cpu_vs_gpu.json)
references.bib        BibTeX form of the reference list
```

### Editing the theme

Everything visual lives in `ttwhitepaper.sty`:

| What | Where |
|---|---|
| Colours | the `\definecolor` block — `ttorange` is the primary, `ttslate` the CPU/counterpoint colour used consistently in every chart |
| Headings, running heads | `\titleformat` / `fancyhdr` blocks |
| Callout boxes | the `ttcallout` (orange left bar) and `ttkey` ("In short") environments |
| Code listings | `\lstdefinestyle{ttpython}` and `ttshell` |
| Table rules | `\ttoprule`, `\ttmidrule`, `\ttbotrule`, `\thead` |
| Figure styles | the `\tikzset` block (`ttnode`, `ttarrow`, …) and `\pgfplotsset` (`ttaxis`, `ttgpu`, `ttcpu`) |

Two details that are easy to trip over if you extend the theme:

- `\code`, `\clsname` and `\thead` all begin with `\leavevmode`. Without it, the leading
  `\color` whatsit lands in a p-column's *vertical* list and the whole cell drops one line
  below its row.
- Chart colour is semantic: **orange = GPU / this library, slate = CPU / baseline**. Keep it
  that way when adding a chart, or the reader has to re-learn the legend each time.

### Updating the numbers

The charts read the CSVs in `data/`, which were extracted from
`../benchmarks/results/cpu_vs_gpu.json`. After re-running

```bash
../benchmarks/run_benchmarks.sh
```

on different hardware, update the CSVs (and the hardware table in
`sections/08-benchmarks.tex`) and rebuild. The figures pick up the new values with no other
changes.

## Figures

| File | Figure |
|---|---|
| `fig-tm-overview.tex` | what a Tsetlin machine model looks like at inference time |
| `fig-architecture.tex` | the four layers of the library |
| `fig-pipeline.tex` | the `update()` pipeline and its subclass hooks |
| `fig-automaton.tex` | one Tsetlin automaton: state line, boundary, feedback types |
| `fig-batched.tex` | sequential commits vs the batched event-counting formulation |
| `fig-conv.tex` | convolutional evaluation and where the random patch is drawn |
| `fig-bench-batch.tex` | throughput vs batch size, training and inference |
| `fig-bench-scale.tex` | GPU speedup vs clause budget and feature count |
| `fig-bench-breakdown.tex` | where one `update()` call spends its time |
| `fig-bench-threads.tex` | CPU thread scaling |
| `fig-bench-fidelity.tex` | what batching costs in accuracy (Noisy XOR, 3 seeds) |

## Licence

Same as the project: MIT.
