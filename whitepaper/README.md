# torchtsetlin white paper

LaTeX source for **_torchtsetlin: Fast, Interpretable Tsetlin Machines for the PyTorch
Ecosystem_** — an orange-themed white paper with TikZ diagrams and pgfplots charts drawn from
the repository's own benchmark records.

Built PDFs: [`main.pdf`](main.pdf), the white paper (~41 pages, A4), and
[`summary.pdf`](summary.pdf), the five-page technical report to cite. Both describe version
0.3.2 and share the theme, figures and data files.

## Build

```bash
make              # two pdflatex passes (needed for the ToC and cross-references)
make summary      # the five-page report, summary.pdf
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
data/                 benchmark CSVs the charts read, plus regen.py which rebuilds
                      them from ../benchmarks/results/versions/<version>/, and
                      regen_literature.py, which writes lit-*.csv and lit-stats.tex
                      from ../docs/assets/literature/literature.db
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

The charts read the CSVs in `data/`, which are generated from one version's
`../benchmarks/results/versions/<version>/cpu_vs_gpu.json` (default: the newest). The CSVs
in the repository are torchtsetlin **0.3.1**'s (`python whitepaper/data/regen.py 0.3.1`
reproduces them exactly). After re-running the benchmarks:

```bash
./benchmarks/run_benchmarks.sh        # from the repo root
python whitepaper/data/regen.py       # rewrite data/*.csv from the newest version's JSON
cd whitepaper && make
```

The figures pick the new values up with no other changes. **The hand-written tables and the
prose figures are not derived from the CSVs** — after a re-run, check these against
`../benchmarks/results/versions/<version>/tables.md` by hand:

- the environment table, model zoo, 3D convolution and end-to-end MNIST tables in
  `sections/08-benchmarks.tex`
- the throughput/speedup numbers quoted in §8 prose (including the segmentation and 3D
  paragraphs of §8.8), the "peak measured" row in §1, and the peak speedup in the title-page
  abstract
- the `10.8×` / `1.3×` annotations in `figures/fig-bench-threads.tex`

The accuracy numbers in §6 (3D shapes, segmentation commit size and context ablations) come
from the examples and the docs' concept pages, not from the benchmark driver.

## Figures

| File | Figure |
|---|---|
| `fig-tm-overview.tex` | what a Tsetlin machine model looks like at inference time |
| `fig-architecture.tex` | the four layers of the library |
| `fig-pipeline.tex` | the `update()` pipeline and its subclass hooks |
| `fig-automaton.tex` | one Tsetlin automaton: state line, boundary, feedback types |
| `fig-batched.tex` | sequential commits vs the batched event-counting formulation |
| `fig-conv.tex` | convolutional evaluation and where the random patch is drawn |
| `fig-structures.tex` | the dense segmentation head and the 3D convolution window |
| `fig-bench-batch.tex` | throughput vs batch size, training and inference |
| `fig-bench-scale.tex` | GPU speedup vs clause budget and feature count |
| `fig-bench-breakdown.tex` | where one `update()` call spends its time |
| `fig-bench-threads.tex` | CPU thread scaling |
| `fig-bench-fidelity.tex` | what batching costs in accuracy (Noisy XOR, 3 seeds) |
| `fig-bench-segmentation.tex` | dense segmentation throughput against commit size |

## Licence

Same as the project: MIT.
