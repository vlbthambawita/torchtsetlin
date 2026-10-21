# JMLR MLOSS submission: torchtsetlin

*torchtsetlin: Fast, Interpretable Tsetlin Machines for the PyTorch Ecosystem*. This is a
submission to the JMLR **Machine Learning Open Source Software** track, which allows at most
4 pages of description plus references
(<https://www.jmlr.org/mloss/mloss-info.html>).

| Path | What it is |
|---|---|
| `paper/paper.tex` | First draft. It uses the official `jmlr2e.sty` from <https://github.com/JmlrOrg/jmlr-style-file>. |
| `paper/outline.tex` | The approved bullet-point structure. |
| `paper/figures/fig-architecture.tex` | The architecture figure, cut from the draft to save space. `\input` it to bring it back. |
| `paper/references.bib` | Verified bibliography. |
| `coverletter/jmlr_coverletter.tex` | The cover letter, from <https://github.com/JmlrOrg/jmlr-coverletter>, with the MLOSS-required fields added. |
| `literature/` | The literature list (`README.md`), the PDFs (`papers/<bibkey>.pdf`), and the fact sheet on existing TM software. |

## Build

```bash
export PATH=/work/vajira/DL2026/.texlive/tl/bin/x86_64-linux:$PATH
cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
cd ../coverletter && pdflatex jmlr_coverletter
```

## Placeholder conventions in the draft

- **Red numbers** (`\res{}`) are measurements still to be taken. Fill them from
  `benchmarks/results/*.json`, never by hand.
- **Red brackets** (`\ph{}`) are other missing facts: the version, the CI platforms, funding.
- **Plots stamped DUMMY DATA** use made-up values. They only fix the format of each figure.

## Still to do before submission (MLOSS review criteria)

1. Head-to-head run time, memory and accuracy against pyTsetlinMachine,
   PyTsetlinMachineCUDA, tmu and green-tsetlin (Table 1, bottom half).
2. Measure test coverage (`pytest --cov`) and report it. Reviewers expect close to 100%.
3. CI on macOS and Windows, or remove that claim from the draft.
4. Cover letter: the software version or tag, evidence of a user community, the action
   editors and reviewers, and disclosure of the arXiv technical report.
