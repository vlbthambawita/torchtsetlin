# Cite torchtsetlin

If you use `torchtsetlin` in your work, please cite the technical report below, **together with
the Tsetlin machine papers whose algorithms you rely on**. Each concept page names the paper
behind every option.

<div class="grid cards" markdown>

-   :material-file-document-outline: **Technical report (5 pages): the document to cite**

    ---

    *torchtsetlin: Fast, Interpretable Tsetlin Machines for the PyTorch Ecosystem*. Software version
    %%version%%.

    [Download the PDF](../assets/papers/torchtsetlin-report.pdf){ .md-button .md-button--primary }

-   :material-book-open-page-variant-outline: **White paper (long form)**

    ---

    The full design, the batched feedback algorithm, benchmarks, the literature catalog and
    limitations.

    [Download the PDF](../assets/papers/torchtsetlin-whitepaper.pdf){ .md-button }

</div>

## BibTeX

```bibtex
@techreport{thambawita2026torchtsetlin,
  author      = {Thambawita, Vajira},
  title       = {torchtsetlin: Fast, Interpretable Tsetlin Machines for the PyTorch Ecosystem},
  institution = {Simula Metropolitan Center for Digital Engineering},
  type        = {Technical report},
  year        = {2026},
  note        = {Software version %%version%%},
  url         = {https://github.com/vlbthambawita/torchtsetlin}
}
```

## Plain text

V. Thambawita. *torchtsetlin: Fast, Interpretable Tsetlin Machines for the PyTorch Ecosystem*. Technical report,
Simula Metropolitan Center for Digital Engineering, 2026. Software version %%version%%.
<https://github.com/vlbthambawita/torchtsetlin>

The repository's `CITATION.cff` carries the same entry, which is what GitHub's
**Cite this repository** button shows.

## Building the PDFs

Both documents are LaTeX sources in `whitepaper/`, sharing one theme, the figures and data
files: `make` builds the white paper (`main.pdf`), and `make summary` builds the report
(`summary.pdf`). Their benchmark charts read the repository's benchmark records. Their
literature counts and charts come from the literature database, via
`python whitepaper/data/regen_literature.py`. The documentation build copies both committed
PDFs into the site.
