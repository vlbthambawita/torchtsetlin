# Cite torchtsetlin

If you use `torchtsetlin` in your work, please cite the preprint below, **together with
the Tsetlin machine papers whose algorithms you rely on**. Each concept page names the paper
behind every option.

<div class="grid cards" markdown>

-   :material-file-document-outline: **Preprint: the document to cite**

    ---

    V. Thambawita, P. Halvorsen and O.-C. Granmo. *torchtsetlin: Fast, Interpretable Tsetlin
    Machines for PyTorch*. arXiv:2611.28195, 2026.

    [Read on arXiv](https://arxiv.org/abs/2611.28195){ .md-button .md-button--primary }

-   :material-file-outline: **Technical report (5 pages)**

    ---

    The short summary of the library that preceded the preprint. Software version
    %%version%%.

    [Download the PDF](../assets/papers/torchtsetlin-report.pdf){ .md-button }

-   :material-book-open-page-variant-outline: **White paper (long form)**

    ---

    The full design, the batched feedback algorithm, benchmarks, the literature catalog and
    limitations.

    [Download the PDF](../assets/papers/torchtsetlin-whitepaper.pdf){ .md-button }

</div>

## BibTeX

```bibtex
@misc{thambawita2026torchtsetlin,
  author        = {Thambawita, Vajira and Halvorsen, P{\aa}l and Granmo, Ole-Christoffer},
  title         = {torchtsetlin: Fast, Interpretable Tsetlin Machines for PyTorch},
  year          = {2026},
  eprint        = {2611.28195},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2611.28195},
  url           = {https://arxiv.org/abs/2611.28195}
}
```

## Plain text

V. Thambawita, P. Halvorsen and O.-C. Granmo. *torchtsetlin: Fast, Interpretable Tsetlin Machines
for PyTorch*. arXiv preprint arXiv:2611.28195, 2026. <https://doi.org/10.48550/arXiv.2611.28195>

The repository's `CITATION.cff` carries the same entry, which is what GitHub's
**Cite this repository** button shows.

## Building the PDFs

Both documents are LaTeX sources in `whitepaper/`, sharing one theme, the figures and data
files: `make` builds the white paper (`main.pdf`), and `make summary` builds the report
(`summary.pdf`). Their benchmark charts read the repository's benchmark records. Their
literature counts and charts come from the literature database, via
`python whitepaper/data/regen_literature.py`. The documentation build copies both committed
PDFs into the site.
