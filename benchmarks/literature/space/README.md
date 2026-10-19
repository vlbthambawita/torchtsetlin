---
title: Tsetlin Machine Literature Explorer
emoji: 😻
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Datasets and reported results from 55 Tsetlin-machine papers
---

# Tsetlin Machine literature explorer

Every dataset and every reported number from 55 Tsetlin-machine papers. That covers the proposed
models and the baselines they were compared against: 128 datasets and about 3,300 results. Use
it to filter, rank the best value per dataset or other grouping, see the best TM result per
dataset, browse datasets with their download links, and run read-only SQL.

The data is `literature.db` (SQLite), built from one JSON record per paper in
[`benchmarks/literature/`](https://github.com/vlbthambawita/torchtsetlin/tree/main/benchmarks/literature)
of the [torchtsetlin](https://github.com/vlbthambawita/torchtsetlin) repository. This folder
(`benchmarks/literature/space/`) is the Space's source. To update the Space:

```bash
python benchmarks/literature/build_catalog.py     # rebuilds space/literature.db
hf upload vlbthambawita/TsetlinMachines benchmarks/literature/space . --repo-type space
```

Numbers are as reported by the authors, not reproduced. See the app's *About* tab before
comparing across papers.
