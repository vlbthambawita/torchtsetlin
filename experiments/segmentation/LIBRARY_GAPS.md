# Library gaps found by the segmentation programme — NOT APPLIED

`src/torchtsetlin/**` is read-only for this programme (PLAN §1). Entries are proposals only.

## LG-001 — dense models commit pixels of one image at a time at realistic sizes

- **Symptom**: with `SegmentationTsetlinMachine` at 128×128 px, F ≥ 108 and ≥ 600 clauses,
  `_chunk_size` returns 1 image, so every `patches_per_commit` window contains pixels of a
  single image regardless of the image batch size. Image batch size then has no effect on
  commit composition, and no ordering inside `update` can decorrelate commits across images.
- **Reproduction**: `python -c` snippet in DECISIONS D-007 (prints `chunk images 1 134217728`).
- **Cause**: `models/segmentation.py::_DenseMixin._chunk_elements_per_example` budgets
  `P * (8C + 2L)` per image against `max_chunk_elements = 2**27` (`models/base.py:92`), but with
  `patches_per_commit` set only the `(P, 2L)` literals are resident for the whole chunk
  (the comment there says so); the `8C` term applies per commit, not per image.
- **Impact here**: the E1 Kvasir `perm`/`disperse` arms only shuffled within one image; the
  corrected arms set `max_chunk_elements` high in the experiment (`cross_image=True`).
- **Proposed fix (NOT APPLIED)**: when `patches_per_commit` is set, budget
  `P * 2L + patches_per_commit * 8C` per chunk; optionally expose a `shuffle_pixels` /
  cross-image commit option once H1 evidence supports it.
- **Risk**: changes commit composition (and therefore results) of existing dense runs whose
  batches were silently split into single-image chunks.
