# Library gaps found in toolbox_cifar10

`src/torchtsetlin/**` is read-only in this programme (README.md, R1); defects are logged here.

## TB-1: full include-cache refresh needs 8 bytes per automaton (memory spike at large models)

- **Found**: 2026-11-09, `train.py --spec therm5 --clauses 64000 --calibrate 20 --device cuda:1`
  (RTX 3080, 10 GB): `OutOfMemoryError: Tried to allocate 6.24 GiB` in
  `models/base.py::_refresh_include`, called from `_apply` on `.to(device)`.
- **Cause**: `inc = self.ta_state >= self.n_states` (bool, 1 byte per automaton), then
  `inc.sum(dim=1)` upcasts the whole `(C, 2F)` mask to int64: 640 000 × 1 308 × 8 B = 6.24 GiB,
  7.4× the uint8 state itself (0.84 GB). A second spike of half that size (3.12 GiB, int32 of the
  same shape) occurs during the first updates.
- **Effect here**: the 4×4/5×5 thermometer specialists at 64 000 clauses/class do not fit the
  10 GB card although their state (2.6 GB incl. the fp16 include cache) would; they run on the
  24 GB card only. Workaround in `train.py`: build the model on the GPU before the data goes there.
- **Suggested fix**: `include_count = inc.sum(dim=1, dtype=torch.int32)` (or row-chunked), and
  find the int32 `(C, 2F)` temporary in the update path.
