# joint_wide_tm — library gaps

Defects or missing hooks of `src/torchtsetlin` found while working here (the library is read-only
in this programme). None is a defect so far; these are the private internals the joint update
relies on, i.e. what a library `WideTsetlinMachine` would have to make public.

| # | What | Where it shows | Status |
|---|---|---|---|
| JW-1 | No hook to add an external term to the vote sum that feedback is computed from. The experiment overrides `_votes` and copies the body of `TsetlinMachine._select_feedback` (to pass another threshold and negative class). | `code/joint.py` | open (feature) |
| JW-2 | No public "learning-mode vote sums" call. `forward()` in `train()` mode gives them, but through the dense `(B, P, C)` evaluation; pass A therefore calls `_triton.conv_eval(..., draw=False)` directly. | `code/joint.py::learning_votes` | open (feature) |
| JW-3 | A captured CUDA graph bakes in private Python attributes that are not in `_graph_key` (here `_joint_T`, `_joint_on`). Harmless here (constant per member), but a subclass that changes such an attribute silently replays the old value. | `code/joint.py` docstring | note |
