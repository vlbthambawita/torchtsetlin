# cifar10_joint_teams — library gaps

`src/torchtsetlin/**` is read-only in this programme (README R1). Defects or missing features found
while working here are recorded here, not fixed in place. The joint update's own gaps (JW-1..3) are
in `../joint_wide_tm/LIBRARY_GAPS.md`.

| # | What | Workaround | Status |
|---|---|---|---|
| CJT-1 | The fast CUDA path only works on the *current* CUDA device. A model on `cuda:1` while the current device is `cuda:0` fails CUDA-graph capture (`cudaErrorStreamCaptureUnsupported` in `_encode`) and the Triton `conv_eval` kernel ("Pointer argument cannot be accessed from Triton (cpu tensor?)"), warns once, and then runs the slow eager torch path for the rest of the process. | every script calls `torch.cuda.set_device(dev)`; the runners see one GPU each | open (note) |
| — | Conv3d patch blocking leaves the fused CUDA path when one example exceeds `max_chunk_elements` (fmnist_bitplanes FB-1, GitHub #17) | `team.build_member` raises the budget to one patch block when needed (not needed at 2 000 clauses/class) | open |
