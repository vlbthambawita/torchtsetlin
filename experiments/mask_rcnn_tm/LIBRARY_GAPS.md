# Library gaps found in the mask_rcnn_tm programme

`src/torchtsetlin/**` is read-only here (rule C1); defects go into this file with a minimal
reproduction when they are first hit. The expected entries MR-G1 to MR-G12 are listed in
`reviews/torchtsetlin-expert.md` (section B.6) of the plan folder.

## 2026-11-25, building the harness (torchtsetlin 0.7.4)

No defect was hit. The harness avoids the expected gaps by design:

| ID | Gap | How the harness avoids it |
|---|---|---|
| MR-G4 | dense models cannot train on a subset of cells | every head is a flat or convolutional model on gathered rows or crops, and predicts the same way |
| MR-G5 | multi-label models have no ignore label | the RPN is one two-class model with the anchor shape as input bits |
| MR-G2 | regression is single-output | ordinal multi-label heads (`CoalescedTsetlinMachine`, `ConvCoalescedTsetlinMachine`) |
| MR-G6 | chunk budgets ignore multi-label feedback tensors | batches of 64 (RPN deltas) and 32 (box head), set by hand |
| MR-G10 | `PyramidEncoder` pools sizes that are not multiples of a scale out of register | crops are 32 px, images 128 px; `boolenc.pyramid` asserts the sides are multiples of the largest scale |
