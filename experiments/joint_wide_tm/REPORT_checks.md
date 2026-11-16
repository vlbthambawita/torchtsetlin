# joint_wide_tm — phase 0 correctness checks

Record: `results/checks/check_joint.json` (NVIDIA GeForce RTX 3080, torchtsetlin 0.7.1, git `20e3cc2d99`). **All passed.**

| Check | What | Result |
|---|---|---|
| T0 | one member, zero offset, own negative: bit-identical to the library model (CPU, CUDA eager, events, graphs) | pass |
| T1 | the score a member learns from equals the external joint score on every path | pass |
| T2 | shared mode: one negative class per example for all members, never the target | pass |
| T3 | CUDA graphs: a replayed update reads the new offset | pass |
| T4 | pass A equals the votes recomputed inside update (with T1) | pass |
| T5 | toy: joint team of a 2x2 and a 3x3 member is not worse than the better member (paired, 2 SE) | pass |
| T6 | `sum` team of identical members = one model with M*n clauses and M*T | pass |
| T8 | state_dict round trip; member files load into the library model | pass |

T5 (toy, 8 seeds, accuracy mean ± SE): m2x2 0.945 ± 0.054, m3x3 0.524 ± 0.018, joint-sum 0.943 ± 0.040, joint-one 0.967 ± 0.024. Paired difference to the better member: joint-sum -0.002 ± 0.019, joint-one +0.022 ± 0.030.

T6: (a) largest difference between the score a split member learns from and the big model's votes: 0. (b) feedback selections per update, paired by batch over 200 batches: `sum` team − big model = +0.69 ± 0.80 (0.9 SE); control (`one` rule, threshold T) = -65.23 ± 2.05 (31.8 SE). (c) trained from scratch, 32 seeds: accuracy team 0.979 ± 0.010, big model 0.960 ± 0.014 (1.0 SE apart).
