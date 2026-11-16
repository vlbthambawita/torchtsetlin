# joint_wide_tm — pricing (T7)

Timed updates after a warm-up, extrapolated to one epoch of 50 000 images (batch 32); evaluation = validation + test. GPU-hours = (train + eval) × epochs.

| Arm | Run | Train s/epoch | Eval s/epoch | Peak MiB | GPU-h, 40 epochs | GPU-h, 100 epochs |
|---|---|---|---|---|---|---|
| I | A10_seed0 | 4.1 | 0.7 | 560 | 0.05 | 0.13 |
| I | H10_seed0 | 23.4 | 3.7 | 1404 | 0.30 | 0.75 |
| I | H3_seed0 | 6.4 | 1.0 | 968 | 0.08 | 0.20 |
| I | H4_seed0 | 6.5 | 1.3 | 906 | 0.09 | 0.22 |
| I | H5_seed0 | 8.2 | 1.6 | 1070 | 0.11 | 0.27 |
| J | P_one_seed0 | 30.4 | 3.6 | 1814 | 0.38 | 0.94 |
| J | P_sum_seed0 | 30.5 | 3.6 | 1814 | 0.38 | 0.95 |
| J | S_one_seed0 | 65.2 | 7.8 | 2558 | 0.81 | 2.03 |
| J | S_sum_seed0 | 57.8 | 7.6 | 2558 | 0.73 | 1.82 |
| W | P_sum_A10_seed0 | 10.2 | 1.6 | 450 | 0.13 | 0.33 |
| W | P_sum_H3_seed0 | 20.3 | 2.7 | 882 | 0.26 | 0.64 |
| W | P_sum_H4_seed0 | 19.6 | 3.5 | 860 | 0.26 | 0.64 |
| W | S_sum_H10_seed0 | 107.2 | 13.4 | 2096 | 1.34 | 3.35 |
| W | S_sum_H3_seed0 | 18.1 | 2.5 | 882 | 0.23 | 0.57 |
| W | S_sum_H5_seed0 | 26.9 | 5.0 | 1116 | 0.35 | 0.89 |

Device: NVIDIA GeForce RTX 3080. Largest peak: 2558 MiB.

| Phase | Runs | GPU-hours |
|---|---|---|
| 1 pilot | 5 members + 4 joint, 40 epochs | 2.9 |
| 2 main | 3 seeds × (5 members + 2 joint + 2 W), 100 epochs; joint priced with rule `sum`, W with each team's most expensive member | 25.0 |
| **Total** | | **27.9** |
