# fmnist_bitplanes — pilot (Phase 3)

Selection on **validation** accuracy only (C = 2 000, seed 0, ≤ 60 epochs); ties go to the
base s. Test accuracy is shown but not used.

| Cell | s | kd | Best val | @ epoch | Test there | Chosen |
|---|---|---|---|---|---|---|
| conv3 × bit8 | 2.5 | 8 | 86.50 | 48 | 85.74 |  |
| conv3 × bit8 | 5 | 8 | 87.22 | 59 | 86.74 | **yes** |
| conv3 × bit8 | 10 | 8 | 86.69 | 46 | 86.53 |  |
| conv3 × gray8 | 2.5 | 8 | 86.80 | 37 | 86.15 |  |
| conv3 × gray8 | 5 | 8 | 87.79 | 49 | 87.19 | **yes** |
| conv3 × gray8 | 10 | 8 | 87.31 | 42 | 86.32 |  |
