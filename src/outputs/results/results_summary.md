# Results summary

## Part 1: reproduction (Reported vs. Yours)

| Quantity | Reported (paper) | Ours |
|---|---|---|
| Number of pose labels | 4800 | 4800 |
| ANN validation accuracy | 54% | 68.6% |
| ANN val. accuracy, correct or adjacent cell | 'incorrect labels are commonly close' | 98.9% |
| PF A (N=1000, a=0.9/10 cyc): position error | ~5 - 19 m | 5.8 m mean (max 12.8) |
| PF A (N=1000, a=0.9/10 cyc): attitude error | ~0 - 3 deg | 4.6 deg mean (max 8.3) |
| PF B (N=5000, a=0.9/cyc): position error | ~3 - 34 m (mostly 5-15 m) | 5.3 m mean (max 9.5) |
| PF B (N=5000, a=0.9/cyc): attitude error | ~0 - 3 deg | 4.4 deg mean (max 7.7) |

## Part 2: extension (XGBoost + bundle adjustment) on the 10 s approach trajectory

| Method | Pos. mean [m] | Pos. RMSE [m] | Pos. max [m] | Att. mean [deg] | Att. max [deg] | ms/frame |
|---|---|---|---|---|---|---|
| ANN label only | 5.57 | 5.91 | 11.19 | 3.43 | 8.80 | 0.0 |
| PF A (N=1000, a=0.9/10 cyc) | 5.80 | 6.45 | 12.77 | 4.58 | 8.26 | 10.6 |
| PF B (N=5000, a=0.9/cyc) | 5.31 | 5.65 | 9.46 | 4.42 | 7.65 | 32.9 |
| XGBoost only | 2.53 | 2.73 | 5.25 | 1.87 | 2.87 | 0.0 |
| XGBoost + BA (single frame) | 2.43 | 2.65 | 5.10 | 1.97 | 3.23 | 9.5 |
| XGBoost + BA (window) | 2.42 | 2.63 | 4.99 | 1.97 | 3.40 | 39.6 |

## Static (independent single-frame) test-set comparison

| Method | Pos. RMSE [m] | Pos. median [m] | Att. mean [deg] |
|---|---|---|---|
| ANN label only (val set) | 6.02 | - | - |
| XGBoost only | 1.55 | 1.00 | 1.43 |
| XGBoost + BA | 1.43 | 0.81 | 1.23 |