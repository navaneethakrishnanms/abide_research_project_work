# Site-centering experiment -- summary (development-set CV, exploratory)
Development subjects: 600 (268 ASD / 332 TC), 19 sites. Same outer folds, inner CV and grids as the fast experiment. Baselines are the uncentered results already saved by that experiment. Site offsets (site mean - global mean) are estimated from the fitted training rows only; sites absent from, or with fewer than 5 rows in, the fitted data get zero offset. The locked holdout was not opened.

| Atlas | Model | Uncentered acc | Centered acc | Paired diff (mean +/- SD) | Folds improved | Uncentered AUC | Centered AUC | AUC diff |
|---|---|---|---|---|---|---|---|---|
| AAL116 | LR_L2 | 0.6417 | 0.6467 +/- 0.0209 | +0.0050 +/- 0.0304 | 2/5 | 0.6944 | 0.7095 | +0.0151 |
| AAL116 | LinearSVM | 0.6417 | 0.6433 +/- 0.0325 | +0.0017 +/- 0.0308 | 3/5 | 0.6944 | 0.7094 | +0.0150 |
| CC200 | RBF_SVM | 0.6483 | 0.6383 +/- 0.0192 | -0.0100 +/- 0.0630 | 2/5 | 0.7000 | 0.7113 | +0.0113 |

Highest centered accuracy: AAL116/LR_L2 at 0.6467. This is the best of three candidates and is **exploratory**, not an unbiased estimate.

Candidates where centering improved accuracy in >=4/5 folds AND the mean gain exceeded the SD of the paired differences: **0 of 3**.

**85% target reached: NO** (best measured CV accuracy 64.7%).

With 5 outer folds no significance test is performed; differences of a few points are within fold-to-fold variation. These are development cross-validation estimates, not an independent test.
