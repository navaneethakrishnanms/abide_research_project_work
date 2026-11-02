# Motion-confound experiment -- summary (development-set CV, exploratory)
600 development subjects (268 ASD / 332 TC), same outer folds, inner CV and grids as the fast experiment. Adjustment: z_adj = z - beta_j*(fd - mean_fd_fit), beta_j = OLS slope on the fitted rows only (re-estimated in every inner-training split); the feature's fitted mean is kept. Unadjusted results are the saved ones from the fast experiment. The locked holdout was not opened; its independence remains unverified.

| Atlas | Model | Unadjusted acc | Adjusted acc | Paired diff (mean +/- SD) | Per-fold diffs | Unadjusted AUC | Adjusted AUC |
|---|---|---|---|---|---|---|---|
| AAL116 | LR_L2 | 0.6417 | 0.6283 +/- 0.0419 | -0.0133 +/- 0.0217 | [-0.0417, -0.0083, -0.0083, 0.0167, -0.025] | 0.6944 | 0.6753 |
| AAL116 | LinearSVM | 0.6417 | 0.6067 +/- 0.0365 | -0.0350 +/- 0.0224 | [-0.0333, -0.0333, -0.05, 0.0, -0.0583] | 0.6944 | 0.6738 |
| CC200 | RBF_SVM | 0.6483 | 0.6283 +/- 0.0209 | -0.0200 +/- 0.0484 | [0.0333, -0.0417, -0.0083, 0.0083, -0.0917] | 0.7000 | 0.6767 |

Adjusted out-of-fold scores vs FD (Spearman, descriptive): AAL116/LR_L2 0.04, AAL116/LinearSVM 0.05, CC200/RBF_SVM 0.10 (unadjusted models were 0.20-0.28).

**FD-only control** (one feature, mean FD, logistic regression, same folds): accuracy 0.5750 +/- 0.0656, balanced accuracy 0.5658, AUC 0.5965 +/- 0.0744 (class-prior accuracy is 0.553).

**Pre-declared reading rule outcome: motion-confound hypothesis UNRESOLVED.** (SUPPORTED needs >=2 of 3 candidates with a >=3-point mean accuracy drop and drops in >=4/5 folds; WEAKENED needs every mean change within +/-1.5 points.)

Caveats: only the linear FD component is removed, so non-linear or site-specific motion effects are not addressed; FD is also related to site and diagnosis, so adjustment can remove real diagnosis-related variance; 5 folds cannot establish significance; correlation and predictive performance do not show causality. No 85% claim: best measured accuracy is in the CSVs.
