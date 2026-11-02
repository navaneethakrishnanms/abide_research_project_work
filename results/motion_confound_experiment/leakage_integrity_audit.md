# Leakage and integrity audit -- motion-confound experiment

| Check | Result | Evidence |
|---|---|---|
| Locked holdout not opened | PASS | Script reads only the train list (via `build_dev_set`) and the cohort CSV filtered to development IDs; grep for holdout file/guard references in the script = 0. Independence of the holdout remains **unverified** (180 of the 780 cohort subjects are outside the train list; IDs not read). |
| Alignment of subjects, labels, FD, folds | PASS | Asserted at start: subject order and labels equal the fast experiment's manifest; FD finite for all 600 subjects; diagnosis in the cohort file equals the dev diagnosis; each outer test fold equals the stored fold assignment. |
| Motion slopes from fitted rows only | PASS | `MotionResidualize.fit()` learns column means, mean FD and OLS slopes from the rows it is fitted on. Independent `np.polyfit` on the outer-training rows reproduced a fitted slope (rtol 1e-8) for all three candidates. |
| Adjustment removes the linear FD component in training data | PASS | Max |corr(FD, adjusted training feature)| = 2.6e-16 (AAL-116) and 2.0e-16 (CC200), i.e. zero up to floating point. |
| Inner CV re-estimates slopes per inner split | PASS | The step is a Pipeline member inside GridSearchCV; inner-validation and test rows only receive the training-derived slopes. |
| Test-row independence | PASS | Scrambling the features and FD of half the test rows left the other rows' predictions unchanged (atol 1e-12), all three candidates. |
| Scaling, feature selection, hyperparameters | PASS | StandardScaler, SelectKBest and GridSearchCV (inner 3-fold ROC-AUC) fitted on training rows only. FD-only control: scaler and logistic regression (C by inner CV) fitted on training rows only. |
| Baseline fairness | PASS | Unadjusted results are the saved fast-experiment folds (same subjects, folds, grids, code path); not recomputed or overwritten. |
| Candidate selection | NOT USED | No best-of-three or adjusted-vs-unadjusted selection was made. |
| Earlier phases and experiments unchanged | PASS | SHA-256 of 311 baseline files: 0 changed; prior result folders' newest file timestamps predate this run; splits folder untouched. |

Limitations: only the linear FD effect is removed; FD is correlated with diagnosis and site, so the adjustment may also remove real diagnosis-related variance; the 600 development subjects overlap the Phase 6-10 cohort; 5-fold CV has limited power; these are CV estimates, not independent test results.
