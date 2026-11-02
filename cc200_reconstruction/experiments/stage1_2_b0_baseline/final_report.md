# Stage 2 report: CC200 baseline B0 (development data only)

Data: the 676 authorized training subjects (313 ASD / 363 TC, 19 sites); features 19,902 = 19,900 Ledoit-Wolf correlation/Fisher-Z edges + age + sex. 10-fold outer CV x seeds 0, 1, 2; inner 5-fold ROC-AUC over C in {0.001, 0.01, 0.1, 1, 10}.
Runtime 7.9 min. Restricted holdout, combined arrays, leave-one-site-out and variants V1-V6 were not touched or run. All numbers are development cross-validation estimates (mean +/- SD over the 3 seeds), not independent-test performance.

## Metrics (arm A = historical-faithful preprocessing; arm B = strict, preprocessing refitted in every inner fold)
| Arm | Threshold reading | Accuracy | Balanced acc. | Sensitivity | Specificity | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| A | (ii) Youden on pooled OOF (historical rule) | 67.21% +/- 0.76 | 66.85% +/- 0.38 | 61.98% +/- 6.29 | 71.72% +/- 6.70 | 65.69% +/- 3.06 | 63.55% +/- 1.84 | 0.7175 +/- 0.0055 |
| A | (i) fixed 0.5 | 65.98% +/- 0.30 | 65.72% +/- 0.35 | 62.19% +/- 1.21 | 69.24% +/- 0.69 | 63.55% +/- 0.26 | 62.86% +/- 0.62 | 0.7175 +/- 0.0055 |
| A | (iii) nested training-only Youden | 66.12% +/- 1.41 | 65.73% +/- 1.28 | 60.38% +/- 5.02 | 71.07% +/- 5.53 | 64.48% +/- 2.75 | 62.22% +/- 2.05 | 0.7175 +/- 0.0055 |
| B | (ii) Youden on pooled OOF (historical rule) | 67.31% +/- 0.78 | 66.93% +/- 0.43 | 61.87% +/- 6.42 | 71.99% +/- 6.81 | 65.89% +/- 3.18 | 63.58% +/- 1.90 | 0.7187 +/- 0.0052 |
| B | (i) fixed 0.5 | 66.07% +/- 0.23 | 65.82% +/- 0.26 | 62.41% +/- 1.03 | 69.24% +/- 0.69 | 63.63% +/- 0.26 | 63.01% +/- 0.50 | 0.7187 +/- 0.0052 |
| B | (iii) nested training-only Youden | 65.68% +/- 1.04 | 65.36% +/- 1.06 | 61.02% +/- 5.54 | 69.70% +/- 5.05 | 63.59% +/- 1.97 | 62.14% +/- 2.36 | 0.7187 +/- 0.0052 |

Fold-to-fold SD of accuracy (arm A, reading ii): 3.95 points (mean over seeds); per-fold values in `fold_metrics.csv`.

## Confusion matrices from the genuine out-of-fold predictions (676 subjects: 313 ASD / 363 TC)
| Arm | Seed | Threshold | TP | FN | FP | TN | Accuracy |
|---|---|---|---|---|---|---|---|
| A | seed 0 | (ii) | 216 | 97 | 129 | 234 | 66.57% |
| A | seed 1 | (ii) | 188 | 125 | 98 | 265 | 67.01% |
| A | seed 2 | (ii) | 178 | 135 | 81 | 282 | 68.05% |
| B | seed 0 | (i) | 194 | 119 | 112 | 251 | 65.83% |
| B | seed 1 | (i) | 199 | 114 | 114 | 249 | 66.27% |
| B | seed 2 | (i) | 193 | 120 | 109 | 254 | 66.12% |
All readings and arms: `seed_metrics.csv`; predictions: `oof_predictions.csv`.

## Comparison with the recorded phase 2h result (arm A, historical threshold rule)
| Seed | Recorded accuracy | Reconstructed | Diff (points) | Recorded AUC | Reconstructed AUC | Same C in folds | Identical confusion matrix |
|---|---|---|---|---|---|---|---|
| 0 | 66.57% | 66.57% | +0.00 | 0.7112 | 0.7112 | 10/10 | yes |
| 1 | 67.01% | 67.01% | +0.00 | 0.7202 | 0.7202 | 10/10 | yes |
| 2 | 68.05% | 68.05% | +0.00 | 0.7211 | 0.7211 | 10/10 | yes |

Mean accuracy: recorded 67.21% (phase 2h: 67.21%), reconstructed 67.21%. Nothing was tuned to move this number.
