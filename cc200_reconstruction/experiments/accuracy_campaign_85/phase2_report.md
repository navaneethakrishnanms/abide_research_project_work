# Phase 2 report: CC200 accuracy campaign (reference = P1_LR, campaign reproduction of the baseline; exact match to the saved baseline verified)
Data: 676 authorized training subjects (313 ASD / 363 TC); outer folds 10 x seeds 0, 1, 2 (identical for all rows); all numbers are development cross-validation estimates (mean over 3 seeds), not independent-test performance. Holdout untouched.
Readings: **R1** = threshold chosen on inner out-of-fold scores of the outer-training partition (primary); R0 = default threshold (0.5 probability / 0 margin). Note: the reference under R0 is the saved baseline (66.07%); under R1 it is 65.68%.
All requested experiments completed.

## 1. Ranking by mean R1 accuracy (primary), with stability
| Rank | Experiment | Accuracy R1 (mean +/- SD over seeds) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD (pts) | Accuracy R0 | Runtime (min per 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P2_LSVM | 66.07% +/- 0.37 | 65.85% | 0.7139 +/- 0.0053 | 62.83% | 68.87% | 63.58% | 63.14% | 4.90 | 65.34% | 2.3 |
| 2 | P1_LR | 65.68% +/- 1.04 | 65.36% | 0.7187 +/- 0.0052 | 61.02% | 69.70% | 63.59% | 62.14% | 4.72 | 66.07% | 2.3 |
| 3 | P2_RBF | 65.29% +/- 1.41 | 64.99% | 0.7176 +/- 0.0045 | 60.92% | 69.05% | 62.95% | 61.91% | 4.91 | 66.72% | 9.6 |
| 4 | P2_XGB | 65.09% +/- 0.82 | 65.26% | 0.7087 +/- 0.0051 | 67.63% | 62.90% | 61.15% | 64.18% | 6.42 | 65.58% | 8.7 |
| 5 | P2_ET | 64.69% +/- 0.95 | 65.12% | 0.6917 +/- 0.0029 | 70.82% | 59.41% | 60.07% | 64.97% | 7.00 | 63.41% | 1.7 |
| 6 | P2_RF | 64.00% +/- 0.73 | 64.63% | 0.6954 +/- 0.0054 | 73.16% | 56.11% | 58.99% | 65.30% | 5.31 | 64.15% | 4.4 |
Stability:
- P2_LSVM: SD of the three seed-level accuracies 0.37 pts; mean fold-to-fold SD 4.90 pts.
- P1_LR: SD of the three seed-level accuracies 1.04 pts; mean fold-to-fold SD 4.72 pts.
- P2_RBF: SD of the three seed-level accuracies 1.41 pts; mean fold-to-fold SD 4.91 pts.
- P2_XGB: SD of the three seed-level accuracies 0.82 pts; mean fold-to-fold SD 6.42 pts.
- P2_ET: SD of the three seed-level accuracies 0.95 pts; mean fold-to-fold SD 7.00 pts.
- P2_RF: SD of the three seed-level accuracies 0.73 pts; mean fold-to-fold SD 5.31 pts.

## 2. Paired comparison with the reference and frozen screening (R1; folds wins/ties/losses out of 30)
| Experiment | Mean gain acc. (pts) | Mean gain bacc. (pts) | Gain by seed (pts) | Fold W/T/L | Sign-test p | Paired-fold SE (pts) | Mean dAUC | Screening | Criteria failed |
|---|---|---|---|---|---|---|---|---|---|
| P2_LSVM | +0.39 | +0.49 | [0.74, 0.89, -0.44] | 11.0/9.0/10.0 | 0.951 | 0.55 | -0.0048 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30 |
| P2_RBF | -0.39 | -0.37 | [-1.04, 1.48, -1.63] | 9.0/8.0/13.0 | 0.992 | 0.62 | -0.0011 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30 |
| P2_XGB | -0.59 | -0.10 | [0.0, 0.74, -2.51] | 11.0/6.0/13.0 | 0.951 | 0.96 | -0.0100 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30; d_mean_auc_not_lower_by_more_than_0.005 |
| P2_ET | -0.99 | -0.24 | [0.44, -0.15, -3.25] | 14.0/1.0/15.0 | 0.708 | 1.10 | -0.0270 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30; d_mean_auc_not_lower_by_more_than_0.005 |
| P2_RF | -1.68 | -0.72 | [-1.78, -0.74, -2.51] | 9.0/4.0/17.0 | 0.992 | 0.89 | -0.0232 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30; d_mean_auc_not_lower_by_more_than_0.005 |
**Outcome:** No candidate passed the frozen screening rule; the reference logistic regression (P1_LR) remains the best defensible model of this phase.  (screening needs +1.5 pts accuracy and balanced accuracy, +0.5 in every seed, >= 20/30 fold wins and AUC within 0.005.) The highest mean R1 accuracy is P2_LSVM (66.07%); with the paired-fold SE shown above, differences below about 1 point are not distinguishable from noise.

## 3. Per-seed results (both readings)
| Experiment | Seed | Reading | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | R0 | 65.83% | 65.56% | 0.7129 | 61.98% | 69.15% | 63.40% | 62.68% | 3.22% |
| P1_LR | 0 | R1 | 64.94% | 65.00% | 0.7129 | 65.81% | 64.19% | 61.31% | 63.48% | 4.06% |
| P1_LR | 1 | R0 | 66.27% | 66.09% | 0.7202 | 63.58% | 68.60% | 63.58% | 63.58% | 4.96% |
| P1_LR | 1 | R1 | 65.24% | 64.53% | 0.7202 | 54.95% | 74.10% | 64.66% | 59.41% | 5.87% |
| P1_LR | 2 | R0 | 66.12% | 65.82% | 0.7230 | 61.66% | 69.97% | 63.91% | 62.76% | 5.85% |
| P1_LR | 2 | R1 | 66.86% | 66.55% | 0.7230 | 62.30% | 70.80% | 64.78% | 63.52% | 4.23% |
| P2_LSVM | 0 | R0 | 65.24% | 65.36% | 0.7083 | 67.09% | 63.64% | 61.40% | 64.12% | 4.02% |
| P2_LSVM | 0 | R1 | 65.68% | 65.76% | 0.7083 | 66.77% | 64.74% | 62.02% | 64.31% | 3.62% |
| P2_LSVM | 1 | R0 | 64.64% | 64.79% | 0.7146 | 66.77% | 62.81% | 60.76% | 63.62% | 6.24% |
| P2_LSVM | 1 | R1 | 66.12% | 65.82% | 0.7146 | 61.66% | 69.97% | 63.91% | 62.76% | 5.57% |
| P2_LSVM | 2 | R0 | 66.12% | 66.19% | 0.7188 | 67.09% | 65.29% | 62.50% | 64.71% | 6.63% |
| P2_LSVM | 2 | R1 | 66.42% | 65.98% | 0.7188 | 60.06% | 71.90% | 64.83% | 62.35% | 5.50% |
| P2_RBF | 0 | R0 | 66.27% | 65.89% | 0.7126 | 60.70% | 71.07% | 64.41% | 62.50% | 3.64% |
| P2_RBF | 0 | R1 | 63.91% | 63.62% | 0.7126 | 59.74% | 67.49% | 61.31% | 60.52% | 4.15% |
| P2_RBF | 1 | R0 | 66.42% | 66.03% | 0.7189 | 60.70% | 71.35% | 64.63% | 62.60% | 5.65% |
| P2_RBF | 1 | R1 | 66.72% | 66.32% | 0.7189 | 61.02% | 71.63% | 64.97% | 62.93% | 5.71% |
| P2_RBF | 2 | R0 | 67.46% | 66.92% | 0.7213 | 59.74% | 74.10% | 66.55% | 62.96% | 5.57% |
| P2_RBF | 2 | R1 | 65.24% | 65.01% | 0.7213 | 61.98% | 68.04% | 62.58% | 62.28% | 4.86% |
| P2_RF | 0 | R0 | 64.50% | 63.88% | 0.6954 | 55.59% | 72.18% | 63.27% | 59.18% | 5.73% |
| P2_RF | 0 | R1 | 63.17% | 63.94% | 0.6954 | 74.44% | 53.44% | 57.96% | 65.17% | 6.59% |
| P2_RF | 1 | R0 | 63.46% | 62.81% | 0.6900 | 53.99% | 71.63% | 62.13% | 57.78% | 5.74% |
| P2_RF | 1 | R1 | 64.50% | 65.09% | 0.6900 | 73.16% | 57.02% | 59.48% | 65.62% | 5.03% |
| P2_RF | 2 | R0 | 64.50% | 63.82% | 0.7009 | 54.63% | 73.00% | 63.57% | 58.76% | 5.63% |
| P2_RF | 2 | R1 | 64.35% | 64.87% | 0.7009 | 71.88% | 57.85% | 59.52% | 65.12% | 4.31% |
| P2_ET | 0 | R0 | 63.02% | 62.48% | 0.6928 | 55.27% | 69.70% | 61.13% | 58.05% | 7.07% |
| P2_ET | 0 | R1 | 65.38% | 65.79% | 0.6928 | 71.25% | 60.33% | 60.76% | 65.59% | 7.34% |
| P2_ET | 1 | R0 | 63.46% | 62.88% | 0.6938 | 54.95% | 70.80% | 61.87% | 58.21% | 5.33% |
| P2_ET | 1 | R1 | 65.09% | 65.75% | 0.6938 | 74.76% | 56.75% | 59.85% | 66.48% | 6.16% |
| P2_ET | 2 | R0 | 63.76% | 63.13% | 0.6884 | 54.63% | 71.63% | 62.41% | 58.26% | 7.11% |
| P2_ET | 2 | R1 | 63.61% | 63.81% | 0.6884 | 66.45% | 61.16% | 59.60% | 62.84% | 7.51% |
| P2_XGB | 0 | R0 | 65.24% | 64.64% | 0.7039 | 56.55% | 72.73% | 64.13% | 60.10% | 6.82% |
| P2_XGB | 0 | R1 | 64.94% | 64.85% | 0.7039 | 63.58% | 66.12% | 61.80% | 62.68% | 6.83% |
| P2_XGB | 1 | R0 | 65.98% | 65.42% | 0.7141 | 57.83% | 73.00% | 64.87% | 61.15% | 4.69% |
| P2_XGB | 1 | R1 | 65.98% | 66.23% | 0.7141 | 69.65% | 62.81% | 61.76% | 65.47% | 6.01% |
| P2_XGB | 2 | R0 | 65.53% | 64.89% | 0.7082 | 56.23% | 73.55% | 64.71% | 60.17% | 8.37% |
| P2_XGB | 2 | R1 | 64.35% | 64.71% | 0.7082 | 69.65% | 59.78% | 59.89% | 64.40% | 6.43% |

## 4. Confusion matrices (pooled out-of-fold predictions, R1; 313 ASD / 363 TC)
| Experiment | Seed | TP | FN | FP | TN | Accuracy | Balanced acc. | AUC |
|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | 206 | 107 | 130 | 233 | 64.94% | 65.00% | 0.7129 |
| P1_LR | 1 | 172 | 141 | 94 | 269 | 65.24% | 64.53% | 0.7202 |
| P1_LR | 2 | 195 | 118 | 106 | 257 | 66.86% | 66.55% | 0.7230 |
| P2_LSVM | 0 | 209 | 104 | 128 | 235 | 65.68% | 65.76% | 0.7083 |
| P2_LSVM | 1 | 193 | 120 | 109 | 254 | 66.12% | 65.82% | 0.7146 |
| P2_LSVM | 2 | 188 | 125 | 102 | 261 | 66.42% | 65.98% | 0.7188 |
| P2_RBF | 0 | 187 | 126 | 118 | 245 | 63.91% | 63.62% | 0.7126 |
| P2_RBF | 1 | 191 | 122 | 103 | 260 | 66.72% | 66.32% | 0.7189 |
| P2_RBF | 2 | 194 | 119 | 116 | 247 | 65.24% | 65.01% | 0.7213 |
| P2_RF | 0 | 233 | 80 | 169 | 194 | 63.17% | 63.94% | 0.6954 |
| P2_RF | 1 | 229 | 84 | 156 | 207 | 64.50% | 65.09% | 0.6900 |
| P2_RF | 2 | 225 | 88 | 153 | 210 | 64.35% | 64.87% | 0.7009 |
| P2_ET | 0 | 223 | 90 | 144 | 219 | 65.38% | 65.79% | 0.6928 |
| P2_ET | 1 | 234 | 79 | 157 | 206 | 65.09% | 65.75% | 0.6938 |
| P2_ET | 2 | 208 | 105 | 141 | 222 | 63.61% | 63.81% | 0.6884 |
| P2_XGB | 0 | 199 | 114 | 123 | 240 | 64.94% | 64.85% | 0.7039 |
| P2_XGB | 1 | 218 | 95 | 135 | 228 | 65.98% | 66.23% | 0.7141 |
| P2_XGB | 2 | 218 | 95 | 146 | 217 | 64.35% | 64.71% | 0.7082 |
Per-fold values, selected hyperparameters, thresholds and runtimes: `results/<experiment>_fold_metrics.csv`; per-subject predictions: `results/<experiment>_oof_predictions_seeds012.csv`; configurations: `results/<experiment>_config.json`.

## 5. Protocol verification (code-level and run-time checks)
- Every learned step is inside an sklearn `Pipeline` refitted by `GridSearchCV` on each inner-training split (demographic imputation and scaling, `StandardScaler`, in-fold `SelectKBest` for the tree models, PCA, L1 selection, classifier); the outer-test fold is only transformed and scored.
- Hyperparameters: chosen by inner 5-fold ROC-AUC on the outer-training rows only. Thresholds (R1): from `cross_val_predict` on the outer-training rows only; the outer-test scores never enter their choice.
- The engine asserts disjoint train/test indices for every fold; subject IDs are unique and no two series are identical (Stage 1); the reference run reproduced the saved baseline's scores with zero difference.
- Tangent-space (Phase 3): the geometric-mean reference is fitted on the outer-training covariances (label-free); the inner CV of that pipeline sees features built with that reference (unsupervised; no labels, no outer-test rows).
- Tree models use an in-fold k = 2000 univariate filter inside the pipeline (runtime control, predeclared); selection never sees outer-test rows.
