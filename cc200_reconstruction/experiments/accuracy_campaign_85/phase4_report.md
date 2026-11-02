# Phase 4 report: CC200 accuracy campaign (reference = P1_LR, campaign reproduction of the baseline; exact match to the saved baseline verified)
Data: 676 authorized training subjects (313 ASD / 363 TC); outer folds 10 x seeds 0, 1, 2 (identical for all rows); all numbers are development cross-validation estimates (mean over 3 seeds), not independent-test performance. Holdout untouched.
Readings: **R1** = threshold chosen on inner out-of-fold scores of the outer-training partition (primary); R0 = default threshold (0.5 probability / 0 margin). Note: the reference under R0 is the saved baseline (66.07%); under R1 it is 65.68%.
All requested experiments completed.

## 1. Ranking by mean R1 accuracy (primary), with stability
| Rank | Experiment | Accuracy R1 (mean +/- SD over seeds) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD (pts) | Accuracy R0 | Runtime (min per 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P1_LR | 65.68% +/- 1.04 | 65.36% | 0.7187 +/- 0.0052 | 61.02% | 69.70% | 63.59% | 62.14% | 4.72 | 66.07% | 2.3 |
| 2 | P4_KBEST_LR | 65.58% +/- 0.95 | 65.60% | 0.7030 +/- 0.0068 | 65.81% | 65.38% | 62.20% | 63.88% | 5.77 | 65.14% | 2.3 |
| 3 | P4_PCA_LR | 64.10% +/- 1.11 | 64.14% | 0.6822 +/- 0.0166 | 64.64% | 63.64% | 60.51% | 62.50% | 5.80 | 64.35% | 3.3 |
| 4 | P4_L1_LR | 63.31% +/- 1.74 | 63.28% | 0.6880 +/- 0.0070 | 62.83% | 63.73% | 59.99% | 61.31% | 5.55 | 63.91% | 1.8 |
Stability:
- P1_LR: SD of the three seed-level accuracies 1.04 pts; mean fold-to-fold SD 4.72 pts.
- P4_KBEST_LR: SD of the three seed-level accuracies 0.95 pts; mean fold-to-fold SD 5.77 pts.
- P4_PCA_LR: SD of the three seed-level accuracies 1.11 pts; mean fold-to-fold SD 5.80 pts.
- P4_L1_LR: SD of the three seed-level accuracies 1.74 pts; mean fold-to-fold SD 5.55 pts.

## 2. Paired comparison with the reference and frozen screening (R1; folds wins/ties/losses out of 30)
| Experiment | Mean gain acc. (pts) | Mean gain bacc. (pts) | Gain by seed (pts) | Fold W/T/L | Sign-test p | Paired-fold SE (pts) | Mean dAUC | Screening | Criteria failed |
|---|---|---|---|---|---|---|---|---|---|
| P4_KBEST_LR | -0.10 | +0.24 | [-0.44, 0.74, -0.59] | 13.0/4.0/13.0 | 0.819 | 0.88 | -0.0157 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30; d_mean_auc_not_lower_by_more_than_0.005 |
| P4_PCA_LR | -1.58 | -1.22 | [0.44, -1.78, -3.4] | 11.0/4.0/15.0 | 0.951 | 0.95 | -0.0365 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30; d_mean_auc_not_lower_by_more_than_0.005 |
| P4_L1_LR | -2.37 | -2.08 | [-0.15, -3.85, -3.11] | 8.0/4.0/18.0 | 0.997 | 1.17 | -0.0307 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30; d_mean_auc_not_lower_by_more_than_0.005 |
**Outcome:** No candidate passed the frozen screening rule; the reference logistic regression (P1_LR) remains the best defensible model of this phase.  (screening needs +1.5 pts accuracy and balanced accuracy, +0.5 in every seed, >= 20/30 fold wins and AUC within 0.005.) The highest mean R1 accuracy is P1_LR (65.68%); with the paired-fold SE shown above, differences below about 1 point are not distinguishable from noise.

## 3. Per-seed results (both readings)
| Experiment | Seed | Reading | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | R0 | 65.83% | 65.56% | 0.7129 | 61.98% | 69.15% | 63.40% | 62.68% | 3.22% |
| P1_LR | 0 | R1 | 64.94% | 65.00% | 0.7129 | 65.81% | 64.19% | 61.31% | 63.48% | 4.06% |
| P1_LR | 1 | R0 | 66.27% | 66.09% | 0.7202 | 63.58% | 68.60% | 63.58% | 63.58% | 4.96% |
| P1_LR | 1 | R1 | 65.24% | 64.53% | 0.7202 | 54.95% | 74.10% | 64.66% | 59.41% | 5.87% |
| P1_LR | 2 | R0 | 66.12% | 65.82% | 0.7230 | 61.66% | 69.97% | 63.91% | 62.76% | 5.85% |
| P1_LR | 2 | R1 | 66.86% | 66.55% | 0.7230 | 62.30% | 70.80% | 64.78% | 63.52% | 4.23% |
| P4_PCA_LR | 0 | R0 | 65.38% | 65.46% | 0.6804 | 66.45% | 64.46% | 61.72% | 64.00% | 5.34% |
| P4_PCA_LR | 0 | R1 | 65.38% | 65.48% | 0.6804 | 66.77% | 64.19% | 61.65% | 64.11% | 4.76% |
| P4_PCA_LR | 1 | R0 | 64.05% | 64.02% | 0.6666 | 63.58% | 64.46% | 60.67% | 62.09% | 4.81% |
| P4_PCA_LR | 1 | R1 | 63.46% | 63.54% | 0.6666 | 64.54% | 62.53% | 59.76% | 62.06% | 5.48% |
| P4_PCA_LR | 2 | R0 | 63.61% | 63.63% | 0.6997 | 63.90% | 63.36% | 60.06% | 61.92% | 6.83% |
| P4_PCA_LR | 2 | R1 | 63.46% | 63.40% | 0.6997 | 62.62% | 64.19% | 60.12% | 61.35% | 7.15% |
| P4_KBEST_LR | 0 | R0 | 63.61% | 63.56% | 0.6956 | 62.94% | 64.19% | 60.24% | 61.56% | 4.79% |
| P4_KBEST_LR | 0 | R1 | 64.50% | 64.74% | 0.6956 | 68.05% | 61.43% | 60.34% | 63.96% | 5.58% |
| P4_KBEST_LR | 1 | R0 | 65.24% | 65.10% | 0.7041 | 63.26% | 66.94% | 62.26% | 62.76% | 5.23% |
| P4_KBEST_LR | 1 | R1 | 65.98% | 65.66% | 0.7041 | 61.34% | 69.97% | 63.79% | 62.54% | 6.49% |
| P4_KBEST_LR | 2 | R0 | 66.57% | 66.36% | 0.7091 | 63.58% | 69.15% | 63.99% | 63.78% | 6.92% |
| P4_KBEST_LR | 2 | R1 | 66.27% | 66.39% | 0.7091 | 68.05% | 64.74% | 62.46% | 65.14% | 5.24% |
| P4_L1_LR | 0 | R0 | 63.76% | 63.55% | 0.6916 | 60.70% | 66.39% | 60.90% | 60.80% | 4.72% |
| P4_L1_LR | 0 | R1 | 64.79% | 64.47% | 0.6916 | 60.06% | 68.87% | 62.46% | 61.24% | 3.39% |
| P4_L1_LR | 1 | R0 | 64.20% | 64.05% | 0.6799 | 61.98% | 66.12% | 61.20% | 61.59% | 6.41% |
| P4_L1_LR | 1 | R1 | 61.39% | 61.41% | 0.6799 | 61.66% | 61.16% | 57.78% | 59.66% | 5.54% |
| P4_L1_LR | 2 | R0 | 63.76% | 63.53% | 0.6925 | 60.38% | 66.67% | 60.97% | 60.67% | 6.82% |
| P4_L1_LR | 2 | R1 | 63.76% | 63.97% | 0.6925 | 66.77% | 61.16% | 59.71% | 63.05% | 7.72% |

## 4. Confusion matrices (pooled out-of-fold predictions, R1; 313 ASD / 363 TC)
| Experiment | Seed | TP | FN | FP | TN | Accuracy | Balanced acc. | AUC |
|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | 206 | 107 | 130 | 233 | 64.94% | 65.00% | 0.7129 |
| P1_LR | 1 | 172 | 141 | 94 | 269 | 65.24% | 64.53% | 0.7202 |
| P1_LR | 2 | 195 | 118 | 106 | 257 | 66.86% | 66.55% | 0.7230 |
| P4_PCA_LR | 0 | 209 | 104 | 130 | 233 | 65.38% | 65.48% | 0.6804 |
| P4_PCA_LR | 1 | 202 | 111 | 136 | 227 | 63.46% | 63.54% | 0.6666 |
| P4_PCA_LR | 2 | 196 | 117 | 130 | 233 | 63.46% | 63.40% | 0.6997 |
| P4_KBEST_LR | 0 | 213 | 100 | 140 | 223 | 64.50% | 64.74% | 0.6956 |
| P4_KBEST_LR | 1 | 192 | 121 | 109 | 254 | 65.98% | 65.66% | 0.7041 |
| P4_KBEST_LR | 2 | 213 | 100 | 128 | 235 | 66.27% | 66.39% | 0.7091 |
| P4_L1_LR | 0 | 188 | 125 | 113 | 250 | 64.79% | 64.47% | 0.6916 |
| P4_L1_LR | 1 | 193 | 120 | 141 | 222 | 61.39% | 61.41% | 0.6799 |
| P4_L1_LR | 2 | 209 | 104 | 141 | 222 | 63.76% | 63.97% | 0.6925 |
Per-fold values, selected hyperparameters, thresholds and runtimes: `results/<experiment>_fold_metrics.csv`; per-subject predictions: `results/<experiment>_oof_predictions_seeds012.csv`; configurations: `results/<experiment>_config.json`.

## 5. Protocol verification (code-level and run-time checks)
- Every learned step is inside an sklearn `Pipeline` refitted by `GridSearchCV` on each inner-training split (demographic imputation and scaling, `StandardScaler`, in-fold `SelectKBest` for the tree models, PCA, L1 selection, classifier); the outer-test fold is only transformed and scored.
- Hyperparameters: chosen by inner 5-fold ROC-AUC on the outer-training rows only. Thresholds (R1): from `cross_val_predict` on the outer-training rows only; the outer-test scores never enter their choice.
- The engine asserts disjoint train/test indices for every fold; subject IDs are unique and no two series are identical (Stage 1); the reference run reproduced the saved baseline's scores with zero difference.
- Tangent-space (Phase 3): the geometric-mean reference is fitted on the outer-training covariances (label-free); the inner CV of that pipeline sees features built with that reference (unsupervised; no labels, no outer-test rows).
- Tree models use an in-fold k = 2000 univariate filter inside the pipeline (runtime control, predeclared); selection never sees outer-test rows.
