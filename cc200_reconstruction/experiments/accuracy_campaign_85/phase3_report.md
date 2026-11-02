# Phase 3 report: CC200 accuracy campaign (reference = P1_LR, campaign reproduction of the baseline; exact match to the saved baseline verified)
Data: 676 authorized training subjects (313 ASD / 363 TC); outer folds 10 x seeds 0, 1, 2 (identical for all rows); all numbers are development cross-validation estimates (mean over 3 seeds), not independent-test performance. Holdout untouched.
Readings: **R1** = threshold chosen on inner out-of-fold scores of the outer-training partition (primary); R0 = default threshold (0.5 probability / 0 margin). Note: the reference under R0 is the saved baseline (66.07%); under R1 it is 65.68%.
All requested experiments completed.

## 1. Ranking by mean R1 accuracy (primary), with stability
| Rank | Experiment | Accuracy R1 (mean +/- SD over seeds) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD (pts) | Accuracy R0 | Runtime (min per 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P3_TS_LR | 66.27% +/- 0.78 | 65.84% | 0.7141 +/- 0.0088 | 59.96% | 71.72% | 64.64% | 62.19% | 4.50 | 66.57% | 12.9 |
| 2 | P1_LR | 65.68% +/- 1.04 | 65.36% | 0.7187 +/- 0.0052 | 61.02% | 69.70% | 63.59% | 62.14% | 4.72 | 66.07% | 2.3 |
| 3 | P3_PC_LR | 60.70% +/- 0.87 | 60.61% | 0.6672 +/- 0.0044 | 59.32% | 61.89% | 57.29% | 58.27% | 5.85 | 62.67% | 1.7 |
Stability:
- P3_TS_LR: SD of the three seed-level accuracies 0.78 pts; mean fold-to-fold SD 4.50 pts.
- P1_LR: SD of the three seed-level accuracies 1.04 pts; mean fold-to-fold SD 4.72 pts.
- P3_PC_LR: SD of the three seed-level accuracies 0.87 pts; mean fold-to-fold SD 5.85 pts.

## 2. Paired comparison with the reference and frozen screening (R1; folds wins/ties/losses out of 30)
| Experiment | Mean gain acc. (pts) | Mean gain bacc. (pts) | Gain by seed (pts) | Fold W/T/L | Sign-test p | Paired-fold SE (pts) | Mean dAUC | Screening | Criteria failed |
|---|---|---|---|---|---|---|---|---|---|
| P3_TS_LR | +0.59 | +0.48 | [0.74, 1.92, -0.89] | 13.0/3.0/14.0 | 0.819 | 0.74 | -0.0045 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30 |
| P3_PC_LR | -4.98 | -4.75 | [-4.59, -3.55, -6.8] | 4.0/2.0/24.0 | 1.000 | 1.06 | -0.0515 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30; d_mean_auc_not_lower_by_more_than_0.005 |
**Outcome:** No candidate passed the frozen screening rule; the reference logistic regression (P1_LR) remains the best defensible model of this phase.  (screening needs +1.5 pts accuracy and balanced accuracy, +0.5 in every seed, >= 20/30 fold wins and AUC within 0.005.) The highest mean R1 accuracy is P3_TS_LR (66.27%); with the paired-fold SE shown above, differences below about 1 point are not distinguishable from noise.

## 3. Per-seed results (both readings)
| Experiment | Seed | Reading | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | R0 | 65.83% | 65.56% | 0.7129 | 61.98% | 69.15% | 63.40% | 62.68% | 3.22% |
| P1_LR | 0 | R1 | 64.94% | 65.00% | 0.7129 | 65.81% | 64.19% | 61.31% | 63.48% | 4.06% |
| P1_LR | 1 | R0 | 66.27% | 66.09% | 0.7202 | 63.58% | 68.60% | 63.58% | 63.58% | 4.96% |
| P1_LR | 1 | R1 | 65.24% | 64.53% | 0.7202 | 54.95% | 74.10% | 64.66% | 59.41% | 5.87% |
| P1_LR | 2 | R0 | 66.12% | 65.82% | 0.7230 | 61.66% | 69.97% | 63.91% | 62.76% | 5.85% |
| P1_LR | 2 | R1 | 66.86% | 66.55% | 0.7230 | 62.30% | 70.80% | 64.78% | 63.52% | 4.23% |
| P3_PC_LR | 0 | R0 | 61.09% | 60.25% | 0.6650 | 48.88% | 71.63% | 59.77% | 53.78% | 6.60% |
| P3_PC_LR | 0 | R1 | 60.36% | 60.31% | 0.6650 | 59.74% | 60.88% | 56.84% | 58.26% | 6.22% |
| P3_PC_LR | 1 | R0 | 64.20% | 63.32% | 0.6722 | 51.44% | 75.21% | 64.14% | 57.09% | 4.79% |
| P3_PC_LR | 1 | R1 | 61.69% | 61.71% | 0.6722 | 61.98% | 61.43% | 58.08% | 59.97% | 4.80% |
| P3_PC_LR | 2 | R0 | 62.72% | 61.79% | 0.6643 | 49.20% | 74.38% | 62.35% | 55.00% | 5.00% |
| P3_PC_LR | 2 | R1 | 60.06% | 59.80% | 0.6643 | 56.23% | 63.36% | 56.96% | 56.59% | 6.51% |
| P3_TS_LR | 0 | R0 | 65.68% | 65.07% | 0.7135 | 56.87% | 73.28% | 64.73% | 60.54% | 4.32% |
| P3_TS_LR | 0 | R1 | 65.68% | 65.27% | 0.7135 | 59.74% | 70.80% | 63.82% | 61.72% | 4.28% |
| P3_TS_LR | 1 | R0 | 67.01% | 66.53% | 0.7233 | 60.06% | 73.00% | 65.73% | 62.77% | 3.11% |
| P3_TS_LR | 1 | R1 | 67.16% | 66.85% | 0.7233 | 62.62% | 71.07% | 65.12% | 63.84% | 4.21% |
| P3_TS_LR | 2 | R0 | 67.01% | 66.53% | 0.7057 | 60.06% | 73.00% | 65.73% | 62.77% | 6.49% |
| P3_TS_LR | 2 | R1 | 65.98% | 65.39% | 0.7057 | 57.51% | 73.28% | 64.98% | 61.02% | 5.02% |

## 4. Confusion matrices (pooled out-of-fold predictions, R1; 313 ASD / 363 TC)
| Experiment | Seed | TP | FN | FP | TN | Accuracy | Balanced acc. | AUC |
|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | 206 | 107 | 130 | 233 | 64.94% | 65.00% | 0.7129 |
| P1_LR | 1 | 172 | 141 | 94 | 269 | 65.24% | 64.53% | 0.7202 |
| P1_LR | 2 | 195 | 118 | 106 | 257 | 66.86% | 66.55% | 0.7230 |
| P3_PC_LR | 0 | 187 | 126 | 142 | 221 | 60.36% | 60.31% | 0.6650 |
| P3_PC_LR | 1 | 194 | 119 | 140 | 223 | 61.69% | 61.71% | 0.6722 |
| P3_PC_LR | 2 | 176 | 137 | 133 | 230 | 60.06% | 59.80% | 0.6643 |
| P3_TS_LR | 0 | 187 | 126 | 106 | 257 | 65.68% | 65.27% | 0.7135 |
| P3_TS_LR | 1 | 196 | 117 | 105 | 258 | 67.16% | 66.85% | 0.7233 |
| P3_TS_LR | 2 | 180 | 133 | 97 | 266 | 65.98% | 65.39% | 0.7057 |
Per-fold values, selected hyperparameters, thresholds and runtimes: `results/<experiment>_fold_metrics.csv`; per-subject predictions: `results/<experiment>_oof_predictions_seeds012.csv`; configurations: `results/<experiment>_config.json`.

## 5. Protocol verification (code-level and run-time checks)
- Every learned step is inside an sklearn `Pipeline` refitted by `GridSearchCV` on each inner-training split (demographic imputation and scaling, `StandardScaler`, in-fold `SelectKBest` for the tree models, PCA, L1 selection, classifier); the outer-test fold is only transformed and scored.
- Hyperparameters: chosen by inner 5-fold ROC-AUC on the outer-training rows only. Thresholds (R1): from `cross_val_predict` on the outer-training rows only; the outer-test scores never enter their choice.
- The engine asserts disjoint train/test indices for every fold; subject IDs are unique and no two series are identical (Stage 1); the reference run reproduced the saved baseline's scores with zero difference.
- Tangent-space (Phase 3): the geometric-mean reference is fitted on the outer-training covariances (label-free); the inner CV of that pipeline sees features built with that reference (unsupervised; no labels, no outer-test rows).
- Tree models use an in-fold k = 2000 univariate filter inside the pipeline (runtime control, predeclared); selection never sees outer-test rows.
