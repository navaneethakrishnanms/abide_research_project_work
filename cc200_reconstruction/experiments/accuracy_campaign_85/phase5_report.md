# Phase 5 report: CC200 accuracy campaign (reference = P1_LR, campaign reproduction of the baseline; exact match to the saved baseline verified)
Data: 676 authorized training subjects (313 ASD / 363 TC); outer folds 10 x seeds 0, 1, 2 (identical for all rows); all numbers are development cross-validation estimates (mean over 3 seeds), not independent-test performance. Holdout untouched.
Readings: **R1** = threshold chosen on inner out-of-fold scores of the outer-training partition (primary); R0 = default threshold (0.5 probability / 0 margin). Note: the reference under R0 is the saved baseline (66.07%); under R1 it is 65.68%.
All requested experiments completed.

## 1. Ranking by mean R1 accuracy (primary), with stability
| Rank | Experiment | Accuracy R1 (mean +/- SD over seeds) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD (pts) | Accuracy R0 | Runtime (min per 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P5_E1_SOFTVOTE | 66.42% +/- 0.78 | 66.30% | 0.7321 +/- 0.0050 | 64.64% | 67.95% | 63.53% | 64.06% | 4.32 | 66.96% | 0.0 |
| 2 | P5_E2_STACK | 65.83% +/- 0.26 | 65.53% | 0.7329 +/- 0.0041 | 61.45% | 69.61% | 63.61% | 62.44% | 4.52 | 66.12% | 0.0 |
| 3 | P1_LR | 65.68% +/- 1.04 | 65.36% | 0.7187 +/- 0.0052 | 61.02% | 69.70% | 63.59% | 62.14% | 4.72 | 66.07% | 2.3 |
Stability:
- P5_E1_SOFTVOTE: SD of the three seed-level accuracies 0.78 pts; mean fold-to-fold SD 4.32 pts.
- P5_E2_STACK: SD of the three seed-level accuracies 0.26 pts; mean fold-to-fold SD 4.52 pts.
- P1_LR: SD of the three seed-level accuracies 1.04 pts; mean fold-to-fold SD 4.72 pts.

## 2. Paired comparison with the reference and frozen screening (R1; folds wins/ties/losses out of 30)
| Experiment | Mean gain acc. (pts) | Mean gain bacc. (pts) | Gain by seed (pts) | Fold W/T/L | Sign-test p | Paired-fold SE (pts) | Mean dAUC | Screening | Criteria failed |
|---|---|---|---|---|---|---|---|---|---|
| P5_E1_SOFTVOTE | +0.74 | +0.94 | [0.89, 0.89, 0.44] | 17.0/3.0/10.0 | 0.292 | 0.53 | +0.0134 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30 |
| P5_E2_STACK | +0.15 | +0.17 | [0.74, 0.44, -0.74] | 13.0/7.0/10.0 | 0.819 | 0.53 | +0.0142 | no | a_mean_gain_accuracy>=1.5pt; a_mean_gain_balanced_accuracy>=1.5pt; b_every_seed_accuracy_gain>=0.5pt; c_fold_wins>=20_of_30 |
**Outcome:** No candidate passed the frozen screening rule; the reference logistic regression (P1_LR) remains the best defensible model of this phase.  (screening needs +1.5 pts accuracy and balanced accuracy, +0.5 in every seed, >= 20/30 fold wins and AUC within 0.005.) The highest mean R1 accuracy is P5_E1_SOFTVOTE (66.42%); with the paired-fold SE shown above, differences below about 1 point are not distinguishable from noise.

## 3. Per-seed results (both readings)
| Experiment | Seed | Reading | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | R0 | 65.83% | 65.56% | 0.7129 | 61.98% | 69.15% | 63.40% | 62.68% | 3.22% |
| P1_LR | 0 | R1 | 64.94% | 65.00% | 0.7129 | 65.81% | 64.19% | 61.31% | 63.48% | 4.06% |
| P1_LR | 1 | R0 | 66.27% | 66.09% | 0.7202 | 63.58% | 68.60% | 63.58% | 63.58% | 4.96% |
| P1_LR | 1 | R1 | 65.24% | 64.53% | 0.7202 | 54.95% | 74.10% | 64.66% | 59.41% | 5.87% |
| P1_LR | 2 | R0 | 66.12% | 65.82% | 0.7230 | 61.66% | 69.97% | 63.91% | 62.76% | 5.85% |
| P1_LR | 2 | R1 | 66.86% | 66.55% | 0.7230 | 62.30% | 70.80% | 64.78% | 63.52% | 4.23% |
| P5_E1_SOFTVOTE | 0 | R0 | 67.16% | 66.72% | 0.7265 | 60.70% | 72.73% | 65.74% | 63.12% | 3.60% |
| P5_E1_SOFTVOTE | 0 | R1 | 65.83% | 65.89% | 0.7265 | 66.77% | 65.01% | 62.20% | 64.41% | 4.58% |
| P5_E1_SOFTVOTE | 1 | R0 | 66.27% | 65.71% | 0.7358 | 58.15% | 73.28% | 65.23% | 61.49% | 4.47% |
| P5_E1_SOFTVOTE | 1 | R1 | 66.12% | 65.90% | 0.7358 | 62.94% | 68.87% | 63.55% | 63.24% | 4.05% |
| P5_E1_SOFTVOTE | 2 | R0 | 67.46% | 66.97% | 0.7341 | 60.38% | 73.55% | 66.32% | 63.21% | 4.14% |
| P5_E1_SOFTVOTE | 2 | R1 | 67.31% | 67.09% | 0.7341 | 64.22% | 69.97% | 64.84% | 64.53% | 4.34% |
| P5_E2_STACK | 0 | R0 | 64.94% | 64.78% | 0.7286 | 62.62% | 66.94% | 62.03% | 62.32% | 4.29% |
| P5_E2_STACK | 0 | R1 | 65.68% | 65.14% | 0.7286 | 57.83% | 72.45% | 64.41% | 60.94% | 3.94% |
| P5_E2_STACK | 1 | R0 | 66.86% | 66.84% | 0.7368 | 66.45% | 67.22% | 63.61% | 65.00% | 4.82% |
| P5_E2_STACK | 1 | R1 | 65.68% | 65.69% | 0.7368 | 65.81% | 65.56% | 62.24% | 63.98% | 5.10% |
| P5_E2_STACK | 2 | R0 | 66.57% | 66.58% | 0.7332 | 66.77% | 66.39% | 63.14% | 64.91% | 6.03% |
| P5_E2_STACK | 2 | R1 | 66.12% | 65.75% | 0.7332 | 60.70% | 70.80% | 64.19% | 62.40% | 4.52% |

## 4. Confusion matrices (pooled out-of-fold predictions, R1; 313 ASD / 363 TC)
| Experiment | Seed | TP | FN | FP | TN | Accuracy | Balanced acc. | AUC |
|---|---|---|---|---|---|---|---|---|
| P1_LR | 0 | 206 | 107 | 130 | 233 | 64.94% | 65.00% | 0.7129 |
| P1_LR | 1 | 172 | 141 | 94 | 269 | 65.24% | 64.53% | 0.7202 |
| P1_LR | 2 | 195 | 118 | 106 | 257 | 66.86% | 66.55% | 0.7230 |
| P5_E1_SOFTVOTE | 0 | 209 | 104 | 127 | 236 | 65.83% | 65.89% | 0.7265 |
| P5_E1_SOFTVOTE | 1 | 197 | 116 | 113 | 250 | 66.12% | 65.90% | 0.7358 |
| P5_E1_SOFTVOTE | 2 | 201 | 112 | 109 | 254 | 67.31% | 67.09% | 0.7341 |
| P5_E2_STACK | 0 | 181 | 132 | 100 | 263 | 65.68% | 65.14% | 0.7286 |
| P5_E2_STACK | 1 | 206 | 107 | 125 | 238 | 65.68% | 65.69% | 0.7368 |
| P5_E2_STACK | 2 | 190 | 123 | 106 | 257 | 66.12% | 65.75% | 0.7332 |
Per-fold values, selected hyperparameters, thresholds and runtimes: `results/<experiment>_fold_metrics.csv`; per-subject predictions: `results/<experiment>_oof_predictions_seeds012.csv`; configurations: `results/<experiment>_config.json`.

## 5. Protocol verification (code-level and run-time checks)
- Every learned step is inside an sklearn `Pipeline` refitted by `GridSearchCV` on each inner-training split (demographic imputation and scaling, `StandardScaler`, in-fold `SelectKBest` for the tree models, PCA, L1 selection, classifier); the outer-test fold is only transformed and scored.
- Hyperparameters: chosen by inner 5-fold ROC-AUC on the outer-training rows only. Thresholds (R1): from `cross_val_predict` on the outer-training rows only; the outer-test scores never enter their choice.
- The engine asserts disjoint train/test indices for every fold; subject IDs are unique and no two series are identical (Stage 1); the reference run reproduced the saved baseline's scores with zero difference.
- Tangent-space (Phase 3): the geometric-mean reference is fitted on the outer-training covariances (label-free); the inner CV of that pipeline sees features built with that reference (unsupervised; no labels, no outer-test rows).
- Tree models use an in-fold k = 2000 univariate filter inside the pipeline (runtime control, predeclared); selection never sees outer-test rows.
