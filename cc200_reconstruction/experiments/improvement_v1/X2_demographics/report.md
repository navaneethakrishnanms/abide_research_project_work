# X2 report: demographic handling (baseline Arm B vs D2 = in-fold age/sex residualization)
Total runtime 4.5 min. Development estimates on the 676 authorized training subjects (CC200 only); holdout and combined arrays not touched. Frozen rule: `improvement_plan_v1.md` section 4.

## 1. 10-fold x 3-seed CV, fixed threshold 0.5 (same outer folds; baseline = saved Arm B predictions)
| Seed | Accuracy base -> D2 (diff pts) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|
| 0 | 65.83% -> 65.98% (+0.15) | 65.56% -> 65.83% (+0.27) | 0.7129 -> 0.7128 | 61.98% -> 63.90% | 69.15% -> 67.77% | 63.40% -> 63.09% | 62.68% -> 63.49% |
| 1 | 66.27% -> 64.94% (-1.33) | 66.09% -> 64.76% (-1.33) | 0.7202 -> 0.7123 | 63.58% -> 62.30% | 68.60% -> 67.22% | 63.58% -> 62.10% | 63.58% -> 62.20% |
| 2 | 66.12% -> 65.53% (-0.59) | 65.82% -> 65.31% (-0.51) | 0.7230 -> 0.7243 | 61.66% -> 62.30% | 69.97% -> 68.32% | 63.91% -> 62.90% | 62.76% -> 62.60% |
Mean over seeds: accuracy 66.07% -> 65.48% (-0.59 pts); balanced accuracy 65.82% -> 65.30% (-0.52 pts); AUC 0.7187 -> 0.7164;
sensitivity 62.41% -> 62.83%; specificity 69.24% -> 67.77%.
Paired outer folds (30): D2 higher in 9, equal in 5, lower in 16; paired difference mean -0.58 pts, SD 2.73, SE 0.50; one-sided sign-test p for >= 9 wins: 0.992.
Confusion matrices (pooled out-of-fold predictions, 313 ASD / 363 TC):
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
| baseline | seed 0 | 194 | 119 | 112 | 251 |
| D2 residualized | seed 0 | 200 | 113 | 117 | 246 |
| baseline | seed 1 | 199 | 114 | 114 | 249 |
| D2 residualized | seed 1 | 195 | 118 | 119 | 244 |
| baseline | seed 2 | 193 | 120 | 109 | 254 |
| D2 residualized | seed 2 | 195 | 118 | 115 | 248 |
Fold-level comparison:
| Seed | Fold | Baseline acc. | D2 acc. | Diff (pts) | Selected C (D2) |
|---|---|---|---|---|---|
| 0 | 0 | 67.65% | 66.18% | -1.47 | 0.001 |
| 0 | 1 | 64.71% | 63.24% | -1.47 | 0.001 |
| 0 | 2 | 70.59% | 69.12% | -1.47 | 0.001 |
| 0 | 3 | 58.82% | 60.29% | +1.47 | 0.001 |
| 0 | 4 | 66.18% | 61.76% | -4.41 | 0.001 |
| 0 | 5 | 67.65% | 69.12% | +1.47 | 0.001 |
| 0 | 6 | 65.67% | 67.16% | +1.49 | 0.001 |
| 0 | 7 | 62.69% | 67.16% | +4.48 | 0.001 |
| 0 | 8 | 67.16% | 67.16% | +0.00 | 0.001 |
| 0 | 9 | 67.16% | 68.66% | +1.49 | 0.001 |
| 1 | 0 | 63.24% | 60.29% | -2.94 | 0.001 |
| 1 | 1 | 77.94% | 73.53% | -4.41 | 0.001 |
| 1 | 2 | 61.76% | 60.29% | -1.47 | 1.0 |
| 1 | 3 | 67.65% | 66.18% | -1.47 | 0.001 |
| 1 | 4 | 61.76% | 60.29% | -1.47 | 0.01 |
| 1 | 5 | 66.18% | 63.24% | -2.94 | 0.001 |
| 1 | 6 | 62.69% | 64.18% | +1.49 | 0.001 |
| 1 | 7 | 70.15% | 70.15% | +0.00 | 0.001 |
| 1 | 8 | 67.16% | 67.16% | +0.00 | 0.001 |
| 1 | 9 | 64.18% | 64.18% | +0.00 | 0.001 |
| 2 | 0 | 66.18% | 73.53% | +7.35 | 0.001 |
| 2 | 1 | 67.65% | 67.65% | +0.00 | 0.001 |
| 2 | 2 | 61.76% | 60.29% | -1.47 | 0.001 |
| 2 | 3 | 64.71% | 60.29% | -4.41 | 0.001 |
| 2 | 4 | 69.12% | 64.71% | -4.41 | 0.001 |
| 2 | 5 | 67.65% | 66.18% | -1.47 | 0.001 |
| 2 | 6 | 76.12% | 73.13% | -2.99 | 0.001 |
| 2 | 7 | 70.15% | 67.16% | -2.99 | 0.001 |
| 2 | 8 | 53.73% | 56.72% | +2.99 | 0.001 |
| 2 | 9 | 64.18% | 65.67% | +1.49 | 0.001 |

## 2. Residualization diagnostic (seed-0 outer-training sets)
Mean per-feature R^2 of age + sex on training rows: 0.0086. Mean |correlation(feature, age)| on held-out rows: 0.1110 before -> 0.1030 after (subsample of every 20th feature).

## 3. Frozen acceptance rule applied
| Result | Criterion |
|---|---|
| FAIL | 1a mean paired gain >= +1.0 pt accuracy |
| FAIL | 1b mean paired gain >= +1.0 pt balanced accuracy |
| FAIL | 1c every seed >= +0.5 pt (accuracy AND balanced accuracy; strictest reading) |
| FAIL | 2 accuracy higher in >= 20 of 30 paired folds |
| PASS | 3a mean AUC not lower by more than 0.005 |
| PASS | 3b mean sensitivity not lower by more than 3 pts |
| PASS | 3c mean specificity not lower by more than 3 pts |
Provisional outcome (criteria 1-4): **NOT MET -> variant not retained; rule not changed**.
