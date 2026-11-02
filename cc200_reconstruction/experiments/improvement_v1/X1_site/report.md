# X1 report: site robustness (baseline Arm B vs in-fold label-free harmonization H)
Runtime of this report step included in total 25.3 min. All numbers are development estimates on the 676 authorized training subjects (CC200 only); holdout and combined arrays not touched. Rule frozen in `improvement_plan_v1.md`.

## 1. 10-fold x 3-seed CV, fixed threshold 0.5 (same outer folds; baseline = saved Arm B predictions)
| Seed | Accuracy base -> H (diff pts) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|
| 0 | 65.83% -> 66.12% (+0.30) | 65.56% -> 66.06% (+0.50) | 0.7129 -> 0.7213 | 61.98% -> 65.18% | 69.15% -> 66.94% | 63.40% -> 62.96% | 62.68% -> 64.05% |
| 1 | 66.27% -> 65.53% (-0.74) | 66.09% -> 65.46% (-0.62) | 0.7202 -> 0.7154 | 63.58% -> 64.54% | 68.60% -> 66.39% | 63.58% -> 62.35% | 63.58% -> 63.42% |
| 2 | 66.12% -> 65.68% (-0.44) | 65.82% -> 65.51% (-0.30) | 0.7230 -> 0.7267 | 61.66% -> 63.26% | 69.97% -> 67.77% | 63.91% -> 62.86% | 62.76% -> 63.06% |
Mean over seeds: accuracy 66.07% -> 65.78% (-0.30 pts); balanced accuracy 65.82% -> 65.68% (-0.14 pts); AUC 0.7187 -> 0.7211.
Paired outer folds (30): H higher in 12, equal in 4, lower in 14; paired difference mean -0.30 pts, SD 3.04, SE 0.56; one-sided sign-test p for >= 12 wins: 0.900.
Confusion matrices (pooled out-of-fold predictions, 313 ASD / 363 TC):
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
| baseline | seed 0 | 194 | 119 | 112 | 251 |
| harmonized | seed 0 | 204 | 109 | 120 | 243 |
| baseline | seed 1 | 199 | 114 | 114 | 249 |
| harmonized | seed 1 | 202 | 111 | 122 | 241 |
| baseline | seed 2 | 193 | 120 | 109 | 254 |
| harmonized | seed 2 | 198 | 115 | 117 | 246 |
Fold-level comparison:
| Seed | Fold | Baseline acc. | H acc. | Diff (pts) | Selected C (H) |
|---|---|---|---|---|---|
| 0 | 0 | 67.65% | 70.59% | +2.94 | 0.001 |
| 0 | 1 | 64.71% | 67.65% | +2.94 | 0.001 |
| 0 | 2 | 70.59% | 64.71% | -5.88 | 0.001 |
| 0 | 3 | 58.82% | 64.71% | +5.88 | 0.001 |
| 0 | 4 | 66.18% | 61.76% | -4.41 | 0.001 |
| 0 | 5 | 67.65% | 66.18% | -1.47 | 0.001 |
| 0 | 6 | 65.67% | 62.69% | -2.99 | 0.001 |
| 0 | 7 | 62.69% | 67.16% | +4.48 | 0.001 |
| 0 | 8 | 67.16% | 65.67% | -1.49 | 0.001 |
| 0 | 9 | 67.16% | 70.15% | +2.99 | 0.001 |
| 1 | 0 | 63.24% | 64.71% | +1.47 | 0.001 |
| 1 | 1 | 77.94% | 77.94% | +0.00 | 0.001 |
| 1 | 2 | 61.76% | 58.82% | -2.94 | 0.001 |
| 1 | 3 | 67.65% | 64.71% | -2.94 | 0.001 |
| 1 | 4 | 61.76% | 63.24% | +1.47 | 0.001 |
| 1 | 5 | 66.18% | 67.65% | +1.47 | 0.001 |
| 1 | 6 | 62.69% | 59.70% | -2.99 | 0.001 |
| 1 | 7 | 70.15% | 73.13% | +2.99 | 0.001 |
| 1 | 8 | 67.16% | 62.69% | -4.48 | 0.001 |
| 1 | 9 | 64.18% | 62.69% | -1.49 | 10.0 |
| 2 | 0 | 66.18% | 69.12% | +2.94 | 0.001 |
| 2 | 1 | 67.65% | 70.59% | +2.94 | 0.001 |
| 2 | 2 | 61.76% | 60.29% | -1.47 | 0.001 |
| 2 | 3 | 64.71% | 64.71% | +0.00 | 0.001 |
| 2 | 4 | 69.12% | 69.12% | +0.00 | 0.001 |
| 2 | 5 | 67.65% | 64.71% | -2.94 | 0.001 |
| 2 | 6 | 76.12% | 71.64% | -4.48 | 0.001 |
| 2 | 7 | 70.15% | 67.16% | -2.99 | 0.001 |
| 2 | 8 | 53.73% | 53.73% | +0.00 | 0.001 |
| 2 | 9 | 64.18% | 65.67% | +1.49 | 0.001 |

## 2. Leave-one-site-out (19 sites; fixed 0.5; inner CV seed 0)
Pooled: baseline accuracy 65.53%, balanced 65.31%, sens 62.30%, spec 68.32%, F1 62.60%, pooled AUC 0.7243 (mixes 19 site-specific models);
H accuracy 65.68%, balanced 65.47%, sens 62.62%, spec 68.32%, F1 62.82%, pooled AUC 0.7250.
Pooled confusion (TP/FN/FP/TN): baseline 195/118/115/248; H 196/117/115/248.
Mean per-site AUC: baseline 0.711, H 0.715. Unseen-site check: maximum absolute change of the held-out site's features by the harmonizer = 0.0 in all 19 sites (unseen sites pass through unchanged).
| Site | n | ASD/TC | Base acc. | H acc. | Diff (pts) | Base AUC | H AUC | Base sens/spec | H sens/spec |
|---|---|---|---|---|---|---|---|---|---|
| NYU | 137 | 59/78 | 65.69% | 65.69% | +0.0 | 0.721 | 0.725 | 62.71% / 67.95% | 61.02% / 69.23% |
| UM_1 | 68 | 27/41 | 69.12% | 69.12% | +0.0 | 0.758 | 0.761 | 62.96% / 73.17% | 66.67% / 70.73% |
| USM | 53 | 34/19 | 79.25% | 81.13% | +1.9 | 0.848 | 0.851 | 82.35% / 73.68% | 85.29% / 73.68% |
| UCLA_1 | 52 | 30/22 | 63.46% | 67.31% | +3.8 | 0.724 | 0.732 | 56.67% / 72.73% | 66.67% / 68.18% |
| PITT | 40 | 19/21 | 70.00% | 67.50% | -2.5 | 0.789 | 0.769 | 47.37% / 90.48% | 42.11% / 90.48% |
| MAX_MUN | 37 | 15/22 | 51.35% | 48.65% | -2.7 | 0.579 | 0.552 | 46.67% / 54.55% | 46.67% / 50.00% |
| TRINITY | 35 | 15/20 | 54.29% | 54.29% | +0.0 | 0.640 | 0.653 | 73.33% / 40.00% | 73.33% / 40.00% |
| YALE | 33 | 18/15 | 81.82% | 81.82% | +0.0 | 0.885 | 0.874 | 77.78% / 86.67% | 77.78% / 86.67% |
| UM_2 | 27 | 10/17 | 85.19% | 85.19% | +0.0 | 0.824 | 0.794 | 90.00% / 82.35% | 90.00% / 82.35% |
| KKI | 27 | 10/17 | 51.85% | 55.56% | +3.7 | 0.647 | 0.706 | 80.00% / 35.29% | 90.00% / 35.29% |
| LEUVEN_2 | 23 | 10/13 | 56.52% | 60.87% | +4.3 | 0.738 | 0.746 | 50.00% / 61.54% | 40.00% / 76.92% |
| LEUVEN_1 | 22 | 11/11 | 54.55% | 54.55% | +0.0 | 0.678 | 0.636 | 18.18% / 90.91% | 18.18% / 90.91% |
| OLIN | 22 | 11/11 | 68.18% | 63.64% | -4.5 | 0.711 | 0.694 | 72.73% / 63.64% | 63.64% / 63.64% |
| SBL | 21 | 10/11 | 38.10% | 38.10% | +0.0 | 0.427 | 0.464 | 0.00% / 72.73% | 0.00% / 72.73% |
| SDSU | 21 | 6/15 | 71.43% | 71.43% | +0.0 | 0.744 | 0.767 | 50.00% / 80.00% | 50.00% / 80.00% |
| STANFORD | 20 | 10/10 | 60.00% | 60.00% | +0.0 | 0.650 | 0.650 | 80.00% / 40.00% | 80.00% / 40.00% |
| UCLA_2 | 17 | 9/8 | 82.35% | 76.47% | -5.9 | 0.819 | 0.819 | 100.00% / 62.50% | 88.89% / 62.50% |
| CALTECH | 12 | 4/8 | 58.33% | 58.33% | +0.0 | 0.625 | 0.594 | 50.00% / 62.50% | 50.00% / 62.50% |
| CMU | 9 | 5/4 | 55.56% | 55.56% | +0.0 | 0.700 | 0.800 | 20.00% / 100.00% | 20.00% / 100.00% |
Per-site values for sites with fewer than about 30 subjects are very noisy.

## 3. Harmonization diagnostic (seed-0 outer-training sets)
Mean per-feature between-site variance share: 0.0697 before -> 0.0093 after.

## 4. Frozen acceptance rule applied
| Result | Criterion |
|---|---|
| FAIL | 1a mean paired gain >= +1.0 pt accuracy |
| FAIL | 1b mean paired gain >= +1.0 pt balanced accuracy |
| FAIL | 1c every seed >= +0.5 pt (accuracy AND balanced accuracy; strictest reading) |
| FAIL | 2 accuracy higher in >= 20 of 30 paired folds |
| PASS | 3a mean AUC not lower by more than 0.005 |
| PASS | 3b mean sensitivity not lower by more than 3 pts |
| PASS | 3c mean specificity not lower by more than 3 pts |
| PASS | LOSO co-requirement: pooled accuracy not worse by more than 0.5 pt |
Provisional outcome before replication (criteria 1-4 and the LOSO co-requirement): **NOT MET -> candidate not retained; rule not changed**.
