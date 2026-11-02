# Pre-registered fresh-seed confirmation: reference P1_LR vs soft vote of (P3_TS_LR, P2_LSVM, P1_LR)
Seeds 3, 4, 5 (new outer partitions of the same 676 authorized training subjects; identical folds for all arms); CC200 only; development cross-validation, not independent-test performance. The protected holdout was **not accessed**.
Pre-registration: `preregistration.md` (criteria fixed before running; evaluated under R1 = threshold chosen from training-only inner scores; R0 = default thresholds is secondary).
Members and settings were read from the saved artifacts; the rebuilt soft vote reproduced the saved seed 0-2 predictions exactly before the fresh seeds were run. Runtime 35.6 min (members, summed fold time: P3_TS_LR 26.2 min, P2_LSVM 4.7 min, P1_LR 4.6 min); no failures.

## 1. Decision against the pre-registered criteria (R1)
| Criterion | Required | Observed | Met |
|---|---|---|---|
| 1. Mean accuracy gain over seeds 3-5 | >= +0.5 pt | +1.18 pt | yes |
| 2. Seeds with positive accuracy gain | >= 2 of 3 | 3 of 3 (gains +2.07, +1.04, +0.44 pt) | yes |
| 3. Mean ROC-AUC gain | >= +0.01 | +0.0221 (by seed +0.0340, +0.0185, +0.0138) | yes |
**PASS: the ensemble is supported by this confirmation (all three pre-registered conditions hold).**
This confirmation does not alter the campaign verdict: no candidate passed the frozen rule, and 85% was not achieved.

## 2. Per-seed results, R1 (primary)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
| reference_P1_LR | 3 | 64.35% | 63.77% | 0.6958 | 55.91% | 71.63% | 62.95% | 59.22% | 5.01% |
| reference_P1_LR | 4 | 66.86% | 66.68% | 0.7238 | 64.22% | 69.15% | 64.22% | 64.22% | 3.50% |
| reference_P1_LR | 5 | 64.64% | 64.24% | 0.7170 | 58.79% | 69.70% | 62.59% | 60.63% | 7.11% |
| soft_vote_E1 | 3 | 66.42% | 66.31% | 0.7298 | 64.86% | 67.77% | 63.44% | 64.14% | 5.54% |
| soft_vote_E1 | 4 | 67.90% | 67.84% | 0.7424 | 67.09% | 68.60% | 64.81% | 65.93% | 5.59% |
| soft_vote_E1 | 5 | 65.09% | 64.85% | 0.7309 | 61.66% | 68.04% | 62.46% | 62.06% | 8.21% |
Paired differences (ensemble minus reference): accuracy +2.07, +1.04, +0.44 pt (mean +1.18, SD over seeds 0.82); balanced accuracy mean +1.44 pt; AUC mean +0.0221.
Paired outer folds (30): ensemble higher in 20, equal in 2, lower in 8; mean fold difference +1.18 pt, SD 4.20, SE 0.77; one-sided sign-test p = 0.049 (descriptive only).
Confusion matrices, R1 (pooled out-of-fold, 313 ASD / 363 TC):
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
| reference_P1_LR | 3 | 175 | 138 | 103 | 260 |
| soft_vote_E1 | 3 | 203 | 110 | 117 | 246 |
| reference_P1_LR | 4 | 201 | 112 | 112 | 251 |
| soft_vote_E1 | 4 | 210 | 103 | 114 | 249 |
| reference_P1_LR | 5 | 184 | 129 | 110 | 253 |
| soft_vote_E1 | 5 | 193 | 120 | 116 | 247 |

## 3. Secondary: default thresholds (R0; not part of the decision)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
| reference_P1_LR | 3 | 66.27% | 66.09% | 0.6958 | 63.58% | 68.60% | 63.58% | 63.58% | 5.39% |
| reference_P1_LR | 4 | 66.57% | 66.30% | 0.7238 | 62.62% | 69.97% | 64.26% | 63.43% | 4.60% |
| reference_P1_LR | 5 | 65.24% | 64.99% | 0.7170 | 61.66% | 68.32% | 62.66% | 62.16% | 7.07% |
| soft_vote_E1 | 3 | 67.16% | 66.74% | 0.7298 | 61.02% | 72.45% | 65.64% | 63.25% | 4.98% |
| soft_vote_E1 | 4 | 68.05% | 67.61% | 0.7424 | 61.66% | 73.55% | 66.78% | 64.12% | 5.27% |
| soft_vote_E1 | 5 | 66.27% | 65.80% | 0.7309 | 59.42% | 72.18% | 64.81% | 62.00% | 8.08% |
R0 accuracy gain by seed: +0.89, +1.48, +1.04 pt (mean +1.13); mean AUC gain +0.0221.

## 4. Members for context (mean over seeds 3-5)
| Arm | Accuracy R1 | ROC-AUC | Accuracy R0 |
|---|---|---|---|
| reference_P1_LR | 65.29% | 0.7122 | 66.03% |
| member_P2_LSVM | 66.07% | 0.7190 | 64.89% |
| member_P3_TS_LR | 66.81% | 0.7200 | 67.46% |
| soft_vote_E1 | 66.47% | 0.7344 | 67.16% |
Per-seed member metrics: `seed_metrics.csv`.

## 5. Fold-level comparison (R1)
| Seed | Fold | Reference acc. | Ensemble acc. | Diff (pt) |
|---|---|---|---|---|
| 3 | 0 | 57.35% | 60.29% | +2.94 |
| 3 | 1 | 69.12% | 70.59% | +1.47 |
| 3 | 2 | 73.53% | 76.47% | +2.94 |
| 3 | 3 | 61.76% | 69.12% | +7.35 |
| 3 | 4 | 66.18% | 61.76% | -4.41 |
| 3 | 5 | 69.12% | 70.59% | +1.47 |
| 3 | 6 | 62.69% | 67.16% | +4.48 |
| 3 | 7 | 61.19% | 58.21% | -2.99 |
| 3 | 8 | 59.70% | 65.67% | +5.97 |
| 3 | 9 | 62.69% | 64.18% | +1.49 |
| 4 | 0 | 67.65% | 77.94% | +10.29 |
| 4 | 1 | 73.53% | 75.00% | +1.47 |
| 4 | 2 | 64.71% | 64.71% | +0.00 |
| 4 | 3 | 66.18% | 70.59% | +4.41 |
| 4 | 4 | 66.18% | 64.71% | -1.47 |
| 4 | 5 | 67.65% | 58.82% | -8.82 |
| 4 | 6 | 68.66% | 68.66% | +0.00 |
| 4 | 7 | 65.67% | 68.66% | +2.99 |
| 4 | 8 | 68.66% | 64.18% | -4.48 |
| 4 | 9 | 59.70% | 65.67% | +5.97 |
| 5 | 0 | 67.65% | 72.06% | +4.41 |
| 5 | 1 | 61.76% | 55.88% | -5.88 |
| 5 | 2 | 69.12% | 66.18% | -2.94 |
| 5 | 3 | 64.71% | 67.65% | +2.94 |
| 5 | 4 | 72.06% | 76.47% | +4.41 |
| 5 | 5 | 67.65% | 69.12% | +1.47 |
| 5 | 6 | 62.69% | 64.18% | +1.49 |
| 5 | 7 | 46.27% | 47.76% | +1.49 |
| 5 | 8 | 67.16% | 62.69% | -4.48 |
| 5 | 9 | 67.16% | 68.66% | +1.49 |

## 6. Integrity
Hashes of inputs, source code, configurations and checkpoints: `hashes.json` (campaign code and selection rule unchanged, equal to the recorded values). Valid subject-level outer folds for seeds 3-5 (each subject in exactly one test fold; sizes 67-68; 31-32 ASD per fold; partitions differ from seeds 0-2). Campaign outputs were only read; the campaign log was not appended to. Holdout, restricted membership file and combined arrays: not accessed.
Limitations: same 676 subjects as in seeds 0-2 (new partitions, not new subjects); 30 paired folds are not independent; accuracy SE about 1.8 pt; the conclusion concerns a small ensembling effect on this cohort only.
