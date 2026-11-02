# Strictly nested tangent-space workflow: final report
CC200 only; 676 authorized training subjects (313 ASD / 363 TC); development cross-validation (10 folds x 3 seeds per stage); the protected holdout was **not accessed**. 85% is the project's target and was **not achieved**; the best accuracies below are in the mid-60s to about 67%.
Plan and frozen criteria: `experiment_plan.md`; method and validity evidence: `methodology.md`; numbers: `results.csv`, `predictions.csv`.

## 0. Stage status
- Stage A: exploratory (seeds 0-2; seeds used earlier for model comparison): RUN
- Stage B: formal confirmation (seeds 6-8; never evaluated before): RUN
- Stage C: supplementary (seeds 3-5; the old tangent model was already seen there, so NOT independent for this hypothesis): RUN

## 1. Priority 1: the tangent-space reference was corrected and the correction was validated
- The old pipeline fitted the tangent reference once on all outer-training rows, so inner-validation subjects had contributed to the reference that transformed them. The corrected evaluator refits the reference inside every inner training split and never on validation or test rows.
- Validation: with the reference scope switched back to the old behaviour, the new evaluator reproduces the saved `P3_TS_LR` results **exactly** (zero difference in scores, selected C, thresholds and inner out-of-fold scores on two real folds); 5/5 synthetic tests show that perturbing inner-validation or outer-test rows changes no fitted reference or any other row's output, and that no outer-test labels reach the evaluator.
- Effect size (see diagnostics under each stage): the strictly inner references differ only slightly from the outer one, so the correction is expected to change results little; the stage tables measure that directly (TS-N minus TS-old).

## Stage A: exploratory (seeds 0-2; seeds used earlier for model comparison)
### Accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, precision, F1 (R1 = threshold from training-only inner scores; primary)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
| LR | 0 | 64.94% | 65.00% | 0.7129 | 65.81% | 64.19% | 61.31% | 63.48% | 4.06% |
| LR | 1 | 65.24% | 64.53% | 0.7202 | 54.95% | 74.10% | 64.66% | 59.41% | 5.87% |
| LR | 2 | 66.86% | 66.55% | 0.7230 | 62.30% | 70.80% | 64.78% | 63.52% | 4.23% |
| TS-old | 0 | 65.68% | 65.27% | 0.7135 | 59.74% | 70.80% | 63.82% | 61.72% | 4.28% |
| TS-old | 1 | 67.16% | 66.85% | 0.7233 | 62.62% | 71.07% | 65.12% | 63.84% | 4.21% |
| TS-old | 2 | 65.98% | 65.39% | 0.7057 | 57.51% | 73.28% | 64.98% | 61.02% | 5.02% |
| TS-N | 0 | 65.53% | 65.18% | 0.7152 | 60.38% | 69.97% | 63.42% | 61.87% | 4.21% |
| TS-N | 1 | 67.90% | 67.60% | 0.7195 | 63.58% | 71.63% | 65.89% | 64.72% | 4.28% |
| TS-N | 2 | 66.27% | 65.71% | 0.7162 | 58.15% | 73.28% | 65.23% | 61.49% | 4.81% |
| H3 soft vote LR+TS-N | 0 | 64.79% | 64.75% | 0.7318 | 64.22% | 65.29% | 61.47% | 62.81% | 3.76% |
| H3 soft vote LR+TS-N | 1 | 65.68% | 65.51% | 0.7402 | 63.26% | 67.77% | 62.86% | 63.06% | 5.89% |
| H3 soft vote LR+TS-N | 2 | 67.90% | 67.47% | 0.7401 | 61.66% | 73.28% | 66.55% | 64.01% | 4.86% |
Secondary (R0, default thresholds): see `results.csv`; paired R0 accuracy gains are in the last column below.
### Confusion matrices (R1; pooled out-of-fold, 313 ASD / 363 TC)
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
| LR | 0 | 206 | 107 | 130 | 233 |
| LR | 1 | 172 | 141 | 94 | 269 |
| LR | 2 | 195 | 118 | 106 | 257 |
| TS-old | 0 | 187 | 126 | 106 | 257 |
| TS-old | 1 | 196 | 117 | 105 | 258 |
| TS-old | 2 | 180 | 133 | 97 | 266 |
| TS-N | 0 | 189 | 124 | 109 | 254 |
| TS-N | 1 | 199 | 114 | 103 | 260 |
| TS-N | 2 | 182 | 131 | 97 | 266 |
| H3 | 0 | 201 | 112 | 126 | 237 |
| H3 | 1 | 198 | 115 | 117 | 246 |
| H3 | 2 | 193 | 120 | 97 | 266 |
### Paired comparisons (R1; gains in percentage points; fold W/T/L out of 30)
| Comparison | Accuracy gain by seed | Mean accuracy gain | Mean balanced-acc. gain | Mean AUC gain | Fold W/T/L | Mean fold diff (SE) | R0 mean acc. gain |
|---|---|---|---|---|---|---|---|
| H1: TS-N minus LR | +0.59, +2.66, -0.59 | +0.89 (SD 1.65) | +0.80 | -0.0017 | 14/2/14 | +0.89 (SE 0.72) | +0.49 |
| Correction effect: TS-N minus TS-old | -0.15, +0.74, +0.30 | +0.30 (SD 0.44) | +0.33 | +0.0028 | 7/19/4 | +0.30 (SE 0.22) | -0.00 |
| H3: soft vote minus LR | -0.15, +0.44, +1.04 | +0.44 (SD 0.59) | +0.55 | +0.0187 | 15/5/10 | +0.45 (SE 0.56) | +1.04 |
| H3 minus TS-N (informational) | -0.74, -2.22, +1.63 | -0.44 (SD 1.94) | -0.25 | +0.0204 | 12/4/14 | -0.44 (SE 0.75) | +0.54 |
### Frozen criteria evaluated on this stage
| Hypothesis | Condition | Observed | Met |
|---|---|---|---|
| H1 (TS-N vs LR) | mean accuracy gain >= +0.5 pt | +0.89 | yes |
| | positive in >= 2 of 3 seeds | 2 of 3 | yes |
| | mean AUC change >= -0.005 | -0.0017 | yes |
| H3 (soft vote vs LR) | mean accuracy gain >= +0.5 pt | +0.44 | NO |
| | positive in >= 2 of 3 seeds | 2 of 3 | yes |
| | mean AUC gain >= +0.01 | +0.0187 | yes |
H1: **PASS**; H3: **FAIL** (descriptive only: this stage is not the pre-registered decision stage)
Strict-nesting diagnostics (means over folds): affine-invariant distance between an inner reference and the outer reference 0.283; relative change of the validation features 2.1%; CPU time for the TS-N fits 366 min.

## Stage B: formal confirmation (seeds 6-8; never evaluated before)
### Accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, precision, F1 (R1 = threshold from training-only inner scores; primary)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
| LR | 6 | 66.72% | 66.35% | 0.7324 | 61.34% | 71.35% | 64.86% | 63.05% | 5.42% |
| LR | 7 | 66.57% | 66.45% | 0.7237 | 64.86% | 68.04% | 63.64% | 64.24% | 3.90% |
| LR | 8 | 66.42% | 65.83% | 0.7285 | 57.83% | 73.83% | 65.58% | 61.46% | 3.58% |
| TS-N | 6 | 66.72% | 66.70% | 0.7164 | 66.45% | 66.94% | 63.41% | 64.90% | 3.90% |
| TS-N | 7 | 66.27% | 65.78% | 0.7202 | 59.11% | 72.45% | 64.91% | 61.87% | 3.20% |
| TS-N | 8 | 68.64% | 68.09% | 0.7273 | 60.70% | 75.48% | 68.10% | 64.19% | 4.38% |
| H3 soft vote LR+TS-N | 6 | 67.60% | 67.35% | 0.7451 | 63.90% | 70.80% | 65.36% | 64.62% | 6.98% |
| H3 soft vote LR+TS-N | 7 | 67.46% | 67.54% | 0.7441 | 68.69% | 66.39% | 63.80% | 66.15% | 1.82% |
| H3 soft vote LR+TS-N | 8 | 68.05% | 68.00% | 0.7527 | 67.41% | 68.60% | 64.92% | 66.14% | 5.48% |
Secondary (R0, default thresholds): see `results.csv`; paired R0 accuracy gains are in the last column below.
### Confusion matrices (R1; pooled out-of-fold, 313 ASD / 363 TC)
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
| LR | 6 | 192 | 121 | 104 | 259 |
| LR | 7 | 203 | 110 | 116 | 247 |
| LR | 8 | 181 | 132 | 95 | 268 |
| TS-N | 6 | 208 | 105 | 120 | 243 |
| TS-N | 7 | 185 | 128 | 100 | 263 |
| TS-N | 8 | 190 | 123 | 89 | 274 |
| H3 | 6 | 200 | 113 | 106 | 257 |
| H3 | 7 | 215 | 98 | 122 | 241 |
| H3 | 8 | 211 | 102 | 114 | 249 |
### Paired comparisons (R1; gains in percentage points; fold W/T/L out of 30)
| Comparison | Accuracy gain by seed | Mean accuracy gain | Mean balanced-acc. gain | Mean AUC gain | Fold W/T/L | Mean fold diff (SE) | R0 mean acc. gain |
|---|---|---|---|---|---|---|---|
| H1: TS-N minus LR | +0.00, -0.30, +2.22 | +0.64 (SD 1.37) | +0.65 | -0.0069 | 16/3/11 | +0.64 (SE 0.88) | +1.58 |
| H3: soft vote minus LR | +0.89, +0.89, +1.63 | +1.13 (SD 0.43) | +1.42 | +0.0191 | 14/6/10 | +1.14 (SE 0.76) | +2.37 |
| H3 minus TS-N (informational) | +0.89, +1.18, -0.59 | +0.49 (SD 0.95) | +0.77 | +0.0260 | 13/4/13 | +0.50 (SE 1.02) | +0.79 |
### Frozen criteria evaluated on this stage
| Hypothesis | Condition | Observed | Met |
|---|---|---|---|
| H1 (TS-N vs LR) | mean accuracy gain >= +0.5 pt | +0.64 | yes |
| | positive in >= 2 of 3 seeds | 1 of 3 | NO |
| | mean AUC change >= -0.005 | -0.0069 | NO |
| H3 (soft vote vs LR) | mean accuracy gain >= +0.5 pt | +1.13 | yes |
| | positive in >= 2 of 3 seeds | 3 of 3 | yes |
| | mean AUC gain >= +0.01 | +0.0191 | yes |
H1: **FAIL**; H3: **PASS**
Strict-nesting diagnostics (means over folds): affine-invariant distance between an inner reference and the outer reference 0.284; relative change of the validation features 2.1%; CPU time for the TS-N fits 355 min.

## Stage C: supplementary (seeds 3-5; the old tangent model was already seen there, so NOT independent for this hypothesis)
### Accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, precision, F1 (R1 = threshold from training-only inner scores; primary)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
| LR | 3 | 64.35% | 63.77% | 0.6958 | 55.91% | 71.63% | 62.95% | 59.22% | 5.01% |
| LR | 4 | 66.86% | 66.68% | 0.7238 | 64.22% | 69.15% | 64.22% | 64.22% | 3.50% |
| LR | 5 | 64.64% | 64.24% | 0.7170 | 58.79% | 69.70% | 62.59% | 60.63% | 7.11% |
| TS-old | 3 | 66.42% | 66.07% | 0.7210 | 61.34% | 70.80% | 64.43% | 62.85% | 4.87% |
| TS-old | 4 | 67.01% | 66.62% | 0.7226 | 61.34% | 71.90% | 65.31% | 63.26% | 4.94% |
| TS-old | 5 | 67.01% | 66.91% | 0.7164 | 65.50% | 68.32% | 64.06% | 64.77% | 6.33% |
| TS-N | 3 | 66.57% | 66.21% | 0.7210 | 61.34% | 71.07% | 64.65% | 62.95% | 5.11% |
| TS-N | 4 | 67.01% | 66.58% | 0.7206 | 60.70% | 72.45% | 65.52% | 63.02% | 4.92% |
| TS-N | 5 | 66.72% | 66.48% | 0.7173 | 63.26% | 69.70% | 64.29% | 63.77% | 5.07% |
| H3 soft vote LR+TS-N | 3 | 68.05% | 67.92% | 0.7313 | 66.13% | 69.70% | 65.30% | 65.71% | 5.36% |
| H3 soft vote LR+TS-N | 4 | 67.16% | 67.46% | 0.7421 | 71.57% | 63.36% | 62.75% | 66.87% | 5.47% |
| H3 soft vote LR+TS-N | 5 | 65.09% | 65.18% | 0.7338 | 66.45% | 63.91% | 61.36% | 63.80% | 7.09% |
Secondary (R0, default thresholds): see `results.csv`; paired R0 accuracy gains are in the last column below.
### Confusion matrices (R1; pooled out-of-fold, 313 ASD / 363 TC)
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
| LR | 3 | 175 | 138 | 103 | 260 |
| LR | 4 | 201 | 112 | 112 | 251 |
| LR | 5 | 184 | 129 | 110 | 253 |
| TS-old | 3 | 192 | 121 | 106 | 257 |
| TS-old | 4 | 192 | 121 | 102 | 261 |
| TS-old | 5 | 205 | 108 | 115 | 248 |
| TS-N | 3 | 192 | 121 | 105 | 258 |
| TS-N | 4 | 190 | 123 | 100 | 263 |
| TS-N | 5 | 198 | 115 | 110 | 253 |
| H3 | 3 | 207 | 106 | 110 | 253 |
| H3 | 4 | 224 | 89 | 133 | 230 |
| H3 | 5 | 208 | 105 | 131 | 232 |
### Paired comparisons (R1; gains in percentage points; fold W/T/L out of 30)
| Comparison | Accuracy gain by seed | Mean accuracy gain | Mean balanced-acc. gain | Mean AUC gain | Fold W/T/L | Mean fold diff (SE) | R0 mean acc. gain |
|---|---|---|---|---|---|---|---|
| H1: TS-N minus LR | +2.22, +0.15, +2.07 | +1.48 (SD 1.16) | +1.52 | +0.0074 | 16/3/11 | +1.50 (SE 1.08) | +1.43 |
| Correction effect: TS-N minus TS-old | +0.15, +0.00, -0.30 | -0.05 (SD 0.23) | -0.11 | -0.0004 | 8/16/6 | -0.05 (SE 0.34) | +0.00 |
| H3: soft vote minus LR | +3.70, +0.30, +0.44 | +1.48 (SD 1.92) | +1.96 | +0.0235 | 18/1/11 | +1.48 (SE 0.82) | +1.38 |
| H3 minus TS-N (informational) | +1.48, +0.15, -1.63 | +0.00 (SD 1.56) | +0.43 | +0.0161 | 14/3/13 | -0.01 (SE 0.81) | -0.05 |
### Frozen criteria evaluated on this stage
| Hypothesis | Condition | Observed | Met |
|---|---|---|---|
| H1 (TS-N vs LR) | mean accuracy gain >= +0.5 pt | +1.48 | yes |
| | positive in >= 2 of 3 seeds | 3 of 3 | yes |
| | mean AUC change >= -0.005 | +0.0074 | yes |
| H3 (soft vote vs LR) | mean accuracy gain >= +0.5 pt | +1.48 | yes |
| | positive in >= 2 of 3 seeds | 3 of 3 | yes |
| | mean AUC gain >= +0.01 | +0.0235 | yes |
H1: **PASS**; H3: **PASS** (descriptive only: this stage is not the pre-registered decision stage)
Strict-nesting diagnostics (means over folds): affine-invariant distance between an inner reference and the outer reference 0.286; relative change of the validation features 2.1%; CPU time for the TS-N fits 351 min.


## 2. Conclusions
### Absolute levels (R1 primary; mean over each stage's three seeds)
| Stage | Arm | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Accuracy R0 |
|---|---|---|---|---|---|---|---|
| Stage A | LR | 65.68% | 65.36% | 0.7187 | 61.02% | 69.70% | 66.07% |
| Stage A | TS-old | 66.27% | 65.84% | 0.7141 | 59.96% | 71.72% | 66.57% |
| Stage A | TS-N | 66.57% | 66.16% | 0.7170 | 60.70% | 71.63% | 66.57% |
| Stage A | H3 soft vote LR+TS-N | 66.12% | 65.91% | 0.7373 | 63.05% | 68.78% | 67.11% |
| Stage B | LR | 66.57% | 66.21% | 0.7282 | 61.34% | 71.07% | 66.91% |
| Stage B | TS-N | 67.21% | 66.86% | 0.7213 | 62.09% | 71.63% | 68.49% |
| Stage B | H3 soft vote LR+TS-N | 67.70% | 67.63% | 0.7473 | 66.67% | 68.60% | 69.28% |
| Stage C | LR | 65.29% | 64.90% | 0.7122 | 59.64% | 70.16% | 66.03% |
| Stage C | TS-old | 66.81% | 66.53% | 0.7200 | 62.73% | 70.34% | 67.46% |
| Stage C | TS-N | 66.77% | 66.42% | 0.7196 | 61.77% | 71.07% | 67.46% |
| Stage C | H3 soft vote LR+TS-N | 66.77% | 66.85% | 0.7357 | 68.05% | 65.66% | 67.41% |
### Formal decisions (frozen criteria, confirmation seeds 6-8 only)
- **H1, strictly nested tangent-space LR vs reference LR: FAIL.** Mean accuracy gain +0.64 pt (needs >= +0.5): met; positive in 1 of 3 seeds (needs 2): not met; mean AUC change -0.0069 (needs >= -0.005): not met.
- **H3, soft vote of LR and strictly nested tangent LR vs reference LR: PASS.** Mean accuracy gain +1.13 pt (needs >= +0.5): met; positive in 3 of 3 seeds (needs 2): met; mean AUC gain +0.0191 (needs >= +0.01): met.
- **Effect of the nesting correction itself:** TS-N minus the old non-nested tangent model: +0.30 pt (Stage A), -0.05 pt (Stage C) (most folds identical; relative feature change about 2%). The correction removed a methodological doubt; it did not change the results materially.
### Descriptive pooling over all nine seeds (A+B+C; NOT a decision: stages A and C were seen before or are not independent for H1, and every seed uses the same 676 subjects)
- TS-N minus LR: mean accuracy gain +1.00 pt, positive in 6 of 9 seeds (zero in 1), mean AUC change -0.0004.
- H3 soft vote minus LR: mean accuracy gain +1.02 pt, positive in 8 of 9 seeds, mean AUC gain +0.0204.
### Answers
1. **Does the corrected tangent-space LR improve on the baseline?** On average it is about +1.0 pt more accurate than the reference LR over nine seeds, but the gain is not consistent: it failed its pre-registered confirmation (positive in only 1 of 3 confirmation seeds and AUC slightly lower), and its gain by seed ranges from -0.59 to +2.66 pt. It is not established as an improvement.
2. **Consistent across folds and seeds?** No for the tangent model alone (paired-fold W/T/L close to even, SE about 0.7-1.1 pt in every stage). The two-model soft vote is more consistent (gain positive in 8 of 9 seeds, AUC +0.020) and passed the formal confirmation, but its mean accuracy gain is about +1.0 pt, and its paired-fold SE is 0.6-0.8 pt.
3. **Best-performing model:** the soft vote of the reference LR and the strictly nested tangent LR (H3): the only candidate that passed a formal confirmation in this workflow, with R1 accuracy 67.70% and AUC 0.747 on the confirmation seeds versus 66.57% and 0.728 for the reference LR. This is a small gain, not the 85% target.
4. **Is further experimentation on this cohort justified?** Not for accuracy: after nine seeds, the nesting correction with its validation, and a pre-registered confirmation, the ceiling of the connectivity features on these 676 subjects is about 66-68% accuracy (AUC about 0.72-0.74), and the effects being discussed are about one point. What remains valuable is data, not modelling: an independent external cohort with the same preprocessing, more subjects per site, frame-level motion data and harmonized reprocessing; a final evaluation on the protected holdout would need a frozen model and your separate authorization.

## 3. Integrity
- Hash inventory (baseline files, covariance cache, campaign code, frozen selection rule, final/audit/confirmation reports, all existing campaign checkpoints and results, confirmation outputs): identical before the first corrected run and now: **True**. Per-stage inventories: `hashes_before_*.json`, `hashes_after_*.json`.
- Subject-level outer folds identical for all arms within a seed (asserted); all fits use only the outer-training rows (or inner splits of them); holdout, `RESTRICTED_id_membership.csv` and the combined arrays were not opened.

## 4. Limitations
Result types: Stage A is exploratory, Stage B is the pre-registered confirmation, Stage C is supplementary and not independent for H1; **no independent external validation exists**. All stages use the same 676 subjects (new partitions are not new subjects); folds are not independent; accuracy SE is about 1.8 points; exploratory results decide nothing; no independent external validation was possible or performed. Nothing here supports the 85% target.
