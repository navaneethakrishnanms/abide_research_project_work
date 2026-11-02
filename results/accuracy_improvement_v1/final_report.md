# Accuracy improvement v1 -- small, predeclared investigation on the frozen AAL-116 baseline

**Result: no candidate reliably improved on the frozen baseline.** 80% was not approached: the best five-fold development accuracy of any candidate/seed is 65.3% (the unchanged baseline with inner seed 2026). All numbers are development cross-validation estimates on 600 subjects, not independent-holdout performance. The restricted holdout, its ID file and CC200 were not used.

## 1. Evidence that motivated the experiments (Phase 1)
From the existing reports, predictions and per-site results (no new fitting):
- Sensitivity is below specificity for the baseline (0.60 vs 0.67) and the balance varies by fold (sensitivity 0.53-0.67), so the decision threshold is a plausible lever. But accuracy was nearly flat for thresholds 0.45-0.70 in the earlier diagnostics.
- Under leave-site-out the 0.5 threshold transferred badly to some sites (LEUVEN_1 and SBL predicted almost all TC: sensitivity 0.09-0.10 with AUC 0.70-0.80), while the average within-site ranking held up.
- The selected C was mid-grid in all five-fold fits and 15 of 19 leave-site-out fits; three sites chose the lower edge (1e-4), so a finer/wider C grid is a weak but cheap refinement.
- Already tested without gain (so not repeated): Elastic Net classifier 62.0%, linear and RBF SVM about 64%, CPM, site centering, motion adjustment, Elastic Net + SSAE 62.8%.
- Baseline pipeline verified from `fast_accuracy_experiment.py` (MeanImputeFisherZ -> StandardScaler -> SelectKBest -> L2 logistic regression, balanced; grid C {1e-4..1e-1} x k {500, 2000, all}; inner stratified 3-fold ROC-AUC) and the stored fold assignments (asserted identical to the manifest).

## 2. Candidates (Phase 2), all predeclared
| ID | Description | Why |
|---|---|---|
| E0 | Control: the frozen baseline procedure, threshold 0.5 | Reference for each inner seed; must reproduce the saved predictions |
| E1 | E0 + decision threshold chosen on training-only inner out-of-fold probabilities (5-fold, selected C and k; thresholds 0.30-0.70 step 0.025; maximize balanced accuracy; ties -> closest to 0.5) | Sensitivity/specificity imbalance, site-dependent threshold failures |
| E2 | Extended C grid {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} x same k grid, threshold 0.5 | Edge hits at 1e-4 |
| E3 | E2 + the same training-only threshold rule | Combination |

Linear SVM and Elastic Net were not rerun: both already exist in the fast experiment and the SSAE experiment with no gain. Rule fixed in advance: a candidate "improves" if, against E0 with the same inner seed, mean accuracy and balanced accuracy both rise by at least 0.02, accuracy improves in at least 4 of 5 folds, and this holds for at least 2 of the 3 seeds (inner-CV seeds 42, 123, 2026; outer folds fixed).

## 3. Results (Phase 3-4), five outer folds, mean +/- SD over folds
| Candidate | Seed | Accuracy | Bal. acc. | Sens. | Spec. | F1 | AUC | Mean threshold |
|---|---|---|---|---|---|---|---|---|
| E0 control | 42 | 0.642 +/- 0.043 | 0.638 | 0.604 | 0.672 | 0.601 | 0.694 | 0.50 |
| E0 control | 123 | 0.647 +/- 0.047 | 0.643 | 0.608 | 0.678 | 0.606 | 0.694 | 0.50 |
| E0 control | 2026 | 0.653 +/- 0.033 | 0.649 | 0.615 | 0.683 | 0.613 | 0.692 | 0.50 |
| E1 threshold | 42 / 123 / 2026 | 0.628 / 0.652 / 0.633 | 0.626 / 0.652 / 0.634 | 0.605 / 0.660 / 0.649 | 0.648 / 0.644 / 0.620 | 0.590 / 0.627 / 0.614 | 0.694 / 0.694 / 0.692 | 0.49 / 0.49 / 0.45 |
| E2 extended grid | 42 / 123 / 2026 | 0.650 / 0.623 / 0.648 | 0.646 / 0.619 / 0.645 | 0.612 / 0.585 / 0.612 | 0.681 / 0.653 / 0.677 | 0.609 / 0.580 / 0.609 | 0.695 / 0.686 / 0.693 | 0.50 |
| E3 extended + threshold | 42 / 123 / 2026 | 0.632 / 0.625 / 0.640 | 0.637 / 0.626 / 0.639 | 0.690 / 0.634 / 0.630 | 0.584 / 0.617 / 0.647 | 0.625 / 0.599 / 0.611 | 0.695 / 0.686 / 0.693 | 0.45 / 0.48 / 0.48 |

Seed-averaged accuracy: E0 0.647, E1 0.638, E2 0.641, E3 0.632 (balanced accuracy 0.643 / 0.638 / 0.637 / 0.634; AUC 0.694 / 0.694 / 0.692 / 0.692). Per-fold values and per-subject anonymized predictions: `fold_metrics.csv`, `oof_predictions.csv`.

Paired differences against E0 with the same seed (accuracy / balanced accuracy; folds with higher accuracy):
| Candidate | Seed 42 | Seed 123 | Seed 2026 | Rule met |
|---|---|---|---|---|
| E1 | -0.013 / -0.012 (1/5) | +0.005 / +0.010 (2/5) | -0.020 / -0.015 (1/5) | 0 of 3 seeds |
| E2 | +0.008 / +0.008 (1/5) | -0.023 / -0.023 (0/5) | -0.005 / -0.005 (1/5) | 0 of 3 seeds |
| E3 | -0.010 / -0.001 (2/5) | -0.022 / -0.017 (1/5) | -0.013 / -0.011 (2/5) | 0 of 3 seeds |

## 4. Findings
- **E0 reproduces the frozen baseline exactly** (seed 42: identical predictions and scores in all five folds). Changing only the inner-CV seed moves the baseline's accuracy between 0.642 and 0.653, so selection noise alone is about one point.
- **Threshold selection trades sensitivity for specificity without improving balance.** E1 raises mean sensitivity by 3 points (0.609 -> 0.638) and lowers specificity by 4 points; balanced accuracy is unchanged to slightly lower (0.643 -> 0.638). The chosen thresholds were unstable: seed-42 E1 thresholds 0.475, 0.625, 0.425, 0.45, 0.475 (range 0.375-0.625 across candidates, folds and seeds), because the training-only objective is noisy with about 480 subjects.
- **A wider C grid did nothing reliable.** E2 changes accuracy by +0.008, -0.023 and -0.005 depending on the seed; AUC is unchanged within 0.01.
- **AUC is unchanged by threshold choice** (as it must be) and by the grid (0.686-0.695 everywhere), confirming that the ranking quality, not the cut-off or the regularization strength, limits accuracy.
- No candidate is consistent across folds (accuracy improved in at most 2 of 5 folds in any seed) or across seeds.

## 5. Secondary: one-pass leave-one-site-out check (seed 42; not used for selection)
| Candidate | Pooled acc. | Bal. acc. | Sens. | Spec. | F1 | Mean abs gap, predicted vs true ASD share (sites n >= 20) |
|---|---|---|---|---|---|---|
| E0 | 0.622 | 0.618 | 0.586 | 0.651 | 0.580 | 0.143 |
| E1 | 0.628 | 0.628 | 0.619 | 0.636 | 0.598 | 0.187 |
| E2 | 0.630 | 0.627 | 0.601 | 0.654 | 0.592 | 0.143 |
| E3 | 0.640 | 0.641 | 0.646 | 0.636 | 0.616 | 0.166 |

The differences are 0.6-1.8 points, well inside the uncertainty of 600 subjects, and E3's higher number here does not override its lower number in the primary evaluation; it was not used to choose anything. The training-only thresholds did not reduce the site-level mismatch between predicted and true ASD share (0.166-0.187 vs 0.143): random inner folds cannot see site shift, so they cannot calibrate a threshold for an unseen site.

## 6. Runtime and configuration
301 s total (34 checkpoints; outer 45 candidate-fold fits plus 19 leave-site-out fits); single-thread numerical libraries; configuration in `config.json`.

## 7. Limitations
- Five folds of 120 subjects, one cohort; no significance tests; the improvement rule (+0.02) is larger than the seed noise (about 0.01) but small effects cannot be excluded or confirmed.
- The development subjects overlap the Phase 3-10 cohort used in earlier design decisions; the baseline procedure was frozen before this run.
- Only logistic-regression variants were tried here (other model families were already tested elsewhere without gain); inner folds were random, not site-grouped.

## 8. Conclusion and next research direction
Within this small experiment set the frozen baseline stands: threshold re-selection and a finer regularization grid do not improve it, and nothing moves it toward 80%. Every approach tried across the project (classifiers, representations, adjustments, deep models, threshold and regularization changes) gives 61-68% accuracy and AUC about 0.69. What the evidence supports is that further tuning of classifiers, thresholds or regularization on this feature set is unlikely to help. What would change the picture is a hypothesis, not a finding: more subjects, or data with better-controlled acquisition (known TR, per-frame motion, harmonized preprocessing from raw data), which would change the signal-to-noise ratio of the connectivity features. Until such data exist, report the baseline as about 64% five-fold and 62% leave-site-out development accuracy.
