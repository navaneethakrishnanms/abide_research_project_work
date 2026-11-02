# Ledoit-Wolf tangent-space connectivity on AAL-116 vs the frozen baseline (accuracy_research_v2)

**Result: no improvement was established.** Two of the three predeclared criteria failed (the accuracy/balanced-accuracy gain and the seed/fold consistency); the no-degradation criterion passed. The best valid new result is 65.2% seed-averaged accuracy against 64.7% for the baseline (+0.5 points). The 85% target was not approached. All results are development cross-validation estimates on 600 subjects, not independent-holdout performance. The restricted holdout, its ID file, CC200 and the 780-subject cohort were not used. This was the only experiment in the run.

## What was done (as predeclared in `experiment_protocol.md`)
- **Representation:** `tangent.py` (read-only): per-subject z-scored ROI series, Ledoit-Wolf covariance, geometric-mean reference fitted on **training subjects only** (inside each inner-training split and on each outer-training set), tangent vectors of the off-diagonal upper triangle (6,670 features, no NaN imputation). Covariances: 600 x 116 x 116, all symmetric positive definite (minimum eigenvalue 0.04, median condition number 166); no reference-estimation convergence warnings.
- **Model/selection:** identical to the baseline apart from the first step: StandardScaler -> SelectKBest -> L2 logistic regression (balanced), C {1e-4..1e-1} x k {500, 2000, all} by inner stratified 3-fold ROC-AUC (seeds 42, 123, 2026); outer folds fixed (the baseline's five folds), threshold 0.5.
- **Baseline for pairing:** control E0 of `accuracy_improvement_v1` (same pipeline; seed 42 reproduces the frozen baseline predictions exactly).

## Primary results: five-fold development CV (mean +/- SD over folds)
| Seed | Representation | Accuracy | Bal. acc. | Sens. | Spec. | Prec. | F1 | AUC | Confusion (tn/fp/fn/tp) |
|---|---|---|---|---|---|---|---|---|---|
| 42 | Baseline (Fisher-Z Pearson) | 0.642 +/- 0.043 | 0.638 | 0.604 | 0.672 | 0.601 | 0.601 | 0.694 +/- 0.037 | 223/109/106/162 |
| 42 | Tangent (Ledoit-Wolf) | 0.645 +/- 0.030 | 0.638 | 0.578 | 0.699 | 0.611 | 0.591 | 0.704 +/- 0.018 | 232/100/113/155 |
| 123 | Baseline | 0.647 +/- 0.047 | 0.643 | 0.608 | 0.678 | 0.607 | 0.606 | 0.694 +/- 0.037 | 225/107/105/163 |
| 123 | Tangent | 0.657 +/- 0.031 | 0.650 | 0.589 | 0.711 | 0.623 | 0.602 | 0.711 +/- 0.019 | 236/96/110/158 |
| 2026 | Baseline | 0.653 +/- 0.033 | 0.649 | 0.615 | 0.683 | 0.615 | 0.613 | 0.692 +/- 0.039 | 227/105/103/165 |
| 2026 | Tangent | 0.655 +/- 0.036 | 0.650 | 0.604 | 0.696 | 0.615 | 0.606 | 0.718 +/- 0.019 | 231/101/106/162 |

Seed-averaged: accuracy 0.652 vs 0.647 (+0.005), balanced accuracy 0.646 vs 0.643 (+0.003), sensitivity 0.590 vs 0.609 (-0.019), specificity 0.702 vs 0.677 (+0.024), F1 0.600 vs 0.607, AUC 0.711 vs 0.694 (+0.018).

Paired per-seed changes (tangent minus baseline): accuracy +0.003, +0.010, +0.002; balanced accuracy +0.000, +0.007, +0.001; accuracy higher in 2, 3 and 2 of 5 folds. Per-fold accuracy changes: seed 42 [0, +2.5, -1.7, +4.2, -3.3] points; seed 123 [+4.2, 0, +4.2, +5.0, -8.3]; seed 2026 [+6.7, +3.3, -5.0, -0.8, -3.3].

## Predeclared criteria
| Criterion | Result |
|---|---|
| 1. Seed-averaged accuracy and balanced accuracy each >= +0.02 | **Failed** (+0.005, +0.003) |
| 2. >= 2 seeds with gains >= 0.02 and accuracy higher in >= 4/5 folds | **Failed** (0 of 3 seeds) |
| 3. No material degradation (AUC >= -0.01; sensitivity and specificity >= -0.03) | Met (AUC +0.018, sensitivity -0.019, specificity +0.024) |
| 4. No integrity failure | Met |

## Observations (post hoc, not part of the criteria)
- **AUC is consistently a little higher with tangent features**: +0.010, +0.017, +0.026 by seed, and higher in 4 of 5 folds for every seed (fold 4 is lower, by 0.04-0.06; fold 0 is higher by 0.07-0.09). The tangent runs are also less variable across folds (accuracy SD 0.030-0.036 vs 0.033-0.047; AUC SD 0.018-0.019 vs 0.037-0.039). Because the same five folds are reused across seeds, this is roughly five independent comparisons, not fifteen.
- The higher ranking did not translate into higher accuracy at the 0.5 threshold: specificity rose and sensitivity fell. A threshold could be selected within training data (earlier experiments showed this redistributes errors rather than improving balanced accuracy), but no such step was in the protocol and none was run.
- **Why accuracy did not improve (hypothesis):** the tangent representation carries essentially the same diagnostic information as the Pearson one at a slightly better-conditioned scale; the earlier CC200 run (Phase 2e, with age and sex covariates) also showed no gain (67.7% vs 67.2%).

## Secondary: one leave-one-site-out pass (seed 42, inductive, not used for decisions, not pooled with the above)
| | Pooled accuracy | Bal. acc. | Sens. | Spec. | F1 | Pooled AUC (mixes 19 models) |
|---|---|---|---|---|---|---|
| Baseline (saved) | 0.622 | 0.618 | 0.586 | 0.651 | 0.580 | 0.684 |
| Tangent | 0.660 | 0.653 | 0.590 | 0.717 | 0.608 | 0.680 |

Tangent accuracy is higher at 10 sites, equal at 2 and lower at 7 (higher at 8 of the 13 sites with n >= 20); large swings occur at small sites (SDSU +0.30, UM_1 +0.19, STANFORD +0.21, KKI -0.13, YALE -0.11, PITT -0.12), and pooled AUC is unchanged. This is a different protocol from the primary evaluation (unseen sites, one seed, 600 subjects, pooled over 19 models), so the +3.8 points is not combined with the five-fold result and is not evidence of an established improvement: the rise comes from specificity (0.651 -> 0.717) at an unchanged pooled AUC, which is the signature of a shift in the score distribution, and five-fold CV shows only +0.5 points. It is a hypothesis for a future, separately predeclared test.

## Integrity
Subjects, labels, sites and folds asserted identical to the manifest; each subject has exactly one out-of-fold prediction per seed (600 unique IDs for each of the three seeds and for the leave-site-out pass); the tangent reference was verified to depend only on training covariances (replacing the test covariances did not change it); 0 of 311 hashed files and `tangent.py` changed; runtime 369 s, peak memory about 1.2 GB (peak resident set 1,190 MB); anonymized IDs `DEV0000`-`DEV0599`.

## Limitations and generalization risks
Five folds of 120 subjects; no significance tests; one atlas and one classifier family; the AUC observation is post hoc; the development subjects overlap the cohort behind earlier design decisions; random-fold results do not show generalization to unseen sites, and the leave-site-out pass is a single, noisy run. No independent-holdout claim.

## Decision
Baseline: 64.2% (seed 42) / 64.7% (seed-averaged) five-fold development accuracy, AUC 0.694 (AAL-116 Fisher-Z Pearson, L2 logistic regression, same five folds). Best valid new result: tangent-space 65.2% seed-averaged, +0.5 points, AUC 0.711; not consistent across folds (accuracy higher in 2-3 of 5), improvement not established, 85% not achieved. Next justified action: **no further modeling on this cohort.** Document the baseline and note the small, consistent AUC advantage of the tangent representation as a hypothesis to test on new, independent data (more subjects, known TR, per-frame motion, an external cohort). This run launches nothing further.
