# Cerebrum-only AAL connectivity (ROIs 1-90) vs the full 116-ROI baseline

**Result: the predeclared improvement rule was NOT met (0 of 3 seeds).** Excluding the cerebellum and vermis did not improve five-fold development accuracy; the 90-ROI representation was slightly lower on average and less stable across inner-CV seeds. These are development cross-validation estimates on 600 subjects, not independent-holdout results. The restricted holdout, its ID file and CC200 were not used.

## Setup (verified)
- **Atlas mapping:** from the verified Phase 4.1 mapping (all 116 rows `mapping_verified` and `header_code_match`): ROIs 1-90 contain no cerebellar/vermis name (last = Temporal_Inf_R, AAL code 8302); ROIs 91-116 are exactly the 26 cerebellar/vermis ROIs (first = Cerebelum_Crus1_L, code 9001).
- **Features:** 4,005 unique edges (90 x 89 / 2) recomputed from the per-subject ROI time series (Pearson, then Fisher-Z in the pipeline). They equal the corresponding columns of the cached 6,670-edge matrix exactly (same order and NaN pattern). Undefined entries: 1,575 for 90 ROIs vs 12,764 for 116.
- **Procedure:** both representations use the frozen baseline pipeline (training-fold mean imputation -> Fisher-Z -> StandardScaler -> SelectKBest -> L2 logistic regression, balanced) and grid (C {1e-4..1e-1} x k {500, 2000, all}, inner stratified 3-fold ROC-AUC), on the same 600 subjects and five outer folds, inner seeds 42, 123, 2026. The 116-ROI run for seed 42 reproduced the saved baseline predictions exactly in all 5 folds. Nothing was chosen from outer-fold scores.
- **Rule (fixed in advance):** mean accuracy and balanced accuracy each up by >= 0.02, accuracy better in >= 4 of 5 folds, in >= 2 of 3 seeds.

## Five-fold results (mean +/- SD over folds)
| Seed | Representation | Accuracy | Bal. acc. | Sens. | Spec. | Prec. | F1 | AUC |
|---|---|---|---|---|---|---|---|---|
| 42 | 116 ROIs (baseline) | 0.642 +/- 0.043 | 0.638 | 0.604 | 0.672 | 0.601 | 0.601 | 0.694 +/- 0.037 |
| 42 | 90 ROIs | 0.640 +/- 0.038 | 0.635 | 0.593 | 0.678 | 0.604 | 0.596 | 0.703 +/- 0.040 |
| 123 | 116 ROIs | 0.647 +/- 0.047 | 0.643 | 0.608 | 0.678 | 0.607 | 0.606 | 0.694 +/- 0.037 |
| 123 | 90 ROIs | 0.612 +/- 0.042 | 0.609 | 0.585 | 0.632 | 0.568 | 0.571 | 0.677 +/- 0.038 |
| 2026 | 116 ROIs | 0.653 +/- 0.033 | 0.649 | 0.615 | 0.683 | 0.615 | 0.613 | 0.692 +/- 0.039 |
| 2026 | 90 ROIs | 0.632 +/- 0.047 | 0.630 | 0.615 | 0.644 | 0.591 | 0.599 | 0.688 +/- 0.034 |

Averaged over seeds: accuracy 0.647 (116) vs 0.628 (90), AUC 0.694 vs 0.689.

Paired differences (90 minus 116, same seed), per-fold accuracy change in parentheses:
| Seed | Accuracy | Balanced acc. | AUC | Sens. / Spec. / F1 | Folds with higher accuracy | Rule met |
|---|---|---|---|---|---|---|
| 42 | -0.002 +/- 0.025 (0, -2.5, +3.3, +0.8, -2.5 pts) | -0.003 | +0.008 | -0.011 / +0.006 / -0.006 | 2/5 | No |
| 123 | -0.035 +/- 0.044 (-5.8, -4.2, +0.8, +0.8, -9.2 pts) | -0.034 | -0.018 | -0.023 / -0.045 / -0.035 | 2/5 | No |
| 2026 | -0.022 +/- 0.030 (-5.8, -2.5, +0.8, +0.8, -4.2 pts) | -0.020 | -0.004 | 0.000 / -0.039 / -0.014 | 2/5 | No |

Fold-level pattern: the 90-ROI version gained accuracy in the same two folds in every seed (folds 2 and 3, +0.8 to +3.3 points) and lost (or tied, seed 42 fold 0) in folds 0, 1 and 4. Its seed-to-seed spread is larger (accuracy 0.612-0.640, versus 0.642-0.653 for the baseline), and it selected a larger C (0.01) more often, i.e. its inner-CV model selection is less stable. AUC in seed 42 is slightly higher (+0.008) but lower or unchanged in the other seeds, so there is no consistent ranking gain either.

## Leave-one-site-out sensitivity (single pass, seed 42; not used to choose anything)
| Representation | Pooled accuracy | Bal. acc. | Sens. | Spec. | F1 | Pooled AUC (mixes 19 models) |
|---|---|---|---|---|---|---|
| 116 ROIs (saved leave-site-out run) | 0.622 | 0.618 | 0.586 | 0.651 | 0.580 | 0.684 |
| 90 ROIs | 0.642 | 0.637 | 0.597 | 0.678 | 0.598 | 0.696 |

Per site (`leave_site_out_per_site.csv`): the 90-ROI accuracy is higher at 7 sites, equal at 5 and lower at 7, with gains up to +0.11 (SBL) and losses up to -0.13 (OLIN), mostly at small sites (CMU, with two subjects, shows AUC 1.0 vs 0.0, which is meaningless).

## Interpretation
- **No improvement from removing the cerebellum under the pre-declared protocol.** The rule failed in all three seeds, and the average change is slightly negative (-0.019 accuracy). The primary five-fold evidence and the leave-site-out pass point in opposite directions (-0.2 points at seed 42 five-fold vs +2.0 points leave-site-out), which is what noise of about +/-2 points between nearly equivalent models looks like; neither is evidence of a real effect, and the leave-site-out result was not used to override the primary rule.
- **The data-quality hypothesis (site-linked cerebellar noise) is not supported as a lever for accuracy**: dropping 40% of the features did not help, and the representation became less stable. It also does not show that cerebellar features carry diagnostic information; the test cannot tell the two apart.
- Nothing here approaches 80%, and the difference between the two representations is smaller than the variation across inner-CV seeds and folds.

## Limitations
- Five folds of 120 subjects; no significance test; the ROI subset was chosen after inspecting this cohort's metadata (not any model score).
- The leave-site-out comparison is one pass at one seed and uses the saved 116-ROI run; site sizes are small.
- The development subjects overlap the Phase 3-10 cohort used in earlier design decisions.

## Recommendation
Keep the 116-ROI baseline. This was the last justified single-variable test suggested by the audit; the data-quality audit and every representation, classifier, threshold and regularization change tried so far lead to the same 61-68% range. Further gains would more plausibly require different data (more subjects, known TR, per-frame motion, harmonized reprocessing) than further changes to this cohort's features.
