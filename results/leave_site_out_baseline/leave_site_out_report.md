# Leave-one-site-out evaluation of the frozen AAL-116 baseline (600 development subjects)

Status: complete -- 19 of 19 sites, no failures (55 s). These are development estimates on unseen sites, not independent-holdout performance. The restricted holdout, its ID file and CC200 were not used.

## Protocol
- Cohort: 600 development subjects (268 ASD / 332 TC) in 19 sites, identical to the fast experiment's manifest (subject order, labels and sites asserted).
- Frozen baseline = the fast experiment's pipeline and procedure, unchanged: training-fold mean imputation -> Fisher-Z -> StandardScaler -> SelectKBest -> L2 logistic regression (balanced). Within every training set (all other sites), C in {1e-4,...,1e-1} and k in {500, 2000, all} are chosen by inner stratified 3-fold ROC-AUC, exactly as for the five-fold baseline. Nothing is tuned on the held-out site. Note: the inner folds are random (not site-grouped), so hyper-parameter choice does not mimic site shift.
- Each site is held out once; each subject has exactly one out-of-site prediction (asserted). A check confirmed that held-out-site rows do not influence one another or the fit.
- Predictions are saved with anonymized IDs (`DEV0000`-`DEV0599` = row order of the development manifest): `oof_predictions.csv` (project_id, site, y_true, score, y_pred).

## Aggregate results (clearly defined)
| Aggregate | Accuracy | Bal. acc. | Sens. | Spec. | F1 | AUC |
|---|---|---|---|---|---|---|
| **Pooled subject-level (all 600 out-of-site predictions)** | **0.622** (Wilson 95% 0.582-0.660) | 0.618 | 0.586 | 0.651 | 0.580 | 0.684* |
| Macro mean over 19 sites (sites weighted equally) | 0.612 | 0.620 | 0.553 | 0.688 | 0.514 | 0.721 |
| Macro mean over the 13 sites with n >= 20 | 0.618 | 0.623 | 0.566 | 0.680 | 0.540 | 0.699 |
| Five-fold random CV (same subjects/pipeline, saved) | 0.642 | 0.638 | 0.604 | 0.672 | 0.601 | 0.692* |

*Pooled AUC combines scores from 19 different models (different scales) and is for reference only. Macro means are not subject-level accuracy: small sites weigh as much as large ones. Site-level accuracy varies widely (SD 0.116 across sites), mostly because most sites are small.

## Per-site results (full table with intervals: `per_site_results.csv`; small sites n < 20 flagged)
| Site | n (ASD/TC) | Acc. | Sens. | Spec. | AUC |
|---|---|---|---|---|---|
| NYU | 133 (57/76) | 0.632 | 0.614 | 0.645 | 0.713 |
| UM_1 | 62 (25/37) | 0.565 | 0.680 | 0.486 | 0.659 |
| USM | 48 (29/19) | 0.750 | 0.759 | 0.737 | 0.797 |
| UCLA_1 | 44 (23/21) | 0.614 | 0.522 | 0.714 | 0.700 |
| PITT | 33 (16/17) | 0.697 | 0.438 | 0.941 | 0.783 |
| TRINITY | 33 (15/18) | 0.606 | 0.667 | 0.556 | 0.707 |
| MAX_MUN | 31 (12/19) | **0.452** | 0.333 | 0.526 | 0.553 |
| YALE | 27 (14/13) | 0.704 | 0.643 | 0.769 | 0.835 |
| UM_2 | 24 (9/15) | 0.833 | 0.778 | 0.867 | 0.852 |
| KKI | 23 (6/17) | 0.522 | 0.833 | 0.412 | 0.588 |
| LEUVEN_1 | 22 (11/11) | 0.545 | **0.091** | 1.000 | 0.702 |
| LEUVEN_2 | 21 (8/13) | 0.619 | 0.500 | 0.692 | 0.596 |
| SDSU | 20 (6/14) | 0.500 | 0.500 | 0.500 | 0.595 |
| STANFORD* | 19 (9/10) | 0.526 | 0.667 | 0.400 | 0.656 |
| SBL* | 18 (10/8) | 0.500 | **0.100** | 1.000 | 0.800 |
| OLIN* | 16 (8/8) | 0.813 | 0.875 | 0.750 | 0.797 |
| UCLA_2* | 12 (5/7) | 0.750 | 1.000 | 0.571 | 0.743 |
| CALTECH* | 12 (4/8) | 0.500 | 0.500 | 0.500 | 0.625 |
| CMU* | 2 (1/1) | 0.500 | 0.000 | 1.000 | 1.000 (2 subjects, meaningless) |

\* n < 20. Every site's test set contains both classes, so AUC is defined for all 19 sites, but for CMU (one subject per class) it carries no information. No site failed and none was omitted. Individual-site intervals are wide (e.g. +/-0.10 to 0.20), so differences between most sites are not meaningful.

## Comparison with the five-fold random CV (`comparison_with_five_fold_cv.csv`)
- **Pooled accuracy: 62.2% vs 64.2% (-2.0 points); balanced accuracy 61.8% vs 63.8%; sensitivity 58.6% vs 60.4%; specificity 65.1% vs 67.2%; pooled AUC 0.684 vs 0.692 (-0.009).** The same 600 subjects were correct in both settings except 49 correct only in the five-fold run and 37 only in the leave-site-out run (descriptive exact McNemar p = 0.24; subjects within a site are not independent, so this is not a rigorous test).
- At site level, leave-site-out accuracy is lower than five-fold accuracy at 9 of 19 sites (6 of 13 with n >= 20): no consistent drop.
- Within-site ranking holds up: mean within-site AUC over the 13 sites with n >= 20 is 0.699 under leave-site-out; the five-fold figure (0.665) is not directly comparable because its scores come from five different fold models.
- **The decision threshold transfers poorly to some unseen sites**: predicted-ASD share deviates from the true ASD share by 0.143 on average per site under leave-site-out vs 0.106 under five-fold; extreme cases are LEUVEN_1 (4.5% predicted ASD vs 50% true; sensitivity 0.09, AUC 0.70), SBL (sensitivity 0.10, AUC 0.80), PITT (24% vs 49%) and KKI (65% vs 26%). In these sites the ranking is better than the accuracy suggests, which fits a site-specific shift in the score distribution (a hypothesis, not tested).

## What the evidence supports
- Performance on unseen sites is only slightly lower than in random folds (pooled accuracy -2.0 points, AUC -0.009), a difference inside the uncertainty of a 600-subject evaluation (pooled accuracy interval +/-3.9 points). It does not show a large site-generalization collapse, and it does not show that random-fold accuracy overstates site generalization by much.
- Accuracy is nevertheless modest (62%) and highly variable across sites; MAX_MUN is below chance (0.452, AUC 0.553) and several sites have strongly unbalanced predictions.
- Nothing here supports a claim of clinically useful accuracy, an independent-holdout result, or 80%.

## Limitations
- 19 sites, 13 with n >= 20; most per-site estimates are very noisy; leave-site-out folds have unequal sizes; no significance tests.
- Hyper-parameters were chosen with random (not site-grouped) inner folds, as in the frozen baseline.
- Sites differ in scan length, spectral content, age, sex mix and motion (representation audit), so site and acquisition effects are confounded with one another here.
- The development cohort overlaps the Phase 3-10 cohort used in earlier design decisions; the baseline itself was frozen before this run.
- Pooled AUC mixes models; the five-fold within-site AUC is not comparable.
