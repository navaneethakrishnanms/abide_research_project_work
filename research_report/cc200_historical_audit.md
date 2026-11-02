# Audit of the historical CC200 ASD-vs-control workflow (reported 74.12% accuracy)

Scope: CC200 only; read-only. Nothing was trained, evaluated or re-run. The restricted holdout list (`splits/phase2_holdout_subjects.txt`), `RESTRICTED_id_membership.csv`, holdout labels/features/predictions
and the CC200 data arrays (`ts_cc200.npy`, `covs_cc200.npy`: array headers only) were not opened. Integrity checks used the 676-subject training list only.

## 1. Evidence for the 74.12% result
| Item | Finding | Source |
|---|---|---|
| Where the number is | `fc_v2_results.csv`, one row "LogReg-L2 (baseline)", exp 2A: holdout (n = 170) accuracy **0.7412**, balanced acc 0.7365, AUC 0.8104, F1 0.7067, sensitivity 0.6709, specificity 0.8022 | `C:\abide\fc_v2_results.csv` (22 Aug 2026 08:57) |
| Same row, cross-validation | CV AUC 0.7101, CV accuracy 0.6701, balanced acc 0.6710, Youden threshold 0.4720; OOF F1 0.656, sens 0.683, spec 0.659 | same row |
| Which run produced it | Almost certainly `python train_fc_v2.py --quick`: the row has **cv_auc_std = cv_acc_std = 0.0** (one seed), **elapsed 0.44 min**, a single model (2A only), and was written 45 s after `train_fc_v2.py` was last saved (08:56:25). `--quick` = 5-fold, 1 seed, LogReg-L2 only, then one holdout evaluation. No log of that run exists. This is inference from the artifacts, not a recorded fact. | `train_fc_v2.py` lines 426-432, 486-488, 518-522 |
| Derived confusion matrix | 126/170 correct; the reported rates imply 79 ASD / 91 control in that holdout: TP 53, FN 26, TN 73, FP 18. Derived arithmetically from the rates, not read from stored predictions. | computed |
| Statistical weight | 95% Clopper-Pearson CI for 126/170: **[66.9%, 80.5%]**. If the true accuracy were the CV value 67.0%, P(>= 126/170) = 0.028 (one-sided); versus 68.6% (all-846 CV), 0.069. | computed |
| Neighbouring runs on the same holdout | `fc_improved_results.csv`: FC-only (correlation) 72.94% (AUC 0.810), + age/sex 72.94% (0.810), + FIQ 73.53% (0.811, flagged by its own script as a confound), tangent FC-only 72.35% (AUC 0.776) | `fc_improved_results.csv` |
| Cross-validated figures for the same family | All 846 subjects, 10-fold x 3 seeds: correlation + L2 logistic 68.6% (AUC 0.749); pooled leave-one-site-out (N = 820): 68.54% (95% CI 65.2-71.7), AUC 0.746. Development-only re-run on the 676 phase-2 training subjects (phases 2e/2h): 67.21%, balanced acc 66.85%, AUC 0.7175. | `fc_results.csv`, `fc_full3.log`, `phase2e_run.log`, `phase2h_run.log` |

**Reading:** every cross-validated estimate of this pipeline is 67-69% accuracy (AUC 0.71-0.75). The 74.12% appears only on one 170-subject holdout, which is about 7 points above the CV value and whose AUC (0.81) is
about 0.10 above the matching CV AUC (0.71), roughly 2-2.5 standard errors at n = 170. That is compatible with sampling luck plus the reuse described in section 4; it is not evidence of a better model.

## 2. Exact configuration (recovered from code; the only parts that cannot be recovered are listed in section 3)
- **Data:** `abide_pcp/ts_cc200.npy` (871 rows, object array of per-subject time series) with `abide_pcp/phenotypic_qc.csv` (871 rows, no duplicate SUB_ID). Subjects with < 100 timepoints removed -> **846 subjects, 391 ASD / 455 TC, 19 sites** (`fc_quick.log`). Label: `DX_GROUP == 1` -> ASD (positive). `covs_cc200.npy` is shape (846, 200, 200).
- **Features:** per subject, z-score each ROI, Ledoit-Wolf covariance (`assume_centered=True`), convert to correlation (cov / (sd sd' + 1e-12)), Fisher-Z `arctanh` with clip +/-0.999999, upper triangle without diagonal (200 x 199 / 2 = **19,900** features, x sqrt(2), irrelevant after scaling) (`tangent.py`). Age and SEX (numeric 1/2) appended (median-imputed, standardized) = 19,902 columns.
- **Model:** StandardScaler (training rows) -> LogisticRegression(L2, lbfgs, max_iter 5000, class_weight balanced); C chosen by GridSearchCV over {0.001, 0.01, 0.1, 1, 10}, ROC-AUC, inner StratifiedKFold(5, shuffle, random_state = outer seed).
- **Split:** `train_test_split(range(846), test_size = 0.20, stratify = y, random_state = 42)` -> 676 train/validation + 170 holdout (79 ASD / 91 TC by the derived counts).
- **CV (quick run):** StratifiedKFold(5, shuffle, random_state = 0); out-of-fold scores; Youden threshold from those same out-of-fold scores.
- **Holdout step:** model refit on all 676 (inner 5-fold, random_state 42), threshold = the Youden value above (0.4720), evaluated once on the 170.
- **Not recorded:** library versions, the exact C chosen for the final fit, the actual fold-level scores.

## 3. Reproducibility status and missing artifacts
| Artifact | Status |
|---|---|
| Fitted estimator / scaler / threshold | **Missing.** Nothing is serialized by `train_fc*.py`. |
| Per-subject predictions or scores (CV or holdout) | **Missing.** Only aggregate metrics were saved. |
| Run log of the 74.12% run | **Missing** (no `fc_v2` log). |
| Feature arrays (`ts_cc200.npy`, `covs_cc200.npy`) | Present (headers consistent: 871 -> 846 subjects). Row-to-phenotype alignment is by shared position and was **not** verified here, because that needs loading arrays that contain holdout subjects. |
| Code and seeds | Present and deterministic (seeds fixed, lbfgs). |
| **Verdict** | The 74.12% is **recorded but not reproduced and not independently verified**. Re-running `train_fc_v2.py --quick` would probably reproduce it numerically (same code, seeds, data), but it is not run here and would simply re-score the already-reused holdout. |

## 4. Leakage and validation risks (ranked)
1. **Holdout reuse (high).** The "locked" holdout was scored for four configurations in `train_fc_improved.py` and again in `train_fc_v2.py`, which prints "improvement vs previous best 72.94%" against it. The design (correlation vs tangent, adding age/sex, classifier) was therefore chosen with sight of holdout results. 74.12% is not a clean one-shot estimate.
2. **Holdout identity (high, for any rerun).** The seed-42 split of 846 subjects is a different partition from the phase-2 split (seed 20260). About 80% of its 170 subjects are expected to lie in the phase-2 training list, and conversely its 676 training subjects contain most of the *locked* phase-2 holdout. A faithful rerun on the old partition would train on subjects whose holdout status the project protects; I did not compute the overlap because that needs the restricted ID list. A reconstruction must therefore use the phase-2 training list, which is a **different subject set** from the historical one.
3. **Threshold chosen on the evaluated predictions (medium, CV only).** The Youden threshold is picked on the same out-of-fold scores that produce the reported CV accuracy (67.0%), a mild optimism. The holdout is scored with a threshold that did not see it (correct).
4. **No site control (medium).** Folds are random, not site-grouped, and 2A has no harmonization. Pooled leave-one-site-out accuracy is 68.5% with large per-site spread (42-88%, small sites unstable).
5. **Small holdout (medium).** n = 170: +/- 6.5 points on accuracy.
6. **Age/sex/FIQ as inputs (low to medium).** Age/sex are legitimate covariates; the FIQ variant is a diagnosis-related confound (the script itself says so) and is not the 74.12% row. Sex differs by diagnosis in this cohort.
7. **Feature construction (low).** Per-subject covariance and correlation use no cross-subject statistics, and the correlation embedding has no fitted reference, so no feature leakage is apparent. Scaler, grid search and threshold are fitted inside training data.
8. **Label alignment / duplicates (checked, clean on the training list):** 676 unique IDs, all found in the phenotype file, no duplicate rows; labels 1/2 only; **313 ASD / 363 TC**; 19 sites (3 sites < 20 subjects, no single-class site); age, sex and site complete; FIQ missing for 46. The holdout subjects were not examined.

## 5. CC200-only reconstruction plan (reconstruction, not reproduction)
1. **Cohort:** the 676 phase-2 training-list subjects only; read the arrays, keep those rows, and discard all other rows without inspection (as the phase-2 scripts did).
2. **Features:** exactly as section 2 (Ledoit-Wolf -> correlation -> Fisher-Z, 19,900 edges), computed from the time series of the selected subjects only.
3. **Candidates:** L2 logistic regression and linear SVM (as proposed), optionally with age/sex as a separate arm; scaling, optional SelectKBest and the C grid fitted inside each training fold.
4. **Evaluation:** stratified 10-fold x 3 seeds with inner 5-fold C selection (nested), plus leave-one-site-out as the site-aware check; thresholds either fixed at 0.5 or chosen only from training-fold data. Report accuracy, balanced accuracy, sensitivity, specificity, F1, ROC-AUC, confusion matrices, fold SDs and per-site results; never report a best fold as overall.
5. **Already done in part:** phases 2e/2h ran this family on the same 676 subjects (67.21%, AUC 0.7175; 30 fits at about 6 s each). What is missing is the leave-site-out arm and fold-internal thresholding.

## 6. Recommended next experiment (not run)
**CC200 development-only nested CV with a leave-one-site-out arm and a training-only threshold** (L2 logistic regression and linear SVM; age/sex as a second arm), on the 676 phase-2 training subjects, restoring the missing site-aware and per-fold evidence.
- **Why:** it answers the one open question the artifacts cannot: whether the pipeline generalizes beyond random folds, on a subject set that respects the holdout lock. It also puts the historical number in context: CV and LOSO point to 67-69%, and only a re-scored reused holdout shows 74%.
- **Estimated runtime:** about 5-10 min (30 outer fits at about 6 s plus 19 site-held-out fits and the SVM arm), single-threaded numerics.
- **Output files (new folder, nothing overwritten):** `results/cc200_dev_reconstruction_v1/` with `protocol.md`, `run_cc200_dev_reconstruction.py`, `fold_metrics.csv`, `oof_predictions.csv`, `leave_site_out_metrics.csv`, `confusion_matrices.csv`, `final_report.md`, `logs/`.
- **Expectation, stated in advance:** about 67% accuracy (AUC about 0.72) in random folds and lower under leave-one-site-out; I do not expect 74% or any value near 85%. A final evaluation on the locked holdout would need your separate authorization and a frozen model first.

## Conclusion
The historical 74.12% is **REPORTED, NOT VERIFIED**: the result and the full recipe exist, but the fitted model, per-subject predictions and run log do not, the figure comes from a reused 170-subject holdout (CI 66.9-80.5%), and all cross-validated estimates of the same pipeline are 67-69%.
