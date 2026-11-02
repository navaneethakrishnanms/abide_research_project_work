# Methodology, audit and limitations

## Data and labels
- Cohort: 600 development subjects (`splits/phase2_train_subjects.txt` intersected with `results/phase2/fd_filtered_subjects.csv`), 19 sites; 268 ASD / 332 typical control.
  The restricted 170-subject holdout, its ID file, CC200 and the overlapping 780-subject cohort were not opened.
- Label alignment (verified earlier in this project and re-asserted here): manifest labels equal `DX_GROUP` in the phenotype file (y = 1 is ASD; 0 mismatches over 600 subjects);
  every prediction file's `y_true` equals the manifest label (checks below). ROI order: the 116 AAL ROI codes in the header of all 600 `rois_aal.1D` files are identical.

## Features and models (unchanged from the experiments being reported)
- Features: upper-triangle Pearson correlations of the 116 ROI series (6,670 edges); training-fold mean imputation of undefined correlations, Fisher-Z (clip 1e-6), StandardScaler,
  SelectKBest(f_classif), L2 logistic regression (balanced class weights); C and k chosen by inner stratified 3-fold ROC-AUC on training rows only. All steps fitted inside each training split.
- Models M2 (tangent-space), M3 (age + sex added), M4 (in-fold ComBat-style harmonization, label-free) and M6 (Elastic Net + SSAE) are variants evaluated with the same cohort and folds.

## Evaluation protocol
- M1-M4/M6: outer StratifiedKFold(5, shuffle, seed 42) on the 600 subjects, identical fold assignment (checked against the manifest); inner-CV seeds 42, 123, 2026 (affect hyper-parameter selection only).
- M5: leave-one-site-out. Every subject has exactly one out-of-fold prediction per run.
- Metrics: computed in `build_evaluation_report.py` from the confusion counts (accuracy, sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), precision = TP/(TP+FP), F1 = 2TP/(2TP+FP+FN),
  balanced accuracy = mean of sensitivity and specificity) and ROC-AUC from the stored scores; each is cross-checked against scikit-learn (agreement to 1e-12).
- Thresholds: the documented threshold (0.5) stored with each prediction. No threshold was tuned on the reported folds in this report. (A separate numerical analysis in `demo/` showed that cross-fitted thresholds do not raise accuracy.)

## Consistency checks (77/77 passed)
| Result | Check |
|---|---|
| PASS | M1: file read |
| PASS | M1 seed 42: 600 rows, 600 unique subjects |
| PASS | M1 seed 42: labels equal manifest labels (class counts 268/332) |
| PASS | M1 seed 42: sites equal manifest sites |
| PASS | M1 seed 42: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M1 seed 42: stored y_pred consistent with its threshold |
| PASS | M1 seed 123: 600 rows, 600 unique subjects |
| PASS | M1 seed 123: labels equal manifest labels (class counts 268/332) |
| PASS | M1 seed 123: sites equal manifest sites |
| PASS | M1 seed 123: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M1 seed 123: stored y_pred consistent with its threshold |
| PASS | M1 seed 2026: 600 rows, 600 unique subjects |
| PASS | M1 seed 2026: labels equal manifest labels (class counts 268/332) |
| PASS | M1 seed 2026: sites equal manifest sites |
| PASS | M1 seed 2026: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M1 seed 2026: stored y_pred consistent with its threshold |
| PASS | M2: file read |
| PASS | M2 seed 42: 600 rows, 600 unique subjects |
| PASS | M2 seed 42: labels equal manifest labels (class counts 268/332) |
| PASS | M2 seed 42: sites equal manifest sites |
| PASS | M2 seed 42: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M2 seed 42: stored y_pred consistent with threshold 0.5 |
| PASS | M2 seed 123: 600 rows, 600 unique subjects |
| PASS | M2 seed 123: labels equal manifest labels (class counts 268/332) |
| PASS | M2 seed 123: sites equal manifest sites |
| PASS | M2 seed 123: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M2 seed 123: stored y_pred consistent with threshold 0.5 |
| PASS | M2 seed 2026: 600 rows, 600 unique subjects |
| PASS | M2 seed 2026: labels equal manifest labels (class counts 268/332) |
| PASS | M2 seed 2026: sites equal manifest sites |
| PASS | M2 seed 2026: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M2 seed 2026: stored y_pred consistent with threshold 0.5 |
| PASS | M3: file read |
| PASS | M3 seed 42: 600 rows, 600 unique subjects |
| PASS | M3 seed 42: labels equal manifest labels (class counts 268/332) |
| PASS | M3 seed 42: sites equal manifest sites |
| PASS | M3 seed 42: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M3 seed 42: stored y_pred consistent with threshold 0.5 |
| PASS | M3 seed 123: 600 rows, 600 unique subjects |
| PASS | M3 seed 123: labels equal manifest labels (class counts 268/332) |
| PASS | M3 seed 123: sites equal manifest sites |
| PASS | M3 seed 123: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M3 seed 123: stored y_pred consistent with threshold 0.5 |
| PASS | M3 seed 2026: 600 rows, 600 unique subjects |
| PASS | M3 seed 2026: labels equal manifest labels (class counts 268/332) |
| PASS | M3 seed 2026: sites equal manifest sites |
| PASS | M3 seed 2026: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M3 seed 2026: stored y_pred consistent with threshold 0.5 |
| PASS | M4: file read |
| PASS | M4 seed 42: 600 rows, 600 unique subjects |
| PASS | M4 seed 42: labels equal manifest labels (class counts 268/332) |
| PASS | M4 seed 42: sites equal manifest sites |
| PASS | M4 seed 42: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M4 seed 42: stored y_pred consistent with its threshold |
| PASS | M4 seed 123: 600 rows, 600 unique subjects |
| PASS | M4 seed 123: labels equal manifest labels (class counts 268/332) |
| PASS | M4 seed 123: sites equal manifest sites |
| PASS | M4 seed 123: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M4 seed 123: stored y_pred consistent with its threshold |
| PASS | M4 seed 2026: 600 rows, 600 unique subjects |
| PASS | M4 seed 2026: labels equal manifest labels (class counts 268/332) |
| PASS | M4 seed 2026: sites equal manifest sites |
| PASS | M4 seed 2026: outer-fold assignment equals manifest (5 folds partition the 600 subjects) |
| PASS | M4 seed 2026: stored y_pred consistent with its threshold |
| PASS | M5: file read |
| PASS | M5 seed n/a (single run): 600 rows, 600 unique subjects |
| PASS | M5 seed n/a (single run): labels equal manifest labels (class counts 268/332) |
| PASS | M5 seed n/a (single run): sites equal manifest sites |
| PASS | M5: each site held out as one group |
| PASS | M5 seed n/a (single run): stored y_pred consistent with its threshold |
| PASS | M1: three seeds present |
| PASS | M2: three seeds present |
| PASS | M3: three seeds present |
| PASS | M4: three seeds present |
| PASS | confusion rows: every row sums to 600 with 268 ASD / 332 TC |
| PASS | metrics_table has no simulated entries |
| PASS | target 84-85% not reached by any measured row |

## Is 84-85% achievable? (Part A conclusion)
- Measured: no. The best measured accuracy is 66.00% (best seed) / 65.33% (best seed-average), AUC at most 0.708. Even an optimistic oracle threshold on the same predictions
  gives about 66.5% (`demo/threshold_85_report.md`). With AUC near 0.70, an accuracy near 85% is not attainable by any threshold.
- Approaches already tested on this cohort without a reliable gain: threshold selection, wider C grids, linear/RBF SVM, L1/Elastic Net, tangent-space, CPM, dynamic FC, site centering
  (including transductive test-site centering), motion adjustment and matching, demographic covariates (age, sex), cerebrum-only ROIs, ComBat-style harmonization (accuracy +0.1 points, 0 of 3 seeds met the rule), SSAE.
- **No further experiment is proposed or was run.** No untested, scientifically justified within-cohort approach is expected to move accuracy by anything close to 20 points. The routes that could
  change the picture need data outside the permitted set: an independent external cohort with the same preprocessing (also required for any claim of validated performance), more subjects per site,
  frame-level motion data and reprocessing from raw images in one harmonized pipeline. If you want one of these evaluated, state the data source first; a protocol, runtime and output files would be given before running.

## Limitations
- Development cross-validation only; no independent test set; the earlier 170-subject holdout overlaps development subjects and was not used.
- Five folds of 120 subjects: accuracy fold SD is 3-5 points, so differences below about 2 points are noise; no significance tests or confidence intervals.
- ROC-AUC is pooled over models trained on different folds (or sites) and is not the AUC of a single model. M6 reports fold means, so pooled metrics and confusion counts are unavailable.
- Random subject-level folds do not measure generalization to unseen sites (leave-one-site-out accuracy is lower, 62.17%); site, scanner, age and head-motion effects are present in the scores.
- Development subjects overlap the cohort behind earlier design decisions; repeated experiments on the same folds create selection pressure, so small differences between variants must not be over-interpreted.
- Sex differs by diagnosis in this cohort (female 13.8% ASD vs 20.8% control); age does not. Predictions are research output, not a diagnostic tool.
- The illustrative example in `evaluation_report.md` Part C is hypothetical and is not part of any table, CSV or conclusion here.
