# CC200-only reconstruction and improvement plan (NOT started; requires your authorization)

Nothing below has been run. Outputs would go to `C:\abide\cc200_reconstruction\experiments\<name>\`; no existing file is overwritten. Targets of 85-90% are aspirations: the existing evidence (CV 67-69%, AUC 0.71-0.75 across three pipelines)
gives no basis for them, and nothing will be reported as achieved unless a valid evaluation measures it.

## Stage 0. Fixed rules (apply to every stage)
- **Data:** the 676 phase-2 training-list subjects only (`splits/phase2_train_subjects.txt`: 313 ASD / 363 TC, 19 sites). The restricted holdout list, its labels/features/predictions and `RESTRICTED_id_membership.csv` are never opened.
- **Loading without touching the holdout:** read the per-subject raw files `abide_pcp/ABIDE_pcp/cpac/filt_global/*_rois_cc200.1D` for the training IDs only, instead of `ts_cc200.npy` (which contains holdout rows). Labels/sites/age/sex: `phenotypic_qc.csv` filtered to the training IDs immediately after reading. Apply the same >= 100 timepoints rule.
- **Every fitted step inside the training partition:** imputation, Fisher-Z statistics (none needed), scaling, feature selection, harmonization, hyperparameters, thresholds.
- **Selection bias control:** nested CV; a predeclared acceptance rule per variant (below); no choice from outer-fold scores; no best-fold or best-seed reporting.
- **Metrics reported for each evaluation:** accuracy, balanced accuracy, sensitivity, specificity, precision, F1, ROC-AUC, confusion matrices (pooled and per fold), fold SDs, per-site results where site-aware, from saved per-subject predictions (saved under the experiment folder).

## Stage 1. Data check on permitted subjects (about 2 min, no model)
1. Build the 676-subject set from the raw files; assert unique IDs, labels, sites, timepoints, no NaN/Inf.
2. Report the number of constant or near-constant ROIs (open question 3) and how the Ledoit-Wolf conversion treats them.
3. Alignment check: confirm that features built from the raw files equal `covs_cc200.npy` rows for these 676 subjects (training subjects only) (open question 4).

## Stage 2. Baseline B0 = the historical recipe on permitted data
- Ledoit-Wolf covariance -> correlation -> Fisher-Z (19,900 edges) + age + sex -> StandardScaler -> L2 logistic regression (balanced), C in {0.001, 0.01, 0.1, 1, 10}, inner 5-fold ROC-AUC.
- Outer: stratified 10-fold x seeds 0, 1, 2 (the phase 2e/2h protocol); threshold fixed at 0.5 (primary) with a nested, training-only Youden threshold as a second reading.
- **Reconstruction check:** B0 should reproduce phase 2h's recorded 67.21% (AUC 0.7175) on the same subjects, folds and seeds. A match supports the reconstruction; a mismatch is reported and investigated before anything else runs.
- Site-aware companion: leave-one-site-out of B0 (19 site-specific models), with pooled and per-site metrics.
- Estimated runtime: about 3-5 min (30 outer fits at about 6 s) plus about 2 min for leave-one-site-out. Files: `experiments/b0_baseline/` with `protocol.md`, script, `fold_metrics.csv`, `oof_predictions.csv`, `leave_site_out_predictions.csv`, `final_report.md`, `logs/`.

## Stage 3. Candidate improvements, each against B0 (one predeclared comparison each)
| ID | Change from B0 | Why it is justified | Notes |
|---|---|---|---|
| V1 | Plain Pearson + Fisher-Z (mean-imputed undefined values) instead of Ledoit-Wolf | tests whether the shrinkage estimator matters | needs explicit constant-ROI handling |
| V2 | Linear SVM (calibrated) instead of logistic regression | requested candidate classifier | inner AUC selection of C |
| V3 | In-fold SelectKBest (k in {500, 2000, 5000}) before the classifier | 19,900 features vs 608 training rows | k chosen in the inner loop only |
| V4 | Age/sex handling: append (B0) vs residualize features on age/sex in-fold vs no demographics | demographics were uninformative earlier; tests encoding choices | all fitted on training rows |
| V5 | In-fold label-free site harmonization (ComBat-style, age/sex covariates) | site effects are the main uncontrolled nuisance | fitted on training rows; unseen sites untouched |
| V6 | Tangent-space (Ledoit-Wolf, geometric-mean reference fitted in-fold) | alternative connectivity estimator, previously similar to correlation | estimated slower (about 20 min) |
Acceptance rule (same as earlier work, predeclared): a variant "improves" only if, in at least 2 of 3 seeds, mean accuracy and mean balanced accuracy are both >= 0.02 above B0 with the same seed and accuracy is higher in >= 7 of 10 folds; its leave-one-site-out result must not be worse. No test set is used; no variant is retained on AUC alone.
Estimated runtime: about 5-10 min per variant (V6 longer). Run one variant at a time, each after its own protocol file, in `experiments/<variant>/`.

## Stage 4. Freeze and final evaluation (separate authorization)
Only after Stage 3: freeze exactly one pipeline (features, preprocessing, hyperparameter grid, threshold rule) in a signed-off spec; fit it on the 676 training subjects once; evaluate it once on the locked holdout, **only on your explicit authorization**, and report the result whatever it is. The holdout overlaps earlier CC200 development at the subject level through the historical splits, so even that result is a less independent test than an external cohort.

## Expected outcome (stated in advance)
About 67% accuracy (AUC about 0.72) in random folds for B0 and for any variant, lower under leave-one-site-out. 74% would be a surprise on permitted data, and 85-90% is not supported by any existing evidence; the realistic route to
a higher, defensible number is an external cohort and more subjects per site, not further tuning on this one.

## What I need from you
Authorization to start Stage 1-2 (reads training-list data only, no holdout). Variants V1-V6 would each be confirmed with a protocol before running.
