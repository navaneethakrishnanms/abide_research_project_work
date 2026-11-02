# CC200 historical pipeline: audit report (read-only; initial task)

Atlas: **CC200 only** (no AAL-116 anywhere). Nothing was trained, evaluated or re-run, and no existing file was modified. The restricted holdout list, `RESTRICTED_id_membership.csv`, holdout labels, features and predictions were not opened,
and the CC200 arrays were not loaded (header and file-hash checks only). Everything new is in `C:\abide\cc200_reconstruction\`.

## 1. Verdict on the 74.12%
**Recorded, internally consistent, NOT reproducible from saved artifacts, and NOT independently verified.**
- It is the holdout accuracy (126/170) of the `fc_v2_results.csv` row "LogReg-L2 (baseline)". Its confusion matrix (TP 53, FN 26, FP 18, TN 73; derived from the recorded rates and class sizes) reproduces the recorded accuracy, balanced accuracy, sensitivity, specificity and F1 (`metrics_table.csv`, row H1).
- The fitted model, the scores/predictions and the run log do not exist. The number cannot be recalculated from predictions.
- It was measured on the **old seed-42 170-subject holdout, which the project's own Phase 1 audit classified as reused/compromised** (`holdout_guard.py` docstring; `phase2a_create_split.py` lines 155-157). `phase2c_baseline_cv.py` (lines 290-293) already states that this figure "is not the same measurement" as dev-only CV.
- Every cross-validated estimate of the same pipeline is 67-69% accuracy: AUC 0.71-0.75. Table in `metrics_table.csv`.

## 2. Verified pipeline versus the requested conceptual workflow
| Step in your workflow | What the code actually does (verified) | Note |
|---|---|---|
| ABIDE-I rs-fMRI -> preprocessed | PCP CPAC, band-pass 0.01-0.1 Hz, global signal regression, quality-checked list (`fetch_abide.py`) | verified from code; 871 subjects |
| CC200 -> 200 ROI time series | `rois_cc200` from `fetch_abide_pcp`, stored in `ts_cc200.npy` (871 rows), 846 kept with >= 100 timepoints | file hashes match the 2026-09-09 records |
| "Pearson" functional connectivity | per-subject z-scoring, **Ledoit-Wolf shrunk covariance** (assume_centered), converted to a correlation matrix, then Fisher-Z (clip +/-0.999999) | **not plain Pearson**; this matters for any reconstruction |
| 200 x 200 matrix -> lower triangle -> 19,900 | **upper** triangle without diagonal, `triu_indices(200, 1)`, row-major | same 19,900 unordered pairs; only column order differs |
| + age and sex -> 19,902 | `AGE_AT_SCAN` and `SEX` (1/2 numeric) as one column each, appended last; median-imputed, standardized | 19,902 columns confirmed |
| Classifier | StandardScaler + L2 logistic regression (lbfgs, max_iter 5000, balanced), C in {0.001, 0.01, 0.1, 1, 10} by inner 5-fold ROC-AUC | no feature selection, no site harmonization |
| Evaluation | 5-fold CV (quick run) -> Youden threshold on pooled OOF -> refit on 676 -> score 170 holdout once | holdout reused (section 5) |
Full detail with sources: `original_pipeline_config.json`.

## 3. Artifact inventory
| Artifact | Status |
|---|---|
| Source code of the three historical scripts | present: `train_fc.py`, `train_fc_improved.py`, `train_fc_v2.py`, `tangent.py`, `fetch_abide.py` |
| Result tables / logs | present: `fc_v2_results.csv`, `fc_improved_results.csv`, `fc_improved_loso.csv`, `fc_results.csv`, `fc_quick.log`, `fc_full*.log`, phase 2c-2h tables and logs |
| Time series / covariances | present: `ts_cc200.npy` (hash verified), `covs_cc200.npy` (shape (846, 200, 200); no recorded hash), 871 raw `*_rois_cc200.1D` files |
| Fitted estimator, scaler, threshold object | **absent** (never serialized) |
| Per-subject predictions / scores | **absent** (only aggregates saved) |
| Confusion matrix | **not stored**; derived (consistent with all recorded metrics) |
| Log of the run behind the 74.12% row | **absent** |
| Seeds, split, hyperparameter grid | recoverable from code (section 2); final-fit C and library versions are not |
Missing items and open questions: `missing_artifacts_and_open_questions.md`.

## 4. Which configuration produced 74.12%
Experiment 2A of `train_fc_v2.py` (correlation features + age/sex, L2 logistic regression). The row has a single seed, zero fold SD and 0.44 min runtime, so it came from `python train_fc_v2.py --quick`
(5-fold, 1 seed, LogReg-L2 only). This identification is **inferred from the artifacts, not recorded**. Neighbouring runs on the same holdout (`fc_improved_results.csv`): FC-only 72.94%, + age/sex 72.94%, + FIQ 73.53% (confound, flagged by its own script), tangent 72.35%.
Age and sex added nothing on the holdout (identical counts for FC-only and + age/sex), so the figure is a property of connectivity features plus this particular 170-subject sample.

## 5. Leakage and validation risks (ranked)
1. **Holdout reuse (high).** The holdout was scored in `train_fc_improved.py` (four configurations), `train_fc_v2.py`, `fast_holdout_eval.py` and `compute_ensemble_holdout.py`; `train_fc_v2.py` prints its improvement over the previous holdout best. The design choices were made with sight of holdout results. Not a clean one-shot estimate.
2. **Statistical weight (medium).** n = 170: 95% CI of 74.12% is [66.9, 80.5]; if the true accuracy were the CV value 67.0%, P(>= 126/170) = 0.028. The holdout AUC (0.81) is also about 0.10 above the matching CV AUC (0.71).
3. **Threshold selection on the evaluated scores (medium, CV only).** The Youden threshold is chosen on the same out-of-fold scores that give the CV accuracy; the holdout is scored with a threshold that did not see it.
4. **No site control (medium).** Random folds; no harmonization in 2A; the split is stratified by diagnosis, not site. Pooled leave-one-site-out accuracy is 68.5% with per-site values from 42% to 88%.
5. **Feature leakage (none found).** Per-subject covariance and correlation use no cross-subject statistics, the correlation embedding has no fitted reference, and scaler, grid search and threshold are fitted inside training data. Demographic imputation uses training-fold medians.
6. **Demographics (low-medium).** Age/sex are legitimate covariates; FIQ is a diagnosis-related confound (the script itself says so) and is not the 74.12% row. Sex differs by diagnosis in this pool.
7. **Partition conflict (blocking for reconstruction).** The seed-42 split and the phase-2 split (seed 20260, site x diagnosis stratified) are different partitions; about 80% of the **locked phase-2 holdout subjects** are expected to lie inside the old 676 "train/validation" set (a random-overlap expectation, not a measurement). The overlap was not computed (it needs the restricted ID list). A faithful rerun of the old CV would therefore train on protected subjects; reconstruction must use the phase-2 training list, i.e. a different subject set.

## 6. Data-integrity checks performed (no holdout data used)
| Check | Result |
|---|---|
| `ts_cc200.npy` SHA-256 vs the 2026-09-09 record | match |
| `phenotypic_qc.csv` SHA-256 vs the record; rows / duplicate SUB_ID | match; 871 rows, 0 duplicates |
| Array shapes (headers only) | ts: (871,) objects; covs: (846, 200, 200); 871 - 25 short = 846, consistent |
| Phase-2 training list | 676 IDs, all unique, all present in the phenotype file, no duplicate rows |
| Labels / classes on the training list | DX_GROUP in {1, 2}; 313 ASD / 363 TC (46.3% ASD); 19 sites, none single-class, 3 sites < 20 subjects |
| Missing values on the training list | age 0, sex 0, site 0, FIQ 46 |
| Pool statistics (from `fc_quick.log`) | 846 subjects, 391 ASD / 455 TC, 19 sites, 116-316 timepoints |
| Train/evaluation overlap in the historical split | none by construction (`train_test_split` returns disjoint index sets); membership not enumerated |
| Alignment of `covs_cc200.npy` rows to `phenotypic_qc.csv` rows | verified by code path only (same keep-mask); not verified on data, because that requires loading arrays containing holdout subjects |

## 7. Verified evidence versus historical claims
- **Verified (code/files):** the pipeline definition, seeds, split rule, grids, the recorded metrics and their mathematical consistency, data provenance and file integrity, the cross-validated figures (67-69%).
- **Historical claim, not reproducible:** that this pipeline "achieves 74.12% accuracy". What is supported is: one reused 170-subject holdout scored 74.12% once, with an unknown final-fit C and no stored predictions.
- **Not supported:** any 85-90% accuracy for CC200. No recorded CC200 evaluation exceeds 74.12%, and only on the reused holdout.

## 8. Recommendation
Do not treat 74.12% as the model's performance. Reconstruct the pipeline on permitted data and measure it with nested, site-aware evaluation (see `reconstruction_plan.md`), expecting roughly 67% in random folds and lower across sites; 85-90% is a target, with no evidence behind it.
No experiment will be run until you authorize the plan.
