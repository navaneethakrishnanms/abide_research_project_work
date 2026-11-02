# CC200 accuracy campaign (target 85%): plan, written before any campaign experiment ran

Atlas: CC200 only. Data: the 676 authorized Phase-2 training subjects (313 ASD / 363 TC, 19 sites). The restricted holdout, its IDs/labels/features/predictions, `RESTRICTED_id_membership.csv` and the combined arrays are not opened. Baseline, X1, X2, their reports,
the cached features and hashes are read-only; everything new is under `experiments/accuracy_campaign_85/`. X1 (site harmonization) and X2 (age/sex residualization) failed their frozen rule and are not part of any candidate.
**85% is a target, not an expectation:** every CC200 evidence so far (three pipelines, 67-69% in random folds, 66% under the clean threshold rule, AUC 0.71-0.75) points to roughly 66%. Nothing is claimed unless a valid evaluation shows it.

## 0. Common design
- Outer folds: `StratifiedKFold(10, shuffle, random_state = seed)`, seeds 0, 1, 2 (identical for every candidate and for the reference). Inner: `StratifiedKFold(5, shuffle, random_state = seed)`, ROC-AUC. Subject-level splits; per fold the engine asserts that train and test indices are disjoint; Stage 1 verified unique IDs and no identical time series (no duplicates or repeated scans).
- Inputs: age and sex appended to every representation (as in the baseline); `DemoFCPrep` (demographic imputation/standardization + StandardScaler) is the first step of every pipeline, so it is refitted inside every inner fold.
- **Decision rules (all models, including the reference):** R0 = the model's default threshold (probability 0.5; margin 0); **R1 = threshold chosen on inner 5-fold out-of-fold scores of the outer-training partition only** (maximizing balanced accuracy, ties toward the default, midpoint between adjacent scores; inner OOF seed = seed + 1000). No threshold is ever chosen on the evaluated rows. R1 is the primary reading; R0 is reported.
- Stored per outer fold: test scores, the inner OOF scores of all training rows (enables leakage-free ensembles), selected hyperparameters, threshold, runtime. Per-subject predictions are saved for every experiment.
- Metrics: accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, precision, F1, pooled confusion matrices per seed, per-fold values, mean and SD over seeds and folds, paired fold differences, site-wise results, runtime.

## 1. Frozen selection rule (file `selection_rule.py`, sha256 `e4389cbebcae96950fab28bb8f81ae5e81c7967a3d29ed82552e28730c266467`)
Primary metric: pooled out-of-fold **accuracy under R1**, mean over seeds 0-2, compared with the campaign's own reference run P1_LR on the same folds. A candidate **passes screening** only if all hold:
(a) mean paired gain >= +1.5 points in accuracy and in balanced accuracy; (b) gain >= +0.5 point in every seed; (c) accuracy higher in >= 20 of the 30 paired outer folds (one-sided sign test p = 0.049); (d) mean AUC not lower than the reference by more than 0.005; (e) no leakage or protocol violation.
A candidate that passes is **replicated** on fresh seeds 3, 4, 5 (new outer partitions; reference rerun too): mean gain >= +1.0 point and positive in >= 2 of 3 seeds. Only then is it a **validated development improvement**. Any gain >= +5 points triggers a mandatory leakage / data-quality investigation before anything else.
Ranking: all candidates are ranked by mean R1 accuracy (with SD over seeds and AUC) and reported whether or not they pass. If none passes, the reference logistic regression remains the best defensible model. About 15 candidates are examined, so the +1.5 floor (about three paired-fold standard errors, SE 0.5 in X1/X2), the sign test and the replication guard against best-of-K selection; the count is reported.
This rule is separate from the X1/X2 rule, which is unchanged. The rule is not edited after results are seen.

## 2. Experiments (run sequentially; a failed variant is logged and the campaign continues; a data-integrity, leakage or reproducibility failure stops it)
Runtime estimates from a time-only benchmark of one outer fold (scores not inspected); about 107 min for experiments 1-11.
| # | ID | Phase | Representation | Model and predeclared search space | Est. runtime |
|---|---|---|---|---|---|
| 1 | P1_LR | 1 | Ledoit-Wolf correlation + Fisher-Z (19,900) + age/sex | balanced L2 logistic regression, C in {0.001, 0.01, 0.1, 1, 10}. **Reference.** Gate: out-of-fold scores, folds and inner AUCs must equal the saved baseline Arm B exactly, else stop | 6 min |
| 2 | P2_LSVM | 2 | same | LinearSVC (balanced), C in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}; margin scores, R1 threshold on inner OOF decision values | 5 min |
| 3 | P2_RBF | 2 | same | RBF SVC (balanced), C in {1, 10} x gamma in {(1/19902)/3, 1/19902} | 20 min |
| 4 | P2_RF | 2 | same, **in-fold SelectKBest(f_classif, k = 2000)** before the model (runtime control; tree models on 19,902 correlated features and about 540 rows are otherwise intractable) | RandomForest, 300 trees, class_weight balanced_subsample, min_samples_leaf in {1, 5} | 10 min |
| 5 | P2_ET | 2 | same + k = 2000 | ExtraTrees, 300 trees, balanced_subsample, min_samples_leaf in {1, 5} | 3 min |
| 6 | P2_XGB | 2 | same + k = 2000 | XGBoost (gradient boosting), 200 trees, learning rate 0.05, subsample 0.8, colsample 0.5, max_depth in {2, 3}; no class weighting (R1 threshold handles imbalance) | 20 min |
| 7 | P3_PC_LR | 3 | partial correlation from the per-subject Ledoit-Wolf precision matrix + Fisher-Z (19,900); label-free, per-subject, cached with provenance and hashes (condition numbers <= 1088, all finite) | as P1_LR | 4 min |
| 8 | P3_TS_LR | 3 | tangent-space vectors; geometric-mean reference fitted on the outer-training covariances only (label-free); inner CV uses features built with that reference (unsupervised, no labels) | as P1_LR | 26 min |
| 9 | P4_PCA_LR | 4 | LW correlation | in-fold PCA(whiten) with n_components in {25, 50, 100} chosen by inner CV, then L2 LR | 7 min |
| 10 | P4_KBEST_LR | 4 | LW correlation | in-fold SelectKBest(f_classif), k in {500, 2000, 5000} chosen by inner CV, then L2 LR | 4 min |
| 11 | P4_L1_LR | 4 | LW correlation | in-fold L1-regularized selection (SelectFromModel, liblinear L1, C in {0.05, 0.2}) then L2 LR with C in {0.01, 0.1, 1}; stability selection is not run (runtime) | 3 min |
Phase 4 is run for logistic regression (the reference model). If a Phase-2/3 candidate passes screening, its Phase-4 variants are added afterwards and screened by the same rule.
Not run: HistGradientBoosting (XGBoost covers gradient boosting); regularized graphical lasso (cost, and the Ledoit-Wolf precision is already regularized).

## 3. Phase 5: ensembles (conditional, predeclared)
Run only if at least two models with mean R1 accuracy >= the reference minus 1.0 point show complementary errors: pairwise phi coefficient of the error indicators (R1 predictions pooled over seeds 0-2) <= 0.70 and disagreement rate >= 8%.
Then two ensembles of the top-3 eligible models by mean R1 accuracy: **E1 soft vote** of Platt-calibrated scores (calibrators fitted on the inner OOF scores of the outer-training rows); **E2 stacking**: balanced L2 logistic regression (C = 1) on the members' inner OOF scores, applied to the members' test scores.
Meta-model inputs are always the training-only inner out-of-fold scores; no model is combined using predictions on subjects it has seen. The R1 threshold for ensembles comes from a 5-fold CV of the meta-model on the training rows. Ensembles count as candidates in the selection rule.

## 4. Phase 6: reporting
`final_report.md` with every experiment including negative results: R0/R1 tables (mean, SD, per seed), paired fold differences, screening outcome per candidate, replication, site-wise accuracy of the reference and the top candidate, failure cases (subjects misclassified in most runs and the sites they belong to), runtime, all hyperparameters, the source/config hashes (`source_and_config_hashes.json`) and an explicit statement on whether 85% was achieved under valid evaluation. The restricted holdout is not touched; a final evaluation would need a frozen model and your separate authorization.

## 5. Provenance (verified before every run)
Cache and baseline hashes as in `improvement_plan_v1.md` section 8. Campaign caches: `cache/covs_train676.npy` sha256 `fea8ce802acf1a24b1daa2210496b21a2cdf4e9f860c5bc86a59eaed287d7b32`; `cache/features_partial_train676.npz` sha256 `6ad6c3139c6c3769a64706259ae4bd0f309758a0c68afda4f8332515f8904fe6` (recomputed Ledoit-Wolf features equal the verified cache to 1e-12).
`campaign.py` sha256 at plan time `f1770d3dc7a2e932ba8bde9446b09513472b681de632dbe5dee4ae9a3d0d33ae`.
