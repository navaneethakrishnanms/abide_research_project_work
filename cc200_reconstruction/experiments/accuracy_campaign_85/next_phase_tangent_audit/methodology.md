# Methodology: strictly nested tangent-space evaluation

CC200 only; 676 authorized training subjects; protected holdout not accessed. Code: `tangent_nested.py` (evaluator, validity tests, hash inventory), `report_next.py` (tables and decisions). Plan and frozen criteria: `experiment_plan.md`.

## 1. Audit of the previous tangent-space preprocessing (Priority 1)
- **Verified in the code** (`campaign.py`, `provider("TS")`): per outer fold, `TangentEmbedding("tangent").fit(covs[tr])` is called once on all outer-training covariances; the training and test matrices are transformed with that reference; the features then enter a `GridSearchCV` (inner 5-fold, chooses C) and a `cross_val_predict` (inner 5-fold, seed + 1000, gives the threshold scores). Consequently every inner-validation subject had contributed to the reference that transformed it. No diagnosis label is involved, and outer-test rows were never used, but this is not strict nesting for hyperparameter and threshold selection.
- **Why it matters little in principle, and why it is corrected anyway:** the reference is an unsupervised summary of 540 matrices; removing 108 of them moves it slightly (measured below: `ref_dist` and relative feature change per fold). The correction removes the doubt rather than assuming it is negligible.

## 2. The corrected procedure (mode "nested")
For each outer fold (training rows `tr`, test rows `te`; outer StratifiedKFold(10, shuffle, seed)):
1. **Final model:** reference G_out = geometric mean of the covariances of `tr` only; tangent vectors (whitening by G_out^-1/2, matrix logarithm, upper triangle x sqrt(2)) for `tr` and `te`; age and sex appended; demographics median-imputed and standardized and all columns standardized with statistics from `tr` only (`DemoFCPrep`).
2. **C selection:** for each of 5 inner splits of `tr` (StratifiedKFold(5, shuffle, seed)): G_k = geometric mean of that split's **inner-training** covariances only; inner-training and inner-validation matrices transformed with G_k (inner-validation rows are transformed, never used to fit); preprocessing fitted on the inner-training rows; logistic regression for each C in {0.001, 0.01, 0.1, 1, 10}; validation AUC. The C with the highest mean AUC is selected (first on ties, as GridSearchCV).
3. **Threshold scores:** the same inner procedure with StratifiedKFold(5, shuffle, seed + 1000) and the selected C produces out-of-fold probabilities for every training row; the R1 threshold maximizes balanced accuracy on them (ties toward 0.5; identical function to the campaign's).
4. The final logistic regression (selected C) is fitted on `tr` and scores `te`; predictions: R0 = score >= 0.5, R1 = score >= threshold. Outer-test labels are not an argument of the evaluator (checked in a test).
Folds, C grid, classifier (balanced L2, lbfgs, max_iter 5000), inner seeds and the metrics are identical to the campaign's, so arms are paired fold by fold.

## 3. Evidence that the implementation is right
- **Old-mode reproduction (real data).** The same evaluator with `mode="old"` (reference fitted once on `tr` and reused by every inner fit, i.e. the previous behaviour) reproduces the saved `P3_TS_LR` results for seed 0, folds 0 and 1 with **zero difference**: test scores, selected C (0.001 and 0.1), threshold and all inner out-of-fold scores identical. The only code difference between "old" and "nested" is where the reference is fitted.
- **Synthetic validity tests (5/5 pass):** (T1) changing inner-validation covariances leaves the reference and the inner-training features bit-identical; (T2) changing outer-test rows (covariances and demographics) leaves C, threshold, inner OOF scores and every other test row's score unchanged; (T3) the evaluator's signature has no outer-test labels; (T4) nested and old modes differ in features; (T5) determinism.
- **Diagnostics stored per fold in nested mode:** mean affine-invariant distance between the inner reference and the outer reference (`ref_dist`) and the relative Frobenius change of the validation features between the two references (`feat_rel_change`).

## 4. Evaluation and decisions
Metrics: accuracy, balanced accuracy, sensitivity, specificity, precision, F1, ROC-AUC (pooled out-of-fold per seed), confusion matrices, per-fold accuracy, mean and SD over seeds, paired fold differences against the reference. Primary reading R1, secondary R0. Decision rules and stage schedule: `experiment_plan.md` sections 2-3 (frozen before the first corrected result). The soft-vote hypothesis H3 reuses the saved ensemble construction on the training-only inner OOF scores of its two members.

## 5. Deviations from the brief and limitations
- **Seeds.** The brief named seeds 3-5 for confirmation. The old tangent model's strong result on those seeds motivated this work, so they are not independent for this hypothesis; the formal confirmation therefore uses never-evaluated seeds 6-8, and seeds 3-5 are run as a supplementary comparison (stated in the plan before any corrected result existed).
- All seeds use the same 676 subjects (new partitions, not new subjects); 30 paired folds are not independent; accuracy SE is about 1.8 points. No result is a statement about generalization to other cohorts or about the protected holdout.
- The geometric mean is computed with the project's own `geometric_mean` (tolerance 1e-5, at most 30 iterations); no warm start from a mean that contains validation rows was used.
