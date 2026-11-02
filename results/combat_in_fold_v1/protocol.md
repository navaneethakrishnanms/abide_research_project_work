# combat_in_fold_v1 -- protocol (written BEFORE any code was run)

## Question
Does removing per-site location and scale effects from the AAL-116 Fisher-Z connectivity features, with every harmonization parameter learned from
training rows only, raise the accuracy of the frozen L2 logistic-regression baseline on the 600 development subjects?

## Fixed setup (identical to the frozen baseline / `accuracy_improvement_v1` control E0)
- Cohort: the 600 development subjects (`splits/phase2_train_subjects.txt` INTERSECT `results/phase2/fd_filtered_subjects.csv`); order, labels, sites asserted
  identical to `results/fast_accuracy_experiment/data/dev_split_manifest.csv`.
- Outer folds: StratifiedKFold(5, shuffle, seed 42), identical to the stored assignment. Inner seeds 42, 123, 2026.
- Features: AAL-116 Pearson correlations (6,670 edges), cached `fc_raw_AAL116.npy`.
- Downstream pipeline unchanged: StandardScaler -> SelectKBest(f_classif) -> LogisticRegression(L2, class_weight balanced, max_iter 2000);
  grid C {1e-4, 1e-3, 1e-2, 1e-1} x k {500, 2000, all}; inner StratifiedKFold(3, seed) ROC-AUC; threshold 0.5.
- Comparator: `results/accuracy_improvement_v1/fold_metrics.csv`, candidate `E0_control_baseline` (same folds and seeds; E0 seed 42 was verified identical
  to the frozen baseline). E0 is read, not re-run.

## The one change (candidate H)
A harmonizer replaces the Fisher-Z step: training-column-mean imputation and Fisher-Z (as before), then a **ComBat-style empirical-Bayes location/scale
adjustment** (Johnson, Li & Rabinovic 2007; own implementation, parametric prior), applied to the 6,670 features:
- Batch = acquisition site. Covariates kept in the model: **age (z-scored with training mean/SD) and sex (female indicator)**.
- **Diagnosis labels are not used anywhere in the harmonizer** (neither in estimation nor in the transform): the standing project rule forbids using
  diagnosis labels to fit preprocessing. Consequence (a limitation, stated in advance): sites differ in ASD share (26%-60%), so some class signal can be
  absorbed into the site terms.
- The map is label-free and identical for training and test rows: `y_adj = (z - mean_i - gamma*_b) / sqrt(delta*2_b) * sqrt(var) + mean_i`, with
  `mean_i = grand_mean + covariates_i . beta`. gamma*, delta*2, grand_mean, beta, var are estimated on the training rows of the current fit only.
- Small batches: training sites with fewer than 5 training subjects are pooled into one "other" batch; if the pool still has fewer than 5 subjects, or the
  test subject's site is not among the fitted batches, the features are left unchanged (gamma = 0, delta2 = 1). Features with pooled variance < 1e-10 are left unchanged.
- The harmonizer is a pipeline step, so inside GridSearchCV it is re-fitted on each inner-training split; the outer-test rows never enter any fit.
- Site, age and sex are passed to the pipeline as three extra input columns and removed by the harmonizer; they are never features of the classifier.

## Leakage controls (checked in code before the full run)
1. Harmonizer `fit` takes no labels; parameters computed from the rows it receives only.
2. Self-test on synthetic data: (a) a planted site shift is removed on held-out rows using parameters fitted on other rows; (b) changing held-out rows
   (features, site, age) does not change the fitted parameters or any other held-out row's output.
3. Outer-test rows are never arguments of any fit (same structure as `accuracy_improvement_v1`).
4. Integrity: SHA-256 of the previously hashed baseline files before and after; existing results not modified; restricted holdout, its ID file, CC200 and the
   780 cohort not opened.

## Predeclared acceptance rule (unchanged from accuracy_improvement_v1)
Candidate H "improves" over E0 only if, for at least 2 of the 3 inner seeds, mean accuracy AND mean balanced accuracy over the 5 outer folds are each
at least 0.02 higher than E0 with the same seed, and accuracy is higher in at least 4 of the 5 folds. Nothing is chosen from outer-fold scores; no
threshold, grid, covariate or rule is changed after seeing results. No significance test is claimed.

## Reporting (all numbers from the run, nothing else)
Per fold and seed: accuracy, balanced accuracy, sensitivity, specificity, precision, F1, AUC; paired differences against E0; selected C and k; one
diagnostic per outer fold (mean per-edge between-site variance share on training rows before and after harmonization). Runtime, warnings, and the verdict.
An improvement, if the rule is met, would be a **development cross-validation** result only, not independent validation, and 85% would remain unsupported
unless the measured accuracy actually reaches it.

## Estimated runtime and outputs
About 5-10 minutes single-threaded numerics (15 outer fits, each a 12-point inner grid x 3 folds + refit; workers n_jobs=6).
New files only, under `results/combat_in_fold_v1/`: `protocol.md`, `combat_harmonizer.py`, `run_combat_in_fold.py`, `checkpoints/`, `logs/`,
`fold_metrics.csv`, `oof_predictions.csv`, `comparison_vs_E0.csv`, `site_variance_diagnostic.csv`, `config.json`, `final_report.md`.
