# Experiment protocol (written before running; not to be changed after seeing results)

## Justification (stated plainly)
The brief lists Ledoit-Wolf shrinkage covariance with a fold-scoped tangent-space reference as the main untested direction. The method has already been evaluated (Phase 2e, CC200, 676 subjects, with age and sex covariates): 67.7% vs 67.2% for correlation, no meaningful gain. The only new elements here are the primary atlas (AAL-116, where the time-points-per-region ratio is higher than for CC200), the FC-only 600-subject protocol, and three inner-CV seeds. The prior probability of a gain is low; the experiment is a cheap replication on the primary atlas that resolves the question under the primary protocol, using the existing verified code (`tangent.py`, read-only). It is the only experiment in this run. If it fails, the run stops and no further modeling is recommended.

## Hypothesis
Tangent-space features from Ledoit-Wolf covariances, referenced to the geometric mean of the training subjects' covariances, carry connectivity information that the Fisher-Z Pearson representation does not, and improve accuracy over the frozen AAL-116 baseline.

## Data and splits
- The same 600 development subjects, labels and sites as the baseline; the same five outer folds (StratifiedKFold(5, shuffle, seed 42)); inner stratified 3-fold CV with seeds 42, 123, 2026 (outer folds fixed).
- AAL-116 ROI time series of the development subjects only. Holdout, `RESTRICTED_id_membership.csv`, CC200 and the 780-subject cohort are not used.

## Representation (tangent.py, unmodified)
Per subject (parameter-free with respect to other subjects): z-score each ROI series, Ledoit-Wolf covariance (assume_centered=True), symmetrize. Per training set: reference G = Riemannian geometric mean of the training covariances (max_iter 30, tol 1e-5); tangent vector = log(G^-1/2 C G^-1/2), off-diagonal upper triangle times sqrt(2) = 6,670 features. In the nested CV the reference is re-estimated inside every inner-training split and on the whole outer-training set for the final model; validation/test subjects are only transformed. No NaN imputation is needed.

## Model and selection
Identical to the baseline apart from the first step: StandardScaler -> SelectKBest -> L2 logistic regression (balanced); grid C {1e-4, 1e-3, 1e-2, 1e-1} x k {500, 2000, all} chosen by inner stratified 3-fold ROC-AUC on training rows; decision threshold 0.5.

## Comparison
Paired, per seed and fold, against the baseline procedure with the same seed (the saved `accuracy_improvement_v1` control E0, whose seed-42 predictions reproduce the frozen baseline exactly). Metrics: accuracy, balanced accuracy, sensitivity, specificity, F1, ROC-AUC, confusion matrix, fold and seed variation, runtime, peak memory. Anonymized out-of-fold predictions with one prediction per subject per run.

## Success criteria (all must hold)
1. Seed-averaged mean accuracy and balanced accuracy each higher than the baseline by at least 0.02.
2. At least two of the three seeds individually show accuracy and balanced accuracy gains of at least 0.02 with accuracy higher in at least 4 of 5 folds.
3. No material degradation: seed-averaged AUC no lower than baseline minus 0.01; sensitivity and specificity each no lower than baseline minus 0.03.
4. No integrity-check failure.
Otherwise the result is reported as "no improvement". No significance test is claimed; five folds of 120 subjects.

## Secondary, reported separately (never pooled with the above)
One leave-one-site-out pass (seed 42) with the same tangent procedure (reference fitted on the 18 training sites only; inductive, applicable to single subjects), compared with the saved leave-site-out baseline (62.2%). It is not used for decisions. Test-site centering and other transductive adaptations are not part of this run.

## Stopping rule
Run the smoke test (one fold) and then the full run once. No parameter, grid or criterion changes after the first outer score is seen. Stop after reporting.
