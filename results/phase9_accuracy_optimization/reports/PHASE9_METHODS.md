# Phase 9 Methods

Outer 5-fold split is identical to Phase 6 (StratifiedKFold, seed=42, subject-level). Each outer fold's
training portion is split 85/15 (stratified, seed=42) into inner train/validation, exactly matching
Phase 6's own internal monitoring split. All hyperparameter selection used ONLY the mean inner-validation
metric across the 5 outer folds. Outer test folds were accessed only for: (1) EXP_00_BASELINE, a one-time
reproduction check of the frozen Phase 6 configuration; (2) the simpler-classifier comparators (Section 13
explicitly permits reporting their test performance as comparators, not for selecting the final model); and
(3) the final locked configuration's one-time evaluation across 3 predefined seeds, performed only after
every hyperparameter had already been fixed from validation data alone.

Selection criterion (defined before any outer-test access for the SSAE track): primary = mean validation
ROC-AUC across the 5 outer folds; secondary = mean validation balanced accuracy; tie-break (within 0.005
ROC-AUC) = prefer the simpler / more strongly regularized configuration.

Staged design (Section 15): (1) baseline reproduction, (2) Elastic Net C/l1_ratio one-at-a-time sweep
(11 configurations, not the full 35-cell grid), (3) SSAE architecture (5 candidates) then
weight_decay/dropout (documented sequential sub-sweeps, not a full cross), (4) learning rate, early
stopping, batch size, (5) class weighting (balanced vs none), (6) simpler-classifier comparison
(Logistic Regression, Linear SVM; RBF SVM skipped for compute-time reasons), (7) multi-seed stability of
the locked configuration (seeds 42/123/2026).
