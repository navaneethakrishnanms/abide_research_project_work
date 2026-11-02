# Integrity and protocol checks -- static AAL-116 diagnostics

| Check | Result | Evidence |
|---|---|---|
| Development cohort used consistently (600 subjects, AAL-116 only) | PASS | All analyses read the fast experiment's AAL116 OOF predictions (4 models x 600 unique subjects, labels and folds matched the manifest) and `fc_raw_AAL116.npy`; CC200 not used. |
| Outer folds unchanged | PASS | The script recomputes `StratifiedKFold(5, shuffle, seed 42)` and asserts every test fold equals the manifest's `outer_fold` assignment (assertion passed). |
| Preprocessing and feature selection fitted within training folds | PASS (by source) | `fast_accuracy_experiment.py` lines 128-129: Pipeline = MeanImputeFisherZ -> StandardScaler -> SelectKBest -> classifier, fitted per outer-training set; outer test folds partition the 600 subjects (assertion at line 210). |
| Hyperparameter selection excludes outer-test data | PASS (by source and artifacts) | `GridSearchCV` with inner `StratifiedKFold(3)` on training rows only (line 175); saved checkpoints hold `inner_cv_auc` and `best_params`. Inner AUC vs outer AUC (AAL-116): LR_L2 0.685/0.694, LinearSVM 0.685/0.694, L1_LR 0.636/0.653, RBF_SVM 0.691/0.698 (no optimism). |
| Diagnostics fit no classifier | PASS | The only new computation is a univariate F-score ranking on training rows of each outer fold (section 7); nothing was selected or tuned from it, and no outer-test result entered it. |
| No holdout access | PASS | The script opens only: the fast experiment's manifest/OOF/fold files, the cohort CSV filtered to development IDs, and the cached development AAL feature matrix. No reference to the holdout list or the restricted membership file other than the docstring. |
| Previous files unchanged | PASS | SHA-256 of the 311 baseline files after the run: 0 changed. Newest-file timestamps in all earlier result folders predate this run. |
| Historical Phases 3-10 labeled correctly | PASS (statement) | Phases 3-10 numbers (61.5%, 64.9%, 63.0%) are cross-validation estimates on the 780-subject cohort, which contains 158 of the 170 designated-holdout subjects (see `evaluation_independence_audit`). No independent-holdout accuracy is claimed anywhere. |
| Missing artifacts | REPORTED | Selected-feature indices for the 600-subject runs were not saved (only the chosen k), so stability could only be described with a new training-rows-only filter computation, labeled as such. Per-frame FD and the TR are not available. |
| Background processes | PASS | None left running. |
