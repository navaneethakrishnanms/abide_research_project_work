# Integrity report

| Check | Result | Evidence |
|---|---|---|
| Cohort, labels, folds | PASS | 600 development subjects; subject order, labels and the five outer test folds asserted identical to the fast experiment's manifest and the frozen baseline. |
| 780-subject cohort | Not used for training | Provenance only; this run trains on the 600-subject development cohort. |
| Preprocessing and Elastic Net fitted on training rows only | PASS | `fit_prep` (mean imputation, Fisher-Z, scaler) and the Elastic Net are fitted per inner-training split and, for the final model, on the entire outer-train; outer-test rows are transformed only. Perturbing the outer-test rows left the outer-train matrix bit-identical (fold 0). |
| Hyperparameter selection | PASS | Elastic Net C and SSAE settings chosen by inner 3-fold ROC-AUC on outer-training rows; early stopping uses a 15% hold-out of the training rows. No outer-test score, threshold or epoch count was used (decisions use argmax). |
| Method selection | PASS | All methods reported; headline (EN+SSAE, seed 42) fixed in advance. |
| Holdout, restricted ID file, CC200 | PASS | Not opened or used. |
| Previously hashed files | PASS | SHA-256 of the 311 baseline files after the run: changed=0. |
| One prediction per subject per fold | PASS | Each subject is in exactly one outer test fold. |
| Background processes | PASS | None left running. |
