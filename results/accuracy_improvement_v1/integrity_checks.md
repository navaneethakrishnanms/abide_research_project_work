# Integrity checks -- accuracy_improvement_v1

| Check | Result | Evidence |
|---|---|---|
| Cohort, labels, sites, folds | PASS | 600 development subjects, 19 sites; order, labels and sites asserted identical to the fast experiment's manifest; outer test folds asserted identical to the stored assignment. |
| Frozen baseline reproduced | PASS | E0 with inner seed 42 gave predictions and scores identical to the saved AAL116 LR_L2 results in all 5 folds. |
| Learned steps use training rows only | PASS | `fit_on_training(Xtr, ytr, ...)` receives only training rows: imputation, Fisher-Z, scaler, SelectKBest, C/k search (inner stratified 3-fold) and the threshold (inner 5-fold out-of-fold probabilities on the training rows) are all fitted there. The outer-test matrix is passed only to `predict_proba` afterwards. |
| No threshold or hyper-parameter chosen with outer-test labels | PASS | Outer labels are used only to compute metrics. |
| No selection from outer scores | PASS | The improvement rule was fixed in advance and applied mechanically; no candidate is presented as an unbiased best-of-four. |
| Leave-site-out used only once, for sensitivity | PASS | Single pass, seed 42, results not used to choose or retune anything; training and test sites disjoint (asserted). |
| One prediction per subject per candidate/seed | PASS | Each subject lies in exactly one outer test fold (and one leave-site-out site); `oof_predictions.csv` and `leave_site_out_oof_predictions.csv` use anonymized IDs `DEV0000`-`DEV0599`, site, y_true, score, threshold, y_pred. |
| Restricted holdout, `RESTRICTED_id_membership.csv`, CC200 | PASS | Not opened or used (the only mention in the script is the docstring). |
| Protected files | PASS | SHA-256 of the 311 baseline files: 0 changed before and after; the SSAE report, the leave-site-out folder and all earlier outputs untouched. |
| Background processes | PASS | None left running. |

Limitations: subject-level scores are saved for every candidate, so all metrics here can be recomputed; significance tests were not performed; five folds only.
