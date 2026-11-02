# Integrity checks -- cerebrum-only experiment

| Check | Result | Evidence |
|---|---|---|
| Atlas mapping verified before selecting ROIs | PASS | Mapping file: 116 rows, indices 1-116, all `mapping_verified` and `header_code_match`; ROIs 1-90 contain no cerebellar/vermis names, ROIs 91-116 are all cerebellar/vermis (asserted). |
| Feature count and recomputation | PASS | 600 x 4,005 matrix recomputed from the per-subject time series; equals the matching columns (via Phase 5A feature index) of the cached 6,670-edge matrix with identical NaN pattern (max difference < 1e-12). |
| Subjects, labels, sites, folds | PASS | Order, labels and sites asserted identical to the fast experiment's manifest; each outer test fold equals the stored assignment. |
| Frozen baseline reproduced | PASS | The 116-ROI run with inner seed 42 gave predictions and scores identical to the saved AAL116 LR_L2 results in all 5 folds. |
| Preprocessing and selection fitted on training rows only | PASS | Imputation, Fisher-Z, scaler, SelectKBest and the C/k search are Pipeline/GridSearchCV steps fitted on outer-training rows; a perturbation check (scrambling half of the test rows) left the other rows' predictions unchanged. The ROI subset is fixed and has no learned parameters. |
| No model selection from outer or leave-site-out results | PASS | The improvement rule was fixed in advance and applied mechanically; leave-site-out is a single sensitivity pass that was not used to choose or retune anything. |
| One prediction per subject | PASS | Each subject lies in exactly one outer test fold per seed and in one leave-site-out site; anonymized IDs `DEV0000`-`DEV0599` in `oof_predictions.csv` and `leave_site_out_oof_predictions.csv` (the 116-ROI leave-site-out rows are copied from the saved `leave_site_out_baseline` run). |
| Holdout, `RESTRICTED_id_membership.csv`, CC200 | PASS | Not opened or used (the script mentions them only in the docstring). |
| Protected files | PASS | SHA-256 of the 311 baseline files: 0 changed before and after; prior reports and experiment folders untouched. |
| Background processes | PASS | None left running. |

Limitations: five folds; no significance tests; metrics can be recomputed from the saved subject-level predictions for all representations and seeds.
