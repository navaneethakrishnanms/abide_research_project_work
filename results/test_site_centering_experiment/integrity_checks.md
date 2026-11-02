# Integrity checks -- test-site centering experiment

| Check | Result | Evidence |
|---|---|---|
| Cohort, labels, sites, splits | PASS | 600 development subjects, 19 sites; order, labels and sites asserted identical to the manifest; training and held-out site sets disjoint for every split. |
| Protocol fixed before evaluation | PASS | Centering rule, supervised pipeline, grid, inner CV and the success rule are in the script docstring written before any run; nothing was changed after results appeared. |
| Held-out labels never used in transformation, fitting, selection or prediction | PASS | `center_by_site(Z, site)` takes no labels; labels are used only on training rows in the Pipeline/GridSearchCV and to compute metrics afterwards. |
| Supervised steps on training sites only | PASS | Imputation, scaler, SelectKBest, C/k search (inner stratified 3-fold) fitted on the other 18 sites; the held-out site is transformed only. |
| Site isolation of the centering | PASS | Perturbing another site's rows left a site's centered features identical; perturbing the held-out site left every other site's centered features identical; all site means after centering are 0 (max 6.6e-16). |
| Transductive dependence documented | PASS | A held-out subject's centered features depend on all subjects of its own site (and itself); no minimum batch size was enforced; CMU (n = 2) is degenerate and flagged. |
| Control isolating the centering effect | PASS | Uncentered arm through the same code path matches the frozen baseline (accuracy identical; pooled AUC 0.6836 vs 0.6837). |
| One prediction per subject | PASS | 600 rows, unique anonymized IDs `DEV0000`-`DEV0599` (manifest row order); no original subject IDs written. |
| Rule applied mechanically | PASS | Rule outcome: FAILED (3 of 4 conditions met); not altered. |
| Holdout, `RESTRICTED_id_membership.csv`, CC200 | PASS | Not opened or used (mentioned only in the docstring). |
| Protected files | PASS | SHA-256 of the 311 baseline files: 0 changed before and after; earlier reports and experiment folders untouched. |
| Background processes / follow-up experiments | PASS | None left running; none launched. |

Limitations: single pass, one inner-CV seed; transductive setting; five-fold control is outside the rule; metrics can be recomputed from `oof_predictions.csv` and `five_fold_control_oof_predictions.csv`.
