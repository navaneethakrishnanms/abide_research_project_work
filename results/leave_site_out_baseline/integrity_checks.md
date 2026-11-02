# Integrity checks -- leave-one-site-out baseline

| Check | Result | Evidence |
|---|---|---|
| Cohort, labels, sites | PASS | 600 development subjects (268 ASD / 332 TC), 19 sites; subject order, labels and site labels asserted identical to the fast experiment's manifest. |
| Frozen baseline configuration | PASS | Same code path as the baseline (`fast_accuracy_experiment.pipe` + `MODELS['LR_L2']` grid, inner StratifiedKFold(3, seed 42), ROC-AUC); nothing tuned on a held-out site. |
| Training sites only | PASS | For every fold, training and test site sets are disjoint (asserted); imputation, Fisher-Z, scaler, SelectKBest and hyper-parameter search are Pipeline/GridSearchCV steps fitted on the other sites only. A perturbation check (scrambling half of a held-out site's rows) left the other rows' predictions unchanged. |
| One prediction per subject | PASS | Coverage count equals 1 for all 600 subjects (asserted); 600 rows in `oof_predictions.csv`. |
| Failures / undefined metrics | PASS (none) | All 19 sites completed; both classes are present in every test site, so AUC is defined for all (meaningless for CMU, n = 2, flagged). Small sites (n < 20) flagged in every table. |
| Aggregates | PASS | Pooled subject-level metrics computed over all 600 predictions; macro means labeled as such; pooled AUC caveat stated. |
| Anonymization | PASS | `oof_predictions.csv` contains `DEV0000`-`DEV0599` (manifest row order), site, y_true, score, y_pred only; no original subject IDs written and no mapping file created. |
| Holdout, restricted ID file, CC200 | PASS | Not opened or used (script contains no such file references other than docstring text). |
| Protected files | PASS | SHA-256 of the 311 baseline files: 0 changed before and after; the SSAE final report and all earlier experiment outputs untouched. |
| Background processes | PASS | None left running. |

Notes: the first execution computed and checkpointed all 19 sites and then stopped at an aggregation bug (a boolean column of mixed type); the fix touched only the reporting code, and the rerun reloaded the checkpoints without refitting (the logged "runtime 0s"). `within_site_auc_comparison.py` is a read-only descriptive script; its five-fold within-site AUCs come from five different fold models and are not directly comparable with the leave-site-out values.
