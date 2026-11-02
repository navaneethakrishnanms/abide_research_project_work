# Leakage and integrity audit -- site-centering experiment

| Check | Result | Evidence |
|---|---|---|
| Locked holdout not opened | PASS | Script reads only `phase2_train_subjects.txt` (via the fast experiment's `build_dev_set`); grep for holdout file/guard references in the new script = 0. |
| Same subjects, labels and folds as the fast experiment | PASS | Asserted at start: subject order and labels equal `dev_split_manifest.csv`, and each outer test fold equals the stored `outer_fold` assignment. |
| Site offsets from training rows only | PASS | `SiteFisherZCenter.fit()` learns column means, Fisher-Z and per-site offsets (site mean - global mean) from the rows it is fitted on. An independent recomputation of one site's offset from the outer-training rows matched the fitted offset (atol 1e-12) for all three candidates. |
| Inner CV uses per-inner-split offsets | PASS | The step is inside the Pipeline, so GridSearchCV refits it on each inner-training split; inner-validation rows only receive offsets. No offset is computed from inner-validation or test rows. |
| Test rows independent of each other | PASS | Replacing half of the test rows' features with junk left predictions of the other rows unchanged (atol 1e-12), for all three candidates. |
| Absent / small sites | PASS (documented) | Sites with fewer than 5 fitted rows (CMU, n=2 overall) get zero offset; 18 of 19 sites had offsets in every fold. Test-site statistics are never used. |
| Scaling, feature selection, hyperparameters | PASS | StandardScaler, SelectKBest and GridSearchCV (inner 3-fold, ROC-AUC) are Pipeline steps fitted on training rows only. |
| Baseline comparison fairness | PASS | Baselines are the fast experiment's saved fold results (same code path, subjects, folds, grids); nothing was recomputed or overwritten. |
| Candidate selection | EXPLORATORY | Three candidates are reported side by side. No best-of-three score is presented as an unbiased estimate. |
| Earlier phases / fast experiment unchanged | PASS | SHA-256 of 311 baseline files: 0 changed; splits folder timestamps unchanged; fast experiment files not rewritten by this run. |

Limitations: development subjects overlap the Phase 6-10 cohort; this is a 5-fold CV comparison with 120 test subjects per fold; no independent test was run.
