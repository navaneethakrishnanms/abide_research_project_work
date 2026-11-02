# Leakage and integrity audit -- fast accuracy experiment

| Check | Result | Evidence |
|---|---|---|
| Locked holdout not opened | PASS | Script reads only `splits/phase2_train_subjects.txt`; no reference to the holdout file or `holdout_guard` in the script (grep = 0). Dev set is defined by intersection with the train list, so holdout subjects are never selected. |
| Development cohort size | PASS | 600 subjects, 268 ASD / 332 TC, 19 sites, all IDs unique (`data/dev_split_manifest.csv`). |
| Atlases kept separate | PASS | AAL-116 (6,670 features) and CC200 (19,900) are separate matrices and separate runs; never concatenated. |
| Identical outer folds | PASS | One StratifiedKFold(5, seed 42) object; fold-to-subject assignment hash identical for all 8 candidates; outer test folds partition the 600 subjects (asserted), train/test disjoint per fold (asserted). |
| Imputation / Fisher-Z / scaling | PASS | `MeanImputeFisherZ` and `StandardScaler` are Pipeline steps fitted on training rows only (inside GridSearchCV and the outer fit). |
| Feature selection (SelectKBest) | PASS | Pipeline step, fitted on training rows only; k chosen by inner CV. |
| Hyperparameter selection | PASS | GridSearchCV with inner 3-fold on outer-training rows only; scoring uses inner-validation labels only. |
| Test-row independence | PASS | Perturbation check: replacing half of the test rows with junk leaves predictions of the other rows unchanged (run on both atlases). |
| Model/atlas selection among candidates | NOT NESTED | The best of eight candidates is picked by mean outer-fold accuracy. The reported best score is exploratory and mildly optimistic. |
| Earlier phases unchanged | PASS | SHA-256 of 311 baseline files (Phases 1-9, scripts, old Phase 10 run) after this experiment: 0 changed. Splits folder timestamps unchanged. |
| Phase 10 corrected results | PASS (by hash baseline) | Outputs only written under `results/fast_accuracy_experiment/`; new script is a new file. |

Caveats: the dev set is not independent of Phase 6-10 development (same 780-cohort subjects), and hyperparameter search ranges here were set by me, not tuned on outer folds. Results are CV estimates only; no independent test was run.
