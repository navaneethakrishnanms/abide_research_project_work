# Integrity and reproducibility -- fast_safe_ssae (FINAL)

## Execution
| Item | Result | Evidence |
|---|---|---|
| Task `bjep10vq8` finished normally | PASS | Task status completed, exit code 0; log line "Reports written"; no Python process remained. |
| Within the 900 s budget | PASS | 645 s total; no "TIME BUDGET" line in the log. |
| Folds and seeds | PASS | Checkpoints `fold0`-`fold4` all `_complete: true`; each holds seeds 42, 123 and 2026. |
| Verification script | PASS (note) | `verify_and_tabulate.py` reported `problems: []` and `trace_all_consistent: true`. The step was flagged "failed" (exit 255) only because pandas FutureWarnings written to stderr are treated as errors by the shell; its outputs were written correctly. |

## Traceability to the development manifest
| Check | Result |
|---|---|
| Subject order, labels, outer folds identical to the verified manifest and frozen baseline | PASS (asserted by the run at start) |
| Confusion-matrix totals equal the manifest fold size (120) | PASS for all 20 entries (5 folds x [Elastic Net + 3 seeds]) |
| ASD count in each confusion matrix (FN + TP) equals the manifest's ASD count for that fold (54, 54, 54, 53, 53) | PASS for all 20 entries |
| Accuracy, balanced accuracy, sensitivity, specificity, precision, F1 recomputed from confusion counts equal the stored values | PASS (maximum absolute difference 0) |
| ROC-AUC independently recomputed | **NOT POSSIBLE**: subject-level scores were not saved and AUC cannot be derived from confusion matrices. AUC values are as recorded by the run. |
| Exactly one valid out-of-fold prediction per subject | **NOT DIRECTLY VERIFIABLE**: no subject-level predictions saved; only implied by fold sizes and the partition of the 600 subjects into five test folds. |

## Leakage controls (from code and log)
| Component | Result | Evidence |
|---|---|---|
| Imputation, Fisher-Z, scaling | PASS | `fit_prep` fitted per inner-training split and on the entire outer-train; test rows only transformed. Perturbation of outer-test rows left the outer-train matrix identical (fold 0; logged PASS in the smoke and full runs). |
| Elastic Net | PASS | Fitted per inner-training split (C selection) and on the entire outer-train (final). |
| SSAE training and early stopping | PASS | 15% hold-out of the training rows; inner-validation and outer-test rows unused for fitting or stopping. |
| Hyperparameter selection (C, SSAE setting) | PASS | Inner 3-fold ROC-AUC on outer-training rows; no outer score used; decisions by argmax (no threshold tuning). |
| Method selection | PASS | All methods reported; headline (EN+SSAE seed 42) predeclared; no winner chosen from outer scores. |
| Inner vs outer scores | PASS | Inner-selected AUC 0.627-0.676 vs outer AUC 0.643-0.707: no optimism. |

## Preservation
| Check | Result |
|---|---|
| 311 previously hashed files | PASS: 0 changed (before and after the run, and again at reporting). |
| Frozen baseline | PASS: read from the saved fast-experiment results; unchanged. |
| Restricted 170-subject holdout, `RESTRICTED_id_membership.csv`, CC200 | PASS: not opened; the run script's only mentions are docstring/report text. |
| New outputs only in the experiment folder and `final_report` | PASS |

## Missing or unavailable artifacts
- Subject-level out-of-fold predictions/scores (not saved by the run).
- `implementation_notes.md` (required by the original run specification, not produced).
- Outer-test metrics for SSAE settings S1, S2, S3 as separate models (never computed; only inner-validation AUC).
- Peak memory.

## Reproducibility
Script `results/fast_safe_ssae/run_fast_safe_ssae.py`, `config.json`, fixed seeds (selection 42; final seeds 42/123/2026), deterministic fold assignment (StratifiedKFold seed 42), single-thread numerical libraries, per-fold checkpoints and log. GPU non-determinism was not controlled; seed-to-seed accuracy differences were under 0.01.
