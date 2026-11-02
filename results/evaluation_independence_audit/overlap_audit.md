# Evaluation-independence audit -- subject overlap

**Verdict: the locked Phase 2 holdout is NOT independent of prior model development. Classification: PARTIALLY OVERLAPPING with prior development.**

Only subject IDs were used. No holdout diagnosis labels, features, predictions or performance were read or printed. The restricted ID-level artifact is `RESTRICTED_id_membership.csv` (membership flags only).

## Authoritative lists (verified)
| List | Source | Size |
|---|---|---|
| 871-subject Phase 1 cohort | `abide_pcp/phenotypic_qc.csv` | 871 |
| 846-subject split pool | union of the two split files (split script keeps subjects with >= 100 CC200 timepoints; 25 of the 871 are not in it) | 846 |
| Development/train list | `splits/phase2_train_subjects.txt` | 676 |
| Locked holdout | `splits/phase2_holdout_subjects.txt` (seed 20260, stratified by site x diagnosis) | 170 |
| FD-filtered cohort used by Phases 3-10 | `results/phase2/fd_filtered_subjects.csv` (equals Phase 6's cohort-verification IDs) | 780 |
| FD-removed | `results/phase2/fd_removed_subjects.csv` (871 = 780 + 91, disjoint) | 91 |
| Development cohort of the recent experiments | fast-experiment manifest (= train list intersect 780 cohort, verified) | 600 |

Checks: holdout and train are disjoint; their union is 846 and lies inside the 871; dev600 is exactly train INTERSECT the 780 cohort.

## Exact overlaps with the holdout (170 subjects)
| Set | Overlap with holdout |
|---|---|
| Train list (676) | 0 |
| Dev600 (fast, site-centering, motion and diagnostic experiments) | **0** |
| 871-subject cohort (Phases 1-2) | 170 |
| 780 FD-filtered cohort (Phases 3-10) | **158 (92.9% of the holdout; 20.3% of the cohort)** |
| FD-removed subjects (91) | 12 |

The 180 cohort-780 subjects outside the train list are 158 holdout subjects plus 22 subjects that were not in the 846-pool at all.

## Which results used holdout subjects
| Phase / experiment | Subjects used | Holdout subjects in them |
|---|---|---|
| Phases 1-2 (inventory, FD filter) | 871 | 170 (descriptive only, no model) |
| Phases 3-5 (AAL feasibility, FC diagnostics, missing-FC analysis, Elastic Net design) | 780 | 158 |
| Phase 6 (61.5%), 7, 8 | 780 | 158 |
| Phase 9 (64.9%) | 780 | 158 |
| Phase 10 old and corrected (63.0%) | 780 | 158 |
| Phases 2c-2h dev-only (about 67.2% for CC200 logistic regression) | 676 | 0 (via `holdout_guard`) |
| fast / site-centering / motion / diagnostics (about 64-65%) | 600 | 0 |

## What can and cannot be claimed
- **Cannot:** call the holdout an independent test of the Phase 6-10 methodology. About 93% of its subjects were part of those phases' training and cross-validation, so Phase 6, 9 and 10 numbers must stay labeled as cross-validation estimates on a cohort that includes the holdout. Do not present them as independent-test performance.
- **Cannot:** claim independence from the early (August) development history. Those scripts (`train_fc_improved.py`, `train_fc_v2.py`, `fast_holdout_eval.py`) used a different, label-stratified split with seed 42 on the same 846-pool. Two independent random 170-subject draws overlap by about 34 subjects, so roughly 80% of today's holdout subjects were probably in the old development partition. This is an inference, not verified: reconstructing the old split would have required reading holdout labels and time series, which this audit does not do.
- **Can (with a caveat):** the 600-subject experiments have zero subject-ID overlap with the holdout. A single future evaluation of one frozen pipeline on the holdout would be free of subject-level training leakage. It would still not be fully independent in the research-process sense: atlas choice, preprocessing and grid ranges were shaped by Phases 6-10 results computed on a cohort that contained the holdout.
- **Evidence on whether the holdout was ever scored:** `experiment_audit_log.csv` shows 4 denied holdout-access attempts (guard self-tests) and 24 clean dev-only experiment records; there is no granted access. `FINAL_MODEL_SPEC.json` does not exist, so the designated final-evaluation script is not runnable. The guard only covered the Phase 2c-2h scripts; Phases 3-10 did not use it.
- **Not established:** that the holdout was never used in some other way (the guard cannot see scripts that bypass it).
