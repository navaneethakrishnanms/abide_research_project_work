# Integrity checks -- accuracy_research_v2 (tangent-space on AAL-116)

| Check | Result | Evidence |
|---|---|---|
| Protocol written before running | PASS | `experiment_protocol.md` (hypothesis, data, splits, parameters, metrics, criteria, stopping rule) preceded the smoke test and the full run; nothing changed afterwards. |
| Cohort, labels, sites, folds | PASS | 600 development subjects; order, labels and sites asserted identical to the manifest; outer test folds identical to the stored assignment. |
| Restricted data not used | PASS | No access to the holdout list or labels, `RESTRICTED_id_membership.csv`, CC200 or the 780-subject cohort (script mentions them only in the docstring). |
| Tangent reference fitted on training subjects only | PASS | Fitted separately in every inner-training split and on each outer-training set; replacing the test covariances with junk left the reference identical (fold 0); validation/test subjects only transformed. |
| Per-subject steps parameter-free across subjects | PASS | z-scoring and Ledoit-Wolf shrinkage use only the subject's own series. |
| Covariance validity | PASS | 600 x 116 x 116, symmetric, minimum eigenvalue 0.04 (all positive definite), median condition number 166; 0 non-converged geometric-mean computations. |
| Hyper-parameters selected inside training data | PASS | C and k by inner stratified 3-fold ROC-AUC on training rows; threshold fixed at 0.5; no outer or leave-site-out score used for any choice. |
| One prediction per subject per run | PASS | 600 rows and 600 unique anonymized IDs for each of seeds 42, 123, 2026 and for the leave-site-out pass. |
| Baseline for pairing | PASS | `accuracy_improvement_v1` control E0 (identical pipeline; seed-42 predictions reproduce the frozen baseline exactly). |
| Success criteria applied as written | PASS | Criteria 1 and 2 failed, 3 and 4 met; reported as "no improvement established". |
| Separate reporting of evaluation types | PASS | Random subject-level five-fold CV and leave-one-site-out are reported in separate tables and never pooled; no transductive adaptation was used in this run. |
| Protected files | PASS | SHA-256 of the 311 baseline files and of `tangent.py` unchanged before and after the run. |
| Resources | Recorded | Runtime 369 s; peak resident memory 1,190 MB; 4 worker processes for reference fitting. |
| Background processes | PASS | None left running. |

Not verified: AUC is as recorded from saved prediction scores (scores are saved in the OOF files, so AUC can be recomputed); no significance tests were run.
