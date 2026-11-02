# Integrity checks -- accuracy_research_v3

| Check | Result | Evidence |
|---|---|---|
| Protocol written before the run, threshold predeclared | PASS | `experiment_protocol.md` preceded the smoke test and full run; arms, grids, criteria unchanged afterwards. |
| Cohort, labels, sites, folds | PASS | 600 development subjects; order, labels and sites asserted identical to the manifest; outer folds identical to the stored assignment. |
| Age and sex alignment | PASS | Complete for all 600; cohort file and phenotype file identical (max age difference 0.0, sex identical); diagnosis from the phenotype file matches the manifest; non-development rows discarded immediately after reading. |
| Only prediction-time-legitimate variables | PASS | Age, binary sex and site (control B2 only); no diagnosis-derived, post-diagnosis, severity or medication variable. |
| Fitted steps on training rows only | PASS | Imputation, Fisher-Z, scaler, SelectKBest, demographic standardization and the C/k/w search (inner stratified 3-fold) are fitted on training rows in `FCDemo`/`DemoOnly`; a perturbation check (scrambling half the test rows, including their age) left the others' predictions unchanged. |
| No labels used to construct features | PASS | Only training-fold SelectKBest uses training labels, as in the baseline. |
| No selection from outer-fold results | PASS | Threshold fixed in advance; arms C and D failed it and no further arm or tuning was run. |
| One prediction per subject per arm and seed | PASS | 600 rows and 600 unique anonymized IDs for each arm and seed (asserted). |
| Baseline pairing | PASS | `accuracy_improvement_v1` control E0 (seed 42 identical to the frozen baseline). |
| Experiment 2 uses saved predictions only | PASS | Baseline and tangent out-of-fold files; no model fitted, no subject removed, no variable invented (motion measure = subject-level mean FD). |
| Restricted data, CC200, 780 cohort | PASS | Not opened or used (script mentions them only in docstrings). |
| Protected files | PASS | SHA-256 of the 311 baseline files: 0 changed before and after. |
| Resources | Recorded | Experiment 1: 70 s, peak resident memory 305 MB. |
| Background processes | PASS | None left running. |

Limitations: five folds; no significance tests; Experiment 2 is descriptive; AUC values are computed from saved scores (recomputable from `oof_predictions.csv`).
