# Integrity checks

- Subjects, labels, folds identical to the development manifest and the frozen baseline (asserted): PASS
- AAL-116 only; CC200, holdout and restricted file not used: PASS
- One OOF prediction per subject for CPM: 600 rows / 600 unique: PASS
- Edge selection, strengths, scaler and logistic model fitted on training rows only; p-threshold chosen by inner 3-fold CV inside each outer-training set: PASS
- Test-row independence perturbation and independent recomputation of edge correlations from training rows (fold 0): PASS
- Previously hashed files after run (311): changed=0: PASS
- No method selection from outer-fold scores (single pre-specified candidate): PASS
- Baseline results read from saved files, not modified: PASS
