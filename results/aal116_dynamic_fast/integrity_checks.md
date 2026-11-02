# Integrity checks

- Only AAL-116 time series of the 600 development subjects were opened (files filtered by development IDs; CC200 not used): PASS
- Outer folds identical to the verified manifest, disjoint, partitioning all 600 subjects (asserted at start): PASS
- One OOF prediction per subject per method: DYN_VAR: 600 rows, 600 unique, DYN_MAC: 600 rows, 600 unique : PASS
- Saved static baseline matched the manifest (600 unique OOF subjects, labels and folds equal): PASS
- Dynamic features are computed per subject from that subject's own time series; imputation, scaling, SelectKBest and C are fitted inside training folds only; inner 3-fold CV picks settings from training rows: PASS
- Test-row independence perturbation (scrambling half the test rows leaves other predictions unchanged): PASS
- Previously hashed files (311) after run: changed=0: PASS
- Holdout labels/features/predictions/performance and the restricted membership file not accessed (script reads only the train list, the cohort CSV filtered to development IDs, dev AAL files): PASS

Limitations: development subjects overlap Phase 6-10 cohort members; 5-fold CV; static baseline pipeline differs from the dynamic one only by the first preprocessing step; window defined in TRs (TR unknown); constant ROIs yield NaN edges handled by the stated rule.
