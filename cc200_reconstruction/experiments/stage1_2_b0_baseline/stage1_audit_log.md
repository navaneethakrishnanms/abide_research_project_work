# Stage 1 audit log: CC200 training-data integrity (676 authorized Phase-2 training subjects)

Run 2026-10-22T06:38:46+00:00 | runtime 9s | **ALL CHECKS PASSED** (20/20)

Scope: per-subject raw CC200 files of the 676 training-list IDs only; `ts_cc200.npy`, `covs_cc200.npy`, the holdout list, `RESTRICTED_id_membership.csv`, `holdout_guard.py` and every holdout label/feature/prediction were not opened.

| Result | Check | Detail |
|---|---|---|
| PASS | training list has 676 unique IDs | 676 lines, 676 unique |
| PASS | every training ID has a raw CC200 file | missing 0 |
| PASS | no training ID has more than one raw file | 0 duplicated |
| PASS | one unique phenotype row per training ID | 676 rows, unique=True |
| PASS | phenotype FILE_ID equals the raw file name for every subject | 676/676 |
| PASS | site in FILE_ID prefix agrees with SITE_ID | 19 sites |
| PASS | labels are DX_GROUP in {1,2} | [1, 2] |
| PASS | class counts match the earlier independent audit (313 ASD / 363 TC) | 313 / 363 |
| PASS | age and sex complete and valid | age 6.5-58.0, sex values [np.int64(1), np.int64(2)], missing 0 |
| PASS | every series is (T x 200) | ROI counts [np.int64(200)] |
| PASS | every series has >= 100 timepoints | T range 116-316 |
| PASS | all time-series values are finite | no NaN/Inf |
| PASS | no two subjects share an identical series | 676 distinct |
| PASS | covariance array is (676, 200, 200) | (676, 200, 200) |
| PASS | covariances symmetric (max asymmetry < 1e-10) | 0.00e+00 |
| PASS | covariances positive definite | min eigenvalue 4.515e-02 |
| PASS | correlations in [-1, 1] with unit diagonal | max|r| 1.000000 |
| PASS | feature matrix is (676, 19900) and finite | (676, 19900) |
| PASS | independent recomputation (own Ledoit-Wolf/arctanh/triu code) equals tangent.py on 20 subjects | max abs diff 0.00e+00 |
| PASS | feature generation is deterministic (recomputed for 5 subjects) | identical |

Information: 13 of 676 subjects have at least one constant ROI (max 12); T range 116-316; 19 sites; ASD 313 / TC 363.
Not done on purpose: comparison of raw-file features with the rows of `covs_cc200.npy` (it needs the combined array that contains restricted holdout rows). The end-to-end check against the recorded phase 2h result replaces it.
Output: `subject_manifest.csv` (IDs of the training list only), `features_cc200_train676.npz` (cache of the 676 x 19,900 features).