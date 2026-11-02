# Missing artifacts and unresolved questions (CC200 reconstruction)

## Missing artifacts
| # | Artifact | Why it matters |
|---|---|---|
| 1 | Fitted estimator (scaler + logistic regression + threshold) behind the 74.12% | The only way to inspect coefficients or re-score; never serialized by `train_fc*.py`. |
| 2 | Per-subject CV and holdout scores/predictions | Needed to recompute any metric, confusion matrix or confidence interval from data. Only aggregates were saved. |
| 3 | Run log for `fc_v2_results.csv` (exact invocation, printed C values, warnings) | The `--quick` identification is inferred from the row, not recorded. |
| 4 | Final-fit regularization C and other chosen hyperparameters | The recorded CSV has CV summaries only. |
| 5 | Software versions at run time (python, numpy, scikit-learn) | Needed for bit-exact reproduction. |
| 6 | Hash of `covs_cc200.npy` and the script that first created it | The file dates from 2026-08-19 16:26; the current `train_fc.py` reuses it if its row count matches; provenance of its row order rests on code, not on a recorded checksum. |
| 7 | Output of the Phase 1 leakage audit that declared the seed-42 holdout compromised | Cited in `holdout_guard.py` and `phase2a_create_split.py`; no audit file was found under `results/phase1`. |

## Unresolved questions
1. Was the 74.12% row produced by `train_fc_v2.py --quick`? (Strongly suggested; not recorded.)
2. How many subjects of the old seed-42 train/validation set are in the locked phase-2 holdout? (Needs the restricted ID list; not computed. Matters because it decides whether the old CV can be re-run at all.)
3. How many of the 200 CC200 ROIs are constant or near-constant in the development subjects, and how do the Ledoit-Wolf conversions treat them? (Needs the time series; not inspected. By construction the code produces no NaNs, but constant ROIs yield near-zero correlations.)
4. Do `covs_cc200.npy` rows align with `phenotypic_qc.csv` rows for every subject? (Code path says yes; not checked on data. A development-only check against the raw `*_rois_cc200.1D` files of training subjects would settle it.)
5. Is the AUC gap between the old holdout (0.81) and CV (0.71) sampling variation (about 2-2.5 standard errors) or a property of those 170 subjects? (Cannot be resolved without predictions; not testable on permitted data.)
6. Does the pipeline generalize across sites better or worse than random folds on the phase-2 training list? (Leave-one-site-out exists only on the full 846 pool, including the reused holdout subjects.)
7. Which preprocessing choice would a different variant require (e.g. no global signal regression, `filt_noglobal`)? (Out of scope; the CC200 files on disk are `filt_global` only.)
