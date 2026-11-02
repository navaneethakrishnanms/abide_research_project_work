# Integrity checks -- FD-matched sensitivity analysis

| Check | Result | Evidence |
|---|---|---|
| Inputs aligned | PASS | Asserted before analysis: 600 unique manifest subjects; exactly one saved AAL-116 LR_L2 OOF prediction each; labels, folds and diagnosis equal the manifest and the cohort file; FD and scores finite. |
| Only development subjects, AAL-116, saved predictions | PASS | Reads the fast experiment's manifest and `oof_predictions.csv` (AAL116/LR_L2 rows) and the cohort CSV filtered to development IDs. CC200 rows are not used. |
| No model trained | PASS | The script fits nothing; it only computes metrics from saved scores and predictions. |
| Matching independent of predictions | PASS | Matching uses only mean FD and diagnosis, a seeded subject order and a caliper fixed in advance (0.2 x SD of FD); scores and errors are first used after matching. |
| Protocol not tuned | PASS | One run; caliper, seeds, bootstrap/random counts were fixed before running and not changed (first execution failed on a column-name bug before any analysis, fixed without altering the protocol). |
| Holdout and restricted file | PASS | Not opened; the only matches for the restricted/holdout names in the script are the docstring. |
| Previously hashed files unchanged | PASS | SHA-256 of the 311 baseline files: 0 changed; newest timestamps in earlier result folders predate this run. |
| Background processes | PASS | None left running. |

Limitations: the matched subset has a narrower FD range; matching is on mean FD only; saved models were trained on the unmatched cohort; bootstrap intervals are descriptive and ignore training uncertainty; twelve reference comparisons without multiplicity adjustment.
