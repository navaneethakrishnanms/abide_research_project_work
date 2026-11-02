# Pre-registered fresh-seed confirmation (written before any seed-3/4/5 result exists)

Source of the design: `independent_audit/audit_report.md` (sha256 `0de9d82a79e883dbf9e46fe0822bf2ce6235cbb9902c91e493a41d9f15951c42`), section 9. This is a separate confirmation of one effect (ensembling), **not** a rescue of the failed campaign screening; the frozen campaign rule (`selection_rule.py`, sha256 `e4389cbebcae96950fab28bb8f81ae5e81c7967a3d29ed82552e28730c266467`) is untouched.
Atlas: CC200 only. Data: the 676 authorized training subjects. The protected holdout, its IDs/labels/features/predictions and `RESTRICTED_id_membership.csv` are not accessed.

## Compared methods (fixed in advance)
- **Reference:** campaign experiment P1_LR, unchanged (Ledoit-Wolf correlation + Fisher-Z + age/sex, balanced L2 logistic regression, C in {0.001, 0.01, 0.1, 1, 10} by inner 5-fold AUC).
- **Ensemble:** P5_E1_SOFTVOTE exactly as saved: members **P3_TS_LR** (tangent-space + L2 LR; reference fitted on the outer-training covariances), **P2_LSVM** (balanced LinearSVC, C in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}) and **P1_LR**; each member's training-only inner out-of-fold scores (inner CV seed = seed + 1000) are Platt-calibrated by a one-dimensional logistic regression (C = 1e6) fitted on the training rows; the ensemble score is the mean of the calibrated probabilities; the decision threshold is chosen on the ensemble's own training-only inner scores (balanced-accuracy-maximizing, default 0.5). No member, setting or threshold rule is chosen or changed after seeing fresh-seed results.
- Members and settings are read from the saved artifacts (`results/P5_E1_SOFTVOTE_config.json`, `campaign.py` `make()` and `run_fold()`, `phase5.py`). Before the fresh seeds are run, the builder is checked to reproduce the saved seed-0-2 soft-vote predictions exactly; if it does not, the experiment stops.

## Design
Seeds 3, 4, 5; outer `StratifiedKFold(10, shuffle, random_state = seed)` over the same 676 subjects, identical for both methods; inner CV seed = outer seed. Everything fitted (scaling, tangent reference, hyperparameters, calibration, thresholds) uses the outer-training partition only; outer-test rows are only scored.
Primary reading **R1** (threshold from training-only inner scores), as in the audit's +0.74-point figure; R0 (default thresholds) is reported as secondary and does not enter the decision.

## Pre-registered success criterion (all three required; evaluated under R1; not changed after results)
1. Mean accuracy gain (ensemble minus reference, pooled out-of-fold accuracy per seed, averaged over seeds 3-5) >= **+0.5 percentage points**.
2. Accuracy gain **positive in at least 2 of the 3** fresh seeds.
3. Mean ROC-AUC gain (pooled out-of-fold AUC per seed, averaged) >= **+0.01**.
All three seed results are reported whatever the outcome. A pass supports only that the ensembling effect is real and small; it does not change the campaign verdict (no candidate passed the frozen rule; 85% not achieved).

## Outputs (this folder only; no existing file is modified)
`preregistration.md`, `confirm.py`, `checkpoints/`, `logs/`, `seed_metrics.csv`, `fold_comparison.csv`, `oof_predictions.csv`, `config.json`, `hashes.json`, `report.md`.
Estimated runtime: about 40 minutes (tangent-space is the slow member).
