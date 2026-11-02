# Independent audit of the CC200 accuracy campaign

Scope: read-only audit of `accuracy_campaign_85` (final report, plan, comparison table, phase reports, saved out-of-fold predictions, checkpoints, hashes). No model was trained and no existing artifact was modified; everything below was re-derived by separate code (`audit.py`) from the saved predictions, scores and stored inner-OOF arrays.
Only the 676 authorized training subjects (cache) were involved; the restricted holdout, its membership and `RESTRICTED_id_membership.csv` were not accessed. Full check list: `audit_checks.csv` (35 checks: 25 PASS, 10 INFO, 0 FAIL after correcting two bugs in my own audit script, listed in section 8).

## 1. Verdict
**The campaign's reported results are reproducible and its conclusion stands: no candidate passed the frozen rule and 85% was not achieved.** I found no leakage and no misapplication of the rule. Three statements in the final report need correction or qualification (section 7): the soft-vote "optimism" caveat is not supported by the evidence, the "39 runs" are not independent, and the 68-subject count depends on which runs are included.

## 2. Metrics reproduce from the saved predictions (question 1)
- Pooled accuracy, balanced accuracy, sensitivity, specificity, precision, F1 and confusion counts for 13 experiments x 3 seeds x 2 readings, recomputed from the stored predictions, equal `seed_metrics.csv` (max difference 5.6e-17); AUC recomputed from the stored scores matches (checks 1a, 1d).
- `all_experiments_comparison.csv` equals the recomputed means exactly (1b); all 13 accuracy cells (mean +/- seed SD) in the final report's ranking table match the recomputed values (1c).
- The reference run's scores equal the pre-campaign Arm B scores exactly (3f).

## 3. Same subjects and compatible folds (question 2)
- Every experiment and seed contains exactly the 676 authorized training subjects, once each (2a); `y_true` and site labels equal the verified cache (2b, 2e).
- The fold assignment is identical across all 13 experiments for each seed (2c) and equals an independent regeneration `StratifiedKFold(10, shuffle, seed).split(y)` (2d).
- Hashes in `source_and_config_hashes.json` equal the current files, including the baseline and cache; the frozen selection rule and `campaign.py` are unchanged since the plan (2f, 2g). The log contains no failure message (2h; note that a case-insensitive search for "FAILED" matches 11 "finished/failed" completion lines, which are not failures).

## 4. Threshold selection and preprocessing leakage (question 3)
- Inner out-of-fold scores exist exactly for the outer-training rows of each fold and are NaN for every outer-test row, for all 33 single-model runs x 10 folds (3a). The per-fold R1 thresholds, recomputed independently from those training-only scores and training labels, equal the stored thresholds (3b; the largest difference, 3e-8, is XGBoost's float32 output and changes no prediction), and R1 predictions equal "test score >= that threshold" (3c).
- A static check of `run_fold` shows every `.fit(...)` and `cross_val_predict(...)` uses only `Xtr`/`y[tr]`; `Xte` appears only in scoring (3d). The tangent reference is fitted on the outer-training covariances only (3e).
- **Residual limitation (documented in the plan, check 3g):** for the tangent-space experiment the reference is built from all outer-training rows, so the inner CV that selects C sees a reference computed with the inner-validation rows (label-free). This affects only hyperparameter selection in that one experiment and never involves outer-test rows. Strictly applied, rule 6 would refit the reference per inner fold.
- Not testable without training: an empirical leakage probe (for example label permutation). The audit therefore rests on structure plus the exact reproduction of the pre-campaign baseline.

## 5. Soft-vote ensemble and selection on the same folds (question 4)
The members (tangent-space LR, linear SVM, LR) were the top 3 eligible models by mean accuracy on the very folds used for evaluation, so selection bias was a legitimate concern. I rebuilt the soft vote from the stored training-only inner-OOF scores (the saved P5_E1 predictions are reproduced exactly, check 4a) and measured the bias:
| Analysis | Mean accuracy gain vs reference (pts) |
|---|---|
| Chosen vote (as reported) | +0.74 (rank 15 of 35 possible 3-model subsets of the 7 eligible models) |
| Mean over all 35 subsets | +0.57 (range -0.94 to +1.97) |
| Vote of all 7 eligible models | +1.04 |
| Vote of all 11 single models (no outcome-based member choice) | +0.99 |
| Leave-one-seed-out selection (members chosen from the other two seeds) | +1.63, +0.89, +0.44 (mean +0.99) |
| Chosen vote minus the mean of its own three members | +0.41 |
- The chosen set is mid-ranked, not an outlier, and the unselected or out-of-seed-selected ensembles gain as much or more. **So the soft vote's advantage is not an artifact of choosing its members on the evaluation folds; if anything, the +0.74 is a slightly conservative estimate of the ensembling effect (about +0.6 to +1.0 points, plus about +0.013 AUC in every seed).** What would be biased is picking the best subset (+1.97), which was not done.
- Caveats: all variants use the same 676 subjects (a held-out seed is a different partition, not new subjects); eligibility itself uses outer results; the paired-fold SE of the chosen vote is 0.53 (17 wins, 3 ties, 10 losses). None of the unselected variants reaches the frozen +1.5-point floor, and the all-model votes were not declared candidates in the plan, so they cannot change the verdict.

## 6. The 68 consistently misclassified subjects (question 5)
- Recomputed from the predictions: 68 subjects (35 ASD, 33 TC) are misclassified in at least 36 of the 39 runs (>= 90%); the arithmetic in the final report is correct (5a, 5d).
- Qualifications: the 39 runs are not independent. The two ensemble runs are functions of three other runs, and all runs share the same features and folds. Excluding the ensembles (33 runs, >= 30 errors) the count is 70, so the number depends on the inclusion rule. Under independent errors at the observed mean error rate (35.2%) the probability of such persistence is about 1e-13, but strong persistence is expected when all models use the same connectivity features. The statement "misclassified in at least 90% of the 39 runs" is accurate; the sentence implying 39 independent runs is not.

## 7. Frozen rule applied consistently (question 6) and the two reference accuracies (question 7)
- I re-implemented the rule independently and compared it with `selection_rule.py` for all 12 candidates: identical outcomes, none passes (6a). The best gain is +0.74 points (floor +1.5) with 17 fold wins (floor 20). The z statistic (mean paired fold difference over its SE, folds treated as independent) is at most 1.39 (soft vote) among 12 candidates, which is what chance alone produces; the only large effect is the harmful one (partial correlation, z = -4.71) (6c). The reference and the 30 paired folds are the same for every comparison (6b).
- **66.07%** = reading R0, default 0.5 threshold, no threshold selection; per seed 65.83%, 66.27%, 66.12%; identical to the saved Arm B "fixed 0.5" predictions (7a).
- **65.68%** = reading R1, per-fold threshold that maximizes balanced accuracy on training-only inner out-of-fold scores; per seed 64.94%, 65.24%, 66.86% (7b). It is the same nested-threshold reading as the Stage-2 "nested Youden" (the predictions are identical for all 2,028 subject-seed pairs, 7c), because maximizing Youden's J and maximizing balanced accuracy select the same cut-off.
- Both are nested cross-validation (the inner CV selects C); only the threshold procedure differs. The campaign request calls 66.07% "nested cross-validated accuracy", which conflates the readings: the nested-threshold figure is 65.68% (7d). The final report states both correctly.

## 8. Discrepancies, corrections and unresolved questions
**Corrections to the final report**
1. "Even this small gain [soft vote] is optimistic" is not supported by the measurements in section 5; the chosen set is not an outlier and unselected ensembles gain as much. The gain is small and below the floor, not inflated.
2. "39 runs" should not be read as 39 independent runs; the 68 depends on including the two ensembles (70 without them).
3. The log search for failures must be case-sensitive (see section 3).
**Process notes**
4. `phase5_report.md` was overwritten once by the generic report script and the complementarity text was restored as `phase5_complementarity.md` (disclosed earlier; the underlying CSV was intact).
5. Two failures in my first audit run were bugs in the audit script (numpy booleans and a too-tight float32 tolerance), not campaign problems; fixed and rerun.
**Unresolved**
6. No independent test exists; all numbers are development estimates on 676 subjects. The old reused holdout overlaps development, and the protected holdout was never used.
7. Tangent-space inner-CV reference (check 3g) is a documented, minor deviation from strict fold-internal fitting.
8. The primary metric is accuracy but R1 maximizes balanced accuracy (the class split is 46% ASD), which lowers accuracy slightly for the reference (65.68% vs 66.07%); it treats all candidates equally and does not bias the comparison, but absolute R1 accuracies are a bit lower than R0 for the same model.
9. Tree models used an in-fold k = 2000 filter, a small RBF/XGBoost grid, and no stability selection or graphical lasso; the negative results apply to these settings, not to every possible configuration.
10. Whether the consistent AUC gain of ensembles (about +0.013) is real cannot be settled on the same subjects and partitions.

## 9. Recommended next scientifically justified experiment
The only effect with a consistent direction is ensembling: +0.6 to +1.0 points of accuracy and +0.013 AUC, under every selection scheme, but below the frozen +1.5 floor and measured on the same 676 subjects.
**If any further experiment on this cohort is run, it should be a pre-registered confirmation of that single effect on fresh outer partitions:** rerun the already-built soft vote of the three declared members (tangent-space LR, linear SVM, LR) and the reference on seeds 3, 4, 5, with the success criterion declared before running (mean gain >= +0.5 point, positive in at least 2 of 3 seeds, AUC gain >= +0.01), explicitly labeled as an estimate of the ensembling effect and not a rescue of the failed screening. Cost: about 40 minutes (tangent-space is the slow member), no new modelling choices. Outcome expectations: accuracy around 66-67%, nowhere near 85%.
Otherwise, the audit supports stopping modelling on this cohort: the remaining lever is data (an independent external cohort with the same preprocessing, more subjects per site, frame-level motion, harmonized reprocessing), not further selection among models.

## 10. Files created by this audit (new folder only)
`audit.py`, `audit_checks.csv`, `screening_margins_recomputed.csv`, `softvote_all_3subsets.csv`, `softvote_leave_one_seed_out_selection.csv`, `audit_report.md`.
