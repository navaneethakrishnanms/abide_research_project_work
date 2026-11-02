# Limitations and next step

## Limitations
1. **Development-history caveat.** The regularized SSAE settings (S2, S3) were informed by earlier nested-CV experiments on the 780-subject cohort, which overlaps the designated holdout (158 of its 170 subjects). The result is development evidence, not a pristine, independently designed evaluation. The holdout was not accessed to investigate this.
2. **Small evaluation.** 600 subjects, five folds of 120, one selection seed, three SSAE seeds for the final fit only; no significance test; differences of a few points are inside the fold-to-fold SD (0.02-0.04).
3. **Small search.** Three Elastic Net strengths (l1 ratio fixed at 0.5) and three SSAE settings; inner CV scores of the three SSAE settings are within 0.005 AUC in four of five folds, so the selected setting (S2 in all folds) was largely decided by the predeclared tie-break rule.
4. **Unstable feature selection and overfitting.** Elastic Net selected 147-580 features with a mean pairwise Jaccard of 0.18 across folds; the SSAE reaches training loss 0.04-0.08 against 0.18-0.31 on its early-stopping hold-out, and two folds used the full 80 epochs.
5. **Evidence gaps.** No subject-level predictions (AUC cannot be recomputed; one-prediction-per-subject only inferred); `implementation_notes.md` and peak memory not recorded; S1/S2/S3 have no separate outer-test results.
6. **Scope.** Random-fold accuracy does not show generalization to unseen sites; sites differ in scan length, spectral content, age, sex mix and motion; per-edge diagnostic effects are at chance level at the median edge (representation audit).
7. **Comparability.** Results apply only to this cohort and these folds and cannot be compared with the 780-subject phases or the CC200 experiments as if they were equivalent.

## What the evidence supports
Across this experiment and the earlier ones (static classifiers, tangent/partial-correlation variants, site centering, motion adjustment, dynamic features, CPM, Elastic Net + SSAE), development accuracy stays at 61-68% and AUC at 0.65-0.72. The specified architecture does not change that. No result supports an 80% target on this data.

## One recommended next step
Document the frozen logistic-regression baseline (about 64% accuracy, AUC 0.69, five-fold development CV) as the project's development result. If one more experiment is wanted, run a **leave-site-out evaluation of the frozen baseline** (about one minute, same pipeline, groups = the 19 sites, with small sites flagged). It addresses the question random folds cannot answer, generalization to unseen acquisition sites, and avoids further accuracy tuning that the evidence does not support.
