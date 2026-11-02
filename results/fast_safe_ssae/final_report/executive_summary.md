# Executive summary -- Elastic Net + SSAE (500-100-2), AAL-116, 600 development subjects

**Status: FINAL.** Task `bjep10vq8` completed normally in 645 s (budget 900 s): 5 of 5 outer folds and 3 of 3 seeds, no failures.

| Model | Accuracy | AUC |
|---|---|---|
| Frozen baseline (L2 logistic regression) | 0.642 +/- 0.043 | 0.694 +/- 0.037 |
| Elastic Net classifier (no SSAE) | 0.620 +/- 0.039 | 0.673 +/- 0.029 |
| EN + SSAE, seed 42 (predeclared headline) | 0.628 +/- 0.022 | 0.678 +/- 0.026 |
| EN + SSAE, seeds 123 / 2026 | 0.632 / 0.625 | 0.679 / 0.678 |

- **SSAE did not beat the frozen baseline:** accuracy -0.013 +/- 0.045 (better in 1 of 5 folds), AUC -0.017 (2 of 5), balanced accuracy -0.017. It was only marginally better than its own Elastic Net classifier (+0.008 accuracy, 3 of 5 folds), inside the fold variation. Results are stable across seeds (under 0.01 accuracy difference).
- **80% was not reached.** Best development cross-validation accuracy is 63.2%; best single fold 70.0% (the baseline). This is a five-fold development estimate, not independent-holdout performance.
- **Integrity:** all checkpoint confusion matrices match the manifest (fold size 120, ASD counts), metrics recompute exactly, perturbation check passed, 0 of 311 protected files changed, holdout/restricted file/CC200 untouched. **Not verifiable:** subject-level predictions were not saved, so AUC cannot be recomputed from confusion matrices; `implementation_notes.md` was not produced; the three SSAE settings have no separate outer-test results (inner-validation AUC 0.645 for each; S2 was chosen in all folds, mostly by the tie-break rule).
- **Caveat:** the regularized SSAE settings were informed by earlier results on a cohort overlapping the designated holdout, so this is development evidence, not a pristine evaluation.
- **Next step:** document the baseline; if one more test is wanted, a leave-site-out evaluation of the frozen baseline (about a minute).
