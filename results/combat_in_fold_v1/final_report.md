# combat_in_fold_v1 -- final report

**Result: the predeclared acceptance rule was NOT met (0 of 3 seeds). In-fold ComBat-style harmonization gave no accuracy improvement.** Seed-averaged accuracy
was 64.8% against 64.7% for the matching baseline (E0), a difference of +0.1 points; 85% is not approached. All numbers are development cross-validation
estimates on the 600 development subjects, not independent-holdout performance. The restricted holdout, its ID file, CC200 and the 780 cohort were not used;
0 of the 311 protected baseline files changed (checked before and after).

## Protocol as run (`protocol.md`, written before any code ran; nothing changed afterwards)
Same cohort, outer folds, inner seeds (42/123/2026), downstream pipeline and grid as `accuracy_improvement_v1` control E0. One change: a label-free,
ComBat-style (parametric empirical-Bayes location and scale) site harmonizer with age and sex covariates replaces the Fisher-Z step; all its parameters
are fitted on the rows of each (inner or outer) training split only. Diagnosis labels are not used in the harmonizer. Sites with fewer than 5 training
subjects are pooled; unseen sites are left unchanged.

Checks before the full run: synthetic self-test passed (held-out between-site variance share 0.752 -> 0.049; held-out rows cannot change the fit or each
other; unseen site left unchanged); labels, sites, age and sex asserted aligned with the manifest and the phenotype file; outer folds identical to E0;
smoke run (seed 42, fold 0) completed before the full run.

## Results (mean over 5 outer folds; E0 read from `accuracy_improvement_v1/fold_metrics.csv`)
| Seed | Acc. E0 -> H | Bal. acc. E0 -> H | Sens. E0 -> H | Spec. E0 -> H | AUC E0 -> H | Folds with higher accuracy | Rule met |
|---|---|---|---|---|---|---|---|
| 42 | 0.642 -> 0.643 (+0.002) | 0.638 -> 0.641 (+0.004) | 0.604 -> 0.623 | 0.672 -> 0.660 | 0.694 -> 0.711 (+0.017) | 2/5 | No |
| 123 | 0.647 -> 0.650 (+0.003) | 0.643 -> 0.647 (+0.004) | 0.608 -> 0.615 | 0.678 -> 0.678 | 0.694 -> 0.707 (+0.013) | 3/5 | No |
| 2026 | 0.653 -> 0.652 (-0.002) | 0.649 -> 0.649 (-0.001) | 0.615 -> 0.623 | 0.683 -> 0.675 | 0.692 -> 0.708 (+0.016) | 2/5 | No |
| Mean | 0.647 -> 0.648 | 0.643 -> 0.646 | | | 0.694 -> 0.709 | | 0/3 seeds |

Required for "improves": accuracy and balanced accuracy each +0.02 or more, accuracy higher in at least 4/5 folds, in at least 2 of 3 seeds. The best
observed difference was +0.003 in accuracy. Per-fold accuracy of H ranges 0.608-0.708 (SD of 3-5 points), so these differences are noise.

Fold-level and per-seed values: `fold_metrics.csv`, `comparison_vs_E0.csv`; anonymized out-of-fold predictions (600 unique subjects per seed):
`oof_predictions.csv`; rule and run configuration: `config.json`.

## Interpretation (what the data show, and what they do not)
- Harmonization did what it is designed to do on the features: the mean per-edge between-site variance share on training rows fell from 0.087-0.089 to
  0.0125-0.0134 (`site_variance_diagnostic.csv`, all five folds). It did not translate into more correct decisions at the 0.5 threshold.
- AUC rose by 0.013-0.017 in every seed (0.694 -> 0.709 on average; higher in 3-4 of 5 folds per seed). This is a small ranking improvement that the
  acceptance rule (accuracy based) does not count and that cannot be called a result here: five folds, no significance test, AUC was not the predeclared
  criterion, and the accuracy did not follow. It is a lead worth noting, not a finding.
- Because the harmonizer could not use diagnosis labels (project rule), site terms may have absorbed part of the class signal at sites with unequal ASD
  share (26%-60%). That would work against a gain; it is a hypothesis, not tested.
- 85% is not supported: the best accuracy under this protocol is 65.2% (seed 2026 mean over folds), and the best single fold 70.8%.

## Runtime, warnings
Full run 93 s (15 outer fits at 6-8 s each, workers n_jobs=6, single-threaded numerics). Warnings in the parent process: 1 (`ResourceWarning`);
worker warnings are not captured. No errors; no background processes left.

## Limitations
Five folds of 120 subjects; no significance tests; random (not site-grouped) outer folds, so unseen-site generalization is not assessed; one harmonization
variant (parametric, age + sex, no diagnosis); development subjects overlap the cohort behind earlier design decisions.

## Conclusion
Rule not met, so by the predeclared criterion in-fold ComBat does not improve this model. Recommendation: do not pursue further harmonization variants on
this cohort. The remaining routes are those stated before (an external cohort with the same preprocessing, more subjects per site, frame-level motion
information, reprocessing from raw data in one pipeline).
