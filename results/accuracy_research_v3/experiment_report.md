# Demographic covariates and generalization diagnosis (accuracy_research_v3)

**Result: no defensible improvement.** Adding age and sex to the FC model did not improve it: neither arm C nor arm D met the predeclared threshold, and age and sex alone carry essentially no predictive information. 85% was not approached (best arm: 65.3% seed-averaged vs 64.7% for the baseline). The experiment failed its purpose, so per the brief **modeling on this cohort stops here.** All results are development cross-validation estimates on 600 subjects, not independent-holdout performance. The restricted holdout, its ID file, CC200 and the 780 cohort were not used; 0 of 311 protected files changed.

## Experiment 1: controlled comparison (protocol: `experiment_protocol.md`, written before the run)
Same 600 subjects, five outer folds, inner-CV seeds 42/123/2026; all preprocessing, selection and tuning inside training folds. Smoke test (seed 42, fold 0) passed before the full run; a perturbation check confirmed test rows do not influence one another or the fit.

| Arm | Seed | Accuracy | Bal. acc. | Sens. | Spec. | F1 | AUC |
|---|---|---|---|---|---|---|---|
| A. FC-only baseline | 42 / 123 / 2026 | 0.642 / 0.647 / 0.653 | 0.638 / 0.643 / 0.649 | 0.604 / 0.608 / 0.615 | 0.672 / 0.678 / 0.683 | 0.601 / 0.606 / 0.613 | 0.694 / 0.694 / 0.692 |
| B. Age + sex only | all | 0.497 | 0.526 | 0.799 | 0.253 | 0.586 | 0.528 / 0.531 / 0.528 |
| B2. Age + sex + site (diagnostic) | 42 / 123 / 2026 | 0.500 / 0.498 / 0.502 | 0.502 / 0.500 / 0.503 | 0.512 / 0.508 / 0.516 | 0.491 | 0.473 / 0.470 / 0.475 | 0.503 / 0.502 / 0.502 |
| C. FC + age + sex, fixed (C = 1e-3, k = 2000, w = 1) | identical across seeds | 0.650 | 0.646 | 0.615 | 0.678 | 0.610 | 0.695 |
| D. FC + age + sex, tuned (C, k, demographic weight by inner CV) | 42 / 123 / 2026 | 0.653 / 0.647 / 0.660 | 0.651 / 0.643 / 0.656 | 0.630 / 0.608 / 0.623 | 0.672 / 0.678 / 0.689 | 0.619 / 0.603 / 0.621 | 0.701 / 0.694 / 0.692 |

Fold SDs are in `metrics_by_seed.csv` (0.03-0.05); fold-level and per-seed metrics, confusion counts and anonymized out-of-fold predictions (one per subject per arm and seed, 600 unique IDs each) are saved.

### Predeclared threshold (C and D versus A)
| | Seed-avg accuracy / balanced accuracy change | Seeds meeting +0.02 with >= 4/5 folds | AUC / sens. / spec. change | Result |
|---|---|---|---|---|
| C | +0.003 / +0.003 | 0 of 3 (accuracy higher in 1 of 5 folds each seed) | +0.001 / +0.006 / +0.000 | **Not an improvement** |
| D | +0.006 / +0.007 | 0 of 3 (higher in 3, 3 and 1 of 5 folds) | +0.002 / +0.011 / +0.002 | **Not an improvement** |

The inner CV chose the up-weighted demographic setting (w = 10) in 12 of 15 fits and still gained nothing measurable, so the lack of gain is not an artifact of over-penalizing the two columns.

### Do demographics help, exploit composition, or vary by site?
- **They carry almost no information in this cohort:** age + sex alone gives accuracy 49.7% (below the 55.3% majority-class rate), balanced accuracy 0.526 and AUC 0.53 (it predicts ASD for 77% of subjects, sensitivity 0.80, specificity 0.25). That matches the small sex effect (female share 13.8% vs 20.8%) and the null age effect.
- **Composition control:** adding site one-hot (B2) gives chance (accuracy 0.50, AUC 0.50). Its within-site AUC is 0.36, i.e. the between-site age/sex pattern points the wrong way inside sites, which shows that the apparent age/sex structure is a cohort-composition effect rather than a within-site diagnostic signal. (Within-site AUC for B is 0.46-0.48 vs 0.52 pooled.)
- **FC arms:** pooled vs mean within-site AUC is 0.692 vs 0.665 for A and 0.692 vs 0.665 for C; D is 0.699/0.671 (seed 42), so adding demographics does not change within-site discrimination.
- **By site:** D minus A accuracy ranges from -0.045 (LEUVEN_1) to +0.067 (SDSU) among the 13 sites with n >= 20 (positive at 8, zero at 1, negative at 4), with intervals of +/-0.10 to 0.20 per site; no consistent site pattern (`per_site_accuracy_by_arm.csv`).

## Experiment 2: generalization diagnosis from saved predictions (descriptive; seed 42; `generalization_diagnosis.md`)
- **Motion:** scores of the baseline correlate with mean FD (Spearman +0.20, +0.21 within sites; tangent +0.23/+0.24). Sensitivity depends strongly on motion: ASD subjects in the low-FD tercile are classified correctly 47% of the time (tangent 38%), in the high-FD tercile 71%; the pattern reverses for controls (low FD 74%, high FD 61%). The model's apparent ASD signal is partly a motion-related signal (FD alone gives AUC 0.60).
- **Age:** scores correlate negatively with age (baseline -0.16 overall, -0.13 within sites, -0.26 among controls; tangent -0.15, -0.15, -0.20). Controls in the youngest tercile are correct only 51% of the time against 78% in the oldest, although age does not differ by diagnosis. Errors also decline with age (Spearman -0.10, p = 0.017, baseline). Adding age as a model input (arms C and D) did not remove this, because the FC features themselves still carry the age-related variance.
- **Sex and class imbalance:** errors do not depend on sex (chi-square p = 0.22); sensitivity 0.60 < specificity 0.67; predicted ASD share 45% (true 45%).
- **Within-site vs between-site:** mean within-site AUC over the 13 larger sites is 0.665 (range 0.48-0.85) against 0.692 pooled; site explains 7.9% of the score variance against 11.3% for true class. The site mean score does not significantly track the site's true ASD share (Spearman 0.16, p = 0.59; tangent 0.45, p = 0.13), nor site mean FD or age. Large site-level mismatches between predicted and true ASD share remain (LEUVEN_1 18% predicted vs 50% true; KKI 48% vs 26%).
- Nothing in this analysis removes subjects, changes the evaluation or invents variables; associations are descriptive and not causal.

## Interpretation
Age and sex do not rescue the FC model: they are almost uninformative about diagnosis in this cohort and mostly encode site composition. The diagnostic work shows what limits the FC classifier: its scores carry age- and motion-related variance that does not separate the diagnostic groups, on top of a weak diagnostic signal. That is a hypothesis for future work, not a tested remedy; a covariate in the final classifier does not remove nuisance variance inside the features, and removing it (training-fold residualization of age, in addition to the already-tested motion residualization) has not been evaluated. Per the brief it was not run.

## Limitations
Five folds of 120 subjects; no significance tests; one FC representation (the frozen baseline); demographics limited to age and sex; the development subjects overlap the cohort behind earlier design decisions; per-site estimates are noisy; random-fold results do not show generalization to unseen sites.

## Final decision
Baseline 64.7% seed-averaged (AUC 0.694); best new arm D 65.3% (+0.6 points, AUC 0.696), not consistent across folds or seeds; 85% not achieved. **Stop modeling on this cohort.** What is needed to make real progress: (1) an independent external cohort with the same preprocessing for a clean held-out test (the current holdout overlaps earlier development); (2) more subjects and more subjects per site; (3) the TR and per-frame motion (to censor frames and control motion at the frame level); (4) reprocessing from raw images in one harmonized pipeline (or a harmonization method fitted on a larger multi-site sample); (5) severity or subtype information (ADOS/ADI-R completeness would first need checking) to address heterogeneity; (6) a preregistered evaluation protocol with a single frozen model.
