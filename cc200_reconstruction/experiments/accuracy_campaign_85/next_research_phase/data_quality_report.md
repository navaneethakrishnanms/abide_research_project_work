# Data-quality and limitation report (CC200, 676 authorized subjects; development cross-validation only)

Sources: `diagnostics.py`, `diag2.py` (code in this folder), tables `diagnostics_*.csv`, `diagnostics_summary.json`. Phenotypic rows were filtered to the authorized list on load and asserted to match the cache (site, label). The protected holdout was not accessed. These are descriptive diagnostics of this cohort, not tests of the 85% target; where a number comes from a fixed-hyperparameter LR (balanced, C=0.001), that is stated.

## 1. Cohort and missingness
- 676 subjects: 313 ASD / 363 TC; 19 sites (n from 9 to 137; 3 sites have n < 20; NYU alone is 20% of the cohort). ASD fraction per site ranges 0.29 (SDSU) to 0.64 (USM), so site and label are partly confounded in a few sites.
- Motion (mean FD, perc. FD), DVARS, outlier fraction, FWHM, EFC, GSR: complete for all 676. FIQ missing for 46 (6.8%); FIQ is a diagnosis-correlated phenotype (AUC 0.61 for ASD vs TC) and is not a model input here. Raw per-frame FD series were not found among the authorized inputs; only per-subject summaries are available, so scrubbing/censoring at the frame level cannot be tested.
- Age: ASD median 14.7 vs TC 14.9 (p = 0.52). Sex: slightly different (p = 0.04, AUC 0.47). Site-only, age-only, sex-only classifiers are at chance (52.4%, 47.8%, 50.6%; table D3), so these are not a shortcut here.

## 2. Motion (strongest confound found)
- ASD subjects move more: median mean-FD 0.089 vs 0.069 mm (Mann-Whitney p = 2.5e-6; AUC 0.60). Mean FD alone gives 57.9% accuracy / AUC 0.60 (three seeds, 10-fold CV, LR). FD + age + sex: 56.0% (D3). Site + FD + age + sex: 53.6%.
- Subjects over published-style cut-offs (D1 table): mean FD > 0.2 mm: 76 (45 ASD / 31 TC, 17 sites); > 0.5 mm: 10 (7/3); perc. FD > 20%: 129 (74/55); > 50%: 29 (18/11). Flagged subjects are skewed to ASD (59% for FD > 0.2 mm vs 46% in the cohort).
- The first principal component of the connectivity features correlates with FD (Spearman 0.43) and itself separates the classes with AUC 0.62; the reference LR's out-of-fold score correlates with FD at rho = 0.25 (D8).
- The classifier still ranks subjects within motion tertiles: reference AUC 0.706 / 0.721 / 0.704 in low/mid/high-FD tertiles vs 0.723 overall (mean of 6 seeds, D8). Accuracy by FD tertile: 67.2 / 65.5 / 64.8% (reference, 9 seeds).
- Experiments (Stage E): excluding FD > 0.2 mm (E1) lowered accuracy on the retained subjects; regressing FD out of every connectivity feature (E2) lowered accuracy by about 1.5 points and AUC by 0.011-0.016. So part of the usable signal is associated with motion (either partly motion-driven or real signal that covaries with motion; these cannot be separated with summary FD alone). This is a caveat on any claim of neural specificity, not a way to improve accuracy.

## 3. Site and acquisition effects
- Fixed LR, leave-one-site-out pooled accuracy 65.5% vs 66.2% for random 10-fold on the same model (D7): almost no site-generalization loss in accuracy; mean within-site AUC 0.72. In-fold site centering/scaling (E3) did not give a confirmed gain.
- Per-site accuracy of the reference (9 seeds, D2) ranges 44% (SBL, n = 21) to 83% (UM_2, n = 27); largest sites: NYU 65.0%, UM_1 67.8%, USM 80.1%. Per-site standard errors are large (about 10 points at n = 20-30) so most differences between sites are not distinguishable from noise; SBL, MAX_MUN (52%) and LEUVEN_1 (55%) are the weakest and the same sites were weak in the earlier AAL leave-site-out run.
- Under leave-one-site-out, several sites get a badly shifted threshold (e.g. LEUVEN_1 predicted ASD share 14% vs 50% true; KKI 70% vs 37%) while within-site ranking is fair (LEUVEN_1 AUC 0.68): site-specific score offsets exist.

## 4. Age subgroup
Reference accuracy (9 seeds): age <= 12 61.1% (n = 165), 12-18 70.7% (n = 304), > 18 62.5% (n = 207). The 2-member vote: 64.5 / 71.5 / 61.9%. Performance is best for adolescents and weaker in children and adults; cohort composition by age is uneven, and the hard subgroups are the ones that dilute the pooled number.

## 5. Sample size, redundancy, pipeline validity
- Learning curve (fixed LR, 3 seeds; training sizes about 135 / 270 / 405 / 540): accuracy 62.4 / 63.8 / 65.3 / 66.2%, AUC 0.664 / 0.685 / 0.707 / 0.719. Still rising at the full size; roughly +0.02-0.03 AUC and +1.5-2 accuracy points per doubling. A crude log-linear extrapolation would need an implausible multiple of the current data to approach 85% (and curves saturate), so more data of the same kind would help modestly but is not a path to 85% by itself.
- Feature redundancy: 19,900 features, but 107 principal components explain 50% of the standardized variance (top 10: 16.5%; 300 PCs: 76%). The matrix is high-dimensional but strongly redundant; the campaign's PCA / k-best / L1 reductions did not beat the full-feature L2 model (65.6% vs 63.3-65.6% vs 65.7%).
- Label-permutation check (fixed LR, 6 permutations): accuracy 50.3% and AUC 0.498 on average (individual runs 46-53%, AUC 0.44-0.56): the pipeline gives chance on random labels, so the 66% is not a leakage artifact of the evaluation code.
- Representativeness: ABIDE-I sites are convenience samples with heterogeneous protocols, age ranges (median per site 9-34) and diagnoses; the cohort is 676 subjects of one release. Whether it represents the clinical population is not testable here.
- Label quality: diagnoses come from the ABIDE phenotypic file (ADOS/ADI-R-based clinical diagnoses by site); label noise cannot be quantified from the available data.

## 6. Ranked likely limitations (evidence; test)
1. **Limited sample size and heterogeneous sites** (learning curve rising; per-site n small; 19 sites). Test: external cohorts (ABIDE-II or others) with matched preprocessing - not done here (needs authorization/licensing).
2. **Weak signal in static FC of this kind** (AUC 0.72 across all families: LR, SVM, trees, tangent, ensembles all within about 2 points). Test: fundamentally different information (frame-level dynamics, task data), not more tuning.
3. **Motion-related signal and confounding** (FD AUC 0.60; FD-residualization costs 1.5 pt; PC1 ~ FD). Test: frame-level FD and censoring (data not available); matched-FD subsets (previously done on a different cohort).
4. **Site/acquisition heterogeneity** (per-site accuracy 44-83%; threshold shift under LOSO) - but accuracy loss under LOSO is small and harmonization gave no confirmed gain. Lower priority.
5. **Redundant high-dimensional features** - evidence of redundancy exists, but feature reduction did not help in the campaign. Lower priority.
6. **Label noise / diagnostic heterogeneity** - plausible, not measurable here.
