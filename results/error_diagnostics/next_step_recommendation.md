# Next-step recommendation (diagnostics only; nothing was trained)

## Confirmed findings (from saved files and descriptive statistics)
1. **Data are clean.** 600 unique subjects, labels match diagnosis, every candidate's out-of-fold table matches the manifest in subjects, labels and folds.
2. **Inner and outer scores agree.** Inner-CV AUC of the selected setting is within about +/-0.017 of outer AUC for all 8 candidates, so there is no sign of leakage or selection overfitting. The weak accuracy is the real out-of-sample level of these features.
3. **Errors are shared across models.** 78 subjects are misclassified by all 8 candidate runs (about 0.2 expected if errors were independent) and 211 by none. The models fail on largely the same subjects, which is why swapping classifier, atlas or centering changes nothing.
4. **Hard subjects skew ASD.** Subjects wrong in at least 7 of 8 runs are 52% ASD; subjects right in at most 1 of 8 are 38% ASD (overall 45%). Sensitivity (54-66%) is below specificity (63-71%) for most candidates.
5. **Motion differs by diagnosis even after the FD <= 0.2 mm filter.** ASD mean FD 0.09 vs TC 0.07 (Mann-Whitney p = 5e-5). Mean FD alone has an AUC-equivalent of **0.596** for ASD vs TC (a descriptive rank statistic on all 600 subjects, not cross-validated), against model out-of-fold AUCs of 0.65-0.70.
6. **Model scores track motion.** Out-of-fold scores correlate with FD (Spearman 0.17-0.28 overall), and also *within* each diagnosis group (0.12-0.27), so the correlation is not just the class difference.
7. **Performance varies by site, consistently across atlases.** Per-site accuracy ranges roughly 46%-79% (sites with n >= 20: Spearman rho between AAL-116 and CC200 = 0.87). ASD prevalence does not differ significantly across sites (p = 0.62), and a prevalence-only site rule scores 52.7%, so site is not a direct label shortcut.
8. **Features are not redundant, and the problem is p >> n.** Fewer than 0.01% of random feature pairs have |r| > 0.8; 80% of variance needs 262 (AAL-116) or 311 (CC200) components for 600 subjects. Phase 6 Elastic Net selected 1,286 features at least once but only 50 in all five folds, i.e. selection is unstable. About 19-30% of features have some imputed value.
9. **Sex is imbalanced by diagnosis** (ASD 14% female vs TC 21%; p = 0.034).

## Hypotheses (not established)
- **H1 (motion):** part of the 64-65% accuracy may reflect motion-related connectivity differences rather than diagnosis. Support: findings 5 and 6. Against: hard-vs-easy subjects do not differ in FD (p = 0.43). Correlation of scores with FD does not show the models *use* motion.
- H2: the shared-hard-subject structure reflects label/phenotype heterogeneity or weak signal in static FC, not a classifier limit. Not testable with the current artifacts.
- H3: site-specific acquisition differences matter, but simple mean centering did not help (previous experiment).

## Holdout independence: unverified
The holdout file was not opened, but 180 of the 780 cohort subjects are not in the train list, and the holdout has 170 subjects; Phases 6-10 used all 780. I cannot rule out that earlier phases (and therefore design choices) used holdout subjects, and I did not read the IDs to check. Treat the holdout as **not demonstrably independent** until someone with the authority to read the ID lists confirms the overlap with the 780 cohort.

## Single recommended experiment (not launched)
**Fold-safe motion adjustment, with a motion-only control.** Same 600 subjects, same outer/inner folds, same pipelines. Inside every training split, fit a linear regression of each FC feature on mean FD (training rows only), subtract the fitted FD effect from training and test rows, then run the three existing candidates (AAL-116 logistic regression, AAL-116 linear SVM, CC200 RBF SVM) and compare paired with their unadjusted results. Add one control: cross-validated logistic regression on mean FD alone (one feature) to put a number on how much diagnosis information is in motion.
- **What would be learned:** if adjusted accuracy falls toward the FD-only control level, much of the current signal is motion-related; if it is unchanged, H1 is not supported.
- **Estimated runtime:** about 2-3 minutes (the previous paired experiment took 56 s).
- **Caution:** the outcome cannot reach 85%; its purpose is to explain the 64-65% ceiling, not to raise it. Selecting the better of adjusted/unadjusted would be exploratory.
