# Feature and data-quality audit -- AAL-116 FC features, 600 development subjects

Read-only; no model trained; holdout, `RESTRICTED_id_membership.csv` and CC200 not touched. Labels were used only for the descriptive QC-confounding associations in section 5. Facts are separated from hypotheses throughout. Tables: `feature_quality_summary.csv`, `site_quality_summary.csv`, `qc_metadata_summary.csv`, `qc_rater2_flag_by_site.csv`, `flagged_features.csv`. Baseline context (from the earlier reports): frozen L2 logistic regression 64.2% five-fold and 62.2% leave-site-out development accuracy, AUC 0.69; no later experiment improved reliably.

## 1. Dimensions and ordering -- OBSERVED, no problem
- Cached matrix is 600 x 6,670 (= 116 x 115 / 2), float64; the feature order equals the Phase 5A feature index for all 6,670 rows (upper triangle, i < j, 1-indexed).
- AAL time-series files exist for 600/600 development subjects, each with 116 ROIs. Recomputing correlations from the raw files for 42 subjects (40 random, first, last) reproduces the cache with a maximum absolute difference of 6.7e-16 and an identical missing-value pattern, so the cache is aligned with the manifest and the feature index.

## 2. Feature values -- OBSERVED
| Check | Result |
|---|---|
| Non-finite | 0 infinite; 12,764 NaN entries (0.32%), in 2,014 features and 40 subjects (up to 1,314 in one subject; only 4 features are NaN in > 5% of subjects). Cause: constant (zero) ROIs (Phase 4.1). Handled by training-fold mean imputation. |
| Range | r from -0.859 to 0.969; no |r| >= 0.99, none equal to 1, so the 1e-6 clip never binds; Fisher-Z from -1.29 to 2.08. |
| Constant / near-constant | 0 features with SD = 0 or Fisher-Z SD < 0.05 (median SD 0.193). |
| Duplicated | 0 exactly duplicated columns; among 3.1 million random feature pairs 0.0013% have |r| > 0.8 and none > 0.95. |
| Extreme values | 59 entries (0.001%) with robust |z| > 6 in 57 features; no subject has more than 0.61% of features extreme. Distributions are close to symmetric (no feature with |skew| > 1; median 0.10). |
There is no defect in the feature matrix itself beyond the documented undefined correlations.

## 3. Subjects and sites -- OBSERVED
- 268 ASD / 332 TC in 19 sites (6 sites have n < 20; CMU has only 2 subjects). No subject has a mean correlation outside +/-0.05 (consistent with global-signal-regressed data).
- Site shift of the mean FC vector is on average 1.6 times what would be expected with no site effect (range 0.9-2.1; `site_quality_summary.csv`): present but moderate. Scan length (116-316 TRs), spectral content, age, sex mix and motion also differ by site (representation audit).
- Subjects with undefined edges are concentrated at SBL (14 of 18), PITT (11 of 33), NYU (4), SDSU (4) and CALTECH (3).

## 4. Metadata availability -- OBSERVED
Available for all 600 subjects: mean FD, number and percent of frames over the FD threshold, DVARS, outlier fraction, func_quality, EFC, FBER, GSR, FWHM, and three rater columns. **Not available:** per-frame motion, TR, scan-level QC files. Rater 1 and rater 3 are 'OK' for all 600 subjects (no information). Rater 2 gives 'OK' for 474 and 'maybe' for 126 subjects.

## 5. Confounding of QC metadata -- OBSERVED (descriptive; ten variables, no multiplicity correction)
- **Motion:** mean FD, frames over threshold and percent FD differ by diagnosis (AUC-equivalent 0.59-0.60, p = 1e-4), and the association persists within sites (about 0.62). All three are also site-dependent (epsilon^2 0.10-0.17).
- **Scanner/acquisition metrics** (DVARS, quality, EFC, FBER, GSR, FWHM): no overall association with diagnosis (AUC-equivalent 0.50-0.53, p 0.17-0.95) but very strongly site-dependent (epsilon^2 0.53-0.82). Within sites, FWHM (0.62) and outlier fraction (0.61) lean toward ASD; these are exploratory and uncorrected.
- **Rater-2 'maybe' flag:** equal by diagnosis (ASD 20.1%, TC 21.7%, p = 0.72) but extremely site-specific: SBL 94%, MAX_MUN 87%, CALTECH 83%, PITT 76%, LEUVEN_2 71%, UCLA_2 50%, against 0-11% at most other sites. Among flagged subjects the notes are mostly 'ic-cerebellum' (some with temporal lobe or parietal). 27 of the 40 subjects with undefined edges are flagged (Fisher p = 5e-11), but only 27 of the 126 flagged subjects have undefined edges. 
- Of the 19 constant ROIs found in Phase 4.1, 11 are cerebellar or vermis ROIs (indices 101-108 and 114-116); the other 8 are orbitofrontal, rectus or temporal-pole ROIs. Cerebellar and vermis ROIs (indices 91-116) are 26 of 116 ROIs but take part in 2,665 of 6,670 edges (40%).

## 6. Pipeline provenance and leakage -- OBSERVED
- Documented pipeline (PCP CPAC derivative, band-pass filtered and global-signal regressed -> AAL ROI series -> Pearson correlation -> Fisher-Z -> 6,670 edges) matches the code: `load_fc` computes `np.corrcoef` on the raw series and takes the upper triangle (verified numerically above); Fisher-Z (clip 1e-6), mean imputation, scaling and SelectKBest are Pipeline steps.
- Every `.fit(` call in the development-cohort scripts (fast, site-centering, motion, dynamic, CPM, SSAE, leave-site-out, accuracy-improvement) receives training-index data or a Pipeline fitted on them; the cached matrix contains no learned statistic. The earlier Phase 9 selection leak (Elastic Net fitted on validation rows) belongs to the 780-cohort phases, not to these scripts. This audit and earlier descriptive audits used global statistics only for description.
- Design-history caveat (unchanged): the development subjects overlap the 780-subject cohort used for earlier design decisions.

## 7. Sufficiency to test another representation without leakage -- OBSERVED
Yes for any representation computed per subject from the AAL series (600/600 files, aligned IDs, no NaN in the series) or from a fixed, predeclared ROI subset; any group-level learned transform (harmonization, PCA, feature selection) must be fitted inside training folds. Limits: no TR or per-frame FD, so frequency-based and scrubbing-based representations cannot be built.

## 8. Hypotheses (not established)
- H1: cerebellar/vermis signal is less reliable at several sites (coverage problems flagged by a rater, constant ROIs concentrated there), adding site-linked noise to 40% of the features.
- H2: part of the motion association with diagnosis survives within-site comparison, so motion-related variance cannot be fully separated from diagnosis with mean FD alone.
- H3: the weak signal itself, not a data defect, limits accuracy (consistent with the null median edge effect found earlier).

## 9. Summary
The feature matrix is clean, correctly ordered and consistent with its documented pipeline, and all fitting is inside training folds. The main data-quality finding is site-specific partial coverage or quality problems in the cerebellum, flagged by a rater and by constant ROIs, which affect many more subjects than the undefined correlations alone reveal.
