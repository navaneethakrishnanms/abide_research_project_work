# Representation audit -- AAL-116, 600 development subjects (no model trained)

**Reference result (unchanged):** frozen static AAL-116 L2 logistic regression, five-fold development cross-validation: accuracy **64.2% +/- 4.3**, AUC **0.694 +/- 0.037**. This is a cross-validation estimate on the development cohort, not an independent-holdout result. Descriptive statistics below use only the 600 development subjects; none was used to select a model.

## 1. Preprocessing (what is verified and where it happens)
Full table: `preprocessing_inventory.csv`.
- **Verified from scripts/artifacts:** the ROI series come from the ABIDE PCP CPAC derivative requested with band-pass filtering and global signal regression (`fetch_abide.py`; Phase 3.6 download summary: strategy `filt_global`; nilearn 0.14 docs: band filtered 0.01-0.1 Hz). Our own code adds only: Pearson correlation of the raw series, training-fold mean imputation, Fisher-Z (clip 1e-6) and a training-fold StandardScaler. There is no frame scrubbing, no scan-length or TR harmonization, no per-ROI standardization of the series (Pearson is invariant to it), and no site harmonization in the baseline. Subjects with mean FD > 0.2 mm were removed (Phase 2).
- **Consistent with the data (not independently verified):** global signal regression (median mean edge correlation 0.002; ROI-mean signal SD about 13% of ROI SD), band-pass filtering (power above half the Nyquist frequency about 0.08% in most sites) and detrending (linear-trend variance share 0.0003).
- **Not verifiable locally:** slice timing, motion realignment, which nuisance regressors were used. The TR is not stored in any local file.

## 2. Cohort variation
Table: `cohort_characteristics.csv` (small sites, n < 20, are flagged).
- **Scan length differs sharply by site:** 116 to 316 TRs (median 176); e.g. UCLA_1, UCLA_2 and MAX_MUN 116, NYU 176, UM_1/UM_2 296. It is not related to diagnosis (mean 196 TRs for both groups).
- **Spectral content differs by site:** the T = 116 sites (UCLA_1, UCLA_2, part of MAX_MUN) have lag-1 autocorrelation about 0.54 and 12% of power above half Nyquist, against 0.75-0.87 and 0.1% elsewhere (KKI intermediate at 0.65). This is what a Hz-defined filter would produce at a longer TR, but the TR is not available here, so that is a hypothesis.
- **Demographics differ by site:** mean age from 10.0 (STANFORD) to 34.3 (SBL); female share 0% at USM, TRINITY, LEUVEN_1 and SBL up to 44% at YALE; ASD share 26% (KKI) to 60% (USM). Overall: 268 ASD / 332 TC, age 17.1, 17.7% female.
- **Motion:** mean FD 0.081 overall (0.060-0.138 by site), higher in ASD (0.089 vs 0.075). DVARS is identical by diagnosis (1.097 vs 1.095); the outlier fraction is slightly higher in ASD (0.0045 vs 0.0038). Quality metrics (DVARS, outlier fraction, percent FD, FBER, EFC) exist for all 600 subjects and have never been used.

## 3. Where the variance in the FC edges lies (`variance_sources_per_edge.csv`)
| Source | Median edge variance share | Expected if no effect | Ratio |
|---|---|---|---|
| Site (19 sites) | 0.073 | 0.030 | 2.4x |
| Mean FD | 0.0042 | 0.0017 | 2.5x |
| Scan length | 0.0042 | 0.0017 | 2.5x |
| Age | 0.0027 | 0.0017 | 1.6x |
| Diagnosis | **0.0016** | 0.0017 | **0.98x** |
| Sex | 0.0013 | 0.0017 | 0.8x |

- The typical edge carries no diagnosis information beyond chance level, while site, motion and scan length each explain more. The tail is different: 17.9% of edges have a nominal diagnosis p < 0.05 (5% expected; the 95th-percentile edge explains 1.3%). Part of this tail is probably site and motion confounding (both are associated with diagnosis here), so it is not evidence of diagnostic signal on its own. This uses diagnosis labels over all 600 development subjects, as a description only.
- **Edge reliability (split-half, contiguous halves, Spearman-Brown corrected to full length):** median 0.53; 0.51 after removing site means; by scan-length tercile 0.51 (short), 0.53 (mid), 0.57 (long). Roughly half of the variance of a typical edge is noise even at full scan length, and longer scans help only a little. Reliability across subjects includes stable site differences, so the within-site figure is the relevant one.

## 4. Information loss and inconsistencies
- 0.32% of edge values are undefined (constant ROIs in 40 subjects, mostly SBL and PITT); handled by fold-scoped imputation. These subjects are not misclassified more often (earlier diagnostics).
- Series are used at native length; sites with only 116 TRs give noisier correlations (reliability rises modestly with length). No data are discarded by our code beyond the FD filter.
- Not an error: the filter/length differences between sites reflect acquisition, not a processing inconsistency inside the project.

## 5. Representations already tested
See `tested_methods_inventory.csv`. Static correlation with L1/L2 logistic regression, linear and RBF SVM (AAL-116 and CC200), Elastic Net + SSAE (780 cohort, overlapping the holdout), partial correlation, tangent-space and raw covariance (CC200, 676 subjects; tangent equalled correlation), training-fold site centering, mean-FD residualization, dynamic variance and change features, CPM. Everything lands at 61-68% accuracy and AUC 0.65-0.72, except dynamic features, which are at chance.

## 6. Untested representations
Ledoit-Wolf/tangent on AAL-116, anatomically grouped block-averaged FC, graph node metrics, truncation to a common length, ALFF-type amplitude features (needs TR). None has evidence behind it from this audit.

## 7. Evidence versus hypotheses
**Observed:** the baseline and its preprocessing are as described; sites differ strongly in scan length, spectral content, age, sex mix and motion; the median edge's diagnosis effect is indistinguishable from zero while site and motion effects are larger; edge reliability is about 0.5; every tested representation gives the same performance level.
**Hypotheses (not tested):** that the 64% ceiling reflects weak diagnostic signal in static FC combined with moderate measurement reliability and site/motion variance; that the spectral differences reflect TR; that aggregating edges would raise signal-to-noise. No claim of independent-holdout performance, generalization to unseen sites, 85% or 98% is made.
