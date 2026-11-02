# Experiment protocol (written before any arm was run; not to be changed after seeing results)

## Scientific justification (stated plainly)
The older 676-subject CC200 results (about 67%) used age and sex as model inputs and a different protocol, so it is unknown how much of them came from demographics. This experiment measures that on the primary FC-only protocol: it is first a **control to interpret the earlier numbers**, and only second a possible improvement. A gain from age and sex would not by itself be a neuroimaging improvement: sex is imbalanced by diagnosis (13.8% female in ASD vs 20.8% in TC, p = 0.034; within-site Mantel-Haenszel odds ratio 0.67) but also extremely uneven across sites (4 sites have no female subjects; p = 7e-8), and age is strongly site-structured (48% of its variance is between sites), so demographic "signal" may just encode cohort composition. Demographics add a prediction-time-legitimate but non-neural predictor; any result must be read with that in mind.

## Data (verified in `audit_summary.md`)
The 600 development subjects, labels, sites and the five outer folds (StratifiedKFold seed 42) of the baseline. Age (years) and sex (1 = male, 2 = female -> binary female) are complete for all 600 subjects, identical in the cohort file and the phenotype file, with no diagnosis-derived or post-diagnosis variables. FC features: AAL-116 Pearson (6,670 edges), cached development matrix. Restricted holdout, `RESTRICTED_id_membership.csv`, CC200 and the 780 cohort are not used.

## Arms (identical folds; inner-CV seeds 42, 123, 2026)
- **A. Baseline:** the frozen FC-only pipeline, read from `accuracy_improvement_v1` control E0 (seed-42 predictions identical to the frozen baseline).
- **B. Age + sex only:** age (training-standardized) and female (training-standardized), L2 logistic regression (balanced), C in {0.01, 0.1, 1, 10} by inner stratified 3-fold ROC-AUC.
- **B2. Diagnostic control (not part of the success rule):** B plus site one-hot, same selection; estimates how much cohort composition (site, age, sex) predicts without imaging. Site is unavailable for a truly new site.
- **C. FC + age + sex, no tuning:** FC edges -> training-fold mean imputation, Fisher-Z, StandardScaler, SelectKBest(k = 2000) -> concatenated with the two training-standardized demographic columns -> L2 logistic regression C = 1e-3 (the baseline's modal setting; fixed in advance, demographic weight 1). Deterministic, so identical across seeds.
- **D. FC + age + sex, tuned:** as C, with C in {1e-4, 1e-3, 1e-2, 1e-1}, k in {500, 2000, all} and a demographic column weight w in {1, 10} (a weight of 10 lowers the effective penalty on age and sex relative to the thousands of FC edges) selected by inner stratified 3-fold ROC-AUC on training rows.
Decision threshold 0.5 for all arms; nothing is selected from outer-fold scores.

## Success threshold for C and D versus A (all required, per arm)
1. Seed-averaged accuracy and balanced accuracy each >= +0.02 over A.
2. At least 2 of 3 seeds individually >= +0.02 on both, with accuracy higher in >= 4 of 5 folds.
3. No material degradation: seed-averaged AUC >= A - 0.01; sensitivity and specificity each >= A - 0.03.
4. No integrity failure.
Otherwise: "no improvement established". Arm B and B2 are reported only against the class-prevalence level and against each other. No significance test is claimed.

## Confounding diagnostics (reported, not criteria)
Pooled AUC versus mean within-site AUC for each arm (a pooled AUC above the within-site AUC indicates between-site/composition information); per-site accuracy of every arm and the change of C and D relative to A by site; accuracy of B against the 55.3% majority-class rate.

## Experiment 2 (saved predictions only, descriptive)
For the baseline and tangent out-of-fold predictions: errors by site, age tercile, sex, mean-FD tercile and class; within-site discrimination versus between-site score distributions. No variable is invented; no subjects are removed; evaluation rules are not changed.

## Stopping rule
One smoke test (seed 42, fold 0), then one full run, then stop. If neither C nor D meets the threshold, the conclusion is that demographics do not establish an improvement; no further modeling on this cohort is run.
