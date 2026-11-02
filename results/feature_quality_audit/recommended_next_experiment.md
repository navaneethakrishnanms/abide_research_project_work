# Recommended next experiment (at most one; not launched)

## Candidate: cerebrum-only AAL representation (drop ROIs 91-116)
Use a fixed, predeclared ROI subset: the 90 cerebral AAL ROIs (indices 1-90), giving 4,005 edges instead of 6,670, and drop the 26 cerebellar and vermis ROIs (indices 91-116) that account for 2,665 edges. The ROI subset has no learned parameters, so it can be applied identically in every fold. Run the frozen baseline pipeline (imputation, Fisher-Z, scaler, SelectKBest, L2 logistic regression with the same grid and inner CV) on the same 600 subjects and five outer folds (inner seeds 42, 123, 2026), paired with the baseline for each seed, plus one leave-site-out pass as a sensitivity check that is not used to choose anything.

### Why this may help (hypotheses, from the audit)
- A rater flagged 126 subjects (21%) as 'maybe' with mostly cerebellar notes; the flag is concentrated at six sites (SBL, MAX_MUN, CALTECH, PITT, LEUVEN_2, UCLA_2) and is unrelated to diagnosis. If cerebellar coverage is partial there, 40% of the features carry site-linked noise.
- 11 of the 19 constant ROIs are cerebellar or vermis; the flag reaches 99 subjects that have no constant ROI, so imputation of undefined values does not cover the problem.
- The test is cheap (about 2 minutes), leak-free and cannot damage the baseline.

### Pre-declared reading rule
The representation "improves" only if, against the baseline with the same inner seed, mean accuracy and balanced accuracy rise by at least 0.02 with accuracy improved in at least 4 of 5 folds, in at least 2 of 3 seeds. No significance test is claimed.

### Limitations and what to expect
- No guarantee: the audit shows no defect in the features, and the earlier results suggest the diagnostic signal is weak overall; the expected gain, if any, is small, and I do not expect it to approach 80%.
- Cerebellar connectivity is reported as informative in ASD; removing it could lower accuracy.
- The flag is by a single rater, its notes' abbreviation ('ic') is not documented locally, and 'maybe' subjects cluster in a few sites, so the effect is confounded with site.
- The ROI subset was chosen after inspecting this cohort's metadata (a design-process caveat), though not after looking at any model score.

## If you prefer no further modeling
Nothing in the audit justifies a broader model search. The most defensible alternatives are to document the baseline as it stands, or to obtain better-controlled data (known TR, per-frame motion, harmonized reprocessing from raw images, and more subjects), which this cohort cannot supply.
