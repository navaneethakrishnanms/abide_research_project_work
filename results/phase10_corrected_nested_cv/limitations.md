# Limitations
- Computational classification/reproduction study; no selected feature is claimed as a biomarker, causal connection or clinical/disease mechanism.
- Not an exact reproduction of the published paper (Strategy C imputation, Elastic Net instead of SVM-RFE, variable input SSAE); no equivalence to its 98.2% is claimed.
- 5 outer folds / 2 inner folds: limited power, noisy selection; greedy stage-wise search not a full grid; tie tolerance 0.005.
- Multi-seed runs reuse each fold's seed-42 selection (selection itself is not repeated per seed).
- Site-grouped track is a separate evaluation; results must not be pooled with the ordinary track.
- Constant-ROI / imputed entries concentrated at SBL and PITT (Phase 4/7) remain a confound; 47 subjects carry imputed values.
- No external validation cohort.
