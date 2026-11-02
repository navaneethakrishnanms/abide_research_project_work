# Methodology (corrected Phase 10)
Cohort: 780 subjects (351 ASD / 429 TC), 20 sites, AAL-116, 6,670 FC features (unchanged).
Outer: 5-fold (ordinary StratifiedKFold seed 42; site-grouped StratifiedGroupKFold seed 42). Inner: 2-fold (same type) inside each outer-train.
Preprocessing (Strategy C mean imputation -> Fisher-Z (clip 1e-6) -> StandardScaler) fitted on the training rows of each split only.
Search per outer fold: EN 11 configs (C grid x l1 0.5 plus l1 0.25/0.75 at C=0.1) with 500x100 SSAE; ARCH ['64x32', '128x32', '128x64', '256x64', '256x128']; WD [0.0001, 0.001, 0.01, 0.05]; dropout [0.1, 0.2, 0.3, 0.5] (+0); LR [0.0001, 0.0003, 0.001, 0.003]; patience [5, 10, 15, None].
Selection: max mean inner ROC-AUC; ties within 0.005 -> simpler (fewer features / fewer params / stronger regularization / smaller LR / earlier stopping).
Final: refit preprocessing + Elastic Net + SSAE on entire outer-train, one outer-test evaluation. Seeds 42/123/2026 vary SSAE init/shuffle only.
Comparators (ordinary only): B stability (6 x 50% inner-train subsamples), H representation/PCA (LR probe), CLF classifier choice by inner CV.
