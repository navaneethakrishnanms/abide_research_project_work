# Next experiment plan (not launched)

## Priority 0 -- reporting decision (no runtime)
Label Phase 6/9/10 as cross-validation estimates on a cohort containing 158 holdout subjects, and treat the holdout as **partially contaminated**. If a final holdout test is wanted later, it should be one pre-declared evaluation of a single frozen pipeline trained on the 600 development subjects only, disclosed as "ID-independent for training, not independent of the research process".

## Single recommended experiment: small dynamic-connectivity comparison on the development cohort
- **Why:** static FC, six classifier/atlas variants, site centering and motion adjustment all sit at 61-65%, with a stable ceiling and errors shared across models. Dynamic features are the one remaining input family whose data already exist for all 600 development subjects (600/600 AAL-116 files, aligned folds, no NaNs).
- **Design (fixed before running):** AAL-116 only; sliding windows of 50 TRs, step 10; per subject, per edge: variance across windows and mean absolute window-to-window change (13,340 features), computed from each subject's own time series only (no cross-subject statistics). Same 600 subjects and outer folds, same pipeline pieces (training-fold imputation/scaling/SelectKBest, inner 3-fold grid). Candidates: L2 logistic regression and linear SVM on (a) dynamic features alone and (b) dynamic features plus static FC, each paired with the static-only result from the fast experiment. Include mean FD as a covariate/adjustment variant, since variability features are motion-sensitive.
- **Cannot be done:** scrubbing/censoring by frame (no per-frame FD); seconds-based windows (no TR).
- **Estimated runtime:** about 5-8 minutes (feature extraction a few minutes, modeling under 2 minutes).
- **Expectation:** no basis to predict an improvement; the result may well remain near 64%. Report as exploratory, paired per fold, no claim beyond the development CV.

## Not recommended now
Further static-FC classifier/atlas variants, deep-network retraining, or any evaluation on the holdout.
