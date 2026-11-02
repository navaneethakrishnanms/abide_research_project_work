# Fast accuracy experiment -- summary (development-set cross-validation)

**85% target reached: NO.** Best measured outer-CV accuracy is **64.8%** (CC200 / RBF SVM). All eight candidates fall between 61.3% and 64.8%.

## Design (verified)
- Development set: **600 subjects (268 ASD / 332 TC), 19 sites** = `phase2_train_subjects.txt` (676) intersected with the 780-subject FD-filtered cohort. The locked 170-subject holdout file was never opened; no independent test was run.
- Outer: StratifiedKFold(5, shuffle, seed 42), 120 subjects per fold, identical for every candidate (fold-assignment hash identical across all 8 candidates).
- Inner: StratifiedKFold(3, seed 42), ROC-AUC grid search over C (and kernel gamma) and SelectKBest k, per outer fold. Mean imputation, Fisher-Z, scaling, k-best selection and the classifier all sit inside one sklearn Pipeline fitted on training rows only.
- Atlases kept separate: AAL-116 (6,670 features) and CC200 (19,900 features). Class weights balanced.

## Results (mean +/- SD over 5 outer folds)
| Atlas | Model | Accuracy | Bal. acc. | Sens. | Spec. | Prec. | F1 | AUC | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| CC200 | RBF SVM | **0.648 +/- 0.063** | 0.641 | 0.571 | 0.710 | 0.627 | 0.594 | 0.700 +/- 0.052 | 22 s |
| AAL116 | Linear SVM | 0.642 +/- 0.051 | 0.643 | 0.660 | 0.626 | 0.593 | 0.623 | 0.694 +/- 0.035 | 9 s |
| AAL116 | LR (L2) | 0.642 +/- 0.043 | 0.638 | 0.604 | 0.672 | 0.601 | 0.601 | 0.694 +/- 0.037 | 12 s |
| CC200 | LR (L2) | 0.642 +/- 0.048 | 0.636 | 0.582 | 0.690 | 0.610 | 0.593 | 0.689 +/- 0.030 | 35 s |
| CC200 | Linear SVM | 0.640 +/- 0.051 | 0.640 | 0.638 | 0.642 | 0.598 | 0.614 | 0.686 +/- 0.032 | 29 s |
| AAL116 | RBF SVM | 0.637 +/- 0.038 | 0.627 | 0.541 | 0.714 | 0.609 | 0.571 | 0.698 +/- 0.042 | 8 s |
| AAL116 | L1 LR | 0.635 +/- 0.028 | 0.634 | 0.623 | 0.644 | 0.589 | 0.602 | 0.653 +/- 0.025 | 4 s |
| CC200 | L1 LR | 0.613 +/- 0.030 | 0.611 | 0.586 | 0.635 | 0.566 | 0.575 | 0.647 +/- 0.025 | 11 s |

All eight candidates completed all 5 folds (40/40 valid checkpoints). Total runtime 136 s.

Per-fold accuracy (folds 0-4): AAL116 LR 0.608/0.658/0.650/0.592/0.700; AAL116 LinearSVM 0.583/0.650/0.667/0.600/0.708; AAL116 L1 0.592/0.658/0.625/0.642/0.658; AAL116 RBF 0.592/0.658/0.667/0.600/0.667; CC200 LR 0.575/0.658/0.700/0.617/0.658; CC200 LinearSVM 0.583/0.625/0.717/0.617/0.658; CC200 L1 0.575/0.617/0.592/0.650/0.633; CC200 RBF 0.592/0.667/0.608/0.625/0.750.
Per-fold confusion matrices (tn/fp/fn/tp) are in `fold_results.csv`; pooled out-of-fold matrices are in `model_comparison.csv` (best candidate pooled: tn 236, fp 96, fn 115, tp 153). Out-of-fold predictions per subject: `oof_predictions.csv`.

## Interpretation
- **The winner's score is exploratory, not an unbiased final estimate.** Hyperparameters and k were chosen by inner CV (properly nested), but the choice of the best model/atlas among eight candidates used the same outer-fold scores, so 64.8% is mildly optimistic.
- **The candidates are statistically indistinguishable.** The spread across the eight means (3.5 points) is smaller than the typical fold SD (4.5 points). RBF-minus-LR paired fold differences: AAL116 -0.005 +/- 0.020, CC200 +0.007 +/- 0.065. No significance test was run, and with 5 folds none would be convincing.
- **No classifier or atlas change moved accuracy.** Accuracy and AUC (0.65-0.70) are in the same range as Phase 6-10, which suggests the limit is in the features and the cohort, not the classifier.
- **Site is not a trivial label shortcut:** a site-majority rule using training-fold prevalence only scores 52.7% (class-prior baseline 55.3%). Per-site accuracy of the best model varies widely (42%-92%), but most sites have n < 35, so those numbers are noisy.
- **Comparison with the earlier development-only CC200 LR baseline (about 67.2%, AUC 0.7175):** our CC200 LR is 64.2% (AUC 0.689). The setups differ (676 subjects including high-motion scans vs 600 FD-filtered, different CV protocol and repeats), so this is not a like-for-like comparison and I do not claim the new result is better or worse. It is not comparable to Phase 10's 63.0% either (different subjects and folds).
- These are cross-validation estimates on development subjects. Nothing here supports a claim about unseen sites or an independent test set.
