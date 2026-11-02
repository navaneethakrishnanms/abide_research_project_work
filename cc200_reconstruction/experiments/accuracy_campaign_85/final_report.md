# CC200 accuracy campaign (target 85%): final report
Atlas: CC200 only. Data: 676 authorized training subjects (313 ASD / 363 TC, 19 sites); development cross-validation (10 folds x seeds 0, 1, 2; identical folds for every experiment). The restricted holdout, its IDs/labels/features/predictions and `RESTRICTED_id_membership.csv` were never opened.
Baseline, X1, X2, their reports, the cached features and hashes were not modified (hashes re-verified before every run). The selection rule (`selection_rule.py`) is unchanged since the plan: `True`; `campaign.py` unchanged since the plan: `True`.

## 1. Answer
**85% was NOT achieved, and nothing in this campaign supports it.** The highest mean accuracy under a valid evaluation is **66.42%** (P5_E1_SOFTVOTE); the highest ROC-AUC is 0.7329. The reference logistic regression reaches 65.68% (nested threshold, R1) or 66.07% (default 0.5 threshold, R0).
**No candidate passed the frozen selection rule** (0 of 12 candidates), so no replication on fresh seeds was triggered and no gain approached the 5-point leakage-investigation trigger (largest gain: +0.74 points). Twelve candidates were examined; the largest gain is within what chance would produce given paired-fold standard errors of 0.5-1.2 points.

## 2. All experiments ranked by mean R1 accuracy (the primary metric), with stability
R1 = threshold chosen on inner out-of-fold scores of the outer-training partition only; R0 = the model's default threshold. Means over 3 seeds; the +/- is the SD of the three seed-level accuracies; "fold SD" is the mean fold-to-fold SD of accuracy.
| Rank | Experiment | Description | Accuracy R1 | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold SD (pts) | Accuracy R0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P5_E1_SOFTVOTE | Soft vote of TS_LR, LSVM, LR (Platt-calibrated, training-only inner OOF) | 66.42% +/- 0.78 | 66.30% | 0.7321 | 64.64% | 67.95% | 63.53% | 64.06% | 4.32 | 66.96% |
| 2 | P3_TS_LR | Tangent-space (in-fold reference) + L2 LR | 66.27% +/- 0.78 | 65.84% | 0.7141 | 59.96% | 71.72% | 64.64% | 62.19% | 4.50 | 66.57% |
| 3 | P2_LSVM | Linear SVM (balanced) | 66.07% +/- 0.37 | 65.85% | 0.7139 | 62.83% | 68.87% | 63.58% | 63.14% | 4.90 | 65.34% |
| 4 | P5_E2_STACK | Stacking of TS_LR, LSVM, LR (meta-LR on training-only inner OOF) | 65.83% +/- 0.26 | 65.53% | 0.7329 | 61.45% | 69.61% | 63.61% | 62.44% | 4.52 | 66.12% |
| 5 | P1_LR | Reference: LW correlation + Fisher-Z + age/sex, balanced L2 logistic regression | 65.68% +/- 1.04 | 65.36% | 0.7187 | 61.02% | 69.70% | 63.59% | 62.14% | 4.72 | 66.07% |
| 6 | P4_KBEST_LR | In-fold SelectKBest (500/2000/5000) + L2 LR | 65.58% +/- 0.95 | 65.60% | 0.7030 | 65.81% | 65.38% | 62.20% | 63.88% | 5.77 | 65.14% |
| 7 | P2_RBF | RBF SVM (balanced) | 65.29% +/- 1.41 | 64.99% | 0.7176 | 60.92% | 69.05% | 62.95% | 61.91% | 4.91 | 66.72% |
| 8 | P2_XGB | XGBoost (in-fold k=2000 filter) | 65.09% +/- 0.82 | 65.26% | 0.7087 | 67.63% | 62.90% | 61.15% | 64.18% | 6.42 | 65.58% |
| 9 | P2_ET | Extra trees (in-fold k=2000 filter) | 64.69% +/- 0.95 | 65.12% | 0.6917 | 70.82% | 59.41% | 60.07% | 64.97% | 7.00 | 63.41% |
| 10 | P4_PCA_LR | In-fold PCA (25/50/100) + L2 LR | 64.10% +/- 1.11 | 64.14% | 0.6822 | 64.64% | 63.64% | 60.51% | 62.50% | 5.80 | 64.35% |
| 11 | P2_RF | Random forest (in-fold k=2000 filter) | 64.00% +/- 0.73 | 64.63% | 0.6954 | 73.16% | 56.11% | 58.99% | 65.30% | 5.31 | 64.15% |
| 12 | P4_L1_LR | In-fold L1 selection + L2 LR | 63.31% +/- 1.74 | 63.28% | 0.6880 | 62.83% | 63.73% | 59.99% | 61.31% | 5.55 | 63.91% |
| 13 | P3_PC_LR | Partial correlation (LW precision) + L2 LR | 60.70% +/- 0.87 | 60.61% | 0.6672 | 59.32% | 61.89% | 57.29% | 58.27% | 5.85 | 62.67% |

## 3. Frozen screening against the reference (R1; thresholds: +1.5 pts accuracy and balanced accuracy, +0.5 in every seed, >= 20/30 fold wins, AUC within 0.005)
| Experiment | Mean gain acc. (pts) | Mean gain bacc. (pts) | Fold wins | Sign-test p | Paired-fold SE (pts) | Screening |
|---|---|---|---|---|---|---|
| P5_E1_SOFTVOTE | +0.74 | +0.94 | 17.0/30 | 0.292 | 0.53 | no |
| P3_TS_LR | +0.59 | +0.48 | 13.0/30 | 0.819 | 0.74 | no |
| P2_LSVM | +0.39 | +0.49 | 11.0/30 | 0.951 | 0.55 | no |
| P5_E2_STACK | +0.15 | +0.17 | 13.0/30 | 0.819 | 0.53 | no |
| P4_KBEST_LR | -0.10 | +0.24 | 13.0/30 | 0.819 | 0.88 | no |
| P2_RBF | -0.39 | -0.37 | 9.0/30 | 0.992 | 0.62 | no |
| P2_XGB | -0.59 | -0.10 | 11.0/30 | 0.951 | 0.96 | no |
| P2_ET | -0.99 | -0.24 | 14.0/30 | 0.708 | 1.10 | no |
| P4_PCA_LR | -1.58 | -1.22 | 11.0/30 | 0.951 | 0.95 | no |
| P2_RF | -1.68 | -0.72 | 9.0/30 | 0.992 | 0.89 | no |
| P4_L1_LR | -2.37 | -2.08 | 8.0/30 | 0.997 | 1.17 | no |
| P3_PC_LR | -4.98 | -4.75 | 4.0/30 | 1.000 | 1.06 | no |
Per-phase detail (per-seed metrics, confusion matrices, criteria failed): `phase2_report.md`, `phase3_report.md`, `phase4_report.md`, `phase5_report.md`; per-fold tables and hyperparameters: `results/<experiment>_fold_metrics.csv`; per-subject predictions: `results/<experiment>_oof_predictions_seeds012.csv`; configs and hashes: `results/<experiment>_config.json`, `source_and_config_hashes.json`.

## 4. What each phase showed (negative results included)
- **Phase 1 (reproduction):** the campaign pipeline reproduced the saved baseline's out-of-fold scores and inner AUCs exactly (difference 0.0). Reference: 65.68% (R1) / 66.07% (R0), AUC 0.7187.
- **Phase 2 (classifiers):** linear SVM ~ logistic regression (+0.39); RBF SVM, XGBoost, extra trees and random forest were equal or worse (-0.39 to -1.68), and the tree models had clearly lower AUC (0.69-0.71). Pairs of models disagree on 7.5-28% of subjects, but none is better.
- **Phase 3 (representations):** partial correlation was clearly worse (-4.98 points, AUC 0.667): inverting 200 x 200 covariances from about 540 training rows amplifies noise. Tangent-space was +0.59 (not distinguishable from noise; AUC slightly lower).
- **Phase 4 (feature reduction):** SelectKBest -0.10, PCA -1.58, L1 selection -2.37; every reduction lost AUC, which suggests that the strongly regularized full-feature model is not leaving usable signal on the table (an inference, not a tested explanation).
- **Phase 5 (ensembles):** the complementarity condition was met (smallest error phi 0.37), so soft voting and stacking of the top-3 eligible models were built from training-only inner out-of-fold scores. Soft vote: 66.42% (+0.74, wins 17/30 folds, AUC +0.013); stacking +0.15. Neither passed. Caveat: the members were chosen as the best models on these same outer folds, so even this small gain is optimistic. The AUC gain is the one consistent signal (about +0.013 in every seed) and the only reason to look at ensembles again, but AUC was not the criterion and it did not translate into accuracy.

## 5. Site-wise performance (R1; mean of 3 seeds; reference vs top-ranked candidate P5_E1_SOFTVOTE)
| Site | n | ASD share | Reference accuracy | P5_E1_SOFTVOTE accuracy | Diff (pts) |
|---|---|---|---|---|---|
| NYU | 137 | 43% | 65.45% | 67.15% | +1.7 |
| UM_1 | 68 | 40% | 66.18% | 69.12% | +2.9 |
| USM | 53 | 64% | 77.99% | 81.76% | +3.8 |
| UCLA_1 | 52 | 58% | 65.38% | 67.31% | +1.9 |
| PITT | 40 | 48% | 70.00% | 73.33% | +3.3 |
| MAX_MUN | 37 | 41% | 54.95% | 43.24% | -11.7 |
| TRINITY | 35 | 43% | 60.00% | 56.19% | -3.8 |
| YALE | 33 | 55% | 75.76% | 75.76% | +0.0 |
| KKI | 27 | 37% | 56.79% | 56.79% | +0.0 |
| UM_2 | 27 | 37% | 83.95% | 80.25% | -3.7 |
| LEUVEN_2 | 23 | 43% | 66.67% | 72.46% | +5.8 |
| LEUVEN_1 | 22 | 50% | 56.06% | 51.52% | -4.5 |
| OLIN | 22 | 50% | 63.64% | 68.18% | +4.5 |
| SBL | 21 | 48% | 46.03% | 42.86% | -3.2 |
| SDSU | 21 | 29% | 65.08% | 73.02% | +7.9 |
| STANFORD | 20 | 50% | 61.67% | 63.33% | +1.7 |
| UCLA_2 | 17 | 53% | 68.63% | 70.59% | +2.0 |
| CALTECH | 12 | 33% | 63.89% | 63.89% | +0.0 |
| CMU | 9 | 56% | 55.56% | 55.56% | +0.0 |
Per-site values for sites with fewer than about 30 subjects are very noisy. Accuracy differs strongly by site (from about 43-46% at the weakest sites to about 84% at the strongest), consistent with the leave-one-site-out results of experiment X1. Removing site mean and scale effects from the features (X1) did not change accuracy, so these differences are not obviously due to such effects; plausible contributors are site-level sample composition and noise at small n (not tested here).

## 6. Failure cases
68 of 676 subjects (10.1%; 35 ASD, 33 TC) are misclassified in at least 90% of the 39 runs (13 experiments x 3 seeds); 254 subjects (37.6%) are correct in at least 90%. Sites contributing most consistently misclassified subjects: UCLA_1 (9), LEUVEN_1 (6), UM_1 (6), NYU (6), MAX_MUN (5). Median per-subject error rate 23%; ASD subjects have a mean error rate of 35.8% against 34.6% for controls.  (`results/subject_error_rates_all_experiments.csv`.) A sizeable block of subjects is misclassified by nearly every model and representation, which suggests that the limiting factor is the information in these connectivity features at this sample size rather than the choice of classifier (an inference from consistent failures, not a tested cause).

## 7. Best defensible CC200 model and the evidence
**The reference model:** Ledoit-Wolf covariance -> correlation -> Fisher-Z (19,900 edges) + age + sex -> standardization -> balanced L2 logistic regression, C chosen by inner 5-fold AUC (typically C = 0.001). Evidence: reproduced exactly; 65.68%-66.07% accuracy depending on the threshold reading, balanced accuracy 65.36%, AUC 0.719; 12 alternatives (five classifiers, two other representations, three feature-reduction schemes, two ensembles) and, earlier, site harmonization and age/sex residualization all failed to beat it by the predeclared margin.
This is a development cross-validation estimate: there is no independent test, and the earlier holdout overlaps development at the subject level. No model here is validated for deployment.

## 8. Limitations and recommendations
- 676 subjects, 19 sites, 10 folds: accuracy SE about 1.8 points, paired-fold SE 0.5-1.2; differences under 1 point cannot be resolved. Multiple candidates were examined (12), which the +1.5 floor and the replication step were designed to absorb.
- The tree models used an in-fold k = 2000 filter for runtime; the tangent reference is fitted on the outer-training rows (label-free); stability selection and graphical-lasso partial correlations were not run (cost; the Ledoit-Wolf precision is already regularized).
- The route to substantially higher accuracy is not further tuning on this cohort: an independent external cohort with the same preprocessing, many more subjects per site, frame-level motion data, severity/subtype information, and reprocessing from raw images. Any final evaluation on the protected holdout needs a frozen model and your separate authorization.
