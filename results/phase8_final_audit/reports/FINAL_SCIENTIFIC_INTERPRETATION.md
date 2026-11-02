# Final Scientific Interpretation

## A. Supported claims

- A 780-subject ABIDE-I cohort (351 ASD, 429 Control, 20 sites) was constructed after FD>0.2mm
  filtering (Phase 1-2).
- AAL-116 ROI time series were obtained via the CPAC pipeline for the full cohort (Phase 3-4).
- Pearson functional connectivity produced 6,670 lower-triangle features per subject (Phase 5A).
- Strategy C fold-scoped mean imputation was applied to the 14,025 undefined FC entries (Phase 6).
- Elastic Net (fixed hyperparameters, not tuned against test data) selected 498-427
  features per fold across 5 folds: [498, 479, 427, 463, 454].
- 50 of 6,670 features were selected in all 5 folds; 93 more in exactly 4/5
  folds.
- Seven independent XAI methods produced model-attribution patterns over the Elastic-Net-selected
  features of each fold.
- ROAR performance changed as attribution-ranked features were progressively removed and the model
  retrained (13 thresholds x 5 folds = 65 runs).

## B. Exploratory claims

- Repeated ROI participation across multiple selected FC connections (see
  `results/phase7_synthesis/roi_analysis/`) -- descriptive, not a ranked "importance" claim.
- Cross-method XAI agreement patterns (support-count distribution: {0: np.int64(1012), 1: np.int64(71), 2: np.int64(27), 3: np.int64(11), 4: np.int64(10), 5: np.int64(19), 6: np.int64(46), 7: np.int64(90)}).
- 87 "multi-evidence computational feature candidates" (>=4/5-fold Elastic Net selection
  AND >=3/7-method XAI top-10% support) -- convergence among computational analyses, not independent
  validation.
- Anatomical/hemisphere patterns suggested by the selected FC connections (e.g. the 50 fully-stable
  features' left-left/right-right/left-right/midline split) -- a descriptive grouping by AAL naming
  convention only.

## C. Unsupported claims

The project cannot establish, and does not claim:

- Validated biological biomarkers.
- Causal brain mechanisms underlying ASD.
- Clinical diagnostic utility.
- Generalization to unseen hospitals/populations (no external validation was performed).
- Superiority over the original paper.
- Equivalence to the original paper (this is an explicitly modified reproduction).
- Causal importance of any individual FC connection.

## Model performance (Phase 6, authoritative)

 fold  n_test  accuracy  balanced_accuracy  precision  sensitivity_recall  specificity       f1  roc_auc
    0     156  0.647436           0.645153   0.611111            0.619718     0.670588 0.615385 0.691135
    1     156  0.589744           0.581395   0.546875            0.500000     0.662791 0.522388 0.669601
    2     156  0.608974           0.600166   0.571429            0.514286     0.686047 0.541353 0.686711
    3     156  0.634615           0.630066   0.594203            0.585714     0.674419 0.589928 0.699252
    4     156  0.596154           0.591196   0.550725            0.542857     0.639535 0.546763 0.638704

Overall (mean +/- SD across 5 folds):

statistic  accuracy  balanced_accuracy  precision  sensitivity_recall  specificity       f1  roc_auc
     mean  0.615385           0.609595   0.574868            0.552515     0.666676 0.563163 0.677081
      std  0.024827           0.026954   0.027685            0.049827     0.017336 0.038245 0.024034

Performance is above chance-level discrimination in terms of ROC-AUC; classification accuracy is
approximately 61.5%; performance varies across folds; sensitivity (55.3%+/-5.0%) is lower than
specificity (66.7%+/-1.7%) on average; the model does not reproduce the paper's reported 98.2% result.

## Overfitting audit

Phase 6's SSAE training showed final training loss ~3.5e-5 against validation loss in the ~0.03-0.15
range, with N approximately 427-498 selected features and roughly 530 effective training subjects
after the internal validation split. This pattern indicates a substantial train/validation
generalization gap in this experiment. This is a statement about the observed training/validation
behavior in this experiment, not a claim that the SSAE architecture is inherently invalid.

## Paper-result context

The original paper (Vidya et al., eClinicalMedicine 2025) reported 98.2% accuracy. This project's
prior audit (Phase 4.1) documented an Expression of Concern associated with that paper regarding
SVM-RFE data leakage; that status was not re-independently-verified as current within this phase, so
it is stated here only as: "the project's prior audit documented an Expression of Concern." This
project's substantially lower accuracy does not, by itself, establish that the paper is incorrect, nor
does it establish that this project's approach is superior -- the two pipelines differ in feature
selection method, feature count, missing-value handling, and other respects documented in
`reports/PAPER_COMPARISON.csv`.
