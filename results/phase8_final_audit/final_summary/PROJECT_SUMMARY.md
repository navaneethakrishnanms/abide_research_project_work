# Final Project Summary

**WHAT:** ABIDE ASD-vs-control classification using resting-state fMRI functional connectivity.

**HOW:** AAL-116 ROI extraction -> Pearson functional connectivity (6,670 features) -> fold-scoped
mean imputation of undefined correlations (Strategy C) -> Fisher-Z transformation -> Elastic Net
feature selection (all non-zero features retained, 427-498 per fold) -> variable-input Stacked Sparse
Autoencoder (N->500->100->2, Softmax) -> 5-fold subject-level cross-validation -> seven-method XAI
attribution -> ROAR (RemOve And Retrain) evaluation -> feature/ROI/hemisphere synthesis.

**RESULT:** Approximately 61.5%+/-2.5% accuracy and 0.677+/-0.024 ROC-AUC (5-fold mean+/-SD), with 50
FC features selected in every fold, 87 features meeting a defined multi-evidence criterion across
Elastic Net and XAI, and a documented ROAR performance curve across 13 removal thresholds.

**IMPORTANT:** This is a modified reproduction of a published pipeline, not an exact replication --
Strategy C imputation, Elastic Net (replacing SVM-RFE), all-non-zero feature retention, and a
variable-input SSAE are all explicit, disclosed deviations. The identified FC features are
multi-evidence computational candidates, not validated biological biomarkers; no causal, clinical, or
biological-experimental claims are made or supported by this project.
