# Abstract-Ready Summary

**BACKGROUND.** Autism spectrum disorder (ASD) classification from resting-state fMRI functional
connectivity has been proposed using AAL-116 parcellation, SVM-RFE feature selection, and a stacked
sparse autoencoder (SSAE), with one published pipeline (Vidya et al., 2025) reporting 98.2% accuracy
-- a result subject to a documented Expression of Concern regarding SVM-RFE data leakage.

**METHODS.** This project performed a modified reproduction using 780 ABIDE-I subjects (351 ASD, 429
Control, 20 sites, FD>0.2mm filtered), AAL-116 ROI time series, Pearson functional connectivity
(6,670 features), fold-scoped mean imputation of 14,025 undefined correlation entries (Strategy C),
Fisher-Z transformation, Elastic Net feature selection (fixed hyperparameters, all non-zero
coefficients retained rather than a fixed count of 1,000), and a variable-input SSAE (N->500->100->2)
evaluated under 5-fold subject-level stratified cross-validation. Seven XAI methods and a 65-run ROAR
(RemOve And Retrain) procedure were applied post-hoc for interpretation.

**RESULTS.** Elastic Net selected 427-498 features per fold (mean 464.2, median 463, SD 23.9). Mean
accuracy was 61.5%+/-2.5%, balanced accuracy 61.0%+/-2.7%, sensitivity 55.3%+/-5.0%, specificity
66.7%+/-1.7%, F1 0.563+/-0.038, ROC-AUC 0.677+/-0.024. Training loss (~3.5e-5) was far below
validation loss (~0.03-0.15), indicating overfitting. Fifty of 6,670 features were selected in all 5
folds; 87 features met a pre-defined multi-evidence criterion (>=4/5-fold Elastic Net selection and
>=3/7 XAI-method top-10% support).

**INTERPRETATION.** The model achieved above-chance discrimination but did not reproduce the original
paper's reported accuracy. A small subset of FC features showed convergent computational evidence
across independent selection and interpretability methods, providing candidates for further
investigation.

**LIMITATIONS.** This is a modified, not exact, reproduction; results reflect a single-dataset,
internally cross-validated pipeline with no external or clinical validation; identified features are
multi-evidence computational candidates, not validated biomarkers; ROAR feature-level provenance
relies on a documented proxy for the exact training-fold ranking; site-concentrated constant-ROI
subjects required imputation for a subset of inputs.
