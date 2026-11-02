# Phase 8 Final Limitations

1. **Modified reproduction.** This project deviates from the original paper's pipeline in several
   explicit, disclosed ways (Strategy C imputation, Elastic Net in place of SVM-RFE, all-non-zero
   feature retention, variable-input SSAE). It is not an exact reproduction and headline performance
   is not directly comparable to the paper's reported 98.2%.
2. **ABIDE multi-site heterogeneity.** The 780-subject cohort spans 20 acquisition sites with
   different scanners/protocols, a well-known source of variance in multi-site fMRI studies.
3. **Site concentration of constant-ROI subjects.** Of the 47 subjects with at least one constant
   ROI, the affected subjects were concentrated at SBL (17/23 subjects, 73.9%) and PITT (13/41,
   31.7%), with smaller counts elsewhere. This is a site-concentrated pattern consistent with a
   possible acquisition/preprocessing effect -- stated here as a hypothesis, not an established cause
   (field-of-view truncation, EPI susceptibility, or scanner protocol were not directly verified in
   this project).
4. **Fold-scoped imputation.** A subset of the model's input values (14,025 of the possible
   subject-feature entries) are training-fold-derived imputed values rather than direct measurements.
5. **Elastic Net feature instability across folds.** Most features (5,384/6,670, 80.7%) were never
   selected in any fold; only 50/6,670 were selected in every fold, consistent with expected
   instability among correlated FC features under L1-type regularization and fold-to-fold sampling
   variation.
6. **Model overfitting.** Final training loss (~3.5e-5) was far below validation loss (~0.03-0.15),
   indicating a substantial train/validation generalization gap in this experiment.
7. **Moderate classification performance.** Accuracy 61.5%+/-2.5%, ROC-AUC 0.677+/-0.024 -- above
   chance-level discrimination but well below the paper's disputed headline figure.
8. **XAI method dependence.** Different XAI methods disagree substantially in which features they
   rank highly; cross-method top-10% agreement in this project's outputs was often partial (most
   pairwise Jaccard values well below 1.0).
9. **ROAR provenance limitation.** Phase 6 did not persist the exact per-feature training-fold
   Integrated Gradients ranking used to order ROAR's feature removal; Phase 7's
   `roar_feature_support.csv` uses saved test-fold Integrated Gradients as a labeled proxy. The
   aggregate ROAR performance curves themselves are direct Phase 6 outputs and remain valid; only the
   individual feature-level attribution of "why a given feature was removed at a given threshold" is
   proxy evidence, not the literal record.
10. **No independent external validation.** All results are from internal 5-fold cross-validation on
    a single dataset; no external/held-out cohort from a different source was used.
11. **No clinical validation.** No clinical outcome, diagnostic-utility trial, or clinician-facing
    evaluation was performed.
12. **No biological/experimental validation.** No wet-lab, imaging-follow-up, or independent
    biological experiment was performed on any identified feature or ROI.
13. **No causal inference.** All feature-selection, XAI, and ROAR evidence is associational/model-
    dependent; none of it establishes causality between an FC connection and ASD status.
14. **No evidence that computationally stable features are biological biomarkers.** Cross-fold
    stability, XAI support, and ROAR association are computational-evidence properties of this
    specific pipeline and dataset; they have not been independently biologically validated.
