# Phase 7 Limitations

1. This is a MODIFIED REPRODUCTION of the original paper, not an exact reproduction.
2. Strategy C introduced fold-scoped imputation for the 14,025 undefined Pearson FC entries -- a
   project-introduced methodological choice the paper does not specify.
3. Elastic Net replaced the paper's SVM-RFE feature-selection step.
4. All-non-zero feature selection replaced the paper's fixed 1,000-feature target.
5. SSAE input dimension varied by fold (427-498), instead of a fixed 1,000.
6. Phase 6 accuracy was approximately 61.5% +/- 2.5% -- far below, and not comparable to, the
   original paper's disputed 98.2% figure (itself under an active Expression of Concern for data
   leakage, per Phase 4.1).
7. ROC-AUC was approximately 0.677 +/- 0.024.
8. Phase 6's training/validation loss curves showed clear overfitting (near-zero training loss vs.
   substantially higher validation loss).
9. XAI attributions are model-dependent -- they explain what THIS trained model responds to, not an
   independent ground truth about the brain.
10. ROAR results are model-dependent and procedure-dependent (reduced fine-tuning epochs were used
    for the 65 ROAR retrains, per Phase 6's documented compute-scoping decision).
11. Elastic Net selection stability does not establish biological importance.
12. XAI cross-method agreement does not establish biological causality.
13. ROAR-driven performance change does not establish biological causality.
14. AAL ROI interpretation used here is anatomical name-mapping only, not disease-mechanism proof.
15. The 50 features selected in every fold are computationally stable within this pipeline, not
    automatically biomarkers.
16. Site effects and preprocessing effects (documented extensively in Phase 4.1/4.2 -- e.g. the
    SBL/PITT constant-ROI concentration) may influence which features appear stable or important,
    independent of any diagnosis-related signal.
17. The 47 constant-ROI subjects were retained via Strategy C's fold-scoped mean imputation; their
    imputed values are not measured brain activity, and any XAI attribution touching their
    imputed features (in folds where they fell in the test set) reflects a fit to a substituted
    statistic for those entries, not a real measurement.

**Additional limitation identified during this phase:** Phase 6 did not persist the exact per-feature
training-fold Integrated-Gradients ranking used internally to order ROAR's feature removal --
`roar/roar_results.csv` contains only aggregate per-threshold performance. Phase 7's
`roar_feature_support.csv` uses the closest available proxy (Phase 6's saved *test-fold* Integrated
Gradients attributions) and explicitly labels this as an approximation, not the literal ROAR removal
record.
