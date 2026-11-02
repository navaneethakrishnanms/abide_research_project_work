# Likely Viva Questions and Defensible Answers

**1. Why did you use ABIDE?**
It is the largest publicly available multi-site ASD resting-state fMRI dataset with existing
preprocessed (CPAC) derivatives, and it is the dataset used by the paper being reproduced.

**2. Why FD filtering?**
Framewise displacement filtering (FD>0.2mm) removes subjects with excessive head motion, a well-known
confound in fMRI functional-connectivity analysis; the threshold follows the paper's stated criterion
(Phase 2), applied to this project's own cohort rather than forcing an exact subject count match.

**3. Why AAL-116?**
It is the atlas used by the paper being reproduced and is available as a CPAC derivative
(`rois_aal`), verified against 116 ROIs per subject.

**4. Why Pearson correlation?**
The standard, paper-matching functional-connectivity construction method between ROI mean time
series.

**5. Why Fisher-Z?**
Pearson correlations are bounded in [-1,1] and not normally distributed; Fisher-Z (arctanh)
transformation is standard practice before using correlations as model inputs or in linear models.

**6. Why did undefined correlations occur?**
19 ROIs had zero-variance (constant) signal in a subset of subjects (47 subjects total), making the
Pearson correlation formula's denominator zero for those ROI pairs -- an undefined, not merely
missing, value (Phase 4.1).

**7. Why did you choose Strategy C?**
After a documented sensitivity comparison of subject exclusion, ROI exclusion, and imputation
(Phase 4.2/5B/5C), fold-scoped imputation (Strategy C) was chosen to preserve the full cohort and
full ROI set while keeping the handling leakage-safe.

**8. Why fold-scoped imputation specifically?**
Imputing with a statistic computed only from each fold's training data prevents information from the
test fold from influencing the imputed values, avoiding data leakage.

**9. Why Elastic Net?**
It provides a well-understood, leakage-controllable feature-selection alternative to SVM-RFE, with L1
sparsity plus L2 stability, and fixed a priori hyperparameters (not tuned against test data).

**10. Why not SVM-RFE?**
SVM-RFE is the paper's original method, but the paper carries a documented Expression of Concern
regarding SVM-RFE-related data leakage (per this project's Phase 4.1 audit trail); Elastic Net was
chosen as a disclosed, leakage-controlled substitute.

**11. Why keep all non-zero features instead of forcing exactly 1,000?**
Forcing exactly 1,000 features would mean either arbitrarily discarding informative non-zero
coefficients or arbitrarily including uninformative zero coefficients; retaining all non-zero
coefficients respects what Elastic Net's regularization actually determined to be informative
(Phase 5D).

**12. Why does feature count vary across folds?**
Elastic Net is fit independently per fold on that fold's training data only (leakage-safe), so
different training samples naturally yield different non-zero coefficient sets and counts (427-498).

**13. Why variable-input SSAE?**
A direct, necessary consequence of the all-non-zero feature-selection decision; each fold already
trains an independent model instance under proper cross-validation.

**14. Why use seven XAI methods?**
Different XAI methods make different assumptions (gradient-based, perturbation-based,
game-theoretic); using seven methods and reporting them separately, plus their cross-method
agreement, avoids overstating the reliability of any single method's attributions.

**15. What does XAI actually tell you?**
Which input features the trained model's output was most sensitive to, under a specific attribution
method's definition of "importance" -- a property of the model and method, not independent evidence
about the brain.

**16. What does ROAR tell you?**
How the retrained model's performance changes as attribution-ranked features are progressively
removed -- evidence of the model's dependence on those features under this specific procedure, not
proof of causal or biological importance.

**17. Why is the accuracy only ~61.5%?**
Possible contributing factors documented in this project include: ABIDE's known multi-site
heterogeneity, the modified (non-paper) feature-selection and imputation methodology, observed
overfitting during SSAE training, and the inherent difficulty of the ASD-vs-control classification
task from resting-state FC alone. No single cause is established as dominant.

**18. What evidence of overfitting did you observe?**
Final training loss (~3.5e-5) far below validation loss (~0.03-0.15) during SSAE fine-tuning across
folds (Phase 6 training_summary.csv).

**19. Why is your result different from the paper?**
The pipelines differ in feature-selection method (Elastic Net vs SVM-RFE), number/variability of
selected features (427-498 variable vs fixed 1,000), and undefined-FC handling (Strategy C imputation
vs undocumented in the paper) -- see `reports/PAPER_COMPARISON.csv` for the full itemized comparison.
The paper's reported figure is also associated with a documented Expression of Concern regarding
SVM-RFE leakage.

**20. Can you call your selected features biomarkers?**
No. They are, at most, "multi-evidence computational feature candidates" -- features with convergent
statistical selection and model-attribution evidence within this pipeline. No independent biological
or clinical validation was performed.

**21. What are your main limitations?**
See `reports/FINAL_LIMITATIONS.md` -- modified (not exact) reproduction, site-concentrated
constant-ROI subjects, fold-scoped imputation, Elastic Net instability, overfitting, moderate
performance, XAI method disagreement, the ROAR provenance/proxy limitation, and the absence of
external, clinical, or biological validation.

**22. How would you improve the study in future work?**
Possible directions (not undertaken here): external cohort validation, addressing the observed
overfitting (e.g. regularization/architecture changes), persisting exact per-feature ROAR removal
provenance, and any biological/clinical follow-up validation of the multi-evidence candidates -- none
of which was performed in this project.
