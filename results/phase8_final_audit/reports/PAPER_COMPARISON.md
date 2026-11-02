# Paper Comparison

Comparison of the original paper (Vidya et al., eClinicalMedicine 2025) against this modified reproduction. Based only on documented project evidence.

| Aspect | Original paper | This project | Difference | Scientific implication |
|---|---|---|---|---|
| Cohort | 884 subjects (paper-reported) | 780 subjects (FD>0.2mm filtered, Phase 2) | Different N and inclusion criteria | Results are not directly comparable sample-for-sample |
| Preprocessing pipeline | Not independently re-verified beyond the paper's own description | CPAC (filt_global), documented and verified in Phase 3/3.6 | Same nominal pipeline family (CPAC) but not independently re-validated against the paper's exact preprocessing run | Assume comparable but not identical preprocessing |
| Atlas | AAL-116 (paper) | AAL-116 (this project) | Same atlas | Directly comparable at the atlas level |
| FC construction | Pearson correlation, 6,670 features (paper) | Pearson correlation, 6,670 features (this project) | Same nominal construction | Directly comparable |
| Missing/undefined FC handling | Not documented as addressed by the paper | Explicitly handled via Strategy C fold-scoped imputation (14,025 undefined entries) | This project introduces explicit handling the paper does not document | A disclosed methodological addition, not present in the original pipeline |
| Feature selection | SVM-RFE, reduced to exactly 1,000 features | Elastic Net, all non-zero features retained (427-498 per fold) | Different algorithm and different, variable output size | Not a like-for-like feature-selection comparison |
| Number of selected features | 1,000 (fixed) | 427-498 (variable per fold) | Substantially fewer, and variable | SSAE input dimensionality differs from the paper by fold |
| SSAE architecture | 1,000 -> ... -> 2 (paper, fixed input) | N(variable, 427-498) -> 500 -> 100 -> 2 | Variable input dimension not present in the paper's design | Direct architectural equivalence cannot be claimed |
| Cross-validation | Not fully documented as independently verified here | 5-fold StratifiedKFold, subject-level, fold-scoped preprocessing | This project's CV structure is fully documented and leakage-controlled | N/A -- documented for this project only |
| XAI | Not part of the original paper's reported pipeline (per this project's documentation) | 7 methods (IntegratedGradients, GuidedBackprop, DeepLift, DeepLiftShap, GradientShap, SHAP, LIME) | This project adds an XAI layer beyond the original paper | Additional interpretability analysis not present in the original paper |
| ROAR | Not part of the original paper's reported pipeline (per this project's documentation) | 13 removal thresholds x 5 folds = 65 runs | This project adds a ROAR layer beyond the original paper | Additional removal-based evidence not present in the original paper |
| Evaluation / headline performance | 98.2% accuracy (paper-reported; paper is documented in Phase 4.1 audit trail as subject to an Expression of Concern for SVM-RFE leakage) | 61.5%+/-2.5% accuracy, 0.677+/-0.024 ROC-AUC (this project, 5-fold mean+/-SD) | Substantially different reported performance | The difference does not by itself establish which methodology is more correct; see Section 6/25 |

**Note:** Differences in reported accuracy alone do not establish which methodology is more correct -- the pipelines differ in feature selection method, feature count, missing-value handling, and other respects listed above.