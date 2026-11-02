# Can any legitimate threshold or metric reach 85%? Analysis of saved AAL-116 out-of-fold predictions

**Answer: No.** In 44 prediction groups from the saved AAL-116 experiments, the highest accuracy at the documented threshold is **66.0%**, the highest with a properly cross-fitted threshold is **65.8%**, and even an optimistic oracle threshold reaches only **66.5%**. The best AUC is **0.708**. No group reaches 85%.

Scope: saved out-of-fold predictions for the 600-subject development cohort only. No model was trained, no hyperparameter search was run, and no restricted holdout file, label, prediction or `RESTRICTED_id_membership.csv` was opened. Every group has exactly 600 unique subjects, one prediction each (asserted). Code: `demo/threshold_85_analysis.py`; full table: `demo/threshold_85_analysis_results.csv` (all six metrics for 44 groups).

## Method
- **Documented threshold:** metrics computed from the stored `y_pred` (each experiment's own rule, 0.5 for most; E1/E3 use thresholds selected inside training data).
- **Cross-fitted threshold:** for each outer fold (or each site, for leave-site-out files), the threshold maximizing balanced accuracy is chosen **only from the predictions of the other folds/sites** and then applied to the held-out one. A subject's prediction never influences its own threshold.
- **Oracle threshold:** best accuracy over all thresholds on the same predictions. This is an optimistic upper bound, **not an estimate**, and is shown only to prove that no threshold could reach 85%.
- AUC is threshold-free.

## Actual results (selected rows; sources are under `C:\abide\results\`)
| Group (source file) | Doc. acc. | Bal. acc. | Sens. | Spec. | F1 | AUC | Cross-fitted acc. / bal. acc. / sens. / spec. | Oracle acc. (optimistic) |
|---|---|---|---|---|---|---|---|---|
| Baseline E0, seed 42 (`accuracy_improvement_v1/oof_predictions.csv`) | 0.642 | 0.638 | 0.604 | 0.672 | 0.601 | 0.692 | 0.640 / 0.646 / 0.705 / 0.587 | 0.653 |
| Baseline E0, seed 123 | 0.647 | 0.643 | 0.608 | 0.678 | 0.606 | 0.691 | 0.642 / 0.645 / 0.672 / 0.617 | 0.655 |
| Baseline E0, seed 2026 | 0.653 | 0.650 | 0.616 | 0.684 | 0.613 | 0.684 | 0.623 / 0.621 / 0.597 / 0.645 | 0.658 |
| Baseline, leave-site-out (`.../leave_site_out_oof_predictions.csv`) | 0.622 | 0.618 | 0.586 | 0.651 | 0.580 | 0.684 | 0.618 / 0.629 / 0.731 / 0.527 | 0.643 |
| Tangent + Ledoit-Wolf, seed 2026 (`accuracy_research_v2/oof_predictions_tangent.csv`) | 0.655 | 0.650 | 0.604 | 0.696 | 0.610 | 0.690 | 0.647 / 0.644 / 0.619 / 0.669 | 0.665 |
| FC + age + sex tuned, seed 2026 (`accuracy_research_v3/oof_predictions.csv`) | **0.660** | 0.656 | 0.623 | 0.690 | 0.621 | 0.686 | 0.657 / 0.654 / 0.631 / 0.678 | 0.662 |
| Test-site centering, leave-site-out (`test_site_centering_experiment/oof_predictions.csv`) | 0.657 | 0.656 | 0.646 | 0.666 | 0.627 | **0.708** | 0.658 / 0.665 / 0.731 / 0.599 | 0.665 |

Other groups (CPM, linear/RBF SVM, L1, cerebrum-only, dynamic FC, extended grids, thresholded variants E1-E3) range from 49.8% to 65.7% documented accuracy; all rows are in the CSV. Dynamic-FC groups are near chance (AUC 0.50-0.52). Some groups are duplicates by construction (cerebrum-only `ROI116_baseline` equals E0; `leave_site_out_baseline` equals E0 leave-site-out).

Findings:
- Choosing a threshold on training/validation data does not help. Cross-fitted accuracy differs from the documented accuracy by -3.1 to +1.8 points, and it is usually lower.
- Moving the threshold only trades sensitivity for specificity (for example sensitivity 0.60 to 0.70 with specificity 0.67 to 0.59), at roughly constant balanced accuracy.
- The ceiling is set by ranking quality: with AUC about 0.69-0.71, no threshold gives more than about 66%.

## Limitations of this analysis
- The cross-fitted thresholds are chosen from other folds' out-of-fold predictions, which came from models trained with the held-out fold's subjects included. The cross-fitted figures are therefore still slightly optimistic. This cannot be removed without retraining, which was not allowed.
- 600 subjects, five folds: fold-level SDs of accuracy are about 3-5 points, and differences below about 2 points are noise. Differences between seeds of the same model are about 1 point.
- Many groups share the same subjects and folds, so they are not independent evidence.
- Scores differ in meaning (probabilities for logistic regression, margins for SVM); thresholds were searched on the score's own scale.
- All estimates are development cross-validation, not independent-test performance. The old CC200 74.1% holdout figure is out of scope here (different atlas, reused holdout, no saved predictions).

## ILLUSTRATIVE NUMERICAL EXAMPLE (not an experimental result)
Assumption: two equal-variance normal score distributions (binormal model), balanced classes, threshold at the midpoint. This is a mathematical illustration only; no real labels or scores are used except the observed best AUC as input to the formula.
- Reaching 85% accuracy under this model requires a class separation d' = 2·Φ⁻¹(0.85) ≈ 2.07, which corresponds to **AUC ≈ 0.93**.
- The best AUC actually observed is 0.708, which under the same model implies an accuracy ceiling of about **65%**, close to the measured 62-66%.
- Conclusion of the example: 85% is not a threshold problem. It would require the model's ranking quality to rise from AUC about 0.70 to about 0.93, which no saved experiment shows.
