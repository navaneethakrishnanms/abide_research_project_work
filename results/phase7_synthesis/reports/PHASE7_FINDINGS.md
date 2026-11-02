# Phase 7 Findings

**Q1. How many features were selected at least once?** 1,286 of 6,670 (19.3%).

**Q2. How many were selected in every fold?** 50.

**Q3. Which features are most stable?** The 50 features selected in all 5/5 folds (see
`feature_stability/stable_features.csv`, filtered to n_folds_selected==5); a further 93 were selected
in 4/5 folds.

**Q4. Which features have strongest cross-method XAI support?** See
`xai/xai_feature_synthesis.csv`, `xai_method_support_count_top10pct` column. Support-count
distribution across all XAI-attributed features: {0: np.int64(1012), 1: np.int64(71), 2: np.int64(27), 3: np.int64(11), 4: np.int64(10), 5: np.int64(19), 6: np.int64(46), 7: np.int64(90)}.

**Q5. Which features show Elastic Net + XAI agreement?** Category "A: Stable EN + strong XAI"
(selected >=4/5 folds AND supported by >=4/7 XAI methods' top-10%): 84
features. Full breakdown: `multi_evidence/elastic_net_xai_intersection.csv`.

**Q6. What does ROAR show?** Baseline (no removal): accuracy 0.6154, AUC 0.6771.
Full curve in `roar/roar_curve_summary.csv`. Performance changes with removal percentage reflect the
retrained model's dependence on the removed features under this procedure -- not proof that those
features are causally necessary for the underlying phenomenon.

**Q7. Do the same features repeatedly appear across Elastic Net, XAI, and ROAR?** Partially -- see
`multi_evidence/consensus_feature_candidates.csv` (87 candidates meeting the
explicit, documented thresholds of >=4/5-fold EN selection and >=3/7-method XAI top-10% support).
This is convergence across evidence sources, not proof of ground truth.

**Q8. Which ROI pairs repeatedly appear?** See `feature_stability/stable_features.csv` and
`multi_evidence/consensus_feature_candidates.csv` for the specific (ROI_i, ROI_j) pairs, by verified
AAL name.

**Q9. Which AAL regions participate in those repeated connections?** See
`roi_analysis/roi_feature_summary.csv`, `n_stable_connections_ge4folds` column, for the per-ROI count
of connections meeting the stability threshold.

**Q10. Where do the methods disagree?** XAI method agreement (`xai/xai_method_agreement.csv`) is
well below 1.0 for most method pairs, meaning each method's top-10% set only partially overlaps with
others' -- full pairwise values are in that file, not summarized away here. Category "C: Unstable EN
+ strong XAI" (81 features) shows XAI methods
attributing importance to features Elastic Net did not select stably across folds -- a real
disagreement between selection-based and attribution-based evidence, reported as such.

**Q11. What can legitimately be called a robust computational finding?** Statements of the form "this
FC feature was selected in 5/5 independent cross-validation folds" or "this ROI pair's connection was
supported by N of 7 independent XAI methods' top-10% ranking" are robust computational findings within
this specific pipeline (Strategy C + Elastic Net + variable-input SSAE) and this specific dataset.

**Q12. What CANNOT yet be called a biological biomarker?** Every single finding in this phase.
Selection frequency, XAI support, and ROAR association are all evidence about *this model's*
behavior on *this dataset* under *this specific, modified-reproduction methodology* -- none of them,
individually or combined, constitutes biological or clinical validation. See
`reports/PHASE7_LIMITATIONS.md`.
