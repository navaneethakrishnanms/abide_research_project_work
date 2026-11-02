# Phase 5D Decision Document — All-Non-Zero Elastic Net Architecture

**No strategy is selected here. No real ABIDE Elastic Net or SSAE computation occurred.**

## 1. Is keeping ALL non-zero Elastic Net features technically valid?

Yes. There is no mathematical or implementation requirement that a feature-selection step produce a fixed count. Retaining every non-zero coefficient is a well-defined, deterministic (given fixed hyperparameters and training data) selection rule.

## 2. Can the selected feature count be greater than 1,000?

Yes, if Elastic Net's regularization is weak enough that more than 1,000 of the 6,670 candidate features retain non-zero coefficients. Confirmed structurally by the synthetic hyperparameter grid (weaker regularization → higher non-zero counts, up to 367 of 400 synthetic features at the weakest setting tested).

## 3. Can it be less than 1,000?

Yes, equally straightforwardly, under stronger regularization (as low as 0 of 400 synthetic features at the strongest setting tested).

## 4. Can the feature count vary between CV folds?

Yes — demonstrated directly: identical hyperparameters (C=0.1, l1_ratio=0.5), fit independently (leakage-safe) on 5 different training folds of the same synthetic dataset, produced 5 different non-zero counts (34, 37, 37, 43, 44) and, per the stability table, different feature identities. This is an expected consequence of fold-scoped fitting, not an anomaly to be corrected.

## 5. Can SSAE use N → 500 → 100 → 2 → Softmax?

Yes. Each CV fold already trains an independent model instance under proper nested CV; allowing that instance's input layer to match its own fold's N is a direct, unremarkable consequence — no framework accommodation is needed beyond instantiating a fresh model per fold, which fold-scoped CV already requires regardless of feature-selection approach.

## 6. What information is preserved compared with fixed top-1,000?

All-non-zero preserves *exactly* whatever Elastic Net's regularization determines to be informative — no features are dropped past the sparsity cutoff simply to hit a numeric target, and no features are force-included below Elastic Net's own zero threshold simply to reach 1,000. The trade-off (§7 of `strategy_ab_comparison.csv`) is that this also means losing a single, stable, cross-fold-comparable feature count and, if N exceeds 1,000, losing the paper's own SSAE input dimensionality as a point of structural comparability.

## 7. What are the computational consequences?

Scale roughly linearly with N: SSAE input-layer parameter count (N × 500 weights for the first layer), RAM for activations/gradients, training time per epoch, and per-feature XAI/ROAR cost all grow with N. Near-6,670 (very weak regularization) approaches the cost profile of skipping feature selection entirely; small N (strong regularization) is cheapest but risks discarding informative correlated features via Elastic Net's L1 pressure. None of this was measured on real data — only discussed qualitatively.

## 8. How should feature stability be recorded?

Per-fold: `fold, feature_index, roi_i, roi_j, elastic_net_coefficient, absolute_coefficient, selected` (using the fixed `feature_index` convention from Phase 5A, not a fold-relative position, since positions shift when N changes). Cross-fold: `selection_frequency = (folds selected) / (total folds)`. Full protocol and a synthetic worked example: `feature_stability_protocol.md`.

## 9. How should selected FC features be mapped back to AAL regions?

Every retained feature must resolve through the same fixed pipeline already established: `feature_index → (roi_i, roi_j) → AAL ROI names (via Phase 4.1's verified aal116_roi_mapping.csv) → anatomical connection`. Five distinct concepts must stay separated in any future reporting, per the task's explicit instruction: (1) Elastic-Net-selected feature, (2) stable feature (cross-fold), (3) XAI-important feature, (4) ROAR-supported feature, (5) biologically interpretable FC connection. A feature satisfying (1) alone is not evidence for (5).

## 10. How will XAI work with fold-specific feature sets?

Every listed method (SHAP, Integrated Gradients, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop, LIME) operates on whatever input a given fold's model actually has — all are structurally compatible with a variable-N input, per fold. The complication is cross-fold aggregation: a feature absent from a fold (because Elastic Net zeroed it) has no attribution in that fold — structurally absent, not zero-importance — and must be reported as such, not conflated with a computed-but-low attribution. Full detail: `xai_implications.md`.

## 11. How will ROAR work with variable N?

Percentage-based removal (the standard ROAR convention) scales naturally: "remove the top 20% of fold k's N_k selected features" is well-defined for any N_k, with the procedure identical across folds even though the absolute count removed differs. Full detail: `roar_implications.md`.

## 12. Is the procedure leakage-safe?

Yes, as documented — provided the fold-scoped structure in `leakage_control_protocol.md` is actually followed: missing-FC handling, Elastic Net fitting, and non-zero feature identification all restricted to training-fold data, with the test fold only ever transformed using training-fold-derived choices, never used to influence any upstream step. This was verified in the synthetic demonstration by construction (the fitting call never references test indices) — not yet implemented for real data.

## 13. Does any human methodological decision remain?

Yes, on two separate axes, neither resolved by this phase:
- **Missing-FC handling**: Strategy A, B, or C (unchanged from Phase 5B/5C — not re-decided here).
- **Feature-selection target**: fixed-1,000 (Phase 5C, Approach B) vs. all-non-zero (this phase, Approach A) — a new axis this phase adds evidence for, without choosing between them. `strategy_ab_comparison.csv` lays out the trade-offs without ranking.

---

**PHASE 5D STATUS: PASS**

**PHASE 5: BLOCKED — human decisions on missing-FC handling (A/B/C) and feature-selection target (fixed-1000 vs. all-non-zero) both remain required.**

**PHASE 6: NOT STARTED.**
