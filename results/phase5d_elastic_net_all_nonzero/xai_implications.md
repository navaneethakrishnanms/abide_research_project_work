# XAI Compatibility — All-Non-Zero Elastic Net Features

**No XAI method was executed. This is a forward-looking compatibility analysis only.**

## General principle

Every XAI method listed operates on whatever input the trained model actually has. If a feature was
eliminated by Elastic Net in a particular fold (coefficient = 0, therefore never passed to that
fold's SSAE), that feature structurally **cannot** receive an SSAE-level attribution in that fold —
there is no input unit for it to attribute to. This is true regardless of which XAI method is used.

## Per-method notes

- **SHAP / GradientShap / DeepLiftShap**: these attribute each present input feature relative to a
  background/reference distribution. Fully compatible with a variable-N input, per fold — the
  background distribution would itself need to be defined over that fold's N_k-dimensional space.
- **Integrated Gradients**: requires a baseline input and integrates gradients along a path to the
  actual input; compatible with any fixed N_k within a fold, no cross-fold constraint.
- **DeepLift**: reference-based, same consideration as DeepLiftShap — compatible per fold.
- **GuidedBackprop**: operates on gradients through the network; compatible per fold, same N_k
  constraint.
- **LIME**: fits a local surrogate model per prediction using the model's actual input features;
  compatible per fold, though the surrogate's own feature count matches N_k.

## Consequence of fold-specific feature sets

None of these methods has a cross-fold aggregation mechanism built in — combining attributions across
5 folds with different N_k and potentially different feature identities requires an explicit mapping
step: attributions must be indexed by the fixed global `feature_index` (Phase 5A's convention, present
regardless of whether a feature was selected in a given fold), with missing/not-applicable entries for
folds where that feature was not selected at all (structurally absent, not just attributed as zero-
importance — these are different things and must not be conflated in reporting). A feature attributed
highly in the 2 folds where it was selected is a different claim than a feature attributed highly in
all 5 folds.

## Not executed here

No SHAP, Integrated Gradients, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop, or LIME
computation was performed on any real or synthetic model output in this phase.
