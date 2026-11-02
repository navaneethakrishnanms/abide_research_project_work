# Elastic Net as a Replacement for SVM-RFE — Procedure Analysis

**No Elastic Net was fit on real ABIDE FC data in this phase. All numeric illustrations below are
SYNTHETIC (see `elastic_net_feature_count_analysis.csv`).**

## How Elastic Net differs from SVM-RFE

The paper's SVM-RFE recursively fits a linear SVM and removes the lowest-ranked feature(s) each
iteration until reaching 1,000 — a wrapper method that repeatedly retrains a classifier. Elastic Net
is instead a single regularized linear model (here, for a classification target, typically
`LogisticRegression(penalty="elasticnet", solver="saga")` or `SGDClassifier(penalty="elasticnet")`
would replace the plain `ElasticNet` regressor used for illustration below) whose objective combines:

- **L1 regularization** (the Lasso term, weighted by `l1_ratio`): drives some coefficients to exactly
  zero, producing sparsity and an implicit feature-selection effect.
- **L2 regularization** (the Ridge term, weighted by `1 - l1_ratio`): shrinks correlated coefficients
  together rather than arbitrarily picking one and zeroing the rest — this is the well-known "grouping
  effect," and it matters directly here because FC features are structurally correlated (many of the
  6,670 pairs share a common ROI).

The combined penalty is controlled by two hyperparameters: `alpha` (overall regularization strength)
and `l1_ratio` (the L1/L2 balance, in [0, 1]). The SYNTHETIC sparsity grid in
`elastic_net_feature_count_analysis.csv` shows nonzero-feature counts varying substantially across a
small `alpha`/`l1_ratio` grid on a 500-feature synthetic design with 20 correlated blocks — the same
qualitative behavior would be expected on the real 6,670-feature FC representation, though the exact
counts cannot be known without fitting on real data (not done in this phase).

## Correlated FC features and the grouping effect

Because many FC features share an ROI (each of the 116 ROIs participates in 115 of the 6,670
features), Elastic Net's L2 component is expected to retain groups of related features together
rather than collapsing to a single representative per group, unlike pure L1/Lasso. This is a
structural property of the method, not a claim about what will happen on ABIDE data specifically —
demonstrated only in synthetic miniature here.

## Coefficient interpretation

An Elastic Net coefficient's magnitude reflects its (regularized) linear association with the target
within the fitted model — it is not equivalent to SVM-RFE's ranking criterion (which is derived from
the linear SVM's margin/weight at each recursive step). Coefficients from the two methods are not
directly comparable in magnitude or interpretation, even where they might select overlapping features.

## Regularization parameters

`alpha` and `l1_ratio` would need to be selected via inner cross-validation on training folds only
(e.g. `ElasticNetCV`/`LogisticRegressionCV` with `penalty="elasticnet"`), never by observing outer-fold
or holdout test accuracy — consistent with this project's existing leakage-avoidance discipline.

## Leakage requirements

Per the task's explicit requirement and this project's established `holdout_guard.py` discipline:
Elastic Net must be **fit on training-fold data only** within each of the 5 CV folds; its selected
features (whichever selection mechanism, Option 1 or 2) must be determined from training-fold data
only and then applied unchanged to that fold's test data. Elastic Net must never see test-fold labels
or values during fitting or feature-ranking.

## Explicitly not claimed here

This document does not claim Elastic Net will produce higher, lower, or comparable accuracy to
SVM-RFE on the real ABIDE data. No such comparison was performed, attempted, or estimated.
