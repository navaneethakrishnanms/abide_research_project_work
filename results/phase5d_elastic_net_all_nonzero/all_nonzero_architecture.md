# All-Non-Zero Elastic Net Architecture — Methodology Analysis

**No real ABIDE Elastic Net fitting occurred. All numeric illustrations are SYNTHETIC
(see `elastic_net_hyperparameter_behavior.csv`, `elastic_net_nonzero_feature_analysis.csv`).**

## Requested architecture

```
AAL-116 -> Pearson FC -> Fisher-Z -> Elastic Net -> ALL NON-ZERO FEATURES -> SSAE -> Softmax
```

replacing the fixed-1000 ranking step with direct retention of whatever set of features Elastic Net's
own regularization leaves non-zero.

## Terminology (used consistently throughout this phase's outputs)

- **TOTAL FEATURES**: all 6,670 FC features.
- **NON-ZERO ELASTIC NET FEATURES**: features whose fitted coefficient is non-zero for a specific
  Elastic Net fit (a specific fold, specific `alpha`/`l1_ratio`). This is a property of *that model
  fit*, not an intrinsic property of the feature.
- **ZERO-COEFFICIENT FEATURES**: the complement — coefficient exactly zero for that fit.
- **STABLE FEATURES**: features non-zero across *multiple independent* CV folds — a reproducibility
  signal, not a biological one.

**A non-zero coefficient means the feature was retained by the specified Elastic Net model under its
specific hyperparameters and training data. It does not by itself establish biological importance.**
Biological interpretation requires additional, separate evidence: selection stability across folds,
XAI attribution, ROAR support, and mapping back to a named AAL ROI pair (§9 of `PHASE5D_DECISION.md`).

## Elastic Net objective (for reference)

For a classification target, `LogisticRegression(penalty="elasticnet", solver="saga")` (or an
equivalent formulation) minimizes:

```
loss(y, Xw) + alpha * [ l1_ratio * ||w||_1 + (1 - l1_ratio) * 0.5 * ||w||_2^2 ]
```

- **L1 term** (`||w||_1`): drives coefficients to exactly zero, producing sparsity.
- **L2 term** (`||w||_2^2`): shrinks correlated coefficients toward each other (the "grouping
  effect"), rather than arbitrarily zeroing all but one of a correlated set.
- **`alpha`**: overall regularization strength — SYNTHETIC demonstration confirms larger `alpha`
  (equivalently, smaller `C` in scikit-learn's inverse-strength convention) drives the non-zero count
  down.
- **`l1_ratio`**: L1/L2 balance in [0, 1] — SYNTHETIC demonstration confirms higher `l1_ratio` (more
  L1-like) also drives the non-zero count down, independent of `alpha`.

**Neither hyperparameter targets a specific feature count.** The resulting non-zero count is an
emergent, data-dependent consequence of the fit — this is exactly why the paper's SVM-RFE (which
explicitly, mechanically ranks down to a stated target of 1,000) and Elastic-Net-all-nonzero are
structurally different procedures, not interchangeable implementations of "the same idea."

## Why different folds may produce different selected feature sets

Each CV fold trains on a different subset of subjects. Because Elastic Net's coefficients (which
determines zero vs. non-zero) are a function of the training data, a different training subset can
and, per the SYNTHETIC 5-fold demonstration in this phase's log, does produce a different non-zero
count AND different feature identities, even with identical hyperparameters. This is expected, not a
bug — it is a direct, unavoidable consequence of leakage-safe, fold-scoped fitting (§5).

## Example count illustration (per the task's own instruction — NOT a real result)

```
6,670 total features -> Elastic Net -> [EXAMPLE ONLY] 2,347 non-zero features -> SSAE
```

2,347 is illustrative only. The actual count for any real ABIDE fold can only be determined by
performing the real fit — deferred to a later phase, not this one.
