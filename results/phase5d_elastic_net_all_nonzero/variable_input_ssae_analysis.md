# Variable-Input SSAE Analysis

**No SSAE was trained. This is a technical-feasibility analysis only.**

## Technical validity

A neural network's input layer is simply a weight matrix of shape `(N, hidden_1)`, where `N` is the
input dimensionality. There is nothing in standard deep-learning frameworks that requires `N` to be
fixed across independently-trained model instances. In a proper nested/outer CV design, **each fold
already trains its own independent model instance** (no weights are shared or carried over between
folds) — so allowing each fold's SSAE to have a different `N` (matching that fold's Elastic-Net-
selected feature count) requires no special accommodation: it is simply instantiating 5 separate
models, each with its own input-layer shape, exactly as 5 separate models would be instantiated even
under the fixed-1000 approach (just with `N=1000` held constant across all 5 in that case).

## Architecture

```
Fold k:  N_k -> 500 -> 100 -> 2 -> Softmax
```

where `N_k` = the number of non-zero Elastic-Net-selected features in fold *k*'s training data. The
downstream structure (`500 -> 100 -> 2 -> Softmax`) is held fixed across all folds; only the input
layer's size varies.

## Example (per the task's instruction — illustrative only, NOT real results)

```
Fold 1: 2,347 -> 500 -> 100 -> 2 -> Softmax
Fold 2: 2,518 -> 500 -> 100 -> 2 -> Softmax
Fold 3: 2,287 -> 500 -> 100 -> 2 -> Softmax
Fold 4: 2,462 -> 500 -> 100 -> 2 -> Softmax
Fold 5: 2,399 -> 500 -> 100 -> 2 -> Softmax
```

These numbers are illustrative only, matching the task's own example, and are not derived from any
real computation in this phase.

## Reproducibility implication

Because `N_k` is fully determined by (training-fold data, `alpha`, `l1_ratio`, random_state), the
per-fold architecture is deterministic and reproducible given those fixed inputs — the variability is
data-dependent, not random or ad hoc.

## Practical consequence: no single shared feature-index list

Under the fixed-1000 approach, one could (if desired) report a single canonical "the 1000 selected
features" list synthesized somehow across folds. Under all-non-zero, there is no single canonical
list by construction — the honest unit of reporting is per-fold selected-feature sets plus the
cross-fold stability analysis (`feature_stability_protocol.md`), not a single merged list presented
as *the* feature set.

## Conclusion

**Yes — `N -> 500 -> 100 -> 2 -> Softmax` with fold-varying `N` is technically valid and
reproducible.** It requires no framework accommodation beyond instantiating a fresh model per fold
(already required for leakage-safe CV regardless of which feature-selection approach is used).
