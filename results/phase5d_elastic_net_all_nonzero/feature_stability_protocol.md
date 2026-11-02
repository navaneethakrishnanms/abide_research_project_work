# Feature Stability Protocol

**No real ABIDE feature stability was computed. The table below uses SYNTHETIC data
(400 synthetic features, 5-fold synthetic CV) purely to illustrate the mechanic.**

## Required per-fold record

For every CV fold, for every feature considered by that fold's Elastic Net fit, record:

| Column | Meaning |
|---|---|
| fold | CV fold index (0-4 for 5-fold) |
| feature_index | 1-6670, per the fixed convention established in Phase 5A (`fc_feature_index.csv`) |
| roi_i, roi_j | 1-indexed AAL ROI numbers for this feature |
| elastic_net_coefficient | signed fitted coefficient for this fold's model |
| absolute_coefficient | `abs(elastic_net_coefficient)` |
| selected | `elastic_net_coefficient != 0` (boolean) |

## Selection frequency

`selection_frequency(feature) = (number of folds where selected) / (total folds)`.

## SYNTHETIC illustration (400 synthetic features, 5 synthetic folds, NOT real ABIDE features)

Top 10 synthetic features by selection frequency this run:

```
 synthetic_feature_index  fold_0_selected  fold_1_selected  fold_2_selected  fold_3_selected  fold_4_selected  selection_frequency
                      48                1                1                1                1                1                  1.0
                       9                1                1                1                1                1                  1.0
                      24                1                1                1                1                1                  1.0
                      28                1                1                1                1                1                  1.0
                      75                1                1                1                1                1                  1.0
                      31                1                1                1                1                1                  1.0
                      60                1                1                1                1                1                  1.0
                      53                1                1                1                1                1                  1.0
                     221                1                1                1                1                1                  1.0
                      52                1                1                1                1                1                  1.0
```

## Why selection frequency matters

A feature selected in only 1 of 5 folds may reflect noise or an idiosyncrasy of that fold's specific
training subjects, rather than a robust signal — Elastic Net's data-dependence (§ above) means this is
expected to happen for borderline features, especially within correlated blocks where the L2 term can
cause different-but-correlated group members to be selected in different folds. A feature selected in
5 of 5 folds is a stronger (though still not sufficient on its own) candidate for further scrutiny via
XAI/ROAR/anatomical interpretation (§9-11).

## Worked example format (per the task's instruction — illustrative only, not a real result)

```
Feature FC_1523:
  Fold 1 = selected
  Fold 2 = selected
  Fold 3 = not selected
  Fold 4 = selected
  Fold 5 = selected
  Selection frequency = 80%
```

This is a format example only — no such feature/result exists yet for the real ABIDE data.
