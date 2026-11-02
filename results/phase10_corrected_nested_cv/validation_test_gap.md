# Validation-test gap (corrected Phase 10)
Inner validation = mean over the 2 inner folds of the SELECTED configuration (max over candidates, so mildly optimistic by selection). Inner-validation rows are scored on data never used for imputation/scaling/Elastic Net/SSAE fitting/early stopping.
Phase 9 reference (seed 42): validation accuracy ~95.5%, validation AUC ~0.994, outer-test accuracy 64.9%, outer-test AUC 0.688.

## ORDINARY nested CV (seed 42)  (5/5 folds)

```
 outer_fold  inner_val_mean_inner_accuracy  accuracy  inner_val_mean_inner_roc_auc  roc_auc  acc_gap  auc_gap
          0                         0.6298    0.6410                        0.6538   0.6969  -0.0112  -0.0431
          1                         0.6651    0.6154                        0.7072   0.6601   0.0497   0.0471
          2                         0.6603    0.6154                        0.7068   0.6774   0.0449   0.0294
          3                         0.6923    0.6795                        0.7343   0.7146   0.0128   0.0197
          4                         0.6667    0.5962                        0.7194   0.6568   0.0705   0.0626
```

Mean inner-val accuracy 0.6628 vs outer-test accuracy 0.6295 (gap +0.0333); mean inner-val AUC 0.7043 vs outer-test AUC 0.6812 (gap +0.0231).

## SITE-GROUPED nested CV  (5/5 folds)

```
 outer_fold  inner_val_mean_inner_accuracy  accuracy  inner_val_mean_inner_roc_auc  roc_auc  acc_gap  auc_gap
          0                         0.6190    0.6857                        0.6532   0.7810  -0.0667  -0.1278
          1                         0.6297    0.6849                        0.6550   0.7295  -0.0552  -0.0745
          2                         0.6507    0.6450                        0.6863   0.6760   0.0057   0.0103
          3                         0.6370    0.6422                        0.6869   0.6808  -0.0052   0.0061
          4                         0.6412    0.6552                        0.6642   0.6998  -0.0140  -0.0356
```

Mean inner-val accuracy 0.6355 vs outer-test accuracy 0.6626 (gap -0.0271); mean inner-val AUC 0.6691 vs outer-test AUC 0.7134 (gap -0.0443).
