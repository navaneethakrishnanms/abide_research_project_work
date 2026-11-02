# AAL-116 dynamic vs static connectivity -- exploratory development-set CV
600 development subjects (268 ASD / 332 TC), 19 sites, same outer folds as the fast experiment. Dynamic windows: 50 TRs, step 10; 7-27 windows/subject (mean 15.1). Classifier: L2 logistic regression, grid and inner 3-fold ROC-AUC identical to the saved static AAL-116 baseline (baseline read from `fast_accuracy_experiment`, not recomputed). The only pipeline difference is a training-fold mean imputer in place of the Fisher-Z step (Fisher-Z is applied per window). Holdout, CC200 and the restricted ID file were not used.

| Method | Accuracy | Bal. acc. | Sens. | Spec. | F1 | AUC |
|---|---|---|---|---|---|---|
| DYN_MAC | 0.5250 +/- 0.0460 | 0.5130 | 0.3991 | 0.6269 | 0.4285 | 0.5164 +/- 0.0315 |
| DYN_VAR | 0.4983 +/- 0.0303 | 0.4854 | 0.3621 | 0.6087 | 0.3889 | 0.4996 +/- 0.0495 |
| STATIC_AAL116 (saved baseline) | 0.6417 +/- 0.0429 | 0.6378 | 0.6042 | 0.6715 | 0.6010 | 0.6944 +/- 0.0373 |

Paired fold-level differences vs the static baseline (dynamic minus static):

| Method | Metric | Mean diff | SD | Folds improved |
|---|---|---|---|---|
| DYN_MAC | accuracy | -0.1167 | 0.0601 | 0/5 |
| DYN_MAC | balanced_accuracy | -0.1248 | 0.0575 | 0/5 |
| DYN_MAC | f1 | -0.1724 | 0.0660 | 0/5 |
| DYN_MAC | roc_auc | -0.1780 | 0.0632 | 0/5 |
| DYN_VAR | accuracy | -0.1433 | 0.0538 | 0/5 |
| DYN_VAR | balanced_accuracy | -0.1525 | 0.0542 | 0/5 |
| DYN_VAR | f1 | -0.2121 | 0.0866 | 0/5 |
| DYN_VAR | roc_auc | -0.1948 | 0.0558 | 0/5 |

Per-fold accuracy diffs: DYN_VAR: [-0.092, -0.117, -0.183, -0.108, -0.217]; DYN_MAC: [-0.033, -0.117, -0.1, -0.133, -0.2]

**Conclusion (pre-declared rule: promising needs a >= +0.02 mean accuracy AND AUC gain with >= 4/5 folds improved in accuracy; worse needs <= -0.02 on both for every method): DYNAMIC FEATURES PERFORM WORSE.**

No significance test is claimed from 5 folds; these are development cross-validation estimates, not independent-holdout results; no 85% claim. Runtime 26s.
