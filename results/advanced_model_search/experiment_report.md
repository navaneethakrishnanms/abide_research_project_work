# CPM (connectome-based predictive modeling) on static AAL-116 -- nested CV, 600 development subjects
Pre-specified single candidate; frozen reference = saved AAL-116 L2 logistic regression (read from `fast_accuracy_experiment`). Edge selection, network strengths, scaling and the logistic model are fitted on training rows only; the edge p-threshold ([0.05, 0.01, 0.005, 0.001]) is chosen by inner 3-fold ROC-AUC. Same five outer folds. Holdout, CC200 and the restricted ID file not used. Cross-validation estimates on the development cohort, not independent-test results.

| Method | Accuracy | Bal. acc. | Sens. | Spec. | Prec. | F1 | AUC |
|---|---|---|---|---|---|---|---|
| BASELINE_AAL116_LR_L2 (frozen) | 0.6417 +/- 0.0429 | 0.6378 | 0.6042 | 0.6715 | 0.6010 | 0.6010 | 0.6944 +/- 0.0373 |
| CPM_AAL116 | 0.6317 +/- 0.0819 | 0.6306 | 0.6231 | 0.6382 | 0.5894 | 0.6039 | 0.6680 +/- 0.0637 |

Paired fold-level differences, CPM minus frozen baseline: accuracy -0.0100 +/- 0.0439 (per fold [-0.075, 0.017, -0.017, -0.017, 0.042], improved in 2/5); AUC -0.0264 +/- 0.0330 (per fold [-0.034, -0.035, -0.036, -0.058, 0.03], improved in 1/5); balanced accuracy -0.0072.

Per-fold CPM: selected p-threshold [0.05, 0.005, 0.01, 0.005, 0.01]; positive/negative edges [(663, 551), (148, 103), (228, 162), (144, 147), (155, 123)]; inner AUC [0.684, 0.692, 0.642, 0.668, 0.641] (outer AUC [0.607, 0.68, 0.666, 0.619, 0.768]).
Edge-set stability across outer folds (mean pairwise Jaccard): positive edges 0.305, negative edges 0.253.

## Generalization checks (descriptive, CPM out-of-fold predictions)
Accuracy by site (n<20 flagged noisy): NYU 0.65 (n=133), UM_1 0.56 (n=62), USM 0.71 (n=48), UCLA_1 0.64 (n=44), PITT 0.67 (n=33), TRINITY 0.58 (n=33), MAX_MUN 0.48 (n=31), YALE 0.78 (n=27), UM_2 0.75 (n=24), KKI 0.48 (n=23), LEUVEN_1 0.50 (n=22), LEUVEN_2 0.67 (n=21), SDSU 0.60 (n=20), STANFORD 0.53 (n=19*), SBL 0.78 (n=18*), OLIN 0.81 (n=16*), UCLA_2 0.58 (n=12*), CALTECH 0.58 (n=12*), CMU 0.50 (n=2*)

Spearman correlation of CPM scores with mean FD: 0.252 (p=3.6e-10); accuracy by sex and diagnosis: female/TC 0.72 (n=69); female/ASD 0.62 (n=37); male/TC 0.62 (n=263); male/ASD 0.62 (n=231).

**Conclusion (rule fixed before running: improvement needs >= +0.02 mean accuracy AND AUC with >= 4/5 folds improved; worse needs <= -0.02 on both): NO RELIABLE DIFFERENCE from the baseline (inconclusive).**

No significance test is claimed from 5 folds; no independent-test or 98%/85% claim. The frozen baseline (about 64.2% accuracy, AUC 0.694) is unchanged. Limitations: one pre-specified method, threshold grid of four values, 600 subjects, site and motion effects not adjusted here, method family chosen after inspecting earlier results.
Runtime 0s.
