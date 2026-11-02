# Static AAL-116 diagnostics (600 development subjects; saved predictions; no model trained)
All associations below are descriptive; none establishes a cause.

## 1. Error consistency across the four AAL-116 models
Mean error rate 0.361. Subjects wrong in 0/1/2/3/4 of 4 models: [296, 69, 34, 74, 127] (expected if errors were independent: [np.float64(99.9), np.float64(225.9), np.float64(191.7), np.float64(72.3), np.float64(10.2)]). Wrong in all 4: **127** (ASD share 0.48); right in all 4: 296 (ASD share 0.42; overall 0.45).
- ASD: n=268; wrong in all 4 models: 61 (23%); right in all 4: 123 (46%)
- TC: n=332; wrong in all 4 models: 66 (20%); right in all 4: 173 (52%)

## 2. Performance by site (small groups flagged)
```
    site  n_subjects  accuracy  acc_ci95_lo  acc_ci95_hi  sensitivity  specificity  small_group_n_lt_20
     NYU         133     0.690        0.649        0.728        0.658        0.714                False
    UM_1          62     0.565        0.502        0.625        0.600        0.541                False
     USM          48     0.734        0.668        0.792        0.793        0.645                False
  UCLA_1          44     0.608        0.534        0.677        0.489        0.738                False
    PITT          33     0.682        0.598        0.755        0.531        0.824                False
 TRINITY          33     0.621        0.536        0.699        0.650        0.597                False
 MAX_MUN          31     0.508        0.421        0.594        0.333        0.618                False
    YALE          27     0.731        0.641        0.806        0.750        0.712                False
    UM_2          24     0.698        0.600        0.781        0.556        0.783                False
     KKI          23     0.522        0.421        0.621        0.500        0.529                False
LEUVEN_1          22     0.466        0.365        0.569        0.205        0.727                False
LEUVEN_2          21     0.643        0.536        0.737        0.438        0.769                False
    SDSU          20     0.575        0.466        0.677        0.583        0.571                False
STANFORD          19     0.566        0.454        0.671        0.833        0.325                 True
     SBL          18     0.639        0.524        0.740        0.650        0.625                 True
    OLIN          16     0.750        0.632        0.840        0.781        0.719                 True
  UCLA_2          12     0.688        0.547        0.801        0.800        0.607                 True
 CALTECH          12     0.708        0.568        0.818        0.438        0.844                 True
     CMU           2     0.500        0.215        0.785        0.000        1.000                 True
```
Sites with n>=20 (k=13): accuracy range 0.47-0.73; heterogeneity of error rate across them chi2=54.2, dof=12, p=2.55e-07 (the four models are not independent, so p is anti-conservative; individual-site estimates are wide, e.g. CIs about +/-0.10-0.17).

## 3. Association of scores and errors with mean FD
```
model                            L1_LR  LR_L2  LinearSVM  RBF_SVM
spearman_score_vs_FD_all         0.167    0.2      0.199    0.219
p_all                              0.0    0.0        0.0      0.0
spearman_within_ASD               0.13  0.201      0.194    0.207
spearman_within_TC               0.143  0.134      0.136    0.162
mean_FD_misclassified            0.081  0.082      0.082    0.083
mean_FD_correct                  0.081  0.081      0.081     0.08
mannwhitney_p_FD_err_vs_correct  0.957  0.797      0.902    0.568
accuracy_low_FD                  0.635   0.65      0.645    0.655
accuracy_mid_FD                   0.61  0.615       0.62    0.615
accuracy_high_FD                  0.66   0.66       0.66     0.64
share_predicted_ASD_low_FD       0.385   0.33      0.375    0.305
share_predicted_ASD_high_FD      0.555  0.555      0.595    0.495
```
Mean FD: ASD 0.089, TC 0.075; AUC-equivalent of FD alone for the diagnosis labels = 0.596 (descriptive rank statistic, all 600 subjects). Predicted-ASD share is higher in the high-FD tercile for the models above, i.e. predictions track motion; whether the models use motion or share a correlated signal cannot be told from these data.

## 4. Demographic balance (development subjects)
Sex (1 male, 2 female) by diagnosis (rows TC, ASD): [[263, 69], [231, 37]]; female share TC 0.208, ASD 0.138; chi2 p=0.034. Age: ASD 17.4, TC 16.9 (Mann-Whitney p=0.981).
LR_L2 accuracy by sex and diagnosis:
```
                 n  accuracy
sexl   y_true               
female 0        69     0.739
       1        37     0.622
male   0       263     0.654
       1       231     0.602
```

## 5. Calibration and class behavior (LR_L2 out-of-fold probabilities)
Brier score 0.2224 (always predicting the prevalence 0.447: 0.2472). Mean predicted ASD probability: ASD 0.556, TC 0.410.
```
              n  mean_pred  observed_ASD_rate
score                                        
(0.0, 0.3]  139      0.189              0.245
(0.3, 0.4]   92      0.350              0.304
(0.4, 0.5]   98      0.450              0.449
(0.5, 0.6]   94      0.555              0.521
(0.6, 0.7]   74      0.648              0.486
(0.7, 1.0]  103      0.802              0.748
```
Threshold behavior of the saved scores (descriptive; not for choosing a threshold, which would need inner CV):
```
 threshold  sensitivity  specificity  accuracy  balanced_accuracy
      0.30        0.873        0.316     0.565              0.595
      0.35        0.828        0.422     0.603              0.625
      0.40        0.769        0.509     0.625              0.639
      0.45        0.698        0.602     0.645              0.650
      0.50        0.604        0.672     0.642              0.638
      0.55        0.519        0.729     0.635              0.624
      0.60        0.422        0.807     0.635              0.614
      0.65        0.354        0.864     0.637              0.609
      0.70        0.287        0.922     0.638              0.605
```
Per-fold AUC (all four models): LR_L2: 0.641 0.715 0.702 0.677 0.738; LinearSVM: 0.645 0.712 0.706 0.675 0.734; L1_LR: 0.634 0.655 0.694 0.649 0.634; RBF_SVM: 0.637 0.729 0.711 0.675 0.739

## 6. Data quality (AAL-116, development subjects)
Matrix (600, 6670); undefined (NaN) correlation entries 12764 (0.32%); subjects with any undefined entries 40; features with any NaN 2014 of 6670; max per subject 1314. Undefined entries arise from constant (zero) ROIs (Phase 4.1). Imputation is the fold-scoped training-column mean inside the Pipeline (verified in the script).
Subjects with undefined entries by site:
```
           n  subjects_with_nan
site                           
SBL       18                 14
PITT      33                 11
NYU      133                  4
SDSU      20                  4
CALTECH   12                  3
CMU        2                  1
KKI       23                  1
MAX_MUN   31                  1
TRINITY   33                  1
```
Mean number of the 4 models wrong: subjects with undefined entries 1.25 (n=40) vs without 1.46 (n=560); Mann-Whitney p=0.535.

## 7. Feature-selection stability
Selected-feature indices were not saved for the 600-subject runs, so existing artifacts only give the chosen k per fold. Descriptive substitute: in each outer fold, on TRAINING rows only (fold-scoped mean imputation, Fisher-Z, standardization), rank the 6,670 features by univariate F-score and compare the top-k sets across folds. This is not a model and nothing is selected or tuned from it.
```
 top_k  mean_pairwise_jaccard_across_folds  min_jaccard  max_jaccard  jaccard_expected_if_random_sets  features_in_top_k_of_all_5_folds  features_in_top_k_of_>=4_folds  features_in_top_k_of_>=1_fold
   500                               0.380        0.357        0.403                            0.039                               136                             252                           1073
  2000                               0.523        0.491        0.550                            0.176                               877                            1367                           3468
```
Historical context (Phase 6, 780-subject cohort that overlaps the designated holdout; Elastic Net, all five folds): 1,286 features selected at least once, only 50 in all five folds.
