# Data integrity diagnostics (existing 600-subject development cohort; no new models)
## 1. IDs and labels
- unique subject IDs in manifest: **True**
- n subjects: **600**
- labels consistent with diagnosis (y == ASD): **True**
- ASD / TC: **268 / 332**
- sites: **19**
- age/sex/FD found for all dev subjects: **True**
- each of 8 candidates: 600 unique OOF subjects, labels and folds equal the manifest: **True**

## 2. Class counts by site, fold, source
Dataset source = ABIDE acquisition site (19 in the development set).

```
            n  asd  tc  asd_pct                            flag
site                                                           
NYU       133   57  76     42.9                                
UM_1       62   25  37     40.3                                
USM        48   29  19     60.4                                
UCLA_1     44   23  21     52.3                                
PITT       33   16  17     48.5                                
TRINITY    33   15  18     45.5                                
MAX_MUN    31   12  19     38.7                                
YALE       27   14  13     51.9                                
UM_2       24    9  15     37.5                                
KKI        23    6  17     26.1                                
LEUVEN_1   22   11  11     50.0                                
LEUVEN_2   21    8  13     38.1                                
SDSU       20    6  14     30.0                                
STANFORD   19    9  10     47.4                    small (n<20)
SBL        18   10   8     55.6                    small (n<20)
OLIN       16    8   8     50.0                    small (n<20)
UCLA_2     12    5   7     41.7                    small (n<20)
CALTECH    12    4   8     33.3  small (n<20) minority class <5
CMU         2    1   1     50.0  small (n<20) minority class <5
```
ASD prevalence differs across sites: chi2=15.6, dof=18, p=0.622 (expected counts <5 in 2 cells, so the p-value is approximate). Range 26%-60%.

Outer folds:
```
              n  asd  asd_pct
outer_fold                   
0           120   54     45.0
1           120   54     45.0
2           120   54     45.0
3           120   53     44.2
4           120   53     44.2
```
Smallest site-by-fold cell: 0; sites missing from some test folds: 2.

### Possible confounds (descriptive, development subjects, ASD vs TC)
- age: ASD mean 17.42 (SD 8.00), TC mean 16.90 (SD 6.65); Mann-Whitney p=0.981
- func_mean_fd: ASD mean 0.09 (SD 0.05), TC mean 0.07 (SD 0.04); Mann-Whitney p=5.33e-05
- sex counts (rows TC/ASD): [[263, 69], [231, 37]]; chi2 p=0.034

## 3. Out-of-fold performance (saved predictions)
```
 atlas     model  tn  fp  fn  tp  accuracy  acc_ci95_lo  acc_ci95_hi  sensitivity  specificity  balanced_accuracy   auc
AAL116     L1_LR 214 118 101 167     0.635        0.596        0.673        0.623        0.645              0.634 0.653
AAL116     LR_L2 223 109 106 162     0.642        0.603        0.679        0.604        0.672              0.638 0.694
AAL116 LinearSVM 208 124  91 177     0.642        0.603        0.679        0.660        0.627              0.643 0.694
AAL116   RBF_SVM 237  95 123 145     0.637        0.597        0.674        0.541        0.714              0.627 0.698
 CC200     L1_LR 211 121 111 157     0.613        0.574        0.651        0.586        0.636              0.611 0.647
 CC200     LR_L2 229 103 112 156     0.642        0.603        0.679        0.582        0.690              0.636 0.689
 CC200 LinearSVM 213 119  97 171     0.640        0.601        0.677        0.638        0.642              0.640 0.686
 CC200   RBF_SVM 236  96 115 153     0.648        0.609        0.685        0.571        0.711              0.641 0.700
```
Per-fold accuracy:
```
scope             fold0  fold1  fold2  fold3  fold4
atlas  model                                       
AAL116 L1_LR      0.592  0.658  0.625  0.642  0.658
       LR_L2      0.608  0.658  0.650  0.592  0.700
       LinearSVM  0.583  0.650  0.667  0.600  0.708
       RBF_SVM    0.592  0.658  0.667  0.600  0.667
CC200  L1_LR      0.575  0.617  0.592  0.650  0.633
       LR_L2      0.575  0.658  0.700  0.617  0.658
       LinearSVM  0.583  0.625  0.717  0.617  0.658
       RBF_SVM    0.592  0.667  0.608  0.625  0.750
```
Inner-CV AUC (selected setting) vs outer AUC:
```
                  inner_auc  outer_auc  optimism
atlas  model                                    
AAL116 L1_LR          0.636      0.653    -0.017
       LR_L2          0.685      0.694    -0.010
       LinearSVM      0.685      0.694    -0.010
       RBF_SVM        0.691      0.698    -0.007
CC200  L1_LR          0.657      0.647     0.009
       LR_L2          0.693      0.689     0.004
       LinearSVM      0.695      0.686     0.009
       RBF_SVM        0.691      0.700    -0.009
```
Fold means over all 8 candidates: fold0=0.588, fold1=0.649, fold2=0.653, fold3=0.618, fold4=0.679

## 4. Errors by site (average over the 4 models of each atlas; sites with n<20 are noisy)
```
              n  acc_AAL116  acc_CC200
group                                 
NYU       133.0       0.690      0.648
UM_1       62.0       0.565      0.653
USM        48.0       0.734      0.745
UCLA_1     44.0       0.608      0.608
PITT       33.0       0.682      0.712
TRINITY    33.0       0.621      0.568
MAX_MUN    31.0       0.508      0.460
YALE       27.0       0.731      0.731
UM_2       24.0       0.698      0.792
KKI        23.0       0.522      0.522
LEUVEN_1   22.0       0.466      0.523
LEUVEN_2   21.0       0.643      0.690
SDSU       20.0       0.575      0.588
STANFORD   19.0       0.566      0.526
SBL        18.0       0.639      0.542
OLIN       16.0       0.750      0.609
UCLA_2     12.0       0.688      0.771
CALTECH    12.0       0.708      0.604
CMU         2.0       0.500      0.625
```
Consistency across atlases (sites with n>=20, k=13): Spearman rho of per-site accuracy = 0.87 (p=0.00012).
Fold-level error rate, AAL116 vs CC200: Spearman rho = 0.90 (k=5, very low power).
Do error rates differ by site beyond chance? (sites with >=10 subjects, 8 model-runs pooled; not independent, so p is anti-conservative) chi2=114.9, dof=17, p=1.42e-16.

## 5. Do the models fail on the same subjects?
Mean error rate 0.363. Subjects misclassified by all 8 candidate runs: **78** (expected 0.2 if errors were independent); by none: **211**.
Distribution of 'number of runs (of 8) that misclassify the subject':
```
n_wrong
0    211
1     76
2     47
3     35
4     47
5     33
6     22
7     51
8     78
```
Consistently wrong (>=7/8) n=129: ASD share 0.52, mean age 16.5, mean FD 0.083. Consistently right (<=1/8) n=287: ASD share 0.38, mean age 17.5, mean FD 0.080. Age Mann-Whitney p=0.0468; FD p=0.426.

### Motion and age as descriptive single-variable rank statistics (all 600 subjects, not cross-validated, no model)
AUC-equivalent of mean framewise displacement for ASD vs TC = 0.596; of age = 0.501 (0.5 = no separation). Compare with the models' out-of-fold AUC of about 0.65-0.70.
Spearman correlation of saved out-of-fold scores with mean FD (scores are per-model scales; within-fold models differ):
```
 atlas     model  rho_all  rho_within_ASD  rho_within_TC
AAL116     L1_LR    0.167           0.130          0.143
AAL116     LR_L2    0.200           0.201          0.134
AAL116 LinearSVM    0.199           0.194          0.136
AAL116   RBF_SVM    0.219           0.207          0.162
 CC200     L1_LR    0.192           0.187          0.124
 CC200     LR_L2    0.211           0.172          0.168
 CC200 LinearSVM    0.234           0.208          0.181
 CC200   RBF_SVM    0.277           0.271          0.215
```

## 6. Features and saved feature selection
Selected k (SelectKBest) per outer fold, as saved:
```
fold                 0     1     2     3     4
atlas  model                                  
AAL116 L1_LR      2000  2000  2000  2000  2000
       LR_L2      2000   all  2000  2000  2000
       LinearSVM  2000   all  2000  2000  2000
       RBF_SVM    2000  2000  2000  2000   500
CC200  L1_LR      2000  2000  2000  2000  2000
       LR_L2       all   all   all  2000   all
       LinearSVM   all   all   all  2000   all
       RBF_SVM    2000  2000  2000  2000  2000
```
Phase 6 (780-cohort) Elastic Net stability, as saved:
```
                      n_folds_selected  n_features
                                     0        5384
                                     1         731
                                     2         268
                                     3         144
                                     4          93
                                     5          50
   total_unique_selected_at_least_once        1286
total_selected_in_every_completed_fold          50
```
- AAL116: matrix (600, 6670), features with any NaN 2014 (30.2%), near-constant features 0; components for 50/80/90% variance (unsupervised, all 600 rows): 80/262/380 (n=600 subjects, max rank 600); among 1,124,250 random feature pairs, |r|>0.5: 0.07%, |r|>0.8: 0.001%.
- CC200: matrix (600, 19900), features with any NaN 3790 (19.0%), near-constant features 0; components for 50/80/90% variance (unsupervised, all 600 rows): 106/311/426 (n=600 subjects, max rank 600); among 1,124,250 random feature pairs, |r|>0.5: 0.04%, |r|>0.8: 0.001%.
These matrix checks use no labels and only describe redundancy; no feature was selected.

## 7. Leakage / independence review (from recorded files only)
- Fast experiment and site-centering audits record: dev set = train list (676) INTERSECT 780 cohort = 600; folds identical across runs; all learned steps inside Pipelines; 0 of 311 earlier-phase hashes changed.
- **The unopened holdout file does not by itself prove independence.** Recorded facts: (a) 780 - 600 = 180 cohort subjects are not in the train list; the holdout has 170 subjects, so most of those 180 could be holdout subjects. Phases 6-10 evaluated and tuned on all 780, so the earlier phases may have used holdout subjects' data, and their design decisions (imputation, Elastic Net settings, architectures) may have been shaped partly by them. I did not read the holdout IDs to check, so this is **unverified**.
- (b) `holdout_guard.py` documents that an older 170-subject holdout was evaluated repeatedly in early development (a holdout-reuse problem); the current split was created afterwards (2026-09-09) as a fresh split. Whether current development choices were informed by holdout data is not demonstrable from these files.
- (c) The dev-set CV scores here are estimates on development subjects, with candidate selection not nested; they are not independent test results.
