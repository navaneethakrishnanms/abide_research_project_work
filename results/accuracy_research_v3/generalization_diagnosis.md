# Experiment 2: generalization diagnosis (descriptive; saved predictions; seed 42)

## BASELINE (Fisher-Z Pearson)  (seed 42, five-fold OOF, n=600)
Accuracy 0.642; sensitivity 0.604; specificity 0.672; predicted-ASD share 0.452 (true 0.447).
By sex and class:
```
                 n  accuracy
sex    y_true               
female 0        69     0.739
       1        37     0.622
male   0       263     0.654
       1       231     0.602
```
By age tercile and class:
```
                n  accuracy
age_t y_true               
young 0       107     0.505
      1        95     0.642
mid   0       118     0.729
      1        81     0.556
old   0       107     0.776
      1        92     0.609
```
By mean-FD tercile and class:
```
                  n  accuracy
fd_t    y_true               
low FD  0       134     0.739
        1        66     0.470
mid FD  0       103     0.641
        1        97     0.588
high FD 0        95     0.611
        1       105     0.705
```
Error vs sex chi2 p=0.221; Spearman(error, age) rho=-0.097 (p=0.017); Spearman(error, FD) rho=0.011 (p=0.797).
Pooled AUC 0.692; mean within-site AUC over 13 sites with n >= 20: 0.665 (range 0.48-0.85); share of score variance explained by site 0.079 vs by true class 0.113. Across the 13 larger sites, the site mean score correlates with the site's true ASD share (Spearman 0.16, p=0.59), with site mean FD (0.23, p=0.46) and with site mean age (-0.23, p=0.46).
Site table (n >= 20): true vs predicted ASD share and mean score:
```
            n  true_asd  pred_asd  mean_score  mean_age  mean_fd  female
site                                                                    
KKI        23     0.261     0.478       0.495    10.170    0.088   0.304
LEUVEN_1   22     0.500     0.182       0.317    22.409    0.083   0.000
LEUVEN_2   21     0.381     0.333       0.404    14.210    0.085   0.286
MAX_MUN    31     0.387     0.387       0.456    25.968    0.074   0.097
NYU       133     0.429     0.436       0.477    15.753    0.064   0.195
PITT       33     0.485     0.364       0.398    19.271    0.119   0.182
SDSU       20     0.300     0.450       0.435    14.214    0.068   0.300
TRINITY    33     0.455     0.545       0.497    16.888    0.086   0.000
UCLA_1     44     0.523     0.364       0.459    13.573    0.072   0.091
UM_1       62     0.403     0.532       0.494    14.037    0.075   0.306
UM_2       24     0.375     0.292       0.434    16.642    0.067   0.042
USM        48     0.604     0.583       0.578    23.208    0.098   0.000
YALE       27     0.519     0.481       0.491    13.108    0.084   0.444
```

## TANGENT (Ledoit-Wolf)  (seed 42, five-fold OOF, n=600)
Accuracy 0.645; sensitivity 0.578; specificity 0.699; predicted-ASD share 0.425 (true 0.447).
By sex and class:
```
                 n  accuracy
sex    y_true               
female 0        69     0.783
       1        37     0.541
male   0       263     0.677
       1       231     0.584
```
By age tercile and class:
```
                n  accuracy
age_t y_true               
young 0       107     0.495
      1        95     0.674
mid   0       118     0.831
      1        81     0.506
old   0       107     0.757
      1        92     0.543
```
By mean-FD tercile and class:
```
                  n  accuracy
fd_t    y_true               
low FD  0       134     0.754
        1        66     0.379
mid FD  0       103     0.709
        1        97     0.588
high FD 0        95     0.611
        1       105     0.695
```
Error vs sex chi2 p=0.251; Spearman(error, age) rho=-0.062 (p=0.128); Spearman(error, FD) rho=-0.003 (p=0.934).
Pooled AUC 0.686; mean within-site AUC over 13 sites with n >= 20: 0.660 (range 0.46-0.78); share of score variance explained by site 0.056 vs by true class 0.084. Across the 13 larger sites, the site mean score correlates with the site's true ASD share (Spearman 0.45, p=0.13), with site mean FD (0.07, p=0.82) and with site mean age (-0.02, p=0.94).
Site table (n >= 20): true vs predicted ASD share and mean score:
```
            n  true_asd  pred_asd  mean_score  mean_age  mean_fd  female
site                                                                    
KKI        23     0.261     0.435       0.453    10.170    0.088   0.304
LEUVEN_1   22     0.500     0.227       0.382    22.409    0.083   0.000
LEUVEN_2   21     0.381     0.143       0.405    14.210    0.085   0.286
MAX_MUN    31     0.387     0.484       0.472    25.968    0.074   0.097
NYU       133     0.429     0.444       0.470    15.753    0.064   0.195
PITT       33     0.485     0.364       0.426    19.271    0.119   0.182
SDSU       20     0.300     0.300       0.434    14.214    0.068   0.300
TRINITY    33     0.455     0.515       0.544    16.888    0.086   0.000
UCLA_1     44     0.523     0.432       0.510    13.573    0.072   0.091
UM_1       62     0.403     0.419       0.472    14.037    0.075   0.306
UM_2       24     0.375     0.250       0.444    16.642    0.067   0.042
USM        48     0.604     0.521       0.522    23.208    0.098   0.000
YALE       27     0.519     0.481       0.502    13.108    0.084   0.444
```
