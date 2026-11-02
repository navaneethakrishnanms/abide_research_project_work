# Phase 9 Results

## Final locked configuration

final_experiment_id  elastic_net_C  elastic_net_l1_ratio  ssae_h1  ssae_h2  dropout  weight_decay  learning_rate  batch_size early_stopping_patience class_weight  pretrain_epochs  finetune_epochs_max  sparsity_weight  primary_random_seed                                                                                                                                                                       selection_criterion
 EXP_9_FINAL_LOCKED           0.05                   0.5      128       32      0.2          0.01          0.001          32                    None     balanced               40                   80           0.0001                   42 Primary: mean validation ROC-AUC across 5 outer folds. Secondary: mean validation balanced accuracy. Tie-break (within 0.005 ROC-AUC): prefer the simpler/more-regularized configuration.

## Phase 6 baseline vs. Phase 9 locked model (mean +/- SD across 5 outer folds, seed=42)

            metric  phase6_baseline_mean  phase6_baseline_sd  phase9_locked_mean  phase9_locked_sd  difference_mean
          accuracy              0.615385            0.024827            0.648718          0.024577         0.033333
 balanced_accuracy              0.609595            0.026954            0.643843          0.025411         0.034247
sensitivity_recall              0.552515            0.049827            0.595292          0.047215         0.042777
       specificity              0.666676            0.017336            0.692394          0.037417         0.025718
                f1              0.563163            0.038245            0.603434          0.033736         0.040271
           roc_auc              0.677081            0.024034            0.687999          0.040411         0.010918

## Fold-level differences (Phase 9 - Phase 6)

 fold  phase6_accuracy  phase9_accuracy  accuracy_diff  phase6_balanced_accuracy  phase9_balanced_accuracy  balanced_accuracy_diff  phase6_roc_auc  phase9_roc_auc  roc_auc_diff
    0         0.647436         0.692308       0.044872                  0.645153                  0.688650                0.043496        0.691135        0.749461      0.058326
    1         0.589744         0.641026       0.051282                  0.581395                  0.630565                0.049169        0.669601        0.671595      0.001993
    2         0.608974         0.634615       0.025641                  0.600166                  0.628738                0.028571        0.686711        0.682226     -0.004485
    3         0.634615         0.641026       0.006410                  0.630066                  0.639867                0.009801        0.699252        0.697342     -0.001910
    4         0.596154         0.634615       0.038462                  0.591196                  0.631395                0.040199        0.638704        0.639369      0.000664

Accuracy improved in 5/5 outer folds. ROC-AUC improved overall: True.
Balanced accuracy improved overall: True. With only 5 outer folds, statistical power is
limited; no significance test is manufactured from these paired differences -- the fold-level table is
reported in full so the reader can judge consistency directly.

## Multi-seed stability of the locked configuration

 random_seed  mean_test_accuracy  sd_test_accuracy  mean_test_balanced_accuracy  mean_test_ROC_AUC  sd_test_ROC_AUC
          42            0.648718          0.024577                     0.643843           0.687999         0.040411
         123            0.638462          0.028524                     0.634295           0.687539         0.036971
        2026            0.637179          0.051001                     0.631518           0.687539         0.032531

## Simpler-classifier comparators (reported, not used to replace the SSAE track)

                    validation_ROC_AUC  test_accuracy  test_ROC_AUC
classifier                                                         
LinearSVM                     0.986630       0.628205      0.675948
LogisticRegression            0.991026       0.644872      0.686508
