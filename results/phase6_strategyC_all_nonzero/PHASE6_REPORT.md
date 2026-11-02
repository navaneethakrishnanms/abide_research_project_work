# Phase 6 Report — Real ABIDE Modeling (Strategy C + All-Non-Zero Elastic Net)



Folds completed: 5/5

Per-fold N (non-zero Elastic Net features): [498, 479, 427, 463, 454]

min=427 max=498 mean=464.2 median=463.0 std=23.9



## Evaluation (per fold)

|   fold |   n_test |   accuracy |   balanced_accuracy |   precision |   sensitivity_recall |   specificity |       f1 |   roc_auc |
|-------:|---------:|-----------:|--------------------:|------------:|---------------------:|--------------:|---------:|----------:|
|      0 |      156 |   0.647436 |            0.645153 |    0.611111 |             0.619718 |      0.670588 | 0.615385 |  0.691135 |
|      1 |      156 |   0.589744 |            0.581395 |    0.546875 |             0.5      |      0.662791 | 0.522388 |  0.669601 |
|      2 |      156 |   0.608974 |            0.600166 |    0.571429 |             0.514286 |      0.686047 | 0.541353 |  0.686711 |
|      3 |      156 |   0.634615 |            0.630066 |    0.594203 |             0.585714 |      0.674419 | 0.589928 |  0.699252 |
|      4 |      156 |   0.596154 |            0.591196 |    0.550725 |             0.542857 |      0.639535 | 0.546763 |  0.638704 |



## SSAE architecture per fold

|   fold |   input_dim_N |   hidden1 |   hidden2 |   output | activation                                            | optimizer   |   learning_rate |   weight_decay |   batch_size |   pretrain_epochs_per_layer |   finetune_epochs |   sparsity_weight |   random_seed |
|-------:|--------------:|----------:|----------:|---------:|:------------------------------------------------------|:------------|----------------:|---------------:|-------------:|----------------------------:|------------------:|------------------:|--------------:|
|      0 |           498 |       500 |       100 |        2 | ReLU (hidden), Softmax (output, via CrossEntropyLoss) | Adam        |           0.001 |         0.0001 |           32 |                          40 |                80 |            0.0001 |            42 |
|      1 |           479 |       500 |       100 |        2 | ReLU (hidden), Softmax (output, via CrossEntropyLoss) | Adam        |           0.001 |         0.0001 |           32 |                          40 |                80 |            0.0001 |            42 |
|      2 |           427 |       500 |       100 |        2 | ReLU (hidden), Softmax (output, via CrossEntropyLoss) | Adam        |           0.001 |         0.0001 |           32 |                          40 |                80 |            0.0001 |            42 |
|      3 |           463 |       500 |       100 |        2 | ReLU (hidden), Softmax (output, via CrossEntropyLoss) | Adam        |           0.001 |         0.0001 |           32 |                          40 |                80 |            0.0001 |            42 |
|      4 |           454 |       500 |       100 |        2 | ReLU (hidden), Softmax (output, via CrossEntropyLoss) | Adam        |           0.001 |         0.0001 |           32 |                          40 |                80 |            0.0001 |            42 |



## Problems encountered

None.



## Paper comparison

This is a MODIFIED REPRODUCTION, not an exact reproduction of the original paper. Differences: (1) Strategy C fold-scoped imputation (paper does not specify this handling), (2) Elastic Net instead of SVM-RFE, (3) ALL non-zero Elastic Net features instead of a fixed 1,000, (4) variable SSAE input dimension N per fold instead of a fixed 1,000. No claim is made that this reproduces or targets the paper's reported accuracy.