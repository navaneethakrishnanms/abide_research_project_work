# FINAL REPORT -- Elastic Net + SSAE (500-100-2) on AAL-116, 600 development subjects

Status: **FINAL** for the experiment as run (5 of 5 outer folds, 3 of 3 seeds complete; log shows normal completion; 645 s of the 900 s budget). These are development cross-validation estimates, not independent-holdout performance.

## 1. Project objective
Test whether the specified workflow (Elastic Net feature selection followed by a stacked sparse autoencoder with a 500-neuron hidden layer, a 100-neuron latent layer and a two-class softmax output) improves on the frozen AAL-116 logistic-regression baseline, with the aspirational target of at least 80% accuracy.

## 2. Final workflow and architecture
ABIDE-I (PCP CPAC, band-pass filtered, global-signal regressed) -> quality control (subject-level mean FD <= 0.2 mm, 780-subject cohort; provenance only) -> AAL-116 ROI time series -> Pearson correlation -> Fisher-Z (clip 1e-6) -> 6,670 unique edges -> Elastic Net (training rows only) -> selected edges (147-580 per fold) -> SSAE encoder 500 -> 100 -> two-class output -> softmax -> ASD or typical control.

Verified from the code (`phase10_corrected_nested_cv.py`, imported by the run): encoder 1 maps N inputs to 500 ReLU units, encoder 2 maps 500 to 100 ReLU units, and a 100-to-2 linear classifier follows a dropout layer; softmax is applied through the cross-entropy loss. Two **reconstruction decoders** (100 -> 500 and 500 -> N) exist only for layer-wise pretraining: each layer is first trained as an autoencoder with mean-squared reconstruction loss plus an L1 sparsity penalty on its activations (40 epochs per layer). They are discarded at classification time because reconstruction and diagnosis have different targets. Fine-tuning then trains the encoders and the classifier head with a **supervised class-weighted cross-entropy loss** (up to 80 epochs; the sparsity penalty applies only during pretraining).

## 3. Dataset and evaluation cohort
600 development subjects (268 ASD / 332 TC, 19 sites) = the train list intersected with the 780-subject FD-filtered cohort; the five outer folds (120 subjects each) are those of the fast experiment. The 780-subject cohort was not trained on. The restricted 170-subject holdout, `RESTRICTED_id_membership.csv` and CC200 were not used.

## 4. Implementation details (as actually executed)
- **Imputation / scaling:** training-row column-mean imputation, Fisher-Z, StandardScaler fitted per inner-training split and, for the final model, on the whole outer-training set.
- **Elastic Net:** logistic regression, saga solver, l1_ratio 0.5, class weights balanced; C chosen from {0.03, 0.1, 0.3} by mean inner 3-fold ROC-AUC of the EN model (ties within 0.005 -> smaller C).
- **SSAE selection:** three predeclared settings compared by inner 3-fold AUC (ties within 0.005 -> preference order S2, S3, S1).
- **Class weighting:** balanced for the Elastic Net and the cross-entropy loss. **Early stopping:** patience 10 on the loss of a stratified 15% hold-out drawn from the training rows, with best-state restore; inner-validation and outer-test rows never influenced fitting or stopping. **Decisions:** argmax (no threshold tuning).
- **Seeds:** 42 (predeclared headline), 123, 2026 for the SSAE only (selection run once with seed 42). Single-thread numerical libraries; GPU for the SSAE.

## 5. Experimental configurations
| Setting | Learning rate | Weight decay | Dropout | Sparsity weight |
|---|---|---|---|---|
| S1 original (Phase 6) | 1e-3 | 1e-4 | 0 | 1e-4 |
| S2 regularized | 1e-4 | 5e-2 | 0.5 | 1e-4 |
| S3 regularized + stronger sparsity | 1e-4 | 5e-2 | 0.5 | 1e-3 |

Only the inner-CV-selected setting was trained on each full outer-training set. **Outer-test results for S1, S2 and S3 as separate models do not exist**; only their inner-validation AUCs do (below).

## 6. Execution status and runtime
Normal completion (exit code 0), 645 s total (900 s budget), no failed or timed-out folds, 5 checkpoints valid. Per fold 103-174 s (selection 81-103 s; fold 3, which chose the largest feature set, was slowest). Peak memory was not recorded.

## 7. Results by fold and seed (accuracy / AUC)
| Fold | Frozen baseline | Elastic Net classifier | SSAE seed 42 | SSAE seed 123 | SSAE seed 2026 | Features | EN C |
|---|---|---|---|---|---|---|---|
| 0 | 0.608 / 0.641 | 0.567 / 0.623 | 0.600 / 0.643 | 0.608 / 0.645 | 0.600 / 0.644 | 149 | 0.03 |
| 1 | 0.658 / 0.715 | 0.675 / 0.696 | 0.658 / 0.707 | 0.658 / 0.704 | 0.658 / 0.708 | 154 | 0.03 |
| 2 | 0.650 / 0.702 | 0.625 / 0.685 | 0.633 / 0.698 | 0.642 / 0.704 | 0.633 / 0.694 | 158 | 0.03 |
| 3 | 0.592 / 0.677 | 0.617 / 0.678 | 0.633 / 0.678 | 0.633 / 0.675 | 0.617 / 0.680 | 580 | 0.3 |
| 4 | 0.700 / 0.738 | 0.617 / 0.685 | 0.617 / 0.665 | 0.617 / 0.667 | 0.617 / 0.665 | 147 | 0.03 |

All sensitivity, specificity, precision, F1 and confusion matrices per fold and seed are in `fold_level_results.csv`.

## 8. Model comparison (mean +/- SD over five folds, 600 subjects each)
| Model | Accuracy | Bal. acc. | Sens. | Spec. | Prec. | F1 | AUC |
|---|---|---|---|---|---|---|---|
| Frozen baseline (L2 logistic regression, saved) | 0.642 +/- 0.043 | 0.638 | 0.604 | 0.672 | 0.601 | 0.601 | 0.694 +/- 0.037 |
| Elastic Net classifier (no SSAE) | 0.620 +/- 0.039 | 0.615 | 0.570 | 0.659 | 0.576 | 0.570 | 0.673 +/- 0.029 |
| EN+SSAE seed 42 (predeclared headline) | 0.628 +/- 0.022 | 0.621 | 0.552 | 0.690 | 0.590 | 0.568 | 0.678 +/- 0.026 |
| EN+SSAE seed 123 | 0.632 +/- 0.020 | 0.624 | 0.559 | 0.690 | 0.593 | 0.573 | 0.679 +/- 0.026 |
| EN+SSAE seed 2026 | 0.625 +/- 0.022 | 0.617 | 0.548 | 0.687 | 0.586 | 0.564 | 0.678 +/- 0.025 |
| SSAE S1 / S2 / S3 as separate outer-test models | UNAVAILABLE (not evaluated) | | | | | | |

Inner-validation AUC of the three SSAE settings (mean over folds, selection data only): S1 0.6448, S2 0.6448, S3 0.6448; per fold S1 vs S2 differ by 0.001-0.005 except fold 3 (S2 0.676 vs S1 0.664), so S2 was chosen in all five folds mostly by the predeclared tie-break rule, not by a clear performance difference. S2 and S3 are practically identical (the sparsity weight change had no measurable effect).

## 9. Did SSAE improve the baseline?
**No.** Paired fold-level differences (SSAE seed 42 minus reference):
| Comparison | Accuracy | Folds improved | AUC | Folds improved | Balanced acc. |
|---|---|---|---|---|---|
| SSAE vs frozen baseline | -0.013 +/- 0.045 | 1/5 | -0.017 +/- 0.032 | 2/5 | -0.017 |
| Elastic Net classifier vs frozen baseline | -0.022 +/- 0.044 | 2/5 | -0.021 +/- 0.020 | 1/5 | -0.023 |
| SSAE vs Elastic Net classifier | +0.008 +/- 0.019 | 3/5 | +0.005 +/- 0.016 | 3/5 | +0.006 |

Seeds 123 and 2026 give the same picture (accuracy -0.010 and -0.017, AUC -0.016 and -0.016 against the baseline; accuracy improved in 1/5 and AUC in 2/5 folds each). The SSAE is slightly and inconsistently better than the Elastic Net classifier it is built on (differences of a fraction of a point, inside the fold variation) and slightly worse than the simple logistic regression baseline. It is less variable across folds (SD 0.022 vs 0.043) but not more accurate. The added complexity (about 250,000-680,000 trainable parameters including decoders) is not justified by these results. The three seeds differ by under 0.01 in accuracy.

## 10. Was the 80% target reached?
**No.** Best mean development cross-validation accuracy is 63.2% (SSAE seed 123; the predeclared headline is 62.8%), the best single fold is 70.0% (frozen baseline, fold 4), and no fold or seed approaches 80%. The result is complete and verifiable (confusion counts), is a five-fold development cross-validation estimate, and is not an independent test. There is no evidence in these results supporting 80%.

## 11. Integrity and reproducibility checks
See `integrity_and_reproducibility.md`. Summary: all 5 folds x 3 seeds complete; 20 of 20 checkpoint entries have confusion matrices summing to the manifest fold size (120) with ASD counts equal to the manifest; metrics recompute exactly from the counts; perturbation check passed; 0 of 311 protected files changed. **Not verifiable:** subject-level predictions were not saved, so one-prediction-per-subject can only be inferred from fold sizes, and ROC-AUC cannot be independently recomputed from confusion matrices. `implementation_notes.md` was not produced by the run (the details above come from the code and logs).

## 12. Limitations and methodological caveats
- The regularized SSAE settings (S2/S3) were informed by earlier nested-CV results on the 780-subject cohort, which overlaps the designated holdout. This is a development-history limitation: the run is useful development evidence but is not a pristine, independently designed evaluation. The holdout was not accessed to investigate this.
- Five folds, 120 subjects per fold, one selection seed; no significance test; three Elastic Net strengths and three SSAE settings only.
- Selected Elastic Net features are unstable across folds (mean pairwise Jaccard 0.18; 147-580 features), and the SSAE overfits the training data (final training loss 0.04-0.08 against 0.18-0.31 on the early-stopping hold-out; two folds ran the full 80 epochs).
- Inner-validation AUCs (0.627-0.676) are in line with outer AUCs (0.643-0.707): there is no sign of selection optimism, and the limit is the signal, not tuning.
- Comparisons are on the same 600 subjects and folds only; they are not comparable with results on the 780 cohort or other protocols.

## 13. Conclusion
The specified Elastic Net + SSAE (500-100-2) workflow ran correctly and safely but did not improve on the simple frozen logistic-regression baseline (62.8% vs 64.2% accuracy, AUC 0.678 vs 0.694), added only a fraction of a point over its own Elastic Net feature-selection stage, and fell far short of 80%.

## 14. One recommended next step
Stop tuning accuracy on this cohort and document the baseline. If one more experiment is wanted, run a site-held-out (leave-site-out) evaluation of the frozen baseline: it takes about a minute, uses the same pipeline, and tests the question that random-fold accuracy cannot answer, namely generalization to unseen acquisition sites.
