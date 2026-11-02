# Next experiment recommendation (not launched)

## What the evidence says
Supported by the saved predictions and manifests (descriptive, development subjects only):
- **No tuning problem is visible.** Inner-CV AUC matches outer AUC (no optimism); the four AAL-116 models have nearly identical per-fold AUC (about 0.64-0.74); calibration is reasonable (Brier 0.222 vs 0.247 for prevalence-only); accuracy is flat (0.635-0.645) for any threshold from 0.45 to 0.70, so the decision threshold is not the limit.
- **Feature filtering is not erratic.** Training-rows-only F-score top-500 sets overlap across folds with Jaccard 0.38 (0.04 expected for random sets; folds share 75% of their training rows, so some overlap is built in).
- **Data quality is a minor issue.** 0.32% of correlations are undefined (40 subjects, mostly SBL and PITT), and those subjects are not misclassified more often (p = 0.54).
- **Errors are shared, as expected for correlated models.** 127 subjects are wrong in all 4 models and 296 are right in all 4 (about 10 and 100 expected if independent). Models that use the same features produce highly correlated scores, so this does not by itself indicate label noise.
- **Site matters.** Accuracy for sites with n >= 20 ranges 0.47-0.73 (heterogeneity p about 3e-7, anti-conservative; single-site intervals are wide, and five sites have fewer than 20 subjects).
- **Scores track motion, errors do not.** Out-of-fold scores correlate with mean FD (Spearman 0.17-0.22, also within each diagnosis group), and the predicted-ASD share rises from about 31-39% in the low-FD tercile to 50-60% in the high-FD tercile. But misclassified and correctly classified subjects have the same FD (p = 0.57-0.96), and accuracy is similar across FD terciles. Mean FD alone has an AUC-equivalent of 0.596 for diagnosis (rank statistic on all 600 subjects).
- Sex is imbalanced by diagnosis (female 21% TC vs 14% ASD, p = 0.034).

Hypotheses (not established): that part of the 64% reflects motion-related signal; that site-specific effects limit generalization; that the features simply carry a weak diagnostic signal. None of these is shown to be causal.

## Single recommended experiment (read-only, no training)
**FD-matched re-evaluation of the existing saved out-of-fold predictions.** Pair each ASD subject with a TC subject of near-identical mean FD (pre-declared caliper, matching within the development cohort only, optionally within site), then compute AUC, accuracy and sensitivity/specificity of the existing AAL-116 OOF scores on the matched subset only, with a bootstrap interval over matched pairs. Compare with the same statistics on random subsets of equal size from the unmatched cohort, so the effect of the smaller sample is separated from the effect of matching.
- **Why this one:** it addresses the one specific, testable question the evidence raises (do scores/performance depend on the ASD-TC motion difference?) without training anything, without selecting a model, and without touching the holdout. It follows the unresolved motion-adjustment result and the 0.596 FD-only AUC.
- **Estimated runtime:** under 2 minutes.
- **Limitations:** the matched subset will be smaller than the full cohort (its size depends on the caliper and is not yet known), so intervals will be wide; matching is on mean FD only (no residual-confounding guarantee); the OOF models were trained on unmatched data; any difference is descriptive, not causal. The result cannot raise accuracy, only qualify how much of 64% may depend on motion.

## If you prefer not to spend more effort
The evidence does not point to a fixable defect in tuning, thresholds, preprocessing or feature selection. Stopping and documenting the static AAL-116 baseline (about 64% five-fold CV accuracy, AUC about 0.69, on the development cohort, not an independent-holdout result) is a defensible alternative.
