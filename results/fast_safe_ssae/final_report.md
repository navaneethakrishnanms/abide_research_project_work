# Final report -- Elastic Net + SSAE (500-100-2) on AAL-116, 600 development subjects (exploratory)

Development five-fold cross-validation estimates; not independent-holdout performance. The restricted holdout, the restricted ID file and CC200 were not used. Reference: frozen AAL-116 L2 logistic regression (saved).

| Method | Accuracy | Bal. acc. | Sens. | Spec. | Prec. | F1 | AUC |
|---|---|---|---|---|---|---|---|
| EN+SSAE (seed 42, predeclared headline) | 0.6283 +/- 0.0217 | 0.6207 | 0.5519 | 0.6896 | 0.5895 | 0.5680 | 0.6779 +/- 0.0255 |
| EN+SSAE seed 123 | 0.6317 +/- 0.0199 | 0.6244 | 0.5593 | 0.6896 | 0.5927 | 0.5734 | 0.6789 +/- 0.0256 |
| EN+SSAE seed 2026 | 0.6250 +/- 0.0220 | 0.6173 | 0.5480 | 0.6865 | 0.5858 | 0.5640 | 0.6781 +/- 0.0250 |
| Elastic Net classifier (control) | 0.6200 +/- 0.0385 | 0.6149 | 0.5704 | 0.6594 | 0.5764 | 0.5697 | 0.6734 +/- 0.0291 |
| Frozen baseline (L2 logistic regression) | 0.6417 +/- 0.0429 | 0.6378 | 0.6042 | 0.6715 | 0.6010 | 0.6010 | 0.6944 +/- 0.0373 |

Paired fold-level differences (method minus reference):

| Method | Reference | Accuracy diff (mean +/- SD) | Folds improved | AUC diff | Balanced acc. diff |
|---|---|---|---|---|---|
| EN+SSAE (seed 42, headline) | frozen baseline | -0.0133 +/- 0.0451 | 1/5 | -0.0165 | -0.0171 |
| Elastic Net classifier | frozen baseline | -0.0217 +/- 0.0443 | 2/5 | -0.0210 | -0.0229 |
| EN+SSAE (seed 42, headline) | Elastic Net classifier | +0.0083 +/- 0.0186 | 3/5 | +0.0045 | +0.0058 |

Per-fold SSAE (seed 42) accuracy: [0.6, 0.658, 0.633, 0.633, 0.617]; baseline: [0.608, 0.658, 0.65, 0.592, 0.7].
Selected per fold -- Elastic Net C: [0.03, 0.03, 0.03, 0.3, 0.03]; SSAE config: ['S2', 'S2', 'S2', 'S2', 'S2']; features: [149, 154, 158, 580, 147]. Selected-feature stability across folds: mean pairwise Jaccard 0.181; features chosen in all 5 folds: 17, in >=4 folds: 57.
Seed stability (mean over folds): EN+SSAE (seed 123): acc 0.6317, AUC 0.6789; EN+SSAE (seed 2026): acc 0.6250, AUC 0.6781; EN+SSAE (seed 42): acc 0.6283, AUC 0.6779.
SSAE trainable parameters (encoders + decoders + classifier) about 339,139; total runtime 645s.

**Verdict (rule fixed in advance: 'adds value' needs >= +0.02 accuracy and AUC over the frozen baseline with >= 4/5 folds improved AND >= +0.02 accuracy over the EN control; 'worse' needs <= -0.02 on both): the Elastic Net + SSAE workflow NO RELIABLE DIFFERENCE from the frozen baseline (inconclusive).**

**80% target: NOT reached** (headline development CV accuracy 62.8%). No significance test is claimed from five folds.

## Limitations and recommendation
- Single dataset of 600 subjects, five folds, one headline seed; three predeclared SSAE settings and three Elastic Net strengths only; settings S2/S3 were informed by earlier Phase 9/10 results on a cohort that overlaps the designated holdout (a design-process caveat).
- Earlier experiments with many representations and classifiers on these data stayed near 64% accuracy / AUC 0.69; this run tests whether the specified architecture changes that, and the answer is in the table above.
- A result of this size does not support 80%. The most informative legitimate next step is not more architecture tuning but understanding the signal limit (edge-level diagnostic effects at chance level, site/motion/scan-length variance) or a site-held-out evaluation design.
