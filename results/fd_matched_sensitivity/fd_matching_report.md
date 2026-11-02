# FD-matched sensitivity analysis of saved static AAL-116 OOF predictions

No model was trained. The 600 development subjects, the saved AAL-116 L2 logistic-regression out-of-fold predictions (one per subject, trained on the unmatched cohort), diagnosis labels, folds and subject-level mean FD were used. The holdout, CC200 and the restricted ID file were not accessed. Descriptive analysis only; not an independent test.

## Protocol (declared before running; not changed afterwards)
Greedy 1:1 nearest-neighbour matching on mean FD without replacement; each ASD subject (seeded random order, seed 0) takes the nearest unused TC within a caliper of 0.2 x SD of FD over the cohort (= 0.0086). Matching used only FD and diagnosis, never scores or errors. Bootstrap: 2,000 resamples of matched pairs (seed 1). Random reference: 2,000 subsets each (seed 2), same size (498) simple random, and class-balanced (249 ASD + 249 TC). The caliper was not adjusted.

## Matching and FD balance
- **249 matched pairs** (93% of ASD subjects); unmatched: 19 ASD and 83 TC.
- Standardized mean FD difference (ASD minus TC): **0.338 before, 0.004 after** (Mann-Whitney p 1e-4 before, 0.96 after). Mean FD 0.0834 (ASD) vs 0.0833 (TC) in the matched set.
- The unmatched remainder is extreme: the 19 ASD have mean FD 0.164 versus 0.049 for the 83 TC (standardized difference 4.2). Matching therefore discards the high-FD ASD tail and many low-FD TC, so the matched subset covers a narrower FD range and is not representative of the full cohort.

## Performance of the saved predictions
| Sample | n | AUC | Accuracy | Bal. acc. | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| Full development cohort (pooled OOF) | 600 | 0.692 | 0.642 | 0.638 | 0.605 | 0.672 | 0.601 |
| FD-matched subset | 498 | 0.680 | 0.627 | 0.627 | 0.610 | 0.643 | 0.620 |

Pair-bootstrap 95% intervals for the matched subset (descriptive; they ignore model-training uncertainty): AUC 0.634-0.725, accuracy 0.584-0.669, sensitivity 0.550-0.675, specificity 0.582-0.699, F1 0.576-0.665. The full-cohort values lie inside every interval.

## Comparison with random subsets of the same size (2,000 each)
- Unstratified: matched AUC 0.680 sits at the 9.8th percentile (reference 95% range 0.673-0.712); accuracy 0.627 at the 3.2nd percentile (range 0.625-0.661); specificity 0.643 at the 0.65th percentile, **outside** the range (0.650-0.696); sensitivity and F1 inside.
- Class-balanced (fairer for accuracy and specificity, since the matched set is 50/50): AUC at the 9.7th percentile, accuracy/balanced accuracy at the 7.0th, specificity at the 1.7th (all inside the 95% range, specificity just inside), sensitivity at the 70th, F1 at the 24th.
- Twelve comparisons were made; with that many, one or two near-boundary values are expected by chance and no multiplicity adjustment was applied.

## Interpretation
- On the FD-matched subset the saved predictions perform slightly lower than on the whole cohort (AUC -0.012, accuracy -0.015, balanced accuracy -0.011) and slightly lower than random subsets (matched values sit in the lower tail, at about the 3rd-10th percentile for AUC and accuracy, mostly inside the 95% reference ranges). The shift is small and its interval overlaps the full-cohort value.
- This is compatible with a small motion contribution to performance (the AUC change is about 0.012 against 0.19 above chance, roughly 6%) and equally with no contribution; this analysis cannot tell the two apart. It is consistent in size with the earlier fold-safe motion-adjustment experiment (AUC about -0.02, unresolved).
- It does **not** show that motion explains the 64% accuracy: matched-subset AUC stays near 0.68 with balanced FD.
- No causal claim, no significance claim (no inferential test was specified), no claim that matching removes all confounding, and no independent-test or 85% claim. Caveats: narrower FD range after matching, mean-FD matching only, models trained on unmatched data, one model analyzed.

## Conclusion
The evidence does not support further investigation of motion confounding as the explanation for the performance ceiling: removing the ASD-TC FD imbalance leaves performance within a few hundredths of the original. The simplest next step is to document the static AAL-116 baseline (about 64% five-fold cross-validation accuracy, AUC about 0.69, on the development cohort; cross-validation estimate, not an independent-holdout result).
