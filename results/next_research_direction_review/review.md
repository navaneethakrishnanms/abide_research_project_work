# Review of all experiments and the next research direction (read-only)

Nothing was trained or modified. The restricted holdout, `RESTRICTED_id_membership.csv` and CC200 as a primary pipeline were not used. Reference: the frozen 116-ROI AAL L2 logistic-regression baseline on the 600-subject development cohort (five-fold CV, three inner-CV seeds): accuracy 64.7%, AUC 0.694 (seed-averaged; 64.2% for the seed-42 headline); leave-one-site-out pooled accuracy 62.2%, AUC 0.684. All results are development estimates, not independent-holdout results; the holdout is not independent of Phases 3-10 (158 of its 170 subjects were in the 780-subject cohort).

## 1. What was tested, and the verified results
Numbers are five-fold development CV means on the same 600 subjects and folds unless stated; "paired" = difference versus the matched baseline.

| Approach | Verified result | Verdict |
|---|---|---|
| Static classifiers on AAL-116 (L2/L1 LR, linear SVM, RBF SVM) | 63.5-64.2% accuracy, AUC 0.65-0.70 | Equivalent |
| Same on CC200 (separate exploratory atlas) | 61.3-64.8%, AUC 0.65-0.70 | Equivalent |
| Earlier 676-subject CC200 development work (Phase 2e-2h): correlation, partial correlation, tangent-space, raw covariance | 63.5-67.7%; tangent 67.7% vs correlation 67.2%; partial correlation lower | Tangent-space no gain (different cohort/protocol, not directly comparable) |
| Training-fold site centering | accuracy +0.005 / +0.002 / -0.010 (3 candidates), AUC +0.011 to +0.015, accuracy improved in 2-3 of 5 folds | No reliable gain |
| Training-fold mean-FD residualization | accuracy -0.013 / -0.035 / -0.020, AUC about -0.02 | Slightly worse, unresolved |
| FD-only control | 57.5% accuracy, AUC 0.597 | Motion alone carries some signal |
| FD-matched re-evaluation of saved predictions | AUC 0.680 (matched) vs 0.692; accuracy 0.627 vs 0.642 | Motion does not explain the ceiling |
| Dynamic connectivity (50/10 TR windows): variance and mean absolute change | accuracy 0.498 / 0.525, AUC 0.500 / 0.516 | Chance |
| CPM (positive/negative network strength) | 63.2%, AUC 0.668 | No gain |
| Elastic Net classifier | 62.0%, AUC 0.673 | Worse |
| Elastic Net + SSAE (500-100-2), 3 seeds | 62.5-63.2%, AUC 0.678-0.679; vs baseline -0.013 accuracy, 1/5 folds better | No gain; adds little over its Elastic Net stage |
| Threshold selection from training-only predictions | seed-avg accuracy 0.638 vs 0.647; sensitivity +3 points, specificity -4 | Trades, does not improve |
| Wider C grid (with/without threshold) | 0.641 / 0.632 vs 0.647 | No gain |
| Cerebrum-only (ROIs 1-90, 4,005 edges) | 0.628 vs 0.647 seed-averaged; rule met in 0 of 3 seeds; leave-site-out 64.2% vs 62.2% (single pass, mixed per site) | No improvement; less stable |
| Leave-one-site-out of the baseline | 62.2% pooled, AUC 0.684; 9 of 19 sites below their five-fold accuracy | Only a 2-point drop |
| Earlier 780-subject phases 6, 9, 10 (EN + SSAE, nested) | 61.5%, 64.9%, 63.0% | Overlap the holdout; development estimates only |

## 2. What should not be repeated
- More classifier families, kernels, regularization grids or hyper-parameter searches on these features (all within about 2 points; the differences are smaller than the effect of changing the inner-CV seed, about 1 point).
- Elastic Net or SSAE variants, deeper networks, ensembles of the tested models (errors are shared across models: 127 of 600 subjects are wrong in all four AAL models, about 10 expected by chance).
- Threshold re-selection from training data (no improvement; chosen thresholds unstable).
- Dynamic-connectivity summaries with the current windowing (chance), mean-FD residualization and training-fold site centering in random folds (no gain), and ROI-subset reductions such as cerebrum-only.
- Anything evaluated on the restricted holdout.

## 3. Untested ideas that are technically feasible with the data on hand, and why most are not worth running
| Idea | Assessment |
|---|---|
| Anatomically grouped block-averaged FC | Weak: CPM's two-feature compression did not help; coarse grouping |
| Graph node metrics (strength, clustering) | Weak: derived from the same edges |
| AAL + CC200 fusion | Weak: both gave the same performance; same time series |
| Ledoit-Wolf / tangent on AAL | Tangent already shows no gain on CC200 |
| Common-length truncation | Redundant with site centering (scan length is a site property) |
| ComBat-style harmonization | Site centering was the simplest version and gave no gain |
| Non-imaging phenotypes (age, sex) as features | Not an imaging improvement and confound-prone (sex is imbalanced by diagnosis; mean FD differs) |
| Test-site unlabeled feature centering (transductive site adaptation) | The only idea with a specific, evidence-based mechanism; see section 4 |

## 4. Evidence for the one remaining mechanism (hypothesis, not a finding)
Observed: under leave-one-site-out, ranking within sites is retained (mean within-site AUC 0.699 over the 13 sites with n >= 20, against a pooled AUC of 0.684), but the 0.5 decision threshold fails at specific unseen sites: LEUVEN_1 predicted 4.5% ASD against 50% true (sensitivity 0.09, AUC 0.70), SBL sensitivity 0.10 (AUC 0.80), PITT 24% predicted against 49% true, KKI 65% against 26%. The mean gap between predicted and true ASD share is 0.143 per site under leave-site-out against 0.106 under five-fold. Training-only thresholds could not repair this (gap 0.166-0.187). Hypothesis: part of the leave-site-out loss is a site-specific shift of the score distribution, which an adaptation that sees only the unlabeled features of the new site could reduce.
Why this could improve generalization rather than fit this cohort: it is evaluated only on held-out sites, uses no test labels and adds no tunable parameters. Bound on the benefit: the total gap between leave-site-out and five-fold random CV is 2.0 accuracy points and 0.009 AUC, so even a perfect repair could recover only about that much; it cannot lift accuracy toward 80%.

## 5. What would be needed if the data cannot support a strong experiment
- Raw images or a reprocessing route to a harmonized pipeline, with the **TR** and **per-frame motion** (frame censoring, scan-length matching, frequency-based features such as ALFF).
- **More subjects** and more balanced sites; per-site sample sizes are small (6 of 19 sites below 20 subjects), which makes site-level evidence very noisy.
- An **independent external cohort** (for example a separate ABIDE release or another study) for a true held-out test, because the current holdout overlaps earlier design work.
- Phenotypic severity scores (ADOS and ADI-R columns exist in the phenotype table; how complete they are for the development subjects was not checked) for subtype or severity analyses; heterogeneity may explain why static group classification stays near 64%.
- Quality ratings are limited: the only informative rater column flags 21% of subjects, concentrated in six sites, with undocumented abbreviations.

## 6. Summary of the evidence
Across about twenty variants spanning classifiers, representations, adjustments, deep models, thresholds and data-quality reductions, development accuracy stays within 61-68% and AUC within 0.65-0.72. The per-edge diagnostic effect at the median edge is at chance level, edge reliability is about 0.5, and site, scan-length and motion variance exceed diagnosis variance. That pattern fits a weak-signal, small-sample, multi-site problem, and is consistent with what is commonly reported for leakage-controlled multi-site ABIDE analyses (stated from general knowledge, not verified from local files). It does not fit a story in which a better classifier or feature subset on this cohort would approach 80%.
