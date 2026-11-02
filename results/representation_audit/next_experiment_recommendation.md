# Next-experiment recommendation

## Does the audit justify a new representation experiment? No clear justification.
The audit explains the ceiling rather than pointing at a defect:
- Preprocessing is sound and consistent with the PCP `filt_global` strategy; the project's own steps are fold-scoped; no avoidable information loss was found.
- The typical edge carries no diagnosis effect beyond chance (variance share 0.98x of the null) while site, motion and scan length each explain more; edge reliability is about 0.5 even at full length.
- Many prior representation/classifier variants on the same data (see `tested_methods_inventory.csv`), spanning high-dimensional, low-dimensional (CPM), dynamic, site-adjusted and motion-adjusted versions, all give the same 61-68% accuracy.

## If one more test is wanted for completeness (weakly justified)
**Anatomically grouped block-averaged FC on AAL-116:** average edges within and between pre-declared groups of AAL regions (for example frontal, parietal, temporal, occipital, limbic/insula, subcortical, cerebellum/vermis, defined from the AAL region names), giving a few dozen features with no learned parameters, then the same L2 logistic regression and folds, paired with the frozen baseline.
- **Rationale:** it raises signal-to-noise if diagnostic differences are spatially smooth, which the low edge reliability and null median edge effect make plausible.
- **Why only weakly justified:** CPM, a two-feature compression, did not help; the tail of diagnosis-associated edges is probably partly site and motion confounding; the grouping is coarse and must be fixed before running.
- **Runtime:** about 1 minute. It would be exploratory, with no significance claim.

## Not recommended
Further classifier or hyperparameter searches, dynamic-connectivity variants, neural networks, or any evaluation on the holdout (which is not independent of Phases 3-10).

## Alternative: stop and document
Document the baseline (about 64% five-fold development CV accuracy, AUC 0.69) together with this audit's explanation: weak per-edge diagnostic signal, moderate edge reliability and substantial site/motion/scan-length variance. If the goal is a result that generalizes to new sites, a site-held-out evaluation design is a more informative next step than further accuracy tuning, because random-fold accuracy does not show generalization to unseen sites. That would be an evaluation experiment, not a representation search, and it was not run here.
