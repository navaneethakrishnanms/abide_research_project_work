# Phase 9 Limitations

- Only 5 outer folds -- limited statistical power for any Phase 6 vs Phase 9 comparison; fold-level
  differences are reported in full rather than collapsed into a significance claim.
- Dataset size (780 subjects) and ABIDE's known multi-site heterogeneity (20 sites) apply here exactly
  as documented in Phase 8.
- Possible site effects on the 47 constant-ROI subjects (documented in Phase 4.1/4.2) are unchanged by
  this phase.
- Hyperparameter-search uncertainty: the Elastic Net and SSAE searches used a controlled, reduced design
  (one-hyperparameter-at-a-time sweeps, not full grids); a different search strategy could select a
  different configuration.
- Computational constraints limited the search: RBF SVM was not evaluated; nested nested-CV per
  hyperparameter (rather than the single 85/15 inner split reused per outer fold) was not performed.
- SSAE overfitting risk, as documented in Phase 6/8, motivated Stage 3-4 of this phase but is not fully
  eliminated by any tested configuration.
- Elastic Net feature-selection instability across folds (Phase 7/8) remains present in Phase 9's
  feature-selection step.
- Strategy C fold-scoped imputation is unchanged and carries the same limitations documented in Phase 8.
- No external validation cohort was used.
- No clinical validation was performed.
- No biological/experimental validation was performed.
- A numerical accuracy increase, if observed, does not by itself establish superiority beyond this
  specific leakage-controlled evaluation framework on this specific dataset.
