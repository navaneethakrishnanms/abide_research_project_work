# Research plan: next research phase (written before any new result)

Atlas CC200 only; 676 authorized subjects (313 ASD / 363 TC); protected holdout, `RESTRICTED_id_membership.csv` and holdout IDs/labels/features/predictions are never opened. Phenotypic rows are filtered to the authorized training list immediately on load. Existing campaign/audit files and the frozen rule (`selection_rule.py`) are read-only; all new output is in this folder.

## Stage D (diagnostics, no model selection)
D1 data-quality audit (motion, QC, site, demographics, missingness, label balance); D2 performance by subgroup using saved out-of-fold predictions (seeds 0-8, reference LR and the 2-member vote); D3 confound-only baselines (site, FD, age, sex) under the campaign's folds; D4 feature-space structure (effective dimension, redundancy); D5 learning curve (accuracy/AUC vs training-set size) to test whether the cohort size limits performance; D6 label-permutation check of the pipeline; D7 leave-one-site-out vs random folds for the reference LR on CC200.
Output: ranked limitations with evidence (`data_quality_report.md`). No subject is removed in D.

## Stage E (experiments, one hypothesis each, frozen criteria)
Reference: campaign reference LR (R1). Same outer folds (StratifiedKFold 10, shuffle) and inner protocol as the campaign. Exploratory seeds 0-2; formal confirmation seeds 6-8 only for a hypothesis that clears exploratory +0.5 pt AND is not a repeat of an earlier null. Success on confirmation: mean accuracy gain >= +0.5 pt, >= 2/3 seeds positive, mean AUC change >= -0.005. Criteria are not changed after results.
- E1 motion-aware QC (only if D1 shows a defensible, pre-stated criterion, e.g. published FD cut-off; applied identically to train and test; reported both on the retained subset and with the removed subjects counted as errors/unscored).
- E2 elastic-net LR (saga, in-fold C/l1_ratio) - model family.
- E3 RBF-SVM / linear SVM were already evaluated in the campaign (P2_*); not repeated. Regularized gradient boosting (P2_XGB) already evaluated; not repeated.
- E4 in-fold stability selection (L1 with subsampling) - only if D4 indicates noise dominance.
- E5 ComBat-like site harmonization in-fold was evaluated earlier (`combat_in_fold_v1`); not repeated.
- E6 additional data: feasibility audit only (no download of restricted data; ABIDE-II/other cohorts need separate licensing, diagnosis definitions and preprocessing consistency).
Deep learning: not run (676 subjects, no defensible design beyond what the earlier MIL/ResNet attempts in the repository showed); documented.

## Stopping rule
Per the brief: if valid experiments remain at 66-70%, stop tuning on this cohort and report that the data may not support 85%. 85% is claimed only with independent evaluation, which this phase does not have.

---
# Addendum A (frozen before any Stage E result; written after Stage D diagnostics, which are descriptive)
Stage D findings that motivate the experiments (detail in `data_quality_report.md`): ASD subjects have higher motion (mean FD median 0.089 vs 0.069 mm; FD alone separates classes with AUC 0.60, 57.9% accuracy); the first principal component of the connectivity features correlates with FD (Spearman 0.43) and itself separates classes with AUC 0.62; the learning curve is still rising at 676 subjects; leave-one-site-out is about as accurate as random folds for the same fixed model; permuted labels give chance.

## Common protocol
CC200, 676 subjects, outer StratifiedKFold(10, shuffle, seed) = campaign folds; inner 5-fold ROC-AUC grid over C in {0.001, 0.01, 0.1, 1, 10} (seed = outer seed); R1 threshold from inner OOF scores (seed+1000); R0 secondary. Every learned step is a pipeline step fitted on the training rows of the (inner or outer) split. The "none" arm of this engine must reproduce the saved reference LR scores before any other arm is trusted.
Stage E-exploratory: seeds 0-2, informs only. Confirmation: seeds 6-8 (never evaluated by these new arms), run only for an arm with exploratory mean gain >= +0.5 pt (R1). Superiority success (all required): mean accuracy gain >= +0.5 pt; positive in >= 2 of 3 seeds; mean AUC change >= -0.005. Because several hypotheses are tested, a single pass is reported as "supported on this cohort", not as proof; no multiplicity adjustment is made and this is stated.

## Hypotheses
- **E1 motion QC.** Excluding subjects with mean FD > 0.2 mm (published convention; fixed now) from training and scoring improves accuracy on the retained subjects relative to the reference trained on all subjects and scored on the same retained subjects (paired by subject and fold). Subjects removed are reported (number, class, sites); the removed subjects are not scored, so this is a result on a restricted population and is never compared with whole-cohort accuracy. Superiority criteria above, evaluated on the retained subset.
- **E2 motion residualization (validity check).** Linearly regressing mean FD out of every connectivity feature (coefficients fitted on training rows only) does not lower accuracy: success = non-inferiority, mean accuracy change >= -0.5 pt on every evaluated seed-set mean and mean AUC change >= -0.005. A drop larger than 0.5 pt is evidence that part of the reference signal is motion-related and is reported as such (a finding, not a failure of the method).
- **E3 site harmonization.** Per-site feature centering and scaling (statistics from training rows only; sites with < 10 training rows pooled; unseen sites unchanged) improves accuracy. Superiority criteria above. (Earlier ComBat-style test was on the AAL development cohort and is not evidence for CC200.)
- **E4 elastic-net logistic regression** (saga; grid C in {0.01, 0.1, 1} x l1_ratio in {0.3, 0.7} chosen by inner AUC). Superiority criteria above. Run only if a 1-fold timing benchmark shows the 3-seed stage completes in < 3 h; otherwise documented as infeasible.
- E5 stability selection: not pre-registered unless E4 indicates that sparsity helps (E4 exploratory gain >= +0.5 pt); the campaign's L1 selection (63.3%) and k-best selection (65.6%) already did not beat the reference.
Not repeated (already in the campaign with this exact cohort and folds): linear/RBF SVM, random forest, extra trees, XGBoost, PCA+LR, k-best, L1 selection, tangent-space and partial-correlation representations, soft-vote/stacking. Deep learning and graph neural networks: not run (676 subjects, 19 sites; the repository's earlier MIL/ResNet attempts did not exceed the linear baselines and no defensible design with adequate data exists here).
