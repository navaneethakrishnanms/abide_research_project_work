# Audit summary (accuracy_research_v2)

Verified from saved reports and result files; no new fitting was done for this audit. All figures are development cross-validation (CV) estimates unless stated; none is an independent-holdout result.

## 1. Existing results and their exact protocols
| Experiment | Cohort / folds | Representation / model | Result | Source |
|---|---|---|---|---|
| **Reference baseline** | 600 development subjects, 5 outer folds (StratifiedKFold seed 42), inner stratified 3-fold (seeds 42/123/2026) | AAL-116 Pearson -> Fisher-Z (6,670 edges), training-fold imputation/scaler/SelectKBest, L2 logistic regression | 64.2% (seed 42), 64.7% seed-averaged; AUC 0.694 | `fast_accuracy_experiment`, `accuracy_improvement_v1` |
| Leave-one-site-out baseline | same 600, 19 sites held out in turn | same | 62.2% pooled, AUC 0.684 (mixed-model pooled AUC) | `leave_site_out_baseline` |
| Linear/RBF SVM, L1 LR (AAL-116 and CC200, separate) | same 600/5 folds | same grid logic | 61.3-64.8%, AUC 0.65-0.70 | `fast_accuracy_experiment` |
| Elastic Net, Elastic Net + SSAE (500-100-2) | same 600/5 folds, 3 seeds | EN selection, SSAE | 62.0%; 62.5-63.2% | `fast_safe_ssae` |
| Threshold selection, wider C grid | same, 3 seeds | LR variants | 63.2-64.1% vs 64.7% | `accuracy_improvement_v1` |
| Site centering (training-fold means) | same 600/5 folds | LR, SVM | accuracy -0.010 to +0.005 | `site_centering_experiment` |
| Mean-FD residualization | same | LR, SVM | accuracy -0.013 to -0.035 | `motion_confound_experiment` |
| Dynamic connectivity (50/10 TR windows) | same | variance, mean absolute change | chance (0.50, 0.53) | `aal116_dynamic_fast` |
| CPM | same | 2-feature network strengths | 63.2%, AUC 0.668 | `advanced_model_search` |
| Cerebrum-only (ROIs 1-90) | same, 3 seeds | LR | 62.8% vs 64.7%; rule met in 0/3 seeds | `cerebrum_only_experiment` |
| **Transductive test-site centering** | same 600, leave-one-site-out | LR | pooled 65.7% (frozen 62.2%), AUC 0.708, **predeclared rule FAILED** (accuracy higher at 6 of 13 sites; 7 needed) | `test_site_centering_experiment` |
| Phases 6, 9, 10 (EN + SSAE) | 780 subjects (158 of 170 designated-holdout subjects included) | EN + SSAE | 61.5%, 64.9%, 63.0% | earlier phase folders |
| **Phases 2c-2h** | 676 train-list subjects (includes high-motion scans), **CC200**, 10-fold x 3 seeds | logistic regression **with age and sex covariates** and a threshold protocol; correlation 67.2%, partial correlation 63.5%, **tangent-space 67.7%**, raw covariance 67.4% | different cohort, atlas, folds and covariates: **not comparable** with the 600-subject results |

Verified detail worth stating: the older ~67% figures include age and sex as model inputs (`phase2e_representations.py`), and were not computed on the FC-only 600-subject protocol, so they should not be quoted as a better baseline.

## 2. Which experiments are exhausted
Classifier families and regularization on static AAL-116 or CC200 FC; Elastic Net/SSAE; thresholds; training-fold site centering; mean-FD residualization; dynamic windows (current setting); CPM; ROI-subset reduction. Each lands at 61-68% with differences smaller than seed/fold noise (about 1 to 2 points).
**Tangent-space is not new:** `tangent.py` already implements per-subject z-scoring, Ledoit-Wolf shrinkage, a geometric-mean reference fitted on training subjects only, and vectorization of the off-diagonal tangent vector; Phase 2e evaluated it on CC200 (676 subjects, with age and sex): 67.7% against 67.2% for Fisher-Z correlation, i.e. no meaningful gain, at about 1,157 s per run. What is untested is only the same method on AAL-116 under the primary FC-only 600-subject protocol.

## 3. Remaining data-quality and preprocessing questions
- 12,764 undefined correlations (0.32%) from constant ROIs, 40 subjects, 2,014 features; a rater flag ("maybe", mostly cerebellar notes) covers 126 subjects concentrated in six sites; site, scan length (116-316 TRs), spectral content, age and sex mix differ by site; mean FD differs by diagnosis (ASD 0.089, TC 0.075) and persists within sites.
- Feature matrix itself is clean (order, values, no duplicates); all development-cohort fits sit inside training folds.
- Not available: TR, per-frame motion, scan-level QC. The signal at the median edge is at chance level (diagnosis variance share 0.98x of the null), and edge reliability is about 0.5.
- Constant ROIs: Pearson/Fisher-Z needs imputation; a Ledoit-Wolf covariance remains positive definite with a constant ROI (it contributes a shrunk, near-isolated row), so a covariance representation handles them without NaN, at the cost of a degenerate row for those subjects.

## 4. Claims that can and cannot be made
Can: the frozen AAL-116 L2 logistic regression gives about 64% five-fold and 62% leave-site-out development accuracy (AUC about 0.69) on 600 subjects; no tested variant improves it reliably; the old CC200 67% figures used extra covariates; test-site centering changed pooled leave-site-out accuracy by +3.5 points in one exploratory run but failed its predeclared rule.
Cannot: independent-holdout performance (the holdout overlaps Phase 3-10 development, 158 of 170 subjects); 85% or any target; that test-site centering is a validated improvement; that it is a single-subject method (it needs an unlabeled batch of the new site); that AUC values from pooled leave-site-out scores are independently verified.
