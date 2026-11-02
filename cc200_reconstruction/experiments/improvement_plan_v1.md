# CC200 improvement plan v1 (protocol only; nothing has been run)

Status: **written before any experiment; no training, leave-one-site-out run or variant has been started.** Atlas: CC200 only; no AAL-116 data, results or evidence are used anywhere in this plan.
Target: find a real, reproducible improvement over the reconstructed baseline. 85-90% is not promised and nothing is tuned to reach it.

## 0. Which experiment should run first
**Experiment X1: site robustness (Candidate E): a site-aware reference for the baseline plus in-fold, label-free site harmonization.** Reasons:
1. It is the only candidate with **no existing CC200 development evidence** (representation, feature selection and classifier were already compared on these same 676 subjects in phases 2e-2g; section 2).
2. A leave-one-site-out reference for the baseline does not exist for the authorized training list; every other candidate would be judged without knowing how site-dependent the baseline is.
3. Site is the largest uncontrolled nuisance factor in multi-site ABIDE data, and the harmonizer's fitting can be made valid for deployment (training-fold parameters; unseen sites left unchanged).
Honest expectation: a gain of about 0 to 1.5 points at best; 85-90% is not plausible from any single candidate.

## 1. Verified baseline and permitted data scope
**Baseline (reported numbers verified against the saved files; all five match):**
| Item | Saved value |
|---|---|
| Arm A recorded-rule accuracy | 67.21% |
| Arm B recorded-rule accuracy | 67.31% |
| Arm B fixed-0.5 accuracy | 66.07% (per seed pooled: 65.83%, 66.27%, 66.12%) |
| Arm B nested-threshold accuracy | 65.68% |
| Arm B ROC-AUC | 0.7187 (per seed 0.7129, 0.7202, 0.7230) |
| Arm B at 0.5, other metrics (seed mean) | balanced accuracy 65.82%, sensitivity 62.41%, specificity 69.24%, precision 63.63%, F1 63.01% |
| Fold-to-fold SD of accuracy (arm B, 0.5) | 4.64 points (per-seed 3.05, 4.70, 5.55); SD of the three seed means 0.23 points |
**Reference arm for all comparisons: Arm B** (strict pipeline: all preprocessing refitted inside every inner fold), fixed threshold 0.5. Arms A and B agree to 0.1 point at 0.5, so the choice does not matter; B is the stricter protocol.

**Data scope (verified before writing this plan):** only the 676 authorized Phase-2 training subjects (manifest IDs equal `splits/phase2_train_subjects.txt` as sets; Stage 1: 21/21 checks passed). Candidates read **only the feature cache**
`stage1_2_b0_baseline/features_cc200_train676.npz` (sha256 `df6e7f9a...fd43c`; full hash in section 8) after re-verifying its hash and the manifest ID set at start. The holdout list, `RESTRICTED_id_membership.csv`, any holdout label/feature/prediction and the combined arrays `ts_cc200.npy` / `covs_cc200.npy` are never opened. Baseline files and checkpoints are read-only.

## 2. Candidate triage (V1-V6 and A-F), using existing CC200 development evidence
Existing evidence is from phases 2e-2g on the same 676 subjects, seeds 0-2, historical protocol, accuracy at the Youden-on-OOF reading (baseline 67.21%); it is a prior, not a substitute for the new protocol.
| Candidate | Plan ID | Clear hypothesis? | Existing CC200 evidence | Decision |
|---|---|---|---|---|
| A. Connectivity representation | V1 plain Pearson, V6 tangent-space | V1: weak (shrinkage matters?). V6: tangent space linearizes covariance geometry | raw covariance 67.36%, tangent 67.70% (+0.49, SD 0.25/0.62), partial correlation 63.46% | V1 dropped (question answered: raw vs shrunk differ by 0.15). OAS / graphical-lasso dropped (no CC200-specific rationale, costly). **V6 optional, X4.** |
| B. Feature selection | V3 SelectKBest | yes (19,900 features vs about 540 training rows) | k = 100 / 500 / 1000 / 2000 / 5000: 62.62 / 64.69 / 65.63 / 66.86 / 67.11%; all features 67.21% | falsified at the historical protocol; **not run** (revisit only if X1-X3 suggest a different regime). |
| C. Classifier | V2 linear SVM | moderate | linear SVM 64.55% vs logistic 65.53% within one earlier run (RBF 65.93%) | **optional, X3** (low prior). |
| D. Demographic handling | V4 | yes (age/sex encoding and whether they help at all) | FC-only vs + age/sex on the old holdout: identical counts; no CC200 dev ablation | **X2.** |
| E. Site robustness | V5 | yes | none for the training list (the all-846 leave-site-out run included the later-reused holdout subjects) | **X1 (first).** |
| F. Decision threshold | none | answered | baseline readings: fixed 0.5 = 65.98-66.07%, nested training-only Youden = 65.68-66.12%, historical Youden-on-OOF = 67.21-67.31% (optimistic by about 1.2-1.3 points) | **no new run**: nested thresholds do not beat 0.5; the recorded 67.2% overstates the clean figure. |

## 3. Common validation design (all experiments)
- Subjects: 676 authorized training subjects; 313 ASD / 363 TC; nothing removed, no case selection.
- Outer folds: StratifiedKFold(10, shuffle, random_state = seed) for seeds 0, 1, 2 (identical to the baseline); inner: StratifiedKFold(5, shuffle, random_state = seed), ROC-AUC, same C grid {0.001, 0.01, 0.1, 1, 10} for logistic regression (SVM grid fixed in advance in X3).
- Everything fitted inside the training partition: imputation, scaling, harmonization, residualization, feature selection, hyperparameters, thresholds. Held-out rows never enter any fit; a perturbation check (altering test rows leaves other test rows' outputs unchanged) is part of each experiment's self-test.
- Threshold: fixed 0.5 for the primary metric (decision function 0 for SVM). Nested training-only Youden and the recorded Youden-on-OOF reading are reported as secondary only.
- **Primary metric:** pooled out-of-fold accuracy at the fixed threshold per seed (676 predictions per seed), averaged over the 3 seeds; paired per-fold accuracies (30 folds) for win counts.
- **Reported for every experiment:** accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, precision, F1, pooled confusion matrices per seed, per-fold values, mean and SD across folds and across seeds, selected hyperparameters, runtime, failures. Predictions saved per subject.

## 4. Predeclared acceptance rule (fixed now; not changed after seeing results)
A candidate is **retained** only if all hold against Arm B with the same seeds and folds:
1. Mean paired improvement over the 3 seeds of **at least +1.0 point in accuracy and +1.0 point in balanced accuracy** (about 7 more correct subjects of 676), and **at least +0.5 point in every seed**.
2. Accuracy higher in **at least 20 of the 30 paired outer folds** (one-sided sign test p = 0.049).
3. No material deterioration elsewhere: ROC-AUC not lower by more than 0.005; sensitivity and specificity each not lower by more than 3 points.
4. No protocol violation or leakage found in the experiment's checks.
5. **Replication:** rerun baseline and candidate on fresh seeds 3, 4, 5 (new outer partitions); the mean accuracy gain must again be at least +0.5 point and positive in at least 2 of 3 seeds. Only then is the candidate called a **validated development improvement**.
X1 adds: pooled leave-one-site-out accuracy not worse than the baseline's by more than 0.5 point.
Multiple comparisons: at most four candidates are examined; the effect-size floor, the sign test and the replication step guard against best-of-K selection, and the number of candidates examined is reported. If a candidate fails, the result is recorded and the rule is not loosened.
Note on noise: Arms A and B give paired-fold differences with SD 0.38 point, but they are near-identical pipelines, so that is not a noise estimate for different candidates. The paired SD for each candidate is reported with its results.

## 5. Experiment specifications
### X1 (run first). Site robustness (Candidate E, V5)
- **Hypothesis / rationale:** multi-site scanner and population differences add a site signature to connectivity; removing per-site location and scale from the training features may let the classifier learn a more diagnosis-specific boundary.
- **Changes:** (a) *Reference:* baseline Arm B under **leave-one-site-out** (19 site-held-out fits; per-site and pooled metrics; the pooled AUC mixes site-specific models). (b) *Variant H:* a new CC200 transformer `cc200_harmonizer.py` (generic ComBat-style empirical-Bayes location/scale adjustment, parametric prior; site = batch; covariates age and sex; **no diagnosis labels**) inserted before the scaler in the Arm-B pipeline, evaluated in the 10 x 3 CV and in leave-one-site-out.
  Training sites with fewer than 5 training subjects are pooled; unseen (held-out) sites are left unchanged; the map is identical for training and test rows; all parameters come from the rows passed to `fit`. Implemented fresh for this experiment (no AAL-116 data or results are read or reused).
- **Primary metric:** per section 3 for the 10 x 3 CV; leave-one-site-out pooled accuracy as the co-requirement.
- **Validation design:** same outer folds as the baseline plus leave-one-site-out; harmonizer refitted inside every inner fold.
- **Runtime (estimate, to be confirmed by a one-fold smoke test):** harmonizer adds about 10-12 s per outer fold: 10 x 3 CV about 10 min, leave-one-site-out for baseline and H about 3 + 6 min; **about 20 min in total**.
- **Acceptance criterion:** section 4 (including the LOSO co-requirement).
- **Leakage / confounding risks:** site is confounded with diagnosis prevalence (ASD share ranges from 29% to 64% across the 19 sites), so a label-free harmonizer can absorb some diagnostic signal into site terms (works against a gain); sites with fewer than 20 subjects give noisy parameters; under leave-one-site-out the harmonizer cannot adjust the unseen site, so any LOSO effect comes only from cleaner training data; harmonizing with test-set statistics (transductive centering) is **not** used because it is not valid for single-subject deployment.
- **Outputs:** `experiments/improvement_v1/X1_site/` with `config.json`, `cc200_harmonizer.py`, `run_x1.py`, `fold_metrics.csv`, `oof_predictions.csv`, `leave_site_out_predictions.csv`, `site_metrics.csv`, `report.md`, `logs/`.

### X2. Demographic handling (Candidate D, V4)
- **Hypothesis:** age/sex as two appended columns among 19,900 standardized features contribute almost nothing; and age-related variance inside the connectivity features may act as a nuisance that in-fold residualization can remove.
- **Changes:** arms D0 = baseline (append), D1 = no demographics (19,900 features), D2 = residualize every feature on [age, female] with coefficients fitted on the training rows (no demographics appended). Age and sex are legitimate predictors; no diagnosis-derived variable is used; missing values (none in this cohort) would use the training median.
- **Primary metric / validation:** section 3 (D1 is a simplification test: judged on non-inferiority within 0.5 point, not as an improvement; D2 judged by section 4).
- **Runtime:** about 5 min (D1) + 8 min (D2) = **about 13 min**.
- **Leakage / confounding risks:** sex differs by diagnosis (female share lower in ASD), so removing sex-related variance could remove real signal; the residualization model is fitted on training rows only; age is strongly site-structured, so residualization partly removes site information as well (overlaps with X1; X2 results are read together with X1).
- **Outputs:** `experiments/improvement_v1/X2_demographics/` (same structure).

### X3 (optional). Classifier (Candidate C, V2)
- **Hypothesis:** a margin-based linear classifier is more robust than logistic regression when features far exceed samples.
- **Change:** `LinearSVC(class_weight="balanced")` in the Arm-B pipeline; C grid fixed now: {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}; inner 5-fold ROC-AUC on the decision function; threshold on the decision function at 0.
- **Prior:** low (about -1 point in an earlier run). **Runtime:** about 5 min. **Risks:** none beyond the common ones. Run only if X1/X2 results leave room, or if you ask for it.

### X4 (optional). Tangent-space representation (Candidate A, V6)
- **Hypothesis:** Euclidean models on tangent-space vectors respect the geometry of covariance matrices; earlier CC200 evidence +0.49 point (within noise).
- **Change:** `TangentEmbedding("tangent")` with the geometric-mean reference fitted on training covariances inside every fold (Fisher-Z not applicable); same classifier and grid.
- **Note:** the cache holds correlation features only; X4 would first need the 676 covariance matrices recomputed from the authorized per-subject files (9 s) and cached with a hash.
- **Runtime:** about 20-25 min (reference estimation about 40 s per fold). **Risks:** reference fitted on training rows only; slower and numerically more fragile. Run last, only if you want it.

## 6. Order and stopping
X1 -> (X2) -> optional X3, X4. After each experiment the results are reported and you approve the next one; no automatic chaining. If X1 fails the rule, the result is recorded and the plan continues without reinterpretation.
Expected outcome, stated in advance: all candidates within about +-1.5 points of the baseline's 66.1% (AUC about 0.72). A validated gain of 1 point would be a real but small result; 85-90% is not plausible from these changes.

## 7. Outputs (under `C:\abide\cc200_reconstruction\experiments\improvement_v1\`; nothing existing overwritten)
`experiment_results.csv` (one row per experiment x arm x seed), `comparison_report.md`, per-experiment folders with `config.json`, scripts, genuine per-subject prediction files and logs, and a final summary naming the best validated candidate and its limitations. This plan stays at `experiments/improvement_plan_v1.md`.

## 8. Provenance for reuse of intermediates (checked again at the start of every run)
| File | sha256 |
|---|---|
| `features_cc200_train676.npz` | `df6e7f9a3d15c7d659c00c2ed0d24d497d6be58d039856b5fb4ce795d39fd43c` |
| `subject_manifest.csv` | `22cb64022ab5947716993b82f19be2814d469f3776241b3a36b5c10b3e93ea7a` |
| `seed_metrics.csv` | `4211d0305ea384cb77075ab2d0a01d1f8b958899c7b5b27b2e36095d2d4c3c46` |
| `fold_metrics.csv` | `bab680c18643666d08ced0c2d30557ccba546058fd0d49b456ceb1fd8c872f2e` |
| `oof_predictions.csv` | `28dc62c095828f68e76d0d2bbd5496c61b472f742e635207bcce72341e73682e` |
| `stage1_checks.json` | `c3f835248dd489221de957994d96f716b3abe06379ae655dd0e5e5a49c94cb55` |
(all in `experiments/stage1_2_b0_baseline/`)

## 9. Decisions needed from you
1. Approve X1 as the first experiment (including the baseline leave-one-site-out run it contains).
2. Confirm that X3 and X4 stay optional and that Candidates B and F need no new run.
3. Confirm the acceptance rule in section 4 as written, because it cannot change once X1 runs.
