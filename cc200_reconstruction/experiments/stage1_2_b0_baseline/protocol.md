# Stage 1-2 protocol: CC200 training-data check and baseline B0 reconstruction (written before any code ran)

Atlas: CC200 only. Authorized scope: Stages 1 and 2 of `cc200_reconstruction/reconstruction_plan.md`. Not run here: leave-one-site-out, variants V1-V6, any final or holdout evaluation.

## Data rules
- Subjects: only the 676 IDs in `splits/phase2_train_subjects.txt`. Per-subject files `abide_pcp/ABIDE_pcp/cpac/filt_global/<FILE_ID>_rois_cc200.1D` are opened only for those IDs; `ts_cc200.npy`, `covs_cc200.npy`,
  the holdout list, `RESTRICTED_id_membership.csv`, `holdout_guard.py` and every holdout label/feature/prediction are not opened. `phenotypic_qc.csv` is filtered to the training IDs immediately after reading and the remaining rows are discarded unread.
- Subject order: the order of `phenotypic_qc.csv` rows restricted to the training IDs (the order phases 2c-2h used), so the folds `StratifiedKFold(10, shuffle, random_state=seed)` are comparable.
- Stop rule: any failed integrity check, or any training-list/provenance ambiguity, stops the run before Stage 2 and is reported.

## Stage 1 checks (logged to `stage1_audit_log.md` and `stage1_checks.json`)
Training list (676 unique IDs); one raw file per ID, none missing, none duplicated; phenotype row per ID (unique); `FILE_ID` equals the raw file name stem; site prefix consistent; labels in {1, 2}; class counts; age/sex complete;
every series is (T x 200) with T >= 100, finite, no all-NaN; constant-ROI count per subject; no identical series shared by two subjects; covariances symmetric, positive definite, 200 x 200; correlations in [-1, 1] with unit diagonal;
features 19,900 per subject, finite, in `np.triu_indices(200, 1)` order; independent recomputation (own Ledoit-Wolf + arctanh code) equals the project's `tangent.py` output; deterministic on recomputation.
Deliberately **not** done: the planned comparison of raw-file features with `covs_cc200.npy` rows (needs the combined array that contains holdout rows).

## Stage 2: baseline B0 (recovered configuration, `original_pipeline_config.json`)
- Features: per-subject z-score, Ledoit-Wolf covariance (assume_centered), correlation, Fisher-Z (clip 0.999999), upper triangle (19,900) from `tangent.py`; age and SEX appended (19,902). These per-subject steps have no cross-subject fitted statistics.
- Model: demographics median-imputed (training rows) and standardized; StandardScaler on all columns; LogisticRegression(L2, lbfgs, max_iter 5000, class_weight balanced); GridSearchCV over C in {0.001, 0.01, 0.1, 1, 10}, scoring ROC-AUC, inner StratifiedKFold(5, shuffle, random_state = seed).
- Outer: StratifiedKFold(10, shuffle, random_state = seed) for seeds 0, 1, 2 (30 outer fits per arm). Out-of-fold probabilities for every subject; one prediction per subject per seed.
- **Arm A (historical-faithful):** preprocessing fitted on each outer-training set and applied to the outer-test set; the inner CV then runs on the already-scaled training matrix (as in phase 2c). This is the arm compared with phase 2h.
- **Arm B (strict, companion):** the same preprocessing refitted inside every inner fold (a Pipeline), so nothing leaks inside the inner selection either. Reported alongside; no choice between arms is made on outer scores.
- Three threshold readings per arm, all fixed in advance: (i) 0.5; (ii) the historical rule, Youden's J on the seed-level pooled out-of-fold scores (comparable to the recorded 67.21%, mildly optimistic because it sees the evaluated scores);
  (iii) nested: Youden's J on inner 5-fold out-of-fold probabilities of each outer-training set only (random_state = seed + 1000), applied to that outer-test fold.
- Metrics: accuracy, balanced accuracy, sensitivity, specificity, precision, F1, ROC-AUC, pooled confusion matrix per seed and reading, per-fold values and SDs, mean +/- SD over seeds. No best-fold or best-seed reporting.

## Comparison with phase 2h (no forcing, no re-tuning)
Per seed: Arm A with reading (ii) against `results/phase2c_baseline_results.csv` (accuracy, tau, AUC, F1, confusion matrix, chosen C per fold); mean against 67.21% (AUC 0.7175). Expected differences (documented): features from the raw 1D files vs the stored
array; arm B's inner-fold preprocessing; threading/BLAS order (n_jobs 4, single-thread numerics); library versions. A match supports the reconstruction; a mismatch is reported and not tuned away.

## Outputs (new folder only; no historical file touched)
`protocol.md`, `b0_components.py`, `run_stage1_2.py`, `stage1_audit_log.md`, `stage1_checks.json`, `subject_manifest.csv`, `features_cc200_train676.npz` (cache), `fold_metrics.csv`, `oof_predictions.csv`, `seed_metrics.csv`, `comparison_vs_phase2h.csv`, `final_report.md`, `checkpoints/`, `logs/`.

## Estimated runtime
Stage 1: about 2-4 min. Stage 2: about 10-20 min for both arms (60 outer fits plus nested-threshold fits), checkpointed per arm and seed.
