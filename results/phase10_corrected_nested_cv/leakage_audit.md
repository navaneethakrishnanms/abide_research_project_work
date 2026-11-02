# Leakage audit -- corrected Phase 10
Self-test (phase10_corrected_selftest.py) all_passed = True

| Item | Result | Evidence |
|---|---|---|
| Imputation (inner) | PASS | fit_prep() on inner-train rows only (build_inner); perturbing inner-val/outer-test rows leaves train output bit-identical. |
| Imputation/scaling (outer) | PASS | fit_prep() on entire outer-train only (outer_prepared); perturbing outer-test rows leaves train matrix + EN selection identical. |
| Scaling (inner) | PASS | StandardScaler inside fit_prep, inner-train only. |
| Fisher-Z | PASS | _fz(): element-wise arctanh with 1e-6 clip, nothing learned. |
| Elastic Net / feature selection (inner) | PASS | get_en()/fit_en() on fd['Xtr'] (inner-train) only. |
| Elastic Net / feature selection (final) | PASS | outer_prepared(): EN fit on entire outer-train only; applied once to outer-test. |
| PCA (Experiment H) | PASS | stage_H(): PCA.fit(fd['Xtr']) per inner split; transform on inner-val. |
| Architecture / regularization / LR / early-stopping selection | PASS | stage_* functions take only the inner list; source scan finds no te_idx/X_te/y_te. |
| Hyperparameter selection scope | PASS | Per outer fold from that fold's own outer-train inner CV; no cross-fold majority vote anywhere (run_fold has no cross-fold state). |
| Classifier selection | PASS | stage_CLF() inner CV only; outer test touched once for the inner-selected classifier. |
| Early stopping | PASS | Uses a 15% stratified hold-out of the TRAINING data (es_split); inner-val/outer-test never used for epoch choice. |
| Seed handling | PASS | Seeds 42/123/2026 predefined; all reported; headline = seed 42 fixed a priori; selection done once (seed 42) per fold. |
| Outer-test evaluation | PASS | Only final_unit()/clf_outer_unit() read te_idx; one evaluation per (fold, seed) plus one for the inner-selected classifier. |
| Site-grouped evaluation | PASS | StratifiedGroupKFold(site) for outer AND inner splits; run_fold asserts no outer site overlap; own selection per site fold. |
| Fold partitions | PASS | Inner subsets disjoint from outer test; group folds share no sites. |

## Residual notes (not strict-nested failures)
- Greedy sequential stage search (EN->ARCH->WD->DROPOUT->LR->ES) per fold; interactions between stages are not searched (scope reduction, disclosed).
- Inner CV has k=2 (small training sets); inner scores are noisy. Tie tolerance 0.005 AUC with complexity tie-break.
- Selected-config inner scores are a max over candidates and therefore slightly optimistic.
- Final refit uses the same 15% training hold-out recipe for early stopping; that hold-out was inside Elastic Net's training data (affects only epoch choice, not test).
- Experiments B/H are comparators (LR probe) and do not feed the main model; CLF chooses a classifier by inner CV and evaluates only that choice on outer test.
