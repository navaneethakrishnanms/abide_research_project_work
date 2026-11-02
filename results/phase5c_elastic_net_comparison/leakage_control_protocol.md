# Leakage Control Protocol — Elastic Net + SSAE (Phase 6+ Reference, NOT Executed Here)

**No training was performed in this phase. This document specifies the required protocol for when
training does eventually occur, consistent with this project's existing `holdout_guard.py` discipline
established in the earlier ML phases.**

## Required structure: 5-fold cross-validation

```
FOR EACH of 5 folds:
    TRAINING DATA (4 folds)
        |
        v
    FC missing-value handling            <- fit ONLY on training-fold data (Strategy C: training-fold
        |                                    mean/median; Strategies A/B: no fitting needed, deterministic)
        v
    Fit Elastic Net (or ElasticNetCV)     <- fit ONLY on training-fold data; alpha/l1_ratio selected via
        |                                    INNER cross-validation on training folds only
        v
    Feature selection (Option 1 or 2)     <- determined ONLY from training-fold Elastic Net coefficients
        |
        v
    Train SSAE                            <- fit ONLY on training-fold data, using only the
        |                                    training-fold-selected features
        v
    Evaluate on the UNTOUCHED test fold   <- test fold transformed using training-fold statistics/
                                              feature list ONLY; test labels/values never used upstream
```

## Explicitly forbidden (the leakage pattern to avoid)

```
ALL 780 SUBJECTS
    |
    v
Imputation (using all 780 subjects' statistics)
    |
    v
Elastic Net (fit on all 780 subjects)
    |
    v
Feature selection (using all 780 subjects)
    |
    v
5-fold CV  <-- by this point, every fold's "test" data already influenced the
               imputation statistic, the Elastic Net fit, and the feature list.
               This is feature-selection leakage -- structurally identical to
               the flaw underlying the paper's own currently-active Expression
               of Concern (Phase 4.1 finding: SVM-RFE was run on the full
               dataset before cross-validation).
```

## Relationship to existing project infrastructure

This protocol is the same discipline already implemented in `holdout_guard.py` for the earlier
CC200-based ML phases (Phase 2C-2H of the original ML track): fold-scoped fitting, no test-label or
test-value use during any preprocessing/selection step, explicit assertions checking for holdout/
test-fold contamination. Reusing that discipline (or an AAL-116-specific equivalent built the same
way) is recommended when training actually begins — not implemented in this phase.
