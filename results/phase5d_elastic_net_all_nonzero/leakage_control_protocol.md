# Leakage Control Protocol — All-Non-Zero Elastic Net (Phase 6+ Reference, NOT Executed Here)

**No training was performed in this phase.**

## Required structure: 5-fold cross-validation

```
FOR EACH of 5 folds:
    TRAINING FOLD
        |
        v
    Apply the selected missing-FC strategy (A/B/C)   <- Strategy C's imputation statistic, if used,
        |                                                fit on training-fold data only
        v
    Fit Elastic Net ONLY on training data             <- alpha/l1_ratio via INNER CV on training folds only
        |
        v
    Identify ALL non-zero coefficients                <- no ranking/truncation step (this is the
        |                                                 change from Phase 5C's Option 2)
        v
    Store selected feature indices (N_k, data-dependent)
        |
        v
    Train downstream SSAE using those N_k features    <- fresh model instance, input layer = N_k
        |
        v
    Apply the SAME selected feature indices to validation/test data  <- test fold's OWN Elastic Net
        |                                                                coefficients are never computed;
        |                                                                it is only ever transformed using
        |                                                                the training fold's selection
        v
    Evaluate only after training is complete
```

## Explicitly forbidden

```
ALL 780 SUBJECTS
    |
    v
Elastic Net (fit on all 780)
    |
    v
select features (using all 780)
    |
    v
5-fold CV   <-- every fold's "test" subjects already influenced which features were
                even available for the CV loop to work with. This is feature-selection
                leakage, structurally identical to the flaw underlying the paper's own
                currently-active Expression of Concern (SVM-RFE run on the full dataset
                before cross-validation -- Phase 4.1 finding).
```

## The test fold must never influence

- Elastic Net fitting (coefficients, hyperparameters)
- Which features end up non-zero
- Imputation statistics (if Strategy C is used)
- Any hyperparameter selection (alpha, l1_ratio) -- these must come from INNER CV on training data only

## Relationship to existing project infrastructure

Same `holdout_guard.py`-style discipline already used in this project's earlier ML phases: fold-scoped
fitting, explicit assertions against test-fold contamination, no test-label or test-value use at any
preprocessing/selection stage.
