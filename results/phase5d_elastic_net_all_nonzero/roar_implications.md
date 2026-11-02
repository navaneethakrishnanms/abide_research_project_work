# ROAR Implications — Variable N Across Folds

**No ROAR was executed. This is a conceptual compatibility analysis only.**

## Conceptual pipeline (not executed)

```
Elastic Net -> N_k selected features (fold k) -> SSAE -> XAI attribution
    -> rank the N_k selected features by attribution
    -> remove a predefined PERCENTAGE of them
    -> retrain (fresh Elastic Net + SSAE fit, still training-fold-only)
    -> evaluate on the untouched test fold
```

## How percentage-based removal handles variable N

ROAR conventionally removes a stated *percentage* (e.g. top 10%, 20%, 50%) of the currently-attributed
feature set, not a fixed absolute count. This scales naturally with a variable N_k: "remove the top
20% of fold *k*'s N_k selected features" is well-defined regardless of whether N_k is 500 or 5,000 —
the absolute number of features removed differs per fold (0.2 × N_k), but the *procedure* is identical
and requires no special-casing for variable N. This is a structural advantage of percentage-based
ROAR over an absolute-count-based variant, specifically in the all-non-zero setting.

## What would need to be fixed before ROAR could run (not done here)

1. Elastic Net selection and SSAE architecture would need to be finalized (this phase only analyzes
   feasibility).
2. A specific XAI method would need to be chosen to produce the attribution ranking ROAR consumes.
3. The removal-percentage schedule (e.g. 10/20/30/...%) would need to be fixed in advance, not chosen
   after observing any accuracy result.

## Not executed here

No feature was removed, no model was retrained, and no ROAR accuracy was calculated, on real or
synthetic data, in this phase.
