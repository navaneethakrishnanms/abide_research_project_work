"""
phase2n_final_holdout_eval.py -- THE ONLY script permitted to evaluate the
locked Phase 2 final holdout (splits/phase2_holdout_subjects.txt).

holdout_guard.get_locked_holdout_ids() checks the caller's filename against
this exact name -- running this logic from a copy, a notebook cell, or any
other script will be rejected by the guard even with the correct
confirmation phrase.

STATUS: NOT YET RUNNABLE. Per the Phase 2 protocol, this script may only be
executed after:
  1. Phase 2C-2M development-set experiments are complete,
  2. a final model has been selected using development-set nested CV only,
     never holdout performance, and
  3. results/FINAL_MODEL_SPEC.json has been written and locked.

Until FINAL_MODEL_SPEC.json exists, this script deliberately refuses to run
(see the guard clause in main()) so it cannot be accidentally executed early
and "spend" the one-time holdout evaluation before the model is actually
finalized.
"""
from __future__ import annotations

import json
import os

from holdout_guard import get_locked_holdout_ids, FINAL_EVAL_CONFIRMATION, verify_split_integrity

FINAL_MODEL_SPEC_PATH = r"C:\abide\results\FINAL_MODEL_SPEC.json"
FINAL_RESULT_PATH     = r"C:\abide\results\FINAL_HOLDOUT_RESULT.csv"


def main():
    if not os.path.exists(FINAL_MODEL_SPEC_PATH):
        raise SystemExit(
            "results/FINAL_MODEL_SPEC.json does not exist yet -- the model "
            "has not been finalized via development-set nested CV (Phase "
            "2C-2M). Refusing to touch the locked holdout. This is "
            "intentional, not a bug."
        )

    with open(FINAL_MODEL_SPEC_PATH) as f:
        spec = json.load(f)

    # Sanity: prove integrity of the split before doing anything else.
    verify_split_integrity()

    # This is the one and only sanctioned call site in the whole project.
    holdout_ids = get_locked_holdout_ids(confirm=FINAL_EVAL_CONFIRMATION)
    print(f"Locked holdout unlocked for final evaluation: {len(holdout_ids)} subjects.")

    # TODO (Phase 2N, once a model spec exists):
    #   1. Load dev subject IDs + fit the FINAL_MODEL_SPEC pipeline on ALL
    #      676 development subjects (preprocessing fit on dev only).
    #   2. Transform the 170 holdout subjects using dev-fitted transformers.
    #   3. Score once. Compute accuracy / balanced accuracy / ROC-AUC / F1 /
    #      sensitivity / specificity / confusion matrix.
    #   4. Write results immediately to FINAL_RESULT_PATH.
    #   5. Do not loop, do not try alternative models, do not re-run.
    raise NotImplementedError(
        "Model-fitting logic intentionally not implemented until Phase 2M "
        "produces a locked FINAL_MODEL_SPEC.json."
    )


if __name__ == "__main__":
    main()
