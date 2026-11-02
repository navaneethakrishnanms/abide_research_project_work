"""
holdout_guard.py -- Safety mechanism protecting the Phase 2 locked final holdout.

Background
----------
Phase 1 audit found the original 170-subject holdout (splits created ad hoc via
`train_test_split(..., random_state=42)` inside train_fc_improved.py /
train_fc_v2.py / fast_holdout_eval.py / compute_ensemble_holdout.py) had been
evaluated against repeatedly across the project's development history -- a
holdout-reuse problem, even though no single script's *fitting* logic leaked.

Phase 2A created a fresh, persisted split (splits/phase2_train_subjects.txt,
splits/phase2_holdout_subjects.txt) specifically so this cannot happen again.
This module is the enforcement layer: every exploratory/development script
should import from here rather than re-deriving its own split, and every
function that could leak the holdout raises loudly instead of failing silently.

What this module does NOT do
-----------------------------
It cannot stop a determined developer from reading splits/phase2_holdout_
subjects.txt directly with open(). It is a guard rail for honest mistakes
(accidentally loading all 846 subjects, accidentally including holdout rows
in a CV fold, accidentally calling a "final eval" helper from a scratch
script), not a cryptographic access-control system. The real protections are:
(a) it's the path of least resistance -- easier to use than to bypass,
(b) it fails LOUDLY and immediately, not silently, and
(c) it makes an accidental holdout touch structurally hard: filenames are
    checked, an explicit confirmation phrase is required, and every access
    is logged.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SPLITS_DIR   = Path(r"C:\abide\splits")
TRAIN_PATH   = SPLITS_DIR / "phase2_train_subjects.txt"
HOLDOUT_PATH = SPLITS_DIR / "phase2_holdout_subjects.txt"
META_PATH    = SPLITS_DIR / "phase2_split_metadata.json"

RESULTS_DIR  = Path(r"C:\abide\results")
AUDIT_LOG    = RESULTS_DIR / "experiment_audit_log.csv"

EXPECTED_N_HOLDOUT = 170
EXPECTED_N_TRAIN   = 676

# Only a script with exactly this filename may call get_locked_holdout_ids().
# This is enforced by inspecting the caller's __file__, not just trusted.
ALLOWED_FINAL_EVAL_SCRIPT = "phase2n_final_holdout_eval.py"

# Must be passed verbatim to get_locked_holdout_ids() as proof of deliberate intent.
FINAL_EVAL_CONFIRMATION = "I_AM_THE_LOCKED_PHASE2_FINAL_EVALUATION_RUN"


class HoldoutLeakageError(RuntimeError):
    """Raised when locked holdout subjects are detected somewhere they must not be."""


class HoldoutAccessDenied(PermissionError):
    """Raised when a script other than the designated final-eval script tries
    to obtain the locked holdout subject IDs, or the confirmation phrase is wrong."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_ids(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_metadata() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


def verify_split_integrity() -> dict:
    """
    Run every structural check the audit requires:
      - exactly 170 holdout subjects
      - exactly 676 train subjects
      - no duplicate IDs in either file
      - zero overlap between train and holdout
      - holdout file content matches the hash stamped in metadata at creation
        time (catches accidental or deliberate tampering with the locked file)

    Raises HoldoutLeakageError on any violation. Returns a small report dict
    on success.
    """
    if not TRAIN_PATH.exists() or not HOLDOUT_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Phase 2 split files not found. Run phase2a_create_split.py first."
        )

    train_ids   = _read_ids(TRAIN_PATH)
    holdout_ids = _read_ids(HOLDOUT_PATH)
    meta        = _load_metadata()

    if len(holdout_ids) != EXPECTED_N_HOLDOUT:
        raise HoldoutLeakageError(
            f"Locked holdout file has {len(holdout_ids)} subjects, "
            f"expected exactly {EXPECTED_N_HOLDOUT}. Refusing to proceed."
        )
    if len(train_ids) != EXPECTED_N_TRAIN:
        raise HoldoutLeakageError(
            f"Train/dev file has {len(train_ids)} subjects, "
            f"expected exactly {EXPECTED_N_TRAIN}. Refusing to proceed."
        )
    if len(set(train_ids)) != len(train_ids):
        raise HoldoutLeakageError("Duplicate subject ID(s) found in train/dev file.")
    if len(set(holdout_ids)) != len(holdout_ids):
        raise HoldoutLeakageError("Duplicate subject ID(s) found in holdout file.")

    overlap = set(train_ids) & set(holdout_ids)
    if overlap:
        raise HoldoutLeakageError(
            f"{len(overlap)} subject(s) appear in BOTH train and holdout: "
            f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )

    current_hash = _sha256_file(HOLDOUT_PATH)
    stamped_hash = meta.get("holdout_ids_sha256")
    if stamped_hash is not None and current_hash != stamped_hash:
        raise HoldoutLeakageError(
            "Holdout file hash does not match the hash stamped in "
            "phase2_split_metadata.json at creation time -- the locked "
            "holdout file may have been modified. Refusing to proceed."
        )

    return {
        "n_train": len(train_ids),
        "n_holdout": len(holdout_ids),
        "overlap": 0,
        "hash_verified": stamped_hash is not None,
    }


def load_dev_subject_ids() -> frozenset[str]:
    """
    The ONLY subject-ID source exploratory/development scripts should use.
    Runs full integrity verification first, then returns the 676 dev IDs.
    """
    verify_split_integrity()
    return frozenset(_read_ids(TRAIN_PATH))


def assert_no_holdout_leakage(subject_ids, context: str = "") -> None:
    """
    Call this on ANY collection of subject IDs that is about to be used for
    training, validation, cross-validation, feature selection, PCA,
    harmonization, hyperparameter tuning, threshold optimization, or ensemble
    weighting. Raises HoldoutLeakageError immediately if any locked holdout
    subject is present.
    """
    holdout_ids = frozenset(_read_ids(HOLDOUT_PATH))
    given = frozenset(str(s) for s in subject_ids)
    hit = given & holdout_ids
    if hit:
        raise HoldoutLeakageError(
            f"[{context or 'unlabeled check'}] {len(hit)} locked holdout "
            f"subject(s) detected in data intended for modeling: "
            f"{sorted(hit)[:10]}{'...' if len(hit) > 10 else ''}. "
            f"This operation has been blocked."
        )


def assert_cv_folds_dev_only(fold_subject_id_arrays, context: str = "") -> None:
    """
    Convenience wrapper for CV loops: pass a list/iterable of subject-ID
    arrays (one per fold, or [train_fold_ids, val_fold_ids, ...]) and every
    one is checked against the locked holdout in a single call.
    """
    for i, ids in enumerate(fold_subject_id_arrays):
        assert_no_holdout_leakage(ids, context=f"{context} fold[{i}]")


def get_locked_holdout_ids(confirm: str) -> frozenset[str]:
    """
    Returns the 170 locked final-holdout subject IDs.

    Guarded by two independent checks, both of which must pass:
      1. The caller's own source file must be named exactly
         ALLOWED_FINAL_EVAL_SCRIPT (checked via the call stack, not a
         self-reported flag).
      2. `confirm` must equal FINAL_EVAL_CONFIRMATION exactly.

    Any call from any other script -- with any confirmation string -- is
    rejected with HoldoutAccessDenied. Every call attempt (successful or not)
    is appended to the experiment audit log.
    """
    caller_frame = inspect.stack()[1]
    caller_file  = os.path.basename(caller_frame.filename)

    allowed = (caller_file == ALLOWED_FINAL_EVAL_SCRIPT) and (confirm == FINAL_EVAL_CONFIRMATION)

    _log_holdout_access_attempt(caller_file, allowed)

    if caller_file != ALLOWED_FINAL_EVAL_SCRIPT:
        raise HoldoutAccessDenied(
            f"get_locked_holdout_ids() was called from '{caller_file}', but "
            f"only '{ALLOWED_FINAL_EVAL_SCRIPT}' is permitted to access the "
            f"locked holdout. Denied."
        )
    if confirm != FINAL_EVAL_CONFIRMATION:
        raise HoldoutAccessDenied(
            "get_locked_holdout_ids() called without the correct confirmation "
            "phrase. Denied -- this is not accidental access."
        )

    verify_split_integrity()
    return frozenset(_read_ids(HOLDOUT_PATH))


def _log_holdout_access_attempt(caller_file: str, allowed: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not AUDIT_LOG.exists()
    with open(AUDIT_LOG, "a") as f:
        if is_new:
            f.write("timestamp_utc,event_type,experiment_name,n_subjects,"
                    "subject_id_hash,holdout_clean,caller_file,notes\n")
        f.write(
            f"{datetime.now(timezone.utc).isoformat()},HOLDOUT_ACCESS_ATTEMPT,"
            f"-,-,-,{'ALLOWED' if allowed else 'DENIED'},{caller_file},"
            f"get_locked_holdout_ids call\n"
        )


def log_experiment(experiment_name: str, subject_ids, notes: str = "") -> None:
    """
    Append-only audit record of which subjects an experiment actually used.
    Automatically re-checks for holdout leakage (defense in depth -- an
    experiment that already passed assert_no_holdout_leakage should never
    trip this, but if it does, this function raises too rather than silently
    logging a contaminated run).
    """
    ids = [str(s) for s in subject_ids]
    assert_no_holdout_leakage(ids, context=f"log_experiment:{experiment_name}")

    id_hash = hashlib.sha256(",".join(sorted(ids)).encode()).hexdigest()[:16]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not AUDIT_LOG.exists()
    with open(AUDIT_LOG, "a") as f:
        if is_new:
            f.write("timestamp_utc,event_type,experiment_name,n_subjects,"
                    "subject_id_hash,holdout_clean,caller_file,notes\n")
        caller_file = os.path.basename(inspect.stack()[1].filename)
        f.write(
            f"{datetime.now(timezone.utc).isoformat()},EXPERIMENT,"
            f"{experiment_name},{len(ids)},{id_hash},CLEAN,{caller_file},"
            f"{notes}\n"
        )


if __name__ == "__main__":
    report = verify_split_integrity()
    print("Split integrity check passed:")
    for k, v in report.items():
        print(f"  {k}: {v}")
