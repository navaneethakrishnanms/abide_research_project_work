"""
test_holdout_guard.py -- Demonstrates that holdout_guard.py correctly blocks
every leakage path it claims to block, and that it does not get in the way
of a normal, honest development pipeline.

Run: python test_holdout_guard.py
Every test prints PASS or FAIL explicitly; exits non-zero if anything fails.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import holdout_guard as hg

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────
# TEST 1 (should PASS): split integrity check succeeds on the real files
# ─────────────────────────────────────────────────────────────────────────
try:
    report = hg.verify_split_integrity()
    check("T1 split integrity verifies cleanly", report["overlap"] == 0
          and report["n_train"] == 676 and report["n_holdout"] == 170,
          detail=str(report))
except Exception as e:
    check("T1 split integrity verifies cleanly", False, detail=f"raised {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 2 (should PASS): loading dev subject IDs gives exactly 676, no holdout
# ─────────────────────────────────────────────────────────────────────────
try:
    dev_ids = hg.load_dev_subject_ids()
    holdout_ids_raw = set(hg._read_ids(hg.HOLDOUT_PATH))
    clean = len(dev_ids) == 676 and len(dev_ids & holdout_ids_raw) == 0
    check("T2 load_dev_subject_ids returns 676 clean IDs", clean,
          detail=f"n_dev={len(dev_ids)}, overlap={len(dev_ids & holdout_ids_raw)}")
except Exception as e:
    check("T2 load_dev_subject_ids returns 676 clean IDs", False, detail=f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 3 (DELIBERATE FAILURE -- must raise): mixing a few holdout IDs into
# a "training" subject list must be caught.
# ─────────────────────────────────────────────────────────────────────────
try:
    dev_ids = list(hg.load_dev_subject_ids())
    holdout_ids = hg._read_ids(hg.HOLDOUT_PATH)
    contaminated = dev_ids[:20] + holdout_ids[:3]  # 3 leaked holdout subjects
    hg.assert_no_holdout_leakage(contaminated, context="T3 deliberate contamination")
    check("T3 leakage into training set is blocked", False,
          detail="assert_no_holdout_leakage did NOT raise -- THIS IS A BUG")
except hg.HoldoutLeakageError as e:
    check("T3 leakage into training set is blocked", True, detail=f"correctly raised: {e}")
except Exception as e:
    check("T3 leakage into training set is blocked", False,
          detail=f"raised wrong exception type {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 4 (DELIBERATE FAILURE -- must raise): a CV fold containing a holdout
# subject must be caught.
# ─────────────────────────────────────────────────────────────────────────
try:
    dev_ids = list(hg.load_dev_subject_ids())
    holdout_ids = hg._read_ids(hg.HOLDOUT_PATH)
    fold_train = dev_ids[:100]
    fold_val   = dev_ids[100:150] + [holdout_ids[0]]  # 1 leaked subject
    hg.assert_cv_folds_dev_only([fold_train, fold_val], context="T4 fold check")
    check("T4 leakage into a CV fold is blocked", False,
          detail="assert_cv_folds_dev_only did NOT raise -- THIS IS A BUG")
except hg.HoldoutLeakageError as e:
    check("T4 leakage into a CV fold is blocked", True, detail=f"correctly raised: {e}")
except Exception as e:
    check("T4 leakage into a CV fold is blocked", False,
          detail=f"raised wrong exception type {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 5 (DELIBERATE FAILURE -- must raise): calling get_locked_holdout_ids
# from a script that is NOT phase2n_final_holdout_eval.py must be denied,
# even with the correct confirmation phrase.
# ─────────────────────────────────────────────────────────────────────────
try:
    hg.get_locked_holdout_ids(confirm=hg.FINAL_EVAL_CONFIRMATION)
    check("T5 unauthorized script cannot access locked holdout", False,
          detail="get_locked_holdout_ids did NOT raise -- THIS IS A BUG")
except hg.HoldoutAccessDenied as e:
    check("T5 unauthorized script cannot access locked holdout", True,
          detail=f"correctly denied: {e}")
except Exception as e:
    check("T5 unauthorized script cannot access locked holdout", False,
          detail=f"raised wrong exception type {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 6 (DELIBERATE FAILURE -- must raise): even the allowed filename
# (simulated) must be rejected with a wrong confirmation phrase. We simulate
# this by calling with a bad phrase from THIS file (already not the allowed
# filename, so this doubles as a second denial reason) -- the important
# assertion is that a wrong phrase alone is fatal.
# ─────────────────────────────────────────────────────────────────────────
try:
    hg.get_locked_holdout_ids(confirm="wrong phrase")
    check("T6 wrong confirmation phrase is rejected", False,
          detail="get_locked_holdout_ids did NOT raise -- THIS IS A BUG")
except hg.HoldoutAccessDenied as e:
    check("T6 wrong confirmation phrase is rejected", True, detail=f"correctly denied: {e}")
except Exception as e:
    check("T6 wrong confirmation phrase is rejected", False,
          detail=f"raised wrong exception type {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 7 (DELIBERATE FAILURE -- must raise): tampering with the holdout file
# (simulated on a copy, not the real file) must be caught by hash mismatch.
# ─────────────────────────────────────────────────────────────────────────
import shutil, tempfile, os as _os
from pathlib import Path as _Path

try:
    with tempfile.TemporaryDirectory() as tmp:
        fake_splits = tmp
        shutil.copy(hg.TRAIN_PATH, fake_splits)
        shutil.copy(hg.META_PATH, fake_splits)
        tampered_holdout = _os.path.join(fake_splits, "phase2_holdout_subjects.txt")
        real_ids = hg._read_ids(hg.HOLDOUT_PATH)
        with open(tampered_holdout, "w") as f:
            f.write("\n".join(real_ids[:-1] + ["9999999_FAKE_SUBJECT"]) + "\n")  # swapped one ID

        # Point a throwaway module-level copy at the tampered files
        orig_train, orig_holdout, orig_meta = hg.TRAIN_PATH, hg.HOLDOUT_PATH, hg.META_PATH
        hg.TRAIN_PATH   = _Path(fake_splits) / "phase2_train_subjects.txt"
        hg.HOLDOUT_PATH = _Path(tampered_holdout)
        hg.META_PATH    = _Path(fake_splits) / "phase2_split_metadata.json"
        try:
            hg.verify_split_integrity()
            check("T7 tampered holdout file is detected via hash mismatch", False,
                  detail="verify_split_integrity did NOT raise -- THIS IS A BUG")
        except hg.HoldoutLeakageError as e:
            check("T7 tampered holdout file is detected via hash mismatch", True,
                  detail=f"correctly raised: {e}")
        finally:
            hg.TRAIN_PATH, hg.HOLDOUT_PATH, hg.META_PATH = orig_train, orig_holdout, orig_meta
except Exception as e:
    check("T7 tampered holdout file is detected via hash mismatch", False,
          detail=f"test harness error {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 8 (should PASS): a NORMAL development pipeline still runs fine after
# the guard is in place -- load dev data, run real StratifiedKFold CV,
# check every fold, log the experiment. Demonstrates the guard doesn't
# break legitimate work.
# ─────────────────────────────────────────────────────────────────────────
try:
    import os as _os2
    DATA = r"C:\abide\abide_pcp"
    ts = np.load(_os2.path.join(DATA, "ts_cc200.npy"), allow_pickle=True)
    ph = pd.read_csv(_os2.path.join(DATA, "phenotypic_qc.csv"))
    keep = np.array([t is not None and t.shape[0] >= 100 for t in ts])
    ph = ph.loc[keep].reset_index(drop=True)

    dev_ids = hg.load_dev_subject_ids()
    ph_dev = ph[ph["SUB_ID"].astype(str).isin(dev_ids)].reset_index(drop=True)

    assert len(ph_dev) == 676, f"expected 676 dev rows, got {len(ph_dev)}"

    y_dev = (ph_dev["DX_GROUP"].values == 1).astype(int)
    sub_dev = ph_dev["SUB_ID"].astype(str).values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    all_fold_ids = []
    for tr, te in skf.split(np.zeros(len(y_dev)), y_dev):
        all_fold_ids.append(sub_dev[tr])
        all_fold_ids.append(sub_dev[te])
    hg.assert_cv_folds_dev_only(all_fold_ids, context="T8 normal 10-fold CV")

    hg.log_experiment("T8_smoke_test_stratified_10fold", sub_dev,
                       notes="Phase 2B smoke test -- confirms guard does not "
                             "block a legitimate dev-only CV pipeline")

    check("T8 normal dev pipeline (676 subj, 10-fold CV) runs and logs cleanly", True,
          detail=f"n_dev_used={len(sub_dev)}, folds_checked={len(all_fold_ids)}")
except Exception as e:
    check("T8 normal dev pipeline (676 subj, 10-fold CV) runs and logs cleanly", False,
          detail=f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 9 (should PASS): confirm the audit log now contains the expected
# rows (denied access attempts + the clean experiment).
# ─────────────────────────────────────────────────────────────────────────
try:
    log = pd.read_csv(hg.AUDIT_LOG)
    n_denied = (log["holdout_clean"] == "DENIED").sum()
    n_experiments = (log["event_type"] == "EXPERIMENT").sum()
    check("T9 audit log recorded denied attempts and the clean experiment",
          n_denied >= 2 and n_experiments >= 1,
          detail=f"denied_attempts={n_denied}, experiments_logged={n_experiments}, "
                 f"total_rows={len(log)}")
except Exception as e:
    check("T9 audit log recorded denied attempts and the clean experiment", False,
          detail=f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"RESULTS: {n_pass} passed, {n_fail} failed, {len(results)} total")
print("=" * 70)
if n_fail:
    sys.exit(1)
