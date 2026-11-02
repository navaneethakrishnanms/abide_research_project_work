"""
phase2a_create_split.py -- Create the NEW, untouched Phase 2 final holdout.

Per the Phase 1 audit, the original 170-subject holdout (random_state=42,
label-only stratification) was evaluated against by multiple scripts during
development (train_fc_improved.py, train_fc_v2.py, fast_holdout_eval.py,
compute_ensemble_holdout.py) and can no longer be treated as a blind test set.

This script draws a FRESH split from the full 846-subject FC cohort, stratified
jointly by (site, diagnosis) so that per-site ASD/TC proportions are preserved
in both partitions -- unlike the original split, which stratified by label only
and produced holdout site proportions ranging from 0% to 54.5% of a site's
subjects (see Phase 1 audit, Section 7).

This script ONLY creates and saves the split. It does not touch any model,
feature, or preprocessing code. Nothing is trained or evaluated here.

Outputs (splits/):
    phase2_train_subjects.txt      -- development subject IDs, one per line
    phase2_holdout_subjects.txt    -- final holdout subject IDs, one per line
    phase2_split_metadata.json     -- full provenance: seed, method, counts,
                                       per-site breakdown, source data hashes
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA        = r"C:\abide\abide_pcp"
TS_PATH     = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH   = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH  = os.path.join(DATA, "phenotypic_qc.csv")
SPLITS_DIR  = r"C:\abide\splits"

MIN_TIMEPOINTS = 100          # identical filter to train_fc_v2.py / fast_holdout_eval.py
PHASE2_SEED    = 20260        # deliberately different from the old SEED=42
HOLDOUT_FRAC   = 0.20

TRAIN_OUT   = os.path.join(SPLITS_DIR, "phase2_train_subjects.txt")
HOLDOUT_OUT = os.path.join(SPLITS_DIR, "phase2_holdout_subjects.txt")
META_OUT    = os.path.join(SPLITS_DIR, "phase2_split_metadata.json")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # ── Load the same 846-subject cohort used throughout the FC pipeline ──────
    ts = np.load(TS_PATH, allow_pickle=True)
    ph = pd.read_csv(PHENO_PATH)
    keep = np.array([t is not None and t.shape[0] >= MIN_TIMEPOINTS for t in ts])
    ts = ts[keep]
    ph = ph.loc[keep].reset_index(drop=True)

    assert ph["SUB_ID"].is_unique, "Duplicate SUB_ID found -- aborting split."

    sub_id = ph["SUB_ID"].astype(str).values
    site   = ph["SITE_ID"].astype(str).values
    dx     = ph["DX_GROUP"].values            # 1 = ASD, 2 = TC
    y      = (dx == 1).astype(int)

    n = len(ph)
    print(f"Loaded {n} subjects (identical filter to the existing FC pipeline)")
    print(f"  ASD: {y.sum()}  |  TC: {(1 - y).sum()}  |  sites: {pd.unique(site).size}")

    # ── Joint (site, diagnosis) stratification key ─────────────────────────────
    strata = pd.Series(site).astype(str) + "|" + pd.Series(dx).astype(str)
    strata_counts = strata.value_counts()
    print(f"\n{len(strata_counts)} (site, diagnosis) strata; "
          f"smallest stratum has {strata_counts.min()} subjects.")
    if strata_counts.min() < 2:
        raise SystemExit(
            "A (site, diagnosis) stratum has < 2 subjects -- sklearn's "
            "stratified split cannot split it. Aborting rather than silently "
            "falling back to a coarser stratification."
        )

    all_idx = np.arange(n)
    train_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=HOLDOUT_FRAC,
        stratify=strata.values,
        random_state=PHASE2_SEED,
    )

    train_ids   = sub_id[train_idx]
    holdout_ids = sub_id[holdout_idx]

    # ── Verification ─────────────────────────────────────────────────────────
    overlap = set(train_ids) & set(holdout_ids)
    assert len(overlap) == 0, f"Subject overlap detected: {overlap}"
    assert len(set(train_ids)) == len(train_ids), "Duplicate ID in train split"
    assert len(set(holdout_ids)) == len(holdout_ids), "Duplicate ID in holdout split"
    assert len(train_ids) + len(holdout_ids) == n

    print(f"\nDevelopment (train): {len(train_idx)} subjects "
          f"(ASD {y[train_idx].sum()} | TC {(1 - y[train_idx]).sum()})")
    print(f"Final holdout       : {len(holdout_idx)} subjects "
          f"(ASD {y[holdout_idx].sum()} | TC {(1 - y[holdout_idx]).sum()})")
    print(f"Subject overlap check: {len(overlap)} shared IDs (must be 0)")

    # ── Site distribution table ─────────────────────────────────────────────
    site_train = pd.Series(site[train_idx]).value_counts()
    site_hold  = pd.Series(site[holdout_idx]).value_counts()
    site_table = pd.DataFrame({"train_dev": site_train, "final_holdout": site_hold}).fillna(0).astype(int)
    site_table["holdout_pct_of_site"] = (
        site_table["final_holdout"] /
        (site_table["train_dev"] + site_table["final_holdout"]) * 100
    ).round(1)

    # Per-site, per-class breakdown too, so ASD/TC proportion-per-site can be checked
    df = pd.DataFrame({"sub_id": sub_id, "site": site, "y": y})
    df_tr = df.loc[train_idx]
    df_ho = df.loc[holdout_idx]
    per_site_class = pd.DataFrame({
        "train_ASD": df_tr[df_tr.y == 1].groupby("site").size(),
        "train_TC":  df_tr[df_tr.y == 0].groupby("site").size(),
        "holdout_ASD": df_ho[df_ho.y == 1].groupby("site").size(),
        "holdout_TC":  df_ho[df_ho.y == 0].groupby("site").size(),
    }).fillna(0).astype(int)

    print("\nSite distribution (train_dev vs final_holdout):")
    print(site_table.sort_index().to_string())

    print("\nPer-site class breakdown:")
    print(per_site_class.sort_index().to_string())

    sites_missing_from_holdout = set(np.unique(site[train_idx])) - set(np.unique(site[holdout_idx]))
    sites_missing_from_train   = set(np.unique(site[holdout_idx])) - set(np.unique(site[train_idx]))
    print(f"\nSites absent from holdout: {sites_missing_from_holdout or 'none'}")
    print(f"Sites absent from train:   {sites_missing_from_train or 'none'}")

    # ── Save subject ID lists ───────────────────────────────────────────────
    with open(TRAIN_OUT, "w") as f:
        f.write("\n".join(str(s) for s in sorted(train_ids)) + "\n")
    with open(HOLDOUT_OUT, "w") as f:
        f.write("\n".join(str(s) for s in sorted(holdout_ids)) + "\n")

    # ── Save metadata (full provenance, reproducibility, site table) ───────
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 2 final locked holdout -- superseudes the reused "
                   "170-subject holdout (random_state=42) identified as "
                   "compromised by the Phase 1 leakage audit.",
        "source_files": {
            "phenotypic_csv": PHENO_PATH,
            "phenotypic_csv_sha256": file_sha256(PHENO_PATH),
            "ts_cc200_npy": TS_PATH,
            "ts_cc200_npy_sha256": file_sha256(TS_PATH),
        },
        "filter": f"timepoints >= {MIN_TIMEPOINTS}",
        "n_total": int(n),
        "split_method": "sklearn.model_selection.train_test_split, "
                         "stratify=site|diagnosis joint key",
        "random_state": PHASE2_SEED,
        "holdout_fraction": HOLDOUT_FRAC,
        "n_train_dev": int(len(train_idx)),
        "n_final_holdout": int(len(holdout_idx)),
        "train_dev_asd": int(y[train_idx].sum()),
        "train_dev_tc": int((1 - y[train_idx]).sum()),
        "final_holdout_asd": int(y[holdout_idx].sum()),
        "final_holdout_tc": int((1 - y[holdout_idx]).sum()),
        "subject_overlap_count": int(len(overlap)),
        "n_sites": int(pd.unique(site).size),
        "sites_missing_from_holdout": sorted(sites_missing_from_holdout),
        "sites_missing_from_train": sorted(sites_missing_from_train),
        "site_distribution": site_table.reset_index().rename(
            columns={"index": "site"}).to_dict(orient="records"),
        "status": "LOCKED -- do not use n_final_holdout subjects for anything "
                  "other than the single final Phase 2N evaluation.",
        "old_holdout_note": "The original 170-subject holdout (seed=42, "
                             "label-only stratified) is DEMOTED to a "
                             "development/reference set as of this file's "
                             "creation. Its historical result (74.12% acc, "
                             "0.8104 AUC) remains valid as a reference number "
                             "but must be labeled 'previously evaluated/reused "
                             "development holdout', not a final result.",
    }
    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:")
    print(f"  {TRAIN_OUT}")
    print(f"  {HOLDOUT_OUT}")
    print(f"  {META_OUT}")


if __name__ == "__main__":
    main()
