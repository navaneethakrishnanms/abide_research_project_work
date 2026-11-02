"""
phase2_fd_filter.py -- Phase 2 (paper-reproduction track): apply the paper's
FD > 0.2mm motion-quality exclusion to the Phase 1 871-subject cohort.

Metadata-only. No fMRI volumes touched, no functional connectivity computed,
no AAL extraction, no training. Reads results/phase1/subject_manifest.csv
(read-only) plus abide_pcp/phenotypic_qc.csv for func_mean_fd, and writes
only to results/phase2/ -- Phase 1 outputs and the unrelated ML results/
phase2c..phase2h_*.csv files (a different project track) are never opened
for writing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE1_MANIFEST = ROOT / "results" / "phase1" / "subject_manifest.csv"
PHENO_QC_PATH   = ROOT / "abide_pcp" / "phenotypic_qc.csv"

OUT_DIR = ROOT / "results" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD_THRESHOLD = 0.2  # paper's stated threshold, exact -- do not change

audit_lines = []


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg)


def main():
    log("Phase 2 FD filtering started (metadata-only, no fMRI processing).")
    log(f"Loading Phase 1 manifest: {PHASE1_MANIFEST}")

    manifest = pd.read_csv(PHASE1_MANIFEST, dtype={"subject_id": str})
    n_initial = len(manifest)
    log(f"Phase 1 manifest loaded: {n_initial} subjects (expected 871)")
    if n_initial != 871:
        log(f"[WARNING] Expected 871 subjects from Phase 1, got {n_initial}. Proceeding with actual count.")

    initial_asd = int((manifest["diagnosis"] == "ASD").sum())
    initial_control = int((manifest["diagnosis"] == "Control").sum())
    log(f"Initial cohort: total={n_initial}, ASD={initial_asd}, Control={initial_control}")

    # ── Pull func_mean_fd from the QC source and join onto the Phase 1 manifest ──
    log(f"Loading FD source: {PHENO_QC_PATH}")
    ph_qc = pd.read_csv(PHENO_QC_PATH, dtype={"SUB_ID": str})
    ph_qc["SUB_ID"] = ph_qc["SUB_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    fd_map = ph_qc.set_index("SUB_ID")["func_mean_fd"]

    manifest["subject_id"] = manifest["subject_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    manifest["func_mean_fd_raw"] = manifest["subject_id"].map(fd_map)

    n_joined = manifest["func_mean_fd_raw"].notna().sum()
    log(f"func_mean_fd joined for {n_joined} / {n_initial} subjects via SUB_ID match")
    if n_joined < n_initial:
        unmatched = manifest.loc[manifest["func_mean_fd_raw"].isna(), "subject_id"].tolist()
        log(f"[WARNING] {n_initial - n_joined} subject(s) had no matching row in phenotypic_qc.csv "
            f"by SUB_ID -- treated as missing FD, not silently dropped: {unmatched[:20]}"
            f"{'...' if len(unmatched) > 20 else ''}")

    # ── Classify FD values explicitly: numeric-valid, NaN/missing, infinite, non-numeric ──
    def classify_fd(raw):
        if pd.isna(raw):
            return "missing", np.nan
        try:
            v = float(raw)
        except (ValueError, TypeError):
            return "invalid_non_numeric", np.nan
        if np.isinf(v):
            return "infinite", np.nan
        return "valid", v

    classes, values = zip(*(classify_fd(v) for v in manifest["func_mean_fd_raw"]))
    manifest["fd_class"] = classes
    manifest["func_mean_fd"] = values

    n_missing  = int((manifest["fd_class"] == "missing").sum())
    n_invalid  = int((manifest["fd_class"] == "invalid_non_numeric").sum())
    n_infinite = int((manifest["fd_class"] == "infinite").sum())
    n_valid    = int((manifest["fd_class"] == "valid").sum())
    log(f"FD value classification: valid={n_valid}, missing={n_missing}, "
        f"invalid_non_numeric={n_invalid}, infinite={n_infinite}")
    assert n_valid + n_missing + n_invalid + n_infinite == n_initial

    # ── FD distribution stats (valid values only) ────────────────────────
    valid_fd = manifest.loc[manifest["fd_class"] == "valid", "func_mean_fd"]
    fd_stats = {
        "min": float(valid_fd.min()), "max": float(valid_fd.max()),
        "mean": float(valid_fd.mean()), "median": float(valid_fd.median()),
        "std": float(valid_fd.std()),
        "n_le_0.2": int((valid_fd <= FD_THRESHOLD).sum()),
        "n_gt_0.2": int((valid_fd > FD_THRESHOLD).sum()),
        "n_missing": n_missing, "n_invalid_non_numeric": n_invalid, "n_infinite": n_infinite,
    }
    log(f"FD stats (valid values, n={n_valid}): min={fd_stats['min']:.4f} max={fd_stats['max']:.4f} "
        f"mean={fd_stats['mean']:.4f} median={fd_stats['median']:.4f} std={fd_stats['std']:.4f}")
    log(f"  <= {FD_THRESHOLD}: {fd_stats['n_le_0.2']}   > {FD_THRESHOLD}: {fd_stats['n_gt_0.2']}")

    manifest.to_csv(OUT_DIR / "fd_distribution.csv", index=False,
                     columns=["subject_id", "diagnosis", "site", "age", "sex", "func_mean_fd", "fd_class"])
    log(f"fd_distribution.csv written -> {OUT_DIR / 'fd_distribution.csv'}")

    # ── Classify each subject: retained / removed / unresolved ──────────
    def decide(row):
        if row["fd_class"] == "valid":
            return "retained" if row["func_mean_fd"] <= FD_THRESHOLD else "removed_fd_high"
        return "unresolved"

    manifest["fd_decision"] = manifest.apply(decide, axis=1)

    retained  = manifest[manifest["fd_decision"] == "retained"].copy()
    removed   = manifest[manifest["fd_decision"] == "removed_fd_high"].copy()
    unresolved = manifest[manifest["fd_decision"] == "unresolved"].copy()

    # removal_reason for the removed-subject CSV
    removed["removal_reason"] = f"FD > {FD_THRESHOLD} mm"
    unresolved_reason_map = {"missing": "FD value missing", "invalid_non_numeric": "FD value non-numeric/invalid",
                              "infinite": "FD value infinite"}
    unresolved["removal_reason"] = unresolved["fd_class"].map(unresolved_reason_map)

    # ── STEP 8 audit checks (before writing anything else) ───────────────
    n_retained, n_removed, n_unresolved = len(retained), len(removed), len(unresolved)
    ids_all = set(manifest["subject_id"])
    ids_retained = set(retained["subject_id"])
    ids_removed = set(removed["subject_id"])
    ids_unresolved = set(unresolved["subject_id"])

    assert ids_retained <= ids_all and ids_removed <= ids_all and ids_unresolved <= ids_all, \
        "Retained/removed/unresolved subjects must all originate from the Phase 1 cohort."
    assert len(ids_retained & ids_removed) == 0, "Retained/removed overlap detected."
    assert len(ids_retained & ids_unresolved) == 0, "Retained/unresolved overlap detected."
    assert len(ids_removed & ids_unresolved) == 0, "Removed/unresolved overlap detected."
    assert n_retained + n_removed + n_unresolved == n_initial, \
        "retained + removed + unresolved must equal the initial cohort size."
    assert manifest["subject_id"].duplicated().sum() == 0, "Duplicate subject_id detected post-join."
    # Diagnosis-unchanged check: compare against the original Phase 1 manifest, re-loaded fresh
    ph1_check = pd.read_csv(PHASE1_MANIFEST, dtype={"subject_id": str})
    ph1_check["subject_id"] = ph1_check["subject_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    dx_check = manifest.merge(ph1_check[["subject_id", "diagnosis"]], on="subject_id",
                               suffixes=("", "_phase1"))
    assert (dx_check["diagnosis"] == dx_check["diagnosis_phase1"]).all(), \
        "Diagnosis label mismatch vs Phase 1 manifest -- labels must never be altered."
    log("AUDIT PASSED: retained/removed/unresolved are disjoint, sum to initial cohort, "
        "all subject IDs trace back to the Phase 1 871-subject cohort, no duplicate IDs, "
        "no diagnosis labels altered.")

    retained_asd = int((retained["diagnosis"] == "ASD").sum())
    retained_control = int((retained["diagnosis"] == "Control").sum())
    removed_asd = int((removed["diagnosis"] == "ASD").sum())
    removed_control = int((removed["diagnosis"] == "Control").sum())
    unresolved_asd = int((unresolved["diagnosis"] == "ASD").sum())
    unresolved_control = int((unresolved["diagnosis"] == "Control").sum())

    retention_pct = round(100 * n_retained / n_initial, 2)
    removal_pct = round(100 * n_removed / n_initial, 2)
    unresolved_pct = round(100 * n_unresolved / n_initial, 2)

    log(f"Retained: total={n_retained}, ASD={retained_asd}, Control={retained_control}")
    log(f"Removed (FD>{FD_THRESHOLD}): total={n_removed}, ASD={removed_asd}, Control={removed_control}")
    log(f"Unresolved (missing/invalid/infinite FD): total={n_unresolved}, "
        f"ASD={unresolved_asd}, Control={unresolved_control}")
    log(f"Retention: {retention_pct}%  Removal: {removal_pct}%  Unresolved: {unresolved_pct}%")

    # ── STEP 3 / 4 outputs ────────────────────────────────────────────────
    keep_cols = ["subject_id", "diagnosis", "site", "age", "sex", "func_mean_fd",
                 "fmri_rois_cc200_available", "preprocessed_available", "fd_qc_available"]
    keep_cols = [c for c in keep_cols if c in retained.columns]
    retained[keep_cols].to_csv(OUT_DIR / "fd_filtered_subjects.csv", index=False)
    log(f"fd_filtered_subjects.csv written: {n_retained} subjects -> {OUT_DIR / 'fd_filtered_subjects.csv'}")

    removed_cols = ["subject_id", "diagnosis", "site", "age", "sex", "func_mean_fd", "removal_reason"]
    removed[removed_cols].to_csv(OUT_DIR / "fd_removed_subjects.csv", index=False)
    log(f"fd_removed_subjects.csv written: {n_removed} subjects -> {OUT_DIR / 'fd_removed_subjects.csv'}")

    unresolved[removed_cols].to_csv(OUT_DIR / "fd_unresolved_subjects.csv", index=False)
    log(f"fd_unresolved_subjects.csv written: {n_unresolved} subjects -> {OUT_DIR / 'fd_unresolved_subjects.csv'}")

    # ── STEP 6 site analysis ─────────────────────────────────────────────
    site_initial = manifest["site"].value_counts()
    site_retained = retained["site"].value_counts()
    site_removed = removed["site"].value_counts()
    site_unresolved = unresolved["site"].value_counts()
    site_table = pd.DataFrame({
        "initial_n": site_initial, "retained_n": site_retained,
        "removed_n": site_removed, "unresolved_n": site_unresolved,
    }).fillna(0).astype(int).sort_index()
    site_table_path = OUT_DIR / "fd_site_summary.csv"
    site_table.to_csv(site_table_path)
    log(f"Site summary written -> {site_table_path}")
    log("Site table (site | initial_n | retained_n | removed_n | unresolved_n):")
    for site, row in site_table.iterrows():
        log(f"  {site:<10} {row['initial_n']:>4} {row['retained_n']:>4} {row['removed_n']:>4} {row['unresolved_n']:>4}")

    # ── STEP 7 age analysis (no filtering applied) ───────────────────────
    age_initial = manifest["age"].dropna()
    age_retained = retained["age"].dropna()
    age_summary = {
        "initial_age_min": float(age_initial.min()), "initial_age_max": float(age_initial.max()),
        "post_fd_age_min": float(age_retained.min()) if len(age_retained) else None,
        "post_fd_age_max": float(age_retained.max()) if len(age_retained) else None,
    }
    log(f"Age range -- initial: {age_summary['initial_age_min']}-{age_summary['initial_age_max']}  "
        f"post-FD: {age_summary['post_fd_age_min']}-{age_summary['post_fd_age_max']} (no age filter applied)")

    # ── STEP 5 summary JSON ───────────────────────────────────────────────
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fd_threshold_mm": FD_THRESHOLD,
        "source_phase1_manifest": str(PHASE1_MANIFEST),
        "source_fd_column": str(PHENO_QC_PATH) + "::func_mean_fd",
        "initial": {"total": n_initial, "asd": initial_asd, "control": initial_control},
        "retained": {"total": n_retained, "asd": retained_asd, "control": retained_control},
        "removed": {"total": n_removed, "asd": removed_asd, "control": removed_control},
        "unresolved": {"total": n_unresolved, "asd": unresolved_asd, "control": unresolved_control},
        "retention_percent": retention_pct,
        "removal_percent": removal_pct,
        "unresolved_percent": unresolved_pct,
        "fd_statistics": fd_stats,
        "age_summary": age_summary,
        "site_summary": site_table.reset_index().rename(columns={"index": "site"}).to_dict(orient="records"),
        "audit": {
            "retained_removed_overlap": len(ids_retained & ids_removed),
            "retained_unresolved_overlap": len(ids_retained & ids_unresolved),
            "removed_unresolved_overlap": len(ids_removed & ids_unresolved),
            "sum_matches_initial": (n_retained + n_removed + n_unresolved == n_initial),
            "duplicate_subject_ids": int(manifest["subject_id"].duplicated().sum()),
            "diagnosis_labels_altered": False,
            "phase1_files_modified": False,
            "previous_project_results_modified": False,
        },
    }
    summary_path = OUT_DIR / "fd_filter_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"fd_filter_summary.json written -> {summary_path}")

    # ── phase2_summary.txt ────────────────────────────────────────────────
    lines = [
        "PHASE 2 SUMMARY -- FD > 0.2mm Motion-Quality Filtering",
        "=" * 60, "",
        "INITIAL COHORT",
        f"  Total = {n_initial}", f"  ASD = {initial_asd}", f"  Controls = {initial_control}", "",
        "AFTER FD FILTERING",
        f"  Total = {n_retained}", f"  ASD = {retained_asd}", f"  Controls = {retained_control}", "",
        "REMOVED (FD > 0.2mm)",
        f"  Total = {n_removed}", f"  ASD = {removed_asd}", f"  Controls = {removed_control}", "",
        "UNRESOLVED (missing/invalid/infinite FD)",
        f"  Total = {n_unresolved}", f"  ASD = {unresolved_asd}", f"  Controls = {unresolved_control}", "",
        f"Retention percentage: {retention_pct}%",
        f"Removal percentage: {removal_pct}%",
        f"Unresolved percentage: {unresolved_pct}%", "",
        "FD STATISTICS (valid values only)",
        f"  Min = {fd_stats['min']:.4f}", f"  Max = {fd_stats['max']:.4f}",
        f"  Mean = {fd_stats['mean']:.4f}", f"  Median = {fd_stats['median']:.4f}",
        f"  Std = {fd_stats['std']:.4f}",
        f"  <=0.2: {fd_stats['n_le_0.2']}   >0.2: {fd_stats['n_gt_0.2']}   "
        f"missing: {fd_stats['n_missing']}   invalid: {fd_stats['n_invalid_non_numeric']}   "
        f"infinite: {fd_stats['n_infinite']}", "",
        "AGE",
        f"  Initial range: {age_summary['initial_age_min']} - {age_summary['initial_age_max']}",
        f"  Post-FD range: {age_summary['post_fd_age_min']} - {age_summary['post_fd_age_max']}",
        "  (No age filtering applied.)",
    ]
    with open(OUT_DIR / "phase2_summary.txt", "w") as f:
        f.write("\n".join(lines))
    log(f"phase2_summary.txt written -> {OUT_DIR / 'phase2_summary.txt'}")

    audit_path = OUT_DIR / "phase2_audit.log"
    audit_lines.append(f"Output files: fd_filtered_subjects.csv, fd_removed_subjects.csv, "
                        f"fd_unresolved_subjects.csv, fd_distribution.csv, fd_site_summary.csv, "
                        f"fd_filter_summary.json, phase2_summary.txt")
    with open(audit_path, "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 2 outputs written to {OUT_DIR}")

    return summary


if __name__ == "__main__":
    main()
