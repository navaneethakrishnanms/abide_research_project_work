"""
phase1_inventory.py -- Phase 1 (paper-target reproduction project): fast,
read-only inventory of existing ABIDE-I data. Does NOT download, preprocess,
train, or compute functional connectivity. Only inspects files/metadata
already on disk and builds a subject-level manifest.

Outputs (results/phase1/, a namespace separate from the ongoing results/
phase2*.csv work -- nothing there is read, modified, or deleted):
    dataset_inventory.json
    subject_manifest.csv
    available_subjects.csv
    phase1_summary.txt
    phase1_audit.log
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHENO_QC_PATH   = ROOT / "abide_pcp" / "phenotypic_qc.csv"
PHENO_ROOT_PATH = ROOT / "Phenotypic_V1_0b.csv"
TS_PATH         = ROOT / "abide_pcp" / "ts_cc200.npy"
COVS_PATH       = ROOT / "abide_pcp" / "covs_cc200.npy"
CPAC_DIR        = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
UCLA_RAW_DIR    = ROOT / "UCLA_1" / "UCLA_1"
MORPH_CSV       = ROOT / "derivatives" / "morphometry.csv"

OUT_DIR = ROOT / "results" / "phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

audit_lines = []


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg)


def main():
    t0 = time.time()
    log("Phase 1 inventory started.")
    log(f"Files inspected: {PHENO_QC_PATH}, {PHENO_ROOT_PATH}, {TS_PATH}, {COVS_PATH}, "
        f"{CPAC_DIR}, {UCLA_RAW_DIR}, {MORPH_CSV}")

    # ── Load the two candidate phenotypic sources ───────────────────────────
    ph_qc = pd.read_csv(PHENO_QC_PATH) if PHENO_QC_PATH.exists() else None
    ph_root = pd.read_csv(PHENO_ROOT_PATH) if PHENO_ROOT_PATH.exists() else None

    log(f"phenotypic_qc.csv: {'FOUND' if ph_qc is not None else 'MISSING'} "
        f"({len(ph_qc) if ph_qc is not None else 0} rows)")
    log(f"Phenotypic_V1_0b.csv: {'FOUND' if ph_root is not None else 'MISSING'} "
        f"({len(ph_root) if ph_root is not None else 0} rows)")

    # phenotypic_qc.csv is the nilearn fetch_abide_pcp() output actually paired
    # with our downloaded CC200 derivatives (871 subjects, quality_checked=True,
    # CPAC pipeline) -- this is the dataset with real derivative data behind it,
    # so it is the primary source for the manifest. Phenotypic_V1_0b.csv (1112
    # rows) is the full raw ABIDE-I phenotypic file with no guarantee every row
    # has downloaded imaging data -- used only for cross-checking columns/counts.
    primary_source = "phenotypic_qc.csv (nilearn fetch_abide_pcp output, paired with downloaded CC200 derivatives)"
    ph = ph_qc if ph_qc is not None else ph_root
    log(f"Primary manifest source selected: {primary_source}")

    log(f"phenotypic_qc.csv columns ({len(ph_qc.columns)}): {list(ph_qc.columns)}" if ph_qc is not None else "")

    # ── QC / FD column detection (search both files, don't assume names) ────
    fd_candidates = ["func_mean_fd", "func_num_fd", "func_perc_fd", "MeanFD", "mean_fd"]
    found_fd_cols = {}
    for label, df in [("phenotypic_qc.csv", ph_qc), ("Phenotypic_V1_0b.csv", ph_root)]:
        if df is None:
            continue
        hits = [c for c in df.columns if any(cand.lower() in c.lower() for cand in fd_candidates)]
        # Also catch generic qc columns
        hits += [c for c in df.columns if "qc_" in c.lower() or c.lower().startswith("fd")]
        hits = sorted(set(hits))
        found_fd_cols[label] = hits
        log(f"FD/QC-like columns in {label}: {hits if hits else 'NONE FOUND'}")

    # ── Raw imaging / derivative availability ────────────────────────────────
    n_cc200_files = len(list(CPAC_DIR.glob("*_rois_cc200.1D"))) if CPAC_DIR.exists() else 0
    log(f"CC200 ROI .1D derivative files on disk: {n_cc200_files} (dir: {CPAC_DIR})")

    ts_shape = None
    if TS_PATH.exists():
        ts = np.load(TS_PATH, allow_pickle=True)
        ts_shape = (len(ts), "variable timepoints x 200 regions per subject")
        log(f"ts_cc200.npy: {len(ts)} subjects' ROI time series (precomputed, on disk)")
    covs_shape = None
    if COVS_PATH.exists():
        covs = np.load(COVS_PATH, mmap_mode="r")
        covs_shape = covs.shape
        log(f"covs_cc200.npy: shape {covs_shape} (precomputed covariances, on disk)")

    n_ucla_raw = 0
    if UCLA_RAW_DIR.exists():
        n_ucla_raw = len([p for p in UCLA_RAW_DIR.iterdir() if p.is_dir()])
        log(f"Raw NIfTI subject folders (UCLA_1 site only): {n_ucla_raw} -- "
            f"this is a single-site raw NIfTI cohort, NOT a full ABIDE-I multi-site raw dataset")

    n_morph = 0
    if MORPH_CSV.exists():
        morph_df = pd.read_csv(MORPH_CSV)
        n_morph = len(morph_df)
        log(f"morphometry.csv: {n_morph} subjects (structural/anatomical derived features, "
            f"separate pipeline, small cohort)")

    # ── Build the subject-level manifest from phenotypic_qc.csv ─────────────
    def col(df, *names):
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([np.nan] * len(df))

    sub_id   = col(ph, "SUB_ID").astype("Int64").astype(str)
    dx_group = col(ph, "DX_GROUP")
    site_id  = col(ph, "SITE_ID").astype(str)
    age      = pd.to_numeric(col(ph, "AGE_AT_SCAN"), errors="coerce").replace(-9999, np.nan)
    sex      = col(ph, "SEX")
    file_id  = col(ph, "FILE_ID").astype(str) if "FILE_ID" in ph.columns else pd.Series([""] * len(ph))

    diagnosis = dx_group.map({1: "ASD", 2: "Control"})
    n_invalid_dx = int(diagnosis.isna().sum() - dx_group.isna().sum())  # values present but not 1/2
    n_missing_dx = int(dx_group.isna().sum())

    has_cc200 = set()
    for p in CPAC_DIR.glob("*_rois_cc200.1D"):
        # filenames look like "<SITE>_<SUBID>_rois_cc200.1D" -- extract the numeric subject id
        stem = p.stem.replace("_rois_cc200", "")
        digits = "".join(ch for ch in stem.split("_")[-1] if ch.isdigit())
        if digits:
            has_cc200.add(str(int(digits)))

    manifest = pd.DataFrame({
        "subject_id": sub_id,
        "file_id": file_id,
        "diagnosis": diagnosis,
        "dx_group_raw": dx_group,
        "site": site_id,
        "age": age,
        "sex": sex,
        "fmri_rois_cc200_available": sub_id.isin(has_cc200),
        "preprocessed_available": sub_id.isin(has_cc200),  # CC200 ROI extraction IS the preprocessed derivative here
        "fd_qc_available": False,  # updated below if FD columns exist and are non-null per row
    })

    for label, cols in found_fd_cols.items():
        if label == "phenotypic_qc.csv" and cols:
            for c in cols:
                manifest["fd_qc_available"] = manifest["fd_qc_available"] | ph_qc[c].notna().values

    manifest_path = OUT_DIR / "subject_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    log(f"subject_manifest.csv written: {len(manifest)} rows -> {manifest_path}")

    # ── Duplicate check (report only, do not remove) ─────────────────────────
    total_rows = len(manifest)
    unique_subjects = manifest["subject_id"].nunique()
    dup_mask = manifest["subject_id"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    dup_ids = sorted(manifest.loc[dup_mask, "subject_id"].unique().tolist())
    log(f"Duplicate check: {total_rows} total rows, {unique_subjects} unique subject_id values, "
        f"{n_dup_rows} duplicated rows across {len(dup_ids)} duplicated subject_id(s): {dup_ids}")

    # ── Missing subject ID check ──────────────────────────────────────────
    n_missing_id = int(sub_id.isna().sum() | (sub_id == "nan")) if False else int((sub_id == "nan").sum())
    log(f"Rows with missing subject_id: {n_missing_id}")

    # ── Label counts ─────────────────────────────────────────────────────
    n_asd = int((diagnosis == "ASD").sum())
    n_control = int((diagnosis == "Control").sum())
    log(f"Diagnosis counts: ASD={n_asd}, Control={n_control}, missing={n_missing_dx}, invalid={n_invalid_dx}")

    # ── Availability manifest ────────────────────────────────────────────
    available = manifest[manifest["fmri_rois_cc200_available"]].copy()
    available_path = OUT_DIR / "available_subjects.csv"
    available.to_csv(available_path, index=False)
    log(f"available_subjects.csv written: {len(available)} subjects with CC200 ROI derivative data on disk")

    # ── Demographics ─────────────────────────────────────────────────────
    age_valid = age.dropna()
    age_min = float(age_valid.min()) if len(age_valid) else None
    age_max = float(age_valid.max()) if len(age_valid) else None
    n_missing_age = int(age.isna().sum())
    sex_counts = sex.value_counts(dropna=False).to_dict()
    n_missing_sex = int(sex.isna().sum())

    n_sites = int(site_id.nunique())
    site_counts = site_id.value_counts().to_dict()

    log(f"Age range: {age_min}-{age_max} (missing: {n_missing_age})")
    log(f"Sex distribution: {sex_counts} (missing: {n_missing_sex})")
    log(f"Sites: {n_sites} -- {sorted(site_counts.keys())}")

    # ── Paper comparison ─────────────────────────────────────────────────
    paper_target = {"total": 884, "asd": 408, "control": 476, "sites": 17,
                     "age_range": [7, 64], "countries": 5}
    match_total = (total_rows == paper_target["total"])
    match_asd = (n_asd == paper_target["asd"])
    match_control = (n_control == paper_target["control"])
    match_sites = (n_sites == paper_target["sites"])
    if match_total and match_asd and match_control and match_sites:
        match_status = "YES"
    elif not match_total and not match_asd and not match_control:
        match_status = "NO"
    else:
        match_status = "PARTIAL"
    log(f"Paper comparison: target={paper_target} | ours: total={total_rows}, ASD={n_asd}, "
        f"Control={n_control}, sites={n_sites} | match_status={match_status}")

    # ── dataset_inventory.json ───────────────────────────────────────────
    inventory = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_location": str(ROOT / "abide_pcp"),
        "primary_phenotypic_source": str(PHENO_QC_PATH),
        "primary_phenotypic_source_rows": len(ph_qc) if ph_qc is not None else None,
        "secondary_phenotypic_source": str(PHENO_ROOT_PATH),
        "secondary_phenotypic_source_rows": len(ph_root) if ph_root is not None else None,
        "phenotypic_qc_columns": list(ph_qc.columns) if ph_qc is not None else [],
        "fd_qc_columns_found": found_fd_cols,
        "cc200_roi_derivative_files_on_disk": n_cc200_files,
        "ts_cc200_npy_present": TS_PATH.exists(),
        "ts_cc200_npy_n_subjects": len(ts) if TS_PATH.exists() else None,
        "covs_cc200_npy_present": COVS_PATH.exists(),
        "covs_cc200_npy_shape": list(covs_shape) if covs_shape is not None else None,
        "raw_nifti_ucla1_subject_count": n_ucla_raw,
        "raw_nifti_note": "UCLA_1 is a SINGLE SITE raw-NIfTI cohort only -- not a full "
                          "multi-site ABIDE-I raw dataset. Multi-site data on disk exists "
                          "only as CPAC-preprocessed CC200 ROI derivatives.",
        "morphometry_csv_subject_count": n_morph,
        "preprocessing_pipeline_available": "CPAC (Configurable Pipeline for the Analysis of "
                                            "Connectomes), band_pass_filtering=True, "
                                            "global_signal_regression=True, quality_checked=True "
                                            "(per fetch_abide.py / abide_pcp/fetch.log)",
        "atlas_available": "CC200 (Craddock 200-region parcellation) -- ROI time series only. "
                           "AAL atlas: NOT FOUND in project files -- would need separate extraction "
                           "(explicitly out of scope for Phase 1 per instructions).",
        "total_subjects_in_manifest": total_rows,
        "unique_subjects": unique_subjects,
        "duplicate_rows": n_dup_rows,
        "duplicate_subject_ids": dup_ids,
        "missing_subject_id_rows": n_missing_id,
        "asd_count": n_asd,
        "control_count": n_control,
        "missing_diagnosis": n_missing_dx,
        "invalid_diagnosis": n_invalid_dx,
        "age_range": [age_min, age_max],
        "missing_age": n_missing_age,
        "sex_distribution": {str(k): int(v) for k, v in sex_counts.items()},
        "missing_sex": n_missing_sex,
        "n_sites": n_sites,
        "site_counts": {str(k): int(v) for k, v in site_counts.items()},
        "subjects_with_fmri_derivative_cc200": int(manifest["fmri_rois_cc200_available"].sum()),
        "subjects_with_preprocessed_data": int(manifest["preprocessed_available"].sum()),
        "subjects_with_fd_qc_info": int(manifest["fd_qc_available"].sum()),
        "paper_target": paper_target,
        "match_with_paper_target": match_status,
        "match_details": {
            "total_matches": match_total, "asd_matches": match_asd,
            "control_matches": match_control, "sites_matches": match_sites,
        },
    }
    inventory_path = OUT_DIR / "dataset_inventory.json"
    with open(inventory_path, "w") as f:
        json.dump(inventory, f, indent=2)
    log(f"dataset_inventory.json written -> {inventory_path}")

    # ── phase1_summary.txt ───────────────────────────────────────────────
    summary_lines = [
        "PHASE 1 SUMMARY -- ABIDE-I Dataset Inventory",
        "=" * 60,
        "",
        "DATASET",
        f"  Source: ABIDE-I (CPAC pipeline, CC200 atlas, quality_checked=True)",
        f"  Total subjects found (phenotypic_qc.csv): {total_rows}",
        f"  ASD count: {n_asd}",
        f"  Control count: {n_control}",
        f"  Unique subject count: {unique_subjects}",
        f"  Duplicate rows: {n_dup_rows}",
        "",
        "DEMOGRAPHICS",
        f"  Age range: {age_min} - {age_max}",
        f"  Missing age: {n_missing_age}",
        f"  Sex distribution: {sex_counts}",
        f"  Missing sex: {n_missing_sex}",
        "",
        "SITE",
        f"  Number of sites: {n_sites}",
        f"  Subjects per site: {site_counts}",
        "",
        "DATA AVAILABILITY",
        f"  Subjects with fMRI (CC200 ROI derivative): {int(manifest['fmri_rois_cc200_available'].sum())}",
        f"  Subjects with preprocessed fMRI: {int(manifest['preprocessed_available'].sum())}",
        f"  Subjects with ROI data: {int(manifest['fmri_rois_cc200_available'].sum())}",
        f"  Subjects with FD/QC information: {int(manifest['fd_qc_available'].sum())}",
        "",
        "PAPER COMPARISON",
        f"  Paper target: 884 total / 408 ASD / 476 Control / 17 sites / age 7-64",
        f"  Our data:     {total_rows} total / {n_asd} ASD / {n_control} Control / {n_sites} sites / age {age_min}-{age_max}",
        f"  Match status: {match_status}",
    ]
    summary_path = OUT_DIR / "phase1_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    log(f"phase1_summary.txt written -> {summary_path}")

    # ── audit log ─────────────────────────────────────────────────────────
    audit_path = OUT_DIR / "phase1_audit.log"
    audit_lines.append(f"Total elapsed: {time.time()-t0:.1f}s")
    audit_lines.append(f"Output files: {manifest_path}, {available_path}, {inventory_path}, {summary_path}")
    with open(audit_path, "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 1 outputs written to {OUT_DIR}")

    return inventory, match_status


if __name__ == "__main__":
    main()
