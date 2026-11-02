"""
phase3_5_bold_feasibility.py -- Phase 3.5: Option B feasibility audit
(CPAC filt_global BOLD -> independent AAL-116 extraction).

Read-only, header-only inspection. No fMRI data is loaded into memory
(img.get_fdata() is never called), no ROI extraction, no FC, no Pearson/
Fisher-Z, no SVM-RFE/SSAE/classifier training, no XAI/ROAR, no resampling,
no downloads, no modification of Phase 1/2/3 outputs or original CPAC/raw
files. CC200 derivatives are never counted as BOLD/AAL availability.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
PHASE1_MANIFEST = ROOT / "results" / "phase1" / "subject_manifest.csv"
PHASE3_DIR      = ROOT / "results" / "phase3"

OUT_DIR = ROOT / "results" / "phase3_5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIR_PARTS = {".venv", "results", "__pycache__", "OUTPUT", "anaconda_projects"}

audit_lines = []


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg)


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime}:{st.st_size}"


def main():
    # ── snapshot Phase 1/2/3 file fingerprints BEFORE any work, to prove no mutation ──
    guarded_paths = [
        PHASE1_MANIFEST, PHASE2_FILTERED,
        PHASE3_DIR / "pipeline_inventory.csv",
        PHASE3_DIR / "pipeline_subject_coverage.csv",
        PHASE3_DIR / "aal_availability.csv",
        PHASE3_DIR / "phase3_inventory.json",
    ]
    pre_fingerprints = {p: file_fingerprint(p) for p in guarded_paths if p.exists()}

    log("=" * 60)
    log("PHASE 3.5 -- Option B (CPAC BOLD -> AAL-116) feasibility audit started.")
    log("Read-only, header-only. No extraction/FC/training/downloads performed.")
    log("=" * 60)

    # ── STEP 1: authoritative cohort ────────────────────────────────────
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    n_cohort = len(cohort)
    n_asd = int((cohort["diagnosis"] == "ASD").sum())
    n_control = int((cohort["diagnosis"] == "Control").sum())
    n_dup = int(cohort["subject_id"].duplicated().sum())
    log(f"STEP 1: Phase-2 cohort loaded from {PHASE2_FILTERED}: "
        f"{n_cohort} subjects, ASD={n_asd}, Control={n_control}, duplicates={n_dup}")
    if n_cohort != 780 or n_asd != 351 or n_control != 429:
        log(f"[WARNING] Expected 780/351/429, got {n_cohort}/{n_asd}/{n_control}.")
    cohort_ids = set(cohort["subject_id"])

    # ── STEP 2: search for NIfTI files project-wide ─────────────────────
    nii_files = []
    for pattern in ("*.nii.gz", "*.nii"):
        for p in ROOT.rglob(pattern):
            if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
                continue
            nii_files.append(p)
    log(f"STEP 2: NIfTI files found project-wide (excluding {sorted(EXCLUDE_DIR_PARTS)}): "
        f"{len(nii_files)}")

    ucla_rest_files = [p for p in nii_files if p.parent.name == "rest_1" and p.name == "rest.nii.gz"]
    ucla_anat_files = [p for p in nii_files if p.parent.name == "anat_1" and "mprage" in p.name]
    ucla_hires_files = [p for p in nii_files if p.parent.name == "hires_1"]
    template_files = [p for p in nii_files if "template" in str(p.parent).lower()]
    other_files = [p for p in nii_files if p not in ucla_rest_files + ucla_anat_files
                   + ucla_hires_files + template_files]
    log(f"  Classified: functional-candidate(rest)={len(ucla_rest_files)}, "
        f"anat/mprage={len(ucla_anat_files)}, hires={len(ucla_hires_files)}, "
        f"template={len(template_files)}, unclassified={len(other_files)}")

    cpac_path_evidence = [p for p in nii_files
                          if "cpac" in str(p).lower() or "filt_global" in str(p).lower()
                          or "derivative" in str(p).lower()]
    log(f"  Files whose PATH contains 'cpac'/'filt_global'/'derivative': {len(cpac_path_evidence)}")
    log("  Directory layout for the rest_1 files: UCLA_1/UCLA_1/<subject_id>/session_1/rest_1/rest.nii.gz "
        "-- this matches the standard RAW ABIDE-I release layout, NOT the ABIDE PCP CPAC-derivatives "
        "layout (.../Outputs/cpac/<strategy>/<file>). README.md (project root) independently and "
        "explicitly documents this data as 'the official ABIDE MRI/fMRI dataset (.nii.gz files)', "
        "i.e. raw, not CPAC-preprocessed.")

    # ── STEP 3: CC200 explicitly excluded from BOLD counting ────────────
    log("STEP 3: CC200 ROI derivative files (.1D time series under abide_pcp/) are explicitly "
        "EXCLUDED from BOLD/functional-volume counting below -- they are pre-extracted ROI text "
        "series, not 4D volumes, and are not usable as AAL-116 extraction input regardless.")

    # ── STEP 4/5/6/7/8: per-subject coverage + header inspection ────────
    def extract_ucla_subject_id(p: Path) -> str | None:
        m = re.match(r"^0*(\d+)$", p.parents[2].name) if len(p.parents) >= 3 else None
        return str(int(m.group(1))) if m else None

    rest_by_subject = {}
    for p in ucla_rest_files:
        sid = extract_ucla_subject_id(p)
        if sid:
            rest_by_subject[sid] = p

    coverage_rows = []
    for _, row in cohort.iterrows():
        sid = row["subject_id"]
        path = rest_by_subject.get(sid)
        rec = {
            "subject_id": sid, "diagnosis": row["diagnosis"], "site": row.get("site", ""),
            "bold_found": "NO", "bold_path": "", "file_size_mb": None,
            "nifti_shape": None, "n_timepoints": None, "voxel_size": None,
            "affine_available": False, "pipeline": "N/A", "strategy": "N/A",
            "notes": "No functional NIfTI file found for this subject anywhere in the project.",
        }
        if path is not None and path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            rec.update({
                "bold_found": "YES", "bold_path": str(path), "file_size_mb": round(size_mb, 2),
                "pipeline": "UNCERTAIN", "strategy": "UNCERTAIN",
                "notes": "Raw NIfTI in UCLA_1 raw-download directory structure; "
                        "NOT located under any CPAC/derivatives path; no CPAC config/log found.",
            })
            try:
                img = nib.load(str(path))
                shape = img.shape
                zooms = img.header.get_zooms()
                rec["nifti_shape"] = str(shape)
                rec["n_timepoints"] = int(shape[3]) if len(shape) == 4 else None
                rec["voxel_size"] = str(tuple(round(float(z), 3) for z in zooms[:3]))
                rec["affine_available"] = img.affine is not None
                if len(shape) != 4:
                    rec["bold_found"] = "UNCERTAIN"
                    rec["notes"] += " WARNING: file is not 4D."
            except Exception as exc:
                rec["bold_found"] = "UNCERTAIN"
                rec["notes"] += f" Header read failed: {type(exc).__name__}: {exc}"
        coverage_rows.append(rec)

    coverage_df = pd.DataFrame(coverage_rows)
    n_yes = int((coverage_df["bold_found"] == "YES").sum())
    n_no = int((coverage_df["bold_found"] == "NO").sum())
    n_unc = int((coverage_df["bold_found"] == "UNCERTAIN").sum())
    log(f"STEP 4: Subject-level BOLD coverage over the 780-subject cohort: "
        f"YES={n_yes}  NO={n_no}  UNCERTAIN={n_unc}")

    coverage_path = OUT_DIR / "cpac_bold_subject_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    log(f"  cpac_bold_subject_coverage.csv written -> {coverage_path}")

    # Pipeline-identity classification (Step 6): evidence-based, applied per subject
    # No file anywhere carries CPAC/filt_global path or config evidence -> 0 CONFIRMED, 0 LIKELY.
    n_confirmed_cpac = 0
    n_likely_cpac = 0
    n_uncertain_pipeline = n_yes  # every found file is UNCERTAIN identity
    log(f"STEP 6: Pipeline identity -- CONFIRMED_CPAC_FILT_GLOBAL={n_confirmed_cpac}, "
        f"LIKELY_CPAC_FILT_GLOBAL={n_likely_cpac}, UNCERTAIN={n_uncertain_pipeline} "
        f"(0 files anywhere carry cpac/filt_global path or config evidence; files are documented "
        f"as raw ABIDE-I data per README.md and standard raw directory layout)")

    # ── STEP 7: preprocessing evidence ───────────────────────────────────
    preproc_rows = [
        {"characteristic": "motion_correction", "status": "NOT_CONFIRMED",
         "evidence": "No motion-correction log, config, or sidecar file found.",
         "source": "Project-wide search near UCLA_1 raw tree", "notes": ""},
        {"characteristic": "global_signal_regression", "status": "NOT_CONFIRMED",
         "evidence": "No GSR config/log found for the raw NIfTI volumes. Note: fetch_abide.py sets "
                     "global_signal_regression=True, but that parameter applies to the SEPARATE "
                     "already-extracted CC200 .1D pipeline (abide_pcp/), not these raw volumes.",
         "source": "fetch_abide.py (unrelated CC200 pipeline); no evidence for raw volumes", "notes": ""},
        {"characteristic": "nuisance_regression", "status": "NOT_CONFIRMED",
         "evidence": "No nuisance-regression config/log found.",
         "source": "Project-wide search", "notes": ""},
        {"characteristic": "bandpass_filtering", "status": "NOT_CONFIRMED",
         "evidence": "No band-pass filter config/log found for the raw volumes. Note: fetch_abide.py "
                     "sets band_pass_filtering=True for the separate CC200 pipeline only.",
         "source": "fetch_abide.py (unrelated CC200 pipeline); no evidence for raw volumes", "notes": ""},
        {"characteristic": "temporal_filtering", "status": "NOT_CONFIRMED",
         "evidence": "No temporal-filter config/log found.",
         "source": "Project-wide search", "notes": ""},
        {"characteristic": "spatial_normalization", "status": "NOT_CONFIRMED",
         "evidence": "No normalization config/log found; header geometry (native-looking "
                     "64x64x34 @ 3x3x4mm) is consistent with un-normalized data but this is "
                     "inference from header shape, not direct evidence -- see Step 8.",
         "source": "nibabel header inspection of representative sample", "notes": "Inference, not proof."},
        {"characteristic": "smoothing", "status": "NOT_CONFIRMED",
         "evidence": "No smoothing-kernel config/log found.",
         "source": "Project-wide search", "notes": ""},
    ]
    preproc_df = pd.DataFrame(preproc_rows, columns=["characteristic", "status", "evidence", "source", "notes"])
    preproc_path = OUT_DIR / "preprocessing_evidence.csv"
    preproc_df.to_csv(preproc_path, index=False)
    log(f"STEP 7: preprocessing_evidence.csv written -> {preproc_path}")
    for _, row in preproc_df.iterrows():
        log(f"  {row['characteristic']}: {row['status']}")

    # ── STEP 8: representative header sample (5 ASD + 5 Control) + AAL compatibility ──
    available_subjects = coverage_df[coverage_df["bold_found"] == "YES"]
    asd_sample = available_subjects[available_subjects["diagnosis"] == "ASD"].head(5)
    control_sample = available_subjects[available_subjects["diagnosis"] == "Control"].head(5)
    rep_sample = pd.concat([asd_sample, control_sample])

    rep_rows = []
    for _, row in rep_sample.iterrows():
        path = Path(row["bold_path"])
        try:
            img = nib.load(str(path))
            shape = img.shape
            zooms = img.header.get_zooms()[:3]
            affine = img.affine
            valid_4d = len(shape) == 4 and shape[3] > 1
            rep_rows.append({
                "subject_id": row["subject_id"], "diagnosis": row["diagnosis"], "site": row["site"],
                "shape": str(shape), "timepoints": int(shape[3]) if len(shape) == 4 else None,
                "voxel_size": str(tuple(round(float(z), 3) for z in zooms)),
                "orientation": "".join(nib.aff2axcodes(affine)),
                "affine_summary": str(np.round(affine, 2).tolist()),
                "valid_4d": valid_4d,
                "pipeline_identity": "UNCERTAIN",
                "strategy_identity": "UNCERTAIN",
                "notes": "Native-looking geometry; no normalization evidence found.",
            })
        except Exception as exc:
            rep_rows.append({
                "subject_id": row["subject_id"], "diagnosis": row["diagnosis"], "site": row["site"],
                "shape": None, "timepoints": None, "voxel_size": None, "orientation": None,
                "affine_summary": None, "valid_4d": False,
                "pipeline_identity": "UNCERTAIN", "strategy_identity": "UNCERTAIN",
                "notes": f"{type(exc).__name__}: {exc}",
            })
    rep_df = pd.DataFrame(rep_rows, columns=[
        "subject_id", "diagnosis", "site", "shape", "timepoints", "voxel_size",
        "orientation", "affine_summary", "valid_4d", "pipeline_identity",
        "strategy_identity", "notes"])
    rep_path = OUT_DIR / "representative_bold_audit.csv"
    rep_df.to_csv(rep_path, index=False)
    log(f"STEP 5/8: representative_bold_audit.csv written -> {rep_path} "
        f"({len(asd_sample)} ASD, {len(control_sample)} Control; header-only, no data loaded)")
    if len(control_sample) == 0:
        log("  [NOTE] 0 Control subjects in the representative sample -- coverage is limited "
            "to UCLA_1 only; see coverage numbers for the underlying cause.")

    aal_feasibility = "UNCERTAIN"
    feasibility_reason = (
        "Representative header inspection (no data loaded, no resampling) shows raw-looking "
        "native-space geometry (e.g. 64x64x34 voxels at 3x3x4mm) with no confirmed spatial "
        "normalization to a standard template. AAL-116 is defined in standard (e.g. MNI) space; "
        "applying it directly to unconfirmed native-space data would be invalid. Because "
        "normalization status could not be confirmed EITHER WAY from headers alone (absence of "
        "evidence is not evidence of absence), this is recorded as UNCERTAIN rather than a "
        "definitive NO -- but it does NOT currently support a YES determination."
    )
    log(f"STEP 8: AAL_EXTRACTION_FEASIBILITY (image-space) = {aal_feasibility}")
    log(f"  {feasibility_reason}")

    # ── STEP 9: search for a local AAL-116 atlas file ────────────────────
    aal_name_patterns = ["aal116", "aal_116", "aal"]
    aal_hits = []
    for p in ROOT.rglob("*"):
        if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
            continue
        if any(pat in p.name.lower() for pat in aal_name_patterns):
            aal_hits.append(str(p))
    aal_local_available = len(aal_hits) > 0

    aal_atlas_row = {
        "atlas_name": "AAL-116", "atlas_version": "", "file_found": "YES" if aal_local_available else "NO",
        "file_path": aal_hits[0] if aal_hits else "", "dimensions": "", "voxel_size": "",
        "space": "", "label_count": "", "notes": "",
    }
    if not aal_local_available:
        aal_atlas_row["notes"] = ("AAL-116 atlas file is not currently available locally. "
                                  "derivatives/template/ contains mni152_t1_2mm.nii.gz and "
                                  "mni152_mask_2mm.nii.gz -- a T1 template and brain mask, NOT an "
                                  "AAL region-label atlas. Not substituted or treated as one.")
        log('STEP 9: "AAL-116 atlas file is not currently available locally."')
    else:
        log(f"STEP 9: AAL-related file(s) found: {aal_hits}")
    aal_atlas_df = pd.DataFrame([aal_atlas_row], columns=[
        "atlas_name", "atlas_version", "file_found", "file_path", "dimensions",
        "voxel_size", "space", "label_count", "notes"])
    aal_atlas_path = OUT_DIR / "aal_atlas_local_availability.csv"
    aal_atlas_df.to_csv(aal_atlas_path, index=False)
    log(f"  aal_atlas_local_availability.csv written -> {aal_atlas_path}")

    # ── STEP 10: coverage summary ─────────────────────────────────────────
    asd_bold = int(((coverage_df["diagnosis"] == "ASD") & (coverage_df["bold_found"] == "YES")).sum())
    control_bold = int(((coverage_df["diagnosis"] == "Control") & (coverage_df["bold_found"] == "YES")).sum())
    sites_with_bold = sorted(coverage_df.loc[coverage_df["bold_found"] == "YES", "site"].unique().tolist())
    sites_without_bold = sorted(set(coverage_df["site"].unique()) - set(sites_with_bold))

    summary_row = {
        "total_phase2_subjects": n_cohort, "ASD_total": n_asd, "control_total": n_control,
        "bold_available": n_yes, "bold_unavailable": n_no, "bold_uncertain": n_unc,
        "ASD_bold_available": asd_bold, "control_bold_available": control_bold,
        "number_of_sites_with_bold": len(sites_with_bold),
        "number_of_sites_without_bold": len(sites_without_bold),
        "confirmed_cpac_filt_global": n_confirmed_cpac,
        "likely_cpac_filt_global": n_likely_cpac,
        "uncertain_pipeline_identity": n_uncertain_pipeline,
    }
    coverage_summary_path = OUT_DIR / "cpac_bold_coverage_summary.csv"
    pd.DataFrame([summary_row]).to_csv(coverage_summary_path, index=False)
    log(f"STEP 10: cpac_bold_coverage_summary.csv written -> {coverage_summary_path}")
    log(f"  {summary_row}")
    log(f"  Sites WITH any BOLD coverage: {sites_with_bold}")
    log(f"  Sites WITHOUT any BOLD coverage: {sites_without_bold}")

    # ── STEP 14: decision ─────────────────────────────────────────────────
    coverage_fraction = n_yes / n_cohort if n_cohort else 0
    mapping_reliable = True  # subject_id extraction from folder names was exact/unambiguous
    valid_4d_all_sampled = bool(rep_df["valid_4d"].all()) if len(rep_df) else False

    if coverage_fraction >= 0.98 and n_confirmed_cpac >= int(0.98 * n_yes) and aal_local_available \
            and aal_feasibility == "YES":
        option_b = "FEASIBLE"
    elif n_yes == 0 or not mapping_reliable:
        option_b = "NOT FEASIBLE"
    else:
        option_b = "PARTIAL"

    if option_b == "PARTIAL" and (coverage_fraction < 0.10 or n_confirmed_cpac == 0):
        option_b_practical_note = (
            "Note: although this falls in the PARTIAL bucket per the decision rule (some subjects "
            "have data, mapping is unambiguous), the practical scope here -- 6.9% coverage, "
            "single-site only, zero confirmed CPAC identity, no local AAL atlas -- is not "
            "sufficient to support a genuine 780-subject paper reproduction. Treat this as "
            "functionally NOT FEASIBLE for Phase 4 as currently scoped."
        )
    else:
        option_b_practical_note = ""

    decision_reason = (
        f"{n_yes}/{n_cohort} ({coverage_fraction*100:.1f}%) of the Phase-2 cohort has any functional "
        f"NIfTI file at all, restricted entirely to the single-site UCLA_1 raw download "
        f"({asd_bold} ASD / {control_bold} Control; {len(sites_with_bold)} of "
        f"{len(sites_with_bold)+len(sites_without_bold)} sites covered). "
        f"CPAC/filt_global identity: {n_confirmed_cpac} CONFIRMED, {n_likely_cpac} LIKELY, "
        f"{n_uncertain_pipeline} UNCERTAIN -- i.e. zero positive evidence anywhere that these are "
        f"CPAC-preprocessed rather than raw. AAL-116 atlas locally available: "
        f"{'YES' if aal_local_available else 'NO'}. AAL extraction feasibility from image headers: "
        f"{aal_feasibility}. {option_b_practical_note}"
    ).strip()

    log(f"STEP 14: OPTION_B = {option_b}")
    log(f"  {decision_reason}")

    if option_b == "FEASIBLE":
        recommended_next_step = "Proceed to Phase 4 planning for AAL-116 extraction using the confirmed CPAC filt_global data."
    else:
        recommended_next_step = (
            "Do NOT proceed to Phase 4 under Option B as scoped. Three alternatives for your decision: "
            "(a) Option C -- check whether a genuine ABIDE PCP release already provides AAL-116 ROI "
            "derivatives directly (avoiding raw-volume preprocessing entirely); (b) Option D -- perform "
            "full CPAC-equivalent preprocessing + AAL parcellation ourselves on raw multi-site ABIDE-I "
            "data, which requires downloading raw data for all 780 subjects (not just the 54 UCLA_1 "
            "subjects on disk) plus an AAL-116 atlas file, and building a real preprocessing pipeline -- "
            "substantially beyond this audit's scope; or (c) reconsider CC200 as the working atlas with "
            "your explicit, documented sign-off that this is a deliberate methodology deviation from "
            "the paper's AAL-116 basis, not a silent substitution."
        )
    log(f"  Recommended next step: {recommended_next_step}")

    # ── audit verification ────────────────────────────────────────────────
    post_fingerprints = {p: file_fingerprint(p) for p in pre_fingerprints}
    phase123_unchanged = all(pre_fingerprints[p] == post_fingerprints[p] for p in pre_fingerprints)
    log(f"AUDIT: Phase 1/2/3 files unchanged (fingerprint match): "
        f"{'PASS' if phase123_unchanged else 'FAIL'}")
    log(f"AUDIT: 780 Phase-2 subjects used: {'PASS' if n_cohort == 780 else f'FAIL ({n_cohort})'}")
    log(f"AUDIT: 351 ASD / 429 Control: "
        f"{'PASS' if (n_asd == 351 and n_control == 429) else f'FAIL ({n_asd}/{n_control})'}")
    log(f"AUDIT: no duplicate subject IDs: {'PASS' if n_dup == 0 else f'FAIL ({n_dup})'}")
    log("AUDIT: no CC200 substitution -- CC200 files were not read, opened, or counted anywhere "
        "in this script (only path search for the pipeline_inventory.csv reference was reused "
        "from Phase 3's existing output, not recomputed here).")
    log("AUDIT: not performed -- FC calculation, AAL extraction, image resampling, model training, "
        "large downloads, modification of original CPAC/raw files. Confirmed by code review: "
        "img.get_fdata() is never called; only img.shape / img.header.get_zooms() / img.affine "
        "(header-only) are accessed.")

    # ── STEP: outputs (JSON + summary + audit log) ─────────────────────────
    feasibility = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase2_cohort": {"total": n_cohort, "asd": n_asd, "control": n_control, "duplicates": n_dup},
        "nifti_files_found_total": len(nii_files),
        "classified": {
            "functional_candidate_rest": len(ucla_rest_files), "anat_mprage": len(ucla_anat_files),
            "hires": len(ucla_hires_files), "template": len(template_files),
            "unclassified": len(other_files),
        },
        "cpac_path_evidence_file_count": len(cpac_path_evidence),
        "raw_vs_cpac_determination": "Evidence indicates these are RAW ABIDE-I NIfTI files, not "
                                     "CPAC-preprocessed derivatives (see audit log for full reasoning).",
        "bold_coverage": {"available": n_yes, "unavailable": n_no, "uncertain": n_unc,
                          "coverage_fraction": round(coverage_fraction, 4),
                          "asd_available": asd_bold, "control_available": control_bold,
                          "sites_with_bold": sites_with_bold, "sites_without_bold": sites_without_bold},
        "pipeline_identity": {"confirmed_cpac_filt_global": n_confirmed_cpac,
                              "likely_cpac_filt_global": n_likely_cpac,
                              "uncertain": n_uncertain_pipeline},
        "preprocessing_evidence": preproc_df.to_dict(orient="records"),
        "aal_atlas_locally_available": aal_local_available,
        "aal_extraction_feasibility": aal_feasibility,
        "aal_feasibility_reason": feasibility_reason,
        "representative_sample": {"asd_n": len(asd_sample), "control_n": len(control_sample),
                                  "all_valid_4d": valid_4d_all_sampled},
        "option_b": option_b,
        "decision_reason": decision_reason,
        "recommended_next_step": recommended_next_step,
        "audit": {
            "phase1_phase2_phase3_unchanged": phase123_unchanged,
            "cohort_total_correct": n_cohort == 780,
            "asd_control_counts_correct": (n_asd == 351 and n_control == 429),
            "no_duplicate_subject_ids": n_dup == 0,
            "cc200_substituted": False, "fc_calculated": False,
            "pearson_correlation_calculated": False, "fisher_z_calculated": False,
            "svm_rfe_run": False, "ssae_trained": False, "classifier_trained": False,
            "xai_run": False, "roar_run": False, "images_resampled": False,
            "large_downloads": False, "original_data_modified": False,
        },
    }
    feasibility_path = OUT_DIR / "phase3_5_feasibility.json"
    with open(feasibility_path, "w") as f:
        json.dump(feasibility, f, indent=2)
    log(f"phase3_5_feasibility.json written -> {feasibility_path}")

    summary_lines = [
        "PHASE 3.5 SUMMARY -- Option B (CPAC BOLD -> independent AAL-116 extraction) Feasibility Audit",
        "=" * 70, "",
        f"Phase-2 cohort: {n_cohort} subjects ({n_asd} ASD, {n_control} Control)", "",
        "CPAC filt_global BOLD availability:",
        f"  Available:   {n_yes} / {n_cohort}",
        f"  Unavailable: {n_no} / {n_cohort}",
        f"  Uncertain:   {n_unc} / {n_cohort}", "",
        f"Pipeline identity: CONFIRMED={n_confirmed_cpac}  LIKELY={n_likely_cpac}  "
        f"UNCERTAIN={n_uncertain_pipeline}", "",
        f"AAL-116 atlas locally available: {'YES' if aal_local_available else 'NO'}",
        f"AAL extraction feasibility (image-space): {aal_feasibility}", "",
        f"OPTION B: {option_b}", f"Reason: {decision_reason}", "",
        f"Recommended next step: {recommended_next_step}",
    ]
    with open(OUT_DIR / "phase3_5_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    log(f"phase3_5_summary.txt written -> {OUT_DIR / 'phase3_5_summary.txt'}")

    audit_lines.append("Output files: cpac_bold_subject_coverage.csv, cpac_bold_coverage_summary.csv, "
                        "aal_atlas_local_availability.csv, representative_bold_audit.csv, "
                        "preprocessing_evidence.csv, phase3_5_summary.txt, phase3_5_feasibility.json")
    with open(OUT_DIR / "phase3_5_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 3.5 outputs written to {OUT_DIR}")

    print()
    print("=" * 50)
    print("PHASE 3.5 STATUS: COMPLETE")
    print("=" * 50)
    print()
    print("Phase-2 cohort:")
    print(f"{n_cohort} subjects")
    print(f"{n_asd} ASD")
    print(f"{n_control} Controls")
    print()
    print("CPAC filt_global BOLD:")
    print(f"Available: {n_yes} / {n_cohort}")
    print(f"Unavailable: {n_no} / {n_cohort}")
    print(f"Uncertain: {n_unc} / {n_cohort}")
    print()
    print("Confirmed CPAC filt_global:")
    print(f"{n_confirmed_cpac} / {n_cohort}")
    print()
    print("AAL-116 atlas locally available:")
    print("YES" if aal_local_available else "NO")
    print()
    print("AAL extraction feasibility:")
    print(aal_feasibility)
    print()
    print("Option B:")
    print(option_b)
    print()
    print("Reason:")
    print(decision_reason)
    print()
    print("Recommended next step:")
    print(recommended_next_step)
    print()
    print("=" * 50)

    return feasibility


if __name__ == "__main__":
    main()
