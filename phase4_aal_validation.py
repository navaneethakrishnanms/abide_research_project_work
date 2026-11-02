"""
phase4_aal_validation.py -- Phase 4: AAL-116 ROI time-series validation and
preparation (NOT computation). Read-only against the authoritative cohort
(results/phase2/fd_filtered_subjects.csv) and the authoritative downloaded
data (the *_rois_aal.1D files actually present on disk -- NOT the stale
aal_download_summary.json, which is historical only and not consulted here
for coverage decisions).

Does NOT compute functional connectivity, Pearson correlation, Fisher-Z,
feature vectors, SVM-RFE, SSAE, Softmax, XAI, or ROAR. Does NOT download
anything. Does NOT modify Phase 1/2/3/3.5/3.6 files or the source .1D
files. Produces a validation/quality report and a manifest only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
AAL_DIR = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
CURRENT_STATUS_JSON = ROOT / "results" / "phase3_6_aal_download" / "aal_current_verified_status.json"

OUT_DIR = ROOT / "results" / "phase4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [
    ROOT / "results" / "phase1", ROOT / "results" / "phase2",
    ROOT / "results" / "phase3", ROOT / "results" / "phase3_5",
    ROOT / "results" / "phase3_6", ROOT / "results" / "phase3_6_verification",
    ROOT / "results" / "phase3_6_aal_download",
]

audit_lines = []


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg, flush=True)


def fp_snapshot(root: Path) -> dict[str, tuple[int, float]]:
    out = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[str(p)] = (st.st_size, st.st_mtime)
    return out


def guard_fingerprints() -> dict[str, tuple[int, float]]:
    fps = {}
    for d in GUARD_DIRS:
        fps.update(fp_snapshot(d))
    return fps


def normalize_id(raw) -> str:
    s = str(raw).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def extract_subject_id_from_aal_filename(name: str) -> str | None:
    base = name.replace("_rois_aal.1D", "")
    digits = "".join(ch for ch in base.split("_")[-1] if ch.isdigit())
    return str(int(digits)) if digits else None


def main():
    log("=" * 60)
    log("PHASE 4 -- AAL-116 ROI time-series validation & preparation (no FC/ML).")
    log("=" * 60)

    pre_guard = guard_fingerprints()

    # ── STEP 12: pre-audit of the authoritative cohort ──────────────────
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    cohort["subject_id_norm"] = cohort["subject_id"].map(normalize_id)
    n_cohort = len(cohort)
    n_dup_cohort = int(cohort["subject_id"].duplicated().sum())
    n_asd = int((cohort["diagnosis"] == "ASD").sum())
    n_control = int((cohort["diagnosis"] == "Control").sum())
    cohort_fp = fp_snapshot(PHASE2_FILTERED.parent)
    log(f"Pre-audit: Phase-2 cohort total={n_cohort}, dup={n_dup_cohort}, ASD={n_asd}, Control={n_control}")
    if not (n_cohort == 780 and n_dup_cohort == 0 and n_asd == 351 and n_control == 429):
        log("[STOP] Phase-2 cohort pre-check failed. Aborting.")
        raise SystemExit(1)
    cohort_ids = set(cohort["subject_id_norm"])
    cohort_lookup = cohort.set_index("subject_id_norm")

    if CURRENT_STATUS_JSON.exists():
        with open(CURRENT_STATUS_JSON) as f:
            current_status = json.load(f)
        log(f"Authoritative current-status reference loaded (informational only, not relied on for "
            f"coverage decisions -- recomputed independently below): overall_pass="
            f"{current_status.get('overall_pass')}, coverage={current_status.get('coverage')}")
    else:
        log("[NOTE] aal_current_verified_status.json not found -- proceeding on direct file inspection alone.")

    # ── STEP 2: locate AAL files ──────────────────────────────────────────
    log(f"AAL data directory: {AAL_DIR}")
    all_aal_files = list(AAL_DIR.rglob("*_rois_aal.1D"))
    id_to_file = {}
    for f in all_aal_files:
        sid = extract_subject_id_from_aal_filename(f.name)
        if sid:
            id_to_file[sid] = f
    log(f"Number of .1D files: {len(all_aal_files)}")
    log(f"Number of unique subject IDs: {len(id_to_file)}")

    # ── STEP 3: select only the authoritative 780 ─────────────────────────
    downloaded_ids = set(id_to_file.keys())
    overlap = cohort_ids & downloaded_ids
    missing = cohort_ids - downloaded_ids
    extra = downloaded_ids - cohort_ids
    log(f"Phase-2 subjects={n_cohort}  AAL available={len(overlap)}  missing={len(missing)}  "
        f"extra_downloaded={len(extra)} (expected 91, NOT used for Phase-4 cohort, left untouched)")
    if missing:
        log(f"[WARNING] Missing subjects (no AAL file): {sorted(missing)}")

    # ── STEP 4/5: inspect structure + per-subject validation ─────────────
    validation_rows = []
    roi_col_stats = None  # will be initialized once ROI count is known

    for sid in sorted(cohort_ids, key=lambda x: int(x) if x.isdigit() else 0):
        row = cohort_lookup.loc[sid]
        diagnosis = row["diagnosis"]
        site = row.get("site", None)
        rec = {
            "subject_id": sid, "diagnosis": diagnosis, "site": site,
            "aal_file_path": None, "file_size_bytes": None,
            "timepoints": None, "roi_count": None,
            "missing_value_count": None, "nan_count": None, "inf_count": None,
            "constant_roi_count": None, "non_numeric": False, "readable": False,
            "valid": False, "notes": "",
        }
        if sid not in id_to_file:
            rec["notes"] = "No AAL .1D file found for this Phase-2 subject."
            validation_rows.append(rec)
            continue

        path = id_to_file[sid]
        rec["aal_file_path"] = str(path)
        try:
            size = path.stat().st_size
            rec["file_size_bytes"] = size
            if size == 0:
                rec["notes"] = "zero-byte file"
                validation_rows.append(rec)
                continue
            arr = np.loadtxt(str(path))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            rec["readable"] = True
            n_tp, n_roi = arr.shape
            rec["timepoints"] = int(n_tp)
            rec["roi_count"] = int(n_roi)

            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            rec["nan_count"] = nan_count
            rec["inf_count"] = inf_count
            rec["missing_value_count"] = nan_count  # .1D numeric format has no distinct "missing"
            # marker separate from NaN -- documented here rather than silently assumed identical.

            col_std = np.nanstd(arr, axis=0)
            constant_cols = int((col_std == 0).sum())
            rec["constant_roi_count"] = constant_cols

            rec["valid"] = (rec["readable"] and n_roi == 116 and nan_count == 0
                            and inf_count == 0 and size > 0)
            if n_roi != 116:
                rec["notes"] += f"ROI count {n_roi} != 116. "
            if nan_count > 0:
                rec["notes"] += f"{nan_count} NaN values. "
            if inf_count > 0:
                rec["notes"] += f"{inf_count} Inf values. "
            if constant_cols > 0:
                rec["notes"] += f"{constant_cols} constant-variance ROI column(s) for this subject. "

        except Exception as exc:
            rec["non_numeric"] = True
            rec["notes"] = f"{type(exc).__name__}: {exc}"
        validation_rows.append(rec)

    validation_df = pd.DataFrame(validation_rows)
    validation_path = OUT_DIR / "phase4_subject_validation.csv"
    validation_df.to_csv(validation_path, index=False)
    log(f"phase4_subject_validation.csv written -> {validation_path} ({len(validation_df)} rows)")

    observed_roi_counts = sorted(set(int(x) for x in validation_df["roi_count"].dropna()))
    n_roi_116 = int((validation_df["roi_count"] == 116).sum())
    n_roi_wrong = int(validation_df["roi_count"].notna().sum()) - n_roi_116
    n_unreadable = int((~validation_df["readable"]).sum())
    n_nan_subjects = int((validation_df["nan_count"].fillna(0) > 0).sum())
    n_inf_subjects = int((validation_df["inf_count"].fillna(0) > 0).sum())
    n_missing_val_subjects = int((validation_df["missing_value_count"].fillna(0) > 0).sum())
    n_const_subjects = int((validation_df["constant_roi_count"].fillna(0) > 0).sum())
    log(f"STEP 4/5: observed ROI counts: {observed_roi_counts}  116-ROI subjects={n_roi_116}  "
        f"wrong-ROI subjects={n_roi_wrong}  unreadable={n_unreadable}")
    log(f"  NaN-affected subjects={n_nan_subjects}  Inf-affected={n_inf_subjects}  "
        f"missing-value-affected={n_missing_val_subjects}  constant-ROI-affected={n_const_subjects}")

    # ── STEP 6: ROI-by-ROI quality audit (across the 780, for valid readable files) ──
    roi_n = 116
    valid_readable = validation_df[validation_df["readable"] & (validation_df["roi_count"] == roi_n)]
    n_valid_readable = len(valid_readable)
    roi_valid_count = np.zeros(roi_n, dtype=int)
    roi_missing_count = np.zeros(roi_n, dtype=int)
    roi_nan_count = np.zeros(roi_n, dtype=int)
    roi_inf_count = np.zeros(roi_n, dtype=int)
    roi_zero_var_count = np.zeros(roi_n, dtype=int)

    for _, r in valid_readable.iterrows():
        arr = np.loadtxt(r["aal_file_path"])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        col_std = np.nanstd(arr, axis=0)
        for c in range(roi_n):
            has_nan = nan_mask[:, c].any()
            has_inf = inf_mask[:, c].any()
            if has_nan:
                roi_nan_count[c] += 1
                roi_missing_count[c] += 1
            if has_inf:
                roi_inf_count[c] += 1
            if col_std[c] == 0:
                roi_zero_var_count[c] += 1
            if not has_nan and not has_inf:
                roi_valid_count[c] += 1

    roi_quality_df = pd.DataFrame({
        "roi_index": np.arange(1, roi_n + 1),
        "n_subjects_valid": roi_valid_count,
        "n_subjects_missing": roi_missing_count,
        "n_subjects_nan": roi_nan_count,
        "n_subjects_inf": roi_inf_count,
        "n_subjects_zero_variance": roi_zero_var_count,
        "n_subjects_inspected": n_valid_readable,
    })
    roi_quality_path = OUT_DIR / "phase4_roi_quality.csv"
    roi_quality_df.to_csv(roi_quality_path, index=False)
    log(f"phase4_roi_quality.csv written -> {roi_quality_path} ({n_valid_readable} subjects inspected per ROI)")
    problematic_rois = roi_quality_df[(roi_quality_df["n_subjects_nan"] > 0) |
                                      (roi_quality_df["n_subjects_inf"] > 0) |
                                      (roi_quality_df["n_subjects_zero_variance"] > 0)]
    log(f"ROIs with any NaN/Inf/zero-variance subject: {len(problematic_rois)} "
        f"(reported, not removed) -- indices: {problematic_rois['roi_index'].tolist()}")

    # ── STEP 7: timepoint distribution ────────────────────────────────────
    tp_series = validation_df["timepoints"].dropna().astype(int)
    if len(tp_series):
        tp_min, tp_max = int(tp_series.min()), int(tp_series.max())
        tp_mean, tp_median, tp_std = float(tp_series.mean()), float(tp_series.median()), float(tp_series.std())
        tp_p25, tp_p75 = float(tp_series.quantile(0.25)), float(tp_series.quantile(0.75))
    else:
        tp_min = tp_max = tp_mean = tp_median = tp_std = tp_p25 = tp_p75 = None
    log(f"STEP 7: timepoints min={tp_min} max={tp_max} mean={tp_mean} median={tp_median} "
        f"std={tp_std} p25={tp_p25} p75={tp_p75}")

    tp_dist = tp_series.value_counts().sort_index()
    tp_dist_df = pd.DataFrame({"timepoint_count": tp_dist.index, "n_subjects": tp_dist.values})
    tp_dist_path = OUT_DIR / "phase4_timepoint_distribution.csv"
    tp_dist_df.to_csv(tp_dist_path, index=False)
    log(f"phase4_timepoint_distribution.csv written -> {tp_dist_path} ({len(tp_dist_df)} unique counts)")

    # ── STEP 8: diagnosis + site integrity ───────────────────────────────
    n_dup_validation = int(validation_df["subject_id"].duplicated().sum())
    asd_count = int((validation_df["diagnosis"] == "ASD").sum())
    control_count = int((validation_df["diagnosis"] == "Control").sum())
    log(f"STEP 8: validation-table ASD={asd_count} Control={control_count} "
        f"(expected 351/429) dup_ids={n_dup_validation}")
    diagnosis_mismatches = 0  # diagnosis is taken directly from the Phase-2 cohort join, by
    # construction there is no independent second diagnosis source to disagree here -- documented,
    # not silently assumed: every row's diagnosis IS cohort's diagnosis (single source).

    site_summary = validation_df.groupby("site").apply(
        lambda g: pd.Series({
            "total": len(g), "ASD": int((g["diagnosis"] == "ASD").sum()),
            "Control": int((g["diagnosis"] == "Control").sum()),
            "valid": int(g["valid"].sum()), "invalid": int((~g["valid"]).sum()),
        }), include_groups=False
    ).reset_index()
    site_summary_path = OUT_DIR / "phase4_site_summary.csv"
    site_summary.to_csv(site_summary_path, index=False)
    log(f"phase4_site_summary.csv written -> {site_summary_path} ({len(site_summary)} sites)")

    # ── STEP 10: manifest (index, not a data copy) ────────────────────────
    manifest_df = validation_df[["subject_id", "diagnosis", "site", "aal_file_path",
                                  "timepoints", "roi_count", "valid"]].copy()
    manifest_df["validation_status"] = np.where(manifest_df["valid"], "VALID", "NEEDS_REVIEW")
    manifest_df = manifest_df.drop(columns=["valid"])
    manifest_path = OUT_DIR / "phase4_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    log(f"phase4_manifest.csv written -> {manifest_path} -- an INDEX pointing to the existing source "
        f".1D files (no time-series data duplicated); a separate processed representation was not "
        f"created because none is needed yet -- Phase 5's FC computation can read directly from "
        f"these paths, and duplicating ~780 small text files would add no value at this stage.")

    n_valid_subjects = int(validation_df["valid"].sum())
    n_invalid_subjects = n_cohort - n_valid_subjects

    # ── STEP 12 (post): audit ──────────────────────────────────────────────
    post_guard = guard_fingerprints()
    guard_unchanged = (pre_guard == post_guard)
    cohort_fp_after = fp_snapshot(PHASE2_FILTERED.parent)
    cohort_unchanged = (cohort_fp == cohort_fp_after)
    source_files_untouched = True  # verified by construction: script only opens .1D files for
    # reading (np.loadtxt), never writes to AAL_DIR -- confirmed by code review below.
    log(f"AUDIT: Phase 1/2/3/3.5/3.6(+verification+download) files unchanged: "
        f"{'PASS' if guard_unchanged else 'FAIL'}")
    log(f"AUDIT: Phase-2 cohort file itself unchanged: {'PASS' if cohort_unchanged else 'FAIL'}")
    log("AUDIT: no writes anywhere under abide_pcp/ (source AAL files) -- confirmed by code review: "
        "this script's only write targets are under results/phase4/.")
    log("AUDIT: no FC/Pearson/Fisher-Z/SVM-RFE/SSAE/Softmax/XAI/ROAR/ML-feature code anywhere in "
        "this script -- confirmed by code review.")
    log("AUDIT: no downloads performed -- confirmed by code review: no network calls, no nilearn "
        "fetch functions imported or called.")

    # ── STEP 13: pass criteria ─────────────────────────────────────────────
    criteria = {
        "780/780 authoritative subjects have AAL data": len(missing) == 0,
        "every subject has 116 ROI columns": n_roi_wrong == 0 and n_roi_116 == n_cohort,
        "files are readable": n_unreadable == 0,
        "no unexpected corruption (NaN/Inf)": (n_nan_subjects == 0 and n_inf_subjects == 0),
        "subject IDs map correctly": len(overlap) == n_cohort,
        "diagnosis labels match Phase 2": (asd_count == 351 and control_count == 429),
        "no duplicate subject IDs": n_dup_validation == 0,
        "previous phases remain unchanged": guard_unchanged and cohort_unchanged,
    }
    all_pass = all(criteria.values())
    phase4_status = "PASS" if all_pass else "FAIL / NEEDS REVIEW"
    log(f"STEP 13: Pass criteria: {criteria}")
    log(f"PHASE 4 STATUS: {phase4_status}")

    # ── outputs: JSON summary ──────────────────────────────────────────────
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_cohort_source": str(PHASE2_FILTERED),
        "authoritative_data_source": str(AAL_DIR),
        "phase2_cohort": {"total": n_cohort, "asd": n_asd, "control": n_control, "duplicates": n_dup_cohort},
        "aal_files_on_disk": len(all_aal_files),
        "unique_subject_ids_on_disk": len(id_to_file),
        "coverage": {"phase2_total": n_cohort, "aal_available": len(overlap),
                    "missing": len(missing), "extra_downloaded": len(extra)},
        "roi_structure": {"observed_roi_counts": observed_roi_counts,
                          "subjects_with_116_rois": n_roi_116, "subjects_with_wrong_roi_count": n_roi_wrong},
        "timepoints": {"min": tp_min, "max": tp_max, "mean": tp_mean, "median": tp_median,
                      "std": tp_std, "p25": tp_p25, "p75": tp_p75},
        "data_quality": {"nan_subjects": n_nan_subjects, "inf_subjects": n_inf_subjects,
                         "missing_value_subjects": n_missing_val_subjects,
                         "constant_roi_subjects": n_const_subjects, "unreadable_files": n_unreadable,
                         "problematic_roi_indices": problematic_rois["roi_index"].tolist()},
        "diagnosis_integrity": {"asd": asd_count, "control": control_count,
                                "diagnosis_mismatches": diagnosis_mismatches},
        "duplicates": {"duplicate_subject_ids": n_dup_validation},
        "valid_subjects": n_valid_subjects, "invalid_subjects": n_invalid_subjects,
        "pass_criteria": criteria, "phase4_status": phase4_status,
        "audit": {
            "previous_phases_unchanged": guard_unchanged, "cohort_file_unchanged": cohort_unchanged,
            "source_aal_files_untouched": source_files_untouched,
            "downloads_performed": False, "fc_computed": False, "fisher_z_computed": False,
            "ml_features_created": False, "model_trained": False,
        },
    }
    with open(OUT_DIR / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"phase4_summary.json written -> {OUT_DIR / 'phase4_summary.json'}")

    next_step = (
        "Phase 4 AAL-116 ROI time-series validation is complete.\n"
        "The verified 780-subject dataset is ready for Phase 5:\n"
        "subject-level Pearson functional connectivity calculation."
        if all_pass else
        "Phase 4 cannot proceed to Phase 5 until the reported data-quality issue is reviewed."
    )

    txt_lines = [
        "PHASE 4 -- AAL-116 ROI Time-Series Validation & Preparation", "=" * 70, "",
        f"Phase-2 cohort: {n_cohort} ({n_asd} ASD, {n_control} Control)",
        f"AAL files on disk: {len(all_aal_files)}  unique IDs: {len(id_to_file)}",
        f"Coverage: {len(overlap)}/{n_cohort}  missing={len(missing)}  extra={len(extra)}", "",
        f"ROI counts observed: {observed_roi_counts}  116-ROI subjects={n_roi_116}  wrong={n_roi_wrong}",
        f"Timepoints: min={tp_min} max={tp_max} mean={tp_mean} median={tp_median} std={tp_std}",
        f"NaN subjects={n_nan_subjects}  Inf subjects={n_inf_subjects}  "
        f"constant-ROI subjects={n_const_subjects}  unreadable={n_unreadable}", "",
        f"ASD={asd_count}  Control={control_count}  dup_ids={n_dup_validation}", "",
        f"Previous phases unchanged: {guard_unchanged}", "",
        f"PHASE 4 STATUS: {phase4_status}", "", next_step,
    ]
    with open(OUT_DIR / "phase4_summary.txt", "w") as f:
        f.write("\n".join(txt_lines))
    log("phase4_summary.txt written.")

    audit_lines.append("Output files: phase4_subject_validation.csv, phase4_roi_quality.csv, "
                        "phase4_timepoint_distribution.csv, phase4_site_summary.csv, "
                        "phase4_summary.json, phase4_summary.txt, phase4_manifest.csv")
    with open(OUT_DIR / "phase4_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 4 outputs written to {OUT_DIR}")

    print()
    print("=" * 50)
    print("PHASE 4 -- AAL-116 ROI TIME-SERIES VALIDATION")
    print("=" * 50)
    print()
    print("AUTHORITATIVE COHORT")
    print("--------------------")
    print(f"Total: {n_cohort}")
    print(f"ASD: {n_asd}")
    print(f"Controls: {n_control}")
    print()
    print("AAL DATA")
    print("--------")
    print(f"Total .1D files on disk: {len(all_aal_files)}")
    print(f"Relevant Phase-2 subjects: {n_cohort}")
    print(f"Coverage: {len(overlap)}/{n_cohort}")
    print(f"Missing: {len(missing)}")
    print(f"Extra: {len(extra)}")
    print()
    print("ROI STRUCTURE")
    print("-------------")
    print(f"Observed ROI count: {observed_roi_counts}")
    print(f"Subjects with 116 ROIs: {n_roi_116}")
    print(f"Subjects with incorrect ROI count: {n_roi_wrong}")
    print()
    print("TIMEPOINTS")
    print("----------")
    print(f"Minimum: {tp_min}")
    print(f"Maximum: {tp_max}")
    print(f"Mean: {tp_mean}")
    print(f"Median: {tp_median}")
    print(f"Standard deviation: {tp_std}")
    print()
    print("DATA QUALITY")
    print("------------")
    print(f"Subjects with NaN: {n_nan_subjects}")
    print(f"Subjects with Inf: {n_inf_subjects}")
    print(f"Subjects with missing values: {n_missing_val_subjects}")
    print(f"Subjects with constant ROI: {n_const_subjects}")
    print(f"Unreadable files: {n_unreadable}")
    print()
    print("DIAGNOSIS INTEGRITY")
    print("-------------------")
    print(f"ASD: {asd_count}")
    print(f"Controls: {control_count}")
    print(f"Diagnosis mismatches: {diagnosis_mismatches}")
    print()
    print("DUPLICATES")
    print("----------")
    print(f"Duplicate subject IDs: {n_dup_validation}")
    print()
    print("PREVIOUS PHASES")
    print("---------------")
    print(f"Phase 1 modified: {'NO' if guard_unchanged else 'YES -- CHECK AUDIT'}")
    print(f"Phase 2 modified: {'NO' if cohort_unchanged else 'YES -- CHECK AUDIT'}")
    print(f"Phase 3 modified: {'NO' if guard_unchanged else 'YES -- CHECK AUDIT'}")
    print(f"Phase 3.5 modified: {'NO' if guard_unchanged else 'YES -- CHECK AUDIT'}")
    print(f"Phase 3.6 data modified: {'NO' if guard_unchanged else 'YES -- CHECK AUDIT'}")
    print()
    print("PROCESSING PERFORMED")
    print("--------------------")
    print("Pearson correlation: NO")
    print("Functional connectivity: NO")
    print("Fisher-Z: NO")
    print("SVM-RFE: NO")
    print("SSAE: NO")
    print("Softmax: NO")
    print("XAI: NO")
    print("ROAR: NO")
    print()
    print("DOWNLOAD PERFORMED")
    print("------------------")
    print("NO")
    print()
    print("PHASE 4 STATUS")
    print("--------------")
    print(phase4_status)
    print()
    print("NEXT STEP")
    print("---------")
    print(next_step)
    print()
    print("=" * 50)

    return summary


if __name__ == "__main__":
    main()
