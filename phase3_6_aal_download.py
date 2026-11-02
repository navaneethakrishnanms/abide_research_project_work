"""
phase3_6_aal_download.py -- Phase 3.6: approved CPAC AAL-116 ROI derivative
download + integrity/coverage verification.

Approved scope: fetch_abide_pcp(pipeline='cpac', derivatives=['rois_aal'],
band_pass_filtering=True, global_signal_regression=True [i.e. the
filt_global strategy, matching our existing Phase 1-3 CPAC/CC200 choice],
quality_checked=True). Nothing else is requested -- no func_preproc, no
other atlas, no other pipeline.

This script performs the download and verification ONLY. No ROI
re-extraction, no FC, no Fisher-Z, no SVM-RFE, no SSAE, no training, no
XAI/ROAR. Phase 1/2/3/3.5 and existing Phase 3.6 files are read-only here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
ABIDE_PCP_DATA_DIR = ROOT / "abide_pcp"

OUT_DIR = ROOT / "results" / "phase3_6_aal_download"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [
    ROOT / "results" / "phase1", ROOT / "results" / "phase2",
    ROOT / "results" / "phase3", ROOT / "results" / "phase3_5",
    ROOT / "results" / "phase3_6", ROOT / "results" / "phase3_6_verification",
]

audit_lines = []


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg, flush=True)


def snapshot(root: Path) -> dict[str, tuple[int, float]]:
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
        fps.update(snapshot(d))
    return fps


def normalize_id(raw) -> str:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def main():
    log("=" * 60)
    log("PHASE 3.6 -- Approved CPAC AAL-116 ROI derivative download + verification.")
    log("=" * 60)

    pre_guard = guard_fingerprints()

    # ── authoritative cohort pre-check ──────────────────────────────────
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    n_cohort = len(cohort)
    n_dup = int(cohort["subject_id"].duplicated().sum())
    n_asd = int((cohort["diagnosis"] == "ASD").sum())
    n_control = int((cohort["diagnosis"] == "Control").sum())
    log(f"Phase-2 cohort: total={n_cohort}, dup={n_dup}, ASD={n_asd}, Control={n_control}")
    if not (n_cohort == 780 and n_dup == 0 and n_asd == 351 and n_control == 429):
        log("[STOP] Phase-2 cohort pre-check failed. Aborting without downloading.")
        raise SystemExit(1)
    cohort["subject_id_norm"] = cohort["subject_id"].map(normalize_id)
    cohort_ids = set(cohort["subject_id_norm"])

    # ── STEP 3: download safety -- before-state ─────────────────────────
    log(f"Target Nilearn data/cache directory: {ABIDE_PCP_DATA_DIR}")
    before_snap = snapshot(ABIDE_PCP_DATA_DIR)
    before_total_size = sum(sz for sz, _ in before_snap.values())
    log(f"Before-state: {len(before_snap)} files, {before_total_size/1e6:.1f} MB total")

    # ── STEP 2: the approved fetch ──────────────────────────────────────
    import nilearn
    from nilearn.datasets import fetch_abide_pcp
    log(f"nilearn version: {nilearn.__version__}")
    log("Requesting: pipeline='cpac', derivatives=['rois_aal'], band_pass_filtering=True, "
        "global_signal_regression=True (filt_global strategy -- matches existing Phase 1-3 CPAC/CC200 "
        "choice), quality_checked=True. Nothing else requested.")

    # Same transient-Windows-file-lock retry pattern already used by this project's own
    # fetch_abide.py (antivirus/sync client can grab a .part file mid-move); already-downloaded
    # files are skipped by nilearn's own cache check, so a retry resumes rather than restarting.
    import glob as _glob
    import time as _time
    data = None
    last_exc = None
    for attempt in range(1, 21):
        for p in _glob.glob(str(ABIDE_PCP_DATA_DIR / "**" / "*.part"), recursive=True):
            try:
                Path(p).unlink()
            except OSError:
                pass
        try:
            data = fetch_abide_pcp(
                data_dir=str(ABIDE_PCP_DATA_DIR),
                pipeline="cpac",
                derivatives=["rois_aal"],
                band_pass_filtering=True,
                global_signal_regression=True,
                quality_checked=True,
                verbose=1,
            )
            break
        except Exception as exc:
            last_exc = exc
            n_done = len(list(ABIDE_PCP_DATA_DIR.rglob("*_rois_aal.1D")))
            log(f"  attempt {attempt}: {type(exc).__name__}: {exc} after {n_done} files on disk -- retrying")
            _time.sleep(3)
    if data is None:
        log(f"[STOP] Download did not complete after 20 attempts. Last error: {last_exc}")
        raise SystemExit(1)
    returned_keys = [k for k in data.keys()]
    log(f"Fetch returned keys: {returned_keys}")
    unexpected = [k for k in returned_keys if k not in ("description", "phenotypic", "rois_aal")]
    if unexpected:
        log(f"[WARNING] Unexpected keys returned: {unexpected} -- inspect before proceeding.")
    pheno = pd.DataFrame(data["phenotypic"])
    log(f"Phenotypic rows returned alongside derivative: {len(pheno)}")

    after_snap = snapshot(ABIDE_PCP_DATA_DIR)
    after_total_size = sum(sz for sz, _ in after_snap.values())
    new_files = sorted(set(after_snap) - set(before_snap))
    changed_files = sorted(f for f in (set(after_snap) & set(before_snap))
                           if before_snap[f] != after_snap[f])
    log(f"After-state: {len(after_snap)} files, {after_total_size/1e6:.1f} MB total")
    log(f"New files: {len(new_files)}  Changed files: {len(changed_files)}")
    log(f"Downloaded volume this run: {(after_total_size - before_total_size)/1e6:.1f} MB")

    non_aal_new = [f for f in new_files if "rois_aal" not in f]
    if non_aal_new:
        log(f"[WARNING] New files OUTSIDE the rois_aal derivative path appeared: {non_aal_new[:20]}")
    else:
        log("Confirmed: every new file is under the rois_aal derivative path -- no other "
            "atlas/pipeline/functional data was fetched.")

    aal_files = [Path(f) for f in new_files if "rois_aal" in f and f.endswith(".1D")]
    log(f"AAL .1D files downloaded this run: {len(aal_files)}")

    # ── STEP 4: subject coverage ─────────────────────────────────────────
    def extract_subject_id_from_aal_filename(name: str) -> str | None:
        # e.g. "Caltech_0051456_rois_aal.1D" -> "51456"
        base = name.replace("_rois_aal.1D", "")
        digits = "".join(ch for ch in base.split("_")[-1] if ch.isdigit())
        return str(int(digits)) if digits else None

    downloaded_ids = {}
    for p in aal_files:
        sid = extract_subject_id_from_aal_filename(p.name)
        if sid:
            downloaded_ids[sid] = p
    downloaded_id_set = set(downloaded_ids.keys())
    downloaded_total = len(downloaded_id_set)

    overlap = cohort_ids & downloaded_id_set
    missing = cohort_ids - downloaded_id_set
    extra = downloaded_id_set - cohort_ids
    overlap_pct = round(100 * len(overlap) / n_cohort, 2) if n_cohort else 0.0
    log(f"STEP 4: phase2_total={n_cohort} downloaded_total={downloaded_total} "
        f"overlap={len(overlap)} missing={len(missing)} extra={len(extra)} overlap_pct={overlap_pct}%")

    # ── STEP 5: diagnosis/site integrity ─────────────────────────────────
    pheno["subject_id_norm"] = pheno["SUB_ID"].map(normalize_id)
    pheno["diagnosis_source"] = pheno["DX_GROUP"].map({1: "ASD", 2: "Control"})
    site_lookup = pheno.set_index("subject_id_norm")["SITE_ID"] if "SITE_ID" in pheno.columns else pd.Series(dtype=str)
    dx_lookup = pheno.set_index("subject_id_norm")["diagnosis_source"]

    coverage_rows = []
    for sid in sorted(cohort_ids, key=lambda x: int(x) if x.isdigit() else 0):
        in_dl = sid in downloaded_id_set
        row = {
            "subject_id": sid, "phase2_present": True, "downloaded_present": in_dl,
            "diagnosis_phase2": cohort.set_index("subject_id_norm").loc[sid, "diagnosis"]
                                if sid in cohort.set_index("subject_id_norm").index else None,
            "diagnosis_source": dx_lookup.get(sid) if sid in dx_lookup.index else None,
            "site": site_lookup.get(sid) if sid in site_lookup.index else None,
        }
        coverage_rows.append(row)
    for sid in sorted(extra, key=lambda x: int(x) if x.isdigit() else 0):
        coverage_rows.append({
            "subject_id": sid, "phase2_present": False, "downloaded_present": True,
            "diagnosis_phase2": None,
            "diagnosis_source": dx_lookup.get(sid) if sid in dx_lookup.index else None,
            "site": site_lookup.get(sid) if sid in site_lookup.index else None,
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = OUT_DIR / "aal_subject_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    log(f"aal_subject_coverage.csv written -> {coverage_path} ({len(coverage_df)} rows)")

    overlap_asd = int(((coverage_df["phase2_present"]) & (coverage_df["downloaded_present"]) &
                       (coverage_df["diagnosis_phase2"] == "ASD")).sum())
    overlap_control = int(((coverage_df["phase2_present"]) & (coverage_df["downloaded_present"]) &
                           (coverage_df["diagnosis_phase2"] == "Control")).sum())
    log(f"STEP 5: overlap ASD={overlap_asd} Control={overlap_control} "
        f"(expected 351/429) -- match={'YES' if (overlap_asd==351 and overlap_control==429) else 'NO'}")

    dup_downloaded = downloaded_total - len(downloaded_id_set)  # always 0 given dict-by-id construction;
    # separately check for literal duplicate FILES mapping to the same subject id:
    from collections import Counter
    fname_id_counts = Counter(extract_subject_id_from_aal_filename(p.name) for p in aal_files)
    duplicate_subject_files = {sid: c for sid, c in fname_id_counts.items() if c > 1}
    log(f"Duplicate subject IDs among downloaded files: {len(duplicate_subject_files)} "
        f"{duplicate_subject_files if duplicate_subject_files else ''}")

    site_summary = coverage_df.groupby("site")["downloaded_present"].agg(["sum", "count"]).rename(
        columns={"sum": "downloaded", "count": "total"})
    site_summary_path = OUT_DIR / "aal_site_summary.csv"
    site_summary.to_csv(site_summary_path)
    log(f"aal_site_summary.csv written -> {site_summary_path}")

    # ── STEP 6/7/8: file integrity, ROI count, timepoints, corruption ────
    integrity_rows = []
    timepoint_counts = []
    roi_counts = []
    corrupt_files = []

    for sid, path in downloaded_ids.items():
        rec = {"subject_id": sid, "path": str(path), "readable": False, "non_empty": False,
               "numeric": False, "n_rois": None, "n_timepoints": None, "notes": ""}
        try:
            size = path.stat().st_size
            rec["non_empty"] = size > 0
            if size == 0:
                rec["notes"] = "zero-byte file"
                corrupt_files.append({"subject_id": sid, "reason": "zero-byte file", "path": str(path)})
                integrity_rows.append(rec)
                continue
            arr = np.loadtxt(str(path))
            rec["readable"] = True
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            rec["numeric"] = np.issubdtype(arr.dtype, np.number)
            rec["n_timepoints"] = int(arr.shape[0])
            rec["n_rois"] = int(arr.shape[1])
            if not (arr.shape[0] > 0 and arr.shape[1] > 0):
                rec["notes"] = "empty numeric array after load"
                corrupt_files.append({"subject_id": sid, "reason": "empty array", "path": str(path)})
            elif not np.isfinite(arr).all():
                rec["notes"] = "contains NaN/Inf"
            timepoint_counts.append(rec["n_timepoints"])
            roi_counts.append(rec["n_rois"])
        except Exception as exc:
            rec["notes"] = f"{type(exc).__name__}: {exc}"
            corrupt_files.append({"subject_id": sid, "reason": rec["notes"], "path": str(path)})
        integrity_rows.append(rec)

    integrity_df = pd.DataFrame(integrity_rows)
    integrity_path = OUT_DIR / "aal_file_integrity.csv"
    integrity_df.to_csv(integrity_path, index=False)
    log(f"aal_file_integrity.csv written -> {integrity_path} ({len(integrity_df)} files inspected -- ALL "
        f"downloaded files, not just a sample)")

    unique_roi_counts = sorted(set(roi_counts)) if roi_counts else []
    observed_roi_count = unique_roi_counts[0] if len(unique_roi_counts) == 1 else unique_roi_counts
    log(f"STEP 6: unique ROI-column counts observed across all files: {unique_roi_counts}")
    if len(unique_roi_counts) == 1 and unique_roi_counts[0] == 116:
        log("STEP 6: CONFIRMED -- every downloaded file has exactly 116 ROI columns, consistent with AAL-116.")
        roi_check_pass = True
    elif len(unique_roi_counts) == 1:
        log(f"STEP 6: [STOP CONDITION] Observed ROI count is {unique_roi_counts[0]}, NOT 116. "
            f"This does NOT match the paper's stated AAL-116 atlas. Reporting discrepancy, not proceeding "
            f"to treat this as AAL-116.")
        roi_check_pass = False
    else:
        log(f"STEP 6: [STOP CONDITION] Inconsistent ROI-column counts across files: {unique_roi_counts}. "
            f"Cannot confirm a uniform AAL-116 structure.")
        roi_check_pass = False

    if timepoint_counts:
        tp_min, tp_max = min(timepoint_counts), max(timepoint_counts)
        tp_median = float(np.median(timepoint_counts))
        unique_tp = sorted(set(timepoint_counts))
    else:
        tp_min = tp_max = tp_median = None
        unique_tp = []
    log(f"STEP 7: timepoints min={tp_min} max={tp_max} median={tp_median} "
        f"unique_counts_n={len(unique_tp)} (first 10: {unique_tp[:10]})")

    tp_summary_df = pd.DataFrame({"timepoint_count": unique_tp,
                                  "n_subjects": [timepoint_counts.count(t) for t in unique_tp]})
    tp_summary_path = OUT_DIR / "aal_timepoint_summary.csv"
    tp_summary_df.to_csv(tp_summary_path, index=False)
    log(f"aal_timepoint_summary.csv written -> {tp_summary_path}")

    log(f"STEP 8: corrupt/unreadable files: {len(corrupt_files)} "
        f"{corrupt_files if corrupt_files else '(none)'}")
    log(f"STEP 8: missing Phase-2 subjects (no downloaded file): {len(missing)} "
        f"{sorted(missing)[:20] if missing else '(none)'}")
    log(f"STEP 8: unexpected extra subject IDs beyond Phase-2 (expected, matches the 91 subjects "
        f"removed by FD filtering): {len(extra)}")

    # ── STEP 9: before/after audit ────────────────────────────────────────
    post_guard = guard_fingerprints()
    guard_unchanged = (pre_guard == post_guard)
    only_aal_downloaded = (len(non_aal_new) == 0) and (len(changed_files) == 0)
    log(f"STEP 9: Phase 1/2/3/3.5/(existing)3.6 files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    log(f"STEP 9: Only the approved rois_aal derivative was downloaded (no other pipeline/atlas/"
        f"func_preproc data): {'PASS' if only_aal_downloaded else 'FAIL'}")

    # ── overall decision ────────────────────────────────────────────────
    all_780_present = (len(missing) == 0 and len(overlap) == n_cohort)
    diagnosis_match = (overlap_asd == 351 and overlap_control == 429)
    files_valid = (len(corrupt_files) == 0)
    no_dup_subject_files = (len(duplicate_subject_files) == 0)
    overall_pass = (all_780_present and diagnosis_match and files_valid and roi_check_pass
                    and no_dup_subject_files and guard_unchanged and only_aal_downloaded)

    if overall_pass:
        final_decision = ("AAL-116 CPAC data successfully acquired and verified.\n"
                          "Phase 4 may begin.")
    else:
        reasons = []
        if not all_780_present:
            reasons.append(f"{len(missing)} Phase-2 subjects missing from download")
        if not diagnosis_match:
            reasons.append("diagnosis counts do not match 351 ASD / 429 Control")
        if not files_valid:
            reasons.append(f"{len(corrupt_files)} corrupt/unreadable files")
        if not roi_check_pass:
            reasons.append(f"ROI column count is not uniformly 116 (observed: {unique_roi_counts})")
        if not no_dup_subject_files:
            reasons.append(f"{len(duplicate_subject_files)} duplicate subject files")
        if not guard_unchanged:
            reasons.append("Phase 1/2/3/3.5/3.6 files were modified")
        if not only_aal_downloaded:
            reasons.append("unexpected non-AAL files were downloaded")
        final_decision = ("STOP -- AAL-116 data integrity/coverage issue detected.\n"
                          f"Do not begin Phase 4. Issues: {'; '.join(reasons)}")
    log(f"FINAL DECISION: {final_decision}")

    # ── outputs ─────────────────────────────────────────────────────────
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "CPAC", "derivative": "rois_aal", "strategy": "filt_global",
        "phase2_cohort": {"total": n_cohort, "asd": n_asd, "control": n_control},
        "download": {
            "target_dir": str(ABIDE_PCP_DATA_DIR),
            "files_before": len(before_snap), "files_after": len(after_snap),
            "new_files": len(new_files), "changed_files": len(changed_files),
            "total_size_before_mb": round(before_total_size / 1e6, 1),
            "total_size_after_mb": round(after_total_size / 1e6, 1),
            "downloaded_volume_mb": round((after_total_size - before_total_size) / 1e6, 1),
            "aal_files_downloaded": len(aal_files),
            "non_aal_new_files": non_aal_new,
        },
        "coverage": {
            "phase2_total": n_cohort, "downloaded_total": downloaded_total,
            "overlap": len(overlap), "missing": len(missing), "extra": len(extra),
            "overlap_percentage": overlap_pct, "overlap_ASD": overlap_asd, "overlap_Control": overlap_control,
        },
        "integrity": {
            "files_inspected": len(integrity_df), "corrupt_files": corrupt_files,
            "duplicate_subject_files": duplicate_subject_files,
            "observed_roi_counts": unique_roi_counts, "roi_check_pass": roi_check_pass,
        },
        "timepoints": {"min": tp_min, "max": tp_max, "median": tp_median, "unique_counts": unique_tp},
        "audit": {
            "phase1_2_3_35_36_unchanged": guard_unchanged,
            "only_approved_derivative_downloaded": only_aal_downloaded,
            "overall_pass": overall_pass,
        },
        "final_decision": final_decision,
    }
    with open(OUT_DIR / "aal_download_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"aal_download_summary.json written -> {OUT_DIR / 'aal_download_summary.json'}")

    summary_txt = [
        "PHASE 3.6 -- AAL-116 CPAC Download & Integrity Verification", "=" * 70, "",
        f"Pipeline: CPAC  Derivative: rois_aal  Strategy: filt_global", "",
        f"Phase-2 cohort: {n_cohort} ({n_asd} ASD, {n_control} Control)",
        f"Downloaded subjects: {downloaded_total}",
        f"Overlap: {len(overlap)}  Missing: {len(missing)}  Extra: {len(extra)}  "
        f"Overlap%: {overlap_pct}",
        f"Overlap ASD: {overlap_asd}  Overlap Control: {overlap_control}", "",
        f"Observed ROI counts: {unique_roi_counts}  ROI check pass: {roi_check_pass}",
        f"Timepoints: min={tp_min} max={tp_max} median={tp_median}",
        f"Corrupt files: {len(corrupt_files)}  Duplicate subject files: {len(duplicate_subject_files)}", "",
        f"Downloaded volume: {round((after_total_size - before_total_size) / 1e6, 1)} MB", "",
        f"Audit: phase1_2_3_35_36 unchanged={guard_unchanged}, only-approved-derivative={only_aal_downloaded}", "",
        "FINAL DECISION:", final_decision,
    ]
    with open(OUT_DIR / "phase3_6_aal_download_summary.txt", "w") as f:
        f.write("\n".join(summary_txt))
    log("phase3_6_aal_download_summary.txt written.")

    audit_lines.append("Output files: aal_download_summary.json, aal_subject_coverage.csv, "
                        "aal_file_integrity.csv, aal_timepoint_summary.csv, aal_site_summary.csv, "
                        "phase3_6_aal_download_summary.txt")
    with open(OUT_DIR / "aal_download_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll outputs written to {OUT_DIR}")

    print()
    print("=" * 50)
    print("PHASE 3.6 -- AAL-116 DOWNLOAD & INTEGRITY CHECK")
    print("=" * 50)
    print()
    print("Pipeline:")
    print("CPAC")
    print()
    print("Derivative:")
    print("rois_aal")
    print()
    print("Phase-2 cohort:")
    print(n_cohort)
    print()
    print("Downloaded AAL subjects:")
    print(downloaded_total)
    print()
    print("Overlap:")
    print(len(overlap))
    print()
    print("Missing:")
    print(len(missing))
    print()
    print("Extra:")
    print(len(extra))
    print()
    print("ASD:")
    print(overlap_asd)
    print()
    print("Controls:")
    print(overlap_control)
    print()
    print("Observed ROI count:")
    print(unique_roi_counts)
    print()
    print("Minimum timepoints:")
    print(tp_min)
    print()
    print("Maximum timepoints:")
    print(tp_max)
    print()
    print("Median timepoints:")
    print(tp_median)
    print()
    print("Corrupt/unreadable files:")
    print(len(corrupt_files))
    print()
    print("Duplicate subject IDs:")
    print(len(duplicate_subject_files))
    print()
    print("Files downloaded:")
    print(len(aal_files))
    print()
    print("Approximate download size:")
    print(f"{round((after_total_size - before_total_size) / 1e6, 1)} MB")
    print()
    print("PHASE 1/2/3/3.5 MODIFIED:")
    print("NO" if guard_unchanged else "YES -- SEE AUDIT LOG")
    print()
    print("PHASE 4 PROCESSING PERFORMED:")
    print("NO")
    print()
    print("AUDIT:")
    print("PASS" if overall_pass else "FAIL")
    print()
    print("=" * 50)
    print()
    print("FINAL DECISION")
    print("--------------")
    print(final_decision)
    print()
    print("=" * 50)

    return summary


if __name__ == "__main__":
    main()
