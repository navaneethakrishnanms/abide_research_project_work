"""
phase3_6_aal_verify_current_status.py -- produces the AUTHORITATIVE,
freshly-verified status of the CPAC AAL-116 dataset, superseding (without
deleting or modifying) the stale/misleading
results/phase3_6_aal_download/aal_download_summary.json.

Why this exists: that file was written by phase3_6_aal_download.py's
SECOND (resumed) run, whose before/after snapshot only measured files
fetched during that specific resumed process (596 files), not the full
cumulative set already on disk from the first run (~867 files) plus the
second (596 new) -- so its "coverage"/"missing" numbers and its FAIL
verdict are an artifact of measuring a delta, not the true current state.
This script re-derives the true state directly from what is actually on
disk right now, with no assumptions carried over from any prior run's
self-reported counters.

Authoritative sources used here (per explicit instruction):
  - results/phase2/fd_filtered_subjects.csv is the authoritative cohort.
  - The CPAC rois_aal .1D files currently present under
    abide_pcp/ABIDE_pcp/cpac/filt_global/ are the authoritative downloaded
    data (not any prior run's summary file).

Read-only with respect to all prior outputs: this script only WRITES a
new file, aal_current_verified_status.json (and a matching .txt), in the
same results/phase3_6_aal_download/ directory. It does not delete, modify,
or overwrite aal_download_summary.json or any other historical file.
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
OUT_DIR = ROOT / "results" / "phase3_6_aal_download"

NEW_JSON = OUT_DIR / "aal_current_verified_status.json"
NEW_TXT = OUT_DIR / "aal_current_verified_status.txt"
STALE_JSON = OUT_DIR / "aal_download_summary.json"  # preserved, never written to here


def main():
    assert STALE_JSON.exists(), "Historical file unexpectedly missing -- not touching anything."
    stale_mtime_before = STALE_JSON.stat().st_mtime

    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    cohort["subject_id_norm"] = cohort["subject_id"].map(lambda x: str(int(float(x))))
    n_cohort = len(cohort)
    asd = int((cohort["diagnosis"] == "ASD").sum())
    control = int((cohort["diagnosis"] == "Control").sum())
    n_dup = int(cohort["subject_id"].duplicated().sum())
    cohort_ids = set(cohort["subject_id_norm"])
    cohort_lookup = cohort.set_index("subject_id_norm")

    files = list(AAL_DIR.rglob("*_rois_aal.1D"))
    id_to_file = {}
    for f in files:
        base = f.name.replace("_rois_aal.1D", "")
        digits = "".join(ch for ch in base.split("_")[-1] if ch.isdigit())
        if digits:
            id_to_file[str(int(digits))] = f
    downloaded_ids = set(id_to_file.keys())

    overlap = cohort_ids & downloaded_ids
    missing = cohort_ids - downloaded_ids
    extra = downloaded_ids - cohort_ids
    overlap_pct = round(100 * len(overlap) / n_cohort, 2) if n_cohort else 0.0
    overlap_asd = sum(1 for sid in overlap if cohort_lookup.loc[sid, "diagnosis"] == "ASD")
    overlap_control = sum(1 for sid in overlap if cohort_lookup.loc[sid, "diagnosis"] == "Control")

    roi_counts = set()
    timepoints = []
    corrupt = []
    for sid, f in id_to_file.items():
        try:
            if f.stat().st_size == 0:
                corrupt.append({"subject_id": sid, "reason": "zero-byte"})
                continue
            arr = np.loadtxt(str(f))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            roi_counts.add(int(arr.shape[1]))
            timepoints.append(int(arr.shape[0]))
            if not np.isfinite(arr).all():
                corrupt.append({"subject_id": sid, "reason": "contains NaN/Inf"})
        except Exception as exc:
            corrupt.append({"subject_id": sid, "reason": f"{type(exc).__name__}: {exc}"})

    part_files_remaining = list(AAL_DIR.rglob("*.part"))
    roi_check_pass = (len(roi_counts) == 1 and next(iter(roi_counts)) == 116)

    all_780_present = len(missing) == 0
    diagnosis_match = (overlap_asd == 351 and overlap_control == 429)
    files_valid = len(corrupt) == 0
    overall_pass = (all_780_present and diagnosis_match and files_valid and roi_check_pass
                    and n_dup == 0 and len(part_files_remaining) == 0)

    status = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Authoritative, freshly re-derived AAL-116/CPAC status. Supersedes "
                  "aal_download_summary.json for decision-making purposes -- that file is "
                  "PRESERVED, UNMODIFIED, and known to undercount coverage because it only "
                  "measured files fetched during one resumed run, not the full set on disk.",
        "authoritative_cohort_source": str(PHASE2_FILTERED),
        "authoritative_data_source": str(AAL_DIR) + " (all currently-present *_rois_aal.1D files)",
        "phase2_cohort": {"total": n_cohort, "asd": asd, "control": control, "duplicates": n_dup},
        "aal_files_on_disk": len(files),
        "unique_subject_ids_on_disk": len(downloaded_ids),
        "coverage": {
            "phase2_total": n_cohort, "downloaded_total": len(downloaded_ids),
            "overlap": len(overlap), "missing": len(missing), "extra": len(extra),
            "overlap_percentage": overlap_pct,
            "overlap_ASD": overlap_asd, "overlap_Control": overlap_control,
        },
        "integrity": {
            "files_inspected": len(id_to_file), "corrupt_files": corrupt,
            "observed_roi_counts": sorted(roi_counts), "roi_check_pass": roi_check_pass,
            "stale_part_files_remaining": [str(p) for p in part_files_remaining],
        },
        "timepoints": {
            "min": min(timepoints) if timepoints else None,
            "max": max(timepoints) if timepoints else None,
            "median": float(np.median(timepoints)) if timepoints else None,
        },
        "overall_pass": overall_pass,
        "decision": ("AAL-116 CPAC data successfully acquired and verified.\nPhase 4 may begin."
                    if overall_pass else
                    "STOP -- AAL-116 data integrity/coverage issue detected.\nDo not begin Phase 4."),
        "supersedes": str(STALE_JSON) + " (preserved as historical record, not deleted or modified; "
                      "that file's coverage numbers are an artifact of measuring only one resumed "
                      "run's incremental download, not the true cumulative state)",
    }

    with open(NEW_JSON, "w") as f:
        json.dump(status, f, indent=2)

    txt_lines = [
        "AAL-116 / CPAC -- CURRENT VERIFIED STATUS (authoritative)", "=" * 60, "",
        f"Generated: {status['generated_utc']}", "",
        f"Phase-2 cohort (authoritative): {n_cohort} ({asd} ASD, {control} Control, {n_dup} duplicates)",
        f"AAL files on disk: {len(files)}", "",
        f"Coverage: {len(overlap)}/{n_cohort} ({overlap_pct}%)  missing={len(missing)}  extra={len(extra)}",
        f"Overlap ASD={overlap_asd}  Overlap Control={overlap_control}", "",
        f"Observed ROI counts: {sorted(roi_counts)}  (pass={roi_check_pass})",
        f"Timepoints: min={status['timepoints']['min']} max={status['timepoints']['max']} "
        f"median={status['timepoints']['median']}",
        f"Corrupt files: {len(corrupt)}  Stale .part files: {len(part_files_remaining)}", "",
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}", "",
        status["decision"], "",
        f"Supersedes (for decision-making; NOT deleted/modified): {STALE_JSON}",
    ]
    with open(NEW_TXT, "w") as f:
        f.write("\n".join(txt_lines))

    stale_mtime_after = STALE_JSON.stat().st_mtime
    assert stale_mtime_before == stale_mtime_after, "Historical file was touched -- this must not happen."

    print(f"Wrote: {NEW_JSON}")
    print(f"Wrote: {NEW_TXT}")
    print(f"Confirmed untouched: {STALE_JSON} (mtime unchanged)")
    print()
    print(f"Coverage: {len(overlap)}/{n_cohort} ({overlap_pct}%)  ROI={sorted(roi_counts)}  "
          f"corrupt={len(corrupt)}  overall_pass={overall_pass}")


if __name__ == "__main__":
    main()
