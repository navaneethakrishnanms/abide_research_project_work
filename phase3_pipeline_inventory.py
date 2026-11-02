"""
phase3_pipeline_inventory.py -- Phase 3 (paper-reproduction track): fast,
read-only audit of which ABIDE preprocessing pipelines (CCS/CPAC/DPARSF/NIAK)
and ROI derivatives (AAL-116, CC200) actually exist on disk for the 780
FD-filtered subjects from Phase 2.

Metadata/filename inspection only. No fMRI volumes opened, no connectivity
computed, no AAL extraction performed, no downloads. Phase 1/Phase 2 files
are read-only inputs, never modified.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
ABIDE_PCP_ROOT  = ROOT / "abide_pcp" / "ABIDE_pcp"

OUT_DIR = ROOT / "results" / "phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

audit_lines = []


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg)


def extract_subject_id(filename: str) -> str | None:
    """'<Site>_<...>_<numeric_id>_rois_<atlas>.1D' -> numeric subject id (no leading zeros)."""
    m = re.search(r"_(\d+)_rois_", filename)
    if not m:
        return None
    return str(int(m.group(1)))


def main():
    log("Phase 3 pipeline/ROI inventory started (metadata-only).")
    log(f"Loading Phase 2 filtered cohort: {PHASE2_FILTERED}")

    filtered = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    n_filtered = len(filtered)
    n_unique = filtered["subject_id"].nunique()
    n_asd = int((filtered["diagnosis"] == "ASD").sum())
    n_control = int((filtered["diagnosis"] == "Control").sum())
    log(f"Filtered cohort loaded: {n_filtered} rows, {n_unique} unique subject IDs, "
        f"ASD={n_asd}, Control={n_control}")
    if n_filtered != 780 or n_unique != 780:
        log(f"[WARNING] Expected 780 unique subjects, got {n_filtered} rows / {n_unique} unique.")
    if n_asd != 351 or n_control != 429:
        log(f"[WARNING] Expected ASD=351/Control=429, got ASD={n_asd}/Control={n_control}.")

    filtered_ids = set(filtered["subject_id"])

    # ── STEP 2: search for pipeline directories/files by name, not content ──
    known_pipeline_names = ["ccs", "cpac", "dparsf", "niak"]
    pipeline_dirs_found = {}
    if ABIDE_PCP_ROOT.exists():
        top_level_dirs = [p.name.lower() for p in ABIDE_PCP_ROOT.iterdir() if p.is_dir()]
        log(f"Top-level directories under {ABIDE_PCP_ROOT}: {top_level_dirs}")
        for pname in known_pipeline_names:
            pipeline_dirs_found[pname] = pname in top_level_dirs
    else:
        log(f"[WARNING] {ABIDE_PCP_ROOT} does not exist.")
        for pname in known_pipeline_names:
            pipeline_dirs_found[pname] = False

    # Also search the whole project tree (excluding .venv) for any file/dir
    # whose name mentions these pipelines or "aal", in case data lives
    # somewhere unexpected. Filename-based only -- no file content opened.
    project_wide_hits = {p: [] for p in known_pipeline_names + ["aal"]}
    for path in ROOT.rglob("*"):
        try:
            if ".venv" in path.parts:
                continue
        except Exception:
            continue
        name_lower = path.name.lower()
        for key in project_wide_hits:
            if key in name_lower:
                project_wide_hits[key].append(str(path))
    for key, hits in project_wide_hits.items():
        log(f"Project-wide filename search for '{key}': {len(hits)} match(es)"
            + (f" -- e.g. {hits[:3]}" if hits else ""))

    # ── ROI atlas file inventory (only CPAC/filt_global confirmed to exist) ──
    roi_records = []  # (pipeline, strategy, atlas, subject_id, filepath)
    if ABIDE_PCP_ROOT.exists():
        for pipeline_dir in ABIDE_PCP_ROOT.iterdir():
            if not pipeline_dir.is_dir():
                continue
            pipeline = pipeline_dir.name
            for strategy_dir in pipeline_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue
                strategy = strategy_dir.name
                for f in strategy_dir.glob("*_rois_*.1D"):
                    m = re.search(r"_rois_([a-zA-Z0-9]+)\.1D$", f.name)
                    atlas = m.group(1) if m else "unknown"
                    sub_id = extract_subject_id(f.name)
                    roi_records.append({
                        "pipeline": pipeline, "strategy": strategy, "atlas": atlas,
                        "subject_id": sub_id, "filename": f.name,
                    })
    roi_df = pd.DataFrame(roi_records)
    log(f"ROI derivative files enumerated: {len(roi_df)} total "
        f"(pipelines: {sorted(roi_df['pipeline'].unique()) if len(roi_df) else []}, "
        f"strategies: {sorted(roi_df['strategy'].unique()) if len(roi_df) else []}, "
        f"atlases: {sorted(roi_df['atlas'].unique()) if len(roi_df) else []})")

    pipeline_inventory_path = OUT_DIR / "pipeline_inventory.csv"
    roi_df.to_csv(pipeline_inventory_path, index=False)
    log(f"pipeline_inventory.csv written -> {pipeline_inventory_path}")

    # ── STEP 3: per-pipeline subject coverage vs the 780 filtered cohort ────
    coverage_rows = []
    for pname in known_pipeline_names:
        sub_df = roi_df[roi_df["pipeline"].str.lower() == pname] if len(roi_df) else pd.DataFrame()
        if len(sub_df) == 0:
            coverage_rows.append({
                "pipeline": pname.upper(), "available": "NO",
                "subjects_available_total": 0, "subjects_in_filtered_cohort": 0,
                "asd_in_filtered_cohort": 0, "control_in_filtered_cohort": 0,
                "missing_from_filtered_cohort": n_filtered,
            })
            continue
        pipeline_ids = set(sub_df["subject_id"].dropna())
        covered = pipeline_ids & filtered_ids
        covered_df = filtered[filtered["subject_id"].isin(covered)]
        coverage_rows.append({
            "pipeline": pname.upper(), "available": "YES",
            "subjects_available_total": len(pipeline_ids),
            "subjects_in_filtered_cohort": len(covered),
            "asd_in_filtered_cohort": int((covered_df["diagnosis"] == "ASD").sum()),
            "control_in_filtered_cohort": int((covered_df["diagnosis"] == "Control").sum()),
            "missing_from_filtered_cohort": n_filtered - len(covered),
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = OUT_DIR / "pipeline_subject_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    log(f"pipeline_subject_coverage.csv written -> {coverage_path}")
    for _, row in coverage_df.iterrows():
        log(f"  {row['pipeline']}: available={row['available']}  "
            f"covered={row['subjects_in_filtered_cohort']}/{n_filtered}  "
            f"ASD={row['asd_in_filtered_cohort']}  Control={row['control_in_filtered_cohort']}")

    # ── STEP 4: AAL availability ─────────────────────────────────────────
    aal_df = roi_df[roi_df["atlas"].str.lower() == "aal"] if len(roi_df) else pd.DataFrame()
    aal_available = len(aal_df) > 0
    log(f"AAL ROI derivative files found: {len(aal_df)} -- AAL_AVAILABLE={'YES' if aal_available else 'NO'}")

    aal_records = []
    if aal_available:
        for _, row in aal_df.iterrows():
            in_cohort = row["subject_id"] in filtered_ids
            aal_records.append({
                "subject_id": row["subject_id"], "pipeline": row["pipeline"],
                "strategy": row["strategy"], "n_rois": None, "n_timepoints": None,
                "in_780_filtered_cohort": in_cohort,
            })
    aal_avail_df = pd.DataFrame(aal_records, columns=[
        "subject_id", "pipeline", "strategy", "n_rois", "n_timepoints", "in_780_filtered_cohort"])
    aal_path = OUT_DIR / "aal_availability.csv"
    aal_avail_df.to_csv(aal_path, index=False)
    log(f"aal_availability.csv written -> {aal_path} ({len(aal_avail_df)} rows)")
    aal_coverage_in_cohort = int(aal_avail_df["in_780_filtered_cohort"].sum()) if len(aal_avail_df) else 0

    # ── STEP 5: CC200 coverage of the 780 filtered cohort (comparison only) ──
    cc200_df = roi_df[roi_df["atlas"].str.lower() == "cc200"] if len(roi_df) else pd.DataFrame()
    cc200_ids = set(cc200_df["subject_id"].dropna())
    cc200_covered = cc200_ids & filtered_ids
    log(f"CC200 ROI derivative files found: {len(cc200_df)} total; "
        f"{len(cc200_covered)}/{n_filtered} of the FD-filtered cohort have CC200 data "
        f"(comparison only -- NOT used for the paper pipeline)")

    # ── STEP 6: strategies present ────────────────────────────────────────
    strategies_present = sorted(roi_df["strategy"].unique().tolist()) if len(roi_df) else []
    known_strategies = ["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"]
    strategy_rows = [{"strategy": s, "present": s in strategies_present} for s in known_strategies]
    strategy_summary_path = OUT_DIR / "pipeline_strategy_summary.csv"
    pd.DataFrame(strategy_rows).to_csv(strategy_summary_path, index=False)
    log(f"pipeline_strategy_summary.csv written -> {strategy_summary_path}. "
        f"Present strategies: {strategies_present}")

    # ── STEP 7/8: paper comparison + recommendation ─────────────────────
    paper_pipelines = ["CCS", "CPAC", "DPARSF", "NIAK"]
    availability_status = {}
    for _, row in coverage_df.iterrows():
        if row["available"] == "NO":
            availability_status[row["pipeline"]] = "NOT AVAILABLE"
        elif row["subjects_in_filtered_cohort"] == n_filtered:
            availability_status[row["pipeline"]] = "AVAILABLE"
        else:
            availability_status[row["pipeline"]] = "PARTIALLY AVAILABLE"

    log(f"Paper pipeline comparison: {availability_status}")
    log(f"AAL 116-ROI comparison: paper=YES, ours={'AVAILABLE' if aal_available else 'NOT AVAILABLE'}")

    if not aal_available:
        recommendation = "STOP -- no AAL-116 ROI derivative data is available for any pipeline."
        reason = ("AAL ROI time-series files ('rois_aal' or equivalent) were not found anywhere "
                  "in the project (0 matches, project-wide filename search). Only CC200 ROI "
                  "derivatives exist, and only for the CPAC pipeline / filt_global strategy. "
                  "None of CCS, DPARSF, or NIAK derivative data is present at all. Proceeding to "
                  "AAL extraction (Phase 4) would require either downloading AAL-parcellated "
                  "derivatives (if available from the ABIDE PCP release) or performing AAL "
                  "extraction from raw/preprocessed volumes ourselves -- both explicitly out of "
                  "scope for this phase.")
    else:
        # (kept for completeness/reproducibility if AAL data is added later)
        priority = ["DPARSF", "CPAC"] + [p for p in paper_pipelines if p not in ("DPARSF", "CPAC")]
        recommendation = "See per-pipeline AAL coverage in aal_availability.csv"
        reason = "AAL data present -- pick by coverage/priority once available."

    # ── STEP 9: JSON + summary + audit ───────────────────────────────────
    inventory = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_phase2_filtered_cohort": str(PHASE2_FILTERED),
        "filtered_cohort": {"total": n_filtered, "unique": n_unique, "asd": n_asd, "control": n_control},
        "abide_pcp_root_inspected": str(ABIDE_PCP_ROOT),
        "top_level_pipeline_dirs_found": [k for k, v in pipeline_dirs_found.items() if v],
        "project_wide_filename_search_hits": {k: len(v) for k, v in project_wide_hits.items()},
        "roi_derivative_files_total": len(roi_df),
        "pipelines_found_on_disk": sorted(roi_df["pipeline"].unique().tolist()) if len(roi_df) else [],
        "strategies_found_on_disk": strategies_present,
        "atlases_found_on_disk": sorted(roi_df["atlas"].unique().tolist()) if len(roi_df) else [],
        "pipeline_coverage": coverage_df.to_dict(orient="records"),
        "aal_available": aal_available,
        "aal_files_found": len(aal_df),
        "aal_coverage_in_780_cohort": aal_coverage_in_cohort,
        "cc200_files_found": len(cc200_df),
        "cc200_coverage_in_780_cohort": len(cc200_covered),
        "cc200_note": "CC200 comparison only -- NOT used for the paper (AAL-based) pipeline.",
        "paper_comparison": {
            "paper_pipelines": paper_pipelines, "paper_atlas": "AAL (116 ROIs)",
            "our_availability_status": availability_status,
            "our_aal_status": "AVAILABLE" if aal_available else "NOT AVAILABLE",
        },
        "recommendation": recommendation,
        "recommendation_reason": reason,
    }
    inventory_path = OUT_DIR / "phase3_inventory.json"
    with open(inventory_path, "w") as f:
        json.dump(inventory, f, indent=2)
    log(f"phase3_inventory.json written -> {inventory_path}")

    summary_lines = [
        "PHASE 3 SUMMARY -- Pipeline & ROI Derivative Availability Audit",
        "=" * 60, "",
        f"Filtered cohort (from Phase 2): {n_filtered}  (ASD={n_asd}, Control={n_control})", "",
        "PIPELINE AVAILABILITY",
    ]
    for _, row in coverage_df.iterrows():
        summary_lines.append(
            f"  {row['pipeline']:<8} available={row['available']:<3}  "
            f"covered={row['subjects_in_filtered_cohort']}/{n_filtered}"
        )
    summary_lines += [
        "",
        f"AAL (116 ROIs) availability: {'YES' if aal_available else 'NO'}",
        f"AAL coverage in 780-subject cohort: {aal_coverage_in_cohort}/{n_filtered}",
        "",
        f"CC200 coverage in 780-subject cohort (comparison only): {len(cc200_covered)}/{n_filtered}",
        "",
        f"Strategies present: {strategies_present}",
        "",
        "PAPER COMPARISON",
        f"  Paper pipelines: {paper_pipelines}",
        f"  Our status: {availability_status}",
        f"  Paper atlas: AAL-116  |  Our AAL status: {'AVAILABLE' if aal_available else 'NOT AVAILABLE'}",
        "",
        "RECOMMENDATION",
        f"  {recommendation}",
        f"  Reason: {reason}",
    ]
    with open(OUT_DIR / "phase3_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    log(f"phase3_summary.txt written -> {OUT_DIR / 'phase3_summary.txt'}")

    audit_lines.append("Output files: pipeline_inventory.csv, pipeline_subject_coverage.csv, "
                        "aal_availability.csv, pipeline_strategy_summary.csv, phase3_summary.txt, "
                        "phase3_inventory.json")
    with open(OUT_DIR / "phase3_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 3 outputs written to {OUT_DIR}")

    return inventory


if __name__ == "__main__":
    main()
