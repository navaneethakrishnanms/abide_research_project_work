"""
phase3_6_aal_source_investigation.py -- Phase 3.6: AAL-116 data source
investigation.

Investigation only. No extraction, no FC, no downloads of subject data,
no modification of Phase 1/2/3/3.5 outputs or original CPAC/raw data.

Findings are drawn from three evidence classes, each labeled:
  - CONFIRMED: verified directly from a locally-installed package's actual
    source code (nilearn 0.14.0, installed in the anaconda env that ran
    fetch_abide.py), or from local file inspection.
  - CONFIRMED (external): verified via WebSearch/WebFetch against the
    official ABIDE PCP GitHub repository and project documentation.
  - INFERRED: a reasoned conclusion from the above evidence, not yet
    empirically verified by actually performing a fetch/download.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
PHASE1_MANIFEST = ROOT / "results" / "phase1" / "subject_manifest.csv"
PHASE3_DIR      = ROOT / "results" / "phase3"
PHASE3_5_DIR    = ROOT / "results" / "phase3_5"

OUT_DIR = ROOT / "results" / "phase3_6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

audit_lines = []


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg)


def fp(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime}:{st.st_size}"


def main():
    guarded = [
        PHASE1_MANIFEST, PHASE2_FILTERED,
        PHASE3_DIR / "pipeline_inventory.csv", PHASE3_DIR / "phase3_inventory.json",
        PHASE3_5_DIR / "cpac_bold_subject_coverage.csv", PHASE3_5_DIR / "phase3_5_feasibility.json",
    ]
    pre = {p: fp(p) for p in guarded if p.exists()}

    log("=" * 60)
    log("PHASE 3.6 -- AAL-116 data source investigation started (no downloads, no extraction).")
    log("=" * 60)

    # ── STEP 1: authoritative cohort ─────────────────────────────────────
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    n_cohort, n_asd, n_control = len(cohort), int((cohort.diagnosis == "ASD").sum()), int((cohort.diagnosis == "Control").sum())
    n_dup = int(cohort["subject_id"].duplicated().sum())
    log(f"STEP 1: Phase-2 cohort loaded: {n_cohort} subjects, ASD={n_asd}, Control={n_control}, dup={n_dup}")

    # ── STEP 1 (local search) -- reuses Phase 3/3.5's already-established facts ──
    local_search_rows = [
        {"search_term": "AAL / AAL116 / AAL_116 / aal.nii(.gz) / aal116.nii(.gz)", "scope": "project-wide filenames",
         "result": "0 matches", "evidence_type": "CONFIRMED", "source": "Phase 3 + Phase 3.5 project-wide rglob search"},
        {"search_term": "AAL ROI time series / AAL connectivity", "scope": "project-wide filenames + directory structure",
         "result": "0 matches", "evidence_type": "CONFIRMED", "source": "Phase 3 pipeline_inventory.csv (871 files, all atlas=cc200)"},
        {"search_term": "README / manifest / JSON / config / script mentions of 'AAL'", "scope": "project-wide file content (docstrings, comments, configs)",
         "result": "No AAL references found in any project script, README, or config", "evidence_type": "CONFIRMED",
         "source": "Manual review of fetch_abide.py, converter.py, README.md, config.py during Phase 3/3.5"},
        {"search_term": "ABIDE PCP / preprocessed connectomes / Nilearn cached metadata", "scope": "abide_pcp/ directory",
         "result": "phenotypic_qc.csv and CC200 .1D derivatives present (from a prior fetch_abide_pcp(derivatives=['rois_cc200']) call); "
                  "no AAL derivative was ever fetched", "evidence_type": "CONFIRMED",
         "source": "abide_pcp/fetch.log, fetch_abide.py source"},
        {"search_term": "Local AAL-116 atlas file (atlas-only, no subject data)", "scope": "project-wide + derivatives/template/",
         "result": "NOT FOUND. derivatives/template/ contains mni152_t1_2mm.nii.gz + mni152_mask_2mm.nii.gz "
                  "(a T1 template and brain mask -- NOT an AAL region-label atlas)", "evidence_type": "CONFIRMED",
         "source": "Phase 3.5 aal_atlas_local_availability.csv"},
    ]
    local_df = pd.DataFrame(local_search_rows)
    local_df.to_csv(OUT_DIR / "aal_local_search.csv", index=False)
    log(f"STEP 1: aal_local_search.csv written -- confirms NOTHING locally available provides AAL-116 information.")

    # ── STEP 2/3: ABIDE PCP + Nilearn investigation (verified from installed source + official docs) ──
    log("STEP 2/3: Inspecting locally-installed nilearn source (anaconda env, the SAME environment "
        "that ran fetch_abide.py) for fetch_abide_pcp's actual, real capabilities -- not assumed.")
    log("  CONFIRMED (nilearn 0.14.0 source, nilearn/datasets/func.py, fetch_abide_pcp): "
        "the `derivatives` parameter accepts 'rois_aal' as a valid value, explicitly documented as "
        "'data specific to anatomical regions defined by the Automatic Anatomical Labeling atlas.' "
        "It is fetched as a per-subject .1D ROI time-series text file -- the SAME format/mechanism "
        "already used successfully in this project for rois_cc200.")
    log("  CONFIRMED (nilearn source): valid `pipeline` values are 'ccs', 'cpac', 'dparsf', 'niak' "
        "-- all four of the paper's pipelines are directly fetchable through the same function "
        "already used in fetch_abide.py, just by changing the `pipeline=` and `derivatives=` arguments.")
    log("  CONFIRMED (nilearn source): download URL pattern is "
        "'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/{pipeline}/{strategy}/"
        "{derivative}/{file_id}_{derivative}.1D' -- i.e. AAL derivatives are hosted on the identical "
        "S3 bucket/PCP release already used for our existing CC200 data, not a separate/unknown source.")
    log("  CONFIRMED (external, github.com/preprocessed-connectomes-project/abide, abide_rois.txt): "
        "the PCP pipeline config references an AAL atlas mask file 'aal_mask_pad.nii.gz' used to "
        "generate the rois_aal derivative. This file does NOT itself state a region count (116) or "
        "AAL version in the text consulted -- see Step 5.")
    log("  CONFIRMED (external, web search of PCP/NITRC documentation): PCP explicitly states "
        "functional preprocessing was performed with CCS, CPAC, DPARSF, and NIAK, each in 4 "
        "strategies (filt_global, filt_noglobal, nofilt_global, nofilt_noglobal), and rois_aal is "
        "listed among the standard derivative types for the release.")
    log("  One search result reported '1102 records available for download from the DPARSF pipeline "
        "for the AAL derivative with band-pass filtering, no global signal regression' -- this is an "
        "UNVERIFIED THIRD-PARTY SUMMARY (not fetched/confirmed by this investigation directly), "
        "cited here as supporting-but-unconfirmed evidence, not as a confirmed fact.")

    abide_pcp_rows = [
        {"resource_name": "rois_aal (AAL-116 ROI time series)", "pipeline": "CPAC", "strategy": "filt_global (or any of the 4 strategies)",
         "atlas": "AAL (aal_mask_pad.nii.gz per PCP config)", "data_type": "AAL-116 ROI time series (.1D)",
         "subject_coverage": "INFERRED: same subject set as our existing 871-subject phenotypic_qc.csv "
                             "(same phenotypic query/QC filter; derivative type does not change which "
                             "subjects are listed) -- NOT yet empirically verified by an actual fetch.",
         "site_coverage": "Same ABIDE-I sites as CPAC/CC200 (20 sites, per Phase 1)",
         "download_size_if_known": "~190-200MB estimated (scaled from the existing 871-file CC200 set: "
                                   "338MB at 200 regions/file -> AAL at 116 regions/file, ~58% column "
                                   "count -> ~196MB; this is an ESTIMATE, not a measured value)",
         "source": "nilearn.datasets.fetch_abide_pcp (installed, v0.14.0) + fcp-indi S3 bucket",
         "verified": "PARTIALLY (mechanism/URL/derivative-name CONFIRMED from source code and official "
                    "docs; exact subject count and 780-cohort overlap NOT YET EMPIRICALLY VERIFIED)",
         "notes": "Would require a new (small) download to confirm; not yet performed, per Phase 3.6 rules."},
        {"resource_name": "rois_aal (AAL-116 ROI time series)", "pipeline": "DPARSF", "strategy": "filt_noglobal (per 3rd-party search summary) or others",
         "atlas": "AAL (aal_mask_pad.nii.gz per PCP config)", "data_type": "AAL-116 ROI time series (.1D)",
         "subject_coverage": "UNCERTAIN: DPARSF pipeline may have a DIFFERENT quality-checked subject "
                             "set than CPAC (QC rater columns can be pipeline-specific in the PCP "
                             "phenotypic file) -- NOT the same 871/780 subjects by default assumption.",
         "site_coverage": "Same nominal ABIDE-I sites; exact per-site DPARSF yield unconfirmed",
         "download_size_if_known": "Same order of magnitude as CPAC estimate above (~150-250MB), unconfirmed",
         "source": "nilearn.datasets.fetch_abide_pcp (pipeline='dparsf') + fcp-indi S3 bucket",
         "verified": "PARTIALLY (mechanism CONFIRMED; DPARSF-specific subject count/QC overlap NOT VERIFIED)",
         "notes": "This is the paper's preferred pipeline (paper reports its best result using DPARSF) "
                 "-- highest-priority candidate to verify next, via a phenotypic-metadata-only check."},
        {"resource_name": "AAL-116 functional connectivity (precomputed)", "pipeline": "N/A", "strategy": "N/A",
         "atlas": "N/A", "data_type": "AAL-116 FC matrices",
         "subject_coverage": "NOT FOUND -- ABIDE PCP / nilearn provide ROI TIME SERIES (rois_aal), not "
                             "precomputed FC matrices. No PCP resource for precomputed AAL FC was "
                             "identified in this investigation.",
         "site_coverage": "N/A", "download_size_if_known": "N/A",
         "source": "N/A", "verified": "NOT FOUND",
         "notes": "Even if such a resource existed, Step 7's rules require verifying its Pearson/"
                 "Fisher-Z/feature-construction procedure before treating it as equivalent -- moot here "
                 "since none was found."},
    ]
    abide_pcp_df = pd.DataFrame(abide_pcp_rows)
    abide_pcp_df.to_csv(OUT_DIR / "abide_pcp_aal_investigation.csv", index=False)
    log("STEP 2: abide_pcp_aal_investigation.csv written.")

    nilearn_rows = [
        {"capability": "fetch_abide_pcp(derivatives=['rois_aal'])", "provides_AAL_ROI_timeseries": True,
         "provides_AAL_parcellated_images": False, "provides_AAL_FC": False,
         "cpac_cc200_only": False, "other_atlases_supported": "cc200, cc400, dosenbach160, ez, ho, tt (rois_*), "
                                                               "plus non-ROI derivatives (alff, falff, reho, vmhc, etc.)",
         "evidence": "CONFIRMED directly from installed nilearn 0.14.0 source "
                    "(inspect.getsource(fetch_abide_pcp), nilearn/datasets/func.py) -- the SAME "
                    "environment that produced this project's existing CC200 data.",
         "notes": "Contrary to an assumption that Nilearn's ABIDE access is CC200-only, the function "
                 "itself is atlas-agnostic; CC200 was simply the derivative this project's "
                 "fetch_abide.py chose to request previously."},
    ]
    nilearn_df = pd.DataFrame(nilearn_rows)
    nilearn_df.to_csv(OUT_DIR / "nilearn_aal_investigation.csv", index=False)
    log("STEP 3: nilearn_aal_investigation.csv written -- Nilearn's fetch_abide_pcp is CONFIRMED to "
        "support AAL ROI time series directly; it was simply never requested before now.")

    # ── STEP 4/5: public AAL atlas source + version identification ─────────
    log("STEP 4: Candidate AAL atlas sources identified via public documentation (NOT downloaded):")
    log("  - The PCP-internal 'aal_mask_pad.nii.gz' used by ABIDE PCP itself to generate rois_aal "
        "(implied to exist server-side; not separately published as a standalone downloadable file "
        "in the materials reviewed here).")
    log("  - The standard AAL atlas is separately distributed through established neuroimaging "
        "software/atlas distributions (e.g. as bundled with common neuroimaging toolboxes, and via "
        "Nilearn's own general atlas-fetching utilities, distinct from fetch_abide_pcp) -- NOT "
        "downloaded or version-confirmed in this investigation.")
    log("STEP 5: AAL-116 VERSION IDENTIFICATION -- ")
    log('  "Paper specifies AAL-116 but does not establish the exact atlas version from the currently '
        'available evidence." The PCP config file (abide_rois.txt) references an AAL mask '
        '(aal_mask_pad.nii.gz) without stating a region count or version (original AAL vs. AAL2 vs. '
        'AAL3, which differ in region count and numbering). The commonly-cited "116 regions" figure '
        'is the standard/original AAL atlas convention, but this has NOT been confirmed against the '
        'specific file PCP uses, nor against the paper\'s own methods section (not available for '
        'direct inspection in this project). This should be treated as INFERRED, not CONFIRMED.')

    public_source_rows = [
        {"atlas_name": "AAL (original/standard)", "version": "UNCONFIRMED (commonly 116 regions, not verified here)",
         "number_of_regions": "116 (INFERRED, standard convention -- not confirmed against PCP's specific file)",
         "format": ".nii/.nii.gz volumetric label atlas", "coordinate_template_space": "UNCONFIRMED (typically MNI space for standard AAL)",
         "source": "General neuroimaging atlas distributions (not identified to a specific verified URL in this investigation)",
         "license_if_available": "UNKNOWN (not investigated)", "compatibility_notes": "Would need version/space confirmation before use",
         "verified": "NOT VERIFIED (identification only, no file inspected)"},
        {"atlas_name": "aal_mask_pad.nii.gz (PCP-internal)", "version": "UNKNOWN",
         "number_of_regions": "UNKNOWN (not stated in the config file consulted)",
         "format": ".nii.gz (implied, per PCP naming convention)", "coordinate_template_space": "UNKNOWN",
         "source": "github.com/preprocessed-connectomes-project/abide (referenced in pipeline config, "
                  "not confirmed to be separately downloadable as a standalone file)",
         "license_if_available": "UNKNOWN", "compatibility_notes": "This is the atlas PCP itself used to "
                                                                    "generate rois_aal -- if rois_aal is used, this specific file's identity matters "
                                                                    "less, since the ROI extraction is already done server-side.",
         "verified": "CONFIRMED referenced in PCP config; file itself NOT independently inspected"},
    ]
    pd.DataFrame(public_source_rows).to_csv(OUT_DIR / "public_aal_atlas_sources.csv", index=False)
    log("STEP 4/5: public_aal_atlas_sources.csv written (identification only, nothing downloaded).")

    # ── STEP 6: cohort compatibility ────────────────────────────────────
    log("STEP 6: Subject-level cohort compatibility CANNOT be populated with real subject_id rows "
        "without performing at least a phenotypic-metadata fetch (which Phase 3.6 rules require "
        "stopping before, per Step 9). Reporting this honestly rather than fabricating subject rows.")
    compat_rows = [{
        "source": "ABIDE PCP rois_aal (CPAC pipeline)", "data_type": "AAL-116 ROI time series",
        "subject_id": "NOT YET VERIFIED (no fetch performed)", "in_phase2_cohort": "INFERRED: likely YES for "
        "all/most of 780 (same underlying phenotypic query as our existing CC200 fetch)",
        "diagnosis": "N/A (not fetched)", "site": "N/A (not fetched)", "pipeline": "CPAC",
        "strategy": "filt_global (matches our existing Phase 1-3 pipeline choice)", "atlas": "AAL",
        "compatibility": "INFERRED HIGH -- pending empirical verification",
        "notes": "Requires a small phenotypic-metadata-only fetch to confirm subject-level overlap; "
                "not performed in this investigation.",
    }, {
        "source": "ABIDE PCP rois_aal (DPARSF pipeline)", "data_type": "AAL-116 ROI time series",
        "subject_id": "NOT YET VERIFIED (no fetch performed)", "in_phase2_cohort": "UNCERTAIN -- DPARSF "
        "may have a different QC'd subject set than CPAC",
        "diagnosis": "N/A (not fetched)", "site": "N/A (not fetched)", "pipeline": "DPARSF",
        "strategy": "UNCONFIRMED (paper's specific strategy not established from available evidence)",
        "atlas": "AAL", "compatibility": "UNCERTAIN -- pending empirical verification",
        "notes": "Highest-priority candidate (matches paper's preferred pipeline) but requires "
                "verification before any conclusion.",
    }]
    pd.DataFrame(compat_rows).to_csv(OUT_DIR / "aal_source_cohort_compatibility.csv", index=False)
    log("STEP 6: aal_source_cohort_compatibility.csv written (no subject IDs fabricated).")

    # ── STEP 7: precomputed FC check ────────────────────────────────────
    log("STEP 7: No AAL-116 precomputed FC resource was identified anywhere (local or via PCP/nilearn "
        "investigation) -- ABIDE PCP provides ROI TIME SERIES, not precomputed FC matrices, for the "
        "AAL derivative. This step is therefore moot: there is no precomputed-FC candidate to evaluate "
        "against Pearson/Fisher-Z/6,670-feature/cohort-matching criteria.")

    # ── STEP 8: existing local data sufficiency ──────────────────────────
    log("STEP 8: Per Phase 3.5 (unchanged, reused here, not recomputed): only 54/780 subjects have any "
        "local functional NIfTI file, all single-site (UCLA_1), none with confirmed CPAC identity. "
        "This does NOT change in Phase 3.6 -- local data remain insufficient for AAL extraction from "
        "existing files for the full 780-subject cohort.")

    # ── STEP 9: download disclosure (NOT executed) ───────────────────────
    log("STEP 9: POTENTIAL DOWNLOAD IDENTIFIED -- STOPPING BEFORE DOWNLOADING, per Phase 3.6 rules.")
    download_disclosure = {
        "WHAT": "AAL-116 ROI time series (.1D files) via nilearn.datasets.fetch_abide_pcp("
                "pipeline='cpac' or 'dparsf', derivatives=['rois_aal'], strategy matching filt_global "
                "or the paper's specified strategy)",
        "WHY": "This is the only verified, documented, legitimate AAL-116 ROI time-series source "
               "identified for ABIDE-I; it uses the same mechanism/S3 bucket already used successfully "
               "for our existing CC200 data, so it is the most direct path to real AAL-116 features "
               "without a full raw-data preprocessing effort.",
        "SIZE": "Estimated ~190-200MB for the CPAC/full-cohort case, based on scaling from the "
                "existing 871-file, 338MB rois_cc200 set (200 regions/file) to 116 regions/file. "
                "This is an ESTIMATE, not a measured size -- actual size unconfirmed until fetched.",
        "COVERAGE": "CPAC: INFERRED ~871 subjects (same as existing phenotypic_qc.csv), pending "
                    "verification. DPARSF: UNCERTAIN subject count, requires its own phenotypic check.",
        "PIPELINE": "CPAC (matches our existing Phase 1-3 choice) or DPARSF (matches the paper's "
                    "reported best-performing pipeline) -- your choice affects which is fetched.",
        "ATLAS": "AAL (aal_mask_pad.nii.gz per PCP's own pipeline config) -- exact version/region-count "
                 "NOT independently confirmed against the paper's methodology (see Step 5).",
        "COHORT_OVERLAP": "INFERRED high overlap with the 780-subject Phase-2 cohort for CPAC (same "
                          "phenotypic source); UNCONFIRMED for DPARSF.",
        "REPRODUCTION_IMPACT": "This would be the FIRST genuine AAL-116 ROI data available in this "
                               "project, directly enabling Phase 4 (Pearson correlation + Fisher-Z + "
                               "SVM-RFE + SSAE) on real AAL features rather than CC200 -- a substantial "
                               "improvement in paper fidelity IF the resource actually covers the 780-"
                               "subject cohort as inferred.",
    }
    for k, v in download_disclosure.items():
        log(f"  {k}: {v}")
    log("  NOT DOWNLOADED. Awaiting explicit user approval before any fetch is performed.")

    # ── STEP 10: option comparison ────────────────────────────────────────
    comparison_rows = [
        {"option": "A", "source": "ABIDE PCP rois_aal (CPAC)", "data_type": "AAL-116 ROI time series",
         "atlas": "AAL", "pipeline": "CPAC", "strategy": "filt_global",
         "subject_coverage": "INFERRED ~871 (matches existing phenotypic_qc.csv)", "site_coverage": "20 sites (inferred, same as CPAC/CC200)",
         "same_780_cohort": "INFERRED YES (not yet empirically verified)", "paper_compatible": "PARTIALLY -- paper's best result uses DPARSF, not CPAC",
         "download_required": "YES (small, ~190-200MB estimated)", "estimated_effort": "LOW (same mechanism as existing CC200 fetch)",
         "methodology_deviation": "Pipeline differs from paper's best-reported pipeline (DPARSF), atlas matches (AAL)",
         "recommendation": "STRONG CANDIDATE", "reason": "Verified mechanism, same subject source as existing 780 cohort, real AAL atlas"},
        {"option": "A2", "source": "ABIDE PCP rois_aal (DPARSF)", "data_type": "AAL-116 ROI time series",
         "atlas": "AAL", "pipeline": "DPARSF", "strategy": "UNCONFIRMED",
         "subject_coverage": "UNCONFIRMED", "site_coverage": "UNCONFIRMED",
         "same_780_cohort": "UNCERTAIN", "paper_compatible": "BEST MATCH -- paper's reported best pipeline",
         "download_required": "YES (small, phenotypic check first, then ROI data)", "estimated_effort": "LOW-MEDIUM (pipeline/subject overlap unverified)",
         "methodology_deviation": "None if subject coverage/strategy confirmed -- matches paper's pipeline exactly",
         "recommendation": "HIGHEST-PRIORITY CANDIDATE, PENDING VERIFICATION", "reason": "Exact pipeline match to paper, but coverage unconfirmed"},
        {"option": "B", "source": "AAL-116 FC derivatives", "data_type": "Precomputed FC",
         "atlas": "AAL", "pipeline": "N/A", "strategy": "N/A", "subject_coverage": "N/A", "site_coverage": "N/A",
         "same_780_cohort": "N/A", "paper_compatible": "N/A", "download_required": "N/A", "estimated_effort": "N/A",
         "methodology_deviation": "N/A", "recommendation": "NOT AVAILABLE", "reason": "No such resource found; PCP provides ROI time series only"},
        {"option": "C", "source": "AAL-116 atlas + compatible BOLD (local)", "data_type": "Atlas + raw/preprocessed BOLD",
         "atlas": "AAL (not locally available)", "pipeline": "UNCERTAIN (local BOLD not confirmed CPAC)",
         "strategy": "UNCERTAIN", "subject_coverage": "54/780 (6.9%, single-site, per Phase 3.5)",
         "site_coverage": "1/20 sites", "same_780_cohort": "PARTIAL (54 only)", "paper_compatible": "NO (insufficient coverage)",
         "download_required": "NO for existing 54; YES for an atlas file if pursued", "estimated_effort": "HIGH (would still need full preprocessing pipeline built)",
         "methodology_deviation": "Would require independent AAL extraction, own preprocessing decisions",
         "recommendation": "NOT RECOMMENDED for full-cohort reproduction", "reason": "Confirmed insufficient in Phase 3.5"},
        {"option": "D", "source": "Full raw ABIDE-I download + preprocessing", "data_type": "Raw BOLD, all 780 subjects, all sites",
         "atlas": "Would apply AAL ourselves after preprocessing", "pipeline": "Self-built (CPAC-equivalent)",
         "strategy": "Self-chosen", "subject_coverage": "Up to 780 (post-download)", "site_coverage": "Up to 20",
         "same_780_cohort": "YES (if fully executed)", "paper_compatible": "POTENTIALLY, but high risk of preprocessing differences vs paper",
         "download_required": "YES (LARGE -- full multi-site raw dataset, hundreds of GB, explicitly excluded by the rules)",
         "estimated_effort": "VERY HIGH", "methodology_deviation": "High risk -- our own preprocessing choices may not match paper's CPAC/DPARSF exactly",
         "recommendation": "LAST RESORT ONLY", "reason": "Explicitly excluded by Phase 3.6 rules (no large downloads) unless separately approved"},
        {"option": "E", "source": "Existing CPAC + CC200", "data_type": "CC200 ROI time series",
         "atlas": "CC200 (NOT AAL)", "pipeline": "CPAC", "strategy": "filt_global",
         "subject_coverage": "780/780", "site_coverage": "20/20",
         "same_780_cohort": "YES", "paper_compatible": "NO -- different atlas entirely, changes feature count/identity from paper's AAL-116",
         "download_required": "NO (already available)", "estimated_effort": "NONE (ready now)",
         "methodology_deviation": "MAJOR -- different atlas is not a faithful paper reproduction",
         "recommendation": "COMPARISON-ONLY, NEVER AS THE PAPER REPRODUCTION", "reason": "Explicitly excluded from being the paper's atlas per project rules"},
    ]
    pd.DataFrame(comparison_rows).to_csv(OUT_DIR / "aal_source_comparison.csv", index=False)
    log("STEP 10: aal_source_comparison.csv written.")

    # ── STEP 11: decision ────────────────────────────────────────────────
    verified_sources_found = 1  # rois_aal via PCP/nilearn -- mechanism CONFIRMED, cohort overlap INFERRED not empirically verified
    roi_timeseries_status = "FOUND (mechanism/documentation CONFIRMED; exact 780-cohort overlap NOT YET EMPIRICALLY VERIFIED)"
    fc_status = "NOT FOUND"
    atlas_status = "NOT FOUND locally; PCP-internal atlas referenced but not independently downloadable/verified as a standalone file"
    same_cohort_status = "UNCERTAIN (INFERRED likely yes for CPAC, unconfirmed for DPARSF -- requires one small metadata fetch to confirm)"
    best_option = "Option A2 (ABIDE PCP rois_aal, DPARSF pipeline) pending verification; Option A (CPAC) as a faster-to-confirm fallback"
    same_cohort_compat = "UNCERTAIN"
    large_download_required = "NO (the identified AAL-116 ROI resource is a small, ~190-200MB estimated download, NOT hundreds of GB; a full raw-data route (Option D) WOULD be large and is not recommended)"
    option_c_decision = "PARTIAL"
    decision_reason = (
        "A legitimate, well-documented AAL-116 ROI time-series source (ABIDE PCP's rois_aal derivative, "
        "accessible through the already-installed nilearn.fetch_abide_pcp) was verified to exist for "
        "all four of the paper's pipelines, confirmed directly from the installed package's source code "
        "and corroborated by the official PCP GitHub repository. This is a materially better position "
        "than Phase 3.5's local-only finding. However, its exact overlap with our specific 780-subject "
        "Phase-2 cohort has NOT been empirically verified -- confirming it requires at minimum a small "
        "phenotypic-metadata fetch (and, for full confidence, the actual ~190-200MB rois_aal download), "
        "neither of which has been performed in this investigation, per the explicit rule to stop and "
        "report before any download."
    )
    recommended_next_step = (
        "Approve a SMALL, explicitly-scoped verification step (not yet performed): fetch only the "
        "DPARSF (and/or CPAC) phenotypic metadata via fetch_abide_pcp to confirm subject-ID overlap "
        "with the 780-subject Phase-2 cohort, before committing to the full ~190-200MB rois_aal ROI "
        "data download. This keeps the investigation-only posture of Phase 3.6 while resolving the "
        "one remaining unknown blocking a FEASIBLE determination."
    )

    log(f"STEP 11: Verified AAL-116 sources found: {verified_sources_found}")
    log(f"  AAL-116 ROI/time-series source: {roi_timeseries_status}")
    log(f"  AAL-116 FC source: {fc_status}")
    log(f"  AAL-116 atlas source: {atlas_status}")
    log(f"  Source covering the same 780 subjects: {same_cohort_status}")
    log(f"  Best verified option: {best_option}")
    log(f"  OPTION C decision: {option_c_decision}")
    log(f"  Reason: {decision_reason}")
    log(f"  Recommended next step: {recommended_next_step}")

    # ── audit ────────────────────────────────────────────────────────────
    post = {p: fp(p) for p in pre}
    unchanged = all(pre[p] == post[p] for p in pre)
    log(f"AUDIT: Phase 1/2/3/3.5 files unchanged: {'PASS' if unchanged else 'FAIL'}")
    log(f"AUDIT: 780-subject cohort remains authoritative, unmodified: "
        f"{'PASS' if (n_cohort == 780 and n_asd == 351 and n_control == 429) else 'FAIL'}")
    log(f"AUDIT: no duplicate subject IDs: {'PASS' if n_dup == 0 else 'FAIL'}")
    log("AUDIT: no CC200 substitution -- CC200 appears only in the comparison table (Option E) as an "
        "explicitly-labeled non-reproduction comparison point, never treated as AAL-116.")
    log("AUDIT: no FC/ROI/model/XAI/ROAR computation performed. No downloads performed -- confirmed by "
        "code review: this script contains no network calls, only WebSearch/WebFetch calls made "
        "earlier in this conversation turn for documentation lookup (not data download), and local "
        "source-code/file inspection.")

    decision = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase2_cohort": {"total": n_cohort, "asd": n_asd, "control": n_control, "duplicates": n_dup},
        "verified_sources_found": verified_sources_found,
        "aal_roi_timeseries_status": roi_timeseries_status,
        "aal_fc_status": fc_status,
        "aal_atlas_status": atlas_status,
        "same_780_cohort_coverage": same_cohort_status,
        "best_verified_option": best_option,
        "same_cohort_compatibility": same_cohort_compat,
        "large_download_required": large_download_required,
        "option_c_decision": option_c_decision,
        "decision_reason": decision_reason,
        "recommended_next_step": recommended_next_step,
        "download_disclosure_not_executed": download_disclosure,
        "audit": {
            "phase1_2_3_35_unchanged": unchanged,
            "cohort_authoritative": (n_cohort == 780 and n_asd == 351 and n_control == 429),
            "no_duplicate_subject_ids": n_dup == 0,
            "cc200_substituted": False, "fc_calculated": False, "roi_extraction_performed": False,
            "model_trained": False, "large_download_performed": False, "original_data_modified": False,
        },
    }
    with open(OUT_DIR / "phase3_6_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    log(f"phase3_6_decision.json written -> {OUT_DIR / 'phase3_6_decision.json'}")

    summary_lines = [
        "PHASE 3.6 SUMMARY -- AAL-116 Data Source Investigation", "=" * 70, "",
        f"Phase-2 cohort: {n_cohort} subjects ({n_asd} ASD, {n_control} Control)", "",
        f"Verified AAL-116 sources found: {verified_sources_found}",
        f"  ROI/time-series: {roi_timeseries_status}",
        f"  FC: {fc_status}", f"  Atlas: {atlas_status}", "",
        f"Same-780-cohort coverage: {same_cohort_status}",
        f"Best verified option: {best_option}", "",
        f"OPTION C: {option_c_decision}", f"Reason: {decision_reason}", "",
        f"Recommended next step: {recommended_next_step}", "",
        "DOWNLOAD DISCLOSURE (NOT EXECUTED):",
    ] + [f"  {k}: {v}" for k, v in download_disclosure.items()]
    with open(OUT_DIR / "phase3_6_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    log("phase3_6_summary.txt written.")

    audit_lines.append("Output files: aal_local_search.csv, abide_pcp_aal_investigation.csv, "
                        "nilearn_aal_investigation.csv, public_aal_atlas_sources.csv, "
                        "aal_source_cohort_compatibility.csv, aal_source_comparison.csv, "
                        "phase3_6_summary.txt, phase3_6_decision.json")
    with open(OUT_DIR / "phase3_6_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 3.6 outputs written to {OUT_DIR}")

    print()
    print("=" * 50)
    print("PHASE 3.6 STATUS: COMPLETE")
    print("=" * 50)
    print()
    print("Phase-2 cohort:")
    print(f"{n_cohort} subjects")
    print(f"{n_asd} ASD")
    print(f"{n_control} Controls")
    print()
    print("Verified AAL-116 sources found:")
    print(verified_sources_found)
    print()
    print("AAL-116 ROI/time-series source:")
    print("FOUND (mechanism confirmed; cohort overlap unconfirmed)")
    print()
    print("AAL-116 FC source:")
    print("NOT FOUND")
    print()
    print("AAL-116 atlas source:")
    print("NOT FOUND (locally); PCP-internal reference only, not independently downloadable/verified")
    print()
    print("Source covering the same 780 subjects:")
    print("UNCERTAIN")
    print()
    print("Best verified option:")
    print(best_option)
    print()
    print("Same-cohort compatibility:")
    print(same_cohort_compat)
    print()
    print("Large download required:")
    print("NO")
    print()
    print("Option C decision:")
    print(option_c_decision)
    print()
    print("Reason:")
    print(decision_reason)
    print()
    print("Recommended next step:")
    print(recommended_next_step)
    print()
    print("=" * 50)

    return decision


if __name__ == "__main__":
    main()
