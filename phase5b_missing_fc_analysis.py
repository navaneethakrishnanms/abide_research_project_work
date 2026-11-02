"""
phase5b_missing_fc_analysis.py -- Phase 5B: methodological investigation of
how to handle the 14,025 undefined Pearson FC entries, comparing six
strategies (A-F) as diagnostic simulations only. No strategy is adopted, no
final FC dataset is written, no Fisher-Z on real data, no SVM-RFE/SSAE/
classifier training.

The full 780x6670 FC matrix (with NaN) is built once, IN MEMORY ONLY, for
this diagnostic run and is never persisted to disk in per-subject form --
only aggregate feature-level and strategy-comparison statistics are saved.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
AAL_DIR = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
PHASE4_1_DIR = ROOT / "results" / "phase4_1_constant_roi_audit"
PHASE5A_DIR = ROOT / "results" / "phase5a_fc_diagnostic"

OUT_DIR = ROOT / "results" / "phase5b_missing_fc_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [
    ROOT / "results" / "phase1", ROOT / "results" / "phase2", ROOT / "results" / "phase3",
    ROOT / "results" / "phase3_5", ROOT / "results" / "phase3_6",
    ROOT / "results" / "phase3_6_verification", ROOT / "results" / "phase3_6_aal_download",
    ROOT / "results" / "phase4", ROOT / "results" / "phase4_1_constant_roi_audit",
    ROOT / "results" / "phase4_2_methodological_sensitivity", ROOT / "results" / "phase5a_fc_diagnostic",
]

audit_lines = []
EXPECTED_UNDEFINED_TOTAL = 14025


def log(msg: str):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg, flush=True)


def fp_snapshot(root: Path) -> dict:
    out = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[str(p)] = (st.st_size, st.st_mtime)
    return out


def guard_fingerprints() -> dict:
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


def extract_subject_id(name: str) -> str | None:
    base = name.replace("_rois_aal.1D", "")
    digits = "".join(ch for ch in base.split("_")[-1] if ch.isdigit())
    return str(int(digits)) if digits else None


def main():
    log("=" * 60)
    log("PHASE 5B -- Missing-FC methodological investigation (simulation only).")
    log("=" * 60)
    pre_guard = guard_fingerprints()

    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    cohort["subject_id_norm"] = cohort["subject_id"].map(normalize_id)
    n_cohort, n_asd, n_control = len(cohort), int((cohort.diagnosis == "ASD").sum()), int((cohort.diagnosis == "Control").sum())
    assert n_cohort == 780 and n_asd == 351 and n_control == 429
    cohort_lookup = cohort.set_index("subject_id_norm")
    n_sites = cohort["site"].nunique()
    log(f"Cohort: {n_cohort} ({n_asd} ASD, {n_control} Control), {n_sites} sites.")

    roi_mapping = pd.read_csv(PHASE4_1_DIR / "aal116_roi_mapping.csv")
    roi_name = roi_mapping.set_index("roi_index")["roi_name"].to_dict()
    feature_index_df = pd.read_csv(PHASE5A_DIR / "fc_feature_index.csv")
    assert len(feature_index_df) == 6670

    id_to_file = {}
    for f in AAL_DIR.glob("*_rois_aal.1D"):
        sid = extract_subject_id(f.name)
        if sid:
            id_to_file[sid] = f
    subject_files = {sid: id_to_file[sid] for sid in cohort["subject_id_norm"] if sid in id_to_file}
    assert len(subject_files) == 780

    # ── build the 780x6670 FC matrix IN MEMORY ONLY (never persisted) ────
    log("\nBuilding the 780x6670 diagnostic FC matrix (in-memory only, never saved to disk)...")
    subj_order = list(cohort["subject_id_norm"])
    diagnosis_arr = cohort_lookup.loc[subj_order, "diagnosis"].values
    site_arr = cohort_lookup.loc[subj_order, "site"].values
    fc_matrix = np.full((780, 6670), np.nan, dtype=float)
    iu = np.triu_indices(116, k=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s_idx, sid in enumerate(subj_order):
            arr = np.loadtxt(str(subject_files[sid]))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            corr = np.corrcoef(arr, rowvar=False)
            fc_matrix[s_idx, :] = corr[iu]

    total_undefined = int(np.isnan(fc_matrix).sum())
    log(f"Total undefined entries in the in-memory diagnostic matrix: {total_undefined} "
        f"(expected {EXPECTED_UNDEFINED_TOTAL}) -> "
        f"{'MATCH' if total_undefined == EXPECTED_UNDEFINED_TOTAL else 'MISMATCH -- INVESTIGATE'}")
    assert total_undefined == EXPECTED_UNDEFINED_TOTAL

    missing_mask = np.isnan(fc_matrix)
    affected_features = int((missing_mask.sum(axis=0) > 0).sum())
    log(f"Affected features (>=1 subject missing): {affected_features}")

    # ═══════════════════════════════════════════════════════════════════
    # FEATURE-WISE MISSINGNESS ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Feature-wise missingness analysis ---")
    asd_mask = (diagnosis_arr == "ASD")
    control_mask = (diagnosis_arr == "Control")

    feat_rows = []
    for k in range(6670):
        col_missing = missing_mask[:, k]
        n_missing = int(col_missing.sum())
        rec = {
            "feature_index": int(feature_index_df.iloc[k]["feature_index"]),
            "roi_i": int(feature_index_df.iloc[k]["roi_i"]), "roi_j": int(feature_index_df.iloc[k]["roi_j"]),
            "roi_i_name": feature_index_df.iloc[k]["roi_i_name"], "roi_j_name": feature_index_df.iloc[k]["roi_j_name"],
            "missing_count": n_missing, "missing_percentage": round(100 * n_missing / 780, 4),
            "affected_sites": ";".join(sorted(set(site_arr[col_missing]))) if n_missing else "",
            "ASD_missing": int((col_missing & asd_mask).sum()),
            "control_missing": int((col_missing & control_mask).sum()),
        }
        feat_rows.append(rec)
    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(OUT_DIR / "feature_missingness_analysis.csv", index=False)
    log(f"feature_missingness_analysis.csv written ({len(feat_df)} rows, {affected_features} with missing_count>0).")

    dist = feat_df.loc[feat_df["missing_count"] > 0, "missing_count"].value_counts().sort_index()
    dist_df = pd.DataFrame({"missing_count": dist.index, "n_features": dist.values})
    dist_df.to_csv(OUT_DIR / "feature_missingness_distribution.csv", index=False)
    log(f"feature_missingness_distribution.csv written ({len(dist_df)} distinct missing-count values).")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY C -- feature-wise imputation investigation (diagnostic only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- STRATEGY C: feature-wise pooled imputation (DIAGNOSTIC SIMULATION ONLY) ---")
    imputation_rows = []
    candidate_diffs = []
    for k in range(6670):
        col = fc_matrix[:, k]
        col_missing = missing_mask[:, k]
        n_missing = int(col_missing.sum())
        if n_missing == 0:
            continue
        observed = col[~col_missing]
        observed_asd = col[~col_missing & asd_mask]
        observed_control = col[~col_missing & control_mask]
        pooled_mean = float(np.mean(observed)) if len(observed) else np.nan
        pooled_median = float(np.median(observed)) if len(observed) else np.nan
        asd_mean = float(np.mean(observed_asd)) if len(observed_asd) else np.nan
        control_mean = float(np.mean(observed_control)) if len(observed_control) else np.nan
        diff = abs(pooled_mean - pooled_median) if not (np.isnan(pooled_mean) or np.isnan(pooled_median)) else np.nan
        if not np.isnan(diff):
            candidate_diffs.append(diff)
        imputation_rows.append({
            "feature_index": int(feature_index_df.iloc[k]["feature_index"]),
            "missing_count": n_missing, "observed_count": len(observed),
            "pooled_mean_observed": pooled_mean, "pooled_median_observed": pooled_median,
            "asd_mean_observed": asd_mean, "control_mean_observed": control_mean,
            "mean_vs_median_abs_diff": diff,
        })
    imputation_df = pd.DataFrame(imputation_rows)
    imputation_summary = {
        "n_features_with_missing": len(imputation_df),
        "n_missing_entries_total": int(imputation_df["missing_count"].sum()),
        "mean_abs_diff_mean_vs_median": float(np.nanmean(candidate_diffs)) if candidate_diffs else None,
        "max_abs_diff_mean_vs_median": float(np.nanmax(candidate_diffs)) if candidate_diffs else None,
        "candidate_pooled_mean_range": [float(np.nanmin(imputation_df["pooled_mean_observed"])),
                                        float(np.nanmax(imputation_df["pooled_mean_observed"]))],
    }
    log(f"Imputation candidate-value comparison (mean vs median, computed from OBSERVED values only, "
        f"pooled across the full 780 -- diagnostic only, NOT applied to any dataset): {imputation_summary}")
    log("IMPORTANT: diagnosis-specific (ASD-mean / Control-mean) values are computed here ONLY to test "
        "whether missingness or feature values differ by diagnosis (see bias check below) -- they are "
        "NOT used to impute anything, and no imputed matrix is used for any downstream ML step.")

    # ── missingness-diagnosis / missingness-site association (descriptive only) ──
    log("\n--- Bias check: is missingness associated with diagnosis or site? (descriptive only) ---")
    per_subject_missing = missing_mask.sum(axis=1)
    asd_missing_rate = float(per_subject_missing[asd_mask].mean())
    control_missing_rate = float(per_subject_missing[control_mask].mean())
    log(f"  Mean undefined-FC-entries per subject: ASD={asd_missing_rate:.3f}  Control={control_missing_rate:.3f} "
        f"(descriptive; no causal claim; consistent with Phase 4.1's finding of no diagnosis association)")

    site_bias_rows = []
    for site in sorted(set(site_arr)):
        site_mask = (site_arr == site)
        n_site = int(site_mask.sum())
        n_site_any_missing = int((per_subject_missing[site_mask] > 0).sum())
        site_bias_rows.append({
            "site": site, "n_subjects": n_site, "n_any_missing": n_site_any_missing,
            "pct_any_missing": round(100 * n_site_any_missing / n_site, 2) if n_site else 0,
            "expected_pct_if_uniform": round(100 * int((per_subject_missing > 0).sum()) / 780, 2),
        })
    site_bias_df = pd.DataFrame(site_bias_rows).sort_values("pct_any_missing", ascending=False)
    log("  Site-level missingness vs. cohort-wide expected rate (descriptive contingency, no causal test):")
    for _, r in site_bias_df[site_bias_df["n_any_missing"] > 0].iterrows():
        log(f"    {r['site']}: observed {r['pct_any_missing']}% vs expected {r['expected_pct_if_uniform']}% "
            f"if uniformly distributed -- {'ABOVE' if r['pct_any_missing'] > r['expected_pct_if_uniform'] else 'AT/BELOW'} expectation")
    log("  This reproduces and quantifies (does not newly discover) the SBL/PITT concentration already "
        "established in Phase 4.1/4.2. Feature-wise pooled imputation, if adopted, would silently "
        "smooth over this site-specific pattern rather than represent it -- flagged as a real "
        "consequence of Strategy C, not a reason to avoid reporting it.")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY A -- subject exclusion (re-derived, sensitivity only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- STRATEGY A: subject exclusion (sensitivity only; Phase-2 cohort NOT modified) ---")
    subj_p41 = pd.read_csv(PHASE4_1_DIR / "constant_roi_subjects.csv", dtype={"subject_id": str})
    subj_p41["subject_id"] = subj_p41["subject_id"].map(normalize_id)
    affected_ids = set(subj_p41.loc[subj_p41["number_of_constant_rois"].fillna(0) > 0, "subject_id"])
    remaining_mask = ~cohort["subject_id_norm"].isin(affected_ids)
    n_remaining = int(remaining_mask.sum())
    n_asd_remaining = int((cohort.loc[remaining_mask, "diagnosis"] == "ASD").sum())
    n_control_remaining = int((cohort.loc[remaining_mask, "diagnosis"] == "Control").sum())
    log(f"  Remaining: {n_remaining} ({n_asd_remaining} ASD, {n_control_remaining} Control). "
        f"For these subjects, the FC matrix is fully finite (0 NaN by construction).")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY B -- ROI exclusion (re-derived, sensitivity only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- STRATEGY B: ROI exclusion (sensitivity only; source AAL data NOT modified) ---")
    roi_summary_p41 = pd.read_csv(PHASE4_1_DIR / "constant_roi_summary.csv")
    affected_roi_indices = roi_summary_p41.loc[roi_summary_p41["number_affected_subjects"] > 0, "roi_index"].tolist()
    remaining_roi_count = 116 - len(affected_roi_indices)
    remaining_features = remaining_roi_count * (remaining_roi_count - 1) // 2
    keep_roi_zero_idx = [i for i in range(116) if (i + 1) not in affected_roi_indices]
    keep_feature_mask = np.array([
        (feature_index_df.iloc[k]["roi_i"] not in affected_roi_indices and
         feature_index_df.iloc[k]["roi_j"] not in affected_roi_indices)
        for k in range(6670)
    ])
    reduced_matrix_nan_count = int(np.isnan(fc_matrix[:, keep_feature_mask]).sum())
    log(f"  Remaining ROIs: {remaining_roi_count}, remaining features: {remaining_features} "
        f"(mask count check: {int(keep_feature_mask.sum())}). NaN count in the reduced matrix: "
        f"{reduced_matrix_nan_count} (0 expected -- {'CONFIRMED all finite' if reduced_matrix_nan_count == 0 else 'MISMATCH'}).")
    assert remaining_features == int(keep_feature_mask.sum())

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY D -- ROI time-series imputation (documentation only, no data modified)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- STRATEGY D: ROI time-series imputation (documentation only -- NOT performed) ---")
    log("  A constant ROI time series (std=0) carries no temporal information to interpolate from -- "
        "there is no partial signal within that ROI's own series to extrapolate; any replacement "
        "value would have to come from OUTSIDE that ROI (e.g. neighboring voxels, other subjects, or "
        "a template), which is not a property of THIS subject's own measured data. Interpolation "
        "(temporal) is meaningless here because the signal is uniformly flat for the ENTIRE scan, not "
        "missing at isolated timepoints -- there are no non-missing neighbors within the series to "
        "interpolate between. Any synthetic value (epsilon noise or otherwise) would inject a "
        "correlation structure with the other 115 ROIs that was never measured, changing Pearson FC "
        "from 'a measurement' to 'a measurement mixed with an invented assumption' for every pair "
        "touching that ROI -- this is not faithful to Pearson-correlation-based FC as the paper "
        "describes it, and is not recommended as a course of action (not evaluated further as an "
        "actionable option here).")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY E -- pairwise / alternative correlation (documentation only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- STRATEGY E: pairwise correlation / alternative correlation (documentation only) ---")
    log("  Pairwise deletion (using only overlapping non-missing timepoints between two series) solves "
        "a DIFFERENT problem -- missing timepoints WITHIN otherwise-varying series. It does NOT solve "
        "a constant-vector problem: a series with std=0 is std=0 over any subset of its timepoints too "
        "(a constant series restricted to a subset is still constant), so Pearson r remains undefined "
        "regardless of which timepoints are included. Alternative correlation measures (e.g. Spearman, "
        "Kendall) are also undefined for a constant input by the same mathematical reason (rank-based "
        "measures require variation to rank), and in any case would depart from the paper's explicitly "
        "stated Pearson-correlation methodology -- not evaluated further as faithful to the paper.")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY F -- constant-ROI masking at feature level (re-derived from 5A)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- STRATEGY F: treat undefined features as structurally missing (re-derived) ---")
    log(f"  Unique affected FC feature positions: {affected_features} (of 6670, {round(100*affected_features/6670,2)}%). "
        f"Per-subject affected counts range 0-1314 (Phase 5A). Affected-feature-set determinism: "
        f"deterministic and fully reproducible (fixed function of the 19 constant-prone ROIs, verified "
        f"in Phase 5A/4.1/4.2). Removing only universally-or-near-universally-valid features (i.e. the "
        f"complement of Strategy B's 19-ROI removal) is mathematically equivalent to Strategy B when the "
        f"threshold is set at 'any missingness'; a laxer threshold (e.g. remove only features missing "
        f"in >=5% of subjects) would remove just the 2 worst features identified in Phase 5A, leaving "
        f"6668 features and still 14,025 - (43+39) = 13,943 undefined entries unresolved -- i.e. a "
        f"partial threshold does NOT eliminate the problem, only the full 19-ROI removal (Strategy B) does.")

    # ═══════════════════════════════════════════════════════════════════
    # LEAKAGE-SAFE IMPLEMENTATION (documentation, if Strategy C were ever adopted)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Leakage-safe implementation, IF feature-wise imputation were ever adopted for an ML pipeline ---")
    log("  Required procedure (documented, NOT implemented/trained here): within each outer-CV training "
        "fold, compute each feature's imputation statistic (e.g. mean of OBSERVED, non-missing values) "
        "using ONLY that fold's training subjects; apply that same training-fold statistic to fill any "
        "missing entries in both the training fold AND the corresponding validation/test fold. At NO "
        "point may a validation/test subject's own value, or any statistic computed using validation/"
        "test subjects (including their diagnosis labels), be used to compute the imputation statistic "
        "-- this exactly mirrors the fold-scoped preprocessing discipline already enforced by this "
        "project's `holdout_guard.py` infrastructure in the earlier ML phases. This is a documentation "
        "of HOW leakage would be avoided if Strategy C were selected -- it is not a recommendation to "
        "select Strategy C, and no such fold-scoped imputation was performed in this script.")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY COMPARISON TABLE (no ranking, no scores)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Strategy comparison (descriptive only, no ranking) ---")
    comparison_rows = [
        {"strategy": "A: Subject exclusion", "subjects_retained": n_remaining, "rois_retained": 116,
         "features_retained": 6670, "finite_matrix_possible": "YES (for remaining subjects)",
         "changes_paper_representation": "Subject count only", "introduces_imputation": "No",
         "potential_leakage": "None (deterministic exclusion list, no label use)",
         "site_impact": "Severe for SBL (-74%), PITT (-32%); see Phase 4.2",
         "diagnosis_impact": f"-{n_asd - n_asd_remaining} ASD, -{n_control - n_control_remaining} Control (negligible ratio shift)",
         "methodological_assumption": "Exclusion threshold (>=1 constant ROI) is a project choice",
         "paper_specified": "Paper does not specify this handling.",
         "notes": "Fully finite matrix for remaining subjects; no invented values."},
        {"strategy": "B: ROI exclusion", "subjects_retained": 780, "rois_retained": remaining_roi_count,
         "features_retained": remaining_features, "finite_matrix_possible": "YES (confirmed, 0 NaN remaining)",
         "changes_paper_representation": "ROI count and feature count (6670->4656)",
         "introduces_imputation": "No",
         "potential_leakage": "None (deterministic ROI list, no label use)",
         "site_impact": "None directly (no subjects removed)",
         "diagnosis_impact": "None directly (no subjects removed)",
         "methodological_assumption": "ROI-removal criterion is a project choice",
         "paper_specified": "Paper does not specify this handling.",
         "notes": "Discards valid signal from 733 unaffected subjects for these 19 regions."},
        {"strategy": "C: Feature-wise imputation", "subjects_retained": 780, "rois_retained": 116,
         "features_retained": 6670, "finite_matrix_possible": "CONDITIONAL (only if a specific "
                                                                "statistic and leakage-safe fold scoping is defined)",
         "changes_paper_representation": "No (shape preserved)", "introduces_imputation": "Yes",
         "potential_leakage": "HIGH RISK if computed pooled/globally (uses information from all "
                              "subjects, including future test subjects); LOW RISK only if computed "
                              "strictly within each training fold (see leakage-safe procedure above)",
         "site_impact": "Would silently smooth over the SBL/PITT concentration rather than represent it",
         "diagnosis_impact": "None if imputation is diagnosis-blind (uses pooled, not ASD/Control-"
                             "specific statistics) -- but computing/reporting ASD vs Control means, as "
                             "done diagnostically in this phase, must never feed the actual imputed value",
         "methodological_assumption": "That a feature's observed-subject mean/median is a reasonable "
                                       "stand-in for an undefined value -- statistically convenient, "
                                       "not paper-faithful (see PHASE5B_DECISION.md)",
         "paper_specified": "Paper does not specify this handling.",
         "notes": "Diagnostic-only simulation performed this phase; NOT used for any dataset."},
        {"strategy": "D: ROI time-series imputation", "subjects_retained": 780, "rois_retained": 116,
         "features_retained": 6670, "finite_matrix_possible": "NOT EVALUATED (not recommended as actionable)",
         "changes_paper_representation": "Would not change shape, but would inject invented signal",
         "introduces_imputation": "Yes, at the raw-signal level (more invasive than feature-level)",
         "potential_leakage": "Not applicable (not evaluated further)",
         "site_impact": "Not evaluated", "diagnosis_impact": "Not evaluated",
         "methodological_assumption": "That a plausible time series can be invented for a signal that "
                                       "was never measured -- not supportable (see documentation above)",
         "paper_specified": "Paper does not specify this handling.",
         "notes": "Documented and explicitly not recommended; not actionable."},
        {"strategy": "E: Pairwise/alternative correlation", "subjects_retained": 780, "rois_retained": 116,
         "features_retained": 6670, "finite_matrix_possible": "NO -- does not resolve a constant-vector problem",
         "changes_paper_representation": "Would depart from Pearson correlation (if a different measure used)",
         "introduces_imputation": "No", "potential_leakage": "Not applicable",
         "site_impact": "Not applicable", "diagnosis_impact": "Not applicable",
         "methodological_assumption": "None -- ruled out mathematically, not a viable candidate",
         "paper_specified": "Paper specifies Pearson correlation explicitly; alternatives would not be a reproduction.",
         "notes": "Mathematically cannot solve the problem; documented for completeness."},
        {"strategy": "F: Structural feature masking", "subjects_retained": 780, "rois_retained": 116,
         "features_retained": "6670 minus however many are masked (partial thresholds do not eliminate "
                              "the problem; full elimination requires the same 19-ROI removal as Strategy B)",
         "finite_matrix_possible": "CONDITIONAL (only equivalent to Strategy B at full threshold)",
         "changes_paper_representation": "Feature count, if any masking threshold is applied",
         "introduces_imputation": "No", "potential_leakage": "None (deterministic feature list)",
         "site_impact": "None directly", "diagnosis_impact": "None directly",
         "methodological_assumption": "Masking threshold choice is a project decision",
         "paper_specified": "Paper does not specify this handling.",
         "notes": "Effectively converges to Strategy B at the threshold needed to fully eliminate NaN."},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "strategy_comparison.csv", index=False)
    log("strategy_comparison.csv written (6 strategies, descriptive only, no scores, no ranking).")

    # ── audit ───────────────────────────────────────────────────────────
    post_guard = guard_fingerprints()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phase 1-5A files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    log("AUDIT: the 780x6670 diagnostic FC matrix built this phase existed only as an in-memory numpy "
        "array (`fc_matrix`) -- confirmed by code review: no np.save/to_csv/to_parquet of the full "
        "matrix anywhere in this script; only aggregate per-feature/per-strategy statistics were "
        "written to disk.")
    log("AUDIT: no Fisher-Z applied to real data, no SVM-RFE, no SSAE, no XAI, no ROAR, no classifier "
        "training, no downloads, no modification of Phase 1-5A files, AAL source files, or the "
        "authoritative Phase-2 cohort -- confirmed by code review.")

    status_ok = guard_unchanged and (total_undefined == EXPECTED_UNDEFINED_TOTAL)
    phase5b_status = "PASS" if status_ok else "FAIL"

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"total": n_cohort, "asd": n_asd, "control": n_control},
        "undefined_fc_total": total_undefined, "affected_features": affected_features,
        "strategies_evaluated": ["A", "B", "C", "D", "E", "F"],
        "finite_matrix_possible_780x6670_without_modification": "NO",
        "finite_matrix_possible_conditionally": "YES -- via Strategy A (fewer subjects), Strategy B "
                                                "(fewer ROIs/features), or Strategy C (with a leakage-safe, "
                                                "explicitly-documented imputation procedure, not yet adopted)",
        "paper_specified_handling": "NO",
        "leakage_safe_implementation_identified": "YES, IF Strategy C is eventually selected (documented "
                                                   "fold-scoped procedure above) -- CONDITIONAL on that selection",
        "imputation_diagnostic": imputation_summary,
        "phase5b_status": phase5b_status,
        "human_decision_required": True,
        "phase5_status": "BLOCKED",
        "audit": {"previous_phases_unchanged": guard_unchanged, "fc_dataset_written": False,
                  "fisher_z_on_real_data": False, "svm_rfe_trained": False, "ssae_trained": False,
                  "classifier_trained": False, "downloads_performed": False,
                  "cohort_modified": False, "aal_files_modified": False},
    }
    with open(OUT_DIR / "phase5b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT_DIR / "phase5b_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2, default=str))
    log("phase5b_summary.json / .txt written.")

    audit_lines.append("Output files: feature_missingness_analysis.csv, feature_missingness_distribution.csv, "
                        "strategy_comparison.csv, phase5b_summary.json/.txt, PHASE5B_DECISION.md")
    with open(OUT_DIR / "phase5b_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 5B outputs written to {OUT_DIR}")

    return summary, {
        "total_undefined": total_undefined, "affected_features": affected_features,
        "n_remaining": n_remaining, "n_asd_remaining": n_asd_remaining, "n_control_remaining": n_control_remaining,
        "remaining_roi_count": remaining_roi_count, "remaining_features": remaining_features,
        "imputation_summary": imputation_summary, "guard_unchanged": guard_unchanged,
        "site_bias_df": site_bias_df,
    }


if __name__ == "__main__":
    main()
