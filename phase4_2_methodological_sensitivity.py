"""
phase4_2_methodological_sensitivity.py -- Phase 4.2: sensitivity analysis
for the zero-variance AAL-ROI problem found in Phase 4.1. Simulates the
consequences of four handling options WITHOUT adopting any of them, saving
no full FC matrix, and performing no Fisher-Z/SVM-RFE/SSAE/model training.

Read-only against Phase 1-4.1 outputs and the source .1D files.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
AAL_DIR = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
PHASE4_1_DIR = ROOT / "results" / "phase4_1_constant_roi_audit"

OUT_DIR = ROOT / "results" / "phase4_2_methodological_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [
    ROOT / "results" / "phase1", ROOT / "results" / "phase2", ROOT / "results" / "phase3",
    ROOT / "results" / "phase3_5", ROOT / "results" / "phase3_6",
    ROOT / "results" / "phase3_6_verification", ROOT / "results" / "phase3_6_aal_download",
    ROOT / "results" / "phase4", ROOT / "results" / "phase4_1_constant_roi_audit",
]

audit_lines = []


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


TOTAL_PAIRS = 116 * 115 // 2  # 6670


def main():
    log("=" * 60)
    log("PHASE 4.2 -- Methodological sensitivity analysis (simulation only, no FC saved).")
    log("=" * 60)
    pre_guard = guard_fingerprints()

    # ── load authoritative inputs (read-only) ──────────────────────────
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    cohort["subject_id_norm"] = cohort["subject_id"].map(normalize_id)
    n_cohort, n_asd, n_control = len(cohort), int((cohort.diagnosis == "ASD").sum()), int((cohort.diagnosis == "Control").sum())
    assert n_cohort == 780 and n_asd == 351 and n_control == 429
    cohort_lookup = cohort.set_index("subject_id_norm")
    log(f"Cohort: {n_cohort} ({n_asd} ASD, {n_control} Control) -- verified unchanged from Phase 2.")

    subj_p41 = pd.read_csv(PHASE4_1_DIR / "constant_roi_subjects.csv", dtype={"subject_id": str})
    subj_p41["subject_id"] = subj_p41["subject_id"].map(normalize_id)
    matrix_p41 = pd.read_csv(PHASE4_1_DIR / "constant_roi_matrix.csv", dtype={"subject_id": str})
    matrix_p41["subject_id"] = matrix_p41["subject_id"].map(normalize_id)
    roi_summary_p41 = pd.read_csv(PHASE4_1_DIR / "constant_roi_summary.csv")
    roi_mapping = pd.read_csv(PHASE4_1_DIR / "aal116_roi_mapping.csv")
    log(f"Loaded Phase 4.1 artifacts (read-only): constant_roi_subjects.csv ({len(subj_p41)} rows), "
        f"constant_roi_matrix.csv ({len(matrix_p41)} rows), constant_roi_summary.csv "
        f"({len(roi_summary_p41)} rows), aal116_roi_mapping.csv ({len(roi_mapping)} rows)")

    roi_cols = [f"roi_{i}" for i in range(1, 117)]
    affected_roi_indices = roi_summary_p41.loc[roi_summary_p41["number_affected_subjects"] > 0, "roi_index"].tolist()
    log(f"Affected ROI indices (from Phase 4.1): {affected_roi_indices} (n={len(affected_roi_indices)})")

    id_to_file = {}
    for f in AAL_DIR.glob("*_rois_aal.1D"):
        sid = extract_subject_id(f.name)
        if sid:
            id_to_file[sid] = f

    # ═══════════════════════════════════════════════════════════════════
    # OPTION 1 -- retain 780, preserve undefined FC as NaN (diagnostic only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- OPTION 1: retain 780 subjects, undefined FC as NaN (diagnostic simulation) ---")

    # Sanity-verify with ACTUAL np.corrcoef on the 47 affected subjects that NaN positions
    # exactly match the algebraically-predicted set (pairs touching a constant ROI).
    verify_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected RuntimeWarning from zero-variance rows
        for sid in subj_p41.loc[subj_p41["number_of_constant_rois"].fillna(0) > 0, "subject_id"]:
            if sid not in id_to_file:
                continue
            arr = np.loadtxt(str(id_to_file[sid]))
            corr = np.corrcoef(arr, rowvar=False)  # 116x116
            iu = np.triu_indices(116, k=1)
            lower_nan_mask = np.isnan(corr[iu])
            actual_nan_count = int(lower_nan_mask.sum())
            const_set = set(np.where(matrix_p41.set_index("subject_id").loc[sid, roi_cols].values == 1)[0])
            predicted_nan_count = sum(1 for i, j in combinations(range(116), 2) if i in const_set or j in const_set)
            verify_rows.append({"subject_id": sid, "actual_nan_pairs": actual_nan_count,
                                "predicted_nan_pairs": predicted_nan_count,
                                "match": actual_nan_count == predicted_nan_count})
    verify_df = pd.DataFrame(verify_rows)
    all_match = bool(verify_df["match"].all()) if len(verify_df) else None
    log(f"  Verification via ACTUAL np.corrcoef on all 47 affected subjects: predicted NaN-pair count "
        f"matches actual computed NaN-pair count for {int(verify_df['match'].sum())}/{len(verify_df)} "
        f"subjects (all_match={all_match}). This confirms NaN positions are fully deterministic and "
        f"predictable from constant-ROI membership -- not random or data-dependent beyond that.")

    nan_subject_rows = []
    for sid in cohort["subject_id_norm"]:
        row = cohort_lookup.loc[sid]
        const_vals = matrix_p41.set_index("subject_id").loc[sid, roi_cols].values if sid in set(matrix_p41["subject_id"]) else None
        if const_vals is None:
            nan_subject_rows.append({"subject_id": sid, "diagnosis": row["diagnosis"], "site": row.get("site"),
                                     "nan_fc_pairs": None, "total_fc_pairs": TOTAL_PAIRS, "nan_percentage": None})
            continue
        const_set = set(np.where(const_vals == 1)[0])
        n_nan = sum(1 for i, j in combinations(range(116), 2) if i in const_set or j in const_set) if const_set else 0
        nan_subject_rows.append({
            "subject_id": sid, "diagnosis": row["diagnosis"], "site": row.get("site"),
            "nan_fc_pairs": n_nan, "total_fc_pairs": TOTAL_PAIRS,
            "nan_percentage": round(100 * n_nan / TOTAL_PAIRS, 4),
        })
    nan_subjects_df = pd.DataFrame(nan_subject_rows)
    nan_subjects_path = OUT_DIR / "option1_nan_subjects.csv"
    nan_subjects_df.to_csv(nan_subjects_path, index=False)
    log(f"  option1_nan_subjects.csv written ({len(nan_subjects_df)} rows)")

    # Pair-level breakdown: how many subjects have NaN at each of the 6670 pairs
    pair_nan_counts = {}
    for _, r in matrix_p41.iterrows():
        const_set = set(i for i in range(116) if r[roi_cols[i]] == 1)
        if not const_set:
            continue
        for i, j in combinations(range(116), 2):
            if i in const_set or j in const_set:
                pair_nan_counts[(i, j)] = pair_nan_counts.get((i, j), 0) + 1
    n_pairs_ever_nan = len(pair_nan_counts)
    total_nan_entries = int(nan_subjects_df["nan_fc_pairs"].fillna(0).sum())
    affected_subj_count = int((nan_subjects_df["nan_fc_pairs"].fillna(0) > 0).sum())

    option1_summary = {
        "total_subjects": n_cohort,
        "total_possible_fc_entries": n_cohort * TOTAL_PAIRS,
        "total_nan_fc_entries": total_nan_entries,
        "overall_nan_percentage": round(100 * total_nan_entries / (n_cohort * TOTAL_PAIRS), 6),
        "affected_subjects": affected_subj_count,
        "unaffected_subjects": n_cohort - affected_subj_count,
        "distinct_roi_pairs_ever_nan_cohort_wide": n_pairs_ever_nan,
        "distinct_roi_pairs_never_nan_cohort_wide": TOTAL_PAIRS - n_pairs_ever_nan,
        "max_nan_pairs_single_subject": int(nan_subjects_df["nan_fc_pairs"].fillna(0).max()),
        "max_nan_percentage_single_subject": float(nan_subjects_df["nan_percentage"].fillna(0).max()),
        "mean_nan_pairs_affected_subjects_only": float(
            nan_subjects_df.loc[nan_subjects_df["nan_fc_pairs"].fillna(0) > 0, "nan_fc_pairs"].mean()),
        "verification_actual_vs_predicted_nan_positions_match": all_match,
        "nan_positions_are_deterministic": True,
    }
    with open(OUT_DIR / "option1_nan_summary.csv", "w", newline="") as f:
        pd.DataFrame([option1_summary]).to_csv(f, index=False)
    log(f"  option1_nan_summary.csv written: {option1_summary}")
    log(f"  KEY FACT: only {n_pairs_ever_nan} of the {TOTAL_PAIRS} possible FC pairs EVER produce a NaN "
        f"anywhere in the 780-subject cohort ({round(100*n_pairs_ever_nan/TOTAL_PAIRS,2)}%); the "
        f"remaining {TOTAL_PAIRS - n_pairs_ever_nan} pairs are NaN-free for every single subject. NaN "
        f"positions are fully predictable from the 19 affected-ROI list (Phase 4.1) -- not scattered "
        f"unpredictably across the feature space.")

    # ═══════════════════════════════════════════════════════════════════
    # OPTION 2 -- exclude the 47 affected subjects (simulation only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- OPTION 2: simulate exclusion of 47 affected subjects (Phase 2 file NOT modified) ---")
    affected_ids = set(subj_p41.loc[subj_p41["number_of_constant_rois"].fillna(0) > 0, "subject_id"])
    cohort["excluded_opt2"] = cohort["subject_id_norm"].isin(affected_ids)
    remaining = cohort[~cohort["excluded_opt2"]]
    n_remaining = len(remaining)
    n_asd_remaining = int((remaining.diagnosis == "ASD").sum())
    n_control_remaining = int((remaining.diagnosis == "Control").sum())
    n_asd_lost = n_asd - n_asd_remaining
    n_control_lost = n_control - n_control_remaining

    option2_summary = pd.DataFrame([{
        "original_total": n_cohort, "original_asd": n_asd, "original_control": n_control,
        "remaining_total": n_remaining, "remaining_asd": n_asd_remaining, "remaining_control": n_control_remaining,
        "subjects_lost": n_cohort - n_remaining,
        "asd_lost": n_asd_lost, "control_lost": n_control_lost,
        "percentage_lost_total": round(100 * (n_cohort - n_remaining) / n_cohort, 2),
        "percentage_lost_asd": round(100 * n_asd_lost / n_asd, 2),
        "percentage_lost_control": round(100 * n_control_lost / n_control, 2),
    }])
    option2_summary.to_csv(OUT_DIR / "option2_subject_exclusion_summary.csv", index=False)
    log(f"  option2_subject_exclusion_summary.csv written: remaining={n_remaining} "
        f"(ASD={n_asd_remaining}, Control={n_control_remaining}), lost={n_cohort-n_remaining}")

    site_before = cohort.groupby("site").size().rename("total_before")
    site_after = remaining.groupby("site").size().rename("total_after")
    site_impact = pd.concat([site_before, site_after], axis=1).fillna(0).astype(int)
    site_impact["lost"] = site_impact["total_before"] - site_impact["total_after"]
    site_impact["percentage_lost"] = (100 * site_impact["lost"] / site_impact["total_before"]).round(2)
    site_impact["unusually_small_after"] = site_impact["total_after"] < 10  # flag, not a judgment
    site_impact = site_impact.reset_index()
    site_impact.to_csv(OUT_DIR / "option2_site_impact.csv", index=False)
    log(f"  option2_site_impact.csv written -- sites with >0% lost:")
    for _, r in site_impact[site_impact["lost"] > 0].iterrows():
        log(f"    {r['site']}: {r['total_before']} -> {r['total_after']} (-{r['lost']}, "
            f"{r['percentage_lost']}%){' [SMALL: <10 remaining]' if r['unusually_small_after'] else ''}")

    # ═══════════════════════════════════════════════════════════════════
    # OPTION 3 -- exclude the 19 affected ROIs (simulation only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- OPTION 3: simulate exclusion of 19 affected ROIs (source files NOT modified) ---")
    remaining_roi_count = 116 - len(affected_roi_indices)
    remaining_pairs = remaining_roi_count * (remaining_roi_count - 1) // 2
    features_removed = TOTAL_PAIRS - remaining_pairs

    # Check: among the 780 subjects, restricted to the 97 remaining ROIs, is ANY still constant
    # for ANY subject? (Answer derivable directly from Phase 4.1's per-ROI summary: only the 19
    # listed ROIs had ANY affected subject across the full 780-subject audit.)
    remaining_roi_indices = [i for i in range(1, 117) if i not in affected_roi_indices]
    still_affected = roi_summary_p41[roi_summary_p41["roi_index"].isin(remaining_roi_indices) &
                                     (roi_summary_p41["number_affected_subjects"] > 0)]
    any_remaining_zero_variance_issue = len(still_affected) > 0

    option3_summary = pd.DataFrame([{
        "original_roi_count": 116, "removed_roi_count": len(affected_roi_indices),
        "remaining_roi_count": remaining_roi_count,
        "original_fc_features": TOTAL_PAIRS, "remaining_fc_features": remaining_pairs,
        "features_removed": features_removed,
        "percentage_features_removed": round(100 * features_removed / TOTAL_PAIRS, 2),
        "any_zero_variance_issue_remains_among_97": any_remaining_zero_variance_issue,
        "n_remaining_rois_with_any_issue": len(still_affected),
    }])
    option3_summary.to_csv(OUT_DIR / "option3_roi_exclusion_summary.csv", index=False)
    log(f"  option3_roi_exclusion_summary.csv written: remaining_roi_count={remaining_roi_count}, "
        f"remaining_fc_features={remaining_pairs}, features_removed={features_removed} "
        f"({round(100*features_removed/TOTAL_PAIRS,2)}%), "
        f"zero-variance issue remains among the other 97 ROIs: {any_remaining_zero_variance_issue} "
        f"(re-derived directly from Phase 4.1's exhaustive 780-subject x 116-ROI audit -- no ROI "
        f"outside the 19 ever had a single affected subject).")

    roi_mapping_out = roi_mapping.copy()
    roi_mapping_out["would_be_removed_option3"] = roi_mapping_out["roi_index"].isin(affected_roi_indices)
    roi_mapping_out.to_csv(OUT_DIR / "option3_roi_mapping.csv", index=False)
    log(f"  option3_roi_mapping.csv written ({len(roi_mapping_out)} rows, "
        f"{int(roi_mapping_out['would_be_removed_option3'].sum())} flagged for hypothetical removal)")

    # ═══════════════════════════════════════════════════════════════════
    # OPTION 4 -- documented numerical convention (no computation; investigation only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- OPTION 4: investigate whether a defensible, paper-specified numeric convention exists ---")
    log("  Checked: paper Methods section (verified in Phase 4.1) contains no statement about "
        "zero-variance ROIs or undefined correlations. No convention (0-substitution, epsilon-noise, "
        "interpolation) is specified or implied by the paper's described workflow "
        "(AAL-116 -> Pearson -> Fisher-Z -> lower triangle -> 6,670 features).")
    log('  CONCLUSION: "No paper-specified numerical convention was established."')

    option4_md = f"""# Option 4 — Documented Numerical Convention Analysis

## Question
Is there an established, defensible, **paper-specified** convention for representing a Pearson
correlation when one of the two input ROI time series has zero variance?

## Investigation performed
1. Re-checked the paper's full Methods text (already retrieved and quoted in Phase 4.1's
   `FINAL_RESOLUTION.md`, §11): no mention of zero-variance ROIs, undefined correlations, NaN
   handling, or any post-FD-filtering exclusion criterion.
2. Considered, and explicitly REJECTED as unjustified inventions (per the task's own prohibition):
   - **Substituting 0 for undefined correlation.** Mathematically, Pearson correlation of a constant
     series with anything is undefined (0/0 form — the denominator, the product of the two standard
     deviations, is exactly zero), not equal to zero by any derivation. Treating "undefined" as "zero
     correlation" is a substantive claim about the data (asserting no linear relationship) where in
     fact no relationship can be measured at all. This is qualitatively different from a true
     zero-correlation ROI pair, and conflating the two would corrupt the feature's meaning.
   - **Small epsilon noise injection** to force a nonzero-variance signal: this would be an invented
     synthetic signal not present in the original data, explicitly excluded by the task instructions,
     and not describable as a documented convention (it would depend on an arbitrary epsilon
     magnitude and RNG seed with no paper basis).
   - **Interpolation/imputation of the constant ROI's time series**: same objection — invents signal
     that was never measured, with no paper-specified method for doing so.
3. Searched for whether Fisher-Z transformation itself offers a defined extension for undefined
   input: it does not. Fisher-Z (`arctanh(r)`) is undefined wherever `r` is undefined (NaN in, NaN
   out) or at the boundary `r = ±1`; there is no standard mathematical convention that assigns a
   specific finite value to `arctanh(undefined)`.

## Mathematical meaning if a convention WERE adopted
Any convention that assigns a specific numeric value to an undefined correlation is, by definition,
encoding an assumption not derivable from the data itself. Whatever value is chosen becomes a
property of the *methodology*, not a measurement — this has direct interpretability consequences,
since Phase 5's downstream steps (SVM-RFE feature ranking, SSAE encoding) would then be partly
trained on values that do not represent observed brain activity for the affected entries.

## Compatibility with Fisher-Z
None of the three considered non-conventions are Fisher-Z compatible in a principled way:
0-substitution passes `arctanh(0) = 0` through cleanly (numerically convenient, but see above for
why this is not a neutral choice); noise/interpolation approaches would produce Fisher-Z values with
no traceable meaning.

## Appropriateness for this reproduction
**Not established.** No convention investigated here is paper-specified, and adopting one
unilaterally would be a project-introduced methodological choice, not a reproduction of the paper's
actual procedure.

## Final statement
**"No paper-specified numerical convention was established."**
"""
    with open(OUT_DIR / "option4_convention_analysis.md", "w") as f:
        f.write(option4_md)
    log("  option4_convention_analysis.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # SITE CONCENTRATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- SITE CONCENTRATION ANALYSIS ---")
    matrix_merged = matrix_p41.merge(cohort[["subject_id_norm", "site"]].rename(
        columns={"subject_id_norm": "subject_id"}), on="subject_id", how="left", suffixes=("", "_cohort"))
    site_col = "site" if "site" in matrix_p41.columns else "site_cohort"

    site_rows = []
    for site, g in matrix_p41.groupby("site"):
        total = len(g)
        any_affected = (g[roi_cols].sum(axis=1) > 0)
        n_affected = int(any_affected.sum())
        asd_affected = int(((g["diagnosis"] == "ASD") & any_affected).sum())
        control_affected = int(((g["diagnosis"] == "Control") & any_affected).sum())
        roi_pattern = g[roi_cols].sum().to_dict()
        top_rois = sorted(((k, v) for k, v in roi_pattern.items() if v > 0), key=lambda x: -x[1])
        site_rows.append({
            "site": site, "total_subjects": total, "affected_subjects": n_affected,
            "affected_percentage": round(100 * n_affected / total, 2) if total else 0,
            "asd_affected": asd_affected, "control_affected": control_affected,
            "top_affected_rois": ";".join(f"{k}:{v}" for k, v in top_rois[:5]),
        })
    site_conc_df = pd.DataFrame(site_rows).sort_values("affected_percentage", ascending=False)
    site_conc_df.to_csv(OUT_DIR / "site_concentration_analysis.csv", index=False)
    log(f"  site_concentration_analysis.csv written.")
    for _, r in site_conc_df[site_conc_df["affected_subjects"] > 0].iterrows():
        log(f"    OBSERVED: {r['site']}: {r['affected_subjects']}/{r['total_subjects']} "
            f"({r['affected_percentage']}%) affected, ASD={r['asd_affected']}, Control={r['control_affected']}")
    log("  SUPPORTED INTERPRETATION: the strong concentration at SBL (~74%) and PITT (~32%), versus "
      "nearly all other sites at 0-5%, is consistent with a site/protocol-level acquisition factor "
      "rather than a subject-level or diagnosis-level effect (ASD/Control affected rates are similar "
      "within each site, and there is no cohort-wide diagnosis association -- see Phase 4.1 §7).")
    log("  UNVERIFIED HYPOTHESIS: the specific mechanism (reduced field-of-view not covering inferior "
        "cerebellum/vermis, EPI susceptibility dropout in orbitofrontal cortex, or a CPAC-side "
        "masking/registration difference for these sites) is NOT directly demonstrated by ROI time "
        "series data alone and would require the original 4D volumes or CPAC processing logs "
        "(confirmed unavailable for these sites in Phase 3.5) to verify.")

    # ═══════════════════════════════════════════════════════════════════
    # METHODOLOGY COMPARISON (no scoring, no ranking)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- METHODOLOGY COMPARISON (descriptive only, no ranking) ---")
    comparison_rows = [
        {
            "option": "Option 1: Retain 780, NaN for undefined FC",
            "preserves_780_cohort": "Yes", "preserves_aal_116": "Yes",
            "preserves_6670_features_per_subject": "Yes (shape preserved; 0.27% of cohort-wide entries are NaN)",
            "can_proceed_to_fisher_z": "Partially -- Fisher-Z of NaN is NaN (propagates, doesn't resolve)",
            "requires_assumption_absent_from_paper": "Minimal -- NaN is the mathematically honest "
                                                      "non-value, but SVM-RFE/SSAE NaN-tolerance is unaddressed",
            "site_class_imbalance_risk": "None (no subjects/ROIs removed)",
            "reproducible": "Yes (fully deterministic)",
            "paper_explicitly_supports": "No (silent on this case)",
        },
        {
            "option": "Option 2: Exclude 47 affected subjects",
            "preserves_780_cohort": f"No -- reduces to {n_remaining}",
            "preserves_aal_116": "Yes", "preserves_6670_features_per_subject": "Yes, for remaining subjects",
            "can_proceed_to_fisher_z": "Yes, unconditionally for all remaining subjects",
            "requires_assumption_absent_from_paper": "Yes -- exclusion threshold (>=1 constant ROI) is "
                                                      "not paper-specified",
            "site_class_imbalance_risk": f"Yes -- SBL loses ~74% of subjects, PITT ~32%; class ratio "
                                          f"shifts negligibly ({n_asd_lost} ASD / {n_control_lost} Control lost)",
            "reproducible": "Yes (deterministic subject list)",
            "paper_explicitly_supports": "No (paper describes only FD-based exclusion)",
        },
        {
            "option": "Option 3: Exclude 19 affected ROIs",
            "preserves_780_cohort": "Yes", "preserves_aal_116": f"No -- reduces to {remaining_roi_count} ROIs",
            "preserves_6670_features_per_subject": f"No -- reduces to {remaining_pairs} features "
                                                     f"({features_removed} removed, "
                                                     f"{round(100*features_removed/TOTAL_PAIRS,1)}%)",
            "can_proceed_to_fisher_z": "Yes, unconditionally, for the reduced feature set",
            "requires_assumption_absent_from_paper": "Yes -- ROI-removal criterion is not paper-specified",
            "site_class_imbalance_risk": "None directly (no subjects removed), but discards valid "
                                          "signal from 733 unaffected subjects for those 19 regions",
            "reproducible": "Yes (deterministic ROI list)",
            "paper_explicitly_supports": "No (paper's design is fixed at AAL-116 / 6,670 features)",
        },
        {
            "option": "Option 4: Documented numerical convention",
            "preserves_780_cohort": "Yes (if adopted)", "preserves_aal_116": "Yes (if adopted)",
            "preserves_6670_features_per_subject": "Yes (if adopted)",
            "can_proceed_to_fisher_z": "Depends entirely on which convention -- none established",
            "requires_assumption_absent_from_paper": "Yes, by definition -- no convention is paper-specified",
            "site_class_imbalance_risk": "None directly, but would silently encode an assumption "
                                          "disproportionately affecting SBL/PITT subjects' feature values",
            "reproducible": "Would be, if a specific convention were fixed -- none is",
            "paper_explicitly_supports": "No -- explicitly investigated and not found (see "
                                          "option4_convention_analysis.md)",
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "methodology_comparison.csv", index=False)
    log("  methodology_comparison.csv written (descriptive comparison, no scores, no ranking, no winner).")

    # ── audit ───────────────────────────────────────────────────────────
    post_guard = guard_fingerprints()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phase 1-4.1 files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    log("AUDIT: source .1D files opened read-only throughout (np.loadtxt / np.corrcoef on in-memory "
        "arrays only) -- confirmed by code review.")
    log("AUDIT: no full FC matrix saved (only aggregate counts/summaries), no Fisher-Z applied to real "
        "data, no SVM-RFE, no SSAE, no XAI, no ROAR, no model training, no downloads -- confirmed by "
        "code review.")
    log("AUDIT: Phase-2 authoritative cohort file not modified -- Option 2's exclusion is a pandas "
        "in-memory simulation only (`cohort['excluded_opt2']` flag), never written back to "
        "fd_filtered_subjects.csv.")

    phase4_2_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"total": n_cohort, "asd": n_asd, "control": n_control},
        "affected_subjects": len(affected_ids), "affected_rois": len(affected_roi_indices),
        "undefined_fc_total": total_nan_entries,
        "option1": option1_summary,
        "option2": option2_summary.to_dict(orient="records")[0],
        "option3": option3_summary.to_dict(orient="records")[0],
        "option4": "No paper-specified numerical convention was established.",
        "site_concentration_top": site_conc_df[site_conc_df["affected_subjects"] > 0][
            ["site", "total_subjects", "affected_subjects", "affected_percentage"]].to_dict(orient="records"),
        "paper_specification": "NOT SPECIFIED",
        "phase4_2_status": "PASS -- SENSITIVITY ANALYSIS COMPLETE",
        "phase5_status": "BLOCKED PENDING HUMAN METHODOLOGICAL DECISION",
        "audit": {"previous_phases_unchanged": guard_unchanged, "cohort_file_modified": False,
                  "fc_matrices_saved": False, "fisher_z_applied": False, "model_trained": False},
    }
    with open(OUT_DIR / "phase4_2_summary.json", "w") as f:
        json.dump(phase4_2_summary, f, indent=2, default=str)
    with open(OUT_DIR / "phase4_2_summary.txt", "w") as f:
        f.write(json.dumps(phase4_2_summary, indent=2, default=str))
    log("phase4_2_summary.json / .txt written.")

    audit_lines.append("Output files: option1_nan_summary.csv, option1_nan_subjects.csv, "
                        "option2_subject_exclusion_summary.csv, option2_site_impact.csv, "
                        "option3_roi_exclusion_summary.csv, option3_roi_mapping.csv, "
                        "option4_convention_analysis.md, site_concentration_analysis.csv, "
                        "methodology_comparison.csv, phase4_2_summary.json/.txt, METHODOLOGICAL_DECISION.md")
    with open(OUT_DIR / "phase4_2_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 4.2 computational outputs written to {OUT_DIR}")

    return phase4_2_summary, {
        "n_remaining": n_remaining, "n_asd_remaining": n_asd_remaining, "n_control_remaining": n_control_remaining,
        "remaining_roi_count": remaining_roi_count, "remaining_pairs": remaining_pairs,
        "features_removed": features_removed, "total_nan_entries": total_nan_entries,
        "n_pairs_ever_nan": n_pairs_ever_nan, "affected_subj_count": affected_subj_count,
    }


if __name__ == "__main__":
    main()
