"""
phase7_synthesis.py -- Phase 7: read-only synthesis of Phase 6's real ABIDE
results (feature stability, XAI cross-method agreement, ROAR, ROI/brain-
region mapping). No retraining, no new model computation, no leakage --
this script only aggregates and cross-references already-saved Phase 6
output files.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
P6 = ROOT / "results" / "phase6_strategyC_all_nonzero"
OUT = ROOT / "results" / "phase7_synthesis"
for sub in ["feature_stability", "xai", "roar", "roi_analysis", "multi_evidence", "visualizations", "reports", "logs"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [ROOT / "results" / d for d in [
    "phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification",
    "phase3_6_aal_download", "phase4", "phase4_1_constant_roi_audit",
    "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic", "phase5b_missing_fc_analysis",
    "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero", "phase6_strategyC_all_nonzero",
]]

audit_lines = []
XAI_METHODS = ["IntegratedGradients", "GuidedBackprop", "DeepLift", "DeepLiftShap",
               "GradientShap", "SHAP", "LIME"]
N_FOLDS = 5


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg, flush=True)


def fp_snapshot(root):
    out = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[str(p)] = (st.st_size, st.st_mtime)
    return out


def guard_fingerprints():
    fps = {}
    for d in GUARD_DIRS:
        fps.update(fp_snapshot(d))
    return fps


def main():
    log("=" * 70)
    log("PHASE 7 -- Synthesis of Phase 6 real ABIDE results (read-only, no retraining).")
    log("=" * 70)
    pre_guard = guard_fingerprints()
    problems = []

    # ═══════════════════════════════════════════════════════════════════
    # 2. FREEZE-CHECK PHASE 6
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Freeze-check: verifying Phase 6 fundamental values before any synthesis ---")
    fold_counts = pd.read_csv(P6 / "elastic_net" / "fold_feature_counts.csv")
    N_vals = fold_counts.sort_values("fold")["N"].tolist()
    expected_N = [498, 479, 427, 463, 454]
    stability_summary = pd.read_csv(P6 / "feature_stability" / "stability_summary.csv")
    unique_selected = int(stability_summary.loc[stability_summary["n_folds_selected"] == "total_unique_selected_at_least_once", "n_features"].iloc[0])
    selected_every_fold = int(stability_summary.loc[stability_summary["n_folds_selected"] == "total_selected_in_every_completed_fold", "n_features"].iloc[0])

    checks = {
        "fold_N_matches": N_vals == expected_N,
        "min_427": min(N_vals) == 427, "max_498": max(N_vals) == 498,
        "mean_464_2": abs(np.mean(N_vals) - 464.2) < 0.1,
        "median_463": np.median(N_vals) == 463,
        # Phase 6's own PHASE6_REPORT.md reports std=23.9 (population std of [498,479,427,463,454]);
        # the user's spec approximated this as "std≈24.1" from memory. Validated against Phase 6's
        # own authoritative computed value, not the user's recollection.
        "std_matches_phase6_report_23_9": abs(np.std(N_vals) - 23.9) < 0.1,
        "unique_selected_1286": unique_selected == 1286,
        "selected_every_fold_50": selected_every_fold == 50,
    }
    for k, v in checks.items():
        log(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        problems.append(f"Phase 6 freeze-check FAILED: {checks}")
        log(f"[STOP] Freeze-check failed: {checks}. Investigating, NOT overwriting Phase 6.")
        _write_blocked("Phase 6 freeze-check failed -- values do not match the expected Phase 6 report.", problems)
        return
    log("Freeze-check: ALL PASS. Phase 6 values verified consistent with the expected report before proceeding.")

    cohort_check = pd.read_csv(P6 / "data_checks" / "data_integrity_summary.csv")
    log(f"Cohort re-confirmed from Phase 6 data_checks: {cohort_check.iloc[0].to_dict()}")

    # ═══════════════════════════════════════════════════════════════════
    # LOAD ALL PHASE 6 SOURCES (read-only)
    # ═══════════════════════════════════════════════════════════════════
    fold_feature_sel = pd.read_csv(P6 / "elastic_net" / "fold_feature_selection.csv")
    freq_df = pd.read_csv(P6 / "elastic_net" / "feature_selection_frequency.csv")
    stability_df = pd.read_csv(P6 / "feature_stability" / "feature_stability.csv")
    xai_df = pd.read_csv(P6 / "xai" / "xai_feature_attributions.csv")
    roar_df = pd.read_csv(P6 / "roar" / "roar_results.csv")
    roi_summary_p6 = pd.read_csv(P6 / "roi_analysis" / "roi_selection_summary.csv")
    log(f"\nLoaded (read-only): fold_feature_selection ({len(fold_feature_sel)} rows), "
        f"feature_selection_frequency ({len(freq_df)}), feature_stability ({len(stability_df)}), "
        f"xai_feature_attributions ({len(xai_df)}), roar_results ({len(roar_df)}), "
        f"roi_selection_summary ({len(roi_summary_p6)})")

    # ═══════════════════════════════════════════════════════════════════
    # 3. FEATURE STABILITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Feature stability analysis ---")
    coef_stats = fold_feature_sel.groupby("feature_index").agg(
        mean_coefficient=("elastic_net_coefficient", "mean"),
        mean_abs_coefficient=("absolute_coefficient", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
    ).reset_index()

    def sign_consistency(group):
        selected = group[group["selected"]]
        if len(selected) == 0:
            return np.nan
        signs = np.sign(selected["elastic_net_coefficient"])
        if len(signs) == 1:
            return 1.0
        return float((signs == signs.iloc[0]).mean())

    sign_df = fold_feature_sel.groupby("feature_index").apply(sign_consistency, include_groups=False).reset_index(name="coefficient_sign_consistency")

    fold_pivot = stability_df[["feature_index"] + [f"fold_{i}_selected" for i in range(N_FOLDS)]]
    stability_full = stability_df.merge(coef_stats, on="feature_index").merge(sign_df, on="feature_index")
    stability_full.to_csv(OUT / "feature_stability" / "feature_stability_summary.csv", index=False)

    cat_counts = stability_df["n_folds_selected"].value_counts().sort_index()
    log(f"  Selection-frequency categories (verified against Phase 6): "
        f"{ {int(k): int(v) for k, v in cat_counts.items()} }")
    expected_cats = {0: 5384, 1: 731, 2: 268, 3: 144, 4: 93, 5: 50}
    cats_match = all(int(cat_counts.get(k, 0)) == v for k, v in expected_cats.items())
    log(f"  Categories match expected Phase 6 values exactly: {cats_match}")
    if not cats_match:
        problems.append(f"Category counts do not match expected: {dict(cat_counts)} vs {expected_cats}")

    stable_features = stability_full[stability_full["n_folds_selected"] >= 4].sort_values(
        ["n_folds_selected", "mean_abs_coefficient"], ascending=[False, False])
    stable_features.to_csv(OUT / "feature_stability" / "stable_features.csv", index=False)
    log(f"  'Stable' defined explicitly as selected in >=4/5 folds (a descriptive engineering "
        f"threshold, not a biological one): {len(stable_features)} features "
        f"({int((stability_df['n_folds_selected']==5).sum())} at 5/5, "
        f"{int((stability_df['n_folds_selected']==4).sum())} at 4/5).")

    fold_stability_rows = []
    for f in range(N_FOLDS):
        col = f"fold_{f}_selected"
        fold_stability_rows.append({"fold": f, "n_selected": int(stability_df[col].sum()),
                                    "matches_reported_N": int(stability_df[col].sum()) == expected_N[f]})
    fold_stability_analysis = pd.DataFrame(fold_stability_rows)
    fold_stability_analysis.to_csv(OUT / "feature_stability" / "fold_stability_analysis.csv", index=False)
    log(f"  fold_stability_analysis.csv written -- per-fold N re-verified from the raw matrix: "
        f"{fold_stability_analysis['matches_reported_N'].all()}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. XAI SYNTHESIS (per method, separately first)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- XAI synthesis: per-method summary (methods analyzed SEPARATELY, not averaged) ---")
    method_rows = []
    for m in XAI_METHODS:
        sub = xai_df[xai_df["method"] == m]
        method_rows.append({
            "method": m, "n_attribution_records": len(sub),
            "unique_features_represented": sub["feature_index"].nunique(),
            "mean_abs_attribution": float(sub["mean_abs_attribution"].mean()),
            "median_abs_attribution": float(sub["mean_abs_attribution"].median()),
            "max_abs_attribution": float(sub["mean_abs_attribution"].max()),
            "note": "mean_abs_attribution here is Phase 6's already-aggregated per-(fold,feature) "
                    "mean |attribution| across that fold's test subjects; sign/direction was not "
                    "preserved by Phase 6 for these XAI outputs (only mean_abs_attribution was saved), "
                    "so signed-attribution/direction-consistency analysis is NOT POSSIBLE from the "
                    "available files and is not fabricated here.",
        })
    xai_method_summary = pd.DataFrame(method_rows)
    xai_method_summary.to_csv(OUT / "xai" / "xai_method_summary.csv", index=False)
    log(f"  xai_method_summary.csv written. LIMITATION noted explicitly: Phase 6 saved only "
        f"mean_abs_attribution per (fold, method, feature) -- not signed attribution -- so this phase "
        f"cannot report mean signed attribution or direction consistency; this is documented as a "
        f"limitation, not silently omitted.")

    # ── top-1/5/10% flags, computed WITHIN each (fold, method)'s own selected feature set ──
    log("\n--- XAI top-K%% flags (computed within each fold's own selected-feature set per method) ---")
    top_flags = []
    for (fold, method), g in xai_df.groupby(["fold", "method"]):
        g = g.sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)
        n = len(g)
        for pct, label in [(0.01, "top1pct"), (0.05, "top5pct"), (0.10, "top10pct")]:
            k = max(1, int(np.ceil(n * pct)))
            g[label] = g.index < k
        top_flags.append(g)
    xai_ranked = pd.concat(top_flags, ignore_index=True)

    # per-feature synthesis: averaged across folds where the feature was selected, per method
    synth_rows = []
    for feat_idx, g in xai_ranked.groupby("feature_index"):
        row = {"feature_index": feat_idx, "roi_i": g["roi_i"].iloc[0], "roi_j": g["roi_j"].iloc[0],
               "roi_i_name": g["roi_i_name"].iloc[0], "roi_j_name": g["roi_j_name"].iloc[0]}
        support_count = 0
        for m in XAI_METHODS:
            gm = g[g["method"] == m]
            row[f"{m}_mean_abs_attribution"] = float(gm["mean_abs_attribution"].mean()) if len(gm) else np.nan
            row[f"{m}_n_folds_present"] = len(gm)
            row[f"{m}_top10pct_any_fold"] = bool(gm["top10pct"].any()) if len(gm) else False
            if row[f"{m}_top10pct_any_fold"]:
                support_count += 1
        row["xai_method_support_count_top10pct"] = support_count
        synth_rows.append(row)
    xai_feature_synthesis = pd.DataFrame(synth_rows)
    xai_feature_synthesis.to_csv(OUT / "xai" / "xai_feature_synthesis.csv", index=False)
    log(f"  xai_feature_synthesis.csv written ({len(xai_feature_synthesis)} features with any XAI "
        f"record -- necessarily the same {xai_df['feature_index'].nunique()} features that were "
        f"Elastic-Net-selected in at least one fold, since XAI was only computed on each fold's "
        f"actual SSAE input).")

    # ═══════════════════════════════════════════════════════════════════
    # 5. XAI CROSS-METHOD AGREEMENT
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- XAI cross-method agreement (top-10%% support count distribution) ---")
    support_dist = xai_feature_synthesis["xai_method_support_count_top10pct"].value_counts().sort_index()
    log(f"  Support-count distribution (features supported by N of 7 methods' top-10%%, any fold): "
        f"{ {int(k): int(v) for k, v in support_dist.items()} }")

    # ═══════════════════════════════════════════════════════════════════
    # 14. XAI METHOD AGREEMENT MATRIX (pairwise top-10% Jaccard, averaged across folds)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- XAI method agreement matrix (pairwise top-10%% Jaccard overlap, per fold, averaged) ---")
    agreement_matrix = pd.DataFrame(index=XAI_METHODS, columns=XAI_METHODS, dtype=float)
    for m1 in XAI_METHODS:
        for m2 in XAI_METHODS:
            jaccards = []
            for fold in range(N_FOLDS):
                s1 = set(xai_ranked[(xai_ranked.fold == fold) & (xai_ranked.method == m1) & (xai_ranked.top10pct)]["feature_index"])
                s2 = set(xai_ranked[(xai_ranked.fold == fold) & (xai_ranked.method == m2) & (xai_ranked.top10pct)]["feature_index"])
                union = s1 | s2
                jaccards.append(len(s1 & s2) / len(union) if union else np.nan)
            agreement_matrix.loc[m1, m2] = np.nanmean(jaccards)
    agreement_matrix.to_csv(OUT / "xai" / "xai_method_agreement.csv")
    log("  xai_method_agreement.csv written -- definition: for each fold, top-10%% feature set per "
        "method (within that fold's selected features), Jaccard overlap between method pairs, "
        "averaged across the 5 folds.")

    overlap_rows = []
    for pct, label in [(1, "top1pct"), (5, "top5pct"), (10, "top10pct")]:
        for fold in range(N_FOLDS):
            sets = {m: set(xai_ranked[(xai_ranked.fold == fold) & (xai_ranked.method == m) & (xai_ranked[f"top{pct}pct"])]["feature_index"])
                    for m in XAI_METHODS}
            all_union = set().union(*sets.values())
            all_intersection = set.intersection(*sets.values()) if all(len(s) for s in sets.values()) else set()
            overlap_rows.append({"threshold": label, "fold": fold, "union_size": len(all_union),
                                 "intersection_all_7_methods": len(all_intersection)})
    xai_top_overlap = pd.DataFrame(overlap_rows)
    xai_top_overlap.to_csv(OUT / "xai" / "xai_top_feature_overlap.csv", index=False)
    log("  xai_top_feature_overlap.csv written.")

    # ═══════════════════════════════════════════════════════════════════
    # 6. XAI + ELASTIC NET AGREEMENT
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Elastic Net stability vs XAI support intersection ---")
    merged = freq_df.merge(xai_feature_synthesis[["feature_index", "xai_method_support_count_top10pct"]],
                           on="feature_index", how="left")
    merged["xai_method_support_count_top10pct"] = merged["xai_method_support_count_top10pct"].fillna(0).astype(int)
    STABLE_THRESH = 4  # documented explicitly, matches feature_stability's definition (>=4/5 folds)
    XAI_STRONG_THRESH = 4  # documented: supported by >=4 of 7 methods' top-10%
    merged["EN_stable"] = merged["n_folds_selected"] >= STABLE_THRESH
    merged["XAI_strong"] = merged["xai_method_support_count_top10pct"] >= XAI_STRONG_THRESH
    def cat(row):
        if row["EN_stable"] and row["XAI_strong"]:
            return "A: Stable EN + strong XAI"
        if row["EN_stable"] and not row["XAI_strong"]:
            return "B: Stable EN + weak XAI"
        if not row["EN_stable"] and row["XAI_strong"]:
            return "C: Unstable EN + strong XAI"
        return "D: Unstable EN + weak XAI"
    merged["category"] = merged.apply(cat, axis=1)
    en_xai_path = OUT / "multi_evidence" / "elastic_net_xai_intersection.csv"
    merged.to_csv(en_xai_path, index=False)
    cat_dist = merged["category"].value_counts()
    log(f"  Thresholds used (documented, not biologically motivated): EN 'stable' = selected in "
        f">={STABLE_THRESH}/5 folds; XAI 'strong' = supported by >={XAI_STRONG_THRESH}/7 methods' top-10%%.")
    log(f"  Category distribution: {dict(cat_dist)}")

    # ═══════════════════════════════════════════════════════════════════
    # 7/8. ROAR ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- ROAR curve analysis ---")
    roar_curve = roar_df.groupby("percentage_removed").agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"), sd_balanced_accuracy=("balanced_accuracy", "std"),
        mean_roc_auc=("roc_auc", "mean"), sd_roc_auc=("roc_auc", "std"),
        mean_f1=("f1", "mean"), sd_f1=("f1", "std"),
        n_folds=("fold", "count"),
    ).reset_index().sort_values("percentage_removed")
    roar_curve.to_csv(OUT / "roar" / "roar_curve_summary.csv", index=False)

    baseline_metrics = pd.read_csv(P6 / "evaluation" / "overall_metrics.csv")
    baseline_acc_mean = baseline_metrics.loc[baseline_metrics.statistic == "mean", "accuracy"].iloc[0]
    baseline_auc_mean = baseline_metrics.loc[baseline_metrics.statistic == "mean", "roc_auc"].iloc[0]
    roar_summary = roar_curve.copy()
    roar_summary["delta_accuracy_vs_baseline"] = roar_summary["mean_accuracy"] - baseline_acc_mean
    roar_summary["delta_roc_auc_vs_baseline"] = roar_summary["mean_roc_auc"] - baseline_auc_mean
    roar_summary.to_csv(OUT / "roar" / "roar_summary.csv", index=False)
    log(f"  ROAR curve (mean accuracy by %% removed): "
        f"{roar_curve[['percentage_removed','mean_accuracy']].round(4).to_dict(orient='records')}")
    log(f"  Baseline (0%% removed, Phase 6 primary result): accuracy={baseline_acc_mean:.4f}, "
        f"AUC={baseline_auc_mean:.4f}")
    log("  IMPORTANT WORDING: a performance decrease after removing top-ranked features shows the "
        "MODEL's dependence on those features under this retraining procedure -- it does not, by "
        "itself, prove any removed feature is biologically/causally necessary for the underlying "
        "diagnostic distinction.")

    # ROAR feature support -- LIMITATION: Phase 6 did not persist the exact per-feature training-fold
    # IG ranking used to choose ROAR removal order, only aggregate performance per threshold. We use
    # the SAVED test-fold IntegratedGradients attribution (same method, different data) as the closest
    # available proxy, explicitly labeled as such.
    ig_test = xai_ranked[xai_ranked.method == "IntegratedGradients"]
    roar_feature_support_rows = []
    for feat_idx, g in ig_test.groupby("feature_index"):
        freq_row = freq_df[freq_df.feature_index == feat_idx]
        thresholds_likely = []
        for _, r in g.iterrows():
            if r["top1pct"]:
                thresholds_likely.append("top1pct")
            elif r["top5pct"]:
                thresholds_likely.append("top5pct")
            elif r["top10pct"]:
                thresholds_likely.append("top10pct")
        roar_feature_support_rows.append({
            "feature_index": feat_idx, "roi_i": g["roi_i"].iloc[0], "roi_j": g["roi_j"].iloc[0],
            "selection_frequency": float(freq_row["selection_frequency"].iloc[0]) if len(freq_row) else np.nan,
            "xai_support_count_top10pct": int(xai_feature_synthesis.loc[
                xai_feature_synthesis.feature_index == feat_idx, "xai_method_support_count_top10pct"].iloc[0])
                if feat_idx in xai_feature_synthesis.feature_index.values else 0,
            "test_fold_IG_mean_abs_attribution": float(g["mean_abs_attribution"].mean()),
            "likely_early_roar_removal_folds": ";".join(str(int(f)) for f in g.loc[g["top10pct"], "fold"]),
            "approximation_note": "Based on test-fold IntegratedGradients (saved by Phase 6), used as "
                                  "a proxy for the training-fold IG ranking Phase 6 actually used "
                                  "internally for ROAR removal order, which was NOT persisted to a "
                                  "per-feature file. This is an approximation, not the exact ROAR "
                                  "removal record.",
        })
    roar_feature_support = pd.DataFrame(roar_feature_support_rows)
    roar_feature_support.to_csv(OUT / "roar" / "roar_feature_support.csv", index=False)
    log(f"  roar_feature_support.csv written ({len(roar_feature_support)} rows) -- LIMITATION "
        f"explicitly documented in the file: exact per-feature ROAR removal order was not saved by "
        f"Phase 6, so this uses the closest available proxy (test-fold IG), not the literal ranking.")

    # ═══════════════════════════════════════════════════════════════════
    # 9/10. MULTI-EVIDENCE FEATURE TABLE + CONSENSUS CANDIDATES
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Multi-evidence feature table ---")
    multi = freq_df.copy()
    multi = multi.merge(coef_stats, on="feature_index", how="left")
    multi = multi.merge(sign_df, on="feature_index", how="left")
    for m in XAI_METHODS:
        col = f"{m}_mean_abs_attribution"
        sub = xai_ranked[xai_ranked.method == m].groupby("feature_index")["mean_abs_attribution"].mean()
        multi[f"{m}_importance"] = multi["feature_index"].map(sub)
    multi = multi.merge(xai_feature_synthesis[["feature_index", "xai_method_support_count_top10pct"]],
                        on="feature_index", how="left").rename(columns={"xai_method_support_count_top10pct": "xai_method_support_count"})
    multi["xai_method_support_count"] = multi["xai_method_support_count"].fillna(0).astype(int)
    multi = multi.merge(roar_feature_support[["feature_index", "likely_early_roar_removal_folds"]],
                        on="feature_index", how="left")
    multi["roar_supported"] = multi["likely_early_roar_removal_folds"].fillna("").astype(str).str.len() > 0
    multi = multi.rename(columns={"likely_early_roar_removal_folds": "roar_threshold_information"})
    multi_path = OUT / "multi_evidence" / "multi_evidence_feature_table.csv"
    multi.to_csv(multi_path, index=False)
    log(f"  multi_evidence_feature_table.csv written ({len(multi)} rows, all 6,670 global features, "
        f"transparent evidence columns -- no combined score computed).")

    multi_evidence_summary = pd.DataFrame([{
        "total_features": 6670, "selected_at_least_once": int((multi.n_folds_selected > 0).sum()),
        "selected_every_fold": int((multi.n_folds_selected == 5).sum()),
        "EN_stable_and_XAI_strong": int((cat_dist.get("A: Stable EN + strong XAI", 0))),
        "EN_stable_and_XAI_weak": int((cat_dist.get("B: Stable EN + weak XAI", 0))),
        "EN_unstable_and_XAI_strong": int((cat_dist.get("C: Unstable EN + strong XAI", 0))),
        "EN_unstable_and_XAI_weak": int((cat_dist.get("D: Unstable EN + weak XAI", 0))),
        "roar_supported_features": int(multi["roar_supported"].sum()),
    }])
    multi_evidence_summary.to_csv(OUT / "multi_evidence" / "multi_evidence_summary.csv", index=False)

    consensus = multi[(multi["n_folds_selected"] >= 4) & (multi["xai_method_support_count"] >= 3)].copy()
    consensus["evidence_description"] = consensus.apply(
        lambda r: f"Selected in {int(r['n_folds_selected'])}/5 folds; supported by "
                  f"{int(r['xai_method_support_count'])}/7 XAI methods' top-10%% (any fold)"
                  f"{'; test-fold IG places it in a top-K bracket in at least one fold' if r['roar_supported'] else ''}.",
        axis=1)
    consensus_out = consensus[["feature_index", "roi_i", "roi_j", "roi_i_name", "roi_j_name",
                                "n_folds_selected", "selection_frequency", "xai_method_support_count",
                                "roar_supported", "evidence_description"]].rename(
        columns={"n_folds_selected": "ElasticNet_selection_frequency_count",
                "selection_frequency": "ElasticNet_selection_frequency",
                "xai_method_support_count": "XAI_method_support_count",
                "roar_supported": "ROAR_support"})
    consensus_out.insert(len(consensus_out.columns) - 1, "XAI_importance_summary",
                         consensus.apply(lambda r: {m: round(r.get(f"{m}_importance", np.nan), 4)
                                                    for m in XAI_METHODS if pd.notna(r.get(f"{m}_importance"))}, axis=1))
    consensus_out.to_csv(OUT / "multi_evidence" / "consensus_feature_candidates.csv", index=False)
    log(f"  consensus_feature_candidates.csv written ({len(consensus_out)} 'multi-evidence "
        f"computational feature candidates' -- NOT labeled biomarkers).")

    # ═══════════════════════════════════════════════════════════════════
    # 11/12/13. ROI-LEVEL SYNTHESIS
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- ROI-level synthesis ---")
    roi_feature_rows = []
    for roi_idx in range(1, 117):
        involved = multi[(multi.roi_i == roi_idx) | (multi.roi_j == roi_idx)]
        stable_involved = involved[involved.n_folds_selected >= 4]
        xai_involved = involved[involved.xai_method_support_count >= 1]
        roar_involved = involved[involved.roar_supported]
        roi_feature_rows.append({
            "roi_index": roi_idx, "roi_name": roi_summary_p6.loc[roi_summary_p6.roi_index == roi_idx, "roi_name"].iloc[0],
            "n_connections_total": len(involved),
            "n_selected_at_least_once": int((involved.n_folds_selected > 0).sum()),
            "n_stable_connections_ge4folds": len(stable_involved),
            "n_xai_supported_connections": len(xai_involved),
            "n_roar_associated_connections": len(roar_involved),
            "mean_selection_frequency": float(involved["selection_frequency"].mean()),
        })
    roi_feature_summary = pd.DataFrame(roi_feature_rows)
    roi_feature_summary.to_csv(OUT / "roi_analysis" / "roi_feature_summary.csv", index=False)
    roi_summary_p6.to_csv(OUT / "roi_analysis" / "roi_connection_summary.csv", index=False)  # re-saved, read-only source
    roi_feature_summary[["roi_index", "roi_name", "n_xai_supported_connections"]].to_csv(
        OUT / "roi_analysis" / "roi_xai_summary.csv", index=False)
    roi_feature_summary[["roi_index", "roi_name", "n_roar_associated_connections"]].to_csv(
        OUT / "roi_analysis" / "roi_roar_summary.csv", index=False)
    roi_feature_summary.to_csv(OUT / "roi_analysis" / "roi_multi_evidence_summary.csv", index=False)
    log("  roi_analysis/ outputs written (no ROI ranked 'best'/'worst').")

    # hemisphere analysis on the 5/5-stable set (computational, not biological)
    def hemi(name):
        if not isinstance(name, str):
            return "unknown"
        if name.endswith("_L"):
            return "L"
        if name.endswith("_R"):
            return "R"
        return "midline"  # e.g. Vermis_*
    stable5 = multi[multi.n_folds_selected == 5].copy()
    stable5["hemi_i"] = stable5["roi_i_name"].map(hemi)
    stable5["hemi_j"] = stable5["roi_j_name"].map(hemi)
    def pair_type(row):
        pair = tuple(sorted([row["hemi_i"], row["hemi_j"]]))
        if pair == ("L", "L"):
            return "left-left"
        if pair == ("R", "R"):
            return "right-right"
        if "midline" in pair:
            return "midline-involved"
        return "left-right"
    stable5["connection_type"] = stable5.apply(pair_type, axis=1)
    hemi_summary = stable5["connection_type"].value_counts()
    log(f"  Hemisphere breakdown of the 50 features selected in all 5 folds: {dict(hemi_summary)} "
        f"(computational grouping from AAL name suffixes only; NOT an 'external anatomical "
        f"interpretation' -- no canonical functional-network labels applied, per instruction).")

    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Generating visualizations ---")
    plt.figure(figsize=(6, 4))
    cat_counts.plot(kind="bar", color="steelblue")
    plt.xlabel("Number of folds selected (0-5)"); plt.ylabel("Number of features")
    plt.title("Elastic Net selection-frequency distribution (6,670 features)")
    plt.tight_layout(); plt.savefig(OUT / "visualizations" / "01_selection_frequency_distribution.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(agreement_matrix.values.astype(float), cmap="viridis", vmin=0, vmax=1)
    plt.xticks(range(7), XAI_METHODS, rotation=45, ha="right"); plt.yticks(range(7), XAI_METHODS)
    plt.colorbar(label="Mean Jaccard overlap (top-10%, avg over 5 folds)")
    plt.title("XAI method agreement matrix"); plt.tight_layout()
    plt.savefig(OUT / "visualizations" / "02_xai_method_agreement_heatmap.png", dpi=120); plt.close()

    plt.figure(figsize=(6, 4))
    support_dist.plot(kind="bar", color="darkorange")
    plt.xlabel("Number of XAI methods (of 7) with feature in top-10%, any fold"); plt.ylabel("Number of features")
    plt.title("Cross-method XAI top-10% support distribution")
    plt.tight_layout(); plt.savefig(OUT / "visualizations" / "03_xai_top_feature_overlap.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 5))
    for metric, col in [("accuracy", "mean_accuracy"), ("roc_auc", "mean_roc_auc")]:
        plt.errorbar(roar_curve["percentage_removed"], roar_curve[col],
                    yerr=roar_curve[col.replace("mean", "sd")], marker="o", label=metric, capsize=3)
    plt.axhline(baseline_acc_mean, color="gray", linestyle="--", linewidth=1, label="baseline accuracy (0% removed)")
    plt.xlabel("% top-ranked features removed"); plt.ylabel("Metric value"); plt.legend()
    plt.title("ROAR performance curve (mean ± SD across 5 folds)")
    plt.tight_layout(); plt.savefig(OUT / "visualizations" / "04_roar_performance_curve.png", dpi=120); plt.close()

    plt.figure(figsize=(6, 5))
    valid = multi.dropna(subset=["IntegratedGradients_importance"])
    plt.scatter(valid["selection_frequency"], valid["IntegratedGradients_importance"], alpha=0.4, s=10)
    plt.xlabel("Elastic Net selection frequency"); plt.ylabel("Mean |IntegratedGradients| attribution")
    plt.title("Feature stability vs XAI importance (Integrated Gradients)")
    plt.tight_layout(); plt.savefig(OUT / "visualizations" / "05_stability_vs_xai_importance.png", dpi=120); plt.close()

    plt.figure(figsize=(10, 4))
    top_rois = roi_feature_summary.sort_values("mean_selection_frequency", ascending=False).head(20)
    plt.bar(top_rois["roi_name"], top_rois["mean_selection_frequency"], color="seagreen")
    plt.xticks(rotation=75, ha="right"); plt.ylabel("Mean selection frequency of connections")
    plt.title("Top 20 ROIs by mean connection selection-frequency (NOT a 'most important region' ranking)")
    plt.tight_layout(); plt.savefig(OUT / "visualizations" / "06_roi_selection_frequency.png", dpi=120); plt.close()

    log("  6 visualization figures written to visualizations/. No brain-surface/atlas-image maps "
        "generated (would require downloading an atlas image not already verified in this project, "
        "per instruction) -- tabular ROI mapping provided instead.")

    # ═══════════════════════════════════════════════════════════════════
    # REPORTS
    # ═══════════════════════════════════════════════════════════════════
    _write_reports(OUT, checks, cat_counts, expected_cats, cats_match, stable_features, support_dist,
                   cat_dist, roar_curve, baseline_acc_mean, baseline_auc_mean, consensus_out,
                   roi_feature_summary, hemi_summary, problems, xai_method_summary)

    # ── audit ───────────────────────────────────────────────────────────
    post_guard = guard_fingerprints()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phase 1-6 files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    if not guard_unchanged:
        problems.append("Prior-phase (including Phase 6) files were modified during Phase 7.")

    status = "PASS" if (not problems and guard_unchanged) else "BLOCKED"
    log(f"\nFINAL PHASE 7 STATUS: {status}")
    if problems:
        log(f"Problems: {problems}")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_check": checks, "phase7_status": status, "problems": problems,
        "n_selected_at_least_once": unique_selected, "n_selected_every_fold": selected_every_fold,
        "category_distribution": {int(k): int(v) for k, v in cat_counts.items()},
        "en_xai_category_distribution": {k: int(v) for k, v in cat_dist.items()},
        "n_consensus_candidates": len(consensus_out),
        "roar_baseline_accuracy": float(baseline_acc_mean), "roar_baseline_auc": float(baseline_auc_mean),
        "audit": {"previous_phases_unchanged": guard_unchanged, "phase6_retrained": False,
                  "new_model_computation": False, "global_feature_selection_performed": False},
    }
    with open(OUT / "phase7_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT / "phase7_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2, default=str))
    with open(OUT / "logs" / "phase7_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    log(f"\nphase7_summary.json/.txt and logs/phase7_audit.log written. STATUS={status}")
    return summary


def _write_blocked(reason, problems):
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "phase7_status": "BLOCKED",
              "reason": reason, "problems": problems}
    with open(OUT / "phase7_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT / "logs" / "phase7_audit.log", "w") as f:
        f.write("\n".join(audit_lines))


def _write_reports(OUT, checks, cat_counts, expected_cats, cats_match, stable_features, support_dist,
                   cat_dist, roar_curve, baseline_acc, baseline_auc, consensus_out, roi_feature_summary,
                   hemi_summary, problems, xai_method_summary):
    findings_md = f"""# Phase 7 Findings

**Q1. How many features were selected at least once?** 1,286 of 6,670 (19.3%).

**Q2. How many were selected in every fold?** 50.

**Q3. Which features are most stable?** The 50 features selected in all 5/5 folds (see
`feature_stability/stable_features.csv`, filtered to n_folds_selected==5); a further 93 were selected
in 4/5 folds.

**Q4. Which features have strongest cross-method XAI support?** See
`xai/xai_feature_synthesis.csv`, `xai_method_support_count_top10pct` column. Support-count
distribution across all XAI-attributed features: {dict(support_dist)}.

**Q5. Which features show Elastic Net + XAI agreement?** Category "A: Stable EN + strong XAI"
(selected >=4/5 folds AND supported by >=4/7 XAI methods' top-10%): {cat_dist.get('A: Stable EN + strong XAI', 0)}
features. Full breakdown: `multi_evidence/elastic_net_xai_intersection.csv`.

**Q6. What does ROAR show?** Baseline (no removal): accuracy {baseline_acc:.4f}, AUC {baseline_auc:.4f}.
Full curve in `roar/roar_curve_summary.csv`. Performance changes with removal percentage reflect the
retrained model's dependence on the removed features under this procedure -- not proof that those
features are causally necessary for the underlying phenomenon.

**Q7. Do the same features repeatedly appear across Elastic Net, XAI, and ROAR?** Partially -- see
`multi_evidence/consensus_feature_candidates.csv` ({len(consensus_out)} candidates meeting the
explicit, documented thresholds of >=4/5-fold EN selection and >=3/7-method XAI top-10% support).
This is convergence across evidence sources, not proof of ground truth.

**Q8. Which ROI pairs repeatedly appear?** See `feature_stability/stable_features.csv` and
`multi_evidence/consensus_feature_candidates.csv` for the specific (ROI_i, ROI_j) pairs, by verified
AAL name.

**Q9. Which AAL regions participate in those repeated connections?** See
`roi_analysis/roi_feature_summary.csv`, `n_stable_connections_ge4folds` column, for the per-ROI count
of connections meeting the stability threshold.

**Q10. Where do the methods disagree?** XAI method agreement (`xai/xai_method_agreement.csv`) is
well below 1.0 for most method pairs, meaning each method's top-10% set only partially overlaps with
others' -- full pairwise values are in that file, not summarized away here. Category "C: Unstable EN
+ strong XAI" ({cat_dist.get('C: Unstable EN + strong XAI', 0)} features) shows XAI methods
attributing importance to features Elastic Net did not select stably across folds -- a real
disagreement between selection-based and attribution-based evidence, reported as such.

**Q11. What can legitimately be called a robust computational finding?** Statements of the form "this
FC feature was selected in 5/5 independent cross-validation folds" or "this ROI pair's connection was
supported by N of 7 independent XAI methods' top-10% ranking" are robust computational findings within
this specific pipeline (Strategy C + Elastic Net + variable-input SSAE) and this specific dataset.

**Q12. What CANNOT yet be called a biological biomarker?** Every single finding in this phase.
Selection frequency, XAI support, and ROAR association are all evidence about *this model's*
behavior on *this dataset* under *this specific, modified-reproduction methodology* -- none of them,
individually or combined, constitutes biological or clinical validation. See
`reports/PHASE7_LIMITATIONS.md`.
"""
    with open(OUT / "reports" / "PHASE7_FINDINGS.md", "w") as f:
        f.write(findings_md)

    limitations_md = """# Phase 7 Limitations

1. This is a MODIFIED REPRODUCTION of the original paper, not an exact reproduction.
2. Strategy C introduced fold-scoped imputation for the 14,025 undefined Pearson FC entries -- a
   project-introduced methodological choice the paper does not specify.
3. Elastic Net replaced the paper's SVM-RFE feature-selection step.
4. All-non-zero feature selection replaced the paper's fixed 1,000-feature target.
5. SSAE input dimension varied by fold (427-498), instead of a fixed 1,000.
6. Phase 6 accuracy was approximately 61.5% +/- 2.5% -- far below, and not comparable to, the
   original paper's disputed 98.2% figure (itself under an active Expression of Concern for data
   leakage, per Phase 4.1).
7. ROC-AUC was approximately 0.677 +/- 0.024.
8. Phase 6's training/validation loss curves showed clear overfitting (near-zero training loss vs.
   substantially higher validation loss).
9. XAI attributions are model-dependent -- they explain what THIS trained model responds to, not an
   independent ground truth about the brain.
10. ROAR results are model-dependent and procedure-dependent (reduced fine-tuning epochs were used
    for the 65 ROAR retrains, per Phase 6's documented compute-scoping decision).
11. Elastic Net selection stability does not establish biological importance.
12. XAI cross-method agreement does not establish biological causality.
13. ROAR-driven performance change does not establish biological causality.
14. AAL ROI interpretation used here is anatomical name-mapping only, not disease-mechanism proof.
15. The 50 features selected in every fold are computationally stable within this pipeline, not
    automatically biomarkers.
16. Site effects and preprocessing effects (documented extensively in Phase 4.1/4.2 -- e.g. the
    SBL/PITT constant-ROI concentration) may influence which features appear stable or important,
    independent of any diagnosis-related signal.
17. The 47 constant-ROI subjects were retained via Strategy C's fold-scoped mean imputation; their
    imputed values are not measured brain activity, and any XAI attribution touching their
    imputed features (in folds where they fell in the test set) reflects a fit to a substituted
    statistic for those entries, not a real measurement.

**Additional limitation identified during this phase:** Phase 6 did not persist the exact per-feature
training-fold Integrated-Gradients ranking used internally to order ROAR's feature removal --
`roar/roar_results.csv` contains only aggregate per-threshold performance. Phase 7's
`roar_feature_support.csv` uses the closest available proxy (Phase 6's saved *test-fold* Integrated
Gradients attributions) and explicitly labels this as an approximation, not the literal ROAR removal
record.
"""
    with open(OUT / "reports" / "PHASE7_LIMITATIONS.md", "w") as f:
        f.write(limitations_md)

    report_md = f"""# Phase 7 Report — Feature Stability + XAI + ROAR + ROI Synthesis

Freeze-check against Phase 6: {'ALL PASS' if all(checks.values()) else 'FAILED -- see problems'}.
Category distribution verified against expected Phase 6 values: {cats_match}.

See `PHASE7_FINDINGS.md` for the Q1-Q12 answers and `PHASE7_LIMITATIONS.md` for the full limitations
list. All underlying tables are in `feature_stability/`, `xai/`, `roar/`, `roi_analysis/`, and
`multi_evidence/`. Visualizations in `visualizations/`.

Problems encountered: {problems if problems else 'None.'}
"""
    with open(OUT / "reports" / "PHASE7_REPORT.md", "w") as f:
        f.write(report_md)


if __name__ == "__main__":
    main()
