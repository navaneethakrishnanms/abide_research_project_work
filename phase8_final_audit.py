"""
phase8_final_audit.py -- Phase 8: final scientific interpretation and
reproducibility audit. Read-only over Phases 1-7. No retraining, no new
model computation, no cohort/methodology changes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
P4_1 = ROOT / "results" / "phase4_1_constant_roi_audit"
P4_2 = ROOT / "results" / "phase4_2_methodological_sensitivity"
P6 = ROOT / "results" / "phase6_strategyC_all_nonzero"
P7 = ROOT / "results" / "phase7_synthesis"
OUT = ROOT / "results" / "phase8_final_audit"
for sub in ["reports", "data_audit", "claim_audit", "reproducibility", "paper_comparison", "final_summary", "viva"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

GUARD_ROOTS = [ROOT / "results" / d for d in [
    "phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification",
    "phase3_6_aal_download", "phase4", "phase4_1_constant_roi_audit",
    "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic", "phase5b_missing_fc_analysis",
    "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero",
    "phase6_strategyC_all_nonzero", "phase7_synthesis",
]]

audit_lines = []


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg, flush=True)


def fp_snapshot():
    out = {}
    for root in GUARD_ROOTS:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    out[str(p)] = (st.st_size, st.st_mtime)
    return out


def main():
    log("=" * 70)
    log("PHASE 8 -- Final scientific interpretation & reproducibility audit (read-only).")
    log("=" * 70)
    pre_guard = fp_snapshot()
    problems = []

    # ═══════════════════════════════════════════════════════════════════
    # 2. COHORT VERIFICATION
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Cohort / authoritative-value verification ---")
    cohort = pd.read_csv(P6 / "data_checks" / "data_integrity_summary.csv").iloc[0]
    fold_counts = pd.read_csv(P6 / "elastic_net" / "fold_feature_counts.csv").sort_values("fold")
    N_vals = fold_counts["N"].tolist()
    stability_summary = pd.read_csv(P6 / "feature_stability" / "stability_summary.csv")
    unique_sel = int(stability_summary.loc[stability_summary.n_folds_selected == "total_unique_selected_at_least_once", "n_features"].iloc[0])
    every_fold = int(stability_summary.loc[stability_summary.n_folds_selected == "total_selected_in_every_completed_fold", "n_features"].iloc[0])
    stability_df = pd.read_csv(P6 / "feature_stability" / "feature_stability.csv")
    ge4 = int((stability_df.n_folds_selected >= 4).sum())

    constant_roi_summary = pd.read_csv(P4_1 / "constant_roi_summary.csv")
    fc_undefined_summary = pd.read_csv(P4_1 / "fc_undefined_summary.csv")

    checks = {
        "n_subjects_780": int(cohort.n_subjects) == 780,
        "n_asd_351": int(cohort.n_asd) == 351,
        "n_control_429": int(cohort.n_control) == 429,
        "n_sites_20": int(cohort.n_sites) == 20,
        "n_rois_116": int(cohort.n_rois) == 116,
        "n_fc_features_6670": int(cohort.n_fc_features) == 6670,
        "fold_N_matches": N_vals == [498, 479, 427, 463, 454],
        "mean_464_2": abs(np.mean(N_vals) - 464.2) < 0.05,
        "median_463": np.median(N_vals) == 463,
        "std_23_9": abs(np.std(N_vals) - 23.9) < 0.1,
        "unique_selected_1286": unique_sel == 1286,
        "selected_every_fold_50": every_fold == 50,
        "selected_ge4_143": ge4 == 143,
    }
    for k, v in checks.items():
        log(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        problems.append(f"Cohort/authoritative-value verification FAILED: {checks}")
        log(f"[STOP] Verification failed: {checks}")
        _write_blocked("Cohort/authoritative-value verification failed against Phase 6/7 files.", problems)
        return

    n_constant_roi_subjects = int(constant_roi_summary["n_subjects_with_constant_roi"].iloc[0]) if "n_subjects_with_constant_roi" in constant_roi_summary.columns else None
    log(f"  constant_roi_summary.csv columns: {list(constant_roi_summary.columns)}")
    log(f"  fc_undefined_summary.csv columns: {list(fc_undefined_summary.columns)}, values: {fc_undefined_summary.iloc[0].to_dict()}")
    log("Freeze-check: ALL PASS.")

    # ═══════════════════════════════════════════════════════════════════
    # 9. FEATURE STABILITY DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════════
    cat_counts = stability_df["n_folds_selected"].value_counts().sort_index()
    log(f"\nFeature stability distribution (authoritative): {dict(cat_counts)}")

    # ═══════════════════════════════════════════════════════════════════
    # 10/11 XAI + intersection (from Phase 7 outputs)
    # ═══════════════════════════════════════════════════════════════════
    xai_synth = pd.read_csv(P7 / "xai" / "xai_feature_synthesis.csv")
    support_dist = xai_synth["xai_method_support_count_top10pct"].value_counts().sort_index()
    log(f"XAI cross-method top-10%% support distribution (authoritative): {dict(support_dist)}")

    en_xai = pd.read_csv(P7 / "multi_evidence" / "elastic_net_xai_intersection.csv")
    cat_dist = en_xai["category"].value_counts()
    log(f"Elastic Net x XAI category distribution (authoritative): {dict(cat_dist)}")

    consensus = pd.read_csv(P7 / "multi_evidence" / "consensus_feature_candidates.csv")
    n_consensus = len(consensus)
    log(f"Multi-evidence computational feature candidates (authoritative): {n_consensus}")

    # ═══════════════════════════════════════════════════════════════════
    # 12 ROAR
    # ═══════════════════════════════════════════════════════════════════
    roar_curve = pd.read_csv(P7 / "roar" / "roar_curve_summary.csv")
    overall = pd.read_csv(P6 / "evaluation" / "overall_metrics.csv")
    fold_metrics = pd.read_csv(P6 / "evaluation" / "fold_metrics.csv")
    # True baseline (0% removed) is Phase 6's own primary evaluation, not the 1%-removed ROAR row
    # (roar_results.csv's lowest recorded threshold is 1%, not 0%).
    baseline_acc_mean = overall.loc[overall.statistic == "mean", "accuracy"].iloc[0]
    baseline_auc_mean = overall.loc[overall.statistic == "mean", "roc_auc"].iloc[0]
    baseline = pd.DataFrame([{"percentage_removed": 0, "mean_accuracy": baseline_acc_mean,
                              "mean_roc_auc": baseline_auc_mean,
                              "source": "Phase 6 overall_metrics.csv (primary result, no features removed)"}])
    log(f"ROAR true baseline (0%% removed, from Phase 6 primary evaluation): "
        f"{baseline.to_dict(orient='records')}")
    log(f"ROAR curve's own lowest recorded threshold is 1%% removed: "
        f"{roar_curve[roar_curve.percentage_removed == roar_curve.percentage_removed.min()].to_dict(orient='records')}")

    # ═══════════════════════════════════════════════════════════════════
    # 15. 50 fully stable features
    # ═══════════════════════════════════════════════════════════════════
    fully_stable = stability_df[stability_df.n_folds_selected == 5].copy()
    fully_stable_export = fully_stable.merge(
        xai_synth[["feature_index", "xai_method_support_count_top10pct"]], on="feature_index", how="left")
    roar_support = pd.read_csv(P7 / "roar" / "roar_feature_support.csv")
    fully_stable_export = fully_stable_export.merge(
        roar_support[["feature_index", "likely_early_roar_removal_folds", "approximation_note"]],
        on="feature_index", how="left")
    fully_stable_export.to_csv(OUT / "data_audit" / "fully_stable_50_features.csv", index=False)
    log(f"\nfully_stable_50_features.csv written ({len(fully_stable_export)} rows).")

    # ═══════════════════════════════════════════════════════════════════
    # 20. REPRODUCIBILITY AUDIT
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Reproducibility audit ---")
    repro_rows = [
        ("DATA", "Dataset source", "COMPLETE", "ABIDE-I via nilearn fetch_abide_pcp, CPAC pipeline, documented in Phase 1/3."),
        ("DATA", "Cohort definition", "COMPLETE", "780 subjects after FD>0.2mm filtering, documented in Phase 2, verified in Phase 6 data_checks."),
        ("DATA", "Subject filtering (FD)", "COMPLETE", "FD threshold and per-subject FD values recorded in Phase 2 fd_filtered_subjects.csv."),
        ("DATA", "AAL source", "COMPLETE", "CPAC rois_aal derivative, provenance documented Phase 3.6."),
        ("DATA", "Pipeline (CPAC) documented", "COMPLETE", "Explicitly recorded as the approved derivative in Phase 3.6."),
        ("DATA", "ROI count verification", "COMPLETE", "116 ROIs verified per subject file in Phase 4."),
        ("FEATURE ENGINEERING", "Pearson correlation method", "COMPLETE", "Standard Pearson correlation on ROI time series, documented Phase 5A."),
        ("FEATURE ENGINEERING", "Feature ordering / lower triangle convention", "COMPLETE", "Fixed feature_index convention (roi_i<roi_j, 1-indexed) in Phase 5A fc_feature_index.csv, reused unchanged through Phase 6-7."),
        ("FEATURE ENGINEERING", "Fisher-Z transform", "COMPLETE", "arctanh with epsilon clipping 1e-6, documented in phase6_pipeline_core.py."),
        ("FEATURE ENGINEERING", "Missing/undefined-FC imputation (Strategy C)", "COMPLETE", "Fold-scoped training-fold mean imputation, rationale documented Phase 5B/5C/5D, implementation in phase6_pipeline_core.py."),
        ("MODEL", "Elastic Net hyperparameters", "COMPLETE", "C=0.1, l1_ratio=0.5, solver=saga, max_iter=3000, tol=1e-4, class_weight=balanced, random_state=42 -- fixed a priori, recorded in elastic_net_parameters.csv."),
        ("MODEL", "SSAE architecture", "COMPLETE", "N(variable)->500->100->2, ReLU hidden / Softmax output, recorded per-fold in fold_architecture.csv."),
        ("MODEL", "Training parameters", "COMPLETE", "Adam lr=1e-3, weight_decay=1e-4, batch_size=32, pretrain 40 epochs/layer, finetune 80 epochs, sparsity_weight=1e-4 -- all recorded."),
        ("MODEL", "Random seed", "COMPLETE", "random_seed=42 recorded for both Elastic Net and SSAE (fold_architecture.csv, elastic_net_parameters.csv)."),
        ("MODEL", "Cross-validation fold structure", "COMPLETE", "5-fold StratifiedKFold, subject-level, fold assignment fixed and reused unchanged across Phase 6/7."),
        ("XAI", "Methods used", "COMPLETE", "7 named methods (IntegratedGradients, GuidedBackprop, DeepLift, DeepLiftShap, GradientShap, SHAP, LIME) recorded."),
        ("XAI", "Software/library versions", "PARTIAL", "captum, shap, lime libraries used (per phase6_pipeline_core.py imports) but exact installed version numbers were not captured into a Phase 6 output file."),
        ("XAI", "Baseline/reference choices", "PARTIAL", "Captum methods' internal baseline defaults were used; not all were explicitly logged as a named baseline value per fold in the output CSVs."),
        ("XAI", "Sampling choices (e.g. LIME subject subsampling)", "COMPLETE", "LIME explicitly documented as limited to 5 representative test subjects per fold (reflected in its row count vs other methods)."),
        ("ROAR", "Removal percentages", "COMPLETE", "13 fixed thresholds (1,5,10,20,30,40,50,60,70,80,90,95,99%) recorded."),
        ("ROAR", "Ranking methodology", "COMPLETE", "Documented as training-fold mean |IntegratedGradients| ranking in phase6_pipeline_core.py source."),
        ("ROAR", "Retraining settings", "COMPLETE", "ROAR_FINETUNE_EPOCHS=40 (reduced from the primary 80) explicitly documented as a compute-scoping decision."),
        ("ROAR", "Per-feature removal provenance (exact list per run)", "MISSING", "Phase 6 persisted only aggregate per-(fold,percentage) performance metrics, not the literal per-feature removal list; Phase 7 uses a documented test-fold IG proxy, explicitly labeled as an approximation, not the exact record."),
    ]
    repro_df = pd.DataFrame(repro_rows, columns=["category", "item", "status", "explanation"])
    repro_df.to_csv(OUT / "reports" / "REPRODUCIBILITY_AUDIT.csv", index=False)
    repro_df.to_csv(OUT / "reproducibility" / "reproducibility_checklist.csv", index=False)
    log(f"  Reproducibility audit: {sum(repro_df.status=='COMPLETE')} COMPLETE, "
        f"{sum(repro_df.status=='PARTIAL')} PARTIAL, {sum(repro_df.status=='MISSING')} MISSING "
        f"(of {len(repro_df)} items). No overall numeric score computed, per instruction.")

    # ═══════════════════════════════════════════════════════════════════
    # 19. CLAIM AUDIT
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Claim / terminology audit across Phase 6-7 reports ---")
    keywords = ["biomarker", "diagnostic biomarker", "biological marker", "disease marker",
                "causal", "mechanism", "clinically useful", "clinical biomarker", "validated"]
    md_files = [P6 / "PHASE6_REPORT.md", P7 / "reports" / "PHASE7_REPORT.md",
                P7 / "reports" / "PHASE7_FINDINGS.md", P7 / "reports" / "PHASE7_LIMITATIONS.md"]
    claim_rows = []
    for f in md_files:
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="cp1252")
        lines = text.splitlines()
        for lineno, line in enumerate(lines, start=1):
            low = line.lower()
            prev_low = lines[lineno - 2].lower() if lineno >= 2 else ""
            context_low = prev_low + " " + low  # word-wrapped sentences can split the negation onto the prior line
            for kw in keywords:
                if kw in low:
                    is_negation = any(neg in context_low for neg in [
                        "not ", "cannot", "does not", "n't", "no evidence", "not yet", "not establish",
                        "not prove", "not automatically", "not independently"])
                    claim_rows.append({
                        "file": str(f.relative_to(ROOT)), "line": lineno, "keyword": kw,
                        "statement": line.strip(),
                        "claim_type": "hedged/negated" if is_negation else "unhedged -- needs review",
                        "supported": "N/A (properly hedged)" if is_negation else "REVIEW REQUIRED",
                        "reason": "Statement explicitly negates or disclaims the keyword term." if is_negation
                                  else "Statement uses the term without an adjacent negation/disclaimer -- manual review needed.",
                        "recommended_wording": "" if is_negation else
                            "Consider 'model-selected feature' / 'stable computational feature' / "
                            "'XAI-supported feature' / 'multi-evidence computational feature candidate' "
                            "in place of an unqualified biomarker/causal/validated claim.",
                    })
                    break
    claim_df = pd.DataFrame(claim_rows) if claim_rows else pd.DataFrame(
        columns=["file", "line", "keyword", "statement", "claim_type", "supported", "reason", "recommended_wording"])
    claim_df.to_csv(OUT / "reports" / "CLAIM_AUDIT.csv", index=False)
    claim_df.to_csv(OUT / "claim_audit" / "overclaim_audit.csv", index=False)
    n_unhedged = int((claim_df["claim_type"] == "unhedged -- needs review").sum()) if len(claim_df) else 0
    log(f"  Scanned {len(md_files)} report files for overclaim keywords: {len(claim_df)} matching lines found, "
        f"{n_unhedged} flagged as 'unhedged -- needs review' (0 expected, since Phase 6/7 reports were "
        f"already written with explicit hedging).")
    if n_unhedged > 0:
        problems.append(f"{n_unhedged} unhedged biomarker/causal/validated statements found in prior reports -- see CLAIM_AUDIT.csv")

    # ═══════════════════════════════════════════════════════════════════
    # 21. DATA PROVENANCE
    # ═══════════════════════════════════════════════════════════════════
    provenance_rows = [
        {"stage": "Phase 1 cohort", "source": "nilearn.datasets.fetch_abide_pcp (ABIDE-I, CPAC, filt_global)", "verified_by": "Phase 1 output CSVs, subject/site counts", "status": "VERIFIED"},
        {"stage": "Phase 2 FD filtering", "source": "Per-subject FD values from phenotypic file, threshold 0.2mm", "verified_by": "fd_filtered_subjects.csv (780 subjects, 351/429, 20 sites)", "status": "VERIFIED"},
        {"stage": "CPAC AAL data", "source": "fetch_abide_pcp(derivatives=['rois_aal'])", "verified_by": "Phase 3.6 download + integrity checks (116 ROIs, timepoint consistency)", "status": "VERIFIED"},
        {"stage": "AAL-116 mapping", "source": "nilearn bundled docs + independent web-retrieved table + DPARSF forum, cross-checked against actual file headers", "verified_by": "aal116_roi_mapping.csv, Phase 4.1", "status": "VERIFIED (3 independent sources)"},
        {"stage": "Phase 4 constant-ROI findings", "source": "Direct computation on the 780-subject AAL time series", "verified_by": "constant_roi_summary.csv, constant_roi_site_summary.csv", "status": "VERIFIED"},
        {"stage": "Strategy C rationale", "source": "Phase 5B/5C/5D methodological comparison, user decision", "verified_by": "PHASE5D_DECISION.md and prior phase reports", "status": "DOCUMENTED (explicit human decision)"},
        {"stage": "Phase 6 feature selection", "source": "Elastic Net fit fold-wise on real FC data", "verified_by": "fold_feature_selection.csv, fold_feature_counts.csv", "status": "VERIFIED"},
        {"stage": "Phase 7 synthesis", "source": "Aggregation of Phase 6 output CSVs only, no new model computation", "verified_by": "Phase 7 freeze-check (this phase and Phase 7 itself)", "status": "VERIFIED"},
    ]
    provenance_df = pd.DataFrame(provenance_rows)
    provenance_df.to_csv(OUT / "data_audit" / "final_data_provenance.csv", index=False)
    cohort_export = pd.DataFrame([cohort.to_dict()])
    cohort_export.to_csv(OUT / "data_audit" / "final_cohort_verification.csv", index=False)
    log("  final_data_provenance.csv and final_cohort_verification.csv written.")

    # ═══════════════════════════════════════════════════════════════════
    # 22. METHODOLOGICAL DECISION AUDIT (table used in report)
    # ═══════════════════════════════════════════════════════════════════
    decisions = [
        ("Retain all 780 subjects", "Preserve the full FD-filtered cohort rather than further excluding subjects for convenience", "Larger, more representative sample; retains the 47 constant-ROI subjects requiring imputation", "Imputed values for those subjects are not raw measurements (see limitation)"),
        ("Retain all 116 ROIs", "Avoid arbitrarily discarding anatomical coverage over a subset of subjects", "Full 6,670-feature space preserved", "19 ROIs carry undefined correlations for some subjects, requiring imputation downstream"),
        ("Preserve undefined FC through the diagnostic stage", "Avoid premature, silent data loss before a methodology was chosen", "Enabled rigorous Phase 4.2 sensitivity analysis of handling options", "Required an explicit, documented resolution (Strategy C) before modeling could proceed"),
        ("Use Strategy C (fold-scoped imputation)", "Chosen by the user after Phase 5B/5C comparison of subject-exclusion vs ROI-exclusion vs imputation", "No subjects or ROIs discarded; leakage-safe because imputation values are training-fold-only", "Introduces synthetic (imputed) values into a subset of model inputs"),
        ("Use training-fold mean imputation specifically", "Simple, leakage-safe, deterministic given the training fold", "Reproducible per fold", "Imputed value is a training-set summary statistic, not a measurement of that subject"),
        ("Use Elastic Net (replacing SVM-RFE)", "SVM-RFE is the paper's method but is associated with a documented Expression of Concern for leakage; Elastic Net offers a leakage-controllable, well-understood alternative", "A deliberate, disclosed methodology deviation from the paper", "Results are not directly comparable to the paper's headline number"),
        ("Keep all non-zero Elastic Net features (not forced to 1,000)", "Avoid forcing an arbitrary fixed count not implied by the data", "Fold-to-fold feature count varies (427-498)", "Loses a single fixed, cross-fold-comparable feature count and the paper's fixed input dimensionality"),
        ("Use variable-input SSAE", "Direct consequence of the all-non-zero decision; each fold already trains an independent model instance under CV", "No forced padding/truncation of the selected feature set", "SSAE parameter count and capacity differ slightly across folds"),
        ("Use XAI and ROAR as post-model interpretation tools only", "Interpretation should not influence model selection or training (avoid leakage/circularity)", "Attribution and removal-based evidence gathered without altering the trained models", "XAI/ROAR findings are properties of these specific trained models under these specific methods, not independent biological evidence"),
    ]
    decisions_df = pd.DataFrame(decisions, columns=["decision", "rationale", "consequence", "limitation"])
    decisions_df.to_csv(OUT / "data_audit" / "methodological_decision_audit.csv", index=False)
    log("  methodological_decision_audit.csv written.")

    # ═══════════════════════════════════════════════════════════════════
    # 5. PAPER COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════════
    comparison_rows = [
        ("Cohort", "884 subjects (paper-reported)", "780 subjects (FD>0.2mm filtered, Phase 2)", "Different N and inclusion criteria", "Results are not directly comparable sample-for-sample"),
        ("Preprocessing pipeline", "Not independently re-verified beyond the paper's own description", "CPAC (filt_global), documented and verified in Phase 3/3.6", "Same nominal pipeline family (CPAC) but not independently re-validated against the paper's exact preprocessing run", "Assume comparable but not identical preprocessing"),
        ("Atlas", "AAL-116 (paper)", "AAL-116 (this project)", "Same atlas", "Directly comparable at the atlas level"),
        ("FC construction", "Pearson correlation, 6,670 features (paper)", "Pearson correlation, 6,670 features (this project)", "Same nominal construction", "Directly comparable"),
        ("Missing/undefined FC handling", "Not documented as addressed by the paper", "Explicitly handled via Strategy C fold-scoped imputation (14,025 undefined entries)", "This project introduces explicit handling the paper does not document", "A disclosed methodological addition, not present in the original pipeline"),
        ("Feature selection", "SVM-RFE, reduced to exactly 1,000 features", "Elastic Net, all non-zero features retained (427-498 per fold)", "Different algorithm and different, variable output size", "Not a like-for-like feature-selection comparison"),
        ("Number of selected features", "1,000 (fixed)", "427-498 (variable per fold)", "Substantially fewer, and variable", "SSAE input dimensionality differs from the paper by fold"),
        ("SSAE architecture", "1,000 -> ... -> 2 (paper, fixed input)", "N(variable, 427-498) -> 500 -> 100 -> 2", "Variable input dimension not present in the paper's design", "Direct architectural equivalence cannot be claimed"),
        ("Cross-validation", "Not fully documented as independently verified here", "5-fold StratifiedKFold, subject-level, fold-scoped preprocessing", "This project's CV structure is fully documented and leakage-controlled", "N/A -- documented for this project only"),
        ("XAI", "Not part of the original paper's reported pipeline (per this project's documentation)", "7 methods (IntegratedGradients, GuidedBackprop, DeepLift, DeepLiftShap, GradientShap, SHAP, LIME)", "This project adds an XAI layer beyond the original paper", "Additional interpretability analysis not present in the original paper"),
        ("ROAR", "Not part of the original paper's reported pipeline (per this project's documentation)", "13 removal thresholds x 5 folds = 65 runs", "This project adds a ROAR layer beyond the original paper", "Additional removal-based evidence not present in the original paper"),
        ("Evaluation / headline performance", "98.2% accuracy (paper-reported; paper is documented in Phase 4.1 audit trail as subject to an Expression of Concern for SVM-RFE leakage)", "61.5%+/-2.5% accuracy, 0.677+/-0.024 ROC-AUC (this project, 5-fold mean+/-SD)", "Substantially different reported performance", "The difference does not by itself establish which methodology is more correct; see Section 6/25"),
    ]
    comparison_df = pd.DataFrame(comparison_rows, columns=["aspect", "original_paper", "this_project", "difference", "scientific_implication"])
    comparison_df.to_csv(OUT / "reports" / "PAPER_COMPARISON.csv", index=False)
    comparison_df.to_csv(OUT / "paper_comparison" / "methodology_comparison.csv", index=False)
    log("  methodology_comparison.csv / PAPER_COMPARISON.csv written.")

    # ═══════════════════════════════════════════════════════════════════
    # AUDIT / INTEGRITY
    # ═══════════════════════════════════════════════════════════════════
    post_guard = fp_snapshot()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phases 1-7 files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    if not guard_unchanged:
        changed = [k for k in post_guard if pre_guard.get(k) != post_guard.get(k)]
        problems.append(f"Prior-phase files modified during Phase 8: {changed[:20]}")

    status = "PASS" if (not problems and guard_unchanged) else "BLOCKED"

    _write_markdown_reports(cohort, N_vals, cat_counts, support_dist, cat_dist, n_consensus,
                            fold_metrics, overall, roar_curve, comparison_df, repro_df, decisions_df,
                            fully_stable_export, status, problems, guard_unchanged)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase8_status": status, "problems": problems,
        "cohort": cohort.to_dict(),
        "elastic_net_fold_N": N_vals,
        "feature_stability_distribution": {int(k): int(v) for k, v in cat_counts.items()},
        "xai_support_distribution": {int(k): int(v) for k, v in support_dist.items()},
        "en_xai_category_distribution": {k: int(v) for k, v in cat_dist.items()},
        "n_multi_evidence_candidates": n_consensus,
        "roar_baseline": baseline.to_dict(orient="records")[0] if len(baseline) else None,
        "audit": {"previous_phases_unchanged": guard_unchanged, "retraining_performed": False,
                  "cohort_changed": False, "methodology_changed": False},
    }
    with open(OUT / "phase8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT / "phase8_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2, default=str))
    with open(OUT / "phase8_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    log(f"\nFINAL PHASE 8 STATUS: {status}")
    return summary


def _write_blocked(reason, problems):
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "phase8_status": "BLOCKED",
              "reason": reason, "problems": problems}
    with open(OUT / "phase8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT / "phase8_audit.log", "w") as f:
        f.write("\n".join(audit_lines))


def _write_markdown_reports(cohort, N_vals, cat_counts, support_dist, cat_dist, n_consensus,
                            fold_metrics, overall, roar_curve, comparison_df, repro_df, decisions_df,
                            fully_stable_export, status, problems, guard_unchanged):

    # ---------- FINAL_LIMITATIONS.md ----------
    limitations_md = """# Phase 8 Final Limitations

1. **Modified reproduction.** This project deviates from the original paper's pipeline in several
   explicit, disclosed ways (Strategy C imputation, Elastic Net in place of SVM-RFE, all-non-zero
   feature retention, variable-input SSAE). It is not an exact reproduction and headline performance
   is not directly comparable to the paper's reported 98.2%.
2. **ABIDE multi-site heterogeneity.** The 780-subject cohort spans 20 acquisition sites with
   different scanners/protocols, a well-known source of variance in multi-site fMRI studies.
3. **Site concentration of constant-ROI subjects.** Of the 47 subjects with at least one constant
   ROI, the affected subjects were concentrated at SBL (17/23 subjects, 73.9%) and PITT (13/41,
   31.7%), with smaller counts elsewhere. This is a site-concentrated pattern consistent with a
   possible acquisition/preprocessing effect -- stated here as a hypothesis, not an established cause
   (field-of-view truncation, EPI susceptibility, or scanner protocol were not directly verified in
   this project).
4. **Fold-scoped imputation.** A subset of the model's input values (14,025 of the possible
   subject-feature entries) are training-fold-derived imputed values rather than direct measurements.
5. **Elastic Net feature instability across folds.** Most features (5,384/6,670, 80.7%) were never
   selected in any fold; only 50/6,670 were selected in every fold, consistent with expected
   instability among correlated FC features under L1-type regularization and fold-to-fold sampling
   variation.
6. **Model overfitting.** Final training loss (~3.5e-5) was far below validation loss (~0.03-0.15),
   indicating a substantial train/validation generalization gap in this experiment.
7. **Moderate classification performance.** Accuracy 61.5%+/-2.5%, ROC-AUC 0.677+/-0.024 -- above
   chance-level discrimination but well below the paper's disputed headline figure.
8. **XAI method dependence.** Different XAI methods disagree substantially in which features they
   rank highly; cross-method top-10% agreement in this project's outputs was often partial (most
   pairwise Jaccard values well below 1.0).
9. **ROAR provenance limitation.** Phase 6 did not persist the exact per-feature training-fold
   Integrated Gradients ranking used to order ROAR's feature removal; Phase 7's
   `roar_feature_support.csv` uses saved test-fold Integrated Gradients as a labeled proxy. The
   aggregate ROAR performance curves themselves are direct Phase 6 outputs and remain valid; only the
   individual feature-level attribution of "why a given feature was removed at a given threshold" is
   proxy evidence, not the literal record.
10. **No independent external validation.** All results are from internal 5-fold cross-validation on
    a single dataset; no external/held-out cohort from a different source was used.
11. **No clinical validation.** No clinical outcome, diagnostic-utility trial, or clinician-facing
    evaluation was performed.
12. **No biological/experimental validation.** No wet-lab, imaging-follow-up, or independent
    biological experiment was performed on any identified feature or ROI.
13. **No causal inference.** All feature-selection, XAI, and ROAR evidence is associational/model-
    dependent; none of it establishes causality between an FC connection and ASD status.
14. **No evidence that computationally stable features are biological biomarkers.** Cross-fold
    stability, XAI support, and ROAR association are computational-evidence properties of this
    specific pipeline and dataset; they have not been independently biologically validated.
"""
    (OUT / "reports" / "FINAL_LIMITATIONS.md").write_text(limitations_md, encoding="utf-8")
    (OUT / "final_summary" / "FINAL_CONCLUSION.md").write_text("""# Final Conclusion

The project establishes a fully documented modified computational reproduction in which
functional-connectivity features from ABIDE were evaluated using fold-scoped preprocessing, Elastic
Net selection, a variable-input SSAE, and multiple post-hoc interpretation methods. The resulting
model demonstrated moderate discriminative performance (accuracy 61.5%+/-2.5%, ROC-AUC
0.677+/-0.024), while a subset of FC features (50 of 6,670) showed cross-fold computational stability
and, for 87 features, convergence across Elastic Net and XAI interpretability analyses. These
findings provide computationally reproducible candidates for further investigation, but they do not
establish biological biomarkers, causal mechanisms, or clinical diagnostic utility.

No overall reproducibility score or methodology ranking is presented; see
`reports/REPRODUCIBILITY_AUDIT.csv` for the itemized checklist and
`reports/PAPER_COMPARISON.csv` for the itemized comparison against the original paper.
""", encoding="utf-8")

    # ---------- FINAL_SCIENTIFIC_INTERPRETATION.md ----------
    interp_md = f"""# Final Scientific Interpretation

## A. Supported claims

- A 780-subject ABIDE-I cohort (351 ASD, 429 Control, 20 sites) was constructed after FD>0.2mm
  filtering (Phase 1-2).
- AAL-116 ROI time series were obtained via the CPAC pipeline for the full cohort (Phase 3-4).
- Pearson functional connectivity produced 6,670 lower-triangle features per subject (Phase 5A).
- Strategy C fold-scoped mean imputation was applied to the 14,025 undefined FC entries (Phase 6).
- Elastic Net (fixed hyperparameters, not tuned against test data) selected {N_vals[0]}-{N_vals[2]}
  features per fold across 5 folds: {N_vals}.
- 50 of 6,670 features were selected in all 5 folds; {int(cat_counts.get(4,0))} more in exactly 4/5
  folds.
- Seven independent XAI methods produced model-attribution patterns over the Elastic-Net-selected
  features of each fold.
- ROAR performance changed as attribution-ranked features were progressively removed and the model
  retrained (13 thresholds x 5 folds = 65 runs).

## B. Exploratory claims

- Repeated ROI participation across multiple selected FC connections (see
  `results/phase7_synthesis/roi_analysis/`) -- descriptive, not a ranked "importance" claim.
- Cross-method XAI agreement patterns (support-count distribution: {dict(support_dist)}).
- {n_consensus} "multi-evidence computational feature candidates" (>=4/5-fold Elastic Net selection
  AND >=3/7-method XAI top-10% support) -- convergence among computational analyses, not independent
  validation.
- Anatomical/hemisphere patterns suggested by the selected FC connections (e.g. the 50 fully-stable
  features' left-left/right-right/left-right/midline split) -- a descriptive grouping by AAL naming
  convention only.

## C. Unsupported claims

The project cannot establish, and does not claim:

- Validated biological biomarkers.
- Causal brain mechanisms underlying ASD.
- Clinical diagnostic utility.
- Generalization to unseen hospitals/populations (no external validation was performed).
- Superiority over the original paper.
- Equivalence to the original paper (this is an explicitly modified reproduction).
- Causal importance of any individual FC connection.

## Model performance (Phase 6, authoritative)

{fold_metrics.to_string(index=False)}

Overall (mean +/- SD across 5 folds):

{overall.to_string(index=False)}

Performance is above chance-level discrimination in terms of ROC-AUC; classification accuracy is
approximately 61.5%; performance varies across folds; sensitivity (55.3%+/-5.0%) is lower than
specificity (66.7%+/-1.7%) on average; the model does not reproduce the paper's reported 98.2% result.

## Overfitting audit

Phase 6's SSAE training showed final training loss ~3.5e-5 against validation loss in the ~0.03-0.15
range, with N approximately 427-498 selected features and roughly 530 effective training subjects
after the internal validation split. This pattern indicates a substantial train/validation
generalization gap in this experiment. This is a statement about the observed training/validation
behavior in this experiment, not a claim that the SSAE architecture is inherently invalid.

## Paper-result context

The original paper (Vidya et al., eClinicalMedicine 2025) reported 98.2% accuracy. This project's
prior audit (Phase 4.1) documented an Expression of Concern associated with that paper regarding
SVM-RFE data leakage; that status was not re-independently-verified as current within this phase, so
it is stated here only as: "the project's prior audit documented an Expression of Concern." This
project's substantially lower accuracy does not, by itself, establish that the paper is incorrect, nor
does it establish that this project's approach is superior -- the two pipelines differ in feature
selection method, feature count, missing-value handling, and other respects documented in
`reports/PAPER_COMPARISON.csv`.
"""
    (OUT / "reports" / "FINAL_SCIENTIFIC_INTERPRETATION.md").write_text(interp_md, encoding="utf-8")

    # ---------- ABSTRACT_READY_SUMMARY.md ----------
    abstract_md = f"""# Abstract-Ready Summary

**BACKGROUND.** Autism spectrum disorder (ASD) classification from resting-state fMRI functional
connectivity has been proposed using AAL-116 parcellation, SVM-RFE feature selection, and a stacked
sparse autoencoder (SSAE), with one published pipeline (Vidya et al., 2025) reporting 98.2% accuracy
-- a result subject to a documented Expression of Concern regarding SVM-RFE data leakage.

**METHODS.** This project performed a modified reproduction using 780 ABIDE-I subjects (351 ASD, 429
Control, 20 sites, FD>0.2mm filtered), AAL-116 ROI time series, Pearson functional connectivity
(6,670 features), fold-scoped mean imputation of 14,025 undefined correlation entries (Strategy C),
Fisher-Z transformation, Elastic Net feature selection (fixed hyperparameters, all non-zero
coefficients retained rather than a fixed count of 1,000), and a variable-input SSAE (N->500->100->2)
evaluated under 5-fold subject-level stratified cross-validation. Seven XAI methods and a 65-run ROAR
(RemOve And Retrain) procedure were applied post-hoc for interpretation.

**RESULTS.** Elastic Net selected 427-498 features per fold (mean 464.2, median 463, SD 23.9). Mean
accuracy was 61.5%+/-2.5%, balanced accuracy 61.0%+/-2.7%, sensitivity 55.3%+/-5.0%, specificity
66.7%+/-1.7%, F1 0.563+/-0.038, ROC-AUC 0.677+/-0.024. Training loss (~3.5e-5) was far below
validation loss (~0.03-0.15), indicating overfitting. Fifty of 6,670 features were selected in all 5
folds; 87 features met a pre-defined multi-evidence criterion (>=4/5-fold Elastic Net selection and
>=3/7 XAI-method top-10% support).

**INTERPRETATION.** The model achieved above-chance discrimination but did not reproduce the original
paper's reported accuracy. A small subset of FC features showed convergent computational evidence
across independent selection and interpretability methods, providing candidates for further
investigation.

**LIMITATIONS.** This is a modified, not exact, reproduction; results reflect a single-dataset,
internally cross-validated pipeline with no external or clinical validation; identified features are
multi-evidence computational candidates, not validated biomarkers; ROAR feature-level provenance
relies on a documented proxy for the exact training-fold ranking; site-concentrated constant-ROI
subjects required imputation for a subset of inputs.
"""
    (OUT / "final_summary" / "ABSTRACT_READY_SUMMARY.md").write_text(abstract_md, encoding="utf-8")

    # ---------- PROJECT_SUMMARY.md ----------
    project_md = """# Final Project Summary

**WHAT:** ABIDE ASD-vs-control classification using resting-state fMRI functional connectivity.

**HOW:** AAL-116 ROI extraction -> Pearson functional connectivity (6,670 features) -> fold-scoped
mean imputation of undefined correlations (Strategy C) -> Fisher-Z transformation -> Elastic Net
feature selection (all non-zero features retained, 427-498 per fold) -> variable-input Stacked Sparse
Autoencoder (N->500->100->2, Softmax) -> 5-fold subject-level cross-validation -> seven-method XAI
attribution -> ROAR (RemOve And Retrain) evaluation -> feature/ROI/hemisphere synthesis.

**RESULT:** Approximately 61.5%+/-2.5% accuracy and 0.677+/-0.024 ROC-AUC (5-fold mean+/-SD), with 50
FC features selected in every fold, 87 features meeting a defined multi-evidence criterion across
Elastic Net and XAI, and a documented ROAR performance curve across 13 removal thresholds.

**IMPORTANT:** This is a modified reproduction of a published pipeline, not an exact replication --
Strategy C imputation, Elastic Net (replacing SVM-RFE), all-non-zero feature retention, and a
variable-input SSAE are all explicit, disclosed deviations. The identified FC features are
multi-evidence computational candidates, not validated biological biomarkers; no causal, clinical, or
biological-experimental claims are made or supported by this project.
"""
    (OUT / "final_summary" / "PROJECT_SUMMARY.md").write_text(project_md, encoding="utf-8")

    # ---------- VIVA_QUESTIONS_AND_ANSWERS.md ----------
    viva_md = f"""# Likely Viva Questions and Defensible Answers

**1. Why did you use ABIDE?**
It is the largest publicly available multi-site ASD resting-state fMRI dataset with existing
preprocessed (CPAC) derivatives, and it is the dataset used by the paper being reproduced.

**2. Why FD filtering?**
Framewise displacement filtering (FD>0.2mm) removes subjects with excessive head motion, a well-known
confound in fMRI functional-connectivity analysis; the threshold follows the paper's stated criterion
(Phase 2), applied to this project's own cohort rather than forcing an exact subject count match.

**3. Why AAL-116?**
It is the atlas used by the paper being reproduced and is available as a CPAC derivative
(`rois_aal`), verified against 116 ROIs per subject.

**4. Why Pearson correlation?**
The standard, paper-matching functional-connectivity construction method between ROI mean time
series.

**5. Why Fisher-Z?**
Pearson correlations are bounded in [-1,1] and not normally distributed; Fisher-Z (arctanh)
transformation is standard practice before using correlations as model inputs or in linear models.

**6. Why did undefined correlations occur?**
19 ROIs had zero-variance (constant) signal in a subset of subjects (47 subjects total), making the
Pearson correlation formula's denominator zero for those ROI pairs -- an undefined, not merely
missing, value (Phase 4.1).

**7. Why did you choose Strategy C?**
After a documented sensitivity comparison of subject exclusion, ROI exclusion, and imputation
(Phase 4.2/5B/5C), fold-scoped imputation (Strategy C) was chosen to preserve the full cohort and
full ROI set while keeping the handling leakage-safe.

**8. Why fold-scoped imputation specifically?**
Imputing with a statistic computed only from each fold's training data prevents information from the
test fold from influencing the imputed values, avoiding data leakage.

**9. Why Elastic Net?**
It provides a well-understood, leakage-controllable feature-selection alternative to SVM-RFE, with L1
sparsity plus L2 stability, and fixed a priori hyperparameters (not tuned against test data).

**10. Why not SVM-RFE?**
SVM-RFE is the paper's original method, but the paper carries a documented Expression of Concern
regarding SVM-RFE-related data leakage (per this project's Phase 4.1 audit trail); Elastic Net was
chosen as a disclosed, leakage-controlled substitute.

**11. Why keep all non-zero features instead of forcing exactly 1,000?**
Forcing exactly 1,000 features would mean either arbitrarily discarding informative non-zero
coefficients or arbitrarily including uninformative zero coefficients; retaining all non-zero
coefficients respects what Elastic Net's regularization actually determined to be informative
(Phase 5D).

**12. Why does feature count vary across folds?**
Elastic Net is fit independently per fold on that fold's training data only (leakage-safe), so
different training samples naturally yield different non-zero coefficient sets and counts (427-498).

**13. Why variable-input SSAE?**
A direct, necessary consequence of the all-non-zero feature-selection decision; each fold already
trains an independent model instance under proper cross-validation.

**14. Why use seven XAI methods?**
Different XAI methods make different assumptions (gradient-based, perturbation-based,
game-theoretic); using seven methods and reporting them separately, plus their cross-method
agreement, avoids overstating the reliability of any single method's attributions.

**15. What does XAI actually tell you?**
Which input features the trained model's output was most sensitive to, under a specific attribution
method's definition of "importance" -- a property of the model and method, not independent evidence
about the brain.

**16. What does ROAR tell you?**
How the retrained model's performance changes as attribution-ranked features are progressively
removed -- evidence of the model's dependence on those features under this specific procedure, not
proof of causal or biological importance.

**17. Why is the accuracy only ~61.5%?**
Possible contributing factors documented in this project include: ABIDE's known multi-site
heterogeneity, the modified (non-paper) feature-selection and imputation methodology, observed
overfitting during SSAE training, and the inherent difficulty of the ASD-vs-control classification
task from resting-state FC alone. No single cause is established as dominant.

**18. What evidence of overfitting did you observe?**
Final training loss (~3.5e-5) far below validation loss (~0.03-0.15) during SSAE fine-tuning across
folds (Phase 6 training_summary.csv).

**19. Why is your result different from the paper?**
The pipelines differ in feature-selection method (Elastic Net vs SVM-RFE), number/variability of
selected features (427-498 variable vs fixed 1,000), and undefined-FC handling (Strategy C imputation
vs undocumented in the paper) -- see `reports/PAPER_COMPARISON.csv` for the full itemized comparison.
The paper's reported figure is also associated with a documented Expression of Concern regarding
SVM-RFE leakage.

**20. Can you call your selected features biomarkers?**
No. They are, at most, "multi-evidence computational feature candidates" -- features with convergent
statistical selection and model-attribution evidence within this pipeline. No independent biological
or clinical validation was performed.

**21. What are your main limitations?**
See `reports/FINAL_LIMITATIONS.md` -- modified (not exact) reproduction, site-concentrated
constant-ROI subjects, fold-scoped imputation, Elastic Net instability, overfitting, moderate
performance, XAI method disagreement, the ROAR provenance/proxy limitation, and the absence of
external, clinical, or biological validation.

**22. How would you improve the study in future work?**
Possible directions (not undertaken here): external cohort validation, addressing the observed
overfitting (e.g. regularization/architecture changes), persisting exact per-feature ROAR removal
provenance, and any biological/clinical follow-up validation of the multi-evidence candidates -- none
of which was performed in this project.
"""
    (OUT / "viva" / "VIVA_QUESTIONS_AND_ANSWERS.md").write_text(viva_md, encoding="utf-8")

    # ---------- PHASE8_FINAL_REPORT.md ----------
    final_report_md = f"""# Phase 8 Final Report

**PHASE 8 STATUS: {status}**

This phase performed a read-only synthesis and audit of Phases 1-7. No retraining, hyperparameter
tuning, cohort change, or methodology change occurred.

## Contents

- `FINAL_SCIENTIFIC_INTERPRETATION.md` -- supported / exploratory / unsupported claims, performance,
  overfitting audit, paper-result context.
- `FINAL_LIMITATIONS.md` -- complete limitations list.
- `CLAIM_AUDIT.csv` -- terminology audit of Phase 6-7 reports for biomarker/causal/validated language.
- `REPRODUCIBILITY_AUDIT.csv` -- itemized COMPLETE/PARTIAL/MISSING checklist (no overall score).
- `PAPER_COMPARISON.csv` -- itemized comparison against the original paper.
- `../final_summary/ABSTRACT_READY_SUMMARY.md`, `PROJECT_SUMMARY.md`, `FINAL_CONCLUSION.md`.
- `../viva/VIVA_QUESTIONS_AND_ANSWERS.md`.
- `../data_audit/` -- provenance and methodological-decision audit tables.

Problems encountered: {problems if problems else 'None.'}
Phases 1-7 confirmed unchanged: {guard_unchanged}.
"""
    (OUT / "reports" / "PHASE8_FINAL_REPORT.md").write_text(final_report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
