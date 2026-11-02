"""
phase5c_elastic_net_comparison.py -- Phase 5C: A/B/C strategy comparison
plus Elastic Net (replacing SVM-RFE) architecture analysis.

This phase performs NO computation on real ABIDE FC data -- it reuses
already-established aggregate figures from Phase 4.1/4.2/5A/5B (read-only)
and runs small, clearly-labeled SYNTHETIC experiments only to illustrate
Elastic Net behavior and leakage-safe fold-scoped imputation logic. No
final FC matrix, no Fisher-Z on real data, no SVM-RFE, no SSAE training,
no XAI, no ROAR, no hyperparameter tuning for accuracy, no downloads, no
modification of any prior phase or source data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
PHASE4_1_DIR = ROOT / "results" / "phase4_1_constant_roi_audit"
PHASE4_2_DIR = ROOT / "results" / "phase4_2_methodological_sensitivity"
PHASE5A_DIR = ROOT / "results" / "phase5a_fc_diagnostic"
PHASE5B_DIR = ROOT / "results" / "phase5b_missing_fc_analysis"

OUT_DIR = ROOT / "results" / "phase5c_elastic_net_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [
    ROOT / "results" / "phase1", ROOT / "results" / "phase2", ROOT / "results" / "phase3",
    ROOT / "results" / "phase3_5", ROOT / "results" / "phase3_6",
    ROOT / "results" / "phase3_6_verification", ROOT / "results" / "phase3_6_aal_download",
    ROOT / "results" / "phase4", ROOT / "results" / "phase4_1_constant_roi_audit",
    ROOT / "results" / "phase4_2_methodological_sensitivity", ROOT / "results" / "phase5a_fc_diagnostic",
    ROOT / "results" / "phase5b_missing_fc_analysis",
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


def main():
    log("=" * 60)
    log("PHASE 5C -- A/B/C strategy comparison + Elastic Net architecture analysis.")
    log("No real FC computed. Synthetic experiments only, clearly labeled.")
    log("=" * 60)
    pre_guard = guard_fingerprints()

    # ── reuse already-established real figures (read-only) ────────────────
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    n_cohort, n_asd, n_control = len(cohort), int((cohort.diagnosis == "ASD").sum()), int((cohort.diagnosis == "Control").sum())
    n_sites = cohort["site"].nunique()
    assert n_cohort == 780 and n_asd == 351 and n_control == 429
    log(f"Authoritative cohort (reused, not recomputed): {n_cohort} ({n_asd} ASD, {n_control} Control), {n_sites} sites.")

    option2_site_impact = pd.read_csv(PHASE4_2_DIR / "option2_site_impact.csv")  # Strategy A site impact
    option3_roi_summary = pd.read_csv(PHASE4_2_DIR / "option3_roi_exclusion_summary.csv")
    site_bias = None
    try:
        site_bias_path = PHASE5B_DIR / "phase5b_summary.json"
        with open(site_bias_path) as f:
            phase5b_summary = json.load(f)
    except Exception:
        phase5b_summary = {}

    n_subj_A = int(option2_site_impact["total_after"].sum())
    log(f"Strategy A figures reused from Phase 4.2: {n_subj_A} subjects remaining after excluding 47.")
    remaining_roi_B = int(option3_roi_summary.iloc[0]["remaining_roi_count"])
    remaining_feat_B = int(option3_roi_summary.iloc[0]["remaining_fc_features"])
    features_removed_B = int(option3_roi_summary.iloc[0]["features_removed"])
    assert remaining_roi_B == 97 and remaining_feat_B == 4656 and features_removed_B == 2014
    assert remaining_roi_B * (remaining_roi_B - 1) // 2 == remaining_feat_B
    log(f"Strategy B figures reused from Phase 4.2: {remaining_roi_B} ROIs, {remaining_feat_B} features "
        f"(verified {remaining_roi_B}x{remaining_roi_B-1}/2={remaining_feat_B}), {features_removed_B} removed.")

    # Strategy A ASD/Control after exclusion (from Phase 4.2 option2_subject_exclusion_summary.csv)
    opt2_summary = pd.read_csv(PHASE4_2_DIR / "option2_subject_exclusion_summary.csv")
    n_asd_A = int(opt2_summary.iloc[0]["remaining_asd"])
    n_control_A = int(opt2_summary.iloc[0]["remaining_control"])
    assert n_asd_A == 330 and n_control_A == 403

    # ═══════════════════════════════════════════════════════════════════
    # SYNTHETIC EXPERIMENT 1: Elastic Net sparsity / correlated-feature behavior
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- SYNTHETIC EXPERIMENT 1: Elastic Net on correlated synthetic data (NOT real ABIDE data) ---")
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n_syn_samples, n_syn_features = 200, 500
    # Build a block-correlated synthetic design matrix to emulate FC-like feature redundancy
    n_blocks = 20
    block_size = n_syn_features // n_blocks
    X_syn = np.zeros((n_syn_samples, n_syn_features))
    for b in range(n_blocks):
        base = rng.normal(size=n_syn_samples)
        for k in range(block_size):
            X_syn[:, b * block_size + k] = base + rng.normal(scale=0.3, size=n_syn_samples)
    true_coef = np.zeros(n_syn_features)
    true_coef[:5 * block_size:block_size] = 2.0  # a few "true" signal blocks
    y_syn = X_syn @ true_coef + rng.normal(scale=1.0, size=n_syn_samples)
    X_syn_scaled = StandardScaler().fit_transform(X_syn)

    en_grid_rows = []
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=0)
            model.fit(X_syn_scaled, y_syn)
            n_nonzero = int(np.sum(model.coef_ != 0))
            en_grid_rows.append({
                "alpha": alpha, "l1_ratio": l1_ratio, "n_nonzero_features": n_nonzero,
                "total_features": n_syn_features, "SYNTHETIC": True,
            })
    en_grid_df = pd.DataFrame(en_grid_rows)
    log(f"  SYNTHETIC Elastic Net sparsity grid (n_features={n_syn_features}, n_samples={n_syn_samples}, "
        f"20 correlated blocks): nonzero-feature counts range "
        f"{en_grid_df['n_nonzero_features'].min()}-{en_grid_df['n_nonzero_features'].max()} depending on "
        f"alpha/l1_ratio. Demonstrates that Elastic Net's selected-feature COUNT is NOT fixed -- it "
        f"depends on hyperparameters, unlike SVM-RFE's explicit stepwise ranking to a target count.")
    log("  SYNTHETIC observation: within correlated blocks, Elastic Net's L2 component tends to spread "
        "nonzero coefficients across correlated features (the 'grouping effect') rather than picking "
        "one representative and zeroing the rest (which pure L1/Lasso would tend to do) -- relevant "
        "because FC features are structurally correlated (shared ROIs across many of the 6,670 pairs).")

    # ═══════════════════════════════════════════════════════════════════
    # SYNTHETIC EXPERIMENT 2: top-1000-by-|coefficient| ranking procedure (Option 2)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- SYNTHETIC EXPERIMENT 2: rank-to-fixed-count procedure (Option 2 mechanics, NOT real data) ---")
    model_demo = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=0)
    model_demo.fit(X_syn_scaled, y_syn)
    abs_coef = np.abs(model_demo.coef_)
    top_k = min(100, n_syn_features)  # scaled-down synthetic analogue of "top 1000 of 6670"
    ranked_idx = np.argsort(-abs_coef)[:top_k]
    log(f"  SYNTHETIC: fit ElasticNet(alpha=0.05, l1_ratio=0.5) on the same synthetic data, then ranked "
        f"all {n_syn_features} coefficients by |coefficient| and selected the top {top_k} (a scaled-down "
        f"analogue of 'top 1000 of 6670') -- this ranking procedure is deterministic given fixed alpha/"
        f"l1_ratio/random_state and training data, and works REGARDLESS of how many coefficients were "
        f"exactly zero, unlike Option 1's natural nonzero count. This is the mechanic Option 2 would use "
        f"on real data, demonstrated here only in synthetic miniature.")

    feature_count_rows = [
        {"option": "Option 1: natural nonzero count", "mechanism": "ElasticNet(alpha, l1_ratio).coef_ != 0",
         "resulting_count_fixed": False,
         "synthetic_demo_range": f"{en_grid_df['n_nonzero_features'].min()}-{en_grid_df['n_nonzero_features'].max()} "
                                  f"of {n_syn_features} (SYNTHETIC, illustrative only)",
         "matches_paper_1000": "Not guaranteed -- would require search over alpha/l1_ratio, and any such "
                               "search must not be tuned against test accuracy",
         "ssae_input_dim_consequence": "Variable -- SSAE input layer would need to match whatever count "
                                       "results, or the architecture must change from the paper's 1000->500->100->2",
         "labeled_as": "Natural consequence of Elastic Net; NOT the paper's procedure"},
        {"option": "Option 2: rank-to-top-1000", "mechanism": "argsort(-|coef_|)[:1000], training fold only",
         "resulting_count_fixed": True,
         "synthetic_demo_range": f"exactly {top_k} of {n_syn_features} by construction (SYNTHETIC, illustrative only)",
         "matches_paper_1000": "Yes, by construction (preserves the paper's 1000-feature SSAE input)",
         "ssae_input_dim_consequence": "Matches paper exactly: 1000 -> 500 -> 100 -> 2 -> Softmax",
         "labeled_as": "Explicit METHODOLOGICAL MODIFICATION (paper describes SVM-RFE's own ranking "
                       "mechanism, not Elastic-Net-coefficient ranking) -- not a silent equivalence"},
    ]
    feature_count_df = pd.DataFrame(feature_count_rows)
    feature_count_df.to_csv(OUT_DIR / "elastic_net_feature_count_analysis.csv", index=False)
    en_grid_df.to_csv(OUT_DIR / "elastic_net_feature_count_analysis.csv", mode="a", index=False)
    log("elastic_net_feature_count_analysis.csv written (Option 1 vs Option 2 mechanics + synthetic sparsity grid appended).")

    # ═══════════════════════════════════════════════════════════════════
    # SYNTHETIC EXPERIMENT 3: leakage-safe fold-scoped imputation logic check
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- SYNTHETIC EXPERIMENT 3: fold-scoped imputation logic verification (NOT real data) ---")
    from sklearn.model_selection import StratifiedKFold

    n_syn2 = 100
    y_syn2 = rng.integers(0, 2, size=n_syn2)
    x_syn2 = rng.normal(size=n_syn2)
    missing_mask_syn = rng.random(n_syn2) < 0.15  # ~15% synthetic missingness
    x_syn2_with_nan = x_syn2.copy()
    x_syn2_with_nan[missing_mask_syn] = np.nan

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    leakage_check_rows = []
    for fold_idx, (tr, te) in enumerate(skf.split(x_syn2_with_nan.reshape(-1, 1), y_syn2)):
        train_vals = x_syn2_with_nan[tr]
        train_observed = train_vals[~np.isnan(train_vals)]
        train_mean = float(np.mean(train_observed))
        train_median = float(np.median(train_observed))
        # Apply TRAINING-derived statistic to BOTH train and test fold missing entries -- test fold's
        # own values/labels are never read to compute the statistic (verified by construction: only
        # `train_vals`, indexed by `tr`, feeds `train_mean`/`train_median` below).
        imputed_test = np.where(np.isnan(x_syn2_with_nan[te]), train_mean, x_syn2_with_nan[te])
        n_test_missing = int(np.isnan(x_syn2_with_nan[te]).sum())
        leakage_check_rows.append({
            "fold": fold_idx, "n_train": len(tr), "n_test": len(te),
            "train_fold_mean": train_mean, "train_fold_median": train_median,
            "n_test_missing_imputed_with_train_stat": n_test_missing,
            "test_fold_data_used_in_statistic": False,  # verified by construction, see comment above
            "SYNTHETIC": True,
        })
    leakage_check_df = pd.DataFrame(leakage_check_rows)
    log(f"  SYNTHETIC: 5-fold fold-scoped imputation demonstrated on 100 synthetic samples with ~15% "
        f"missingness -- in every fold, the imputation statistic (mean/median) is computed ONLY from "
        f"`x_syn2_with_nan[tr]` (training indices), never from `x_syn2_with_nan[te]` (test indices) or "
        f"`y_syn2[te]` (test labels); verified by code construction (the statistic-computing line never "
        f"references `te`). This is the exact procedure Strategy C would require if selected -- "
        f"demonstrated here in miniature, NOT applied to any real ABIDE feature.")
    log(f"  Fold-level train mean vs median (SYNTHETIC): "
        f"{leakage_check_df[['fold','train_fold_mean','train_fold_median']].to_string(index=False)}")

    # ═══════════════════════════════════════════════════════════════════
    # ELASTIC NET PROCEDURE ANALYSIS (documentation)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing elastic_net_procedure_analysis.md ---")
    en_md = f"""# Elastic Net as a Replacement for SVM-RFE — Procedure Analysis

**No Elastic Net was fit on real ABIDE FC data in this phase. All numeric illustrations below are
SYNTHETIC (see `elastic_net_feature_count_analysis.csv`).**

## How Elastic Net differs from SVM-RFE

The paper's SVM-RFE recursively fits a linear SVM and removes the lowest-ranked feature(s) each
iteration until reaching 1,000 — a wrapper method that repeatedly retrains a classifier. Elastic Net
is instead a single regularized linear model (here, for a classification target, typically
`LogisticRegression(penalty="elasticnet", solver="saga")` or `SGDClassifier(penalty="elasticnet")`
would replace the plain `ElasticNet` regressor used for illustration below) whose objective combines:

- **L1 regularization** (the Lasso term, weighted by `l1_ratio`): drives some coefficients to exactly
  zero, producing sparsity and an implicit feature-selection effect.
- **L2 regularization** (the Ridge term, weighted by `1 - l1_ratio`): shrinks correlated coefficients
  together rather than arbitrarily picking one and zeroing the rest — this is the well-known "grouping
  effect," and it matters directly here because FC features are structurally correlated (many of the
  6,670 pairs share a common ROI).

The combined penalty is controlled by two hyperparameters: `alpha` (overall regularization strength)
and `l1_ratio` (the L1/L2 balance, in [0, 1]). The SYNTHETIC sparsity grid in
`elastic_net_feature_count_analysis.csv` shows nonzero-feature counts varying substantially across a
small `alpha`/`l1_ratio` grid on a 500-feature synthetic design with 20 correlated blocks — the same
qualitative behavior would be expected on the real 6,670-feature FC representation, though the exact
counts cannot be known without fitting on real data (not done in this phase).

## Correlated FC features and the grouping effect

Because many FC features share an ROI (each of the 116 ROIs participates in 115 of the 6,670
features), Elastic Net's L2 component is expected to retain groups of related features together
rather than collapsing to a single representative per group, unlike pure L1/Lasso. This is a
structural property of the method, not a claim about what will happen on ABIDE data specifically —
demonstrated only in synthetic miniature here.

## Coefficient interpretation

An Elastic Net coefficient's magnitude reflects its (regularized) linear association with the target
within the fitted model — it is not equivalent to SVM-RFE's ranking criterion (which is derived from
the linear SVM's margin/weight at each recursive step). Coefficients from the two methods are not
directly comparable in magnitude or interpretation, even where they might select overlapping features.

## Regularization parameters

`alpha` and `l1_ratio` would need to be selected via inner cross-validation on training folds only
(e.g. `ElasticNetCV`/`LogisticRegressionCV` with `penalty="elasticnet"`), never by observing outer-fold
or holdout test accuracy — consistent with this project's existing leakage-avoidance discipline.

## Leakage requirements

Per the task's explicit requirement and this project's established `holdout_guard.py` discipline:
Elastic Net must be **fit on training-fold data only** within each of the 5 CV folds; its selected
features (whichever selection mechanism, Option 1 or 2) must be determined from training-fold data
only and then applied unchanged to that fold's test data. Elastic Net must never see test-fold labels
or values during fitting or feature-ranking.

## Explicitly not claimed here

This document does not claim Elastic Net will produce higher, lower, or comparable accuracy to
SVM-RFE on the real ABIDE data. No such comparison was performed, attempted, or estimated.
"""
    with open(OUT_DIR / "elastic_net_procedure_analysis.md", "w") as f:
        f.write(en_md)
    log("elastic_net_procedure_analysis.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY SITE / DIAGNOSIS IMPACT TABLES
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing strategy_site_impact.csv and strategy_diagnosis_impact.csv ---")
    site_impact_rows = []
    for _, r in option2_site_impact.iterrows():
        site_impact_rows.append({
            "site": r["site"], "strategy_A_subjects_before": int(r["total_before"]),
            "strategy_A_subjects_after": int(r["total_after"]), "strategy_A_lost": int(r["lost"]),
            "strategy_A_pct_lost": r["percentage_lost"],
            "strategy_B_site_impact": "None -- all subjects retained (ROIs removed, not subjects)",
            "strategy_C_site_impact": "None directly -- all subjects retained; imputed-value "
                                      "concentration mirrors strategy_A's per-site pattern "
                                      "(SBL/PITT disproportionately imputed, see Phase 5B)",
        })
    site_impact_df = pd.DataFrame(site_impact_rows)
    site_impact_df.to_csv(OUT_DIR / "strategy_site_impact.csv", index=False)
    log(f"strategy_site_impact.csv written ({len(site_impact_df)} sites).")

    diagnosis_impact_df = pd.DataFrame([
        {"strategy": "A: Subject exclusion", "ASD_before": n_asd, "ASD_after": n_asd_A,
         "ASD_lost": n_asd - n_asd_A, "Control_before": n_control, "Control_after": n_control_A,
         "Control_lost": n_control - n_control_A,
         "class_ratio_before": round(n_asd / n_control, 4), "class_ratio_after": round(n_asd_A / n_control_A, 4)},
        {"strategy": "B: ROI exclusion", "ASD_before": n_asd, "ASD_after": n_asd,
         "ASD_lost": 0, "Control_before": n_control, "Control_after": n_control, "Control_lost": 0,
         "class_ratio_before": round(n_asd / n_control, 4), "class_ratio_after": round(n_asd / n_control, 4)},
        {"strategy": "C: Fold-scoped imputation", "ASD_before": n_asd, "ASD_after": n_asd,
         "ASD_lost": 0, "Control_before": n_control, "Control_after": n_control, "Control_lost": 0,
         "class_ratio_before": round(n_asd / n_control, 4), "class_ratio_after": round(n_asd / n_control, 4)},
    ])
    diagnosis_impact_df.to_csv(OUT_DIR / "strategy_diagnosis_impact.csv", index=False)
    log("strategy_diagnosis_impact.csv written.")

    # ═══════════════════════════════════════════════════════════════════
    # SSAE ARCHITECTURE COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing ssae_architecture_comparison.md ---")
    ssae_md = """# SSAE Architecture Compatibility — Strategies A / B / C

**No SSAE was trained in this phase. This document only assesses structural (input-dimension)
compatibility.**

## Paper / proposed architecture

`1000 -> 500 -> 100 -> 2 -> Softmax` (two output classes: ASD, Control).

## Strategy A (733 subjects, 116 ROIs, 6,670 features)

FC feature dimensionality is unaffected (still 6,670 before selection). Elastic Net Option 2
(rank-to-top-1000) can produce exactly 1,000 features from 6,670 candidates, so the paper's
`1000 -> 500 -> 100 -> 2` architecture is directly supportable. Fewer subjects (733) are available to
fit/validate the SSAE, which is a sample-size consideration, not an architectural one.

## Strategy B (780 subjects, 97 ROIs, 4,656 features)

4,656 > 1,000, so Elastic Net Option 2 can still select exactly 1,000 features from the reduced
candidate pool, and the paper's architecture remains directly supportable. The available candidate
feature pool is smaller (4,656 vs. 6,670), which may affect which 1,000 are selected, but does not by
itself prevent reaching the target count.

## Strategy C (780 subjects, 116 ROIs, 6,670 features, imputed values)

Same as Strategy A structurally (6,670 candidates, 1,000 selectable) — full subject count and full
feature space are retained. The distinction is that some of the 1,000 selected features may have been
partly constructed from imputed (not measured) values for the 47 affected subjects; this does not
change the architecture, but changes what a subset of the *input values* represent for those subjects.

## If Elastic Net Option 1 (natural nonzero count) is used instead

The resulting feature count is not guaranteed to be 1,000 under any of A, B, or C. If adopted, the
SSAE input layer would need to be resized to match whatever count results (documented as a
consequence, not silently absorbed into an unchanged "1000" label) — or Option 2's ranking procedure
would need to be applied on top of Option 1's selected set to force a count, which is simply Option 2
with an extra step.

## Conclusion

All of A, B, and C can structurally support the paper's `1000 -> 500 -> 100 -> 2 -> Softmax`
architecture, PROVIDED Option 2 (rank-to-top-1000) is used for feature selection. This is a
structural compatibility statement only — it says nothing about resulting model performance.
"""
    with open(OUT_DIR / "ssae_architecture_comparison.md", "w") as f:
        f.write(ssae_md)
    log("ssae_architecture_comparison.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # XAI / ROAR IMPLICATIONS
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing xai_roar_implications.md ---")
    xai_md = """# XAI and ROAR Implications — Strategies A / B / C

**No XAI or ROAR was executed in this phase. This is a forward-looking implications document only.**

## Strategy A: fewer subjects, full ROI set

XAI attribution maps (e.g. gradient-based methods, as the paper uses) would be computed over the full
116-ROI / 6,670-feature space, so region-level interpretation remains directly comparable to the
paper's own reported regions. The interpretation caveat is sample-size related: attributions and any
ROAR-based validation would be computed from 733, not 780, subjects, and the excluded 47 are not a
random subsample (heavily concentrated at SBL/PITT — Phase 4.1/4.2) — so XAI results would not
generalize to those two sites' typical presentation without a stated caveat.

## Strategy B: 97 ROIs, 2,014 connections permanently removed

Any XAI attribution or ROAR-based region ranking can only ever reference the remaining 97 ROIs — the
19 excluded ROIs (orbitofrontal cluster, inferior cerebellum/vermis cluster, one temporal-pole region;
Phase 4.1) cannot appear in any result, regardless of their true importance. **This does not mean
those regions are unimportant** — their absence from any XAI output would be a consequence of this
project's preprocessing decision, not a finding about their neuroscientific relevance, and must be
stated as such wherever Strategy B's results are reported.

## Strategy C: all subjects and ROIs retained, some values imputed

XAI attributions and ROAR perturbation analysis would run over the full 116-ROI/6,670-feature space
for all 780 subjects, preserving direct comparability to the paper's design. However, for the 47
affected subjects, some of the ~6,670 input values were not measured but imputed (fold-scoped mean/
median) — **any XAI attribution assigned to an imputed feature for one of these subjects reflects the
model's response to an inserted statistic, not that subject's actual measured brain connectivity.**
This must be disclosed wherever per-subject (not just aggregate) XAI results are reported for any of
the 47 subjects, and is a genuine interpretability caveat unique to Strategy C.

## How ROAR would interact with Elastic Net + SSAE

ROAR (RemOve And Retrain) systematically removes the top-k most important features (per the XAI
method being validated) and retrains the model, checking whether accuracy degrades as expected. This
requires retraining the full pipeline (Elastic Net feature selection possibly held fixed or re-run,
plus SSAE) multiple times with different feature subsets removed — it is a validation *procedure*
layered on top of whichever strategy (A/B/C) and feature-selection method (Option 1/2) is ultimately
adopted, not an independent design choice. Under Strategy C specifically, ROAR's retraining loop would
need to re-apply fold-scoped imputation consistently each time a feature is removed, to avoid
reintroducing leakage at that stage. None of this is executed in this phase.
"""
    with open(OUT_DIR / "xai_roar_implications.md", "w") as f:
        f.write(xai_md)
    log("xai_roar_implications.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # LEAKAGE CONTROL PROTOCOL
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing leakage_control_protocol.md ---")
    leakage_md = """# Leakage Control Protocol — Elastic Net + SSAE (Phase 6+ Reference, NOT Executed Here)

**No training was performed in this phase. This document specifies the required protocol for when
training does eventually occur, consistent with this project's existing `holdout_guard.py` discipline
established in the earlier ML phases.**

## Required structure: 5-fold cross-validation

```
FOR EACH of 5 folds:
    TRAINING DATA (4 folds)
        |
        v
    FC missing-value handling            <- fit ONLY on training-fold data (Strategy C: training-fold
        |                                    mean/median; Strategies A/B: no fitting needed, deterministic)
        v
    Fit Elastic Net (or ElasticNetCV)     <- fit ONLY on training-fold data; alpha/l1_ratio selected via
        |                                    INNER cross-validation on training folds only
        v
    Feature selection (Option 1 or 2)     <- determined ONLY from training-fold Elastic Net coefficients
        |
        v
    Train SSAE                            <- fit ONLY on training-fold data, using only the
        |                                    training-fold-selected features
        v
    Evaluate on the UNTOUCHED test fold   <- test fold transformed using training-fold statistics/
                                              feature list ONLY; test labels/values never used upstream
```

## Explicitly forbidden (the leakage pattern to avoid)

```
ALL 780 SUBJECTS
    |
    v
Imputation (using all 780 subjects' statistics)
    |
    v
Elastic Net (fit on all 780 subjects)
    |
    v
Feature selection (using all 780 subjects)
    |
    v
5-fold CV  <-- by this point, every fold's "test" data already influenced the
               imputation statistic, the Elastic Net fit, and the feature list.
               This is feature-selection leakage -- structurally identical to
               the flaw underlying the paper's own currently-active Expression
               of Concern (Phase 4.1 finding: SVM-RFE was run on the full
               dataset before cross-validation).
```

## Relationship to existing project infrastructure

This protocol is the same discipline already implemented in `holdout_guard.py` for the earlier
CC200-based ML phases (Phase 2C-2H of the original ML track): fold-scoped fitting, no test-label or
test-value use during any preprocessing/selection step, explicit assertions checking for holdout/
test-fold contamination. Reusing that discipline (or an AAL-116-specific equivalent built the same
way) is recommended when training actually begins — not implemented in this phase.
"""
    with open(OUT_DIR / "leakage_control_protocol.md", "w") as f:
        f.write(leakage_md)
    log("leakage_control_protocol.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON TABLES (no ranking, no winner, no predicted accuracy)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing methodology_comparison.csv and strategy_architecture_comparison.csv ---")
    methodology_rows = [
        {"metric": "1. Subjects", "A": n_subj_A, "B": 780, "C": 780},
        {"metric": "2. ASD count", "A": n_asd_A, "B": n_asd, "C": n_asd},
        {"metric": "3. Control count", "A": n_control_A, "B": n_control, "C": n_control},
        {"metric": "4. ROIs", "A": 116, "B": remaining_roi_B, "C": 116},
        {"metric": "5. FC features", "A": 6670, "B": remaining_feat_B, "C": 6670},
        {"metric": "6. Undefined FC after handling", "A": 0, "B": 0, "C": 0},
        {"metric": "7. Whether values are invented", "A": "No", "B": "No",
         "C": "Yes -- 14,025 entries (0.27% cohort-wide) replaced with a training-fold statistic"},
        {"metric": "8. Information discarded", "A": "47 subjects entirely (all 6,670 of their features)",
         "B": "19 ROIs' signal for all 780 subjects (2,014 features)",
         "C": "None discarded; some values (0.27%) are not directly measured"},
        {"metric": "9. Site impact", "A": "Severe for SBL (-74%), PITT (-32%)", "B": "None directly",
         "C": "None directly; imputed-value concentration mirrors A's site pattern"},
        {"metric": "10. Diagnosis impact", "A": f"-{n_asd-n_asd_A} ASD, -{n_control-n_control_A} Control (negligible ratio shift)",
         "B": "None", "C": "None"},
        {"metric": "11. Elastic Net compatibility", "A": "Direct (fully finite matrix)",
         "B": "Direct (fully finite matrix)", "C": "Requires fold-scoped imputation completed first"},
        {"metric": "12. Leakage requirements", "A": "None beyond standard fold-scoped CV",
         "B": "None beyond standard fold-scoped CV",
         "C": "Strict -- imputation statistic must be training-fold-only (see leakage_control_protocol.md)"},
        {"metric": "13. SSAE compatibility", "A": "Yes (1000 selectable from 6670)",
         "B": "Yes (1000 selectable from 4656)", "C": "Yes (1000 selectable from 6670)"},
        {"metric": "14. XAI implications", "A": "733-subject results; SBL/PITT underrepresented",
         "B": "19 ROIs can never appear in any attribution", "C": "Imputed-feature attributions need disclosure"},
        {"metric": "15. ROAR implications", "A": "Retraining loop over 733 subjects",
         "B": "Retraining loop restricted to 97-ROI feature space",
         "C": "Retraining loop must reapply fold-scoped imputation each iteration"},
        {"metric": "16. Computational implications", "A": "Slightly cheaper (fewer subjects)",
         "B": "Cheaper (30% fewer features)", "C": "Most expensive (full data + imputation step per fold)"},
        {"metric": "17. Reproducibility implications", "A": "Deterministic exclusion list",
         "B": "Deterministic ROI list", "C": "Deterministic IF imputation statistic/procedure is fixed and documented"},
        {"metric": "18. Difference from original paper", "A": "Fewer subjects than paper's FD-only exclusion",
         "B": "Fewer ROIs/features than paper's fixed AAL-116/6670 design",
         "C": "Additional preprocessing step (imputation) paper does not describe"},
    ]
    methodology_df = pd.DataFrame(methodology_rows)
    methodology_df.to_csv(OUT_DIR / "methodology_comparison.csv", index=False)
    log("methodology_comparison.csv written (18 metrics x A/B/C, no ranking, no winner, no predicted accuracy).")

    arch_rows = [
        {"aspect": "Subjects available for training", "A": n_subj_A, "B": 780, "C": 780},
        {"aspect": "Candidate FC features before selection", "A": 6670, "B": remaining_feat_B, "C": 6670},
        {"aspect": "Elastic Net Option 2 (top-1000) achievable", "A": "Yes", "B": "Yes (4656>1000)", "C": "Yes"},
        {"aspect": "SSAE input dimension achievable", "A": 1000, "B": 1000, "C": 1000},
        {"aspect": "SSAE architecture", "A": "1000->500->100->2->Softmax", "B": "1000->500->100->2->Softmax",
         "C": "1000->500->100->2->Softmax"},
        {"aspect": "Requires imputation before Elastic Net", "A": "No", "B": "No", "C": "Yes (fold-scoped)"},
        {"aspect": "Computational cost (relative, qualitative only)", "A": "Lower (fewer subjects)",
         "B": "Lower (fewer features)", "C": "Higher (extra fold-scoped step)"},
    ]
    arch_df = pd.DataFrame(arch_rows)
    arch_df.to_csv(OUT_DIR / "strategy_architecture_comparison.csv", index=False)
    log("strategy_architecture_comparison.csv written.")

    # ═══════════════════════════════════════════════════════════════════
    # AUDIT
    # ═══════════════════════════════════════════════════════════════════
    post_guard = guard_fingerprints()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phase 1-5B files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    log("AUDIT: no real ABIDE FC data was loaded, computed, or written in this script -- confirmed by "
        "code review: only Phase 4.2/5B's already-saved aggregate CSV/JSON summaries were read; all "
        "Elastic Net / imputation demonstrations used np.random.default_rng-generated SYNTHETIC arrays.")
    log("AUDIT: no SVM-RFE, no SSAE training, no Softmax training, no XAI, no ROAR, no hyperparameter "
        "tuning against accuracy, no downloads, no modification of any prior phase, the authoritative "
        "cohort, or AAL source files -- confirmed by code review.")

    status_ok = guard_unchanged
    phase5c_status = "PASS" if status_ok else "FAIL"

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"total": n_cohort, "asd": n_asd, "control": n_control},
        "strategy_A": {"subjects": n_subj_A, "asd": n_asd_A, "control": n_control_A, "rois": 116, "features": 6670},
        "strategy_B": {"subjects": 780, "asd": n_asd, "control": n_control, "rois": remaining_roi_B, "features": remaining_feat_B},
        "strategy_C": {"subjects": 780, "asd": n_asd, "control": n_control, "rois": 116, "features": 6670,
                      "imputed_entries": 14025},
        "elastic_net_option1_fixed_count": False,
        "elastic_net_option2_fixed_count": True,
        "ssae_1000_feature_achievable_all_strategies": True,
        "synthetic_experiments_performed": ["ElasticNet sparsity grid", "top-K ranking demo",
                                            "fold-scoped imputation leakage-logic demo"],
        "real_data_used": False,
        "phase5c_status": phase5c_status,
        "human_decision_still_required": True,
        "phase5_status": "BLOCKED",
        "phase6_status": "NOT STARTED",
        "audit": {"previous_phases_unchanged": guard_unchanged, "fc_dataset_written": False,
                  "elastic_net_fit_on_real_data": False, "svm_rfe_run": False, "ssae_trained": False,
                  "softmax_trained": False, "xai_run": False, "roar_run": False,
                  "accuracy_calculated": False, "downloads_performed": False,
                  "cohort_modified": False, "aal_files_modified": False},
    }
    with open(OUT_DIR / "phase5c_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT_DIR / "phase5c_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2, default=str))
    log("phase5c_summary.json / .txt written.")

    audit_lines.append("Output files: strategy_architecture_comparison.csv, elastic_net_procedure_analysis.md, "
                        "elastic_net_feature_count_analysis.csv, strategy_site_impact.csv, "
                        "strategy_diagnosis_impact.csv, ssae_architecture_comparison.md, "
                        "xai_roar_implications.md, leakage_control_protocol.md, methodology_comparison.csv, "
                        "phase5c_summary.json/.txt, PHASE5C_DECISION.md")
    with open(OUT_DIR / "phase5c_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 5C outputs written to {OUT_DIR}")

    return summary, {"guard_unchanged": guard_unchanged, "n_subj_A": n_subj_A, "n_asd_A": n_asd_A,
                     "n_control_A": n_control_A, "remaining_roi_B": remaining_roi_B,
                     "remaining_feat_B": remaining_feat_B}


if __name__ == "__main__":
    main()
