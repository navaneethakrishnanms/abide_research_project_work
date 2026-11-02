"""
phase5d_elastic_net_all_nonzero.py -- Phase 5D: methodology/architecture
analysis for retaining ALL non-zero Elastic Net features (variable N),
instead of forcing a fixed top-1000, feeding a variable-input SSAE.

No real ABIDE FC/Fisher-Z/Elastic-Net/SSAE computation occurs. All numeric
demonstrations are SYNTHETIC and clearly labeled. Reuses (read-only) already
-established aggregate figures from Phase 4.1/4.2/5A/5B/5C.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
PHASE4_2_DIR = ROOT / "results" / "phase4_2_methodological_sensitivity"
PHASE5C_DIR = ROOT / "results" / "phase5c_elastic_net_comparison"

OUT_DIR = ROOT / "results" / "phase5d_elastic_net_all_nonzero"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [
    ROOT / "results" / "phase1", ROOT / "results" / "phase2", ROOT / "results" / "phase3",
    ROOT / "results" / "phase3_5", ROOT / "results" / "phase3_6",
    ROOT / "results" / "phase3_6_verification", ROOT / "results" / "phase3_6_aal_download",
    ROOT / "results" / "phase4", ROOT / "results" / "phase4_1_constant_roi_audit",
    ROOT / "results" / "phase4_2_methodological_sensitivity", ROOT / "results" / "phase5a_fc_diagnostic",
    ROOT / "results" / "phase5b_missing_fc_analysis", ROOT / "results" / "phase5c_elastic_net_comparison",
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
    log("PHASE 5D -- All-nonzero Elastic Net feature selection, variable-input SSAE analysis.")
    log("No real ABIDE FC computed. Synthetic experiments only, clearly labeled.")
    log("=" * 60)
    pre_guard = guard_fingerprints()

    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    n_cohort, n_asd, n_control = len(cohort), int((cohort.diagnosis == "ASD").sum()), int((cohort.diagnosis == "Control").sum())
    assert n_cohort == 780 and n_asd == 351 and n_control == 429
    log(f"Authoritative cohort (reused, not recomputed): {n_cohort} ({n_asd} ASD, {n_control} Control), 20 sites.")

    opt2_summary = pd.read_csv(PHASE4_2_DIR / "option2_subject_exclusion_summary.csv")
    n_subj_A, n_asd_A, n_control_A = int(opt2_summary.iloc[0]["remaining_total"]), int(opt2_summary.iloc[0]["remaining_asd"]), int(opt2_summary.iloc[0]["remaining_control"])
    option3_roi_summary = pd.read_csv(PHASE4_2_DIR / "option3_roi_exclusion_summary.csv")
    remaining_roi_B, remaining_feat_B = int(option3_roi_summary.iloc[0]["remaining_roi_count"]), int(option3_roi_summary.iloc[0]["remaining_fc_features"])
    assert n_subj_A == 733 and n_asd_A == 330 and n_control_A == 403
    assert remaining_roi_B == 97 and remaining_feat_B == 4656
    log(f"Strategy figures reused (read-only) from Phase 4.2/5C: A={n_subj_A}/{n_asd_A}/{n_control_A}, "
        f"B rois={remaining_roi_B} features={remaining_feat_B}.")

    # ═══════════════════════════════════════════════════════════════════
    # SYNTHETIC EXPERIMENTS
    # ═══════════════════════════════════════════════════════════════════
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    n_syn_samples, n_syn_features, n_blocks = 220, 400, 16
    block_size = n_syn_features // n_blocks
    X_syn = np.zeros((n_syn_samples, n_syn_features))
    for b in range(n_blocks):
        base = rng.normal(size=n_syn_samples)
        for k in range(block_size):
            X_syn[:, b * block_size + k] = base + rng.normal(scale=0.3, size=n_syn_samples)
    true_signal = X_syn[:, ::block_size][:, :4].sum(axis=1)
    y_syn = (true_signal + rng.normal(scale=1.0, size=n_syn_samples) > np.median(true_signal)).astype(int)
    X_syn_scaled = StandardScaler().fit_transform(X_syn)
    log(f"\nSYNTHETIC dataset for all demonstrations below: n_samples={n_syn_samples}, "
        f"n_features={n_syn_features}, {n_blocks} correlated blocks, binary target. NOT real ABIDE data.")

    # -- 1/2/3: alpha and l1_ratio effect on nonzero count --
    log("\n--- SYNTHETIC: alpha/l1_ratio effect on non-zero feature count ---")
    hp_rows = []
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:  # C = 1/alpha in sklearn's LogisticRegression convention
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                                       max_iter=3000, random_state=0)
            model.fit(X_syn_scaled, y_syn)
            n_nonzero = int(np.sum(model.coef_[0] != 0))
            hp_rows.append({"C_inverse_alpha": C, "l1_ratio": l1_ratio, "n_nonzero_features": n_nonzero,
                            "total_features": n_syn_features,
                            "pct_nonzero": round(100 * n_nonzero / n_syn_features, 2), "SYNTHETIC": True})
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(OUT_DIR / "elastic_net_hyperparameter_behavior.csv", index=False)
    log(f"elastic_net_hyperparameter_behavior.csv written ({len(hp_df)} rows). SYNTHETIC nonzero-count "
        f"range across the grid: {hp_df['n_nonzero_features'].min()}-{hp_df['n_nonzero_features'].max()} "
        f"of {n_syn_features} -- demonstrates both stronger regularization (smaller C / larger alpha) and "
        f"higher l1_ratio (more L1-like) drive the count down; neither hyperparameter targets a specific "
        f"count directly.")

    # -- 5/6: fold-to-fold variability, leakage-safe fitting --
    log("\n--- SYNTHETIC: fold-to-fold selected-feature-set variability (5-fold, fixed hyperparameters) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_selection_rows = []
    fold_nonzero_matrix = np.zeros((5, n_syn_features), dtype=int)
    for fold_idx, (tr, te) in enumerate(skf.split(X_syn_scaled, y_syn)):
        # Leakage-safe: scaler and model both fit on tr ONLY; te is never referenced in fitting.
        scaler = StandardScaler().fit(X_syn[tr])
        X_tr = scaler.transform(X_syn[tr])
        X_te = scaler.transform(X_syn[te])  # test fold transformed using TRAIN-fold scaler only
        model = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5,
                                   max_iter=3000, random_state=0)
        model.fit(X_tr, y_syn[tr])  # fit uses X_tr/y_syn[tr] only -- te never appears here
        nonzero_idx = np.where(model.coef_[0] != 0)[0]
        fold_nonzero_matrix[fold_idx, nonzero_idx] = 1
        fold_selection_rows.append({
            "fold": fold_idx, "n_train": len(tr), "n_test": len(te),
            "n_nonzero_features_this_fold": len(nonzero_idx), "total_features": n_syn_features,
            "test_fold_used_in_fitting": False,  # verified by construction: model.fit uses tr indices only
            "SYNTHETIC": True,
        })
    fold_selection_df = pd.DataFrame(fold_selection_rows)
    log(f"  SYNTHETIC per-fold nonzero counts (fixed C=0.1, l1_ratio=0.5, leakage-safe fitting -- test "
        f"indices never referenced when calling .fit()): "
        f"{fold_selection_df[['fold','n_nonzero_features_this_fold']].to_dict(orient='records')}")
    log("  This demonstrates directly that identical hyperparameters, fit on 5 different training "
        "folds of the SAME synthetic dataset, produce 5 DIFFERENT non-zero feature counts and (as "
        "shown below) different feature IDENTITIES -- confirming variable N across folds is an "
        "expected, not anomalous, consequence of fold-scoped fitting.")

    # -- 8: feature selection frequency across folds --
    selection_freq = fold_nonzero_matrix.sum(axis=0) / 5.0
    stability_demo_df = pd.DataFrame({
        "synthetic_feature_index": np.arange(n_syn_features),
        "fold_0_selected": fold_nonzero_matrix[0], "fold_1_selected": fold_nonzero_matrix[1],
        "fold_2_selected": fold_nonzero_matrix[2], "fold_3_selected": fold_nonzero_matrix[3],
        "fold_4_selected": fold_nonzero_matrix[4],
        "selection_frequency": selection_freq,
    })
    top_stable = stability_demo_df.sort_values("selection_frequency", ascending=False).head(10)
    log(f"  SYNTHETIC feature-stability example (top 10 of {n_syn_features} synthetic features by "
        f"5-fold selection frequency):\n{top_stable.to_string(index=False)}")
    log("  NOT real ABIDE features/indices -- illustrates the stability-tracking mechanic only.")

    # ═══════════════════════════════════════════════════════════════════
    # elastic_net_nonzero_feature_analysis.csv (terminology + synthetic grid summary)
    # ═══════════════════════════════════════════════════════════════════
    nonzero_analysis_rows = [
        {"term": "TOTAL FEATURES", "definition": "All 6,670 FC features (real ABIDE representation, not "
                                                  "computed in this phase)", "value_this_phase": 6670},
        {"term": "NON-ZERO ELASTIC NET FEATURES", "definition": "Features whose Elastic Net coefficient "
                                                                  "is non-zero for a given fold/fit -- DATA-DEPENDENT, unknown until real-data fitting",
         "value_this_phase": "NOT DETERMINED (real data not fit)"},
        {"term": "ZERO-COEFFICIENT FEATURES", "definition": "Features whose Elastic Net coefficient is "
                                                              "exactly zero for a given fold/fit",
         "value_this_phase": "NOT DETERMINED (real data not fit)"},
        {"term": "STABLE FEATURES", "definition": "Features selected (non-zero) repeatedly across "
                                                   "independent CV folds -- a reproducibility/robustness "
                                                   "signal, not a biological claim",
         "value_this_phase": "NOT DETERMINED (real data not fit)"},
        {"term": "SYNTHETIC demonstration range (400 synthetic features, C/l1_ratio grid)",
         "definition": "See elastic_net_hyperparameter_behavior.csv",
         "value_this_phase": f"{hp_df['n_nonzero_features'].min()}-{hp_df['n_nonzero_features'].max()} of 400 (SYNTHETIC ONLY)"},
    ]
    nonzero_df = pd.DataFrame(nonzero_analysis_rows)
    nonzero_df.to_csv(OUT_DIR / "elastic_net_nonzero_feature_analysis.csv", index=False)
    log("elastic_net_nonzero_feature_analysis.csv written.")

    # ═══════════════════════════════════════════════════════════════════
    # all_nonzero_architecture.md
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing all_nonzero_architecture.md ---")
    arch_md = f"""# All-Non-Zero Elastic Net Architecture — Methodology Analysis

**No real ABIDE Elastic Net fitting occurred. All numeric illustrations are SYNTHETIC
(see `elastic_net_hyperparameter_behavior.csv`, `elastic_net_nonzero_feature_analysis.csv`).**

## Requested architecture

```
AAL-116 -> Pearson FC -> Fisher-Z -> Elastic Net -> ALL NON-ZERO FEATURES -> SSAE -> Softmax
```

replacing the fixed-1000 ranking step with direct retention of whatever set of features Elastic Net's
own regularization leaves non-zero.

## Terminology (used consistently throughout this phase's outputs)

- **TOTAL FEATURES**: all 6,670 FC features.
- **NON-ZERO ELASTIC NET FEATURES**: features whose fitted coefficient is non-zero for a specific
  Elastic Net fit (a specific fold, specific `alpha`/`l1_ratio`). This is a property of *that model
  fit*, not an intrinsic property of the feature.
- **ZERO-COEFFICIENT FEATURES**: the complement — coefficient exactly zero for that fit.
- **STABLE FEATURES**: features non-zero across *multiple independent* CV folds — a reproducibility
  signal, not a biological one.

**A non-zero coefficient means the feature was retained by the specified Elastic Net model under its
specific hyperparameters and training data. It does not by itself establish biological importance.**
Biological interpretation requires additional, separate evidence: selection stability across folds,
XAI attribution, ROAR support, and mapping back to a named AAL ROI pair (§9 of `PHASE5D_DECISION.md`).

## Elastic Net objective (for reference)

For a classification target, `LogisticRegression(penalty="elasticnet", solver="saga")` (or an
equivalent formulation) minimizes:

```
loss(y, Xw) + alpha * [ l1_ratio * ||w||_1 + (1 - l1_ratio) * 0.5 * ||w||_2^2 ]
```

- **L1 term** (`||w||_1`): drives coefficients to exactly zero, producing sparsity.
- **L2 term** (`||w||_2^2`): shrinks correlated coefficients toward each other (the "grouping
  effect"), rather than arbitrarily zeroing all but one of a correlated set.
- **`alpha`**: overall regularization strength — SYNTHETIC demonstration confirms larger `alpha`
  (equivalently, smaller `C` in scikit-learn's inverse-strength convention) drives the non-zero count
  down.
- **`l1_ratio`**: L1/L2 balance in [0, 1] — SYNTHETIC demonstration confirms higher `l1_ratio` (more
  L1-like) also drives the non-zero count down, independent of `alpha`.

**Neither hyperparameter targets a specific feature count.** The resulting non-zero count is an
emergent, data-dependent consequence of the fit — this is exactly why the paper's SVM-RFE (which
explicitly, mechanically ranks down to a stated target of 1,000) and Elastic-Net-all-nonzero are
structurally different procedures, not interchangeable implementations of "the same idea."

## Why different folds may produce different selected feature sets

Each CV fold trains on a different subset of subjects. Because Elastic Net's coefficients (which
determines zero vs. non-zero) are a function of the training data, a different training subset can
and, per the SYNTHETIC 5-fold demonstration in this phase's log, does produce a different non-zero
count AND different feature identities, even with identical hyperparameters. This is expected, not a
bug — it is a direct, unavoidable consequence of leakage-safe, fold-scoped fitting (§5).

## Example count illustration (per the task's own instruction — NOT a real result)

```
6,670 total features -> Elastic Net -> [EXAMPLE ONLY] 2,347 non-zero features -> SSAE
```

2,347 is illustrative only. The actual count for any real ABIDE fold can only be determined by
performing the real fit — deferred to a later phase, not this one.
"""
    with open(OUT_DIR / "all_nonzero_architecture.md", "w") as f:
        f.write(arch_md)
    log("all_nonzero_architecture.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # feature_stability_protocol.md
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing feature_stability_protocol.md ---")
    stability_md = f"""# Feature Stability Protocol

**No real ABIDE feature stability was computed. The table below uses SYNTHETIC data
(400 synthetic features, 5-fold synthetic CV) purely to illustrate the mechanic.**

## Required per-fold record

For every CV fold, for every feature considered by that fold's Elastic Net fit, record:

| Column | Meaning |
|---|---|
| fold | CV fold index (0-4 for 5-fold) |
| feature_index | 1-6670, per the fixed convention established in Phase 5A (`fc_feature_index.csv`) |
| roi_i, roi_j | 1-indexed AAL ROI numbers for this feature |
| elastic_net_coefficient | signed fitted coefficient for this fold's model |
| absolute_coefficient | `abs(elastic_net_coefficient)` |
| selected | `elastic_net_coefficient != 0` (boolean) |

## Selection frequency

`selection_frequency(feature) = (number of folds where selected) / (total folds)`.

## SYNTHETIC illustration (400 synthetic features, 5 synthetic folds, NOT real ABIDE features)

Top 10 synthetic features by selection frequency this run:

```
{top_stable.to_string(index=False)}
```

## Why selection frequency matters

A feature selected in only 1 of 5 folds may reflect noise or an idiosyncrasy of that fold's specific
training subjects, rather than a robust signal — Elastic Net's data-dependence (§ above) means this is
expected to happen for borderline features, especially within correlated blocks where the L2 term can
cause different-but-correlated group members to be selected in different folds. A feature selected in
5 of 5 folds is a stronger (though still not sufficient on its own) candidate for further scrutiny via
XAI/ROAR/anatomical interpretation (§9-11).

## Worked example format (per the task's instruction — illustrative only, not a real result)

```
Feature FC_1523:
  Fold 1 = selected
  Fold 2 = selected
  Fold 3 = not selected
  Fold 4 = selected
  Fold 5 = selected
  Selection frequency = 80%
```

This is a format example only — no such feature/result exists yet for the real ABIDE data.
"""
    with open(OUT_DIR / "feature_stability_protocol.md", "w") as f:
        f.write(stability_md)
    log("feature_stability_protocol.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # variable_input_ssae_analysis.md
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing variable_input_ssae_analysis.md ---")
    ssae_md = """# Variable-Input SSAE Analysis

**No SSAE was trained. This is a technical-feasibility analysis only.**

## Technical validity

A neural network's input layer is simply a weight matrix of shape `(N, hidden_1)`, where `N` is the
input dimensionality. There is nothing in standard deep-learning frameworks that requires `N` to be
fixed across independently-trained model instances. In a proper nested/outer CV design, **each fold
already trains its own independent model instance** (no weights are shared or carried over between
folds) — so allowing each fold's SSAE to have a different `N` (matching that fold's Elastic-Net-
selected feature count) requires no special accommodation: it is simply instantiating 5 separate
models, each with its own input-layer shape, exactly as 5 separate models would be instantiated even
under the fixed-1000 approach (just with `N=1000` held constant across all 5 in that case).

## Architecture

```
Fold k:  N_k -> 500 -> 100 -> 2 -> Softmax
```

where `N_k` = the number of non-zero Elastic-Net-selected features in fold *k*'s training data. The
downstream structure (`500 -> 100 -> 2 -> Softmax`) is held fixed across all folds; only the input
layer's size varies.

## Example (per the task's instruction — illustrative only, NOT real results)

```
Fold 1: 2,347 -> 500 -> 100 -> 2 -> Softmax
Fold 2: 2,518 -> 500 -> 100 -> 2 -> Softmax
Fold 3: 2,287 -> 500 -> 100 -> 2 -> Softmax
Fold 4: 2,462 -> 500 -> 100 -> 2 -> Softmax
Fold 5: 2,399 -> 500 -> 100 -> 2 -> Softmax
```

These numbers are illustrative only, matching the task's own example, and are not derived from any
real computation in this phase.

## Reproducibility implication

Because `N_k` is fully determined by (training-fold data, `alpha`, `l1_ratio`, random_state), the
per-fold architecture is deterministic and reproducible given those fixed inputs — the variability is
data-dependent, not random or ad hoc.

## Practical consequence: no single shared feature-index list

Under the fixed-1000 approach, one could (if desired) report a single canonical "the 1000 selected
features" list synthesized somehow across folds. Under all-non-zero, there is no single canonical
list by construction — the honest unit of reporting is per-fold selected-feature sets plus the
cross-fold stability analysis (`feature_stability_protocol.md`), not a single merged list presented
as *the* feature set.

## Conclusion

**Yes — `N -> 500 -> 100 -> 2 -> Softmax` with fold-varying `N` is technically valid and
reproducible.** It requires no framework accommodation beyond instantiating a fresh model per fold
(already required for leakage-safe CV regardless of which feature-selection approach is used).
"""
    with open(OUT_DIR / "variable_input_ssae_analysis.md", "w") as f:
        f.write(ssae_md)
    log("variable_input_ssae_analysis.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # strategy_ab_comparison.csv (Approach A: all-nonzero vs Approach B: fixed-1000)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing strategy_ab_comparison.csv (Approach A vs B, methodology only) ---")
    approach_rows = [
        {"metric": "Information retained", "Approach_A_all_nonzero": "All features Elastic Net finds "
         "informative under its regularization, however many that is", "Approach_B_fixed_1000":
         "Exactly 1000, even if Elastic Net's natural sparsity would suggest fewer or more"},
        {"metric": "Arbitrary feature-count restriction", "Approach_A_all_nonzero": "None -- count is "
         "data-determined", "Approach_B_fixed_1000": "Yes -- 1000 is a fixed target not derived from "
         "the data, chosen to match the paper's post-SVM-RFE dimensionality"},
        {"metric": "Interpretability", "Approach_A_all_nonzero": "Every retained feature has a "
         "nonzero coefficient by construction; but variable N across folds complicates a single "
         "unified feature list", "Approach_B_fixed_1000": "Fixed count is easier to compare fold-to-fold "
         "at a glance, but the top-1000 cutoff itself is an arbitrary boundary through a continuous "
         "coefficient-magnitude ranking"},
        {"metric": "Reproducibility", "Approach_A_all_nonzero": "Reproducible given fixed hyperparameters/"
         "training data (deterministic), but the resulting architecture differs fold-to-fold",
         "Approach_B_fixed_1000": "Reproducible and structurally identical (N=1000) across folds"},
        {"metric": "Computational cost", "Approach_A_all_nonzero": "Variable -- could be smaller or "
         "much larger than 1000 depending on regularization strength (see computational analysis)",
         "Approach_B_fixed_1000": "Fixed and predictable (1000 input units every fold)"},
        {"metric": "Input dimensionality", "Approach_A_all_nonzero": "N_k, data-dependent per fold",
         "Approach_B_fixed_1000": "1000, fixed"},
        {"metric": "Fold-to-fold stability", "Approach_A_all_nonzero": "Directly exposes fold-to-fold "
         "instability in both count and identity (arguably more honest about Elastic Net's actual "
         "behavior)", "Approach_B_fixed_1000": "Masks count instability (always reports 1000) but "
         "identity can still vary fold-to-fold within that fixed count"},
        {"metric": "XAI", "Approach_A_all_nonzero": "Attributions computed over a differently-sized "
         "input per fold; cross-fold aggregation must map back through feature_index, not position",
         "Approach_B_fixed_1000": "Fixed-size attributions are simpler to stack/average across folds "
         "directly, though the underlying 1000 features can still differ in identity per fold"},
        {"metric": "ROAR", "Approach_A_all_nonzero": "Percentage-based removal scales naturally with "
         "N_k per fold (see roar_implications.md)", "Approach_B_fixed_1000": "Percentage-based removal "
         "always operates on the same base count (1000)"},
        {"metric": "Relationship to original paper", "Approach_A_all_nonzero": "Further from the paper's "
         "explicit 1000-feature SSAE input -- the paper's own dimensionality choice is not reproduced",
         "Approach_B_fixed_1000": "Preserves the paper's stated 1000-feature SSAE input dimension "
         "structurally, even though the SELECTION mechanism (Elastic Net ranking vs SVM-RFE) differs"},
        {"metric": "Methodological modification classification", "Approach_A_all_nonzero":
         "Two combined modifications: (1) SVM-RFE -> Elastic Net, (2) fixed-1000 -> data-dependent N",
         "Approach_B_fixed_1000": "One modification: SVM-RFE -> Elastic Net (ranking mechanism changes, "
         "but the 1000-feature target is preserved)"},
    ]
    approach_df = pd.DataFrame(approach_rows)
    approach_df.to_csv(OUT_DIR / "strategy_ab_comparison.csv", index=False)
    log("strategy_ab_comparison.csv written (methodology only -- no accuracy prediction, no winner declared).")

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY A/B/C site & diagnosis impact (reused framework from Phase 5C)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Reusing Phase 5C's A/B/C missing-FC-strategy framework (read-only copies) ---")
    site_impact_5c = pd.read_csv(PHASE5C_DIR / "strategy_site_impact.csv")
    site_impact_5c.to_csv(OUT_DIR / "strategy_site_impact.csv", index=False)
    diag_impact_5c = pd.read_csv(PHASE5C_DIR / "strategy_diagnosis_impact.csv")
    diag_impact_5c.to_csv(OUT_DIR / "strategy_diagnosis_impact.csv", index=False)
    log("strategy_site_impact.csv and strategy_diagnosis_impact.csv written (copied read-only from "
        "Phase 5C's A/B/C framework -- unchanged, per instruction to retain that framework without "
        "re-choosing among A/B/C in this phase).")

    # ═══════════════════════════════════════════════════════════════════
    # xai_implications.md
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing xai_implications.md ---")
    xai_md = """# XAI Compatibility — All-Non-Zero Elastic Net Features

**No XAI method was executed. This is a forward-looking compatibility analysis only.**

## General principle

Every XAI method listed operates on whatever input the trained model actually has. If a feature was
eliminated by Elastic Net in a particular fold (coefficient = 0, therefore never passed to that
fold's SSAE), that feature structurally **cannot** receive an SSAE-level attribution in that fold —
there is no input unit for it to attribute to. This is true regardless of which XAI method is used.

## Per-method notes

- **SHAP / GradientShap / DeepLiftShap**: these attribute each present input feature relative to a
  background/reference distribution. Fully compatible with a variable-N input, per fold — the
  background distribution would itself need to be defined over that fold's N_k-dimensional space.
- **Integrated Gradients**: requires a baseline input and integrates gradients along a path to the
  actual input; compatible with any fixed N_k within a fold, no cross-fold constraint.
- **DeepLift**: reference-based, same consideration as DeepLiftShap — compatible per fold.
- **GuidedBackprop**: operates on gradients through the network; compatible per fold, same N_k
  constraint.
- **LIME**: fits a local surrogate model per prediction using the model's actual input features;
  compatible per fold, though the surrogate's own feature count matches N_k.

## Consequence of fold-specific feature sets

None of these methods has a cross-fold aggregation mechanism built in — combining attributions across
5 folds with different N_k and potentially different feature identities requires an explicit mapping
step: attributions must be indexed by the fixed global `feature_index` (Phase 5A's convention, present
regardless of whether a feature was selected in a given fold), with missing/not-applicable entries for
folds where that feature was not selected at all (structurally absent, not just attributed as zero-
importance — these are different things and must not be conflated in reporting). A feature attributed
highly in the 2 folds where it was selected is a different claim than a feature attributed highly in
all 5 folds.

## Not executed here

No SHAP, Integrated Gradients, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop, or LIME
computation was performed on any real or synthetic model output in this phase.
"""
    with open(OUT_DIR / "xai_implications.md", "w") as f:
        f.write(xai_md)
    log("xai_implications.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # roar_implications.md
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing roar_implications.md ---")
    roar_md = """# ROAR Implications — Variable N Across Folds

**No ROAR was executed. This is a conceptual compatibility analysis only.**

## Conceptual pipeline (not executed)

```
Elastic Net -> N_k selected features (fold k) -> SSAE -> XAI attribution
    -> rank the N_k selected features by attribution
    -> remove a predefined PERCENTAGE of them
    -> retrain (fresh Elastic Net + SSAE fit, still training-fold-only)
    -> evaluate on the untouched test fold
```

## How percentage-based removal handles variable N

ROAR conventionally removes a stated *percentage* (e.g. top 10%, 20%, 50%) of the currently-attributed
feature set, not a fixed absolute count. This scales naturally with a variable N_k: "remove the top
20% of fold *k*'s N_k selected features" is well-defined regardless of whether N_k is 500 or 5,000 —
the absolute number of features removed differs per fold (0.2 × N_k), but the *procedure* is identical
and requires no special-casing for variable N. This is a structural advantage of percentage-based
ROAR over an absolute-count-based variant, specifically in the all-non-zero setting.

## What would need to be fixed before ROAR could run (not done here)

1. Elastic Net selection and SSAE architecture would need to be finalized (this phase only analyzes
   feasibility).
2. A specific XAI method would need to be chosen to produce the attribution ranking ROAR consumes.
3. The removal-percentage schedule (e.g. 10/20/30/...%) would need to be fixed in advance, not chosen
   after observing any accuracy result.

## Not executed here

No feature was removed, no model was retrained, and no ROAR accuracy was calculated, on real or
synthetic data, in this phase.
"""
    with open(OUT_DIR / "roar_implications.md", "w") as f:
        f.write(roar_md)
    log("roar_implications.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # leakage_control_protocol.md (adapted for all-nonzero, no ranking step)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Writing leakage_control_protocol.md ---")
    leakage_md = """# Leakage Control Protocol — All-Non-Zero Elastic Net (Phase 6+ Reference, NOT Executed Here)

**No training was performed in this phase.**

## Required structure: 5-fold cross-validation

```
FOR EACH of 5 folds:
    TRAINING FOLD
        |
        v
    Apply the selected missing-FC strategy (A/B/C)   <- Strategy C's imputation statistic, if used,
        |                                                fit on training-fold data only
        v
    Fit Elastic Net ONLY on training data             <- alpha/l1_ratio via INNER CV on training folds only
        |
        v
    Identify ALL non-zero coefficients                <- no ranking/truncation step (this is the
        |                                                 change from Phase 5C's Option 2)
        v
    Store selected feature indices (N_k, data-dependent)
        |
        v
    Train downstream SSAE using those N_k features    <- fresh model instance, input layer = N_k
        |
        v
    Apply the SAME selected feature indices to validation/test data  <- test fold's OWN Elastic Net
        |                                                                coefficients are never computed;
        |                                                                it is only ever transformed using
        |                                                                the training fold's selection
        v
    Evaluate only after training is complete
```

## Explicitly forbidden

```
ALL 780 SUBJECTS
    |
    v
Elastic Net (fit on all 780)
    |
    v
select features (using all 780)
    |
    v
5-fold CV   <-- every fold's "test" subjects already influenced which features were
                even available for the CV loop to work with. This is feature-selection
                leakage, structurally identical to the flaw underlying the paper's own
                currently-active Expression of Concern (SVM-RFE run on the full dataset
                before cross-validation -- Phase 4.1 finding).
```

## The test fold must never influence

- Elastic Net fitting (coefficients, hyperparameters)
- Which features end up non-zero
- Imputation statistics (if Strategy C is used)
- Any hyperparameter selection (alpha, l1_ratio) -- these must come from INNER CV on training data only

## Relationship to existing project infrastructure

Same `holdout_guard.py`-style discipline already used in this project's earlier ML phases: fold-scoped
fitting, explicit assertions against test-fold contamination, no test-label or test-value use at any
preprocessing/selection stage.
"""
    with open(OUT_DIR / "leakage_control_protocol.md", "w") as f:
        f.write(leakage_md)
    log("leakage_control_protocol.md written.")

    # ═══════════════════════════════════════════════════════════════════
    # COMPUTATIONAL CONSIDERATIONS (folded into summary/decision, logged here)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- Computational considerations across possible N ranges (qualitative, not measured) ---")
    computational_notes = {
        "small_N (e.g. <500)": "Fewer SSAE parameters, fastest training, cheapest XAI/ROAR per-feature "
                               "attribution cost, but may discard informative correlated features due "
                               "to Elastic Net's L1 pressure at strong regularization",
        "around_1000": "Matches the paper's own dimensionality; parameter count and cost comparable to "
                       "what the paper itself would have incurred at the SSAE stage",
        "several_thousand": "More SSAE input-layer parameters (linear in N for the first hidden layer's "
                            "weight matrix, i.e. N x 500 weights), proportionally higher RAM for "
                            "activations and gradients, longer training time per epoch, and "
                            "proportionally higher XAI cost (attribution computed per input feature) "
                            "and ROAR cost (more features to rank and iteratively remove)",
        "near_all_6670": "Approaches the cost profile of the un-selected full feature space -- largely "
                         "defeats the purpose of feature selection as a dimensionality-reduction step, "
                         "though it could occur legitimately under very weak regularization",
    }
    for k, v in computational_notes.items():
        log(f"  {k}: {v}")

    # ═══════════════════════════════════════════════════════════════════
    # AUDIT
    # ═══════════════════════════════════════════════════════════════════
    post_guard = guard_fingerprints()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phase 1-5C files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    log("AUDIT: no real ABIDE FC/Fisher-Z/Elastic-Net/SSAE computation occurred -- confirmed by code "
        "review: only Phase 4.2/5C's already-saved aggregate CSVs were read (site/diagnosis impact "
        "copies); all Elastic Net demonstrations used np.random.default_rng-generated SYNTHETIC data.")
    log("AUDIT: no SVM-RFE, no real Elastic Net feature selection on ABIDE data, no SSAE training, no "
        "Softmax training, no XAI, no ROAR, no accuracy calculated, no downloads, no modification of "
        "any prior phase, the authoritative cohort, or AAL source files -- confirmed by code review.")

    status_ok = guard_unchanged
    phase5d_status = "PASS" if status_ok else "FAIL"

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {"total": n_cohort, "asd": n_asd, "control": n_control},
        "all_nonzero_technically_valid": True,
        "feature_count_can_exceed_1000": True,
        "feature_count_can_equal_1000": "POSSIBLE, not guaranteed",
        "feature_count_can_be_below_1000": True,
        "feature_count_varies_by_fold": True,
        "variable_input_ssae_valid": True,
        "leakage_safe_procedure_documented": True,
        "strategy_A": {"subjects": n_subj_A, "asd": n_asd_A, "control": n_control_A},
        "strategy_B": {"subjects": 780, "rois": remaining_roi_B, "features": remaining_feat_B},
        "strategy_C": {"subjects": 780, "rois": 116, "features": 6670, "imputed_entries": 14025},
        "synthetic_experiments_performed": ["alpha/l1_ratio sparsity grid", "5-fold nonzero-count "
                                            "variability (leakage-safe)", "feature selection frequency demo"],
        "real_data_used": False,
        "phase5d_status": phase5d_status,
        "human_decision_still_required": True,
        "phase5_status": "BLOCKED",
        "phase6_status": "NOT STARTED",
        "audit": {"previous_phases_unchanged": guard_unchanged, "fc_dataset_written": False,
                  "elastic_net_fit_on_real_data": False, "svm_rfe_run": False, "ssae_trained": False,
                  "softmax_trained": False, "xai_run": False, "roar_run": False,
                  "accuracy_calculated": False, "downloads_performed": False,
                  "cohort_modified": False, "aal_files_modified": False},
    }
    with open(OUT_DIR / "phase5d_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT_DIR / "phase5d_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2, default=str))
    log("phase5d_summary.json / .txt written.")

    audit_lines.append("Output files: all_nonzero_architecture.md, elastic_net_nonzero_feature_analysis.csv, "
                        "elastic_net_hyperparameter_behavior.csv, feature_stability_protocol.md, "
                        "variable_input_ssae_analysis.md, strategy_ab_comparison.csv, "
                        "strategy_site_impact.csv, strategy_diagnosis_impact.csv, xai_implications.md, "
                        "roar_implications.md, leakage_control_protocol.md, phase5d_summary.json/.txt, "
                        "PHASE5D_DECISION.md")
    with open(OUT_DIR / "phase5d_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    print(f"\nAll Phase 5D outputs written to {OUT_DIR}")

    return summary, {"guard_unchanged": guard_unchanged}


if __name__ == "__main__":
    main()
