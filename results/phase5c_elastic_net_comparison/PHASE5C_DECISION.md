# Phase 5C Decision Document — A/B/C + Elastic Net Architecture

**No strategy is selected here. No Elastic Net or SSAE was trained. No real ABIDE FC data was computed in this phase.**

## Purpose

Give the researcher the architectural and methodological evidence needed to choose among Strategies A/B/C, and to understand how replacing SVM-RFE with Elastic Net would interact with each, before any final FC generation or model training begins.

## Strategy A — Subject Exclusion (733 subjects, 330 ASD, 403 Control, 116 ROIs, 6,670 features)

Fully finite matrix, no invented values. Cost: 47 subjects lost entirely (their full feature vectors), concentrated at SBL (−74%) and PITT (−32%). Elastic Net and SSAE are directly compatible with no extra preprocessing step. XAI/ROAR results would apply to 733, not 780, subjects, with two specific sites underrepresented in whatever conclusions follow.

## Strategy B — ROI Exclusion (780 subjects, 351 ASD, 429 Control, 97 ROIs, 4,656 features)

Fully finite matrix, no invented values, no subjects lost. Cost: 2,014 of 6,670 features (30.2%) removed for all 780 subjects, including the 733 with no data problem at all — this permanently removes the orbitofrontal and inferior-cerebellum/vermis regions (and one temporal-pole region) from any possible XAI/ROAR finding, **not because they are unimportant, but because this project's preprocessing removed them.** Elastic Net Option 2 (top-1000) remains achievable (4,656 > 1,000).

## Strategy C — Fold-Scoped Feature-Wise Imputation (780 subjects, 351 ASD, 429 Control, 116 ROIs, 6,670 features)

No subjects or ROIs lost, but 14,025 entries (0.27% of all cohort-wide entries) are not measured values — they are training-fold statistics substituted in. This is an explicit, disclosed **methodological modification**, not something the paper describes. It requires the most additional machinery: a fold-scoped imputation step must run before Elastic Net in every CV fold (demonstrated in synthetic miniature this phase, not on real data), and any per-subject XAI attribution touching one of the 47 affected subjects' imputed features must carry that caveat explicitly.

## Elastic Net Procedure

Confirmed (synthetically) that Elastic Net's L1+L2 combination produces a feature count that varies with `alpha`/`l1_ratio` (Option 1) — no fixed count is guaranteed. **Option 2 (rank by |coefficient|, take top 1,000, within training folds only) is required to preserve the paper's `1000 → 500 → 100 → 2 → Softmax` SSAE architecture**, and is explicitly labeled a methodological modification (the paper's SVM-RFE has its own distinct ranking mechanism). All three strategies (A, B, C) can structurally support the 1,000-feature target under Option 2, since each strategy's candidate feature pool (6,670 for A/C, 4,656 for B) exceeds 1,000.

## Leakage Requirements

Identical discipline across all three strategies: Elastic Net must be fit on training-fold data only, with `alpha`/`l1_ratio` selected via inner CV, never against outer-fold or holdout accuracy. Strategy C adds one more fold-scoped requirement on top: the imputation statistic itself must also be training-fold-only. The forbidden pattern — fitting anything (imputation statistic, Elastic Net, feature ranking) on all 780 subjects before the 5-fold CV split — is structurally the same class of error as the data leakage already identified in the original paper's own SVM-RFE step (Phase 4.1's Expression-of-Concern finding). Full protocol: `leakage_control_protocol.md`.

## SSAE Compatibility

`1000 → 500 → 100 → 2 → Softmax` is achievable under all three strategies (given Option 2 feature selection). The output layer stays at 2 classes (ASD, Control) with Softmax throughout — logistic regression was not substituted into the SSAE architecture anywhere in this phase, per instruction; it remains available only as a separate baseline classifier, not part of the SSAE.

## XAI / ROAR Implications

Each strategy carries a distinct interpretability caveat that must travel with any future result: A → results don't represent SBL/PITT well; B → 19 real anatomical regions can never appear in any finding, regardless of true importance; C → per-subject attributions for the 47 affected subjects may partly reflect an imputed statistic, not measured brain activity. None of this was executed in this phase.

## Whether a Human Methodological Decision Is Still Required

**Yes.** This phase adds architectural detail (Elastic Net behavior, 1,000-feature achievability, SSAE compatibility, leakage protocol) but does not resolve the underlying question from Phase 4.2/5B: which of A, B, or C to adopt, and, if C, formal approval of the specific fold-scoped imputation procedure. No ranking or accuracy prediction was produced, consistent with the task's explicit prohibition.

---

**PHASE 5C STATUS: PASS**

**PHASE 5: BLOCKED — human decision among A/B/C still required.**

**PHASE 6: NOT STARTED.**
