# Methodological Decision Point — Zero-Variance AAL ROI Handling

**This document does not choose a method. It exists to put everything needed for that decision in one place.**

## 1. Problem Definition

Computing Pearson functional connectivity (FC) on the AAL-116 ROI time series for the 780-subject Phase-2 cohort produces mathematically undefined values wherever one of the two ROIs in a pair has zero variance across its time series (a constant signal). The paper's pipeline (AAL-116 → Pearson → Fisher-Z → lower triangle → 6,670 features → SVM-RFE → SSAE) assumes every one of the 6,670 features is a real number; it does not describe what to do when one is not.

## 2. Verified Facts

- 780 subjects (351 ASD, 429 Control), 20 sites — authoritative, unchanged from Phase 2.
- 47/780 subjects (6.03%) have ≥1 of 116 AAL ROIs with exactly zero variance.
- 19/116 ROIs (16.4%) are implicated, in two anatomically distinct clusters (orbitofrontal; inferior cerebellum/vermis) plus one temporal-pole region — mapping independently verified against three sources (Phase 4.1 §6).
- All constant values are exactly 0.0 (124/124 affected subject-ROI instances).
- An independent second parser exactly reproduced every sampled affected file (30/30) — this is not a parsing bug.
- 14,025 of 5,202,600 possible cohort-wide FC entries (0.27%) are undefined. Only 2,014 of the 6,670 possible FC pairs (30.2%) can ever be undefined for any subject; the other 4,656 pairs (69.8%) are never affected, for any subject, anywhere in the cohort.
- Per-subject impact is highly skewed: median 0 undefined pairs, mean 17.98, max 1,314 (19.7% of one subject's feature vector).
- Affected subjects concentrate at 2 of 20 sites: SBL (17/23, 74%) and PITT (13/41, 32%), together 64% of all affected subjects.
- No diagnosis association (ASD-affected rate 6.0% vs. Control-affected rate 6.1% — indistinguishable).
- The paper's Methods section, directly verified, does not mention zero-variance ROIs, undefined correlations, NaN handling, or any exclusion criterion beyond FD > 0.2mm.

## 3. What Pearson Correlation Mathematically Requires

Pearson's r between two series requires both to have nonzero variance — the formula's denominator is the product of the two series' standard deviations. When one is exactly zero, the value is the indeterminate form 0/0, not a computable number and not equal to zero by any derivation. This is a property of the mathematics, not an implementation choice.

## 4. Observed Constant-ROI Problem (recap)

See §2 above and Phase 4.1's `FINAL_RESOLUTION.md` for full detail, including the ROI-by-ROI and subject-by-subject breakdowns.

## 5. Consequences of Each Handling Option (quantified this phase)

| | Option 1: NaN | Option 2: Exclude 47 subjects | Option 3: Exclude 19 ROIs | Option 4: Numeric convention |
|---|---|---|---|---|
| Subjects retained | 780 | 733 (330 ASD / 403 Control) | 780 | 780 (if adopted) |
| ROIs retained | 116 | 116 | 97 | 116 (if adopted) |
| FC features/subject | 6,670 (0.27% NaN cohort-wide) | 6,670 | 4,656 (2,014 removed, 30.2%) | 6,670 (if adopted) |
| Worst single-site impact | none (no removal) | SBL 23→6 (−74%), PITT 41→28 (−32%) | none (no subject removal) | none (no removal) |
| Class balance shift | none | 351→330 ASD (−6.0%), 429→403 Control (−6.1%) — negligible ratio shift | none | none |
| Fisher-Z compatible | NaN propagates, does not resolve | Yes, unconditionally | Yes, unconditionally | Depends on convention — none established |
| Paper-specified | No | No | No | No — investigated and not found |

Full detail: `option1_nan_summary.csv`/`option1_nan_subjects.csv`, `option2_subject_exclusion_summary.csv`/`option2_site_impact.csv`, `option3_roi_exclusion_summary.csv`/`option3_roi_mapping.csv`, `option4_convention_analysis.md`.

## 6. Paper Compatibility

None of the four options is paper-specified (§2, confirmed again this phase). Options 2 and 3 each alter a dimension the paper treats as fixed (subject count or 6,670-feature shape, respectively). Option 1 preserves both dimensions but leaves 0.27% of entries without a defined value, which the paper's described SVM-RFE/SSAE pipeline is not documented to handle. Option 4 preserves both dimensions only by introducing a numeric value the paper never specifies.

## 7. Reproducibility Implications

All four options are individually reproducible (deterministic subject lists, ROI lists, or NaN positions — verified this phase via independent `np.corrcoef` computation matching algebraic prediction for all 47 affected subjects, 47/47 match). The reproducibility question is not "can this be done consistently" — it is "which of several non-paper-specified choices should stand in for what the paper's authors actually did," which cannot be resolved from the paper text.

## 8. Site-Distribution Implications

Option 2 is the only option with a material site-distribution effect, and it is severe for two sites: SBL would retain only 6 of 23 subjects (a 74% loss, leaving a very small site-level sample), and PITT would retain 28 of 41 (a 32% loss). Options 1, 3, and 4 have no direct site-distribution effect (no subjects removed), though Option 1's NaN entries and Option 4's substituted values are concentrated in SBL/PITT subjects' feature vectors specifically.

## 9. Remaining Ambiguity

- The root cause of the constant-ROI pattern (acquisition FOV, susceptibility artifact, or CPAC-side masking) remains an **unverified hypothesis**, not a demonstrated fact — the data needed to confirm it (original 4D volumes or CPAC logs for SBL/PITT) is not available in this project (Phase 3.5).
- Whether a faithful SVM-RFE/SSAE reproduction can accept NaN inputs (Option 1) is unknown — this project has not inspected any reference implementation of the paper's SVM-RFE/SSAE code (none is available; the paper does not publish one, and the paper's own reported accuracy is currently under an Expression of Concern for an unrelated data-leakage issue in that same SVM-RFE step — see Phase 4.1 §12).
- No option is disqualified by the evidence gathered; each remains viable with a different, documented cost.

## 10. Explicit Decision Point

**A human methodological decision is required before Phase 5 can proceed.** This phase has not made that decision and does not recommend one option over another. The decision-maker should weigh:
- Option 1 preserves the most data and the paper's exact 6,670-feature shape, at the cost of a downstream NaN-handling question this project cannot resolve from the paper alone.
- Option 2 gives a clean, fully-numeric downstream pipeline at the cost of losing 6% of subjects and severely thinning two specific sites.
- Option 3 gives a clean, fully-numeric downstream pipeline at the cost of deviating from the paper's fixed 116-ROI/6,670-feature design for all 780 subjects, including the 733 with no data problem at all.
- Option 4 is not currently available as a defensible choice — no paper-specified or otherwise principled convention was found.

---

**PHASE 4.2 STATUS: PASS — SENSITIVITY ANALYSIS COMPLETE**

**PHASE 5: BLOCKED PENDING HUMAN METHODOLOGICAL DECISION**
