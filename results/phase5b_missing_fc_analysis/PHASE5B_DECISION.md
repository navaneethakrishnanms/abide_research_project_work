# Phase 5B Decision Document — Handling the 14,025 Undefined Pearson FC Entries

**This document does not finalize a method. It exists to give the human decision-maker everything established across Phase 4.1, 4.2, 5A, and this phase in one place.**

## A. What Is Mathematically Established

- Pearson correlation requires nonzero variance in both inputs; where either is exactly zero, the result is the indeterminate form 0/0 — undefined, not zero, not approximately anything.
- 14,025 of 5,202,600 possible cohort-wide FC entries (0.27%) are undefined, affecting 47/780 subjects (6.03%) and 2,014/6,670 features (30.2%) — re-confirmed by an independently-built, in-memory-only 780×6,670 matrix this phase (exact match to Phase 5A's 14,025).
- Pairwise deletion and rank-based alternative correlations (Strategy E) cannot resolve a constant-vector problem — a constant series remains constant over any sub-selection of timepoints, and rank-based measures are equally undefined for constant input. This is ruled out mathematically, not by preference.
- Time-series-level imputation (Strategy D) would require inventing a signal with no basis in that ROI's own measured data (a flat series contains no partial information to interpolate from) — assessed as not actionable, not merely undesirable.
- Candidate pooled-imputation values (Strategy C) are numerically close regardless of method (mean vs. median differ by 0.005 on average, 0.035 at most, across all 2,014 affected features) — the *choice of statistic* is not the hard part of Strategy C; the *fact that it would be invented at all* is.

## B. What the Paper Specifies

AAL-116 → Pearson correlation → Fisher-Z → lower triangle → 6,670 features → SVM-RFE → 1,000 features → SSAE. Confirmed (Phase 4.1) directly from the paper's own Methods text.

## C. What the Paper Does Not Specify

Any handling of zero-variance ROIs, undefined correlations, NaN values, ROI removal, or subject removal beyond the stated FD > 0.2mm threshold. This remains the case after this phase's investigation — no new paper evidence was found or invented.

## D. Strategies That Preserve the Original 780/116/6,670 Structure

- **Strategy C** (feature-wise imputation) is the only strategy that preserves all three dimensions simultaneously (780 subjects, 116 ROIs, 6,670 features) with a finite matrix — but only *conditionally*, pending a specific, leakage-safe implementation being formally defined (§F).
- Strategy A preserves 116/6,670 but not 780 (→733). Strategy B preserves 780 but not 116/6,670 (→97/4,656). Strategy F converges to Strategy B if pursued to full resolution.

## E. Strategies That Introduce Assumptions

Every strategy that produces a finite matrix introduces at least one assumption absent from the paper:
- **A**: the exclusion threshold ("≥1 constant ROI disqualifies a subject") is a project choice.
- **B**: the ROI-removal list, while data-driven, is still a project-introduced exclusion criterion.
- **C**: the imputation statistic itself (mean, median, or otherwise) stands in for a value that was never measured — this is **statistically convenient, not paper-faithful**. Explicitly: using an observed-data pooled mean is a common, defensible *statistical* practice in general, but it is not what the paper describes doing, and adopting it would be a project decision presented as if it were part of the reproduction, which it is not, unless labeled as a deviation.
- **D, E**: not actionable / not mathematically viable, as established in §A.
- **F**: converges to Strategy B's assumption at the only threshold that actually resolves the problem.

## F. Leakage-Safe Implementation, If a Strategy Were Selected

- **A, B, F (at full threshold)**: leakage-free by construction — exclusion lists are deterministic and independent of any train/test split or diagnosis label.
- **C**: **leakage-safe only if implemented with strict fold-scoping** — imputation statistics must be computed from training-fold subjects' observed values only, and applied unchanged to validation/test subjects, mirroring this project's existing `holdout_guard.py` discipline from the earlier ML phases (documented in full in the script log; not implemented or trained here). Computing a single pooled statistic across all 780 subjects (as this phase's diagnostic simulation did, for characterization purposes only) would leak information from every subject, including future test subjects, into every other subject's feature values, and must never be used as the actual pipeline's imputation method.
- **D, E**: not applicable (not viable strategies).

## G. Additional Finding This Phase: Missingness Is Not Uniform Across Sites

Site-level missingness rate vs. the cohort-wide expected rate (6.03%) if uniformly distributed:

| Site | Observed | vs. expected (6.03%) |
|---|---|---|
| SBL | 73.9% | far above |
| CMU | 50.0% | far above (n=2, small site) |
| PITT | 31.7% | above |
| CALTECH | 20.0% | above |
| SDSU | 15.4% | above |
| MAX_MUN, KKI, NYU, TRINITY | 2.4–5.0% | at/below |
| 11 other sites | 0% | below |

This is descriptive, not causal (consistent with Phase 4.1/4.2's findings, not a new causal claim). It matters directly for Strategy C: **pooled imputation would silently smooth over a real, measurable site-specific pattern rather than represent it** — a consequence to weigh, not a reason to avoid reporting it here.

## Exact Human Decision Required

None of Strategies A, B, or C (the three viable candidates) is paper-specified, and each has a different, real cost:
- **A** costs 6% of subjects and severely thins SBL/PITT.
- **B** costs 30% of the feature space for all 780 subjects, including the 733 with no data problem.
- **C** costs neither subjects nor features, but requires accepting an invented value for 0.27% of entries, requires a formally-specified leakage-safe implementation before it could ever be used in a real ML pipeline, and would mask the SBL/PITT pattern documented in §G.

**This phase does not select among A, B, or C.** That selection — and, if C is chosen, the exact leakage-safe implementation to formalize — is the decision point handed back for human approval.

---

**PHASE 5B STATUS: PASS — INVESTIGATION COMPLETE**

**PHASE 5: BLOCKED PENDING HUMAN METHODOLOGICAL DECISION**
