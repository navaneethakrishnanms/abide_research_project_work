# XAI and ROAR Implications — Strategies A / B / C

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
