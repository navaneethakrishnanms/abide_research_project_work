# Handling Options Analysis — Constant-Variance AAL ROIs

**Status: informational comparison only. No option has been selected. No data has been modified, removed, or imputed.**

## Grounding facts (all verified, see `phase4_1_summary.json` and companion CSVs)

- 47 of 780 subjects (6.03%) have ≥1 AAL ROI with exactly zero variance across their time series.
- 19 of 116 ROIs are implicated across those 47 subjects.
- All constant values are exactly **0.0** (not some other repeated constant) — 100% of affected subject-ROI pairs.
- 27 of the 47 affected subjects have **more than one** constant ROI simultaneously (up to 12 in one subject).
- Affected subjects are heavily concentrated at two sites: **SBL (17/23 subjects, 74%)** and **PITT (13/41, 32%)** — together 30 of 47 affected subjects (64%). Five other sites have 1-5 affected subjects each; twelve sites have zero.
- Total undefined Pearson FC entries across the cohort: **14,025** out of 780 × 6,670 = 5,202,600 possible entries (0.27% of all entries cohort-wide). Per-subject impact is highly skewed: median = 0 undefined pairs (most subjects, including most of the 780, are entirely unaffected), mean among the whole cohort = 17.98, but the worst single subject (SBL, 12 constant ROIs) has 1,314 undefined pairs — **19.7% of that one subject's 6,670-dim feature vector**.

## OPTION A — Remove affected subjects

**Mechanics:** Drop the 47 subjects with ≥1 constant ROI from the Phase-4/5 cohort.

- Subjects lost: 47 (6.03% of 780).
- ROIs lost: 0.
- Effect on 6,670-dim representation: unaffected — every remaining subject has a complete, valid feature vector.
- Effect on class balance: removes 21 ASD (of the affected 47 per the ASD/Control breakdown: ASD-affected=21, Control-affected=26) and 26 Control, i.e. 351→330 ASD, 429→403 Control. Ratio shifts only marginally (0.4498→0.4501 ASD fraction — negligible balance effect).
- Site-distribution consequences: **material** — SBL would lose 17 of its 23 subjects (74%), leaving only 6; PITT would lose 13 of 41 (32%), leaving 28. This meaningfully thins two sites' representation in the final cohort, which matters for the paper's stated 17-site, 5-country design (Phase 1 already noted our site count doesn't exactly match the paper's).
- Reproducibility: straightforward and fully reproducible (deterministic subject list, already enumerated in `constant_roi_subjects.csv`).
- Compatibility with the paper: the paper does not describe this step (see Phase 4.6) — this would be a **project-introduced** exclusion criterion beyond FD filtering, not a paper-specified one.
- Arbitrary assumptions introduced: **yes** — no stated threshold in the paper for "how many constant ROIs justify exclusion"; the cutoff here (≥1) is a project choice.
- Consideration: viable, but changes the reported cohort size away from the Phase-2-verified 780, which itself is already a deliberate, audited reproduction artifact.

## OPTION B — Remove affected ROIs

**Mechanics:** Drop the 19 implicated AAL regions entirely, reducing the atlas to 97 ROIs for all 780 subjects.

- Subjects lost: 0.
- ROIs lost: 19 (16.4% of AAL-116).
- Effect on 6,670-dim representation: **fundamental** — the paper's feature space is explicitly built on 116×115/2 = 6,670 lower-triangle AAL features; using 97 ROIs would produce 97×96/2 = 4,656 features, a different-shaped representation than the paper's SVM-RFE/SSAE pipeline is defined for.
- Effect on class balance: none directly (no subjects removed).
- Site-distribution consequences: none directly (no subjects removed), but every subject — including the 733 with no constant-ROI problem at all — would lose real, valid signal from those 19 regions, including the well-characterized orbitofrontal and cerebellar/vermis regions, which are neuroanatomically relevant regions in ASD literature.
- Reproducibility: fully reproducible (fixed ROI list).
- Compatibility with the paper: **breaks** the stated 6,670-feature design outright — the paper does not describe reducing the atlas.
- Arbitrary assumptions introduced: **yes** — discards clean, valid data from 733 unaffected subjects to solve a problem that, for them, doesn't exist.
- Consideration: this option's cost (losing 16% of the atlas and deviating from 6,670 features for the entire cohort) is disproportionate to a problem affecting only 6% of subjects.

## OPTION C — Retain subjects/ROIs, represent undefined FC as NaN

**Mechanics:** Compute Pearson FC normally; wherever a pair involves a constant ROI, the correlation is mathematically undefined and is stored as `NaN` rather than a numeric value.

- Subjects lost: 0. ROIs lost: 0.
- Effect on 6,670-dim representation: preserved in shape (every subject still has a 6,670-length vector), but **733 of 780 subjects (94%) have zero NaNs and only 47 (6%) have any**, and even among those 47 the NaN burden is usually small (median non-zero case is well under 20% of that subject's vector; only the single worst case reaches ~20%).
- Effect on class balance: none — no subjects removed.
- Site-distribution consequences: none — no subjects removed, though SBL/PITT feature vectors would disproportionately carry the NaN entries.
- Reproducibility: fully reproducible and fully transparent — the NaN pattern is deterministic and traceable back to `constant_roi_matrix.csv`.
- Compatibility with the paper: the paper's SVM-RFE and SSAE steps are not described as NaN-tolerant (see Phase 4.6) — **downstream compatibility is unresolved**, not proven safe, without knowing whether the paper's own implementation (or a faithful reproduction of it) handles missing entries, and if so how.
- Arbitrary assumptions introduced: **minimal at this stage** — NaN is the mathematically honest representation of "undefined," not a invented number — but a decision is still required for every step downstream that cannot itself accept NaN (SVM-RFE, standard SSAE input layers, etc.), so this option defers rather than resolves the methodological question.
- Consideration: the least lossy, most mathematically honest immediate representation, but it pushes the real decision (what happens to NaN at SVM-RFE/SSAE time) one step later rather than answering it.

## OPTION D — Documented numerical convention for undefined correlations

**Mechanics:** Replace undefined correlations with a specific, justified numeric convention (e.g., 0, indicating "no measurable linear relationship" — a common convention in some FC pipelines) — but **only if the paper or an established reproduction protocol specifies this.**

- Subjects lost: 0. ROIs lost: 0.
- Effect on 6,670-dim representation: preserved in shape and fully numeric (no downstream NaN-handling burden).
- Effect on class balance: none.
- Site-distribution consequences: none directly, though it silently treats SBL/PITT subjects' missing cerebellar/vermis signal as "zero correlation," which is a substantive, non-neutral assumption about brain connectivity where no data actually exists.
- Reproducibility: fully reproducible if the convention is fixed and documented.
- Compatibility with the paper: **UNRESOLVED — Phase 4.6 found the paper does not specify this.** Using 0 (or any other convention) here would be a project-introduced convention, not a paper-specified one, unless independently justified.
- Arbitrary assumptions introduced: **yes, significant** — substituting 0 (or any fixed value) for "undefined" conflates "no data" with "no relationship," which is a real methodological claim, not a neutral placeholder.
- Consideration: **this option is explicitly flagged as requiring justification the project does not currently have** (per Phase 4.6's finding). It should not be adopted merely because it is numerically convenient for downstream steps.

## OPTION E — Other evidence-based strategy (not yet adopted, listed for completeness)

Given the strong **site-concentration** finding (SBL 74%, PITT 32% of their subjects affected, versus near-zero at other sites) and the fact that constant values are 100% exactly zero (consistent with field-of-view truncation or a masking artifact rather than random signal noise), a possible additional strategy — **not evaluated in depth here, and not chosen** — would be a **site-aware or region-aware sensitivity analysis**: report Phase 5 results both with and without SBL/PITT (or both with and without the 19 affected ROIs), to characterize how much the constant-ROI issue could be driving any downstream result. This is a robustness-check strategy, not a data-handling strategy, and would supplement rather than replace a decision among Options A-D.

## Explicit non-ranking statement

No option above is recommended as "best." Each has a different, real cost (subjects lost, ROIs lost, deviation from the paper's 6,670-feature design, or an unresolved downstream-compatibility question). The choice depends on (a) what the paper's own methodology actually specifies — which Phase 4.6 found it does **not** — and (b) a judgment call about acceptable reproduction fidelity that is not this audit's to make unilaterally.
