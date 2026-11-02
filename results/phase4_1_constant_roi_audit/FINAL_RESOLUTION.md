# Phase 4.1 Final Resolution Report — Constant-Variance AAL ROIs

## 1. Executive Summary

47 of 780 authoritative Phase-2 subjects (6.03%) have at least one AAL-116 ROI with exactly zero variance in their time series — meaning Pearson correlation is mathematically undefined for any FC pair involving that ROI for that subject. The affected 19 ROIs cluster tightly in two anatomically and mechanistically distinct groups: (a) orbitofrontal regions, a well-known susceptibility-artifact zone in EPI acquisition, and (b) inferior cerebellum + vermis, consistent with field-of-view truncation. The problem is **heavily site-concentrated** — SBL (74% of its subjects) and PITT (32%) account for 64% of all affected subjects — which is strong evidence against a random/parsing artifact and strong evidence for a site/protocol-specific acquisition cause. An independent second parser exactly reproduced every sampled affected file (30/30 match), ruling out a bug in this project's own code. **The paper's Methods section does not specify any handling for this situation.** Separately, and importantly for the reproduction effort as a whole: **the paper's central 98.2% accuracy result is currently under a published Expression of Concern for data leakage in its SVM-RFE step** (feature selection performed on the full dataset before cross-validation) — a fact independently verified via PubMed/PMC/ScienceDirect, not part of this ROI investigation but directly relevant to how this reproduction should be scoped going forward. **Phase 5 is NOT READY to begin**: a methodological decision on how to handle the 47 affected subjects is required, and this audit deliberately does not make that decision — see §14.

## 2. Authoritative Cohort Confirmation

- Source: `results/phase2/fd_filtered_subjects.csv` (unchanged; fingerprint-verified, see §9 and `phase4_1_final_audit.log`).
- 780 subjects, 351 ASD, 429 Control, 0 duplicates — re-verified directly in this audit.
- AAL data source: `abide_pcp/ABIDE_pcp/cpac/filt_global/*_rois_aal.1D`, 871 files on disk, 780/780 Phase-2 subjects covered, 0 missing.

## 3. Number of Affected Subjects

**47 of 780 (6.03%)**, computed directly by re-reading every one of the 780 subjects' `.1D` files and testing each of the 116 ROI columns for exactly-zero variance (`np.var(column) == 0.0`). Full list: `constant_roi_subjects.csv`.

## 4. Number of Affected ROIs

**19 of 116 (16.4%)**.

## 5. Exact Affected ROI Indices

`6, 9, 10, 16, 25, 27, 28, 87, 101, 102, 103, 104, 105, 106, 107, 108, 114, 115, 116`

## 6. Verified ROI Names

Cross-validated against three independent sources (see `aal116_roi_mapping.csv` for full detail and per-row citation): (S1) nilearn 0.14.0's locally-installed package documentation, which gives worked examples confirming codes 2001→Precentral_L and 5021→Lingual_L; (S2) an independently retrieved AAL ROI_MNI_V4 code/name table matching our file header exactly for codes 2001-4101; (S3) a DPARSF-forum reference (restfmri.net) confirming codes 9081/9082/9100. All 116 codes were also directly extracted from the `.1D` file headers themselves (identical across all 871 files — confirmed programmatically) and matched exactly, position-for-position, against the standard AAL systematic numbering scheme validated by S1-S3.

| ROI index | AAL code | Region name | Cluster |
|---|---|---|---|
| 6 | 2112 | Frontal_Sup_Orb_R | Orbitofrontal |
| 9 | 2211 | Frontal_Mid_Orb_L | Orbitofrontal |
| 10 | 2212 | Frontal_Mid_Orb_R | Orbitofrontal |
| 16 | 2322 | Frontal_Inf_Orb_R | Orbitofrontal |
| 25 | 2611 | Frontal_Med_Orb_L | Orbitofrontal |
| 27 | 2701 | Rectus_L | Orbitofrontal |
| 28 | 2702 | Rectus_R | Orbitofrontal |
| 87 | 8211 | Temporal_Pole_Mid_L | Temporal pole (susceptibility-adjacent) |
| 101 | 9051 | Cerebelum_7b_L | Inferior cerebellum |
| 102 | 9052 | Cerebelum_7b_R | Inferior cerebellum |
| 103 | 9061 | Cerebelum_8_L | Inferior cerebellum |
| 104 | 9062 | Cerebelum_8_R | Inferior cerebellum |
| 105 | 9071 | Cerebelum_9_L | Inferior cerebellum |
| 106 | 9072 | Cerebelum_9_R | Inferior cerebellum |
| 107 | 9081 | Cerebelum_10_L | Inferior cerebellum |
| 108 | 9082 | Cerebelum_10_R | Inferior cerebellum |
| 114 | 9150 | Vermis_8 | Inferior vermis |
| 115 | 9160 | Vermis_9 | Inferior vermis |
| 116 | 9170 | Vermis_10 | Inferior vermis |

**This directly corrects an earlier working assumption** ("do not assume cerebellar/vermis merely because indices are near the end of the list") — the assumption turned out to be substantively correct for 12 of the 19 ROIs (101-108, 114-116), but the remaining 7 (orbitofrontal cluster + one temporal pole ROI) are anatomically distinct and were only established by actually performing this mapping, not by guessing from index position.

## 7. Subject / Site / Diagnosis Distribution

- Diagnosis: 21 ASD affected (of 351, 6.0%), 26 Control affected (of 429, 6.1%) — **essentially identical rate in both groups**, no diagnosis-linked pattern (`constant_roi_diagnosis_summary.csv`).
- Site (`constant_roi_site_summary.csv`), sites with any affected subjects:

| Site | Total subjects | Affected | % of site |
|---|---|---|---|
| SBL | 23 | 17 | 73.9% |
| PITT | 41 | 13 | 31.7% |
| NYU | 168 | 5 | 3.0% |
| SDSU | 26 | 4 | 15.4% |
| CALTECH | 15 | 3 | 20.0% |
| MAX_MUN | 40 | 2 | 5.0% |
| CMU | 2 | 1 | 50.0% |
| KKI | 29 | 1 | 3.4% |
| TRINITY | 42 | 1 | 2.4% |
| (11 other sites) | — | 0 | 0.0% |

SBL and PITT together account for 30 of 47 affected subjects (64%) despite being only 64 of 780 subjects (8.2%) cohort-wide — a striking, non-random concentration.

## 8. Constant-Value Analysis

- **100% of affected subject-ROI pairs (124 total instances) have a constant value of exactly 0.0** — no other repeated constant was observed anywhere (`constant_roi_details.csv`).
- 27 of the 47 affected subjects have more than one constant ROI simultaneously (range 1-12 per subject); the single worst subject (SBL, subject 51558) has 12 constant ROIs, nearly the entire inferior-cerebellum+vermis block plus the temporal-pole ROI.
- **FACTS OBSERVED**: exact-zero constant values, strong site concentration (SBL/PITT), strong anatomical clustering (orbitofrontal + inferior cerebellum/vermis), co-occurrence within subjects, no diagnosis association, parsing independently verified correct.
- **POSSIBLE INTERPRETATIONS (not proven, offered as evidence-consistent hypotheses only)**: (a) the orbitofrontal + temporal-pole cluster is consistent with the well-documented EPI susceptibility-artifact/signal-dropout zone near air-tissue interfaces (sinuses); (b) the inferior-cerebellum/vermis cluster is consistent with field-of-view truncation — i.e., the most inferior slices of some acquisition protocols not fully covering the cerebellum — which would explain both the anatomical clustering (all affected regions are physically adjacent, at the bottom of the brain) and the site concentration (a site-specific scan protocol/FOV setting); (c) a CPAC-side masking/registration artifact specific to certain sites' data cannot be ruled out and was not directly testable from ROI time series alone. No definitive cause determination is made here — this would require access to the original 4D volumes or CPAC processing logs for the affected sites, which are outside this project's available data (per Phase 3.5).

## 9. Parsing Integrity Result

**PASS.** 30 affected files were independently re-parsed with a second, manually-written parser (raw `readlines()`, skip `#`-header, whitespace-split, `float()`) and compared against the original `numpy.loadtxt`-based parsing used throughout this project. All 30 matched exactly in both shape and values (`np.allclose`). No duplicate file paths, no subject ID mapping to more than one file, and all 871 files share an identical 116-code header (ROI ordering is consistent project-wide). **The constant-ROI finding is not an artifact of this project's parsing, orientation, or subject-mapping code.**

## 10. Pearson FC Impact

Simulation only — no FC matrices were computed or saved.

- Total undefined FC entries across the cohort: **14,025** (of 780 × 6,670 = 5,202,600 possible entries cohort-wide, 0.27%).
- Affected subjects: 47/780 (6.03%).
- Undefined pairs per subject: max 1,314 (19.7% of that one subject's 6,670-dim vector), mean 17.98 (skewed by the affected subset), **median 0** (the typical subject, including all 733 unaffected ones, has zero undefined pairs).
- Full per-subject detail: `fc_undefined_impact.csv`; aggregate: `fc_undefined_summary.csv`.

## 11. What the Paper Specifies

Directly verified from the paper's full text (Vidya et al., eClinicalMedicine 2025, PMC12573454): the Methods section states only that "Applying an FD threshold of 0.2 mm yielded 884 usable samples (408 ASD, 476 controls)" across 17 sites. **No mention anywhere in the Methods of zero-variance ROIs, constant/flat time series, undefined or NaN Pearson correlations, ROI exclusion, or any subject exclusion criterion beyond the FD threshold.**

## 12. What the Paper Does NOT Specify

**"Paper does not specify handling for this case."** There is no described procedure for zero-variance ROIs, undefined correlations, missing ROI signal, or any post-FD-filtering data-quality exclusion. This gap is not filled with an invented procedure anywhere in this project.

**Additional critical context, verified independently (PubMed, PMC, ScienceDirect):** the paper's reported 98.2% accuracy result is currently subject to a **published Expression of Concern** (eClinicalMedicine, 2026 Jun 25, vol. 97, p. 104052), stemming from a companion critique ("Data leakage in feature selection invalidates reported classification accuracy for autism diagnosis from fMRI," PMC13382443) identifying that the paper's SVM-RFE feature selection (6,670→1,000 features) was performed on the **entire dataset before cross-validation** — textbook feature-selection data leakage. The original authors have confirmed a performance drop upon re-analysis and are working on a correction as of this writing. **This does not change the ROI-constant-variance findings above, but it materially affects how much weight the paper's 98.2% figure should carry as a reproduction target**, and is exactly the class of leakage this project's own ML pipeline work (Phase 2 holdout-guard infrastructure) was independently built to avoid.

## 13. Handling Options

See `handling_options_analysis.md` for the full comparison of Options A-E (remove subjects / remove ROIs / retain with NaN / documented numeric convention / other evidence-based strategy). No option is ranked or selected there.

## 14. Recommended Next Methodological Decision

**No methodological decision is made in this report.** The evidence gathered here (site concentration, anatomical clustering, all-zero constant values, confirmed parsing integrity, quantified FC impact, and confirmation that the paper is silent on this exact scenario) is sufficient to *inform* a decision, but the decision itself — which of Options A-E (or a variant) to adopt — is a methodological judgment call this audit is explicitly instructed not to make unilaterally. **Evidence is insufficient, by design, for this audit to select a final method** — that selection requires a human decision about acceptable reproduction fidelity, informed by the trade-offs documented in `handling_options_analysis.md` and by the Expression-of-Concern context in §12.

## 15. Explicit Statement: Is Phase 5 Ready?

**NOT READY.** Phase 5 (Pearson FC computation) cannot proceed until a handling decision is made for the 47 affected subjects / 19 affected ROIs, because Pearson correlation is undefined for 14,025 specific FC entries and no numeric value can be written for them without either (a) a chosen convention (Option D, currently unjustified by the paper), (b) an accepted NaN representation with a defined downstream plan (Option C, whose SVM-RFE/SSAE compatibility is unresolved), or (c) removing the affected subjects or ROIs (Options A/B, each with real costs documented above).

## 16. Unresolved Issues

1. **Primary blocker**: no handling decision made for the 47 affected subjects (per §14 — awaiting your direction).
2. Root cause of the constant-ROI pattern (susceptibility artifact vs. FOV truncation vs. CPAC-side masking) is not definitively established — only anatomically/statistically consistent hypotheses are offered (§8), since the original 4D volumes and CPAC processing logs for SBL/PITT are not available in this project (confirmed in Phase 3.5: only 54/780 subjects have any local raw functional NIfTI, none confirmed CPAC, and none from SBL or PITT specifically — not independently checked in this audit but consistent with Phase 3.5's findings).
3. Downstream compatibility of any NaN-retention strategy (Option C) with a faithful SVM-RFE/SSAE reproduction is unknown, since the paper doesn't specify NaN handling and this audit did not attempt to inspect any SVM-RFE/SSAE reference implementation.
4. The paper's own headline accuracy result is under active correction (Expression of Concern) — the target this reproduction is calibrated against may itself change; this is a project-level consideration, not something this audit resolves.
