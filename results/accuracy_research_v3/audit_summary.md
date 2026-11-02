# Audit summary (accuracy_research_v3)

## Numbers verified from saved files
| Item | Verified value | Source |
|---|---|---|
| AAL-116 Fisher-Z Pearson + L2 logistic regression | 64.7% seed-averaged (64.2%, 64.7%, 65.3% by seed), AUC 0.694 | `accuracy_improvement_v1` control E0 (seed 42 identical to the frozen baseline) |
| AAL-116 tangent-space + Ledoit-Wolf | 65.2% seed-averaged (+0.5 points), AUC 0.711; improvement criteria failed (accuracy higher in 2-3 of 5 folds) | `accuracy_research_v2` |
| Elastic Net + SSAE (500-100-2) | 62.5-63.2% (3 seeds), 1/5 folds better than baseline; no improvement | `fast_safe_ssae` |
| Test-site centering | 65.7% pooled leave-one-site-out (62.2% frozen), AUC 0.708; **predeclared rule failed** (accuracy higher at 6 of 13 sites, 7 needed); transductive | `test_site_centering_experiment` |
| Prior CC200 results (~67%) | Phases 2c-2h: 676 train-list subjects, CC200, 10-fold x 3 seeds, logistic regression **with age and sex** and a threshold protocol -- not comparable with the FC-only 600-subject results | `phase2e_representations.py` |

## Demographic metadata (new audit, 600 development subjects)
- Age (years) and sex are **complete** (0 missing); the cohort file and the phenotype file agree exactly (maximum age difference 0.0; sex identical for all); sex coded 1/2 (binary female used); diagnosis from the phenotype file matches the manifest. Age 6.5-58.0 years.
- **Sex differs by diagnosis:** female 13.8% in ASD vs 20.8% in TC (chi-square p = 0.034; crude odds ratio of being female among ASD 0.61, Mantel-Haenszel within-site 0.67). **Age does not:** ASD 17.4 (SD 8.0), TC 16.9 (SD 6.7), Mann-Whitney p = 0.98; within sites the ASD subjects are slightly younger (AUC-equivalent 0.459, p = 0.08).
- **Both are strongly site-structured:** sex differs across sites (p = 7e-8; USM, TRINITY, LEUVEN_1 and SBL have no female subjects) and 48% of the age variance is between sites (mean age from 10.0 at STANFORD to 34.3 at SBL). Per-site table: `demographics_by_site.csv`. So a model using age and sex can partly encode which site (cohort) a subject comes from, not neurobiology.
- Legitimacy: age and sex are available at prediction time, are not diagnosis-derived and encode no target information; no post-diagnosis variable (treatment, severity scores, medication) was used.

## What can and cannot be claimed
Can: demographics are available and clean; the older 67% figures are not a like-for-like baseline. Cannot: independent-holdout performance (the holdout overlaps Phase 3-10 development), 85%, or that test-site centering is validated. All results here are exploratory development evidence.
