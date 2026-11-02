# Test-site feature centering for leave-one-site-out evaluation (single exploratory experiment)

**Predeclared success rule: FAILED.** Three of the four conditions were met; the fourth was not. The rule was not changed afterward. Results are development estimates on 600 subjects in a **transductive** setting (the whole held-out site is available as an unlabeled batch), not independent-holdout performance. The restricted holdout, its ID file and CC200 were not used.

## Protocol (fixed before any score was seen)
- After the element-wise Fisher-Z (clip 1e-6, undefined entries kept as NaN), every site, training or held-out, is centered by subtracting its own site mean vector (nan-mean over all subjects of that site, taken as one batch). Training sites use their own means; the held-out site uses only its own unlabeled rows. No labels, predictions or other sites' rows enter any centering.
- Supervised steps on the training sites only: mean imputation, StandardScaler, SelectKBest and L2 logistic regression (balanced), with the frozen grid (C {1e-4..1e-1} x k {500, 2000, all}) and inner stratified 3-fold CV (seed 42).
- **Transductive, no minimum batch size:** a held-out subject's centered features depend on every other subject of its site and on itself (shrinking its own deviation by 1 - 1/n); there is no dependence across sites, and no training row depends on the held-out site. The method cannot score a single subject. No minimum batch size was enforced (uniform rule); 6 of the 19 sites have n < 20, and CMU (n = 2) is degenerate (its two centered subjects become exact negatives of each other).
- Arms: CENTERED; UNCENTERED_SAME_PIPELINE (same code without centering, to isolate the centering effect; it matches the frozen baseline to 1e-4 in AUC and exactly in accuracy); FROZEN baseline (saved leave-site-out predictions). Also a five-fold control with the same centering (not part of the rule).
- Implementation checks (passed): another site's rows do not change a site's centered features; the held-out site's rows do not change any other site's; all site means are zero after centering (max 6.6e-16); labels are not an argument of the centering function.

## Pooled leave-one-site-out results (600 out-of-site predictions)
| Arm | Accuracy | Bal. acc. | Sens. | Spec. | F1 | Pooled AUC* | Mean abs gap, predicted vs true ASD share (13 sites n >= 20) |
|---|---|---|---|---|---|---|---|
| Frozen baseline | 0.622 | 0.618 | 0.586 | 0.651 | 0.580 | 0.684 | 0.143 |
| Uncentered, same pipeline | 0.622 | 0.618 | 0.586 | 0.651 | 0.580 | 0.684 | 0.143 |
| **Centered** | **0.657** | **0.656** | 0.646 | 0.666 | 0.627 | **0.708** | **0.085** |

*Pooled AUC combines scores from 19 models. Gains over the frozen baseline: accuracy +0.035, balanced accuracy +0.037, pooled AUC +0.025.

## Rule evaluation
| Condition | Result |
|---|---|
| Pooled accuracy up by >= 0.02 | Met (+0.035) |
| Pooled balanced accuracy up by >= 0.02 | Met (+0.037) |
| Pooled AUC not lower | Met (+0.025) |
| Accuracy higher at a majority (>= 7) of the 13 sites with n >= 20 | **Not met: higher at 6, lower at 1 (UM_2), equal at 6** |

## Per-site results (full table: `per_site_results.csv`)
Higher: NYU +0.045, UM_1 +0.065, UCLA_1 +0.023, MAX_MUN +0.032, KKI +0.174, SDSU +0.150. Equal: USM, TRINITY, PITT, YALE, LEUVEN_1, LEUVEN_2. Lower: UM_2 -0.083. Smaller sites: STANFORD +0.053, SBL +0.167, OLIN -0.063, CALTECH 0, UCLA_2 0, CMU +0.500 (two subjects, an artifact of the degenerate centering). The gains are concentrated at sites where the frozen model's predicted-ASD share was far off: SBL 6% -> 44% (true 56%), PITT 24% -> 42% (true 48%), KKI 65% -> 39% (true 26%), LEUVEN_1 5% -> 59% (true 50%, accuracy unchanged), STANFORD 63% -> 37%.

## Interpretation
- **Where the gain comes from (observed):** mean within-site AUC over the 13 larger sites is unchanged (0.699 frozen vs 0.696 centered; it rose at 6 sites and fell at 7), while the predicted-versus-true ASD-share gap fell from 0.143 to 0.085 and sensitivity rose 0.586 -> 0.646. So the improvement is a correction of site-specific score shifts, not better within-site discrimination, and the pooled AUC rise reflects better alignment of scores across sites.
- **The five-fold control shows no change** (centered accuracy 0.637, AUC 0.692, vs 0.642 and 0.692 for the saved baseline): centering did not help where sites are already represented in training, which fits the shift explanation.
- **A correction to my earlier expectation:** I had stated that the benefit was bounded by the 2.0-point gap between leave-site-out and five-fold accuracy. That was wrong: centered leave-site-out accuracy (65.7%) exceeds the five-fold baseline (64.2%), because the transductive correction uses information from the held-out site that random folds' baseline does not use.
- **Caveats on the size of the gain:** excluding CMU, the pooled accuracy gain is +0.033 (descriptive, post hoc; it does not change the rule outcome). Roughly 20 additional subjects are classified correctly out of 600; gains at KKI, SDSU and SBL are 3-4 subjects each, and per-site intervals are wide (+/-0.10 to 0.20). No significance test is claimed.
- **Prevalence confound:** site ASD share varies from 26% to 60%; centering by the site mean removes some diagnosis-related signal along with the site offset (UM_2 lost accuracy; high- and low-prevalence sites are the risk).

## What the evidence supports
Test-site unlabeled feature centering improved pooled leave-one-site-out accuracy by about 3.5 points and pooled AUC by 0.025 in this one exploratory run, but the predeclared rule failed because the per-site majority condition was not met. It requires transductive access to a whole unlabeled site (at least about 20 subjects for a stable mean), cannot be used for single new subjects, and was evaluated on a single pass at one seed on 600 subjects with an overlap to earlier design decisions. It does not support a claim of 80% accuracy or of independent-holdout performance. I did not launch any follow-up.
