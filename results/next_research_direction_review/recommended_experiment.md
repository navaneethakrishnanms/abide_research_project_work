# Recommended experiment (one; not launched)

## Justification in one paragraph
Only one untested direction has a specific mechanism backed by observed data: under leave-one-site-out the baseline keeps its within-site ranking (mean within-site AUC 0.699 on 13 sites, pooled AUC 0.684) but misplaces the decision boundary at particular unseen sites (LEUVEN_1 4.5% predicted ASD vs 50% true, SBL sensitivity 0.10, PITT 24% vs 49%, KKI 65% vs 26%). Training-only thresholds cannot see this because random inner folds never contain an unseen site. A site-adaptation step that uses only the **unlabeled** features of the new site is the natural test.

## Experiment: test-site feature centering in leave-one-site-out (transductive site adaptation)
- **Cohort and evaluation:** the same 600 development subjects and the same 19 leave-one-site-out splits as `leave_site_out_baseline`; compare with its saved predictions (pooled accuracy 62.2%, balanced 61.8%, AUC 0.684).
- **Method (all fixed in advance, no tunable parameters):** after Fisher-Z, subtract from every subject the mean feature vector of that subject's own site: training subjects use their training-site means; the held-out site uses the mean of its own unlabeled rows. Then the unchanged baseline pipeline and grid (scaler, SelectKBest, L2 logistic regression, inner stratified 3-fold CV on the training sites). No test labels are used anywhere. Optional secondary arm with the identical rule applied to the 116-ROI scores only (re-centering each site's scores by their own mean) is not recommended, since it would force predicted ASD share toward a constant.
- **Predeclared success rule:** pooled leave-site-out accuracy and balanced accuracy both up by at least 0.02 against the saved baseline, pooled AUC not lower, and accuracy higher at a majority of the 13 sites with n >= 20. Report all results, per-site changes, and the predicted-vs-true ASD share by site. No significance test.
- **Runtime:** about 1-2 minutes. **Control:** repeat the five-fold evaluation with the same centering to confirm it does not change in-distribution performance (the earlier site-centering experiment found no gain there).

## Why it may help (hypothesis)
It targets the one observed failure mode of unseen-site prediction (site-specific score shift), uses no labels, adds no parameters, and is evaluated only on sites absent from training.

## Limits and what to expect
- **Bounded benefit:** the whole gap between leave-site-out and five-fold random CV is 2.0 accuracy points (62.2% vs 64.2%) and 0.009 AUC; even a complete repair would recover only about that much. This experiment cannot move accuracy toward 80%, and I do not expect more than a small change.
- **Transductive:** it needs a batch of unlabeled subjects from the new site (at least about 20 for a stable mean); it cannot score a single new subject. Sites with fewer than 20 subjects will be noisy, and CMU (2 subjects) is not meaningful.
- **Prevalence confound:** ASD share differs by site (26-60%); centering by site mean removes some diagnosis-related signal along with the site offset, which could lower accuracy at high- or low-prevalence sites.
- **Design caveat:** the idea was formed after inspecting leave-site-out per-site results, so it is exploratory; the success rule above is the guard against reading noise as an effect.

## If you prefer not to run it
No further useful experiment remains on the current data. The evidence supports documenting the baseline (about 64% five-fold and 62% leave-site-out development accuracy, AUC about 0.69) and moving to new data: raw images or reprocessing with TR and per-frame motion, more subjects, and an independent external cohort for an honest held-out test.
