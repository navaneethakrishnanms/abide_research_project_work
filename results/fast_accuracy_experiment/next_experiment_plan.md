# Next experiment plan (not launched)

## Single recommended experiment: training-fold site-wise feature centering
**Why this one.** Eight models on two atlases all land at 61-65% (differences smaller than fold SD), so more classifier variants will not help. ABIDE's dominant non-diagnostic variance is acquisition site; per-site accuracy of the best model varies 42%-92%. A site-prior baseline scores only 52.7%, so site is not a label shortcut -- the aim is to remove site-specific feature offsets, not to exploit site.

**Design (to document before running).**
- Same 600 development subjects, same outer 5 folds, same inner 3-fold, same Pipeline.
- New Pipeline step after Fisher-Z: subtract the **per-site mean of the training rows** from every feature. A test subject is shifted by the training mean of its own site (all 19 sites appear in training; sites with fewer than ~5 training subjects fall back to the global training mean). No test rows contribute to any statistic. No outer-test labels or site labels beyond the subject's own site id are used for fitting.
- Candidates: LR (L2) and Linear SVM on AAL-116 (best-behaved, fastest), plus RBF SVM on CC200 -- 3 candidates, each compared with its uncentered counterpart from this run. This is a paired comparison, not a new search.
- Report the paired per-fold difference and its SD. Do not claim improvement unless the difference is consistent across folds; a 5-fold design cannot establish significance.
- Estimated runtime: about 2-3 minutes (the whole current experiment took 136 s).

## Evaluation design needed for any final number
To get an unbiased estimate, the choice among candidates (model x atlas x site-centering on/off) must occur inside each outer training fold (inner CV picks the candidate, refit, one test). Estimated runtime about 3-6 minutes at this scale. Until then the best score is exploratory. The locked 170-subject holdout stays unopened; any evaluation on it should be a single, pre-declared run of one frozen pipeline.

## Not recommended now
Deep-network retraining, repeating the 95-unit nested run, or exhaustive hyperparameter searches: no evidence in these results that they would move accuracy above the 65-70% range seen so far.
