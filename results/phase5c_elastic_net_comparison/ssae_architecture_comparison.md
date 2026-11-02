# SSAE Architecture Compatibility — Strategies A / B / C

**No SSAE was trained in this phase. This document only assesses structural (input-dimension)
compatibility.**

## Paper / proposed architecture

`1000 -> 500 -> 100 -> 2 -> Softmax` (two output classes: ASD, Control).

## Strategy A (733 subjects, 116 ROIs, 6,670 features)

FC feature dimensionality is unaffected (still 6,670 before selection). Elastic Net Option 2
(rank-to-top-1000) can produce exactly 1,000 features from 6,670 candidates, so the paper's
`1000 -> 500 -> 100 -> 2` architecture is directly supportable. Fewer subjects (733) are available to
fit/validate the SSAE, which is a sample-size consideration, not an architectural one.

## Strategy B (780 subjects, 97 ROIs, 4,656 features)

4,656 > 1,000, so Elastic Net Option 2 can still select exactly 1,000 features from the reduced
candidate pool, and the paper's architecture remains directly supportable. The available candidate
feature pool is smaller (4,656 vs. 6,670), which may affect which 1,000 are selected, but does not by
itself prevent reaching the target count.

## Strategy C (780 subjects, 116 ROIs, 6,670 features, imputed values)

Same as Strategy A structurally (6,670 candidates, 1,000 selectable) — full subject count and full
feature space are retained. The distinction is that some of the 1,000 selected features may have been
partly constructed from imputed (not measured) values for the 47 affected subjects; this does not
change the architecture, but changes what a subset of the *input values* represent for those subjects.

## If Elastic Net Option 1 (natural nonzero count) is used instead

The resulting feature count is not guaranteed to be 1,000 under any of A, B, or C. If adopted, the
SSAE input layer would need to be resized to match whatever count results (documented as a
consequence, not silently absorbed into an unchanged "1000" label) — or Option 2's ranking procedure
would need to be applied on top of Option 1's selected set to force a count, which is simply Option 2
with an extra step.

## Conclusion

All of A, B, and C can structurally support the paper's `1000 -> 500 -> 100 -> 2 -> Softmax`
architecture, PROVIDED Option 2 (rank-to-top-1000) is used for feature selection. This is a
structural compatibility statement only — it says nothing about resulting model performance.
