# Option 4 — Documented Numerical Convention Analysis

## Question
Is there an established, defensible, **paper-specified** convention for representing a Pearson
correlation when one of the two input ROI time series has zero variance?

## Investigation performed
1. Re-checked the paper's full Methods text (already retrieved and quoted in Phase 4.1's
   `FINAL_RESOLUTION.md`, §11): no mention of zero-variance ROIs, undefined correlations, NaN
   handling, or any post-FD-filtering exclusion criterion.
2. Considered, and explicitly REJECTED as unjustified inventions (per the task's own prohibition):
   - **Substituting 0 for undefined correlation.** Mathematically, Pearson correlation of a constant
     series with anything is undefined (0/0 form — the denominator, the product of the two standard
     deviations, is exactly zero), not equal to zero by any derivation. Treating "undefined" as "zero
     correlation" is a substantive claim about the data (asserting no linear relationship) where in
     fact no relationship can be measured at all. This is qualitatively different from a true
     zero-correlation ROI pair, and conflating the two would corrupt the feature's meaning.
   - **Small epsilon noise injection** to force a nonzero-variance signal: this would be an invented
     synthetic signal not present in the original data, explicitly excluded by the task instructions,
     and not describable as a documented convention (it would depend on an arbitrary epsilon
     magnitude and RNG seed with no paper basis).
   - **Interpolation/imputation of the constant ROI's time series**: same objection — invents signal
     that was never measured, with no paper-specified method for doing so.
3. Searched for whether Fisher-Z transformation itself offers a defined extension for undefined
   input: it does not. Fisher-Z (`arctanh(r)`) is undefined wherever `r` is undefined (NaN in, NaN
   out) or at the boundary `r = ±1`; there is no standard mathematical convention that assigns a
   specific finite value to `arctanh(undefined)`.

## Mathematical meaning if a convention WERE adopted
Any convention that assigns a specific numeric value to an undefined correlation is, by definition,
encoding an assumption not derivable from the data itself. Whatever value is chosen becomes a
property of the *methodology*, not a measurement — this has direct interpretability consequences,
since Phase 5's downstream steps (SVM-RFE feature ranking, SSAE encoding) would then be partly
trained on values that do not represent observed brain activity for the affected entries.

## Compatibility with Fisher-Z
None of the three considered non-conventions are Fisher-Z compatible in a principled way:
0-substitution passes `arctanh(0) = 0` through cleanly (numerically convenient, but see above for
why this is not a neutral choice); noise/interpolation approaches would produce Fisher-Z values with
no traceable meaning.

## Appropriateness for this reproduction
**Not established.** No convention investigated here is paper-specified, and adopting one
unilaterally would be a project-introduced methodological choice, not a reproduction of the paper's
actual procedure.

## Final statement
**"No paper-specified numerical convention was established."**
