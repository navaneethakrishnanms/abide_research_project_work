# SVM-RFE NaN Compatibility Investigation

**No SVM-RFE was trained on this project's FC data. This is an isolated compatibility check using
small synthetic arrays against the installed scikit-learn (1.7.2).**

## Empirical results

| Component | Accepts NaN? |
|---|---|
| `sklearn.preprocessing.StandardScaler` | YES |
| `sklearn.svm.SVC` (the classifier SVM-RFE wraps) | NO |
| `sklearn.feature_selection.RFE(SVC(...))` | NO |

`SVC.fit()` raised: `ValueError: Input X contains NaN.
SVC does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values`
`RFE(SVC).fit()` raised: `ValueError: Input X contains NaN.
SVC does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values`

## What would mathematically happen

`SVC` (and therefore `RFE(SVC, ...)`, the standard scikit-learn SVM-RFE construction) validates its
input with scikit-learn's internal `check_array`, which by default requires all values to be finite.
Passing the 780×6,670 diagnostic FC matrix from this phase directly into `RFE(SVC(...))` would raise
`ValueError: Input X contains NaN` immediately, before any fitting occurs — it would not silently
produce a degraded or approximate result; it fails hard.

## Conclusion

**"An additional missing-value handling decision is required before SVM-RFE."**

This phase does not invent that decision. The options remain those enumerated in Phase 4.2's
`METHODOLOGICAL_DECISION.md` (retain-with-NaN requires a downstream imputation/removal step before
SVM-RFE specifically, even if NaN is preserved through Phase 5A and Fisher-Z; exclude-subjects;
exclude-ROIs; or a documented numeric convention — none of which is paper-specified).
