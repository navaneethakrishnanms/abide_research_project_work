"""Preprocessing component for the CC200 B0 baseline: demographics median-imputed + standardized, then StandardScaler on [connectivity | demographics].
Identical to the order of operations in phase2c_baseline_cv.py; as a transformer it can be fitted on the outer-training rows (arm A) or inside every inner fold (arm B)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class DemoFCPrep(BaseEstimator, TransformerMixin):
    """X = [ connectivity features | age | sex ]  (last two columns are demographics)."""

    def fit(self, X, y=None):
        F, D = X[:, :-2], X[:, -2:].copy()
        self.med_ = np.nanmedian(D, axis=0)
        D = np.where(np.isnan(D), self.med_[None, :], D)
        self.ds_ = StandardScaler().fit(D)
        self.fs_ = StandardScaler().fit(np.hstack([F, self.ds_.transform(D)]))
        return self

    def transform(self, X):
        F, D = X[:, :-2], X[:, -2:].copy()
        D = np.where(np.isnan(D), self.med_[None, :], D)
        return self.fs_.transform(np.hstack([F, self.ds_.transform(D)]))
