"""Self-contained copy of the baseline transformer (identical to fast_accuracy_experiment.MeanImputeFisherZ)."""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MeanImputeFisherZ(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmean(X, axis=0)
        self.means_ = np.where(np.isnan(m), 0.0, m)
        return self

    def transform(self, X):
        X = np.where(np.isnan(X), self.means_[None, :], X)
        return np.arctanh(np.clip(X, -1 + 1e-6, 1 - 1e-6))
