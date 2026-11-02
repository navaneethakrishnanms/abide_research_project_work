"""X2 arm D2: in-fold age/sex residualization of connectivity features (CC200). Label-free. Input X = [ features | age | sex (1/2) ]; output = residual features only.
fit(): OLS of every feature on [1, age_z, female] using the rows passed in; transform(): feature - fitted value from that row's own age and sex."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AgeSexResidualizer(BaseEstimator, TransformerMixin):
    def _design(self, X):
        age, sex = X[:, -2].astype(float), X[:, -1].astype(float)
        age = np.where(np.isnan(age), self.age_med_, age); sex = np.where(np.isnan(sex), self.sex_med_, sex)
        return np.column_stack([np.ones(len(X)), (age - self.age_mu_) / self.age_sd_, (sex == 2).astype(float)])

    def fit(self, X, y=None):                                  # y is deliberately ignored
        self.age_med_, self.sex_med_ = float(np.nanmedian(X[:, -2])), float(np.nanmedian(X[:, -1]))
        a = np.where(np.isnan(X[:, -2]), self.age_med_, X[:, -2]); self.age_mu_, self.age_sd_ = float(a.mean()), float(max(a.std(), 1e-6))
        self.coef_ = np.linalg.lstsq(self._design(X), X[:, :-2], rcond=None)[0]
        return self

    def transform(self, X):
        return X[:, :-2] - self._design(X) @ self.coef_
