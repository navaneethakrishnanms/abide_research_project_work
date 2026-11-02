"""ComBat-style empirical-Bayes location/scale harmonizer (parametric prior), label-free, used as a scikit-learn pipeline step.
Input columns of X: [ connectivity features | site code | age | female ]. Every parameter is learned from the rows given to fit() only.
Batch labels: >= 0 a fitted site, -1 the pooled "other" batch of small training sites, -2 = left unchanged (excluded / unseen site)."""
import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.insert(0, r"C:\abide")
from fast_accuracy_experiment import MeanImputeFisherZ  # noqa: E402

MIN_BATCH = 5
VAR_FLOOR = 1e-10
POOL, SKIP = -1, -2


def _eb(s_b, g_hat, d_hat2, g_bar, t2, a, b, tol=1e-6, max_iter=1000):
    """Iterative empirical-Bayes solution for one batch (Johnson et al. 2007, parametric)."""
    n = s_b.shape[0]
    g_old, d_old = g_hat.copy(), d_hat2.copy()
    g_new, d_new = g_old, d_old
    for _ in range(max_iter):
        g_new = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        d_new = (0.5 * ((s_b - g_new) ** 2).sum(0) + b) / (n / 2.0 + a - 1.0)
        if max(np.abs(g_new - g_old).max(), np.abs(d_new - d_old).max()) < tol:
            break
        g_old, d_old = g_new, d_new
    return g_new, d_new


class ComBatParams:
    def fit(self, Z, lab, cov):
        """Z (n x p) harmonization input, lab batch labels (see module doc), cov (n x k) covariates."""
        n, p = Z.shape
        use = lab != SKIP
        self.batches_ = sorted(set(int(b) for b in lab[use]))
        Zu, lu, cu = Z[use], lab[use], cov[use]
        nb = len(self.batches_)
        B = np.stack([(lu == b).astype(float) for b in self.batches_], 1)
        D = np.hstack([B, cu])
        coef = np.linalg.lstsq(D, Zu, rcond=None)[0]
        n_b = B.sum(0)
        self.grand_ = (n_b / n_b.sum()) @ coef[:nb]
        self.beta_ = coef[nb:]
        self.var_ = ((Zu - D @ coef) ** 2).mean(0)
        self.valid_ = self.var_ > VAR_FLOOR
        v = self.valid_
        sd = np.sqrt(np.where(v, self.var_, 1.0))
        s = (Zu - (self.grand_ + cu @ self.beta_)) / sd
        g_hat = np.stack([s[lu == b].mean(0) for b in self.batches_])
        d_hat2 = np.stack([s[lu == b].var(0, ddof=1) for b in self.batches_])
        self.gamma_ = np.zeros((nb, p))
        self.delta2_ = np.ones((nb, p))
        for j, b in enumerate(self.batches_):
            gv, dv = g_hat[j, v], d_hat2[j, v]
            g_bar, t2 = gv.mean(), max(gv.var(ddof=1), 1e-12)
            m, s2 = dv.mean(), max(dv.var(ddof=1), 1e-12)
            a, bb = (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2
            self.gamma_[j, v], self.delta2_[j, v] = _eb(s[lu == b][:, v], gv, dv, g_bar, t2, a, bb)
        return self

    def transform(self, Z, lab, cov):
        out = Z.copy()
        mean_i = self.grand_ + cov @ self.beta_
        sd = np.sqrt(np.where(self.valid_, self.var_, 1.0))
        for j, b in enumerate(self.batches_):
            rows = lab == b
            if not rows.any():
                continue
            adj = (((Z[rows] - mean_i[rows]) / sd) - self.gamma_[j]) / np.sqrt(self.delta2_[j]) * sd + mean_i[rows]
            out[rows] = np.where(self.valid_[None, :], adj, Z[rows])
        return out


class HarmFZ(BaseEstimator, TransformerMixin):
    """Training-column-mean imputation + Fisher-Z (as the baseline), then label-free ComBat-style harmonization.
    X's last 3 columns: site code, age, female. fit() does not use y."""

    def fit(self, X, y=None):
        F, site, age, fem = X[:, :-3], X[:, -3].astype(int), X[:, -2], X[:, -1]
        self.fz_ = MeanImputeFisherZ().fit(F)
        self.age_mu_, self.age_sd_ = float(age.mean()), float(max(age.std(), 1e-6))
        ub, cnt = np.unique(site, return_counts=True)
        self.known_ = [int(b) for b, c in zip(ub, cnt) if c >= MIN_BATCH]
        self.small_ = [int(b) for b, c in zip(ub, cnt) if c < MIN_BATCH]
        self.has_pool_ = int(sum(c for b, c in zip(ub, cnt) if c < MIN_BATCH)) >= MIN_BATCH
        self.cp_ = ComBatParams().fit(self.fz_.transform(F), self._labels(site, fitting=True), self._cov(age, fem))
        return self

    def _labels(self, site, fitting=False):
        lab = np.full(len(site), SKIP, int)
        lab[np.isin(site, self.known_)] = site[np.isin(site, self.known_)]
        if self.has_pool_:
            lab[np.isin(site, self.small_)] = POOL
        return lab

    def _cov(self, age, fem):
        return np.stack([(age - self.age_mu_) / self.age_sd_, fem], 1)

    def transform(self, X):
        F, site, age, fem = X[:, :-3], X[:, -3].astype(int), X[:, -2], X[:, -1]
        return self.cp_.transform(self.fz_.transform(F), self._labels(site), self._cov(age, fem))
