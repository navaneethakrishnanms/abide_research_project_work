"""Label-free, ComBat-style empirical-Bayes site harmonizer for the CC200 X1 experiment (written fresh; generic algorithm, Johnson/Li/Rabinovic 2007, parametric prior).
Input X columns: [ connectivity features | age | female | site code ]. Output columns: [ harmonized features | age | female ] (site code removed).
Deployment validity (checked by tests in run_x1.py):
  * fit() uses only the rows it receives and never uses diagnosis labels (y is ignored);
  * transform() maps every row from that row's own features, age, sex and site plus the fitted parameters; other test rows and test-set statistics are never used;
  * a site that was not fitted (unseen, or too small to estimate) is passed through unchanged (identity), so an unseen site is never adjusted with its own statistics.
Batch labels: >= 0 a fitted site, -1 pooled "other" (training sites with < MIN_BATCH subjects, if together >= MIN_BATCH), -2 = unchanged."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

MIN_BATCH = 5
VAR_FLOOR = 1e-10
POOL, SKIP = -1, -2


def _eb(s_b, g_hat, d_hat2, g_bar, t2, a, b, tol=1e-6, max_iter=1000):
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


class CC200Harmonizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):                       # y is deliberately ignored: no diagnosis labels
        F, age, fem, site = X[:, :-3], X[:, -3], X[:, -2], X[:, -1].astype(int)
        self.age_mu_, self.age_sd_ = float(age.mean()), float(max(age.std(), 1e-6))
        ub, cnt = np.unique(site, return_counts=True)
        self.known_ = [int(b) for b, c in zip(ub, cnt) if c >= MIN_BATCH]
        self.small_ = [int(b) for b, c in zip(ub, cnt) if c < MIN_BATCH]
        self.has_pool_ = int(sum(c for b, c in zip(ub, cnt) if c < MIN_BATCH)) >= MIN_BATCH
        lab = self._labels(site)
        use = lab != SKIP
        self.batches_ = sorted(set(int(b) for b in lab[use]))
        Z, lu, cu = F[use], lab[use], self._cov(age, fem)[use]
        nb = len(self.batches_)
        B = np.stack([(lu == b).astype(float) for b in self.batches_], 1)
        D = np.hstack([B, cu])
        coef = np.linalg.lstsq(D, Z, rcond=None)[0]
        n_b = B.sum(0)
        self.grand_ = (n_b / n_b.sum()) @ coef[:nb]
        self.beta_ = coef[nb:]
        self.var_ = ((Z - D @ coef) ** 2).mean(0)
        self.valid_ = self.var_ > VAR_FLOOR
        v = self.valid_
        sd = np.sqrt(np.where(v, self.var_, 1.0))
        s = (Z - (self.grand_ + cu @ self.beta_)) / sd
        g_hat = np.stack([s[lu == b].mean(0) for b in self.batches_])
        d_hat2 = np.stack([s[lu == b].var(0, ddof=1) for b in self.batches_])
        p = F.shape[1]
        self.gamma_, self.delta2_ = np.zeros((nb, p)), np.ones((nb, p))
        for j, b in enumerate(self.batches_):
            gv, dv = g_hat[j, v], d_hat2[j, v]
            g_bar, t2 = gv.mean(), max(gv.var(ddof=1), 1e-12)
            m, s2 = dv.mean(), max(dv.var(ddof=1), 1e-12)
            a, bb = (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2
            self.gamma_[j, v], self.delta2_[j, v] = _eb(s[lu == b][:, v], gv, dv, g_bar, t2, a, bb)
        return self

    def _labels(self, site):
        lab = np.full(len(site), SKIP, int)
        k = np.isin(site, self.known_)
        lab[k] = site[k]
        if self.has_pool_:
            lab[np.isin(site, self.small_)] = POOL
        return lab

    def _cov(self, age, fem):
        return np.stack([(age - self.age_mu_) / self.age_sd_, fem], 1)

    def transform(self, X):
        F, age, fem, site = X[:, :-3], X[:, -3], X[:, -2], X[:, -1].astype(int)
        out = F.copy()
        lab = self._labels(site)
        mean_i = self.grand_ + self._cov(age, fem) @ self.beta_
        sd = np.sqrt(np.where(self.valid_, self.var_, 1.0))
        for j, b in enumerate(self.batches_):
            rows = lab == b
            if rows.any():
                adj = (((F[rows] - mean_i[rows]) / sd) - self.gamma_[j]) / np.sqrt(self.delta2_[j]) * sd + mean_i[rows]
                out[rows] = np.where(self.valid_[None, :], adj, F[rows])
        return np.hstack([out, age[:, None], fem[:, None]])
