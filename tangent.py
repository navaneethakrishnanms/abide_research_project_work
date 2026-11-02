"""
tangent.py - Fast tangent-space embedding of functional connectivity.

nilearn's ConnectivityMeasure(kind="tangent") is correct but loops in Python
over subjects for every matrix logarithm, which makes it far too slow to sit
inside a nested cross-validation: a single fit on a few hundred subjects runs
for minutes, and the full protocol needs roughly sixty of them.

This does the same mathematics with batched NumPy. Every matrix function goes
through one `np.linalg.eigh` call on a stacked (n_subjects, n_regions,
n_regions) array instead of one call per subject, which is where essentially
all of the speedup comes from.

The maths, for reference:

    covariances     C_i           - Ledoit-Wolf shrunk, guaranteed SPD
    reference       G             - Riemannian geometric mean of the training C_i
    tangent vector  log(G^-1/2 C_i G^-1/2)

The geometric mean is the fold-dependent part, so it is estimated on training
subjects only and then applied to everyone - which is exactly the leak that
makes naive tangent-space pipelines report inflated scores.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf


# ── batched SPD matrix functions ─────────────────────────────────────────────

def _eig_apply(mats, fn, clip_positive):
    """Apply a scalar function to the eigenvalues of stacked symmetric matrices.

    `clip_positive` floors the eigenvalues before `fn` is applied. That is
    required for log and the square roots, whose input must be positive
    definite and which would otherwise blow up on the tiny negative
    eigenvalues that finite precision produces. It must NOT be used for the
    exponential: that one is applied to a tangent vector, which is symmetric
    but indefinite, and flooring its negative eigenvalues at zero turns the
    result into approximately the identity. Doing so silently froze the
    geometric mean on its first iteration - the residual norm sat at 1.47e+01
    for 25 iterations without moving.
    """
    w, V = np.linalg.eigh(mats)
    if clip_positive:
        w = np.clip(w, 1e-10, None)
    w = fn(w)
    return V @ (w[..., None] * np.swapaxes(V, -1, -2))


def spd_log(mats):
    return _eig_apply(mats, np.log, clip_positive=True)


def sym_exp(mats):
    """Matrix exponential of a symmetric (possibly indefinite) matrix."""
    return _eig_apply(mats, np.exp, clip_positive=False)


def spd_sqrt(mats):
    return _eig_apply(mats, np.sqrt, clip_positive=True)


def spd_inv_sqrt(mats):
    return _eig_apply(mats, lambda w: 1.0 / np.sqrt(w), clip_positive=True)


def geometric_mean(covs, max_iter=30, tol=1e-5, return_history=False):
    """Riemannian (Frechet) mean of a stack of SPD matrices.

    Gradient descent on the manifold: repeatedly whiten by the current
    estimate, average in the tangent space, and map the correction back. The
    residual norm must fall monotonically toward zero; a flat trace means the
    update step is not being applied and the returned matrix is not a mean.
    """
    G = covs.mean(axis=0)
    history = []
    for _ in range(max_iter):
        G_inv_sqrt = spd_inv_sqrt(G[None])[0]
        G_sqrt = spd_sqrt(G[None])[0]

        whitened = G_inv_sqrt @ covs @ G_inv_sqrt
        tangent_mean = spd_log(whitened).mean(axis=0)

        norm = float(np.linalg.norm(tangent_mean))
        history.append(norm)
        if norm < tol:
            break

        G = G_sqrt @ sym_exp(tangent_mean[None])[0] @ G_sqrt
        G = (G + G.T) / 2.0                      # keep it exactly symmetric

    if len(history) >= 3 and history[-1] > 0.5 * history[0]:
        warnings.warn(
            f"geometric_mean did not converge: residual went "
            f"{history[0]:.3e} -> {history[-1]:.3e} in {len(history)} iterations. "
            "The returned matrix is not a valid Riemannian mean.",
            RuntimeWarning, stacklevel=2,
        )
    return (G, history) if return_history else G


# ── covariance estimation ────────────────────────────────────────────────────

def covariances(series, shrinkage="ledoit-wolf"):
    """Per-subject covariance from a list of (timepoints, regions) arrays.

    Ledoit-Wolf shrinkage is not cosmetic here: with 200 regions and as few as
    116 timepoints the sample covariance is singular, and the matrix logarithm
    of a singular matrix is undefined.
    """
    out = []
    for ts in series:
        x = np.asarray(ts, dtype=np.float64)
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        if shrinkage == "ledoit-wolf":
            c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
        else:
            c = np.cov(x, rowvar=False)
            c.flat[:: c.shape[0] + 1] += 1e-3
        out.append((c + c.T) / 2.0)
    return np.stack(out)


def vectorize(mats):
    """Upper triangle (excluding diagonal), with off-diagonals scaled by sqrt(2).

    The scaling preserves the Frobenius norm, so Euclidean distance between
    vectors equals the matrix distance the embedding was built to linearise.
    """
    n = mats.shape[-1]
    iu = np.triu_indices(n, k=1)
    return mats[..., iu[0], iu[1]] * np.sqrt(2.0)


# ── scikit-learn transformer ─────────────────────────────────────────────────

class TangentEmbedding(BaseEstimator, TransformerMixin):
    """Tangent-space (or plain correlation) connectivity features.

    Fit estimates the reference point from training subjects only. X is a
    sequence of (timepoints, regions) arrays, or precomputed covariances when
    `precomputed=True`.
    """

    def __init__(self, kind="tangent", precomputed=False):
        self.kind = kind
        self.precomputed = precomputed

    def _covs(self, X):
        if self.precomputed:
            return np.asarray(X, dtype=np.float64)
        return covariances(list(X))

    def fit(self, X, y=None):
        covs = self._covs(X)
        if self.kind == "tangent":
            self.reference_ = geometric_mean(covs)
            self.whitener_ = spd_inv_sqrt(self.reference_[None])[0]
        return self

    def transform(self, X):
        covs = self._covs(X)
        if self.kind == "tangent":
            whitened = self.whitener_ @ covs @ self.whitener_
            return vectorize(spd_log(whitened))
        if self.kind == "correlation":
            d = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
            corr = covs / (d[:, :, None] * d[:, None, :] + 1e-12)
            return vectorize(np.arctanh(np.clip(corr, -0.999999, 0.999999)))
        if self.kind == "partial":
            # Invert through the eigendecomposition rather than np.linalg.pinv.
            # The covariances are Ledoit-Wolf shrunk and therefore SPD, so the
            # eigenvalues are all the inverse needs - and pinv's SVD raises
            # "SVD did not converge" on some of these 200x200 matrices, which
            # killed a three-hour run at the last pipeline.
            prec = _eig_apply(covs, lambda w: 1.0 / w, clip_positive=True)
            d = np.sqrt(np.abs(np.diagonal(prec, axis1=-2, axis2=-1)))
            pc = -prec / (d[:, :, None] * d[:, None, :] + 1e-12)
            return vectorize(np.arctanh(np.clip(pc, -0.999999, 0.999999)))
        raise ValueError(f"unknown kind: {self.kind}")
