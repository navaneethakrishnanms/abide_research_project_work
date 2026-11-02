"""
evaluation.py - Leakage-free evaluation for ABIDE classification.

Every estimator here follows one rule: nothing derived from a test subject
(scaler statistics, feature ranking, hyperparameters, decision threshold, or
the choice of which model to keep) is ever computed on that subject.

Use `nested_cv` for the headline number and `leave_one_site_out` for the
generalisation number. Always report `permutation_test` alongside them.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
)


# ── metrics ──────────────────────────────────────────────────────────────────

def sensitivity_specificity(y_true, y_pred):
    """Returns (sensitivity, specificity) for the positive class 1."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec


def youden_threshold(y_true, y_prob):
    """Decision threshold maximising sensitivity + specificity - 1.

    No clipping: a threshold that lands outside [0.35, 0.65] is telling you the
    model is miscalibrated, and that is information worth seeing rather than
    hiding.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial confidence interval for an observed accuracy."""
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def summarise(y_true, y_prob, threshold=0.5, label=""):
    """Full metric set for one set of out-of-fold predictions."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    lo, hi = clopper_pearson(int(round(acc * n)), n)
    sens, spec = sensitivity_specificity(y_true, y_pred)
    baseline = max(y_true.mean(), 1 - y_true.mean())
    # Is the result better than always predicting the majority class?
    p_vs_base = stats.binomtest(
        int(round(acc * n)), n, baseline, alternative="greater"
    ).pvalue
    return {
        "label": label,
        "n": n,
        "accuracy": acc,
        "acc_ci": (lo, hi),
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "baseline": baseline,
        "p_vs_baseline": p_vs_base,
        "threshold": threshold,
    }


def print_summary(s):
    print(f"  {s['label']}")
    print(
        f"    accuracy      {s['accuracy']*100:6.2f}%  "
        f"95% CI [{s['acc_ci'][0]*100:.1f}, {s['acc_ci'][1]*100:.1f}]   (N={s['n']})"
    )
    print(f"    ROC-AUC       {s['auc']:6.3f}")
    print(f"    F1            {s['f1']:6.3f}")
    print(f"    sensitivity   {s['sensitivity']*100:6.2f}%   specificity {s['specificity']*100:6.2f}%")
    print(
        f"    baseline      {s['baseline']*100:6.2f}%   "
        f"p(better than baseline) = {s['p_vs_baseline']:.4f}"
    )


# ── estimators ───────────────────────────────────────────────────────────────

def nested_cv(estimator, X, y, param_grid=None, outer_splits=10, inner_splits=5,
              seeds=(0, 1, 2, 3, 4), scoring="roc_auc", tune_threshold=False,
              verbose=True, label="nested CV", n_jobs=4):
    """Repeated stratified nested cross-validation.

    The outer loop produces the performance estimate. The inner loop selects
    hyperparameters and, optionally, the decision threshold - both using only
    outer-training data. Repeating over several seeds is not optional at these
    sample sizes: a single split's number is mostly noise.

    Returns (list of per-seed summaries, dict of aggregate statistics).
    """
    X, y = np.asarray(X), np.asarray(y)
    per_seed = []

    for seed in seeds:
        oof = np.zeros(len(y), dtype=float)
        thresholds = []
        outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed)

        for tr, te in outer.split(X, y):
            if param_grid:
                inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
                search = GridSearchCV(
                    clone(estimator), param_grid, cv=inner, scoring=scoring, n_jobs=n_jobs
                )
                search.fit(X[tr], y[tr])
                model = search.best_estimator_
            else:
                model = clone(estimator).fit(X[tr], y[tr])

            if tune_threshold:
                # Threshold comes from an inner split of the training data only.
                inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
                inner_prob = np.zeros(len(tr), dtype=float)
                for itr, iva in inner.split(X[tr], y[tr]):
                    m = clone(model).fit(X[tr][itr], y[tr][itr])
                    inner_prob[iva] = m.predict_proba(X[tr][iva])[:, 1]
                thresholds.append(youden_threshold(y[tr], inner_prob))

            oof[te] = model.predict_proba(X[te])[:, 1]

        thr = float(np.mean(thresholds)) if thresholds else 0.5
        per_seed.append(summarise(y, oof, thr, label=f"{label} seed={seed}"))

    agg = {
        "accuracy_mean": np.mean([s["accuracy"] for s in per_seed]),
        "accuracy_std": np.std([s["accuracy"] for s in per_seed]),
        "auc_mean": np.mean([s["auc"] for s in per_seed]),
        "auc_std": np.std([s["auc"] for s in per_seed]),
        "sens_mean": np.mean([s["sensitivity"] for s in per_seed]),
        "spec_mean": np.mean([s["specificity"] for s in per_seed]),
        "baseline": per_seed[0]["baseline"],
        "n": per_seed[0]["n"],
    }
    if verbose:
        print(f"\n  {label}  ({len(seeds)} seeds x {outer_splits}-fold, N={agg['n']})")
        print(
            f"    accuracy   {agg['accuracy_mean']*100:6.2f}% +/- {agg['accuracy_std']*100:.2f}"
            f"    (baseline {agg['baseline']*100:.2f}%)"
        )
        print(f"    ROC-AUC    {agg['auc_mean']:6.3f} +/- {agg['auc_std']:.3f}")
        print(
            f"    sens/spec  {agg['sens_mean']*100:6.2f}% / {agg['spec_mean']*100:.2f}%"
        )
    return per_seed, agg


def leave_one_site_out(estimator, X, y, sites, param_grid=None, inner_splits=5,
                       scoring="roc_auc", verbose=True, min_site_n=20, n_jobs=4):
    """Train on every site but one, test on the held-out site.

    This is the honest generalisation test for a multi-site dataset, and it is
    the number a reviewer will care about most: it answers "does this work on a
    scanner the model has never seen?" rather than "does this work on subjects
    drawn from the same scanners as training?".
    """
    X, y, sites = np.asarray(X), np.asarray(y), np.asarray(sites)
    logo = LeaveOneGroupOut()
    rows, oof = [], np.full(len(y), np.nan)

    for tr, te in logo.split(X, y, groups=sites):
        site = sites[te][0]
        if len(te) < min_site_n or len(np.unique(y[te])) < 2:
            continue
        if param_grid:
            inner = StratifiedKFold(inner_splits, shuffle=True, random_state=0)
            search = GridSearchCV(
                clone(estimator), param_grid, cv=inner, scoring=scoring, n_jobs=n_jobs
            )
            search.fit(X[tr], y[tr])
            model = search.best_estimator_
        else:
            model = clone(estimator).fit(X[tr], y[tr])

        prob = model.predict_proba(X[te])[:, 1]
        oof[te] = prob
        s = summarise(y[te], prob, 0.5, label=site)
        rows.append(s)
        if verbose:
            print(
                f"    {site:<16} N={s['n']:>4}  acc {s['accuracy']*100:5.1f}%  "
                f"AUC {s['auc']:.3f}  (baseline {s['baseline']*100:.1f}%)"
            )

    mask = ~np.isnan(oof)
    pooled = summarise(y[mask], oof[mask], 0.5, label="pooled across held-out sites")
    if verbose:
        print()
        print(
            f"    mean over sites: acc {np.mean([r['accuracy'] for r in rows])*100:.2f}%"
            f" +/- {np.std([r['accuracy'] for r in rows])*100:.2f}"
            f"   AUC {np.mean([r['auc'] for r in rows]):.3f}"
        )
        print_summary(pooled)
    return rows, pooled


def permutation_test(estimator, X, y, n_permutations=500, cv_splits=10,
                     scoring="roc_auc", seed=0, verbose=True, n_jobs=4):
    """Compare the real score against scores from shuffled labels.

    If the true score sits inside the null distribution, the representation
    carries no usable signal and no amount of model tuning will change that.
    """
    from sklearn.model_selection import permutation_test_score

    # Disable any joblib transformer cache before permuting. joblib keys the
    # cached fit on (X, y), and a permutation test changes y every iteration -
    # so nothing is ever reused and every permutation writes a fresh copy of
    # the transformed features. With 200 permutations x 10 folds x ~120 MB that
    # filled 455 GB of disk before it was caught. Caching helps a grid search,
    # where y is fixed; here it is strictly harmful.
    estimator = clone(estimator)
    if hasattr(estimator, "memory"):
        estimator.set_params(memory=None)

    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=seed)
    score, perm_scores, pvalue = permutation_test_score(
        estimator, np.asarray(X), np.asarray(y), cv=cv,
        n_permutations=n_permutations, scoring=scoring, n_jobs=n_jobs, random_state=seed,
    )
    if verbose:
        print(
            f"    permutation ({n_permutations} shuffles): true {scoring} {score:.3f} | "
            f"null {perm_scores.mean():.3f} +/- {perm_scores.std():.3f} | p = {pvalue:.4f}"
        )
    return score, perm_scores, pvalue


def confound_report(y, sites=None, **variables):
    """Group differences on nuisance variables (motion, age, IQ, site).

    Run this before believing any result. A classifier that separates groups on
    head motion has learned head motion, not neurobiology.
    """
    y = np.asarray(y)
    print("  Confound check (group 1 vs group 0):")
    for name, v in variables.items():
        v = np.asarray(v, dtype=float)
        ok = ~np.isnan(v)
        a, b = v[ok & (y == 1)], v[ok & (y == 0)]
        if len(a) < 3 or len(b) < 3:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        flag = "  <-- CONFOUND" if p < 0.05 else ""
        print(
            f"    {name:<22} {a.mean():9.3f} vs {b.mean():9.3f}   "
            f"d={d:+.2f}  p={p:.4f}{flag}"
        )
    if sites is not None:
        sites = np.asarray(sites)
        tab = {s: y[sites == s].mean() for s in np.unique(sites)}
        spread = max(tab.values()) - min(tab.values())
        print(f"    class balance varies across sites by {spread*100:.1f} percentage points")
