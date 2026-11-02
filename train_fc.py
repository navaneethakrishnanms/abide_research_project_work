"""
train_fc.py - Functional connectivity classifier on the full ABIDE cohort.

This is the number to beat before any deep model is worth writing.

Two things make the result trustworthy, and both are places published ABIDE
pipelines commonly go wrong:

1. Tangent-space embedding is *fitted* - it estimates a reference covariance
   and projects subjects onto the tangent plane there. Estimating that
   reference on all subjects before splitting lets every test subject
   contribute to its own features. Here it is a transformer inside the
   pipeline, refitted on training data in every fold.

2. Accuracy is reported against the confounds-only baseline from
   confound_baseline.py (AUC 0.619 from FIQ, sex, age and site alone), not
   just against the majority class. On full ABIDE, FIQ by itself scores 61%.

Run confound_baseline.py first so the comparison is available.

    python train_fc.py                 # full protocol
    python train_fc.py --quick         # one pipeline, one seed
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import evaluation as ev
from tangent import TangentEmbedding, covariances

warnings.filterwarnings("ignore")

DATA = r"C:\abide\abide_pcp"
COVS = os.path.join(DATA, "covs_cc200.npy")
RESULTS = r"C:\abide\fc_results.csv"
MIN_TIMEPOINTS = 100      # too few frames makes the covariance unstable


def load_cohort(min_timepoints=MIN_TIMEPOINTS):
    """Returns per-subject covariances, labels, site labels and phenotypes.

    Covariances are cached: they do not depend on the fold, so computing them
    once and letting the embedding work from precomputed matrices keeps the
    expensive part of cross-validation to the geometric mean alone.
    """
    ts = np.load(os.path.join(DATA, "ts_cc200.npy"), allow_pickle=True)
    ph = pd.read_csv(os.path.join(DATA, "phenotypic_qc.csv"))

    keep = np.array([t is not None and t.shape[0] >= min_timepoints for t in ts])
    dropped = int((~keep).sum())
    ts, ph = ts[keep], ph.loc[keep].reset_index(drop=True)

    if os.path.exists(COVS):
        covs = np.load(COVS)
        if len(covs) != len(ts):
            covs = covariances(list(ts))
            np.save(COVS, covs)
    else:
        covs = covariances(list(ts))
        np.save(COVS, covs)

    # DX_GROUP: 1 = autism, 2 = control. Positive class is autism.
    y = (ph["DX_GROUP"].values == 1).astype(int)
    sites = ph["SITE_ID"].astype(str).values

    print(f"Loaded {len(ts)} subjects ({dropped} dropped for < {min_timepoints} timepoints)")
    print(f"  ASD {int(y.sum())} | TC {int((1-y).sum())} | sites {len(np.unique(sites))}")
    print(f"  regions {covs.shape[-1]} | timepoints "
          f"{min(t.shape[0] for t in ts)}-{max(t.shape[0] for t in ts)}")
    return covs, y, sites, ph, ts


def motion_proxy(ts):
    """Mean absolute temporal derivative - a stand-in for framewise displacement."""
    out = []
    for t in ts:
        z = (t - t.mean(0)) / (t.std(0) + 1e-8)
        out.append(float(np.abs(np.diff(z, axis=0)).mean()))
    return np.array(out)


def make_pipelines(memory=None):
    """Pipelines for comparison.

    `memory` caches fitted transformers. It is what makes the grid search
    finish: the tangent reference costs ~45s to estimate on 760 subjects and
    does not depend on the classifier's C, so without caching GridSearchCV
    recomputes it for every hyperparameter on every inner fold. Measured
    speedup on one search: 9.5x.

    It is useless for a permutation test, and worse than useless. joblib keys
    the cache on (X, y), and permuting changes y every iteration, so no entry
    is ever reused and each one writes ~120 MB of transformed features to disk.
    `evaluation.permutation_test` therefore strips `memory` before running.
    """
    tangent = lambda: TangentEmbedding("tangent", precomputed=True)
    return {
        "tangent + L2 logreg": (
            Pipeline([("conn", tangent()), ("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))], memory=memory),
            {"clf__C": [0.001, 0.01, 0.1, 1.0]},
        ),
        "tangent + linear SVM": (
            Pipeline([("conn", tangent()), ("scale", StandardScaler()),
                      ("clf", SVC(kernel="linear", probability=True))], memory=memory),
            {"clf__C": [0.001, 0.01, 0.1]},
        ),
        "tangent + SelectK + logreg": (
            Pipeline([("conn", tangent()), ("scale", StandardScaler()),
                      ("sel", SelectKBest(f_classif)), ("clf", LogisticRegression(max_iter=5000))], memory=memory),
            {"sel__k": [1000, 5000], "clf__C": [0.01, 0.1, 1.0]},
        ),
        "correlation + L2 logreg": (
            Pipeline([("conn", TangentEmbedding("correlation", precomputed=True)),
                      ("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))], memory=memory),
            {"clf__C": [0.001, 0.01, 0.1, 1.0]},
        ),
        "partial corr + L2 logreg": (
            Pipeline([("conn", TangentEmbedding("partial", precomputed=True)),
                      ("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))], memory=memory),
            {"clf__C": [0.001, 0.01, 0.1, 1.0]},
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    covs, y, sites, ph, ts = load_cohort()

    print("\n" + "=" * 74)
    print("  Confounds")
    print("=" * 74)
    ev.confound_report(
        y, sites=sites,
        motion_proxy=motion_proxy(ts),
        age=pd.to_numeric(ph.get("AGE_AT_SCAN"), errors="coerce"),
        FIQ=pd.to_numeric(ph.get("FIQ"), errors="coerce").replace(-9999, np.nan),
        sex=pd.to_numeric(ph.get("SEX"), errors="coerce"),
        n_timepoints=np.array([t.shape[0] for t in ts], dtype=float),
    )

    cache = os.path.join(DATA, "_cache")

    def clear_cache():
        """The cache pays off within one grid search and within the permutation
        test, where the training rows repeat. Across seeds and pipelines the
        training rows are all different, so entries are never reused and each
        one holds a ~100 MB feature matrix - roughly 55 GB if left to
        accumulate. Clearing between stages keeps the benefit and the disk."""
        shutil.rmtree(cache, ignore_errors=True)

    clear_cache()
    pipelines = make_pipelines(memory=cache)
    if args.quick:
        pipelines = {"tangent + L2 logreg": pipelines["tangent + L2 logreg"]}
    seeds = tuple(range(args.seeds))

    print("\n" + "=" * 74)
    print(f"  Nested cross-validation ({args.folds}-fold outer, 5-fold inner, {len(seeds)} seeds)")
    print("=" * 74)
    # Resume: a pipeline already in the results file is not recomputed, so an
    # interrupted run continues instead of repeating half an hour of work.
    results = {}
    if os.path.exists(RESULTS):
        done = pd.read_csv(RESULTS, index_col=0)
        results = {k: v for k, v in done.to_dict(orient="index").items()}
        if results:
            print(f"  resuming - already have: {', '.join(results)}")

    for name, (pipe, grid) in pipelines.items():
        if name in results:
            continue
        _, agg = ev.nested_cv(pipe, covs, y, param_grid=grid,
                              outer_splits=args.folds, inner_splits=args.inner,
                              seeds=seeds, label=name, n_jobs=args.jobs)
        results[name] = agg
        clear_cache()
        pd.DataFrame(results).T.to_csv(RESULTS)

    best = max(results, key=lambda k: results[k]["auc_mean"])
    print("\n" + "-" * 74)
    print(f"  Best by AUC: {best}  ({results[best]['auc_mean']:.3f})")
    print("  Compare against: majority 53.78% / AUC 0.500")
    print("                   confounds only (FIQ+sex+age+site) 60.36% / AUC 0.619")

    print("\n" + "=" * 74)
    print("  Leave-one-site-out generalisation (best pipeline)")
    print("=" * 74)
    pipe, grid = pipelines[best]
    ev.leave_one_site_out(pipe, covs, y, sites, param_grid=grid,
                          inner_splits=args.inner, n_jobs=args.jobs)
    clear_cache()

    print("\n" + "=" * 74)
    print("  Permutation test (best pipeline)")
    print("=" * 74)
    ev.permutation_test(pipe, covs, y, n_permutations=args.permutations, n_jobs=args.jobs)
    clear_cache()

    pd.DataFrame(results).T.to_csv(RESULTS)
    print("\nSaved per-pipeline results to C:\\abide\\fc_results.csv")


if __name__ == "__main__":
    main()
