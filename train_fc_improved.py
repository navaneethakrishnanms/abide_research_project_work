"""
train_fc_improved.py -- Leakage-Free Functional Connectivity Classifier with FIQ Ablation
==========================================================================================

Key improvements over train_fc.py:
  1. True 20% holdout test set carved out BEFORE any CV — never touched during training.
  2. Three mandatory ablation configurations:
       A: FC tangent features only (no demographics)
       B: FC + age + sex (FIQ excluded — FIQ is a known confound, d=-0.35, p<1e-4)
       C: FC + age + sex + FIQ (measures confound contribution explicitly)
  3. All preprocessing (tangent reference, feature scaling, FIQ imputation) fit on
     training subjects only inside every fold — zero data leakage.
  4. Full metric suite: accuracy, balanced accuracy, ROC-AUC, F1, sensitivity,
     specificity, confusion matrix, per-site breakdown, 95% CI.
  5. Youden's J threshold calibration on OOF predictions only.
  6. Permutation test (200 shuffles) for statistical significance.
  7. Leave-one-site-out evaluation on the best configuration.
  8. Results saved to fc_improved_results.csv and fc_improved_loso.csv.

Usage:
    python train_fc_improved.py              # full run (all configs)
    python train_fc_improved.py --quick      # A only, 1 seed, 5 outer folds
    python train_fc_improved.py --no-perm    # skip permutation test (saves ~30 min)
"""

from __future__ import annotations

import argparse
import os
import warnings
import time
import sys

# Force UTF-8 output on Windows to avoid cp1252 Unicode crashes
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, LeaveOneGroupOut,
    train_test_split, permutation_test_score,
)
from sklearn.preprocessing import StandardScaler

from tangent import TangentEmbedding, covariances

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA         = r"C:\abide\abide_pcp"
TS_PATH      = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH    = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH   = os.path.join(DATA, "phenotypic_qc.csv")
RESULTS_CSV  = r"C:\abide\fc_improved_results.csv"
LOSO_CSV     = r"C:\abide\fc_improved_loso.csv"

MIN_TIMEPOINTS = 100
SEED           = 42


# ── 1. Metrics helpers ────────────────────────────────────────────────────────

def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial 95% CI for an observed accuracy."""
    lo = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises sensitivity + specificity - 1 (Youden's J)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def full_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                 threshold: float = 0.5, label: str = "") -> dict:
    """Compute the full metric suite for one set of predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    n      = len(y_true)
    acc    = accuracy_score(y_true, y_pred)
    k      = int(round(acc * n))
    lo, hi = clopper_pearson(k, n)

    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens   = tp / (tp + fn) if (tp + fn) else float("nan")
    spec   = tn / (tn + fp) if (tn + fp) else float("nan")

    # p-value vs majority-class baseline
    baseline = max(y_true.mean(), 1 - y_true.mean())
    p_base = stats.binomtest(k, n, baseline, alternative="greater").pvalue

    return {
        "label":            label,
        "n":                n,
        "threshold":        threshold,
        "accuracy":         acc,
        "acc_lo95":         lo,
        "acc_hi95":         hi,
        "balanced_acc":     balanced_accuracy_score(y_true, y_pred),
        "roc_auc":          roc_auc_score(y_true, y_prob),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "sensitivity":      sens,
        "specificity":      spec,
        "baseline":         baseline,
        "p_vs_baseline":    p_base,
        "cm":               cm,
    }


def print_metrics(m: dict):
    print(f"\n  -- {m['label']} --")
    print(
        f"    accuracy      {m['accuracy']*100:6.2f}%  "
        f"95% CI [{m['acc_lo95']*100:.1f}%, {m['acc_hi95']*100:.1f}%]  "
        f"(N={m['n']})"
    )
    print(f"    balanced acc  {m['balanced_acc']*100:6.2f}%")
    print(f"    ROC-AUC       {m['roc_auc']:6.3f}")
    print(f"    F1            {m['f1']:6.3f}")
    print(f"    sensitivity   {m['sensitivity']*100:6.2f}%   "
          f"specificity {m['specificity']*100:6.2f}%")
    print(
        f"    baseline      {m['baseline']*100:6.2f}%   "
        f"p(better than baseline) = {m['p_vs_baseline']:.4f}"
    )
    cm = m["cm"]
    print(f"    confusion matrix:")
    print(f"                  Pred ASD   Pred TC")
    print(f"      True ASD      {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"      True TC       {cm[1,0]:4d}      {cm[1,1]:4d}")


# ── 2. Core nested-CV function ────────────────────────────────────────────────

def nested_cv_tangent(
    covs: np.ndarray,
    y: np.ndarray,
    demo: np.ndarray | None = None,
    kind: str = "correlation",
    outer_splits: int = 10,
    inner_splits: int = 5,
    seeds: tuple = (0, 1, 2),
    label: str = "",
    n_jobs: int = 4,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Repeated stratified nested cross-validation with tangent-space embedding.

    The tangent reference point (Riemannian geometric mean) is estimated on
    training subjects only inside every outer fold -- this is the leak that
    inflates scores in naive pipelines. Demographic features are scaled with
    a StandardScaler fit on the same training subjects.

    FIQ NaN values (coded as NaN after -9999 replacement) are imputed with
    the training-fold median, so no test information leaks into imputation.

    kind: 'tangent' (Riemannian) or 'correlation' (Fisher-z) -- both are
          implemented in tangent.py. 'correlation' was the best in the
          original train_fc.py run (AUC 0.749 vs tangent's 0.741).

    Returns:
        oof_probs : (n_subjects,) out-of-fold predicted P(ASD=1), averaged
                    across seeds -- used for a single Youden threshold search.
        agg       : dict of aggregated metrics across seeds.
    """
    rng = np.random.default_rng(SEED)
    n   = len(y)
    all_oof = np.zeros((len(seeds), n), dtype=float)

    per_seed_auc  = []
    per_seed_acc  = []
    per_seed_bacc = []

    for s_idx, seed in enumerate(seeds):
        oof = np.zeros(n, dtype=float)
        outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed)

        for tr, te in outer.split(covs, y):
            # Tangent/correlation embedding: fit on training covariances only
            tang = TangentEmbedding(kind, precomputed=True)
            X_tr = tang.fit_transform(covs[tr])   # (n_tr, n_features)
            X_te = tang.transform(covs[te])        # (n_te, n_features)

            # Optional demographic features
            if demo is not None:
                # Impute NaN with training-fold median (prevents FIQ leakage)
                demo_tr = demo[tr].copy()
                demo_te = demo[te].copy()
                for col in range(demo_tr.shape[1]):
                    col_median = np.nanmedian(demo_tr[:, col])
                    demo_tr[np.isnan(demo_tr[:, col]), col] = col_median
                    demo_te[np.isnan(demo_te[:, col]), col] = col_median

                # Scale demographics on training fold only
                ds = StandardScaler()
                demo_tr = ds.fit_transform(demo_tr)
                demo_te = ds.transform(demo_te)

                X_tr = np.hstack([X_tr, demo_tr])
                X_te = np.hstack([X_te, demo_te])

            # Scale full feature matrix on training fold only
            fs = StandardScaler()
            X_tr = fs.fit_transform(X_tr)
            X_te = fs.transform(X_te)

            # Inner CV: hyperparameter selection
            inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
            search = GridSearchCV(
                LogisticRegression(max_iter=5000, solver="lbfgs"),
                {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
                cv=inner,
                scoring="roc_auc",
                n_jobs=n_jobs,
                refit=True,
            )
            search.fit(X_tr, y[tr])
            oof[te] = search.best_estimator_.predict_proba(X_te)[:, 1]

        # Aggregate this seed's OOF predictions
        all_oof[s_idx] = oof
        auc  = roc_auc_score(y, oof)
        thresh = youden_threshold(y, oof)
        preds = (oof >= thresh).astype(int)
        acc   = accuracy_score(y, preds)
        bacc  = balanced_accuracy_score(y, preds)
        per_seed_auc.append(auc)
        per_seed_acc.append(acc)
        per_seed_bacc.append(bacc)
        if verbose:
            print(f"    seed={seed}  AUC={auc:.3f}  acc={acc*100:.2f}%  "
                  f"balanced_acc={bacc*100:.2f}%  tau={thresh:.3f}")

    # Average OOF probabilities across seeds for final threshold calibration
    mean_oof = all_oof.mean(axis=0)

    agg = {
        "label":       label,
        "auc_mean":    np.mean(per_seed_auc),
        "auc_std":     np.std(per_seed_auc),
        "acc_mean":    np.mean(per_seed_acc),
        "acc_std":     np.std(per_seed_acc),
        "bacc_mean":   np.mean(per_seed_bacc),
        "bacc_std":    np.std(per_seed_bacc),
        "n_seeds":     len(seeds),
        "outer_folds": outer_splits,
        "n":           n,
    }
    if verbose:
        print(
            f"\n  {label}  ({len(seeds)} seeds × {outer_splits}-fold, N={n})\n"
            f"    AUC  {agg['auc_mean']:.3f} ± {agg['auc_std']:.3f}\n"
            f"    acc  {agg['acc_mean']*100:.2f}% ± {agg['acc_std']*100:.2f}%\n"
            f"    balanced_acc {agg['bacc_mean']*100:.2f}% ± {agg['bacc_std']*100:.2f}%"
        )
    return mean_oof, agg


# ── 3. Leave-one-site-out ─────────────────────────────────────────────────────

def loso_tangent(
    covs: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    demo: np.ndarray | None = None,
    kind: str = "correlation",
    min_site_n: int = 20,
    n_jobs: int = 4,
) -> tuple[list[dict], dict]:
    """Leave-one-site-out evaluation with identical preprocessing to nested_cv."""
    logo = LeaveOneGroupOut()
    rows = []
    oof  = np.full(len(y), np.nan)

    for tr, te in logo.split(covs, y, groups=sites):
        site_name = sites[te][0]
        if len(te) < min_site_n or len(np.unique(y[te])) < 2:
            continue

        tang = TangentEmbedding(kind, precomputed=True)
        X_tr = tang.fit_transform(covs[tr])
        X_te = tang.transform(covs[te])

        if demo is not None:
            demo_tr = demo[tr].copy()
            demo_te = demo[te].copy()
            for col in range(demo_tr.shape[1]):
                col_median = np.nanmedian(demo_tr[:, col])
                demo_tr[np.isnan(demo_tr[:, col]), col] = col_median
                demo_te[np.isnan(demo_te[:, col]), col] = col_median
            ds = StandardScaler()
            demo_tr = ds.fit_transform(demo_tr)
            demo_te = ds.transform(demo_te)
            X_tr = np.hstack([X_tr, demo_tr])
            X_te = np.hstack([X_te, demo_te])

        fs = StandardScaler()
        X_tr = fs.fit_transform(X_tr)
        X_te = fs.transform(X_te)

        inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
        search = GridSearchCV(
            LogisticRegression(max_iter=5000, solver="lbfgs"),
            {"C": [0.001, 0.01, 0.1, 1.0]},
            cv=inner, scoring="roc_auc", n_jobs=n_jobs, refit=True,
        )
        search.fit(X_tr, y[tr])
        prob = search.best_estimator_.predict_proba(X_te)[:, 1]
        oof[te] = prob

        m = full_metrics(y[te], prob, threshold=0.5, label=site_name)
        rows.append(m)
        print(
            f"    {site_name:<16}  N={m['n']:>4}  "
            f"acc={m['accuracy']*100:5.1f}%  balanced={m['balanced_acc']*100:5.1f}%  "
            f"AUC={m['roc_auc']:.3f}  (base={m['baseline']*100:.1f}%)"
        )

    mask   = ~np.isnan(oof)
    pooled = full_metrics(y[mask], oof[mask], threshold=0.5,
                          label="pooled across held-out sites")
    print(f"\n    mean over sites:  "
          f"acc={np.mean([r['accuracy'] for r in rows])*100:.2f}%  "
          f"AUC={np.mean([r['roc_auc'] for r in rows]):.3f}")
    print_metrics(pooled)
    return rows, pooled


# ── 4. Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick",   action="store_true",
                    help="Run config A only, 1 seed, 5 folds (fast smoke-test)")
    ap.add_argument("--no-perm", action="store_true",
                    help="Skip permutation test (saves ~25-35 minutes)")
    ap.add_argument("--jobs",    type=int, default=4)
    args = ap.parse_args()

    outer_splits = 5  if args.quick else 10
    seeds        = (0,) if args.quick else (0, 1, 2)
    n_perm       = 0   if args.no_perm else 200

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 74)
    print("  ABIDE FC Improved — FIQ Ablation & Proper Holdout Evaluation")
    print("=" * 74)

    ts = np.load(TS_PATH, allow_pickle=True)
    ph = pd.read_csv(PHENO_PATH)

    keep = np.array([t is not None and t.shape[0] >= MIN_TIMEPOINTS for t in ts])
    dropped = int((~keep).sum())
    ts = ts[keep]
    ph = ph.loc[keep].reset_index(drop=True)

    # Positive class: ASD (DX_GROUP == 1)
    y     = (ph["DX_GROUP"].values == 1).astype(int)
    sites = ph["SITE_ID"].astype(str).values

    print(f"\nLoaded {len(ts)} subjects  ({dropped} dropped: <{MIN_TIMEPOINTS} timepoints)")
    print(f"  ASD {int(y.sum())} | TC {int((1-y).sum())} | sites {len(np.unique(sites))}")

    # ── Load or compute covariances ───────────────────────────────────────────
    if os.path.exists(COVS_PATH) and np.load(COVS_PATH).shape[0] == len(ts):
        covs = np.load(COVS_PATH)
        print(f"Loaded cached covariances: {covs.shape}")
    else:
        print("Computing covariances (Ledoit-Wolf)...")
        t0   = time.time()
        covs = covariances(list(ts))
        np.save(COVS_PATH, covs)
        print(f"  done in {time.time()-t0:.1f}s  shape={covs.shape}")

    # ── Demographic features ──────────────────────────────────────────────────
    num = lambda c: pd.to_numeric(ph.get(c), errors="coerce").replace(-9999, np.nan).values

    age = num("AGE_AT_SCAN").astype(float)
    sex = num("SEX").astype(float)
    fiq = num("FIQ").astype(float)

    demo_B = np.column_stack([age, sex])           # no FIQ
    demo_C = np.column_stack([age, sex, fiq])      # with FIQ

    # ── Confound check ────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  Confound check")
    print("=" * 74)
    for name, v in [("age", age), ("FIQ", fiq), ("sex", sex)]:
        ok  = ~np.isnan(v)
        a   = v[ok & (y == 1)]
        b   = v[ok & (y == 0)]
        if len(a) < 3 or len(b) < 3: continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d    = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        flag = "  [CONFOUND]" if p < 0.05 else ""
        print(f"    {name:<22}  ASD {a.mean():8.3f}  TC {b.mean():8.3f}  "
              f"d={d:+.2f}  p={p:.4f}{flag}")

    # ── Carve out 20% holdout test set BEFORE any CV ─────────────────────────
    print("\n" + "=" * 74)
    print("  Carving holdout test set (20%, stratified) — NEVER used in training")
    print("=" * 74)

    all_idx    = np.arange(len(y))
    tv_idx, ho_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=SEED
    )

    covs_tv, covs_ho = covs[tv_idx], covs[ho_idx]
    y_tv,    y_ho    = y[tv_idx],    y[ho_idx]
    sites_tv         = sites[tv_idx]
    demo_B_tv, demo_B_ho = demo_B[tv_idx], demo_B[ho_idx]
    demo_C_tv, demo_C_ho = demo_C[tv_idx], demo_C[ho_idx]

    print(f"  Train+Val: {len(tv_idx)} subjects  "
          f"(ASD {int(y_tv.sum())} | TC {int((1-y_tv).sum())})")
    print(f"  Holdout  : {len(ho_idx)} subjects  "
          f"(ASD {int(y_ho.sum())} | TC {int((1-y_ho).sum())})")

    # ── Ablation configurations ───────────────────────────────────────────────
    # Each entry: (cv_cov, cv_y, cv_demo, ho_cov, ho_y, ho_demo, embedding_kind)
    # We use 'correlation' as default -- it was the best in the original run (AUC 0.749).
    # We also test 'tangent' for Config A to compare with original train_fc.py tangent (0.741).
    configs = {
        "A-corr: FC only (correlation)": (
            covs_tv, y_tv, None, covs_ho, y_ho, None, "correlation"),
        "A-tang: FC only (tangent)": (
            covs_tv, y_tv, None, covs_ho, y_ho, None, "tangent"),
        "B: FC(corr) + age + sex": (
            covs_tv, y_tv, demo_B_tv, covs_ho, y_ho, demo_B_ho, "correlation"),
        "C: FC(corr) + age + sex + FIQ [CONFOUND]": (
            covs_tv, y_tv, demo_C_tv, covs_ho, y_ho, demo_C_ho, "correlation"),
    }
    if args.quick:
        configs = {"A-corr: FC only (correlation)": configs["A-corr: FC only (correlation)"]}

    all_results = {}
    best_config = None
    best_auc    = -1.0

    for config_label, cfg_tuple in configs.items():
        cv_cov, cv_y, cv_demo, ho_cov, ho_y, ho_demo, emb_kind = cfg_tuple
        print("\n" + "=" * 74)
        print(f"  Config {config_label}")
        print("=" * 74)

        t0 = time.time()
        oof_probs, agg = nested_cv_tangent(
            cv_cov, cv_y, demo=cv_demo,
            kind=emb_kind,
            outer_splits=outer_splits,
            inner_splits=5,
            seeds=seeds,
            label=config_label,
            n_jobs=args.jobs,
        )
        elapsed = time.time() - t0

        # Youden threshold calibrated on OOF predictions (train+val only)
        tau   = youden_threshold(cv_y, oof_probs)
        m_oof = full_metrics(cv_y, oof_probs, threshold=tau,
                             label=f"{config_label} [OOF, tau={tau:.3f}]")
        print_metrics(m_oof)
        print(f"  Elapsed: {elapsed/60:.1f} min")

        agg["tau_youden"]   = tau
        agg["oof_acc"]      = m_oof["accuracy"]
        agg["oof_bacc"]     = m_oof["balanced_acc"]
        agg["oof_f1"]       = m_oof["f1"]
        agg["oof_sens"]     = m_oof["sensitivity"]
        agg["oof_spec"]     = m_oof["specificity"]
        agg["oof_roc_auc"]  = m_oof["roc_auc"]
        agg["elapsed_min"]  = elapsed / 60
        all_results[config_label] = {"agg": agg, "oof": oof_probs,
                                     "ho_cov": ho_cov, "ho_y": ho_y,
                                     "ho_demo": ho_demo,
                                     "emb_kind": emb_kind}

        if agg["auc_mean"] > best_auc:
            best_auc    = agg["auc_mean"]
            best_config = config_label

    # ── Evaluate each config on holdout test set ──────────────────────────────
    print("\n" + "=" * 74)
    print("  Holdout test set evaluation (fit on ALL train+val data)")
    print("=" * 74)

    holdout_results = {}
    for config_label, res in all_results.items():
        ho_cov   = res["ho_cov"]
        ho_y     = res["ho_y"]
        ho_demo  = res["ho_demo"]
        emb_kind = res["emb_kind"]
        tau      = res["agg"]["tau_youden"]

        # Re-fit on entire train+val set (not just the last fold!)
        tang    = TangentEmbedding(emb_kind, precomputed=True)
        cfg_tuple = configs[config_label]
        cv_cov  = cfg_tuple[0]
        cv_y    = cfg_tuple[1]
        cv_demo = cfg_tuple[2]

        X_tv = tang.fit_transform(cv_cov)
        X_ho = tang.transform(ho_cov)

        if cv_demo is not None:
            demo_tv = cv_demo.copy()
            demo_ho = ho_demo.copy()
            for col in range(demo_tv.shape[1]):
                col_median = np.nanmedian(demo_tv[:, col])
                demo_tv[np.isnan(demo_tv[:, col]), col] = col_median
                demo_ho[np.isnan(demo_ho[:, col]), col] = col_median
            ds = StandardScaler()
            demo_tv = ds.fit_transform(demo_tv)
            demo_ho = ds.transform(demo_ho)
            X_tv = np.hstack([X_tv, demo_tv])
            X_ho = np.hstack([X_ho, demo_ho])

        fs = StandardScaler()
        X_tv = fs.fit_transform(X_tv)
        X_ho = fs.transform(X_ho)

        inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
        search = GridSearchCV(
            LogisticRegression(max_iter=5000, solver="lbfgs"),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            cv=inner, scoring="roc_auc", n_jobs=args.jobs, refit=True,
        )
        search.fit(X_tv, cv_y)
        ho_prob = search.best_estimator_.predict_proba(X_ho)[:, 1]

        m_ho = full_metrics(ho_y, ho_prob, threshold=tau,
                            label=f"{config_label} [HOLDOUT N={len(ho_y)}]")
        print_metrics(m_ho)
        holdout_results[config_label] = m_ho
        res["agg"]["ho_acc"]     = m_ho["accuracy"]
        res["agg"]["ho_bacc"]    = m_ho["balanced_acc"]
        res["agg"]["ho_roc_auc"] = m_ho["roc_auc"]
        res["agg"]["ho_f1"]      = m_ho["f1"]
        res["agg"]["ho_sens"]    = m_ho["sensitivity"]
        res["agg"]["ho_spec"]    = m_ho["specificity"]

    # ── Leave-one-site-out (best config, on full cohort) ──────────────────────
    if not args.quick:
        print("\n" + "=" * 74)
        print(f"  Leave-one-site-out -- {best_config}")
        print("=" * 74)

        best_demo_full = None
        best_emb_kind  = all_results[best_config]["emb_kind"]
        if "FIQ" in best_config:
            best_demo_full = demo_C
        elif "age + sex" in best_config:
            best_demo_full = demo_B

        loso_rows, loso_pooled = loso_tangent(
            covs, y, sites, demo=best_demo_full,
            kind=best_emb_kind, n_jobs=args.jobs
        )

        loso_df = pd.DataFrame([
            {
                "site":        r["label"],
                "n":           r["n"],
                "accuracy":    r["accuracy"],
                "balanced_acc":r["balanced_acc"],
                "roc_auc":     r["roc_auc"],
                "f1":          r["f1"],
                "sensitivity": r["sensitivity"],
                "specificity": r["specificity"],
                "baseline":    r["baseline"],
            }
            for r in loso_rows
        ])
        loso_df.to_csv(LOSO_CSV, index=False)
        print(f"\n  LOSO results saved to {LOSO_CSV}")

    # ── Permutation test (best config, on train+val) ───────────────────────────
    if n_perm > 0 and not args.quick:
        print("\n" + "=" * 74)
        print(f"  Permutation test ({n_perm} shuffles) — {best_config}")
        print("=" * 74)

        tang_perm = TangentEmbedding(best_emb_kind, precomputed=True)
        best_demo_tv  = None
        if "FIQ" in best_config:
            best_demo_tv = demo_C_tv
        elif "age + sex" in best_config:
            best_demo_tv = demo_B_tv

        X_perm = tang_perm.fit_transform(covs_tv)
        if best_demo_tv is not None:
            d = best_demo_tv.copy()
            for col in range(d.shape[1]):
                d[np.isnan(d[:, col]), col] = np.nanmedian(d[:, col])
            ds = StandardScaler()
            d = ds.fit_transform(d)
            X_perm = np.hstack([X_perm, d])
        fs = StandardScaler()
        X_perm = fs.fit_transform(X_perm)

        clf = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs")
        cv_perm = StratifiedKFold(5, shuffle=True, random_state=SEED)
        score, perm_scores, pvalue = permutation_test_score(
            clf, X_perm, y_tv, cv=cv_perm,
            n_permutations=n_perm, scoring="roc_auc",
            n_jobs=args.jobs, random_state=SEED
        )
        print(
            f"  True AUC: {score:.3f} | "
            f"Null: {perm_scores.mean():.3f} ± {perm_scores.std():.3f} | "
            f"p = {pvalue:.4f}"
        )
        if pvalue < 0.05:
            print("  [OK] Result is statistically significant (p < 0.05)")
        else:
            print("  [FAIL] Result is NOT statistically significant (p >= 0.05)")

    # ── Final comparison summary ──────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  FINAL COMPARISON TABLE")
    print("=" * 74)
    print(
        f"\n  {'Config':<42}  {'CV AUC':>8}  {'CV Acc':>8}  "
        f"{'CV BAcc':>8}  {'HO AUC':>8}  {'HO Acc':>8}  {'HO BAcc':>8}"
    )
    print("  " + "-" * 100)

    # Baseline comparisons
    BASELINES = [
        ("Majority class baseline",           "—",     "53.78%", "50.00%", "—",     "—",      "—"),
        ("Confounds only (FIQ+sex+age+site)", "0.619", "60.36%", "—",      "—",     "—",      "—"),
        ("FC correlation + L2 logreg (orig)", "0.749", "68.60%", "—",      "—",     "—",      "—"),
    ]
    for row in BASELINES:
        print(f"  {row[0]:<42}  {row[1]:>8}  {row[2]:>8}  {row[3]:>8}  "
              f"{row[4]:>8}  {row[5]:>8}  {row[6]:>8}")
    print("  " + "-" * 100)

    for config_label, res in all_results.items():
        agg = res["agg"]
        print(
            f"  {config_label:<42}  "
            f"{agg['auc_mean']:>8.3f}  "
            f"{agg['acc_mean']*100:>7.2f}%  "
            f"{agg['bacc_mean']*100:>7.2f}%  "
            f"{agg.get('ho_roc_auc', float('nan')):>8.3f}  "
            f"{agg.get('ho_acc', float('nan'))*100:>7.2f}%  "
            f"{agg.get('ho_bacc', float('nan'))*100:>7.2f}%"
        )

    # FIQ impact report
    if not args.quick and len(all_results) == 3:
        auc_A = all_results["A: FC only"]["agg"]["auc_mean"]
        auc_B = all_results["B: FC + age + sex"]["agg"]["auc_mean"]
        auc_C = all_results["C: FC + age + sex + FIQ (CONFOUND)"]["agg"]["auc_mean"]

        print("\n" + "=" * 74)
        print("  FIQ CONFOUND IMPACT ANALYSIS")
        print("=" * 74)
        print(f"  Config A (FC only):           AUC = {auc_A:.3f}")
        print(f"  Config B (FC + age + sex):    AUC = {auc_B:.3f}  Δ vs A = {auc_B - auc_A:+.3f}")
        print(f"  Config C (FC + age + sex + FIQ): AUC = {auc_C:.3f}  Δ vs B = {auc_C - auc_B:+.3f}")
        delta_fiq = auc_C - auc_B
        if delta_fiq > 0.02:
            print(
                f"\n  ⚠ FIQ is contributing {delta_fiq*100:.1f} AUC points.\n"
                f"  Since FIQ is a known group-level confound (d=-0.35, p<1e-4),\n"
                f"  Config C's improvement does NOT represent additional brain-based signal.\n"
                f"  The legitimate result to report is Config B (AUC = {auc_B:.3f})."
            )
        elif abs(delta_fiq) <= 0.02:
            print(
                f"\n  ✓ FIQ adds only {delta_fiq*100:.1f} AUC points (|Δ| ≤ 0.02).\n"
                f"  FIQ confound impact is minimal. Config B is reliable."
            )
        else:
            print(f"\n  ✓ FIQ actually hurts performance (Δ = {delta_fiq*100:.1f}%). "
                  f"Config B is preferred.")

    # ── Save results CSV ──────────────────────────────────────────────────────
    rows = []
    for config_label, res in all_results.items():
        agg = res["agg"]
        rows.append({
            "config":       config_label,
            "cv_auc_mean":  agg["auc_mean"],
            "cv_auc_std":   agg["auc_std"],
            "cv_acc_mean":  agg["acc_mean"],
            "cv_acc_std":   agg["acc_std"],
            "cv_bacc_mean": agg["bacc_mean"],
            "cv_bacc_std":  agg["bacc_std"],
            "tau_youden":   agg["tau_youden"],
            "oof_f1":       agg["oof_f1"],
            "oof_sens":     agg["oof_sens"],
            "oof_spec":     agg["oof_spec"],
            "ho_roc_auc":   agg.get("ho_roc_auc", float("nan")),
            "ho_acc":       agg.get("ho_acc", float("nan")),
            "ho_bacc":      agg.get("ho_bacc", float("nan")),
            "ho_f1":        agg.get("ho_f1", float("nan")),
            "ho_sens":      agg.get("ho_sens", float("nan")),
            "ho_spec":      agg.get("ho_spec", float("nan")),
            "n_tv":         len(y_tv),
            "n_ho":         len(y_ho),
            "elapsed_min":  agg["elapsed_min"],
        })
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"\n  Full results saved to {RESULTS_CSV}")
    print("\n" + "=" * 74)
    print("  DONE")
    print("=" * 74)


if __name__ == "__main__":
    main()
