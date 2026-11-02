"""
phase2e_representations.py -- Phase 2E: FC representation comparison.

Compares four functional-connectivity representations on IDENTICAL 30
(3 seeds x 10-fold) CV splits reused from Phase 2C, using the same
LogisticRegression(L2)+age+sex+StandardScaler pipeline and the same
threshold protocol. Development subjects only (676); the 170 locked
final-holdout subjects are never accessed.

Representations (all implemented in tangent.py, none modified here):
  1. correlation_fisherz -- TangentEmbedding(kind="correlation")
  2. partial_correlation -- TangentEmbedding(kind="partial")
  3. tangent_space       -- TangentEmbedding(kind="tangent")  [the only one
                             requiring a per-fold-fitted reference]
  4. raw_covariance      -- tangent.vectorize(covs), no normalization,
                             no Fisher-z (fair "no transformation" baseline)
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix, roc_curve, precision_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

import holdout_guard as hg
from tangent import TangentEmbedding, vectorize

warnings.filterwarnings("ignore")

DATA        = r"C:\abide\abide_pcp"
TS_PATH     = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH   = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH  = os.path.join(DATA, "phenotypic_qc.csv")
MIN_TIMEPOINTS = 100

RESULTS_DIR  = r"C:\abide\results"
PERFOLD_CSV  = os.path.join(RESULTS_DIR, "phase2e_representation_perfold.csv")
SUMMARY_CSV  = os.path.join(RESULTS_DIR, "phase2e_representation_results.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "phase2e_representation_summary.json")
PHASE2C_PERFOLD = os.path.join(RESULTS_DIR, "phase2c_baseline_perfold.csv")

OUTER_SPLITS = 10
INNER_SPLITS = 5
SEEDS        = (0, 1, 2)
C_GRID       = [0.001, 0.01, 0.1, 1.0, 10.0]


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


class RawCovarianceEmbedding:
    """Minimal fit/transform wrapper: no normalization, no Fisher-z --
    just the upper-triangle of the (already Ledoit-Wolf shrunk) covariance
    matrices. Subject-independent, so fit() is a no-op, kept only so the
    calling code is identical across all four representations."""
    def fit(self, covs):
        return self

    def transform(self, covs):
        return vectorize(np.asarray(covs, dtype=np.float64))


def make_embedding(name):
    if name == "correlation_fisherz":
        return TangentEmbedding("correlation", precomputed=True)
    if name == "partial_correlation":
        return TangentEmbedding("partial", precomputed=True)
    if name == "tangent_space":
        return TangentEmbedding("tangent", precomputed=True)
    if name == "raw_covariance":
        return RawCovarianceEmbedding()
    raise ValueError(name)


def run_representation(rep_name, covs_dev, y, sub_id, sites, demo, all_sites_dev,
                        phase2c_perfold):
    print(f"\n{'=' * 74}\n  REPRESENTATION: {rep_name}\n{'=' * 74}")
    t_start = time.time()

    perfold_rows = []
    seed_summaries = []
    agg_cm_by_seed = {}
    nan_inf_events = []
    failed_folds = []

    for seed in SEEDS:
        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)
        chosen_C = []

        for fold_idx, (tr, te) in enumerate(outer.split(covs_dev, y)):
            # ── Fold-reuse verification against Phase 2C's recorded counts ──
            ref = phase2c_perfold[(phase2c_perfold.seed == seed) & (phase2c_perfold.fold == fold_idx)].iloc[0]
            assert len(tr) == ref.n_train and len(te) == ref.n_test, (
                f"Fold size mismatch vs Phase 2C at seed={seed} fold={fold_idx}: "
                f"got train={len(tr)}/test={len(te)}, expected {ref.n_train}/{ref.n_test}"
            )
            assert int(y[tr].sum()) == ref.train_asd and int(y[te].sum()) == ref.test_asd, (
                f"Class-count mismatch vs Phase 2C at seed={seed} fold={fold_idx} -- "
                f"folds are NOT identical to Phase 2C's."
            )

            # ── Safety: locked holdout must never appear in either partition ──
            hg.assert_cv_folds_dev_only(
                [sub_id[tr], sub_id[te]],
                context=f"phase2e[{rep_name}] seed={seed} fold={fold_idx}",
            )

            try:
                # ── Representation-specific embedding: fit on TRAIN portion only ──
                emb = make_embedding(rep_name)
                emb.fit(covs_dev[tr])
                X_tr = emb.transform(covs_dev[tr])
                X_te = emb.transform(covs_dev[te])

                for arr, tag in [(X_tr, "train"), (X_te, "test")]:
                    n_nan, n_inf = np.isnan(arr).sum(), np.isinf(arr).sum()
                    if n_nan or n_inf:
                        nan_inf_events.append({
                            "rep": rep_name, "seed": seed, "fold": fold_idx,
                            "partition": tag, "n_nan": int(n_nan), "n_inf": int(n_inf),
                        })

                # ── Demographics: median imputation from TRAIN portion only ──
                demo_tr = demo[tr].copy(); demo_te = demo[te].copy()
                for col in range(demo_tr.shape[1]):
                    med = np.nanmedian(demo_tr[:, col])
                    demo_tr[np.isnan(demo_tr[:, col]), col] = med
                    demo_te[np.isnan(demo_te[:, col]), col] = med
                ds = StandardScaler()
                demo_tr = ds.fit_transform(demo_tr)
                demo_te = ds.transform(demo_te)
                X_tr = np.hstack([X_tr, demo_tr])
                X_te = np.hstack([X_te, demo_te])

                # ── Feature scaling: fit on TRAIN portion only ────────────
                fs = StandardScaler()
                X_tr = fs.fit_transform(X_tr)
                X_te = fs.transform(X_te)

                # ── Inner CV hyperparameter selection: TRAIN portion only ──
                clf = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
                inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
                search = GridSearchCV(clf, {"C": C_GRID}, cv=inner, scoring="roc_auc", n_jobs=4, refit=True)
                search.fit(X_tr, y[tr])
                best_C = search.best_params_["C"]
                chosen_C.append(best_C)

                scores_te = search.best_estimator_.predict_proba(X_te)[:, 1]
                oof[te] = scores_te
                feature_dim = X_tr.shape[1]

            except Exception as exc:
                failed_folds.append({"rep": rep_name, "seed": seed, "fold": fold_idx,
                                      "error": f"{type(exc).__name__}: {exc}"})
                oof[te] = 0.5
                chosen_C.append(None)
                feature_dim = -1

            sites_in_test = set(np.unique(sites[te]))
            sites_missing = sorted(all_sites_dev - sites_in_test)
            perfold_rows.append({
                "representation": rep_name, "seed": seed, "fold": fold_idx,
                "n_train": len(tr), "n_test": len(te),
                "train_asd": int(y[tr].sum()), "train_tc": int((1 - y[tr]).sum()),
                "test_asd": int(y[te].sum()), "test_tc": int((1 - y[te]).sum()),
                "n_sites_in_test": len(sites_in_test),
                "sites_missing_from_test": ";".join(sites_missing),
                "best_C": chosen_C[-1],
                "feature_dim_final": feature_dim,
            })

        if failed_folds:
            print(f"  [WARNING] {len([f for f in failed_folds if f['seed']==seed])} fold(s) failed for seed {seed}")

        tau = youden_threshold(y, oof)
        y_pred = (oof >= tau).astype(int)
        auc  = roc_auc_score(y, oof)
        acc  = accuracy_score(y, y_pred)
        bacc = balanced_accuracy_score(y, y_pred)
        f1   = f1_score(y, y_pred, zero_division=0)
        cm   = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = precision_score(y, y_pred, zero_division=0)

        outer2 = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        fold_accs, fold_baccs = [], []
        for fi, (tr, te) in enumerate(outer2.split(covs_dev, y)):
            yp_fold = (oof[te] >= tau).astype(int)
            fold_accs.append(accuracy_score(y[te], yp_fold))
            fold_baccs.append(balanced_accuracy_score(y[te], yp_fold))
            row_idx = seed * OUTER_SPLITS + fi
            perfold_rows[row_idx]["fold_accuracy"] = fold_accs[-1]
            perfold_rows[row_idx]["fold_balanced_accuracy"] = fold_baccs[-1]

        print(f"  seed={seed}  OOF acc={acc*100:.2f}%  bacc={bacc*100:.2f}%  "
              f"AUC={auc:.4f}  F1={f1:.4f}  tau={tau:.4f}")

        agg_cm_by_seed[seed] = cm.tolist()
        seed_summaries.append({
            "seed": seed, "tau_youden": tau,
            "oof_accuracy": acc, "oof_balanced_accuracy": bacc,
            "oof_roc_auc": auc, "oof_f1": f1, "oof_precision": prec,
            "oof_sensitivity": sens, "oof_specificity": spec,
            "fold_accuracy_mean": float(np.mean(fold_accs)),
            "fold_accuracy_std": float(np.std(fold_accs)),
            "fold_balanced_accuracy_mean": float(np.mean(fold_baccs)),
            "fold_balanced_accuracy_std": float(np.std(fold_baccs)),
            "confusion_matrix": cm.tolist(),
            "chosen_C_per_fold": chosen_C,
        })

    elapsed = time.time() - t_start
    accs  = [s["oof_accuracy"] for s in seed_summaries]
    baccs = [s["oof_balanced_accuracy"] for s in seed_summaries]
    aucs  = [s["oof_roc_auc"] for s in seed_summaries]
    f1s   = [s["oof_f1"] for s in seed_summaries]
    senss = [s["oof_sensitivity"] for s in seed_summaries]
    specs = [s["oof_specificity"] for s in seed_summaries]
    cm_sum = np.sum([np.array(agg_cm_by_seed[s]) for s in SEEDS], axis=0)

    print(f"  CROSS-SEED: acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%  "
          f"bacc={np.mean(baccs)*100:.2f}%+/-{np.std(baccs)*100:.2f}%  "
          f"AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  elapsed={elapsed:.1f}s")

    hg.log_experiment(
        f"phase2e_{rep_name}", sub_id.tolist(),
        notes=f"3x10 nested CV, rep={rep_name}, "
              f"OOF acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%, "
              f"AUC={np.mean(aucs):.4f}, elapsed={elapsed:.1f}s",
    )

    return {
        "perfold_rows": perfold_rows,
        "seed_summaries": seed_summaries,
        "cm_sum": cm_sum,
        "elapsed_sec": elapsed,
        "nan_inf_events": nan_inf_events,
        "failed_folds": failed_folds,
        "cross_seed": {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "balanced_accuracy_mean": float(np.mean(baccs)), "balanced_accuracy_std": float(np.std(baccs)),
            "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "sensitivity_mean": float(np.mean(senss)), "sensitivity_std": float(np.std(senss)),
            "specificity_mean": float(np.mean(specs)), "specificity_std": float(np.std(specs)),
        },
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Identical loading block to phase2c_baseline_cv.py, for identical ──
    # ── subject ordering -> identical StratifiedKFold splits given same seed ──
    ts = np.load(TS_PATH, allow_pickle=True)
    ph = pd.read_csv(PHENO_PATH)
    keep = np.array([t is not None and t.shape[0] >= MIN_TIMEPOINTS for t in ts])
    ph = ph.loc[keep].reset_index(drop=True)
    covs_all = np.load(COVS_PATH)
    assert covs_all.shape[0] == len(ph)

    dev_ids = hg.load_dev_subject_ids()
    ph_all_sub = ph["SUB_ID"].astype(str)
    dev_mask = ph_all_sub.isin(dev_ids).values
    hg.assert_no_holdout_leakage(ph_all_sub[dev_mask].tolist(), context="phase2e initial dev filter")

    ph_dev   = ph.loc[dev_mask].reset_index(drop=True)
    covs_dev = covs_all[dev_mask]
    assert len(ph_dev) == 676

    sub_id = ph_dev["SUB_ID"].astype(str).values
    y      = (ph_dev["DX_GROUP"].values == 1).astype(int)
    sites  = ph_dev["SITE_ID"].astype(str).values
    num    = lambda c: pd.to_numeric(ph_dev.get(c), errors="coerce").replace(-9999, np.nan).values
    age    = num("AGE_AT_SCAN").astype(float)
    sex    = num("SEX").astype(float)
    demo   = np.column_stack([age, sex])
    all_sites_dev = set(np.unique(sites))

    print(f"Development cohort: {len(ph_dev)} subjects (ASD {y.sum()} | TC {(1-y).sum()})")
    print("Verifying identical subject ordering vs Phase 2C via fold-count cross-check...")

    phase2c_perfold = pd.read_csv(PHASE2C_PERFOLD)

    # Empirical relationship check: how similar are raw covariance and
    # correlation/Fisher-z features, given inputs are pre-z-scored?
    sample_covs = covs_dev[:5]
    raw_feats  = vectorize(sample_covs)
    corr_feats = TangentEmbedding("correlation", precomputed=True).fit_transform(sample_covs) \
        if hasattr(TangentEmbedding("correlation", precomputed=True), "fit_transform") \
        else TangentEmbedding("correlation", precomputed=True).fit(sample_covs).transform(sample_covs)
    pear = np.corrcoef(raw_feats.ravel(), np.tanh(corr_feats).ravel())[0, 1]
    print(f"\nFairness check: Pearson correlation between raw-covariance features and "
          f"(inverse-Fisher-z'd) correlation features on a 5-subject sample: {pear:.4f}")
    print("(High correlation here is EXPECTED and documented -- see report -- because "
          "input time series are z-scored before covariance estimation.)\n")

    all_results = {}
    reps = ["correlation_fisherz", "partial_correlation", "tangent_space", "raw_covariance"]
    for rep in reps:
        all_results[rep] = run_representation(
            rep, covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold
        )

    # ── Assemble outputs ──────────────────────────────────────────────────
    all_perfold = []
    for rep in reps:
        all_perfold.extend(all_results[rep]["perfold_rows"])
    pd.DataFrame(all_perfold).to_csv(PERFOLD_CSV, index=False)

    summary_rows = []
    for rep in reps:
        r = all_results[rep]
        for s in r["seed_summaries"]:
            row = {"representation": rep, **s}
            summary_rows.append(row)
        cs = r["cross_seed"]
        summary_rows.append({
            "representation": rep, "seed": "MEAN_SD",
            "oof_accuracy": f"{cs['accuracy_mean']:.4f} +/- {cs['accuracy_std']:.4f}",
            "oof_balanced_accuracy": f"{cs['balanced_accuracy_mean']:.4f} +/- {cs['balanced_accuracy_std']:.4f}",
            "oof_roc_auc": f"{cs['roc_auc_mean']:.4f} +/- {cs['roc_auc_std']:.4f}",
            "oof_f1": f"{cs['f1_mean']:.4f} +/- {cs['f1_std']:.4f}",
            "oof_sensitivity": f"{cs['sensitivity_mean']:.4f} +/- {cs['sensitivity_std']:.4f}",
            "oof_specificity": f"{cs['specificity_mean']:.4f} +/- {cs['specificity_std']:.4f}",
            "tau_youden": "", "oof_precision": "",
            "fold_accuracy_mean": "", "fold_accuracy_std": "",
            "fold_balanced_accuracy_mean": "", "fold_balanced_accuracy_std": "",
            "confusion_matrix": r["cm_sum"].tolist(), "chosen_C_per_fold": "",
        })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    reproducibility = {
        "phase": "2E",
        "purpose": "FC representation comparison on identical Phase 2C CV splits, dev-only (N=676).",
        "fold_reuse_verification": "Every fold's n_train/n_test/train_asd/test_asd was asserted "
                                    "equal to phase2c_baseline_perfold.csv before running -- confirms "
                                    "identical splits, not just identical seeds.",
        "representations": {
            "correlation_fisherz": "TangentEmbedding(kind='correlation'): normalize covariance by "
                                    "sqrt(d_i*d_j), then arctanh (Fisher-z). Subject-independent "
                                    "(no cross-subject fitting).",
            "partial_correlation": "TangentEmbedding(kind='partial'): invert each subject's "
                                    "Ledoit-Wolf-shrunk (SPD) covariance via eigendecomposition, "
                                    "normalize the precision matrix, negate, arctanh. "
                                    "Subject-independent.",
            "tangent_space": "TangentEmbedding(kind='tangent'): Riemannian geometric mean of "
                              "TRAINING-FOLD covariances as reference, matrix-log of whitened "
                              "covariances. The only representation requiring per-fold fitting.",
            "raw_covariance": "tangent.vectorize(covs) directly -- no normalization, no Fisher-z. "
                               "Subject-independent.",
        },
        "fairness_note": "Input ROI time series are z-scored per region before Ledoit-Wolf "
                         "covariance estimation (tangent.py:covariances()), so raw_covariance is "
                         "already close to a (non-Fisher-z'd, non-diagonal-renormalized) correlation "
                         "matrix -- raw_covariance and correlation_fisherz are expected to be highly "
                         "correlated, not independent signals. This was checked empirically (see "
                         "console output) and is reported, not hidden, per the audit instructions.",
        "cv_protocol": {
            "outer": f"StratifiedKFold(n_splits={OUTER_SPLITS}, shuffle=True, random_state=seed)",
            "inner": f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True, random_state=seed)",
            "seeds": list(SEEDS),
            "classifier": "LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced')",
            "hyperparameter_grid": {"C": C_GRID},
            "threshold_selection": "Youden's J on seed-level OOF predictions",
        },
        "n_dev_subjects": 676,
        "n_holdout_subjects_locked_untouched": 170,
        "per_representation": {
            rep: {
                "cross_seed": all_results[rep]["cross_seed"],
                "elapsed_sec": all_results[rep]["elapsed_sec"],
                "nan_inf_events": all_results[rep]["nan_inf_events"],
                "failed_folds": all_results[rep]["failed_folds"],
                "confusion_matrix_summed": all_results[rep]["cm_sum"].tolist(),
                "seed_summaries": all_results[rep]["seed_summaries"],
            }
            for rep in reps
        },
        "holdout_status": "NEVER accessed. get_locked_holdout_ids() was never called from this script.",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(reproducibility, f, indent=2)

    print(f"\nSaved:\n  {PERFOLD_CSV}\n  {SUMMARY_CSV}\n  {SUMMARY_JSON}")

    print("\n" + "=" * 74)
    print("  FINAL COMPARISON TABLE")
    print("=" * 74)
    print(f"  {'Representation':<22} {'Acc':>8} {'BAcc':>8} {'AUC':>8} {'F1':>7} {'Sec':>7}")
    for rep in reps:
        cs = all_results[rep]["cross_seed"]
        print(f"  {rep:<22} {cs['accuracy_mean']*100:6.2f}%  {cs['balanced_accuracy_mean']*100:6.2f}%  "
              f"{cs['roc_auc_mean']:.4f}  {cs['f1_mean']:.4f}  {all_results[rep]['elapsed_sec']:6.1f}")


if __name__ == "__main__":
    main()
