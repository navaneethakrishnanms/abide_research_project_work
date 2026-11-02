"""
phase2g_fast_classifier_comparison.py -- FAST DEVELOPMENT VERSION of Phase 2G.

v2: the first attempt at this fast version stalled on LogReg_L2 (~16 minutes,
no seed completed) despite PCA(200)-reduced features and a small grid. The
process was confirmed alive (not hung) but consuming far more CPU time than
wall-clock time would justify for this workload -- the classic signature of
thread oversubscription: GridSearchCV(n_jobs=-1) spawns multiple worker
processes, and each one's NumPy/SciPy calls ALSO spawn multiple BLAS threads
(OpenMP/MKL/OpenBLAS), so a machine with N cores ends up running many times N
threads, fighting each other for the same cores instead of using them.

Fix: cap every threading layer to 1 thread (env vars set BEFORE numpy/sklearn
are imported -- setting them after import has no effect) and use n_jobs=1
throughout. With PCA(200) already doing the heavy dimensionality cut, single-
threaded execution should be more than fast enough, and it removes an entire
class of "is it hung or just fighting itself" uncertainty.

This script also supports --smoke: runs ONLY LogReg_L2, seed=0, to prove the
fix works before committing to the full 3-seed x 6-classifier run.

Same leakage-safety design as v1: the correlation/Fisher-z embedding is
subject-independent (verified in Phase 2E) and computed once for the whole
676-subject development set; every fold-dependent step (demographic
imputation, scaling, PCA, hyperparameter search) is still fit strictly
inside each training fold. The 170 locked holdout subjects are never
accessed.
"""
from __future__ import annotations

# ── MUST happen before numpy/scipy/sklearn/xgboost are imported ────────────
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, confusion_matrix, roc_curve, precision_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

import holdout_guard as hg
from tangent import TangentEmbedding

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

DATA        = r"C:\abide\abide_pcp"
TS_PATH     = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH   = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH  = os.path.join(DATA, "phenotypic_qc.csv")
MIN_TIMEPOINTS = 100

RESULTS_DIR  = r"C:\abide\results"
PERFOLD_CSV  = os.path.join(RESULTS_DIR, "phase2g_fast_classifier_perfold.csv")
SUMMARY_CSV  = os.path.join(RESULTS_DIR, "phase2g_fast_classifier_summary.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "phase2g_fast_classifier_summary.json")

OUTER_SPLITS = 5
INNER_SPLITS = 3
PCA_COMPONENTS = 200
SEED_STATIC  = 42
N_JOBS = 1   # conservative everywhere -- see module docstring


def make_classifiers():
    clfs = {
        "LogReg_L2": (
            LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        ),
        "LogReg_L1": (
            LogisticRegression(max_iter=5000, solver="liblinear", penalty="l1",
                                class_weight="balanced"),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        ),
        "ElasticNet": (
            LogisticRegression(max_iter=3000, solver="saga", penalty="elasticnet",
                                class_weight="balanced"),
            {"C": [0.1, 1.0], "l1_ratio": [0.5, 0.8]},   # very small grid, per instructions
        ),
        "LinearSVM": (
            CalibratedClassifierCV(
                LinearSVC(max_iter=5000, class_weight="balanced"), cv=3
            ),
            {"estimator__C": [0.001, 0.01, 0.1, 1.0]},
        ),
        "RBF_SVM": (
            SVC(probability=True, class_weight="balanced"),
            {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.001, 0.01]},
        ),
    }
    if XGB_AVAILABLE:
        clfs["XGBoost"] = (
            xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False,
                               scale_pos_weight=1.0, n_jobs=N_JOBS),
            {"n_estimators": [100, 200], "max_depth": [3, 4],
             "learning_rate": [0.05, 0.1], "subsample": [0.8], "colsample_bytree": [0.8]},
        )
    return clfs


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


def score_clf(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def run_classifier(name, clf, grid, X_fc_full, y, sub_id, sites, demo, all_sites_dev, seeds):
    print(f"\n{'=' * 74}\n  CLASSIFIER (FAST v2, n_jobs=1): {name}\n{'=' * 74}", flush=True)
    t_start = time.time()

    perfold_rows = []
    seed_summaries = []
    agg_cm_by_seed = {}
    failed_folds = []

    for seed in seeds:
        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)
        chosen_params = []
        fold_t0 = time.time()

        for fold_idx, (tr, te) in enumerate(outer.split(X_fc_full, y)):
            hg.assert_cv_folds_dev_only(
                [sub_id[tr], sub_id[te]],
                context=f"phase2g_fast[{name}] seed={seed} fold={fold_idx}",
            )

            try:
                X_tr = X_fc_full[tr].copy()
                X_te = X_fc_full[te].copy()

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

                fs = StandardScaler()
                X_tr = fs.fit_transform(X_tr)
                X_te = fs.transform(X_te)

                n_comp = min(PCA_COMPONENTS, X_tr.shape[0] - 1, X_tr.shape[1])
                pca = PCA(n_components=n_comp, random_state=SEED_STATIC)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)

                inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
                search = GridSearchCV(clf, grid, cv=inner, scoring="roc_auc", n_jobs=N_JOBS, refit=True)
                search.fit(X_tr, y[tr])
                chosen_params.append(search.best_params_)

                scores_te = score_clf(search.best_estimator_, X_te)
                oof[te] = scores_te
                n_pca = n_comp
                fold_elapsed = time.time() - fold_t0
                print(f"    seed={seed} fold={fold_idx} done ({fold_elapsed:.1f}s cumulative)", flush=True)

            except Exception as exc:
                failed_folds.append({"classifier": name, "seed": seed, "fold": fold_idx,
                                      "error": f"{type(exc).__name__}: {exc}"})
                oof[te] = 0.5
                chosen_params.append(None)
                n_pca = -1
                print(f"    seed={seed} fold={fold_idx} FAILED: {exc}", flush=True)

            sites_in_test = set(np.unique(sites[te]))
            sites_missing = sorted(all_sites_dev - sites_in_test)
            perfold_rows.append({
                "phase": "2G_fast", "representation": "correlation_fisherz",
                "feature_selection": "PCA", "k": n_pca,
                "classifier": name, "seed": seed, "outer_fold": fold_idx,
                "n_train": len(tr), "n_test": len(te),
                "train_ASD": int(y[tr].sum()), "train_TC": int((1 - y[tr]).sum()),
                "test_ASD": int(y[te].sum()), "test_TC": int((1 - y[te]).sum()),
                "n_sites_in_test": len(sites_in_test),
                "sites_missing_from_test": ";".join(sites_missing),
                "selected_features": n_pca,
                "best_hyperparameters": str(chosen_params[-1]),
            })

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
        fold_accs, fold_baccs, fold_aucs, fold_f1s, fold_sens, fold_specs = [], [], [], [], [], []
        for fi, (tr, te) in enumerate(outer2.split(X_fc_full, y)):
            yp_fold = (oof[te] >= tau).astype(int)
            fold_accs.append(accuracy_score(y[te], yp_fold))
            fold_baccs.append(balanced_accuracy_score(y[te], yp_fold))
            fold_f1s.append(f1_score(y[te], yp_fold, zero_division=0))
            cm_f = confusion_matrix(y[te], yp_fold, labels=[0, 1])
            tnf, fpf, fnf, tpf = cm_f.ravel()
            fold_sens.append(tpf / (tpf + fnf) if (tpf + fnf) else float("nan"))
            fold_specs.append(tnf / (tnf + fpf) if (tnf + fpf) else float("nan"))
            try:
                fold_aucs.append(roc_auc_score(y[te], oof[te]))
            except ValueError:
                fold_aucs.append(float("nan"))
            row_idx = seed * OUTER_SPLITS + fi
            perfold_rows[row_idx]["accuracy"] = fold_accs[-1]
            perfold_rows[row_idx]["balanced_accuracy"] = fold_baccs[-1]
            perfold_rows[row_idx]["roc_auc"] = fold_aucs[-1]
            perfold_rows[row_idx]["f1"] = fold_f1s[-1]
            perfold_rows[row_idx]["sensitivity"] = fold_sens[-1]
            perfold_rows[row_idx]["specificity"] = fold_specs[-1]

        print(f"  seed={seed}  OOF acc={acc*100:.2f}%  bacc={bacc*100:.2f}%  "
              f"AUC={auc:.4f}  F1={f1:.4f}  ({time.time()-fold_t0:.1f}s for this seed)", flush=True)

        agg_cm_by_seed[seed] = cm.tolist()
        seed_summaries.append({
            "seed": seed, "tau_youden": tau,
            "oof_accuracy": acc, "oof_balanced_accuracy": bacc,
            "oof_roc_auc": auc, "oof_f1": f1, "oof_precision": prec,
            "oof_sensitivity": sens, "oof_specificity": spec,
            "fold_accuracy_mean": float(np.mean(fold_accs)), "fold_accuracy_std": float(np.std(fold_accs)),
            "fold_balanced_accuracy_mean": float(np.mean(fold_baccs)), "fold_balanced_accuracy_std": float(np.std(fold_baccs)),
            "confusion_matrix": cm.tolist(), "chosen_params_per_fold": chosen_params,
        })

    elapsed = time.time() - t_start
    for row in perfold_rows:
        row["runtime"] = elapsed

    accs  = [s["oof_accuracy"] for s in seed_summaries]
    baccs = [s["oof_balanced_accuracy"] for s in seed_summaries]
    aucs  = [s["oof_roc_auc"] for s in seed_summaries]
    f1s   = [s["oof_f1"] for s in seed_summaries]
    senss = [s["oof_sensitivity"] for s in seed_summaries]
    specs = [s["oof_specificity"] for s in seed_summaries]
    cm_sum = np.sum([np.array(agg_cm_by_seed[s]) for s in seeds], axis=0)

    print(f"  CROSS-SEED: acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%  "
          f"bacc={np.mean(baccs)*100:.2f}%+/-{np.std(baccs)*100:.2f}%  "
          f"AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  elapsed={elapsed:.1f}s  "
          f"failed_folds={len(failed_folds)}", flush=True)

    hg.log_experiment(
        f"phase2g_fast_{name}", sub_id.tolist(),
        notes=f"FAST v2 (n_jobs=1, thread-capped) dev protocol: {len(seeds)}x{OUTER_SPLITS} outer / "
              f"{INNER_SPLITS} inner CV, PCA(200), classifier={name}, "
              f"OOF acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%, "
              f"AUC={np.mean(aucs):.4f}, elapsed={elapsed:.1f}s, failed_folds={len(failed_folds)}",
    )

    return {
        "name": name, "perfold_rows": perfold_rows, "seed_summaries": seed_summaries,
        "cm_sum": cm_sum, "elapsed_sec": elapsed, "failed_folds": failed_folds,
        "cross_seed": {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "balanced_accuracy_mean": float(np.mean(baccs)), "balanced_accuracy_std": float(np.std(baccs)),
            "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "sensitivity_mean": float(np.mean(senss)), "sensitivity_std": float(np.std(senss)),
            "specificity_mean": float(np.mean(specs)), "specificity_std": float(np.std(specs)),
        },
    }


def load_dev_data():
    ts = np.load(TS_PATH, allow_pickle=True)
    ph = pd.read_csv(PHENO_PATH)
    keep = np.array([t is not None and t.shape[0] >= MIN_TIMEPOINTS for t in ts])
    ph = ph.loc[keep].reset_index(drop=True)
    covs_all = np.load(COVS_PATH)
    assert covs_all.shape[0] == len(ph)

    dev_ids = hg.load_dev_subject_ids()
    ph_all_sub = ph["SUB_ID"].astype(str)
    dev_mask = ph_all_sub.isin(dev_ids).values
    hg.assert_no_holdout_leakage(ph_all_sub[dev_mask].tolist(), context="phase2g_fast initial dev filter")

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
    return covs_dev, y, sub_id, sites, demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="Run ONLY LogReg_L2, seed=0, to verify the threading fix before the full run.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    covs_dev, y, sub_id, sites, demo = load_dev_data()
    all_sites_dev = set(np.unique(sites))

    print(f"Development cohort: {len(y)} subjects (ASD {y.sum()} | TC {(1-y).sum()})", flush=True)
    print(f"XGBoost available: {XGB_AVAILABLE}", flush=True)
    print(f"Thread env vars: OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}  |  N_JOBS={N_JOBS}", flush=True)
    print(f"MODE: {'SMOKE TEST (LogReg_L2, seed=0 only)' if args.smoke else 'FULL RUN (3 seeds, 6 classifiers)'}", flush=True)

    t0 = time.time()
    emb = TangentEmbedding("correlation", precomputed=True)
    X_fc_full = emb.fit(covs_dev).transform(covs_dev)
    print(f"Precomputed correlation/Fisher-z features for all {len(y)} dev subjects "
          f"in {time.time()-t0:.1f}s -- shape {X_fc_full.shape}", flush=True)

    classifiers = make_classifiers()

    if args.smoke:
        name = "LogReg_L2"
        clf, grid = classifiers[name]
        t0 = time.time()
        result = run_classifier(name, clf, grid, X_fc_full, y, sub_id, sites, demo, all_sites_dev, seeds=(0,))
        elapsed = time.time() - t0
        print(f"\n{'=' * 74}\n  SMOKE TEST RESULT\n{'=' * 74}")
        print(f"  Elapsed: {elapsed:.1f}s for 1 seed x {OUTER_SPLITS} outer folds "
              f"({INNER_SPLITS}-fold inner CV, PCA(200), n_jobs={N_JOBS})")
        print(f"  Accuracy: {result['cross_seed']['accuracy_mean']*100:.2f}%  "
              f"AUC: {result['cross_seed']['roc_auc_mean']:.4f}")
        print(f"  Failed folds: {len(result['failed_folds'])}")
        if elapsed < 300:
            print("  [OK] Smoke test completed in a reasonable time. Safe to proceed to the full run.")
        else:
            print("  [WARNING] Smoke test still slow -- investigate further before the full run.")
        return

    seeds = (0, 1, 2)
    all_results = {}
    for name, (clf, grid) in classifiers.items():
        all_results[name] = run_classifier(
            name, clf, grid, X_fc_full, y, sub_id, sites, demo, all_sites_dev, seeds
        )

    if not XGB_AVAILABLE:
        print("\n[NOTE] xgboost not importable -- skipped, recorded as unavailable.")

    all_perfold = []
    for name, r in all_results.items():
        all_perfold.extend(r["perfold_rows"])
    pd.DataFrame(all_perfold).to_csv(PERFOLD_CSV, index=False)

    summary_rows = []
    for name, r in all_results.items():
        for s in r["seed_summaries"]:
            summary_rows.append({"classifier": name, **s})
        cs = r["cross_seed"]
        summary_rows.append({
            "classifier": name, "seed": "MEAN_SD",
            "oof_accuracy": f"{cs['accuracy_mean']:.4f} +/- {cs['accuracy_std']:.4f}",
            "oof_balanced_accuracy": f"{cs['balanced_accuracy_mean']:.4f} +/- {cs['balanced_accuracy_std']:.4f}",
            "oof_roc_auc": f"{cs['roc_auc_mean']:.4f} +/- {cs['roc_auc_std']:.4f}",
            "oof_f1": f"{cs['f1_mean']:.4f} +/- {cs['f1_std']:.4f}",
            "oof_sensitivity": f"{cs['sensitivity_mean']:.4f} +/- {cs['sensitivity_std']:.4f}",
            "oof_specificity": f"{cs['specificity_mean']:.4f} +/- {cs['specificity_std']:.4f}",
            "tau_youden": "", "oof_precision": "",
            "fold_accuracy_mean": "", "fold_accuracy_std": "",
            "fold_balanced_accuracy_mean": "", "fold_balanced_accuracy_std": "",
            "confusion_matrix": r["cm_sum"].tolist(), "chosen_params_per_fold": "",
            "elapsed_sec": r["elapsed_sec"], "n_failed_folds": len(r["failed_folds"]),
        })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    usable = {k: v for k, v in all_results.items() if len(v["failed_folds"]) < len(seeds) * OUTER_SPLITS}
    ranked = sorted(usable.items(),
                     key=lambda kv: (kv[1]["cross_seed"]["roc_auc_mean"],
                                      kv[1]["cross_seed"]["balanced_accuracy_mean"]),
                     reverse=True)
    winner_name, winner = ranked[0] if ranked else (None, None)

    reproducibility = {
        "phase": "2G_fast_v2",
        "purpose": "FAST DEVELOPMENT VERSION of Phase 2G classifier comparison, v2 (threading fixed). "
                   "5 outer / 3 inner folds, PCA(200) for all classifiers, n_jobs=1 everywhere, "
                   "OMP/MKL/OPENBLAS/NUMEXPR capped to 1 thread each.",
        "v1_failure_reason": "v1 (n_jobs=-1) stalled on LogReg_L2 for >16 minutes with no fold "
                             "completed, consistent with BLAS/joblib thread oversubscription. "
                             "Process was killed (PID confirmed via CommandLine before termination) "
                             "and rewritten with capped threading; Phase 2F results and the old "
                             "phase2g_LogReg_L2/L1 audit entries were verified untouched afterward.",
        "feature_config": "correlation_fisherz (precomputed once, subject-independent) + age + sex, "
                          "then per-fold: StandardScaler(fit train) -> PCA(200, fit train)",
        "classifiers_tested": list(classifiers.keys()),
        "xgboost_available": XGB_AVAILABLE,
        "cv_protocol": {
            "outer": f"StratifiedKFold(n_splits={OUTER_SPLITS}, shuffle=True, random_state=seed)",
            "inner": f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True, random_state=seed)",
            "seeds": list(seeds),
            "pca_components": PCA_COMPONENTS,
            "n_jobs": N_JOBS,
            "threshold_selection": "Youden's J on seed-level OOF predictions",
        },
        "n_dev_subjects": 676,
        "n_holdout_subjects_locked_untouched": 170,
        "ranking_criterion": "CV ROC-AUC primary, balanced accuracy secondary",
        "winner": winner_name,
        "winner_cross_seed": winner["cross_seed"] if winner else None,
        "per_classifier": {
            name: {
                "cross_seed": r["cross_seed"], "elapsed_sec": r["elapsed_sec"],
                "failed_folds": r["failed_folds"],
                "confusion_matrix_summed": r["cm_sum"].tolist(),
                "seed_summaries": r["seed_summaries"],
            }
            for name, r in all_results.items()
        },
        "holdout_status": "NEVER accessed. get_locked_holdout_ids() was never called from this script.",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(reproducibility, f, indent=2)

    print(f"\nSaved:\n  {PERFOLD_CSV}\n  {SUMMARY_CSV}\n  {SUMMARY_JSON}")
    print("\n" + "=" * 74)
    print("  FINAL FAST COMPARISON TABLE (ranked by CV AUC)")
    print("=" * 74)
    print(f"  {'Classifier':<14} {'Acc':>8} {'BAcc':>8} {'AUC':>8} {'F1':>7} {'Sec':>7} {'Failed':>7}")
    for name, r in ranked:
        cs = r["cross_seed"]
        print(f"  {name:<14} {cs['accuracy_mean']*100:6.2f}%  {cs['balanced_accuracy_mean']*100:6.2f}%  "
              f"{cs['roc_auc_mean']:.4f}  {cs['f1_mean']:.4f}  {r['elapsed_sec']:6.1f}  {len(r['failed_folds']):6d}")
    for name, r in all_results.items():
        if name not in usable:
            print(f"  {name:<14} FAILED on all folds -- excluded from ranking")
    if winner_name:
        print(f"\n  WINNER (by CV AUC, then balanced accuracy): {winner_name}")


if __name__ == "__main__":
    main()
