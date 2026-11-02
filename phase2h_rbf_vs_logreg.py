"""
phase2h_rbf_vs_logreg.py -- Phase 2H: rigorous RBF-SVM vs Logistic Regression
L2 head-to-head, under the ORIGINAL rigorous protocol (10 outer / 5 inner
folds, 3 seeds, full 19,900 correlation/Fisher-z features + age+sex, NO PCA,
NO fold-count reduction) -- reusing the exact Phase 2C fold definitions,
verified the same way Phase 2E/2F/2G did.

This deliberately does NOT use the fast Phase 2G protocol (5-fold, PCA(200)):
that RBF_SVM win was on a cheaper probe and is explicitly not trusted here.
Phase 2H exists to check whether RBF-SVM still wins under the real protocol.

Threading fix from Phase 2G (thread oversubscription caused a multi-hour
stall) applied identically: OMP/MKL/OPENBLAS/NUMEXPR capped to 1 thread,
n_jobs=1 everywhere.

Development subjects only (676). The 170 locked holdout subjects are never
accessed.
"""
from __future__ import annotations

# ── MUST happen before numpy/scipy/sklearn are imported ────────────────────
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
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
from sklearn.svm import SVC

import holdout_guard as hg
from tangent import TangentEmbedding

warnings.filterwarnings("ignore")

DATA        = r"C:\abide\abide_pcp"
TS_PATH     = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH   = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH  = os.path.join(DATA, "phenotypic_qc.csv")
MIN_TIMEPOINTS = 100

RESULTS_DIR     = r"C:\abide\results"
PERFOLD_CSV     = os.path.join(RESULTS_DIR, "phase2h_rbf_vs_logreg_perfold.csv")
SUMMARY_CSV     = os.path.join(RESULTS_DIR, "phase2h_rbf_vs_logreg_summary.csv")
SUMMARY_JSON    = os.path.join(RESULTS_DIR, "phase2h_rbf_vs_logreg_summary.json")
PHASE2C_PERFOLD = os.path.join(RESULTS_DIR, "phase2c_baseline_perfold.csv")

OUTER_SPLITS = 10
INNER_SPLITS = 5
SEEDS        = (0, 1, 2)
N_JOBS       = 1


def make_models(include_gamma_auto: bool):
    gammas = ["scale", "auto"] if include_gamma_auto else ["scale"]
    return {
        "LogReg_L2": (
            LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        ),
        "RBF_SVM": (
            SVC(probability=True, class_weight="balanced"),
            {"C": [0.1, 1.0, 10.0], "gamma": gammas},
        ),
    }


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


def score_clf(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def run_model(name, clf, grid, covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold,
              time_budget_sec=None):
    print(f"\n{'=' * 74}\n  MODEL: {name}\n{'=' * 74}", flush=True)
    t_start = time.time()

    perfold_rows = []
    seed_summaries = []
    agg_cm_by_seed = {}
    failed_folds = []
    aborted = False

    for seed in SEEDS:
        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)
        chosen_params = []

        for fold_idx, (tr, te) in enumerate(outer.split(covs_dev, y)):
            if time_budget_sec is not None and (time.time() - t_start) > time_budget_sec:
                print(f"  [TIME BUDGET EXCEEDED at {time.time()-t_start:.1f}s -- "
                      f"stopping {name} safely at seed={seed} fold={fold_idx}]", flush=True)
                aborted = True
                break

            ref = phase2c_perfold[(phase2c_perfold.seed == seed) & (phase2c_perfold.fold == fold_idx)].iloc[0]
            assert len(tr) == ref.n_train and len(te) == ref.n_test
            assert int(y[tr].sum()) == ref.train_asd and int(y[te].sum()) == ref.test_asd

            hg.assert_cv_folds_dev_only(
                [sub_id[tr], sub_id[te]],
                context=f"phase2h[{name}] seed={seed} fold={fold_idx}",
            )

            fold_t0 = time.time()
            try:
                emb = TangentEmbedding("correlation", precomputed=True)
                X_tr = emb.fit(covs_dev[tr]).transform(covs_dev[tr])
                X_te = emb.transform(covs_dev[te])

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

                inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
                search = GridSearchCV(clf, grid, cv=inner, scoring="roc_auc", n_jobs=N_JOBS, refit=True)
                search.fit(X_tr, y[tr])
                chosen_params.append(search.best_params_)

                scores_te = score_clf(search.best_estimator_, X_te)
                oof[te] = scores_te
                feature_dim = X_tr.shape[1]
                fold_elapsed = time.time() - fold_t0
                print(f"    seed={seed} fold={fold_idx} done in {fold_elapsed:.1f}s "
                      f"(total {time.time()-t_start:.1f}s)  best={search.best_params_}", flush=True)

            except Exception as exc:
                failed_folds.append({"model": name, "seed": seed, "fold": fold_idx,
                                      "error": f"{type(exc).__name__}: {exc}"})
                oof[te] = 0.5
                chosen_params.append(None)
                feature_dim = -1
                print(f"    seed={seed} fold={fold_idx} FAILED: {exc}", flush=True)

            sites_in_test = set(np.unique(sites[te]))
            sites_missing = sorted(all_sites_dev - sites_in_test)
            perfold_rows.append({
                "phase": "2H", "representation": "correlation_fisherz",
                "feature_selection": "none", "k": "all_19900",
                "classifier": name, "seed": seed, "outer_fold": fold_idx,
                "n_train": len(tr), "n_test": len(te),
                "train_ASD": int(y[tr].sum()), "train_TC": int((1 - y[tr]).sum()),
                "test_ASD": int(y[te].sum()), "test_TC": int((1 - y[te]).sum()),
                "n_sites_in_test": len(sites_in_test),
                "sites_missing_from_test": ";".join(sites_missing),
                "selected_features": feature_dim,
                "best_hyperparameters": str(chosen_params[-1]),
                "fold_runtime_sec": time.time() - fold_t0,
            })

        if aborted:
            break

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
        for fi, (tr, te) in enumerate(outer2.split(covs_dev, y)):
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
              f"AUC={auc:.4f}  F1={f1:.4f}", flush=True)

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

    result = {
        "name": name, "perfold_rows": perfold_rows, "seed_summaries": seed_summaries,
        "elapsed_sec": elapsed, "failed_folds": failed_folds, "aborted": aborted,
    }

    if aborted or not seed_summaries:
        print(f"  [ABORTED] {name} did not complete all seeds/folds within the time budget.", flush=True)
        result["cross_seed"] = None
        result["cm_sum"] = None
        return result

    accs  = [s["oof_accuracy"] for s in seed_summaries]
    baccs = [s["oof_balanced_accuracy"] for s in seed_summaries]
    aucs  = [s["oof_roc_auc"] for s in seed_summaries]
    f1s   = [s["oof_f1"] for s in seed_summaries]
    senss = [s["oof_sensitivity"] for s in seed_summaries]
    specs = [s["oof_specificity"] for s in seed_summaries]
    cm_sum = np.sum([np.array(agg_cm_by_seed[s]) for s in SEEDS], axis=0)

    print(f"  CROSS-SEED: acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%  "
          f"bacc={np.mean(baccs)*100:.2f}%+/-{np.std(baccs)*100:.2f}%  "
          f"AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  elapsed={elapsed:.1f}s  "
          f"failed_folds={len(failed_folds)}", flush=True)

    hg.log_experiment(
        f"phase2h_{name}", sub_id.tolist(),
        notes=f"RIGOROUS protocol: 3x10 outer / 5 inner CV, full 19900+2 features, NO PCA, "
              f"model={name}, OOF acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%, "
              f"AUC={np.mean(aucs):.4f}, elapsed={elapsed:.1f}s, failed_folds={len(failed_folds)}",
    )

    result["cross_seed"] = {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(baccs)), "balanced_accuracy_std": float(np.std(baccs)),
        "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "sensitivity_mean": float(np.mean(senss)), "sensitivity_std": float(np.std(senss)),
        "specificity_mean": float(np.mean(specs)), "specificity_std": float(np.std(specs)),
    }
    result["cm_sum"] = cm_sum
    return result


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
    hg.assert_no_holdout_leakage(ph_all_sub[dev_mask].tolist(), context="phase2h initial dev filter")

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
    ap.add_argument("--gamma-auto", action="store_true",
                     help="Also test gamma='auto' for RBF SVM (doubles its grid).")
    ap.add_argument("--rbf-time-budget", type=float, default=None,
                     help="Max seconds to spend on RBF SVM before aborting safely (optional).")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    covs_dev, y, sub_id, sites, demo = load_dev_data()
    all_sites_dev = set(np.unique(sites))
    phase2c_perfold = pd.read_csv(PHASE2C_PERFOLD)

    print(f"Development cohort: {len(y)} subjects (ASD {y.sum()} | TC {(1-y).sum()})", flush=True)
    print(f"Thread env: OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}  "
          f"N_JOBS={N_JOBS}", flush=True)
    print(f"gamma_auto={args.gamma_auto}  rbf_time_budget={args.rbf_time_budget}", flush=True)

    models = make_models(args.gamma_auto)
    all_results = {}

    # LogReg first (known fast, establishes the reused-fold baseline for this exact run)
    all_results["LogReg_L2"] = run_model(
        "LogReg_L2", *models["LogReg_L2"], covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold
    )

    all_results["RBF_SVM"] = run_model(
        "RBF_SVM", *models["RBF_SVM"], covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold,
        time_budget_sec=args.rbf_time_budget,
    )

    all_perfold = []
    for name, r in all_results.items():
        all_perfold.extend(r["perfold_rows"])
    pd.DataFrame(all_perfold).to_csv(PERFOLD_CSV, index=False)

    summary_rows = []
    for name, r in all_results.items():
        for s in r["seed_summaries"]:
            summary_rows.append({"classifier": name, **s})
        if r["cross_seed"] is not None:
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
                "confusion_matrix": r["cm_sum"].tolist() if r["cm_sum"] is not None else "",
                "chosen_params_per_fold": "",
                "elapsed_sec": r["elapsed_sec"], "n_failed_folds": len(r["failed_folds"]),
                "aborted": r["aborted"],
            })
        else:
            summary_rows.append({
                "classifier": name, "seed": "ABORTED",
                "oof_accuracy": "", "oof_balanced_accuracy": "", "oof_roc_auc": "",
                "oof_f1": "", "oof_sensitivity": "", "oof_specificity": "",
                "tau_youden": "", "oof_precision": "",
                "fold_accuracy_mean": "", "fold_accuracy_std": "",
                "fold_balanced_accuracy_mean": "", "fold_balanced_accuracy_std": "",
                "confusion_matrix": "", "chosen_params_per_fold": "",
                "elapsed_sec": r["elapsed_sec"], "n_failed_folds": len(r["failed_folds"]),
                "aborted": r["aborted"],
            })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    reproducibility = {
        "phase": "2H",
        "purpose": "Rigorous RBF-SVM vs Logistic Regression L2 head-to-head, ORIGINAL protocol "
                   "(10 outer / 5 inner folds, 3 seeds, full 19,900+2 features, no PCA), reusing "
                   "the exact Phase 2C fold definitions (verified per-fold against "
                   "phase2c_baseline_perfold.csv).",
        "gamma_auto_tested": args.gamma_auto,
        "rbf_time_budget_sec": args.rbf_time_budget,
        "cv_protocol": {
            "outer": f"StratifiedKFold(n_splits={OUTER_SPLITS}, shuffle=True, random_state=seed)",
            "inner": f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True, random_state=seed)",
            "seeds": list(SEEDS),
            "threshold_selection": "Youden's J on seed-level OOF predictions",
        },
        "n_dev_subjects": 676,
        "n_holdout_subjects_locked_untouched": 170,
        "per_model": {
            name: {
                "cross_seed": r["cross_seed"], "elapsed_sec": r["elapsed_sec"],
                "failed_folds": r["failed_folds"], "aborted": r["aborted"],
                "confusion_matrix_summed": r["cm_sum"].tolist() if r["cm_sum"] is not None else None,
                "seed_summaries": r["seed_summaries"],
            }
            for name, r in all_results.items()
        },
        "holdout_status": "NEVER accessed. get_locked_holdout_ids() was never called from this script.",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(reproducibility, f, indent=2)

    print(f"\nSaved:\n  {PERFOLD_CSV}\n  {SUMMARY_CSV}\n  {SUMMARY_JSON}", flush=True)

    print("\n" + "=" * 74)
    print("  PHASE 2H FINAL COMPARISON")
    print("=" * 74)
    for name, r in all_results.items():
        if r["cross_seed"] is None:
            print(f"  {name}: ABORTED after {r['elapsed_sec']:.1f}s -- see report for details")
            continue
        cs = r["cross_seed"]
        print(f"  {name:<12} acc={cs['accuracy_mean']*100:6.2f}%+/-{cs['accuracy_std']*100:.2f}%  "
              f"bacc={cs['balanced_accuracy_mean']*100:6.2f}%+/-{cs['balanced_accuracy_std']*100:.2f}%  "
              f"AUC={cs['roc_auc_mean']:.4f}+/-{cs['roc_auc_std']:.4f}  "
              f"F1={cs['f1_mean']:.4f}  elapsed={r['elapsed_sec']:.1f}s  failed={len(r['failed_folds'])}")


if __name__ == "__main__":
    main()
