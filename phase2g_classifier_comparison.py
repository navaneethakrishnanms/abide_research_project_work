"""
phase2g_classifier_comparison.py -- Phase 2G: classifier comparison using the
Phase 2F winning feature configuration (all 19,900 correlation/Fisher-z
features + age+sex, i.e. no SelectKBest reduction beat the full feature set
on CV AUC/balanced accuracy), on the SAME 30 (3 seeds x 10-fold) CV splits
reused from Phase 2C/2E/2F.

Development subjects only (676). The 170 locked holdout subjects are never
accessed.
"""
from __future__ import annotations

import json
import os
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

RESULTS_DIR     = r"C:\abide\results"
PERFOLD_CSV     = os.path.join(RESULTS_DIR, "phase2g_classifier_perfold.csv")
SUMMARY_CSV     = os.path.join(RESULTS_DIR, "phase2g_classifier_summary.csv")
SUMMARY_JSON    = os.path.join(RESULTS_DIR, "phase2g_classifier_summary.json")
PHASE2C_PERFOLD = os.path.join(RESULTS_DIR, "phase2c_baseline_perfold.csv")

OUTER_SPLITS = 10
INNER_SPLITS = 5
SEEDS        = (0, 1, 2)
SEED_STATIC  = 42  # for classifier internal random_state where needed


def make_classifiers():
    clfs = {
        "LogReg_L2": (
            LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            None,
        ),
        "LogReg_L1": (
            LogisticRegression(max_iter=5000, solver="liblinear", penalty="l1",
                                class_weight="balanced"),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            None,
        ),
        "ElasticNet": (
            LogisticRegression(max_iter=3000, solver="saga", penalty="elasticnet",
                                class_weight="balanced"),
            {"C": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
            None,
        ),
        "LinearSVM": (
            CalibratedClassifierCV(
                LinearSVC(max_iter=5000, class_weight="balanced"), cv=3
            ),
            {"estimator__C": [0.001, 0.01, 0.1, 1.0]},
            None,
        ),
        "RBF_SVM_PCA200": (
            SVC(probability=True, class_weight="balanced"),
            {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.001, 0.01]},
            200,  # PCA components -- explicit computational-feasibility step
        ),
    }
    if XGB_AVAILABLE:
        clfs["XGBoost"] = (
            xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False,
                               scale_pos_weight=1.0, n_jobs=1),
            {"n_estimators": [200, 500], "max_depth": [3, 5],
             "learning_rate": [0.01, 0.05], "subsample": [0.8], "colsample_bytree": [0.8]},
            None,
        )
    return clfs


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


def score_clf(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def run_classifier(name, clf, grid, pca_components,
                    covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold):
    print(f"\n{'=' * 74}\n  CLASSIFIER: {name}\n{'=' * 74}")
    t_start = time.time()

    perfold_rows = []
    seed_summaries = []
    agg_cm_by_seed = {}
    failed_folds = []

    for seed in SEEDS:
        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)
        chosen_params = []

        for fold_idx, (tr, te) in enumerate(outer.split(covs_dev, y)):
            ref = phase2c_perfold[(phase2c_perfold.seed == seed) & (phase2c_perfold.fold == fold_idx)].iloc[0]
            assert len(tr) == ref.n_train and len(te) == ref.n_test
            assert int(y[tr].sum()) == ref.train_asd and int(y[te].sum()) == ref.test_asd

            hg.assert_cv_folds_dev_only(
                [sub_id[tr], sub_id[te]],
                context=f"phase2g[{name}] seed={seed} fold={fold_idx}",
            )

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

                if pca_components is not None:
                    n_comp = min(pca_components, X_tr.shape[0] - 1, X_tr.shape[1])
                    pca = PCA(n_components=n_comp, random_state=SEED_STATIC)
                    X_tr = pca.fit_transform(X_tr)
                    X_te = pca.transform(X_te)

                inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
                search = GridSearchCV(clf, grid, cv=inner, scoring="roc_auc", n_jobs=4, refit=True)
                search.fit(X_tr, y[tr])
                chosen_params.append(search.best_params_)

                scores_te = score_clf(search.best_estimator_, X_te)
                oof[te] = scores_te

            except Exception as exc:
                failed_folds.append({"classifier": name, "seed": seed, "fold": fold_idx,
                                      "error": f"{type(exc).__name__}: {exc}"})
                oof[te] = 0.5
                chosen_params.append(None)

            sites_in_test = set(np.unique(sites[te]))
            sites_missing = sorted(all_sites_dev - sites_in_test)
            perfold_rows.append({
                "phase": "2G", "representation": "correlation_fisherz",
                "feature_selection": "none", "k": "all",
                "classifier": name, "seed": seed, "outer_fold": fold_idx,
                "n_train": len(tr), "n_test": len(te),
                "train_ASD": int(y[tr].sum()), "train_TC": int((1 - y[tr]).sum()),
                "test_ASD": int(y[te].sum()), "test_TC": int((1 - y[te]).sum()),
                "n_sites_in_test": len(sites_in_test),
                "sites_missing_from_test": ";".join(sites_missing),
                "selected_features": pca_components if pca_components else "all",
                "best_hyperparameters": str(chosen_params[-1]),
            })

        if not any(np.isnan(v) if isinstance(v, float) else False for v in [1]):
            pass
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
              f"AUC={auc:.4f}  F1={f1:.4f}")

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

    if failed_folds:
        # If every fold failed, seed_summaries wasn't computed meaningfully -- mark unusable.
        n_total_folds = len(SEEDS) * OUTER_SPLITS
        if len(failed_folds) == n_total_folds:
            print(f"  [FAILURE] Classifier {name} FAILED on all {n_total_folds} folds -- excluded from ranking.")

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
          f"failed_folds={len(failed_folds)}")

    hg.log_experiment(
        f"phase2g_{name}", sub_id.tolist(),
        notes=f"3x10 nested CV, classifier={name}, "
              f"OOF acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%, "
              f"AUC={np.mean(aucs):.4f}, elapsed={elapsed:.1f}s, "
              f"failed_folds={len(failed_folds)}",
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
    hg.assert_no_holdout_leakage(ph_all_sub[dev_mask].tolist(), context="phase2g initial dev filter")

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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    covs_dev, y, sub_id, sites, demo = load_dev_data()
    all_sites_dev = set(np.unique(sites))
    phase2c_perfold = pd.read_csv(PHASE2C_PERFOLD)

    print(f"Development cohort: {len(y)} subjects (ASD {y.sum()} | TC {(1-y).sum()})")
    print(f"XGBoost available: {XGB_AVAILABLE}")

    classifiers = make_classifiers()
    all_results = {}
    for name, (clf, grid, pca_c) in classifiers.items():
        all_results[name] = run_classifier(
            name, clf, grid, pca_c, covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold
        )

    if not XGB_AVAILABLE:
        print("\n[NOTE] xgboost not importable in this environment -- XGBoost skipped, "
              "recorded as unavailable rather than fabricated.")

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

    usable = {k: v for k, v in all_results.items() if len(v["failed_folds"]) < len(SEEDS) * OUTER_SPLITS}
    ranked = sorted(usable.items(),
                     key=lambda kv: (kv[1]["cross_seed"]["roc_auc_mean"],
                                      kv[1]["cross_seed"]["balanced_accuracy_mean"]),
                     reverse=True)
    winner_name, winner = ranked[0] if ranked else (None, None)

    reproducibility = {
        "phase": "2G",
        "purpose": "Classifier comparison on the Phase 2F winning feature configuration "
                   "(correlation_fisherz, all 19,900 features + age+sex), identical Phase "
                   "2C/2E/2F CV splits, dev-only (N=676).",
        "feature_config_used": "correlation_fisherz, all 19900 FC features + age + sex "
                               "(Phase 2F winner: no SelectKBest reduction beat full features)",
        "classifiers_tested": list(classifiers.keys()),
        "xgboost_available": XGB_AVAILABLE,
        "cv_protocol": {
            "outer": f"StratifiedKFold(n_splits={OUTER_SPLITS}, shuffle=True, random_state=seed)",
            "inner": f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True, random_state=seed)",
            "seeds": list(SEEDS),
            "threshold_selection": "Youden's J on seed-level OOF predictions",
        },
        "n_dev_subjects": 676,
        "n_holdout_subjects_locked_untouched": 170,
        "ranking_criterion": "CV ROC-AUC primary, balanced accuracy secondary (not raw accuracy)",
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
    print("  FINAL COMPARISON TABLE (ranked by CV AUC)")
    print("=" * 74)
    print(f"  {'Classifier':<18} {'Acc':>8} {'BAcc':>8} {'AUC':>8} {'F1':>7} {'Sec':>7} {'Failed':>7}")
    for name, r in ranked:
        cs = r["cross_seed"]
        print(f"  {name:<18} {cs['accuracy_mean']*100:6.2f}%  {cs['balanced_accuracy_mean']*100:6.2f}%  "
              f"{cs['roc_auc_mean']:.4f}  {cs['f1_mean']:.4f}  {r['elapsed_sec']:6.1f}  {len(r['failed_folds']):6d}")
    for name, r in all_results.items():
        if name not in usable:
            print(f"  {name:<18} FAILED on all folds -- excluded from ranking")
    if winner_name:
        print(f"\n  WINNER (by CV AUC, then balanced accuracy): {winner_name}")


if __name__ == "__main__":
    main()
