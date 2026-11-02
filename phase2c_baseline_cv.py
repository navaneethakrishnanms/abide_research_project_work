"""
phase2c_baseline_cv.py -- Phase 2C: Development-only nested CV baseline.

Reproduces the strongest verified FC pipeline (correlation/Fisher-z FC +
age+sex demographics + StandardScaler + Logistic Regression L2 -- the exact
configuration behind the 74.12%/0.8104 result in fc_v2_results.csv, EXP-2A)
on ONLY the 676 development subjects from the new Phase 2 split.

The 170 locked final-holdout subjects (splits/phase2_holdout_subjects.txt)
are never loaded, inspected, or referenced beyond the safety checks below.

Every subject array that enters training/validation/tuning is explicitly
checked against the locked holdout via holdout_guard before use.
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
from tangent import TangentEmbedding

warnings.filterwarnings("ignore")

DATA        = r"C:\abide\abide_pcp"
TS_PATH     = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH   = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH  = os.path.join(DATA, "phenotypic_qc.csv")
MIN_TIMEPOINTS = 100

RESULTS_DIR   = r"C:\abide\results"
PERFOLD_CSV   = os.path.join(RESULTS_DIR, "phase2c_baseline_perfold.csv")
SUMMARY_CSV   = os.path.join(RESULTS_DIR, "phase2c_baseline_results.csv")
SUMMARY_JSON  = os.path.join(RESULTS_DIR, "phase2c_baseline_summary.json")

OUTER_SPLITS = 10
INNER_SPLITS = 5
SEEDS        = (0, 1, 2)
C_GRID       = [0.001, 0.01, 0.1, 1.0, 10.0]


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load full cohort, apply identical filter to the rest of the pipeline ──
    ts = np.load(TS_PATH, allow_pickle=True)
    ph = pd.read_csv(PHENO_PATH)
    keep = np.array([t is not None and t.shape[0] >= MIN_TIMEPOINTS for t in ts])
    ph = ph.loc[keep].reset_index(drop=True)
    # covs_cc200.npy was saved already aligned to the post-filter 846-subject
    # cohort (same convention as train_fc_v2.py / fast_holdout_eval.py) --
    # it must NOT be re-indexed with `keep` again.
    covs_all = np.load(COVS_PATH)
    assert covs_all.shape[0] == len(ph), (
        f"covs_cc200.npy has {covs_all.shape[0]} rows but filtered phenotypic "
        f"has {len(ph)} rows -- alignment assumption violated, aborting."
    )

    # ── 2. Restrict to development subjects ONLY, via the guard ──────────────
    dev_ids = hg.load_dev_subject_ids()          # verifies integrity, returns 676 IDs
    ph_all_sub = ph["SUB_ID"].astype(str)
    dev_mask = ph_all_sub.isin(dev_ids).values
    hg.assert_no_holdout_leakage(ph_all_sub[dev_mask].tolist(),
                                  context="phase2c initial dev filter")

    ph_dev   = ph.loc[dev_mask].reset_index(drop=True)
    covs_dev = covs_all[dev_mask]
    assert len(ph_dev) == 676, f"expected 676 dev subjects, got {len(ph_dev)}"

    sub_id = ph_dev["SUB_ID"].astype(str).values
    y      = (ph_dev["DX_GROUP"].values == 1).astype(int)
    sites  = ph_dev["SITE_ID"].astype(str).values
    num    = lambda c: pd.to_numeric(ph_dev.get(c), errors="coerce").replace(-9999, np.nan).values
    age    = num("AGE_AT_SCAN").astype(float)
    sex    = num("SEX").astype(float)
    demo   = np.column_stack([age, sex])

    n_features_raw = covs_dev.shape[1] * (covs_dev.shape[1] - 1) // 2
    print(f"Development cohort: {len(ph_dev)} subjects "
          f"(ASD {y.sum()} | TC {(1 - y).sum()}) | sites={pd.unique(sites).size}")
    print(f"Covariance shape: {covs_dev.shape} -> upper-triangle features: {n_features_raw}")
    print("Holdout guard: dev-only filter verified clean (0 holdout subjects present).\n")

    all_sites_dev = set(np.unique(sites))

    perfold_rows = []
    seed_summaries = []
    agg_cm_by_seed = {}

    for seed in SEEDS:
        print(f"{'=' * 74}\n  SEED {seed}\n{'=' * 74}")
        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)
        chosen_C = []

        for fold_idx, (tr, te) in enumerate(outer.split(covs_dev, y)):
            # ── Safety: check both partitions of this fold against the locked holdout ──
            hg.assert_cv_folds_dev_only(
                [sub_id[tr], sub_id[te]],
                context=f"phase2c seed={seed} fold={fold_idx}",
            )

            # ── Embedding: fit tangent/correlation reference on TRAIN portion only ──
            tang = TangentEmbedding("correlation", precomputed=True)
            X_tr = tang.fit_transform(covs_dev[tr])
            X_te = tang.transform(covs_dev[te])

            # ── Demographics: median imputation from TRAIN portion only ──────
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

            # ── Feature scaling: fit on TRAIN portion only ────────────────────
            fs = StandardScaler()
            X_tr = fs.fit_transform(X_tr)
            X_te = fs.transform(X_te)

            # ── Inner CV: hyperparameter selection on TRAIN portion only ─────
            clf = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
            inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
            search = GridSearchCV(clf, {"C": C_GRID}, cv=inner, scoring="roc_auc", n_jobs=4, refit=True)
            search.fit(X_tr, y[tr])
            best_C = search.best_params_["C"]
            chosen_C.append(best_C)

            scores_te = search.best_estimator_.predict_proba(X_te)[:, 1]
            oof[te] = scores_te

            # ── Per-fold descriptive stats ─────────────────────────────────
            sites_in_test = set(np.unique(sites[te]))
            sites_missing_from_test = sorted(all_sites_dev - sites_in_test)
            perfold_rows.append({
                "seed": seed, "fold": fold_idx,
                "n_train": len(tr), "n_test": len(te),
                "train_asd": int(y[tr].sum()), "train_tc": int((1 - y[tr]).sum()),
                "test_asd": int(y[te].sum()), "test_tc": int((1 - y[te]).sum()),
                "n_sites_in_test": len(sites_in_test),
                "sites_missing_from_test": ";".join(sites_missing_from_test),
                "best_C": best_C,
                "feature_dim_final": X_tr.shape[1],
            })

        # ── Seed-level OOF metrics ─────────────────────────────────────────
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

        # Per-fold accuracy/bacc AT THE SEED-LEVEL OOF THRESHOLD (not re-tuned per fold)
        outer2 = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        fold_accs, fold_baccs = [], []
        for fi, (tr, te) in enumerate(outer2.split(covs_dev, y)):
            yp_fold = (oof[te] >= tau).astype(int)
            fold_accs.append(accuracy_score(y[te], yp_fold))
            fold_baccs.append(balanced_accuracy_score(y[te], yp_fold))
            perfold_rows[seed * OUTER_SPLITS + fi]["fold_accuracy"] = fold_accs[-1]
            perfold_rows[seed * OUTER_SPLITS + fi]["fold_balanced_accuracy"] = fold_baccs[-1]

        print(f"  Chosen C per fold: {chosen_C}")
        print(f"  OOF tau (Youden)  : {tau:.4f}")
        print(f"  OOF accuracy      : {acc*100:.2f}%   balanced acc: {bacc*100:.2f}%")
        print(f"  OOF ROC-AUC       : {auc:.4f}   F1: {f1:.4f}")
        print(f"  sensitivity/spec  : {sens*100:.2f}% / {spec*100:.2f}%")
        print(f"  per-fold accuracy : mean={np.mean(fold_accs)*100:.2f}% "
              f"std={np.std(fold_accs)*100:.2f}%")
        print(f"  confusion matrix  :\n{cm}\n")

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

    # ── Cross-seed aggregation ───────────────────────────────────────────
    accs  = [s["oof_accuracy"] for s in seed_summaries]
    baccs = [s["oof_balanced_accuracy"] for s in seed_summaries]
    aucs  = [s["oof_roc_auc"] for s in seed_summaries]
    f1s   = [s["oof_f1"] for s in seed_summaries]
    senss = [s["oof_sensitivity"] for s in seed_summaries]
    specs = [s["oof_specificity"] for s in seed_summaries]

    cm_sum = np.sum([np.array(agg_cm_by_seed[s]) for s in SEEDS], axis=0)

    print("=" * 74)
    print("  CROSS-SEED SUMMARY (3 seeds x 10-fold nested CV, N=676 dev subjects)")
    print("=" * 74)
    print(f"  Accuracy         : {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    print(f"  Balanced accuracy: {np.mean(baccs)*100:.2f}% +/- {np.std(baccs)*100:.2f}%")
    print(f"  ROC-AUC          : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  F1               : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Sensitivity      : {np.mean(senss)*100:.2f}% +/- {np.std(senss)*100:.2f}%")
    print(f"  Specificity      : {np.mean(specs)*100:.2f}% +/- {np.std(specs)*100:.2f}%")
    print(f"  Aggregated confusion matrix (summed over 3 seeds):\n{cm_sum}")

    # ── Save outputs ──────────────────────────────────────────────────────
    pd.DataFrame(perfold_rows).to_csv(PERFOLD_CSV, index=False)

    summary_rows = seed_summaries + [{
        "seed": "MEAN_SD",
        "oof_accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
        "oof_balanced_accuracy": f"{np.mean(baccs):.4f} +/- {np.std(baccs):.4f}",
        "oof_roc_auc": f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}",
        "oof_f1": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
        "oof_sensitivity": f"{np.mean(senss):.4f} +/- {np.std(senss):.4f}",
        "oof_specificity": f"{np.mean(specs):.4f} +/- {np.std(specs):.4f}",
        "tau_youden": "", "oof_precision": "",
        "fold_accuracy_mean": "", "fold_accuracy_std": "",
        "fold_balanced_accuracy_mean": "", "fold_balanced_accuracy_std": "",
        "confusion_matrix": cm_sum.tolist(), "chosen_C_per_fold": "",
    }]
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    reproducibility = {
        "phase": "2C",
        "purpose": "Development-only nested CV baseline, reproducing the FC(correlation)"
                   "+age+sex+LogReg-L2 config behind the original 74.12%/0.8104 result, "
                   "on the NEW dev-only 676-subject split (holdout excluded throughout).",
        "n_dev_subjects": int(len(ph_dev)),
        "n_holdout_subjects_locked_untouched": 170,
        "n_features_raw_upper_triangle": int(n_features_raw),
        "n_features_final_with_demo": int(perfold_rows[0]["feature_dim_final"]),
        "atlas": "CC200 (200 ROIs)",
        "embedding": "TangentEmbedding(kind='correlation', precomputed=True) -- Ledoit-Wolf "
                     "covariance -> Fisher-z correlation -> upper-triangle vectorization",
        "demographics": ["AGE_AT_SCAN", "SEX"],
        "demographic_imputation": "np.nanmedian on training fold only",
        "scaler": "StandardScaler, fit on training fold only",
        "classifier": "LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced')",
        "hyperparameter_grid": {"C": C_GRID},
        "cv_protocol": {
            "outer": f"StratifiedKFold(n_splits={OUTER_SPLITS}, shuffle=True, random_state=seed)",
            "inner": f"StratifiedKFold(n_splits={INNER_SPLITS}, shuffle=True, random_state=seed)",
            "seeds": list(SEEDS),
            "hyperparameter_selection": "GridSearchCV, scoring='roc_auc', inner CV only",
            "threshold_selection": "Youden's J on outer-loop OOF predictions (seed-level), "
                                    "never on any test fold in isolation, never on holdout",
        },
        "cross_seed_summary": {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "balanced_accuracy_mean": float(np.mean(baccs)), "balanced_accuracy_std": float(np.std(baccs)),
            "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "sensitivity_mean": float(np.mean(senss)), "sensitivity_std": float(np.std(senss)),
            "specificity_mean": float(np.mean(specs)), "specificity_std": float(np.std(specs)),
            "aggregated_confusion_matrix": cm_sum.tolist(),
        },
        "per_seed": seed_summaries,
        "holdout_status": "NEVER accessed. Only holdout_guard.load_dev_subject_ids() and "
                          "assert_no_holdout_leakage()/assert_cv_folds_dev_only() were called; "
                          "get_locked_holdout_ids() was never called from this script.",
        "comparison_note": "Compared against fc_v2_results.csv EXP-2A (74.12% acc, 0.8104 AUC), "
                           "which was measured on the OLD, reused 170-subject holdout (seed=42) "
                           "-- NOT this run's dev-only CV number. Both numbers are reported "
                           "side by side in the Phase 2C report but are not the same measurement.",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(reproducibility, f, indent=2)

    hg.log_experiment(
        "phase2c_baseline_nested_cv",
        sub_id.tolist(),
        notes=f"3 seeds x 10-fold nested CV, FC(correlation)+age+sex+LogReg-L2, "
              f"OOF acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%",
    )

    print(f"\nSaved:\n  {PERFOLD_CSV}\n  {SUMMARY_CSV}\n  {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
