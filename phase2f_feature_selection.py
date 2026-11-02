"""
phase2f_feature_selection.py -- Phase 2F: feature selection / dimensionality
reduction comparison, on the SAME 30 (3 seeds x 10-fold) CV splits reused
from Phase 2C/2E. Representation fixed to correlation_fisherz (Phase 2E
winner-by-tie, cheapest of the statistically-equivalent representations).

Configs: all 19,900 features, and SelectKBest(f_classif, k) for
k in {100, 500, 1000, 2000, 5000}.

SelectKBest is fit ONLY on each outer fold's training partition (same
convention as StandardScaler in every prior phase script) -- never on the
full 676, never using outer-test-fold labels.

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
from sklearn.feature_selection import SelectKBest, f_classif
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

RESULTS_DIR     = r"C:\abide\results"
PERFOLD_CSV     = os.path.join(RESULTS_DIR, "phase2f_feature_selection_perfold.csv")
SUMMARY_CSV     = os.path.join(RESULTS_DIR, "phase2f_feature_selection_summary.csv")
SUMMARY_JSON    = os.path.join(RESULTS_DIR, "phase2f_feature_selection_summary.json")
PHASE2C_PERFOLD = os.path.join(RESULTS_DIR, "phase2c_baseline_perfold.csv")

OUTER_SPLITS = 10
INNER_SPLITS = 5
SEEDS        = (0, 1, 2)
C_GRID       = [0.001, 0.01, 0.1, 1.0, 10.0]
K_VALUES     = [None, 100, 500, 1000, 2000, 5000]   # None = all features


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])


def run_config(k, covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold):
    label = "all_19900" if k is None else f"SelectKBest_k{k}"
    print(f"\n{'=' * 74}\n  FEATURE CONFIG: {label}\n{'=' * 74}")
    t_start = time.time()

    perfold_rows = []
    seed_summaries = []
    agg_cm_by_seed = {}
    failed_folds = []

    for seed in SEEDS:
        outer = StratifiedKFold(OUTER_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=float)
        chosen_C = []
        n_selected_list = []

        for fold_idx, (tr, te) in enumerate(outer.split(covs_dev, y)):
            ref = phase2c_perfold[(phase2c_perfold.seed == seed) & (phase2c_perfold.fold == fold_idx)].iloc[0]
            assert len(tr) == ref.n_train and len(te) == ref.n_test
            assert int(y[tr].sum()) == ref.train_asd and int(y[te].sum()) == ref.test_asd

            hg.assert_cv_folds_dev_only(
                [sub_id[tr], sub_id[te]],
                context=f"phase2f[{label}] seed={seed} fold={fold_idx}",
            )

            try:
                # ── Embedding: correlation/Fisher-z, subject-independent ──
                emb = TangentEmbedding("correlation", precomputed=True)
                X_tr = emb.fit(covs_dev[tr]).transform(covs_dev[tr])
                X_te = emb.transform(covs_dev[te])

                # ── Feature selection: fit ONLY on this fold's training portion ──
                if k is not None:
                    skb = SelectKBest(score_func=f_classif, k=min(k, X_tr.shape[1]))
                    X_tr = skb.fit_transform(X_tr, y[tr])
                    X_te = skb.transform(X_te)
                n_selected = X_tr.shape[1]
                n_selected_list.append(n_selected)

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

            except Exception as exc:
                failed_folds.append({"config": label, "seed": seed, "fold": fold_idx,
                                      "error": f"{type(exc).__name__}: {exc}"})
                oof[te] = 0.5
                chosen_C.append(None)
                n_selected_list.append(-1)

            sites_in_test = set(np.unique(sites[te]))
            sites_missing = sorted(all_sites_dev - sites_in_test)
            perfold_rows.append({
                "phase": "2F", "representation": "correlation_fisherz",
                "feature_selection": "none" if k is None else "SelectKBest_f_classif",
                "k": "all" if k is None else k,
                "classifier": "LogReg_L2",
                "seed": seed, "outer_fold": fold_idx,
                "n_train": len(tr), "n_test": len(te),
                "train_ASD": int(y[tr].sum()), "train_TC": int((1 - y[tr]).sum()),
                "test_ASD": int(y[te].sum()), "test_TC": int((1 - y[te]).sum()),
                "n_sites_in_test": len(sites_in_test),
                "sites_missing_from_test": ";".join(sites_missing),
                "selected_features": n_selected_list[-1],
                "best_hyperparameters": f"C={chosen_C[-1]}",
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
                fold_aucs.append(float("nan"))  # single-class fold, rare
            row_idx = seed * OUTER_SPLITS + fi
            perfold_rows[row_idx]["accuracy"] = fold_accs[-1]
            perfold_rows[row_idx]["balanced_accuracy"] = fold_baccs[-1]
            perfold_rows[row_idx]["roc_auc"] = fold_aucs[-1]
            perfold_rows[row_idx]["f1"] = fold_f1s[-1]
            perfold_rows[row_idx]["sensitivity"] = fold_sens[-1]
            perfold_rows[row_idx]["specificity"] = fold_specs[-1]
            perfold_rows[row_idx]["runtime"] = None  # filled with per-config total below

        print(f"  seed={seed}  OOF acc={acc*100:.2f}%  bacc={bacc*100:.2f}%  "
              f"AUC={auc:.4f}  F1={f1:.4f}  n_features~{n_selected_list[0] if n_selected_list else 'NA'}")

        agg_cm_by_seed[seed] = cm.tolist()
        seed_summaries.append({
            "seed": seed, "tau_youden": tau,
            "oof_accuracy": acc, "oof_balanced_accuracy": bacc,
            "oof_roc_auc": auc, "oof_f1": f1, "oof_precision": prec,
            "oof_sensitivity": sens, "oof_specificity": spec,
            "fold_accuracy_mean": float(np.mean(fold_accs)), "fold_accuracy_std": float(np.std(fold_accs)),
            "fold_balanced_accuracy_mean": float(np.mean(fold_baccs)), "fold_balanced_accuracy_std": float(np.std(fold_baccs)),
            "confusion_matrix": cm.tolist(), "chosen_C_per_fold": chosen_C,
            "n_selected_features": n_selected_list,
        })

    elapsed = time.time() - t_start
    for row in perfold_rows:
        row["runtime"] = elapsed  # per-config total, same value on every row of this config

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
        f"phase2f_{label}", sub_id.tolist(),
        notes=f"3x10 nested CV, feature_selection={label}, "
              f"OOF acc={np.mean(accs)*100:.2f}%+/-{np.std(accs)*100:.2f}%, "
              f"AUC={np.mean(aucs):.4f}, elapsed={elapsed:.1f}s",
    )

    return {
        "label": label, "k": k,
        "perfold_rows": perfold_rows, "seed_summaries": seed_summaries,
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
    hg.assert_no_holdout_leakage(ph_all_sub[dev_mask].tolist(), context="phase2f initial dev filter")

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

    all_results = {}
    for k in K_VALUES:
        r = run_config(k, covs_dev, y, sub_id, sites, demo, all_sites_dev, phase2c_perfold)
        all_results[r["label"]] = r

    # ── Assemble outputs ──────────────────────────────────────────────────
    all_perfold = []
    for label, r in all_results.items():
        all_perfold.extend(r["perfold_rows"])
    pd.DataFrame(all_perfold).to_csv(PERFOLD_CSV, index=False)

    summary_rows = []
    for label, r in all_results.items():
        for s in r["seed_summaries"]:
            summary_rows.append({"feature_selection": label, "k": r["k"], **s})
        cs = r["cross_seed"]
        summary_rows.append({
            "feature_selection": label, "k": r["k"], "seed": "MEAN_SD",
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
            "n_selected_features": "", "elapsed_sec": r["elapsed_sec"],
        })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    # Determine winner by CV AUC primarily, balanced accuracy secondarily
    ranked = sorted(all_results.items(),
                     key=lambda kv: (kv[1]["cross_seed"]["roc_auc_mean"],
                                      kv[1]["cross_seed"]["balanced_accuracy_mean"]),
                     reverse=True)
    winner_label, winner = ranked[0]

    reproducibility = {
        "phase": "2F",
        "purpose": "Feature selection / dimensionality reduction comparison, correlation_fisherz "
                   "representation, identical Phase 2C/2E CV splits, dev-only (N=676).",
        "configs_tested": [r["label"] for r in all_results.values()],
        "selection_method": "SelectKBest(f_classif), fit on outer-fold TRAINING partition only",
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
        "ranking_criterion": "CV ROC-AUC primary, balanced accuracy secondary (not raw accuracy)",
        "winner": winner_label,
        "winner_cross_seed": winner["cross_seed"],
        "per_config": {
            label: {
                "k": r["k"], "cross_seed": r["cross_seed"], "elapsed_sec": r["elapsed_sec"],
                "failed_folds": r["failed_folds"],
                "confusion_matrix_summed": r["cm_sum"].tolist(),
                "seed_summaries": r["seed_summaries"],
            }
            for label, r in all_results.items()
        },
        "holdout_status": "NEVER accessed. get_locked_holdout_ids() was never called from this script.",
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(reproducibility, f, indent=2)

    print(f"\nSaved:\n  {PERFOLD_CSV}\n  {SUMMARY_CSV}\n  {SUMMARY_JSON}")
    print("\n" + "=" * 74)
    print("  FINAL COMPARISON TABLE (ranked by CV AUC)")
    print("=" * 74)
    print(f"  {'Config':<20} {'k':>8} {'Acc':>8} {'BAcc':>8} {'AUC':>8} {'F1':>7} {'Sec':>7}")
    for label, r in ranked:
        cs = r["cross_seed"]
        print(f"  {label:<20} {str(r['k']):>8} {cs['accuracy_mean']*100:6.2f}%  "
              f"{cs['balanced_accuracy_mean']*100:6.2f}%  {cs['roc_auc_mean']:.4f}  "
              f"{cs['f1_mean']:.4f}  {r['elapsed_sec']:6.1f}")
    print(f"\n  WINNER (by CV AUC, then balanced accuracy): {winner_label}")


if __name__ == "__main__":
    main()
