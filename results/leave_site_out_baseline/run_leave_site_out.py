"""
Leave-one-site-out evaluation of the FROZEN AAL-116 L2 logistic-regression baseline (600 development subjects).

Frozen baseline = the exact pipeline/procedure of the fast experiment: MeanImputeFisherZ -> StandardScaler -> SelectKBest ->
L2 logistic regression (balanced), C in {1e-4,1e-3,1e-2,1e-1} x k in {500, 2000, all} chosen by inner StratifiedKFold(3, seed 42)
ROC-AUC on the TRAINING rows (unchanged code path: fx.pipe + fx.MODELS['LR_L2']). For each held-out site the whole procedure is
refit on the other sites only; nothing is tuned on the held-out site. Holdout, restricted ID file and CC200 are never touched.
Subject-level predictions are saved with anonymized project IDs (DEV0000..DEV0599 = row order of the development manifest).
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"C:\abide")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
R = ROOT / "results"
BASE = R / "fast_accuracy_experiment"
OUT = R / "leave_site_out_baseline"
for s in ("checkpoints", "logs"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "leave_site_out.log"
SMALL_N = 20
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def metrics(y, pred, score):
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan
    return {"accuracy": (tn + tp) / len(y), "balanced_accuracy": np.nanmean([sens, spec]) if not (np.isnan(sens) or np.isnan(spec)) else np.nan,
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1, "roc_auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def sha_all():
    dirs = ["phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification", "phase3_6_aal_download", "phase4", "phase4_1_constant_roi_audit",
            "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic", "phase5b_missing_fc_analysis", "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero",
            "phase6_strategyC_all_nonzero", "phase7_synthesis", "phase8_final_audit", "phase9_accuracy_optimization", "phase10_advanced_accuracy"]
    files = []
    for d in dirs:
        if (R / d).exists():
            files += [f for f in (R / d).rglob("*") if f.is_file()]
    files += [f for f in ROOT.glob("phase*.py") if not f.name.startswith("phase10")] + [ROOT / "phase10_advanced_accuracy.py"]
    return {str(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(set(files))}


def main():
    t0 = time.time()
    log("=" * 60 + "\nLEAVE-ONE-SITE-OUT (frozen AAL-116 L2 logistic regression) start")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    drift = [k for k, v in sha_all().items() if k in pre and pre[k]["sha256"] != v]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}")
    assert not drift
    dev, _ = fx.build_dev_set()
    y = dev["y"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all() and (man["site"].values == dev["site"].values).all()
    site = dev["site"].values
    sites = sorted(pd.unique(site), key=lambda s: -(site == s).sum())
    assert len(sites) == 19 and all(len(np.unique(y[site == s])) >= 1 for s in sites)
    X = fx.load_fc(dev, "AAL116")
    log(f"Development cohort {len(dev)} subjects ({int(y.sum())} ASD / {int((1 - y).sum())} TC), {len(sites)} sites; AAL-116 matrix {X.shape}")
    # frozen five-fold reference (saved)
    f5 = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str})
    f5 = f5[(f5.atlas == "AAL116") & (f5.model == "LR_L2")].set_index("subject_id").loc[dev["sid"]]
    assert (f5.y_true.values == y).all()
    # correctness check: held-out-site rows must not influence the fit or each other
    make, grid = fx.MODELS["LR_L2"]
    s0 = sites[0]; te0 = np.where(site == s0)[0]; tr0 = np.where(site != s0)[0]
    p = fx.pipe(make(), None).set_params(sel__k=500, clf__C=1e-3).fit(X[tr0], y[tr0])
    base = p.predict_proba(X[te0])[:, 1]
    Xj = X[te0].copy(); half = np.arange(len(te0)) % 2 == 1
    Xj[half] = np.random.default_rng(0).uniform(-0.9, 0.9, Xj[half].shape)
    ok = bool(np.allclose(base[~half], p.predict_proba(Xj)[:, 1][~half], atol=1e-12))
    log(f"Check: held-out-site rows do not influence one another or the fit: {'PASS' if ok else 'FAIL'}")
    assert ok
    rows, oof = [], []
    covered = np.zeros(len(y), int)
    for s in sites:
        path = OUT / "checkpoints" / f"site_{s}.json"
        ck = fx.load_ck(path)
        te = np.where(site == s)[0]; tr = np.where(site != s)[0]
        assert not (set(site[tr]) & set(site[te])) and len(te) + len(tr) == len(y)
        if ck is None:
            gs = GridSearchCV(fx.pipe(make(), None), grid, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=fx.SEED), n_jobs=6, refit=True, error_score="raise")
            t1 = time.time(); gs.fit(X[tr], y[tr])
            est = gs.best_estimator_
            sc = est.predict_proba(X[te])[:, 1]; pr = est.predict(X[te])
            ck = {"site": s, "n_train": int(len(tr)), "n_test": int(len(te)), "train_asd": int(y[tr].sum()), "train_tc": int((1 - y[tr]).sum()), "test_asd": int(y[te].sum()), "test_tc": int((1 - y[te]).sum()),
                  "n_train_sites": int(len(set(site[tr]))), "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in gs.best_params_.items()},
                  "inner_cv_auc": float(gs.best_score_), "metrics": metrics(y[te], pr, sc), "seconds": round(time.time() - t1, 1),
                  "test_idx": [int(i) for i in te], "score": [float(v) for v in sc], "pred": [int(v) for v in pr]}
            fx.save_ck(path, ck)
        m = ck["metrics"]
        log(f"[{s}] test n={ck['n_test']} ({ck['test_asd']} ASD/{ck['test_tc']} TC): acc={m['accuracy']:.3f} auc={m['roc_auc'] if not np.isnan(m['roc_auc']) else float('nan'):.3f} params={ck['best_params']} ({ck['seconds']}s)")
        for i in ck["test_idx"]:
            covered[i] += 1
        lo, hi = wilson(m["tn"] + m["tp"], ck["n_test"])
        rows.append({"site": s, "n_test": ck["n_test"], "test_ASD": ck["test_asd"], "test_TC": ck["test_tc"], "n_train": ck["n_train"], "train_ASD": ck["train_asd"], "train_TC": ck["train_tc"],
                     "n_train_sites": ck["n_train_sites"], "small_site_n_lt_20": ck["n_test"] < SMALL_N, **{k: m[k] for k in MET}, "roc_auc": m["roc_auc"],
                     "auc_note": "" if not np.isnan(m["roc_auc"]) else "AUC undefined (single class in test site)", "acc_ci95_lo": lo, "acc_ci95_hi": hi,
                     "selected_C": ck["best_params"]["clf__C"], "selected_k": ck["best_params"]["sel__k"], "inner_cv_auc": ck["inner_cv_auc"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"]})
        for i, sc_, p_ in zip(ck["test_idx"], ck["score"], ck["pred"]):
            oof.append({"project_id": f"DEV{i:04d}", "site": s, "y_true": int(y[i]), "score": sc_, "y_pred": p_})
    assert (covered == 1).all(), "every subject must be tested exactly once"
    report(dev, y, site, sites, pd.DataFrame(rows), pd.DataFrame(oof), f5, time.time() - t0, pre)


def report(dev, y, site, sites, ps, oof, f5, total_s, pre):
    ps.to_csv(OUT / "per_site_results.csv", index=False)
    oof = oof.sort_values("project_id").reset_index(drop=True)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    yt, sc, pr = oof.y_true.values, oof.score.values, oof.y_pred.values
    pooled = metrics(yt, pr, sc)
    big = ps[(~ps.small_site_n_lt_20) & ps.roc_auc.notna()]
    agg = [{"aggregate": "POOLED subject-level (all 600 out-of-site predictions)", "n_subjects": 600, **{k: pooled[k] for k in MET}, "roc_auc": pooled["roc_auc"],
            "note": "accuracy etc. are over all subjects; pooled AUC mixes scores from 19 different models (different scales), interpret with care"},
           {"aggregate": "MACRO mean over all 19 sites (each site weighted equally)", "n_subjects": 600, **{k: float(ps[k].mean()) for k in MET}, "roc_auc": float(ps.roc_auc.mean()),
            "note": "not subject-level accuracy; small sites weigh as much as large ones"},
           {"aggregate": f"MACRO mean over the {len(big)} sites with n>=20 and both classes", "n_subjects": int(big.n_test.sum()), **{k: float(big[k].mean()) for k in MET}, "roc_auc": float(big.roc_auc.mean()),
            "note": "not subject-level accuracy"},
           {"aggregate": "Site-level spread (SD across all 19 sites)", "n_subjects": 600, **{k: float(ps[k].std(ddof=1)) for k in MET}, "roc_auc": float(ps.roc_auc.std(ddof=1)), "note": "descriptive"}]
    lo, hi = wilson(pooled["tn"] + pooled["tp"], 600)
    agg[0]["accuracy_ci95_lo"], agg[0]["accuracy_ci95_hi"] = lo, hi
    pd.DataFrame(agg).to_csv(OUT / "aggregate_metrics.csv", index=False)
    # comparison with the saved five-fold random CV (same subjects, same pipeline)
    f5p = f5.reset_index().rename(columns={"subject_id": "sid"})
    f5p["project_id"] = [f"DEV{i:04d}" for i in range(600)]
    f5m = metrics(f5p.y_true.values, f5p.y_pred.values, f5p.score.values)
    both = oof.merge(f5p[["project_id", "y_pred", "score"]].rename(columns={"y_pred": "pred_5fold", "score": "score_5fold"}), on="project_id")
    ok_lso = (both.y_pred == both.y_true).values; ok_5 = (both.pred_5fold == both.y_true).values
    b = int((ok_5 & ~ok_lso).sum()); c = int((~ok_5 & ok_lso).sum())
    mcn = stats.binomtest(b, b + c, 0.5).pvalue if b + c else np.nan
    cmp_rows = [{"scope": "all 600 subjects (pooled)", "n": 600, "acc_5fold_random_cv": f5m["accuracy"], "acc_leave_site_out": pooled["accuracy"], "acc_diff": pooled["accuracy"] - f5m["accuracy"],
                 "balanced_acc_5fold": f5m["balanced_accuracy"], "balanced_acc_lso": pooled["balanced_accuracy"], "sens_5fold": f5m["sensitivity"], "sens_lso": pooled["sensitivity"],
                 "spec_5fold": f5m["specificity"], "spec_lso": pooled["specificity"], "auc_5fold_pooled": f5m["roc_auc"], "auc_lso_pooled": pooled["roc_auc"],
                 "subjects_correct_only_in_5fold": b, "subjects_correct_only_in_lso": c, "mcnemar_exact_p_descriptive": mcn}]
    for _, r in ps.iterrows():
        m = (both.site == r.site)
        cmp_rows.append({"scope": f"site {r.site}", "n": int(r.n_test), "acc_5fold_random_cv": float(ok_5[m].mean()), "acc_leave_site_out": float(ok_lso[m].mean()), "acc_diff": float(ok_lso[m].mean() - ok_5[m].mean()),
                         "small_site_n_lt_20": bool(r.small_site_n_lt_20)})
    cmpdf = pd.DataFrame(cmp_rows); cmpdf.to_csv(OUT / "comparison_with_five_fold_cv.csv", index=False)
    sd = cmpdf.iloc[1:]
    nbig = sd[~sd.small_site_n_lt_20.astype(bool)]
    summary = {"pooled": pooled, "five_fold_pooled": f5m, "acc_diff": pooled["accuracy"] - f5m["accuracy"], "auc_diff_pooled": pooled["roc_auc"] - f5m["roc_auc"], "mcnemar_b_c_p": [b, c, mcn],
               "sites_lso_lower_than_5fold": int((sd.acc_diff < 0).sum()), "sites_total": len(sd), "sites_n_ge_20_lower": int((nbig.acc_diff < 0).sum()), "sites_n_ge_20": len(nbig),
               "acc_ci95": [lo, hi], "macro_acc_all": float(ps.accuracy.mean()), "macro_auc_ge20": float(big.roc_auc.mean()), "runtime_s": round(total_s, 1),
               "sites_below_chance_ge20": ps[(~ps.small_site_n_lt_20) & (ps.accuracy < 0.5)].site.tolist(), "n_aucs_defined": int(ps.roc_auc.notna().sum())}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    now = sha_all(); drift = [k for k in pre if k in now and pre[k]["sha256"] != now[k]]
    json.dump({"protocol": "leave-one-site-out, 19 sites, frozen AAL-116 L2 logistic regression (fast experiment pipeline + grid, inner StratifiedKFold(3, seed 42) on training sites only)", "cohort": "600 development subjects",
               "anonymization": "project_id = DEV{row index of dev_split_manifest}; no original subject IDs written", "threads": "OMP/MKL/OPENBLAS=1", "runtime_s": round(total_s, 1), "protected_files_changed": len(drift)}, open(OUT / "config.json", "w"), indent=1)
    log(f"Outputs written; protected files changed={len(drift)}; runtime {total_s:.0f}s")
    log("\n" + ps[["site", "n_test", "test_ASD", "test_TC", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "small_site_n_lt_20"]].round(3).to_string(index=False))
    log("\n" + pd.DataFrame(agg)[["aggregate", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc"]].round(4).to_string(index=False))
    log("\n" + cmpdf.iloc[[0]].round(4).T.to_string(header=False))
    log(json.dumps({k: summary[k] for k in ("acc_diff", "auc_diff_pooled", "mcnemar_b_c_p", "sites_lso_lower_than_5fold", "sites_n_ge_20_lower", "sites_n_ge_20", "sites_below_chance_ge20", "n_aucs_defined")}, default=float))


if __name__ == "__main__":
    main()
