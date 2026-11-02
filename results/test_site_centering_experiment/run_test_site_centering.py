"""
Test-site feature centering for leave-one-site-out evaluation (single exploratory experiment).

PROTOCOL (fixed before any score was seen)
  Z = arctanh(clip(r, +/-(1-1e-6)))            element-wise, parameter-free; undefined correlations stay NaN
  Zc[i] = Z[i] - nanmean(Z[rows of the SAME site as i], axis=0)       every site (training or held-out) is centered as one batch by its own
                                                                       mean; no labels, predictions or other sites' rows are used.
  Then the unchanged baseline: SimpleImputer(mean, fitted on TRAINING rows) -> StandardScaler -> SelectKBest -> L2 LogisticRegression (balanced),
  grid C {1e-4..1e-1} x k {500, 2000, all}, inner StratifiedKFold(3, seed 42) ROC-AUC on the training sites only.
  Transductive: a held-out subject's centered features depend on all other subjects of its site (and itself); needs the whole site as a batch.
  No minimum batch size is enforced (uniform rule); sites with n < 20 are flagged; CMU (n = 2) is degenerate.
ARMS: CENTERED; UNCENTERED_SAME_PIPELINE (same code without centering, to isolate the centering effect); FROZEN (saved leave_site_out_baseline predictions).
PREDECLARED RULE (vs FROZEN, pooled over 600 out-of-site predictions): accuracy +0.02, balanced accuracy +0.02, pooled AUC not lower,
  accuracy higher at a majority (>= 7) of the 13 sites with n >= 20. Control (not part of the rule): five-fold CV with the same centering.
Holdout, restricted ID file and CC200 are never touched.
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
R = ROOT / "results"
BASE = R / "fast_accuracy_experiment"
OUT = R / "test_site_centering_experiment"
for s in ("checkpoints", "logs"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "test_site_centering.log"
GRID = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1], "sel__k": [500, 2000, "all"]}
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
BUDGET_S = 900


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def to_z(X):
    return np.arctanh(np.clip(X, -1 + 1e-6, 1 - 1e-6))


def center_by_site(Z, site):
    """Each site is one batch: subtract that site's own nan-mean (features undefined for a whole site get mean 0)."""
    out = Z.copy()
    for s in np.unique(site):
        m = site == s
        with np.errstate(all="ignore"):
            mu = np.nanmean(Z[m], axis=0)
        out[m] = Z[m] - np.where(np.isnan(mu), 0.0, mu)
    return out


def pipe():
    return Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])


def fit_predict(Xtr, ytr, Xte):
    gs = GridSearchCV(pipe(), GRID, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=fx.SEED), n_jobs=6, refit=True, error_score="raise")
    gs.fit(Xtr, ytr)
    return [float(v) for v in gs.best_estimator_.predict_proba(Xte)[:, 1]], {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in gs.best_params_.items()}


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
    log("=" * 60 + "\nTEST-SITE CENTERING (leave-one-site-out) start")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    drift = [k for k, v in sha_all().items() if k in pre and pre[k]["sha256"] != v]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}"); assert not drift
    dev, _ = fx.build_dev_set(); y = dev["y"].values; site = dev["site"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all() and (man["site"].values == site).all()
    X = fx.load_fc(dev, "AAL116"); Z = to_z(X); Zc = center_by_site(Z, site)
    log(f"Development cohort {len(dev)} ({int(y.sum())} ASD / {int((1 - y).sum())} TC), {len(set(site))} sites; features {X.shape}; undefined entries {int(np.isnan(X).sum())} (kept as NaN through centering)")
    # ---- implementation checks of the centering transform
    rng = np.random.default_rng(0)
    s_te = "NYU"; s_other = "UM_1"
    Zp = Z.copy(); Zp[site == s_other] = rng.normal(size=Zp[site == s_other].shape)                 # perturb ANOTHER site
    iso1 = np.array_equal(np.nan_to_num(center_by_site(Zp, site)[site == s_te]), np.nan_to_num(Zc[site == s_te]))
    Zp2 = Z.copy(); Zp2[site == s_te] = rng.normal(size=Zp2[site == s_te].shape)                      # perturb the held-out site
    iso2 = np.array_equal(np.nan_to_num(center_by_site(Zp2, site)[site != s_te]), np.nan_to_num(Zc[site != s_te]))
    mean_zero = float(np.nanmax(np.abs([np.nanmean(Zc[site == s], axis=0) for s in np.unique(site)])))
    log(f"Checks: another site's rows do not change this site's centered features: {'PASS' if iso1 else 'FAIL'}; held-out site's rows do not change any other site's centered features: {'PASS' if iso2 else 'FAIL'}; "
        f"max |site mean| after centering = {mean_zero:.1e}; labels are not an argument of center_by_site")
    assert iso1 and iso2 and mean_zero < 1e-10
    frozen = pd.read_csv(R / "leave_site_out_baseline" / "oof_predictions.csv").set_index("project_id").loc[[f"DEV{i:04d}" for i in range(600)]]
    assert (frozen.y_true.values == y).all() and (frozen.site.values == site).all()
    # ---- leave-one-site-out, both arms
    sites = sorted(set(site), key=lambda s: -(site == s).sum())
    res = {}
    for s in sites:
        path = OUT / "checkpoints" / f"lso_{s}.json"
        ck = fx.load_ck(path)
        te = np.where(site == s)[0]; tr = np.where(site != s)[0]
        assert not (set(site[tr]) & set(site[te]))
        if ck is None:
            if time.time() - t0 > BUDGET_S:
                log("time budget exceeded; stopping"); return
            t1 = time.time()
            sc_c, best_c = fit_predict(Zc[tr], y[tr], Zc[te])
            sc_u, best_u = fit_predict(Z[tr], y[tr], Z[te])
            ck = {"site": s, "test_idx": [int(i) for i in te], "centered": {"score": sc_c, "best": best_c}, "uncentered": {"score": sc_u, "best": best_u}, "seconds": round(time.time() - t1, 1)}
            fx.save_ck(path, ck)
        log(f"[{s}] n={len(te)} centered {ck['centered']['best']} | uncentered {ck['uncentered']['best']} ({ck['seconds']}s)")
        res[s] = ck
    # ---- five-fold control (seed 42), same centering over each site's full batch
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    ff = []
    for fold, (tr, te) in enumerate(folds):
        path = OUT / "checkpoints" / f"fivefold_{fold}.json"
        ck = fx.load_ck(path)
        if ck is None:
            sc_c, best_c = fit_predict(Zc[tr], y[tr], Zc[te])
            ck = {"fold": fold, "test_idx": [int(i) for i in te], "score": sc_c, "best": best_c}
            fx.save_ck(path, ck)
        ff.append(ck)
    report(dev, y, site, sites, res, ff, frozen, time.time() - t0, pre, Z, Zc)


def report(dev, y, site, sites, res, ff, frozen, total_s, pre, Z, Zc):
    oof = []
    for s, ck in res.items():
        for i, sc_c, sc_u in zip(ck["test_idx"], ck["centered"]["score"], ck["uncentered"]["score"]):
            oof.append({"project_id": f"DEV{i:04d}", "site": s, "y_true": int(y[i]), "score_centered": sc_c, "pred_centered": int(sc_c > 0.5), "score_uncentered_same_pipeline": sc_u,
                        "pred_uncentered_same_pipeline": int(sc_u > 0.5), "score_frozen_baseline": float(frozen.loc[f"DEV{i:04d}", "score"]), "pred_frozen_baseline": int(frozen.loc[f"DEV{i:04d}", "y_pred"])})
    o = pd.DataFrame(oof).sort_values("project_id").reset_index(drop=True); o.to_csv(OUT / "oof_predictions.csv", index=False)
    assert len(o) == 600 and o.project_id.is_unique
    arms = {"CENTERED": ("score_centered", "pred_centered"), "UNCENTERED_SAME_PIPELINE": ("score_uncentered_same_pipeline", "pred_uncentered_same_pipeline"), "FROZEN_BASELINE": ("score_frozen_baseline", "pred_frozen_baseline")}
    agg, per = [], []
    for name, (sc, pr) in arms.items():
        m = fx.metrics(o.y_true.values, o[pr].values, o[sc].values)
        gap = np.mean([abs(g[pr].mean() - g.y_true.mean()) for _, g in o.groupby("site") if len(g) >= 20])
        agg.append({"arm": name, **{k: m[k] for k in MET}, "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"], "mean_abs_gap_predicted_vs_true_ASD_share_sites_n_ge_20": float(gap),
                    "note": "pooled over 600 out-of-site predictions; pooled AUC mixes 19 models"})
        for s, g in o.groupby("site"):
            mm = fx.metrics(g.y_true.values, g[pr].values, g[sc].values) if g.y_true.nunique() == 2 else {}
            per.append({"arm": name, "site": s, "n": len(g), "small_site_n_lt_20": len(g) < 20, "accuracy": float((g.y_true == g[pr]).mean()), "sensitivity": mm.get("sensitivity", np.nan), "specificity": mm.get("specificity", np.nan),
                        "roc_auc": mm.get("roc_auc", np.nan), "predicted_ASD_share": float(g[pr].mean()), "true_ASD_share": float(g.y_true.mean())})
    adf = pd.DataFrame(agg); adf.to_csv(OUT / "aggregate_metrics.csv", index=False)
    pdf = pd.DataFrame(per); piv = pdf.pivot(index="site", columns="arm", values=["n", "accuracy", "roc_auc", "predicted_ASD_share", "true_ASD_share"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]; piv = piv.reset_index()
    piv["acc_diff_centered_minus_frozen"] = piv.accuracy_CENTERED - piv.accuracy_FROZEN_BASELINE
    piv["acc_diff_centered_minus_uncentered_same_pipeline"] = piv.accuracy_CENTERED - piv.accuracy_UNCENTERED_SAME_PIPELINE
    piv["n"] = piv["n_CENTERED"]; piv["small_site_n_lt_20"] = piv.n < 20
    piv.drop(columns=[c for c in piv.columns if c.startswith("n_") or c.startswith("true_ASD_share_") and not c.endswith("CENTERED")]).to_csv(OUT / "per_site_results.csv", index=False)
    a = adf.set_index("arm")
    big = piv[piv.n >= 20]
    cond = {"pooled_accuracy_gain_ge_0.02": bool(a.loc["CENTERED", "accuracy"] - a.loc["FROZEN_BASELINE", "accuracy"] >= 0.02 - 1e-12),
            "pooled_balanced_accuracy_gain_ge_0.02": bool(a.loc["CENTERED", "balanced_accuracy"] - a.loc["FROZEN_BASELINE", "balanced_accuracy"] >= 0.02 - 1e-12),
            "pooled_auc_not_lower": bool(a.loc["CENTERED", "roc_auc"] >= a.loc["FROZEN_BASELINE", "roc_auc"] - 1e-12),
            "accuracy_higher_at_majority_of_13_sites_n_ge_20": bool((big.acc_diff_centered_minus_frozen > 0).sum() >= 7)}
    passed = all(cond.values())
    # five-fold control
    ffo = []
    for ck in ff:
        for i, s_ in zip(ck["test_idx"], ck["score"]):
            ffo.append({"project_id": f"DEV{i:04d}", "fold": ck["fold"], "y_true": int(y[i]), "score": s_, "y_pred": int(s_ > 0.5)})
    fo = pd.DataFrame(ffo).sort_values("project_id"); fo.to_csv(OUT / "five_fold_control_oof_predictions.csv", index=False)
    fm = fx.metrics(fo.y_true.values, fo.y_pred.values, fo.score.values)
    fsaved = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str}); fsaved = fsaved[(fsaved.atlas == "AAL116") & (fsaved.model == "LR_L2")]
    fb = fx.metrics(fsaved.y_true.values, fsaved.y_pred.values, fsaved.score.values)
    pd.DataFrame([{"arm": "five-fold CENTERED (control)", **{k: fm[k] for k in MET}}, {"arm": "five-fold FROZEN baseline (saved)", **{k: fb[k] for k in MET}}]).to_csv(OUT / "five_fold_control.csv", index=False)
    now = sha_all(); drift = [k for k in pre if k in now and pre[k]["sha256"] != now[k]]
    json.dump({"rule_conditions": cond, "rule_passed": passed, "pooled_gain_accuracy": float(a.loc["CENTERED", "accuracy"] - a.loc["FROZEN_BASELINE", "accuracy"]),
               "pooled_gain_balanced_accuracy": float(a.loc["CENTERED", "balanced_accuracy"] - a.loc["FROZEN_BASELINE", "balanced_accuracy"]),
               "pooled_auc_change": float(a.loc["CENTERED", "roc_auc"] - a.loc["FROZEN_BASELINE", "roc_auc"]),
               "sites_n_ge_20_accuracy_higher_lower_equal": [int((big.acc_diff_centered_minus_frozen > 0).sum()), int((big.acc_diff_centered_minus_frozen < 0).sum()), int((big.acc_diff_centered_minus_frozen == 0).sum())],
               "protocol": "see docstring of run_test_site_centering.py", "protected_files_changed": len(drift), "runtime_s": round(total_s, 1)}, open(OUT / "config.json", "w"), indent=1)
    log(f"Outputs written; RULE PASSED: {passed}; conditions {cond}; protected files changed={len(drift)}; runtime {total_s:.0f}s")
    log("\n" + adf[["arm", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "mean_abs_gap_predicted_vs_true_ASD_share_sites_n_ge_20"]].round(4).to_string(index=False))
    log("\n" + pd.read_csv(OUT / "five_fold_control.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
