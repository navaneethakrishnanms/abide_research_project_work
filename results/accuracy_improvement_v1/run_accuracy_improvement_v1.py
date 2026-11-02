"""
accuracy_improvement_v1 -- small, predeclared accuracy-improvement investigation on the frozen AAL-116 L2 logistic-regression baseline.
600 development subjects, the five fixed outer folds of the fast experiment (primary), plus a one-pass leave-one-site-out sensitivity run.

CANDIDATES (all fitted per outer-training set; nothing learned from the outer-test rows)
  E0  control = frozen baseline procedure: grid C {1e-4,1e-3,1e-2,1e-1} x k {500,2000,all}, inner StratifiedKFold(3) ROC-AUC, threshold 0.5
  E1  E0 + decision threshold chosen on TRAINING-ONLY inner out-of-fold probabilities (5-fold cross_val_predict with the selected C,k;
      threshold in {0.300,0.325,...,0.700} maximizing balanced accuracy; ties -> closest to 0.5)
  E2  extended C grid {1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1} x same k grid, threshold 0.5
  E3  E2 + the same training-only threshold rule
 Inner-CV seeds 42, 123, 2026 (affect hyper-parameter and threshold selection; outer folds fixed).
 PREDECLARED RULE: a candidate "improves" only if, against E0 with the same seed, mean accuracy AND balanced accuracy rise by >= 0.02,
 accuracy improves in >= 4/5 folds, and this holds for >= 2 of the 3 seeds. No significance test is claimed. Nothing is chosen from outer scores.
 SECONDARY (one pass, seed 42, not used for selection): the same four candidates under leave-one-site-out.
Holdout, restricted ID file and CC200 are never touched.
"""
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"C:\abide")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
R = ROOT / "results"
BASE = R / "fast_accuracy_experiment"
OUT = R / "accuracy_improvement_v1"
for s in ("checkpoints", "logs"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "accuracy_improvement_v1.log"
SEEDS = [42, 123, 2026]
KS = [500, 2000, "all"]
GRID_BASE = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1], "sel__k": KS}
GRID_EXT = {"clf__C": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "sel__k": KS}
THR = np.round(np.arange(0.30, 0.7001, 0.025), 3)
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
BUDGET_S = 1500


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def lr():
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def fit_on_training(Xtr, ytr, grid, seed):
    """Everything learned here uses ONLY the training rows passed in (the outer-test rows are never an argument)."""
    gs = GridSearchCV(fx.pipe(lr(), None), grid, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=seed), n_jobs=6, refit=True, error_score="raise")
    gs.fit(Xtr, ytr)
    best = gs.best_params_
    oof = cross_val_predict(fx.pipe(lr(), None).set_params(**best), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1000), method="predict_proba", n_jobs=5)[:, 1]
    bacc = np.array([balanced_accuracy_score(ytr, (oof >= t).astype(int)) for t in THR])
    top = np.where(bacc >= bacc.max() - 1e-12)[0]
    thr = float(THR[top[np.argmin(np.abs(THR[top] - 0.5))]])
    return gs.best_estimator_, {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best.items()}, thr, float(gs.best_score_), float(bacc.max())


def candidates_for(Xtr, ytr, Xte, seed):
    out = {}
    for gname, grid in (("base", GRID_BASE), ("ext", GRID_EXT)):
        est, best, thr, inner_auc, inner_bacc = fit_on_training(Xtr, ytr, grid, seed)
        sc = est.predict_proba(Xte)[:, 1]
        out[gname] = {"best": best, "thr": thr, "inner_auc": inner_auc, "inner_oof_bacc_at_thr": inner_bacc, "score": [float(v) for v in sc]}
    return out


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


CAND = {"E0_control_baseline": ("base", False), "E1_baseline_grid_threshold": ("base", True), "E2_extended_C_grid": ("ext", False), "E3_extended_grid_threshold": ("ext", True)}


def main():
    t0 = time.time()
    log("=" * 60 + "\nACCURACY IMPROVEMENT v1 start")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    drift = [k for k, v in sha_all().items() if k in pre and pre[k]["sha256"] != v]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}"); assert not drift
    dev, _ = fx.build_dev_set(); y = dev["y"].values; site = dev["site"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all() and (man["site"].values == site).all()
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds))
    X = fx.load_fc(dev, "AAL116")
    log(f"Development cohort {len(dev)} ({int(y.sum())} ASD / {int((1 - y).sum())} TC), {len(set(site))} sites; AAL-116 {X.shape}; outer folds identical to the frozen baseline's")
    saved = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str})
    saved = saved[(saved.atlas == "AAL116") & (saved.model == "LR_L2")].set_index("subject_id").loc[dev["sid"]]
    # ---- primary: five outer folds x three inner seeds
    prim = []
    for seed in SEEDS:
        for fold, (tr, te) in enumerate(folds):
            path = OUT / "checkpoints" / f"outer_seed{seed}_fold{fold}.json"
            ck = fx.load_ck(path)
            if ck is None:
                if time.time() - t0 > BUDGET_S:
                    log("time budget exceeded; stopping"); return
                t1 = time.time(); c = candidates_for(X[tr], y[tr], X[te], seed)
                ck = {"seed": seed, "fold": fold, "test_idx": [int(i) for i in te], "res": c, "seconds": round(time.time() - t1, 1)}
                fx.save_ck(path, ck)
            log(f"[outer] seed {seed} fold {fold}: base C,k={ck['res']['base']['best']} thr={ck['res']['base']['thr']} | ext C,k={ck['res']['ext']['best']} thr={ck['res']['ext']['thr']} ({ck['seconds']}s)")
            prim.append(ck)
    # reproduction check: E0 with seed 42 must equal the saved frozen baseline predictions
    rep = []
    for ck in prim:
        if ck["seed"] == 42:
            idx = ck["test_idx"]; sc = np.array(ck["res"]["base"]["score"])
            rep.append(bool(((sc > 0.5).astype(int) == saved.y_pred.values[idx]).all() and np.allclose(sc, saved.score.values[idx], atol=1e-9)))
    log(f"Reproduction of the frozen baseline (E0, seed 42) vs saved predictions: {'EXACT MATCH in all 5 folds' if all(rep) else 'MISMATCH ' + str(rep)}")
    # ---- secondary: leave-one-site-out, seed 42 only
    lso = []
    sites = sorted(set(site), key=lambda s: -(site == s).sum())
    for s in sites:
        path = OUT / "checkpoints" / f"lso_{s}.json"
        ck = fx.load_ck(path)
        te = np.where(site == s)[0]; tr = np.where(site != s)[0]
        assert not (set(site[tr]) & set(site[te]))
        if ck is None:
            if time.time() - t0 > BUDGET_S:
                log("time budget exceeded during leave-site-out; stopping"); return
            t1 = time.time(); c = candidates_for(X[tr], y[tr], X[te], 42)
            ck = {"site": s, "test_idx": [int(i) for i in te], "res": c, "seconds": round(time.time() - t1, 1)}
            fx.save_ck(path, ck)
        log(f"[lso] {s}: base thr={ck['res']['base']['thr']} ext thr={ck['res']['ext']['thr']} ({ck['seconds']}s)")
        lso.append(ck)
    report(dev, y, site, folds, prim, lso, rep, time.time() - t0, pre)


def cand_pred(ck_res, name):
    g, use_thr = CAND[name]
    sc = np.array(ck_res[g]["score"]); thr = ck_res[g]["thr"] if use_thr else 0.5
    return sc, thr, (sc >= thr).astype(int) if use_thr else (sc > 0.5).astype(int)


def report(dev, y, site, folds, prim, lso, rep, total_s, pre):
    rows, oof = [], []
    for ck in prim:
        te = np.array(ck["test_idx"])
        for name in CAND:
            sc, thr, pr = cand_pred(ck["res"], name)
            m = fx.metrics(y[te], pr, sc)
            g = CAND[name][0]
            rows.append({"candidate": name, "seed": ck["seed"], "fold": ck["fold"], "threshold": thr, "selected_C": ck["res"][g]["best"]["clf__C"], "selected_k": ck["res"][g]["best"]["sel__k"], **m})
            for i, s_, p_ in zip(te, sc, pr):
                oof.append({"candidate": name, "seed": ck["seed"], "fold": ck["fold"], "project_id": f"DEV{i:04d}", "site": site[i], "y_true": int(y[i]), "score": float(s_), "threshold": thr, "y_pred": int(p_)})
    fm = pd.DataFrame(rows); fm.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(oof).to_csv(OUT / "oof_predictions.csv", index=False)
    # summary and paired differences vs E0 (same seed)
    summ, pair = [], []
    for name in CAND:
        for seed in SEEDS:
            g = fm[(fm.candidate == name) & (fm.seed == seed)].sort_values("fold")
            r = {"candidate": name, "seed": seed, **{f"{m}_mean": float(g[m].mean()) for m in MET}, **{f"{m}_sd": float(g[m].std(ddof=1)) for m in MET}, "mean_threshold": float(g.threshold.mean())}
            if name != "E0_control_baseline":
                b = fm[(fm.candidate == "E0_control_baseline") & (fm.seed == seed)].sort_values("fold")
                for m in MET:
                    d = g[m].values - b[m].values
                    r[f"{m}_diff_mean"], r[f"{m}_diff_sd"], r[f"{m}_folds_improved"] = float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum())
            summ.append(r)
    sdf = pd.DataFrame(summ); sdf.to_csv(OUT / "candidate_summary_by_seed.csv", index=False)
    verdict = {}
    for name in CAND:
        if name == "E0_control_baseline":
            continue
        s = sdf[(sdf.candidate == name)]
        ok = ((s.accuracy_diff_mean >= 0.02) & (s.balanced_accuracy_diff_mean >= 0.02) & (s.accuracy_folds_improved >= 4))
        verdict[name] = {"seeds_meeting_rule": int(ok.sum()), "improves": bool(ok.sum() >= 2), "acc_diff_by_seed": [round(float(v), 4) for v in s.accuracy_diff_mean], "bacc_diff_by_seed": [round(float(v), 4) for v in s.balanced_accuracy_diff_mean],
                         "auc_diff_by_seed": [round(float(v), 4) for v in s.roc_auc_diff_mean], "folds_acc_improved_by_seed": [int(v) for v in s.accuracy_folds_improved]}
    # seed-averaged table
    avg = fm.groupby(["candidate", "seed"])[MET].mean().groupby("candidate").agg(["mean", "std"])
    avg.columns = [f"{a}_{b}_over_seeds" for a, b in avg.columns]; avg.to_csv(OUT / "candidate_summary_seed_averaged.csv")
    # leave-site-out secondary
    lrows, loof = [], []
    for ck in lso:
        te = np.array(ck["test_idx"])
        for name in CAND:
            sc, thr, pr = cand_pred(ck["res"], name)
            for i, s_, p_ in zip(te, sc, pr):
                loof.append({"candidate": name, "project_id": f"DEV{i:04d}", "site": ck["site"], "y_true": int(y[i]), "score": float(s_), "threshold": thr, "y_pred": int(p_)})
    lo = pd.DataFrame(loof); lo.to_csv(OUT / "leave_site_out_oof_predictions.csv", index=False)
    lm = []
    for name in CAND:
        g = lo[lo.candidate == name]
        m = fx.metrics(g.y_true.values, g.y_pred.values, g.score.values)
        per = [fx.metrics(gs.y_true.values, gs.y_pred.values, gs.score.values) for _, gs in g.groupby("site") if gs.site.nunique() == 1]
        gap = np.mean([abs(gs.y_pred.mean() - gs.y_true.mean()) for _, gs in g.groupby("site") if len(gs) >= 20])
        lm.append({"candidate": name, "pooled_accuracy": m["accuracy"], "pooled_balanced_accuracy": m["balanced_accuracy"], "pooled_sensitivity": m["sensitivity"], "pooled_specificity": m["specificity"], "pooled_f1": m["f1"],
                   "pooled_auc_mixed_models": m["roc_auc"], "mean_abs_gap_predicted_vs_true_ASD_share_sites_n_ge_20": float(gap)})
    ldf = pd.DataFrame(lm); ldf.to_csv(OUT / "leave_site_out_secondary_summary.csv", index=False)
    now = sha_all(); drift = [k for k in pre if k in now and pre[k]["sha256"] != now[k]]
    json.dump({"candidates": {k: {"grid": ("baseline" if v[0] == "base" else "extended"), "training_only_threshold": v[1]} for k, v in CAND.items()}, "grid_base": {k: [str(x) for x in v] for k, v in GRID_BASE.items()},
               "grid_ext": {k: [str(x) for x in v] for k, v in GRID_EXT.items()}, "threshold_grid": [float(t) for t in THR], "inner_cv": "StratifiedKFold(3, seed) roc_auc; threshold from StratifiedKFold(5, seed+1000) inner OOF on training rows",
               "seeds": SEEDS, "outer_cv": "StratifiedKFold(5, shuffle, seed 42) identical to fast experiment", "improvement_rule": "mean accuracy and balanced accuracy >= +0.02 vs E0 (same seed), accuracy improved in >=4/5 folds, in >=2 of 3 seeds",
               "baseline_reproduction_exact": bool(all(rep)), "verdict": verdict, "protected_files_changed": len(drift), "runtime_s": round(total_s, 1)}, open(OUT / "config.json", "w"), indent=1)
    log(f"Outputs written; baseline reproduction exact: {all(rep)}; protected files changed={len(drift)}; runtime {total_s:.0f}s")
    log("\n" + sdf[["candidate", "seed", "accuracy_mean", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "roc_auc_mean", "mean_threshold"]].round(4).to_string(index=False))
    log("\n" + json.dumps(verdict, indent=1, default=float))
    log("\n" + ldf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
