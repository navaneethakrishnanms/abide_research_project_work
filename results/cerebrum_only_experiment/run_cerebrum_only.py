"""
Cerebrum-only AAL functional connectivity (ROIs 1-90, 4,005 edges) vs the full 116-ROI baseline (6,670 edges).
600 development subjects, the five fixed outer folds of the fast experiment, inner-CV seeds 42/123/2026, plus one leave-one-site-out pass.

Both representations use the SAME frozen baseline procedure: training-fold mean imputation -> Fisher-Z (clip 1e-6) -> StandardScaler ->
SelectKBest -> L2 logistic regression (balanced); grid C {1e-4,1e-3,1e-2,1e-1} x k {500, 2000, all} by inner stratified 3-fold ROC-AUC on the
training rows (fx.pipe + the same grid). The ROI subset is fixed in advance (indices 1-90; the mapping is asserted below) and has no learned parameters.
PREDECLARED RULE: 'improvement' only if, against the 116-ROI baseline with the same inner seed, mean accuracy AND balanced accuracy each rise by >= 0.02,
accuracy improves in >= 4/5 folds, and this holds for >= 2 of 3 seeds. Leave-one-site-out is a single sensitivity pass (seed 42), not used to choose anything.
Holdout, the restricted ID file and CC200 are never touched.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
R = ROOT / "results"
BASE = R / "fast_accuracy_experiment"
OUT = R / "cerebrum_only_experiment"
for s in ("checkpoints", "logs", "data"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "cerebrum_only.log"
SEEDS = [42, 123, 2026]
GRID = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1], "sel__k": [500, 2000, "all"]}
N_CERE = 90
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
BUDGET_S = 1200


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def fit_predict(Xtr, ytr, Xte, seed):
    gs = GridSearchCV(fx.pipe(LogisticRegression(class_weight="balanced", max_iter=2000), None), GRID, scoring="roc_auc",
                      cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=seed), n_jobs=6, refit=True, error_score="raise")
    gs.fit(Xtr, ytr)
    sc = gs.best_estimator_.predict_proba(Xte)[:, 1]
    return [float(v) for v in sc], {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in gs.best_params_.items()}, float(gs.best_score_)


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
    log("=" * 60 + "\nCEREBRUM-ONLY EXPERIMENT start")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    drift = [k for k, v in sha_all().items() if k in pre and pre[k]["sha256"] != v]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}"); assert not drift
    # ---- atlas mapping verification (before selecting ROIs)
    mp = pd.read_csv(R / "phase4_1_constant_roi_audit" / "aal116_roi_mapping.csv")
    assert len(mp) == 116 and (mp.roi_index.values == np.arange(1, 117)).all() and mp.mapping_verified.all() and mp.header_code_match.all()
    cer = mp.roi_name.str.contains("Cerebel|Vermis")
    assert not cer[:N_CERE].any() and cer[N_CERE:].all() and cer.sum() == 26
    log(f"Atlas mapping verified: ROIs 1-{N_CERE} are cerebral (last: {mp.roi_name[N_CERE - 1]}, code {mp.aal_code[N_CERE - 1]}); ROIs {N_CERE + 1}-116 are the 26 cerebellar/vermis ROIs (first: {mp.roi_name[N_CERE]})")
    dev, _ = fx.build_dev_set(); y = dev["y"].values; site = dev["site"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all() and (man["site"].values == site).all()
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds))
    # ---- recompute 4,005 cerebral edges from the per-subject ROI time series
    cache = OUT / "data" / "fc_raw_cerebrum90.npy"
    X116 = fx.load_fc(dev, "AAL116")
    if cache.exists():
        X90 = np.load(cache)
    else:
        files = {}
        for f in fx.AAL_DIR.glob("*_rois_aal.1D"):
            s = fx.subj_id_from_file(f.name, "rois_aal")
            if s in set(dev["sid"]):
                files[s] = f
        assert len(files) == 600
        iu90 = np.triu_indices(N_CERE, 1)
        X90 = np.full((600, len(iu90[0])), np.nan)
        for i, s in enumerate(dev["sid"]):
            ts = np.loadtxt(str(files[s]))
            assert ts.shape[1] == 116
            with np.errstate(all="ignore"):
                X90[i] = np.corrcoef(ts[:, :N_CERE], rowvar=False)[iu90]
        np.save(cache, X90)
    assert X90.shape == (600, N_CERE * (N_CERE - 1) // 2) == (600, 4005)
    idx = pd.read_csv(R / "phase5a_fc_diagnostic" / "fc_feature_index.csv")
    keep = np.where((idx.roi_i.values <= N_CERE) & (idx.roi_j.values <= N_CERE))[0]
    assert len(keep) == 4005
    same = np.array_equal(np.isnan(X90), np.isnan(X116[:, keep])) and float(np.nanmax(np.abs(X90 - X116[:, keep]))) < 1e-12
    log(f"Recomputed 4,005-edge matrix equals the corresponding columns of the cached 6,670-edge matrix (same order and NaN pattern): {'PASS' if same else 'FAIL'}; NaN entries {int(np.isnan(X90).sum())} (116-ROI: {int(np.isnan(X116).sum())})")
    assert same
    # ---- leakage check on one fold
    tr0, te0 = folds[0]
    p = fx.pipe(LogisticRegression(class_weight="balanced", max_iter=2000), None).set_params(sel__k=500, clf__C=1e-3).fit(X90[tr0], y[tr0])
    base = p.predict_proba(X90[te0])[:, 1]
    Xj = X90[te0].copy(); half = np.arange(len(te0)) % 2 == 1; Xj[half] = np.random.default_rng(0).uniform(-0.9, 0.9, Xj[half].shape)
    ok = bool(np.allclose(base[~half], p.predict_proba(Xj)[:, 1][~half], atol=1e-12)); log(f"Check: test rows do not influence one another or the fit: {'PASS' if ok else 'FAIL'}"); assert ok
    saved = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str})
    saved = saved[(saved.atlas == "AAL116") & (saved.model == "LR_L2")].set_index("subject_id").loc[dev["sid"]]
    reps = {"ROI116_baseline": X116, "ROI90_cerebrum": X90}
    prim = []
    for seed in SEEDS:
        for fold, (tr, te) in enumerate(folds):
            path = OUT / "checkpoints" / f"outer_seed{seed}_fold{fold}.json"
            ck = fx.load_ck(path)
            if ck is None:
                if time.time() - t0 > BUDGET_S:
                    log("time budget exceeded; stopping"); return
                t1 = time.time(); res = {}
                for name, X in reps.items():
                    sc, best, ia = fit_predict(X[tr], y[tr], X[te], seed)
                    res[name] = {"score": sc, "best": best, "inner_auc": ia}
                ck = {"seed": seed, "fold": fold, "test_idx": [int(i) for i in te], "res": res, "seconds": round(time.time() - t1, 1)}
                fx.save_ck(path, ck)
            log(f"[outer] seed {seed} fold {fold}: 116 {ck['res']['ROI116_baseline']['best']} | 90 {ck['res']['ROI90_cerebrum']['best']} ({ck['seconds']}s)")
            prim.append(ck)
    rep = [bool(((np.array(ck["res"]["ROI116_baseline"]["score"]) > 0.5).astype(int) == saved.y_pred.values[ck["test_idx"]]).all() and np.allclose(ck["res"]["ROI116_baseline"]["score"], saved.score.values[ck["test_idx"]], atol=1e-9)) for ck in prim if ck["seed"] == 42]
    log(f"Reproduction of the frozen 116-ROI baseline (seed 42) vs saved predictions: {'EXACT in all 5 folds' if all(rep) else rep}")
    # ---- leave-one-site-out, 90-ROI only (the 116-ROI result is the saved leave_site_out_baseline run)
    lso = []
    for s in sorted(set(site), key=lambda s: -(site == s).sum()):
        path = OUT / "checkpoints" / f"lso_{s}.json"
        ck = fx.load_ck(path)
        te = np.where(site == s)[0]; tr = np.where(site != s)[0]
        assert not (set(site[tr]) & set(site[te]))
        if ck is None:
            if time.time() - t0 > BUDGET_S:
                log("time budget exceeded in leave-site-out; stopping"); return
            t1 = time.time(); sc, best, ia = fit_predict(X90[tr], y[tr], X90[te], 42)
            ck = {"site": s, "test_idx": [int(i) for i in te], "score": sc, "best": best, "inner_auc": ia, "seconds": round(time.time() - t1, 1)}
            fx.save_ck(path, ck)
        lso.append(ck)
    log(f"Leave-site-out (90 ROI) done for {len(lso)} sites")
    report(dev, y, site, folds, prim, lso, rep, same, time.time() - t0, pre)


def report(dev, y, site, folds, prim, lso, rep, same, total_s, pre):
    rows, oof = [], []
    for ck in prim:
        te = np.array(ck["test_idx"])
        for name, r in ck["res"].items():
            sc = np.array(r["score"]); pr = (sc > 0.5).astype(int)
            m = fx.metrics(y[te], pr, sc)
            rows.append({"representation": name, "seed": ck["seed"], "fold": ck["fold"], "selected_C": r["best"]["clf__C"], "selected_k": r["best"]["sel__k"], "inner_cv_auc": r["inner_auc"], **m})
            for i, s_, p_ in zip(te, sc, pr):
                oof.append({"representation": name, "seed": ck["seed"], "fold": ck["fold"], "project_id": f"DEV{i:04d}", "site": site[i], "y_true": int(y[i]), "score": float(s_), "y_pred": int(p_)})
    fm = pd.DataFrame(rows); fm.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(oof).to_csv(OUT / "oof_predictions.csv", index=False)
    summ, verdict = [], {}
    for seed in SEEDS:
        a = fm[(fm.representation == "ROI90_cerebrum") & (fm.seed == seed)].sort_values("fold"); b = fm[(fm.representation == "ROI116_baseline") & (fm.seed == seed)].sort_values("fold")
        for name, g in (("ROI116_baseline", b), ("ROI90_cerebrum", a)):
            summ.append({"representation": name, "seed": seed, **{f"{m}_mean": float(g[m].mean()) for m in MET}, **{f"{m}_sd": float(g[m].std(ddof=1)) for m in MET}})
        d = {m: a[m].values - b[m].values for m in MET}
        verdict[seed] = {"acc_diff": float(d["accuracy"].mean()), "bacc_diff": float(d["balanced_accuracy"].mean()), "auc_diff": float(d["roc_auc"].mean()), "acc_diff_sd": float(d["accuracy"].std(ddof=1)),
                         "folds_acc_improved": int((d["accuracy"] > 0).sum()), "per_fold_acc_diff": [round(float(v), 4) for v in d["accuracy"]], "per_fold_auc_diff": [round(float(v), 4) for v in d["roc_auc"]],
                         "sens_diff": float(d["sensitivity"].mean()), "spec_diff": float(d["specificity"].mean()), "f1_diff": float(d["f1"].mean()),
                         "meets_rule": bool(d["accuracy"].mean() >= 0.02 and d["balanced_accuracy"].mean() >= 0.02 and (d["accuracy"] > 0).sum() >= 4)}
    sdf = pd.DataFrame(summ); sdf.to_csv(OUT / "summary_by_seed.csv", index=False)
    improves = sum(v["meets_rule"] for v in verdict.values()) >= 2
    # leave-site-out comparison
    lo = []
    for ck in lso:
        te = np.array(ck["test_idx"]); sc = np.array(ck["score"])
        for i, s_ in zip(te, sc):
            lo.append({"representation": "ROI90_cerebrum", "project_id": f"DEV{i:04d}", "site": ck["site"], "y_true": int(y[i]), "score": float(s_), "y_pred": int(s_ > 0.5)})
    saved_lso = pd.read_csv(R / "leave_site_out_baseline" / "oof_predictions.csv").assign(representation="ROI116_baseline")
    lo = pd.concat([saved_lso[["representation", "project_id", "site", "y_true", "score", "y_pred"]], pd.DataFrame(lo)], ignore_index=True)
    lo.to_csv(OUT / "leave_site_out_oof_predictions.csv", index=False)
    lrows, per_site = [], []
    for name in ("ROI116_baseline", "ROI90_cerebrum"):
        g = lo[lo.representation == name]
        m = fx.metrics(g.y_true.values, g.y_pred.values, g.score.values)
        lrows.append({"representation": name, **{k: m[k] for k in MET}, "note": "pooled over 600 out-of-site predictions; pooled AUC mixes 19 models"})
        for s, gs in g.groupby("site"):
            mm = fx.metrics(gs.y_true.values, gs.y_pred.values, gs.score.values) if gs.y_true.nunique() == 2 else {}
            per_site.append({"representation": name, "site": s, "n": len(gs), "small_site_n_lt_20": len(gs) < 20, "accuracy": float((gs.y_true == gs.y_pred).mean()), "roc_auc": mm.get("roc_auc", np.nan)})
    ldf = pd.DataFrame(lrows); ldf.to_csv(OUT / "leave_site_out_summary.csv", index=False)
    ps = pd.DataFrame(per_site).pivot(index="site", columns="representation", values=["n", "accuracy", "roc_auc"])
    ps.columns = [f"{a}_{b}" for a, b in ps.columns]; ps = ps.reset_index(); ps["acc_diff_90_minus_116"] = ps.accuracy_ROI90_cerebrum - ps.accuracy_ROI116_baseline
    ps.to_csv(OUT / "leave_site_out_per_site.csv", index=False)
    now = sha_all(); drift = [k for k in pre if k in now and pre[k]["sha256"] != now[k]]
    json.dump({"representations": {"ROI116_baseline": "6,670 edges (AAL ROIs 1-116)", "ROI90_cerebrum": "4,005 edges (AAL ROIs 1-90; cerebellum and vermis, ROIs 91-116, removed)"}, "pipeline": "frozen baseline (fx.pipe + grid)",
               "grid": {k: [str(x) for x in v] for k, v in GRID.items()}, "seeds": SEEDS, "outer_cv": "StratifiedKFold(5, shuffle, seed 42), identical to fast experiment", "rule": "acc & bacc >= +0.02, acc improved in >=4/5 folds, in >=2/3 seeds",
               "verdict_by_seed": verdict, "improvement_rule_passed": bool(improves), "baseline_reproduced_exactly": bool(all(rep)), "recomputed_matrix_equals_cached_subset": bool(same), "protected_files_changed": len(drift), "runtime_s": round(total_s, 1)}, open(OUT / "config.json", "w"), indent=1)
    log(f"Outputs written; improvement rule passed: {improves}; reproduction exact: {all(rep)}; protected files changed={len(drift)}; runtime {total_s:.0f}s")
    log("\n" + sdf[["representation", "seed", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "roc_auc_mean"]].round(4).to_string(index=False))
    log("\n" + json.dumps(verdict, indent=1))
    log("\n" + ldf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
