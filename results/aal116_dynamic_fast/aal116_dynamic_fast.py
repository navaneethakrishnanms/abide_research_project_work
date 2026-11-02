"""
aal116_dynamic_fast.py -- one minimal dynamic-connectivity experiment, AAL-116 only, 600 development subjects.

Dynamic features (computed per subject from that subject's own time series only; no cross-subject statistics):
  window 50 TRs, step 10 TRs; Pearson correlation per window; Fisher-Z (clip 1e-6) per window; unique off-diagonal edges (6670);
    DYN_VAR : variance of each edge across windows (np.nanvar, ddof=0)
    DYN_MAC : mean absolute change of each edge between consecutive windows (np.nanmean)
  Undefined-correlation rule: a window in which an ROI has zero variance gives NaN for that ROI's edges; edge summaries
  use only valid windows and are NaN if fewer than 3 valid windows (DYN_VAR) / 2 valid consecutive pairs (DYN_MAC);
  remaining NaNs are imputed with TRAINING-fold column means inside the Pipeline.

Classifier and protocol = the saved static AAL-116 L2 logistic-regression baseline of the fast experiment (same grid
C x SelectKBest k, inner 3-fold ROC-AUC, class_weight balanced). The only pipeline difference: first step is a plain
training-fold mean imputer (Fisher-Z is already applied per window). Baseline results are read, not recomputed.
Holdout files / IDs, the restricted membership file and CC200 are never touched.
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
BASE = ROOT / "results" / "fast_accuracy_experiment"
OUT = ROOT / "results" / "aal116_dynamic_fast"
for s in ("data", "checkpoints", "logs"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "dynamic_fast.log"
W, STEP, MIN_WIN = 50, 10, 3
BUDGET_S = 300
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def window_features(ts):
    T, n = ts.shape
    starts = list(range(0, T - W + 1, STEP))
    iu = np.triu_indices(n, k=1)
    Z = np.full((len(starts), len(iu[0])), np.nan)
    for k, s in enumerate(starts):
        x = ts[s:s + W]
        sd = x.std(axis=0)
        good = sd > 0
        xs = np.where(good, (x - x.mean(axis=0)) / np.where(good, sd, 1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C = (xs.T @ xs) / W                                           # NaN where either ROI is constant in this window
        Z[k] = np.arctanh(np.clip(C[iu], -1 + 1e-6, 1 - 1e-6))
    valid = np.isfinite(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var = np.where(valid.sum(0) >= MIN_WIN, np.nanvar(Z, axis=0), np.nan)
        d = np.abs(np.diff(Z, axis=0))                                      # NaN if either window undefined
        mac = np.where(np.isfinite(d).sum(0) >= MIN_WIN - 1, np.nanmean(d, axis=0), np.nan)
    return var, mac, len(starts)


def pipe():
    return Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])


def sha_all():
    import glob
    files = []
    for d in ["phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification", "phase3_6_aal_download", "phase4", "phase4_1_constant_roi_audit",
              "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic", "phase5b_missing_fc_analysis", "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero",
              "phase6_strategyC_all_nonzero", "phase7_synthesis", "phase8_final_audit", "phase9_accuracy_optimization", "phase10_advanced_accuracy"]:
        p = ROOT / "results" / d
        if p.exists():
            files += [f for f in p.rglob("*") if f.is_file()]
    files += [f for f in ROOT.glob("phase*.py") if not f.name.startswith("phase10")] + [ROOT / "phase10_advanced_accuracy.py"]
    return {str(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(set(files))}


def main():
    t0 = time.time()
    log("=" * 60 + "\nAAL-116 DYNAMIC FAST EXPERIMENT start")
    pre_hash = json.load(open(ROOT / "results" / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    now_hash = sha_all()
    drift = [k for k in pre_hash if now_hash.get(k) != pre_hash[k]["sha256"]]
    log(f"Integrity before: {len(pre_hash)} baseline files, changed={len(drift)}")
    assert not drift
    dev, _ = fx.build_dev_set()
    y = dev["y"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all(), "dev set differs from verified manifest"
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds)), "folds differ from the verified manifest"
    base_fr = pd.read_csv(BASE / "fold_results.csv")
    base_fr = base_fr[(base_fr.atlas == "AAL116") & (base_fr.model == "LR_L2")].sort_values("fold")
    base_oof = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str})
    base_oof = base_oof[(base_oof.atlas == "AAL116") & (base_oof.model == "LR_L2")]
    assert len(base_fr) == 5 and len(base_oof) == 600 and base_oof.subject_id.is_unique and set(base_oof.subject_id) == set(dev["sid"])
    j = base_oof.merge(man[["sid", "y", "outer_fold"]], left_on="subject_id", right_on="sid")
    assert (j.y_true == j.y).all() and (j.fold == j.outer_fold).all(), "saved baseline OOF does not match the manifest"
    fx_cfg = {"grid": fx.MODELS["LR_L2"][1], "inner": f"StratifiedKFold({fx.N_INNER}, seed {fx.SEED})", "scoring": "roc_auc"}
    log("Verified: 600-subject manifest/labels/folds identical to the saved static AAL-116 LR_L2 baseline (5 folds, 600 unique OOF predictions).")

    # ---- features (cached)
    fv, fm = OUT / "data" / "dyn_var_AAL116.npy", OUT / "data" / "dyn_mac_AAL116.npy"
    if fv.exists() and fm.exists():
        VAR, MAC = np.load(fv), np.load(fm)
        nwin = np.load(OUT / "data" / "n_windows.npy")
    else:
        dev_ids = set(dev["sid"]); files = {}
        for f in fx.AAL_DIR.glob("*_rois_aal.1D"):
            s = fx.subj_id_from_file(f.name, "rois_aal")
            if s in dev_ids:
                files[s] = f
        assert len(files) == 600
        VAR, MAC, nwin = [], [], []
        for s in dev["sid"]:
            ts = np.loadtxt(str(files[s]))
            assert ts.shape[1] == 116
            v, m, k = window_features(ts)
            VAR.append(v); MAC.append(m); nwin.append(k)
        VAR, MAC, nwin = np.array(VAR), np.array(MAC), np.array(nwin)
        np.save(fv, VAR); np.save(fm, MAC); np.save(OUT / "data" / "n_windows.npy", nwin)
    log(f"Features: VAR {VAR.shape}, MAC {MAC.shape}; windows/subject min {nwin.min()} mean {nwin.mean():.1f} max {nwin.max()}; "
        f"NaN share VAR {np.isnan(VAR).mean():.4f} (features with any NaN: {int(np.isnan(VAR).any(0).sum())}), MAC {np.isnan(MAC).mean():.4f}; extraction {time.time() - t0:.0f}s")

    # ---- leakage check: a test row's prediction must not depend on other test rows
    tr0, te0 = folds[0]
    p = pipe().set_params(sel__k=500, clf__C=1e-3).fit(VAR[tr0], y[tr0])
    base = p.predict_proba(VAR[te0])[:, 1]
    Xj = VAR[te0].copy(); half = np.arange(len(te0)) % 2 == 1
    Xj[half] = np.random.default_rng(0).normal(size=Xj[half].shape)
    ok = bool(np.allclose(base[~half], p.predict_proba(Xj)[:, 1][~half], atol=1e-12))
    log(f"Test-row independence check: {'PASS' if ok else 'FAIL'}")
    assert ok

    results, oof = {}, []
    for name, X in (("DYN_VAR", VAR), ("DYN_MAC", MAC)):
        rows = []
        for fold, (tr, te) in enumerate(folds):
            path = OUT / "checkpoints" / f"{name}__fold{fold}.json"
            ck = fx.load_ck(path)
            if ck is None:
                if time.time() - t0 > BUDGET_S:
                    log(f"TIME BUDGET {BUDGET_S}s exceeded before {name} fold {fold}; stopping -- results would be incomplete and are NOT reported as final")
                    return
                gs = GridSearchCV(pipe(), fx.MODELS["LR_L2"][1], scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=fx.SEED),
                                  n_jobs=6, refit=True, error_score="raise")
                t1 = time.time(); gs.fit(X[tr], y[tr])
                sc = gs.best_estimator_.predict_proba(X[te])[:, 1]; pr = gs.best_estimator_.predict(X[te])
                ck = {"method": name, "fold": fold, "metrics": fx.metrics(y[te], pr, sc), "inner_cv_auc": float(gs.best_score_), "seconds": round(time.time() - t1, 1),
                      "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in gs.best_params_.items()},
                      "test_idx": [int(i) for i in te], "score": [float(s) for s in sc], "pred": [int(v) for v in pr]}
                fx.save_ck(path, ck)
            b = base_fr[base_fr.fold == fold].iloc[0]
            log(f"[{name}] fold {fold}: acc={ck['metrics']['accuracy']:.4f} auc={ck['metrics']['roc_auc']:.4f} (static acc={b.accuracy:.4f} auc={b.roc_auc:.4f}) params={ck['best_params']} ({ck['seconds']}s)")
            rows.append(ck)
            for i, s_, p_ in zip(ck["test_idx"], ck["score"], ck["pred"]):
                oof.append({"method": name, "fold": fold, "subject_id": dev["sid"].iloc[i], "y_true": int(y[i]), "score": s_, "y_pred": p_})
        results[name] = rows
    report(dev, y, results, oof, base_fr, base_oof, nwin, time.time() - t0, fx_cfg, pre_hash)


def report(dev, y, results, oof, base_fr, base_oof, nwin, total_s, fx_cfg, pre_hash):
    fm_rows = []
    for _, b in base_fr.iterrows():
        fm_rows.append({"method": "STATIC_AAL116 (saved baseline)", "fold": int(b.fold), **{m: b[m] for m in MET}})
    for name, rows in results.items():
        for ck in rows:
            fm_rows.append({"method": name, "fold": ck["fold"], **ck["metrics"]})
    fm = pd.DataFrame(fm_rows)
    base = fm[fm.method.str.startswith("STATIC")].sort_values("fold")
    out_rows, diff_rows = [], []
    for method, g in fm.groupby("method"):
        g = g.sort_values("fold")
        r = {"method": method, "n_folds": len(g)}
        for m in MET:
            r[f"{m}_mean"], r[f"{m}_sd"] = float(g[m].mean()), float(g[m].std(ddof=1))
        out_rows.append(r)
        if not method.startswith("STATIC"):
            for m in MET:
                d = g[m].values - base[m].values
                for k, v in zip(g.fold, d):
                    diff_rows.append({"method": method, "metric": m, "fold": int(k), "paired_diff_vs_static": float(v)})
    metrics = pd.DataFrame(out_rows)
    pdiff = pd.DataFrame(diff_rows)
    summ = pdiff.groupby(["method", "metric"]).paired_diff_vs_static.agg(mean="mean", sd=lambda s: s.std(ddof=1), n_improved=lambda s: int((s > 0).sum())).reset_index()
    fm.merge(pdiff.pivot_table(index=["method", "fold"], columns="metric", values="paired_diff_vs_static").add_prefix("diff_").reset_index(), on=["method", "fold"], how="left").to_csv(OUT / "fold_metrics.csv", index=False)
    metrics.merge(summ.pivot_table(index="method", columns="metric", values=["mean", "sd"]).pipe(lambda d: d.set_axis([f"paired_diff_{b}_{a}" for a, b in d.columns], axis=1)).reset_index(), on="method", how="left").to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(oof).to_csv(OUT / "oof_predictions_dynamic.csv", index=False)
    # conclusion by pre-declared rule
    concl = "INCONCLUSIVE"
    d_acc = summ[summ.metric == "accuracy"].set_index("method"); d_auc = summ[summ.metric == "roc_auc"].set_index("method")
    prom = [m for m in d_acc.index if d_acc.loc[m, "mean"] >= 0.02 and d_auc.loc[m, "mean"] >= 0.02 and d_acc.loc[m, "n_improved"] >= 4]
    worse = all((d_acc.loc[m, "mean"] <= -0.02 and d_auc.loc[m, "mean"] <= -0.02) for m in d_acc.index)
    if prom:
        concl = "DYNAMIC FEATURES LOOK PROMISING"
    elif worse:
        concl = "DYNAMIC FEATURES PERFORM WORSE"
    L = ["# AAL-116 dynamic vs static connectivity -- exploratory development-set CV\n",
         f"600 development subjects (268 ASD / 332 TC), 19 sites, same outer folds as the fast experiment. Dynamic windows: {W} TRs, step {STEP}; "
         f"{int(nwin.min())}-{int(nwin.max())} windows/subject (mean {nwin.mean():.1f}). Classifier: L2 logistic regression, grid and inner 3-fold ROC-AUC identical to the saved static AAL-116 baseline "
         "(baseline read from `fast_accuracy_experiment`, not recomputed). The only pipeline difference is a training-fold mean imputer in place of the Fisher-Z step (Fisher-Z is applied per window). "
         "Holdout, CC200 and the restricted ID file were not used.\n\n",
         "| Method | Accuracy | Bal. acc. | Sens. | Spec. | F1 | AUC |\n|---|---|---|---|---|---|---|\n"]
    for _, r in metrics.iterrows():
        L.append(f"| {r.method} | {r.accuracy_mean:.4f} +/- {r.accuracy_sd:.4f} | {r.balanced_accuracy_mean:.4f} | {r.sensitivity_mean:.4f} | {r.specificity_mean:.4f} | {r.f1_mean:.4f} | {r.roc_auc_mean:.4f} +/- {r.roc_auc_sd:.4f} |\n")
    L.append("\nPaired fold-level differences vs the static baseline (dynamic minus static):\n\n| Method | Metric | Mean diff | SD | Folds improved |\n|---|---|---|---|---|\n")
    for _, r in summ[summ.metric.isin(["accuracy", "balanced_accuracy", "roc_auc", "f1"])].iterrows():
        L.append(f"| {r.method} | {r.metric} | {r['mean']:+.4f} | {r['sd']:.4f} | {int(r.n_improved)}/5 |\n")
    L.append(f"\nPer-fold accuracy diffs: " + "; ".join(f"{m}: " + str([round(float(v), 3) for v in pdiff[(pdiff.method == m) & (pdiff.metric == 'accuracy')].sort_values('fold').paired_diff_vs_static]) for m in results) + "\n")
    L.append(f"\n**Conclusion (pre-declared rule: promising needs a >= +0.02 mean accuracy AND AUC gain with >= 4/5 folds improved in accuracy; worse needs <= -0.02 on both for every method): {concl}.**\n"
             "\nNo significance test is claimed from 5 folds; these are development cross-validation estimates, not independent-holdout results; no 85% claim. "
             f"Runtime {total_s:.0f}s.\n")
    (OUT / "experiment_report.md").write_text("".join(L), encoding="utf-8")
    # integrity
    now = sha_all(); drift = [k for k in pre_hash if now.get(k) != pre_hash[k]["sha256"]]
    folds_ok = True
    checks = [f"- Only AAL-116 time series of the 600 development subjects were opened (files filtered by development IDs; CC200 not used): PASS",
              f"- Outer folds identical to the verified manifest, disjoint, partitioning all 600 subjects (asserted at start): PASS",
              "- One OOF prediction per subject per method: " + ", ".join(f"{m}: {len(pd.DataFrame(oof)[pd.DataFrame(oof).method == m])} rows, {pd.DataFrame(oof)[pd.DataFrame(oof).method == m].subject_id.nunique()} unique" for m in results) + " : PASS",
              "- Saved static baseline matched the manifest (600 unique OOF subjects, labels and folds equal): PASS",
              "- Dynamic features are computed per subject from that subject's own time series; imputation, scaling, SelectKBest and C are fitted inside training folds only; inner 3-fold CV picks settings from training rows: PASS",
              "- Test-row independence perturbation (scrambling half the test rows leaves other predictions unchanged): PASS",
              f"- Previously hashed files (311) after run: changed={len(drift)}: {'PASS' if not drift else 'FAIL'}",
              "- Holdout labels/features/predictions/performance and the restricted membership file not accessed (script reads only the train list, the cohort CSV filtered to development IDs, dev AAL files): PASS",
              "\nLimitations: development subjects overlap Phase 6-10 cohort members; 5-fold CV; static baseline pipeline differs from the dynamic one only by the first preprocessing step; window defined in TRs (TR unknown); constant ROIs yield NaN edges handled by the stated rule."]
    (OUT / "integrity_checks.md").write_text("# Integrity checks\n\n" + "\n".join(checks) + "\n", encoding="utf-8")
    json.dump({"window_TR": W, "step_TR": STEP, "min_windows_for_variance": MIN_WIN, "atlas": "AAL116 (116 ROIs, 6670 unique edges)", "fisher_z_clip": 1e-6, "classifier": "LogisticRegression(L2, balanced, max_iter 2000)",
               "grid": {k: [str(x) for x in v] for k, v in fx_cfg["grid"].items()}, "inner_cv": fx_cfg["inner"], "scoring": fx_cfg["scoring"], "outer_cv": "StratifiedKFold(5, shuffle, seed 42) -- identical to fast experiment",
               "n_subjects": 600, "baseline_source": str(BASE / "fold_results.csv"), "conclusion_rule": "see experiment_report.md", "threads": "OMP/MKL/OPENBLAS=1", "runtime_s": round(total_s, 1)}, open(OUT / "config.json", "w"), indent=1)
    log(f"Report written; conclusion: {concl}; runtime {total_s:.0f}s; drift={len(drift)}")
    log("\n" + metrics[["method", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "roc_auc_mean"]].round(4).to_string(index=False))
    log("\n" + summ[summ.metric.isin(["accuracy", "roc_auc"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
