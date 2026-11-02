"""
run_tangent_aal116.py -- Ledoit-Wolf tangent-space connectivity on AAL-116 vs the frozen Fisher-Z baseline (protocol: experiment_protocol.md).
600 development subjects, the five fixed outer folds, inner-CV seeds 42/123/2026, plus one leave-one-site-out secondary pass.
Reuses tangent.py (unmodified). The geometric-mean reference is fitted on TRAINING subjects only: inside every inner-training split
and on the whole outer-training set. Holdout, restricted ID file, CC200 and the 780 cohort are never touched.
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
import psutil
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fast_accuracy_experiment as fx
import tangent as tg

ROOT = Path(r"C:\abide")
R = ROOT / "results"
BASE = R / "fast_accuracy_experiment"
OUT = R / "accuracy_research_v2"
for s in ("checkpoints", "logs", "data"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "tangent_aal116.log"
SEEDS = [42, 123, 2026]
GRID = [(C, k) for C in (1e-4, 1e-3, 1e-2, 1e-1) for k in (500, 2000, "all")]
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
BUDGET_S = 3000
PROC = psutil.Process()
PEAK = {"rss": 0}


def mem():
    PEAK["rss"] = max(PEAK["rss"], PROC.memory_info().rss)


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def make_pipe(C, k):
    return Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k)), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000))])


def embed(covs, tr, others):
    """Fit the tangent reference on covs[tr] only; return features for tr and for each index set in `others` (worker-safe)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        emb = tg.TangentEmbedding("tangent", precomputed=True).fit(covs[tr])
        feats = [emb.transform(covs[tr])] + [emb.transform(covs[o]) for o in others]
        nconv = sum(1 for x in w if "did not converge" in str(x.message))
    return feats, nconv


def sha_all():
    dirs = ["phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification", "phase3_6_aal_download", "phase4", "phase4_1_constant_roi_audit",
            "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic", "phase5b_missing_fc_analysis", "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero",
            "phase6_strategyC_all_nonzero", "phase7_synthesis", "phase8_final_audit", "phase9_accuracy_optimization", "phase10_advanced_accuracy"]
    files = []
    for d in dirs:
        if (R / d).exists():
            files += [f for f in (R / d).rglob("*") if f.is_file()]
    files += [f for f in ROOT.glob("phase*.py") if not f.name.startswith("phase10")] + [ROOT / "phase10_advanced_accuracy.py", ROOT / "tangent.py"]
    return {str(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(set(files))}


def nested_predict(covs, y, tr, te, seed, outer_cache=None):
    """Nested CV for one outer split: inner splits (seed) pick (C,k) by mean inner ROC-AUC; final model on the whole outer-train."""
    inner = list(StratifiedKFold(fx.N_INNER, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr]))
    tasks = [(tr[a], [tr[b]]) for a, b in inner]
    jobs = [delayed(embed)(covs, t, o) for t, o in tasks]
    if outer_cache is None:
        jobs.append(delayed(embed)(covs, tr, [te]))
    out = Parallel(n_jobs=min(4, len(jobs)), backend="loky")(jobs)
    nconv = sum(o[1] for o in out)
    mem()
    aucs = np.zeros(len(GRID))
    for (a, b), ((ftr, fva), _) in zip(inner, out[:len(inner)]):
        ya, yb = y[tr][a], y[tr][b]
        for gi, (C, k) in enumerate(GRID):
            p = make_pipe(C, k).fit(ftr, ya)
            aucs[gi] += roc_auc_score(yb, p.predict_proba(fva)[:, 1]) / len(inner)
    best = int(np.argmax(aucs)); C, k = GRID[best]
    if outer_cache is None:
        (ftr_o, fte_o), n2 = out[-1]; nconv += n2
        outer_cache = (ftr_o, fte_o)
    ftr_o, fte_o = outer_cache
    p = make_pipe(C, k).fit(ftr_o, y[tr])
    sc = p.predict_proba(fte_o)[:, 1]
    return [float(v) for v in sc], {"C": C, "k": k}, float(aucs[best]), nconv, outer_cache


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    log("=" * 60 + f"\nTANGENT AAL-116 EXPERIMENT start (smoke={smoke})")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    now = sha_all(); drift = [k for k, v in now.items() if k in pre and pre[k]["sha256"] != v]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}"); assert not drift
    dev, _ = fx.build_dev_set(); y = dev["y"].values; site = dev["site"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all() and (man["site"].values == site).all()
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds))
    cache = OUT / "data" / "covs_aal116_ledoit_wolf.npy"
    if cache.exists():
        covs = np.load(cache)
    else:
        files = {}
        for f in fx.AAL_DIR.glob("*_rois_aal.1D"):
            s = fx.subj_id_from_file(f.name, "rois_aal")
            if s in set(dev["sid"]):
                files[s] = f
        assert len(files) == 600
        covs = tg.covariances([np.loadtxt(str(files[s])) for s in dev["sid"]])
        np.save(cache, covs)
    w = np.linalg.eigvalsh(covs)
    log(f"Covariances {covs.shape}: symmetric={bool(np.allclose(covs, np.swapaxes(covs, 1, 2)))}, min eigenvalue {w.min():.2e} (all > 0: {bool((w > 0).all())}); condition number median {np.median(w[:, -1] / w[:, 0]):.1f}")
    assert (w > 0).all()
    # reference/test independence check on fold 0
    tr0, te0 = folds[0]
    e1 = tg.TangentEmbedding("tangent", precomputed=True).fit(covs[tr0])
    junk = covs.copy(); junk[te0] = np.eye(116) * 3.0
    e2 = tg.TangentEmbedding("tangent", precomputed=True).fit(junk[tr0])
    ok = bool(np.array_equal(e1.reference_, e2.reference_))
    log(f"Check: tangent reference depends only on training covariances (test covariances replaced by junk -> identical reference): {'PASS' if ok else 'FAIL'}"); assert ok
    t1 = time.time(); e1.transform(covs[te0]); log(f"transform of 120 subjects: {time.time() - t1:.1f}s")
    base = pd.read_csv(R / "accuracy_improvement_v1" / "fold_metrics.csv"); base = base[base.candidate == "E0_control_baseline"]
    rows, oof = [], []
    nconv_total = 0
    for seed in SEEDS:
        if smoke and seed != 42:
            continue
        for fold, (tr, te) in enumerate(folds):
            if smoke and fold > 0:
                break
            path = OUT / "checkpoints" / f"outer_seed{seed}_fold{fold}.json"
            ck = None if smoke else fx.load_ck(path)
            if ck is None:
                if time.time() - t0 > BUDGET_S:
                    log("time budget exceeded; stopping (partial results are not reported as final)"); return
                t2 = time.time()
                key = ("outer", fold)
                oc = _OUTER.get(key)
                sc, best, ia, nc, oc = nested_predict(covs, y, tr, te, seed, oc)
                _OUTER[key] = oc
                ck = {"seed": seed, "fold": fold, "test_idx": [int(i) for i in te], "score": sc, "best": best, "inner_auc": ia, "n_nonconverged_reference": nc, "seconds": round(time.time() - t2, 1)}
                if not smoke:
                    fx.save_ck(path, ck)
            nconv_total += ck["n_nonconverged_reference"]
            m = fx.metrics(y[te], (np.array(ck["score"]) > 0.5).astype(int), np.array(ck["score"]))
            b = base[(base.seed == seed) & (base.fold == fold)].iloc[0]
            log(f"[outer] seed {seed} fold {fold}: tangent acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} (baseline acc={b.accuracy:.4f} auc={b.roc_auc:.4f}) best={ck['best']} inner_auc={ck['inner_auc']:.4f} ({ck['seconds']}s)")
            rows.append({"seed": seed, "fold": fold, **m, "selected_C": ck["best"]["C"], "selected_k": ck["best"]["k"], "inner_cv_auc": ck["inner_auc"], "seconds": ck["seconds"]})
            for i, s_ in zip(ck["test_idx"], ck["score"]):
                oof.append({"seed": seed, "fold": fold, "project_id": f"DEV{i:04d}", "site": site[i], "y_true": int(y[i]), "score": s_, "y_pred": int(s_ > 0.5)})
    if smoke:
        log(f"SMOKE done in {time.time() - t0:.0f}s; peak RSS {PEAK['rss'] / 1e6:.0f} MB; non-converged references: {nconv_total}")
        return
    primary_report(dev, y, site, rows, oof, base, nconv_total, time.time() - t0, pre)
    lso(covs, dev, y, site, t0)
    finalize(pre, time.time() - t0)


_OUTER = {}


def primary_report(dev, y, site, rows, oof, base, nconv, total_s, pre):
    fm = pd.DataFrame(rows); fm.to_csv(OUT / "fold_metrics_tangent.csv", index=False)
    o = pd.DataFrame(oof); o.to_csv(OUT / "oof_predictions_tangent.csv", index=False)
    assert all(g.project_id.is_unique and len(g) == 600 for _, g in o.groupby("seed"))
    summ, verdict = [], {}
    for seed in SEEDS:
        a = fm[fm.seed == seed].sort_values("fold"); b = base[base.seed == seed].sort_values("fold")
        for name, g in (("BASELINE_FisherZ_Pearson", b), ("TANGENT_LedoitWolf", a)):
            summ.append({"representation": name, "seed": seed, **{f"{m}_mean": float(g[m].mean()) for m in MET}, **{f"{m}_sd": float(g[m].std(ddof=1)) for m in MET},
                         "tn": int(g.tn.sum()), "fp": int(g.fp.sum()), "fn": int(g.fn.sum()), "tp": int(g.tp.sum())})
        d = {m: a[m].values - b[m].values for m in MET}
        verdict[seed] = {**{f"{m}_diff": float(d[m].mean()) for m in MET}, "acc_diff_sd": float(d["accuracy"].std(ddof=1)), "folds_acc_improved": int((d["accuracy"] > 0).sum()),
                         "per_fold_acc_diff": [round(float(v), 4) for v in d["accuracy"]], "per_fold_auc_diff": [round(float(v), 4) for v in d["roc_auc"]],
                         "seed_meets_gain": bool(d["accuracy"].mean() >= 0.02 and d["balanced_accuracy"].mean() >= 0.02 and (d["accuracy"] > 0).sum() >= 4)}
    sdf = pd.DataFrame(summ); sdf.to_csv(OUT / "summary_by_seed.csv", index=False)
    avg = lambda key: float(np.mean([verdict[s][key] for s in SEEDS]))
    crit = {"1_seed_avg_acc_and_bacc_gain_ge_0.02": bool(avg("accuracy_diff") >= 0.02 and avg("balanced_accuracy_diff") >= 0.02),
            "2_at_least_2_seeds_meet_gain_and_4of5_folds": bool(sum(verdict[s]["seed_meets_gain"] for s in SEEDS) >= 2),
            "3_no_material_degradation_auc_ge_-0.01_sens_spec_ge_-0.03": bool(avg("roc_auc_diff") >= -0.01 and avg("sensitivity_diff") >= -0.03 and avg("specificity_diff") >= -0.03)}
    passed = all(crit.values())
    json.dump({"criteria": crit, "all_criteria_met": passed, "verdict_by_seed": verdict, "seed_avg_diff": {m: avg(f"{m}_diff") for m in MET}, "non_converged_references": nconv,
               "peak_rss_mb": PEAK["rss"] / 1e6, "primary_runtime_s": round(total_s, 1)}, open(OUT / "primary_result.json", "w"), indent=1)
    log(f"PRIMARY: criteria {crit}; improvement established: {passed}")
    log("\n" + sdf[["representation", "seed", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "roc_auc_mean"]].round(4).to_string(index=False))
    log(json.dumps({s: {k: v for k, v in verdict[s].items() if k in ("accuracy_diff", "balanced_accuracy_diff", "roc_auc_diff", "folds_acc_improved", "seed_meets_gain")} for s in SEEDS}, default=float))


def lso(covs, dev, y, site, t0):
    log("Secondary: leave-one-site-out (seed 42; inductive; not used for decisions)")
    rows = []
    for s in sorted(set(site), key=lambda s: -(site == s).sum()):
        path = OUT / "checkpoints" / f"lso_{s}.json"
        ck = fx.load_ck(path)
        te = np.where(site == s)[0]; tr = np.where(site != s)[0]
        assert not (set(site[tr]) & set(site[te]))
        if ck is None:
            if time.time() - t0 > BUDGET_S:
                log("time budget exceeded in leave-site-out; stopping"); return
            t1 = time.time(); sc, best, ia, nc, _ = nested_predict(covs, y, tr, te, 42, None)
            ck = {"site": s, "test_idx": [int(i) for i in te], "score": sc, "best": best, "inner_auc": ia, "n_nonconverged_reference": nc, "seconds": round(time.time() - t1, 1)}
            fx.save_ck(path, ck)
        log(f"[lso] {s}: n={len(te)} best={ck['best']} ({ck['seconds']}s)")
        for i, s_ in zip(ck["test_idx"], ck["score"]):
            rows.append({"project_id": f"DEV{i:04d}", "site": s, "y_true": int(y[i]), "score": s_, "y_pred": int(s_ > 0.5)})
    o = pd.DataFrame(rows).sort_values("project_id"); o.to_csv(OUT / "leave_site_out_oof_predictions_tangent.csv", index=False)
    m = fx.metrics(o.y_true.values, o.y_pred.values, o.score.values)
    fz = pd.read_csv(R / "leave_site_out_baseline" / "oof_predictions.csv"); mb = fx.metrics(fz.y_true.values, fz.y_pred.values, fz.score.values)
    per = []
    fzs = fz.set_index("project_id")
    for s, g in o.groupby("site"):
        gb = fzs.loc[g.project_id]
        per.append({"site": s, "n": len(g), "small_site_n_lt_20": len(g) < 20, "accuracy_tangent": float((g.y_true == g.y_pred).mean()), "accuracy_baseline": float((gb.y_true == gb.y_pred).mean())})
    pd.DataFrame(per).to_csv(OUT / "leave_site_out_per_site.csv", index=False)
    pd.DataFrame([{"representation": "TANGENT_LedoitWolf (leave-site-out)", **{k: m[k] for k in MET}}, {"representation": "BASELINE_FisherZ_Pearson (leave-site-out, saved)", **{k: mb[k] for k in MET}}]).to_csv(OUT / "leave_site_out_summary.csv", index=False)
    log(f"LSO pooled: tangent acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} | baseline acc={mb['accuracy']:.4f} auc={mb['roc_auc']:.4f}")


def finalize(pre, total_s):
    now = sha_all(); drift = [k for k in pre if k in now and pre[k]["sha256"] != now[k]]
    cfg = json.load(open(OUT / "primary_result.json"))
    json.dump({"representation": "per-subject z-score -> Ledoit-Wolf covariance -> tangent space (geometric-mean reference fitted on training subjects only) -> 6,670 off-diagonal features (tangent.py)",
               "model": "StandardScaler -> SelectKBest -> L2 logistic regression (balanced)", "grid": [[C, str(k)] for C, k in GRID], "seeds": SEEDS, "outer_cv": "StratifiedKFold(5, shuffle, seed 42) identical to the baseline",
               "inner_cv": "StratifiedKFold(3, shuffle, seed)", "baseline": "accuracy_improvement_v1 control E0 (reproduces the frozen baseline exactly at seed 42)", "protected_files_changed": len(drift), "total_runtime_s": round(total_s, 1),
               "peak_rss_mb": PEAK["rss"] / 1e6, "threads": "OMP/MKL/OPENBLAS=1; 4 loky workers for reference fitting"}, open(OUT / "config.json", "w"), indent=1)
    log(f"Finalized; protected files (incl. tangent.py) changed={len(drift)}; total runtime {total_s:.0f}s; peak RSS {PEAK['rss'] / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
