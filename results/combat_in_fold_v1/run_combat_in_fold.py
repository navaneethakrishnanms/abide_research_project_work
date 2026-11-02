"""combat_in_fold_v1 -- ONE predeclared experiment (see protocol.md): in-fold ComBat-style site harmonization before the frozen AAL-116
L2 logistic-regression baseline. 600 development subjects, outer folds and inner seeds identical to accuracy_improvement_v1 control E0.
Usage:  python run_combat_in_fold.py [--smoke]     (--smoke: self-test + seed 42 fold 0 only, no report)
Restricted holdout, its ID file, CC200 and the 780 cohort are never opened."""
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"C:\abide")
sys.path.insert(1, str(HERE))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import fast_accuracy_experiment as fx
from combat_harmonizer import HarmFZ

ROOT = Path(r"C:\abide"); R = ROOT / "results"; BASE = R / "fast_accuracy_experiment"; OUT = HERE
for s in ("checkpoints", "logs"):
    (OUT / s).mkdir(exist_ok=True)
LOG = OUT / "logs" / "combat_in_fold_v1.log"
SEEDS = [42, 123, 2026]
GRID = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1], "sel__k": [500, 2000, "all"]}
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
BUDGET_S = 2400
NAME = "H_incombat_age_sex"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def lr():
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def make_pipe():
    base = fx.pipe(lr(), None)                      # baseline: fz, sc, sel, clf
    return Pipeline([("fz", HarmFZ())] + base.steps[1:])


def eta2_site(Z, site):
    mu = Z.mean(0); tot = ((Z - mu) ** 2).sum(0)
    bet = sum((site == g).sum() * (Z[site == g].mean(0) - mu) ** 2 for g in np.unique(site))
    ok = tot > 1e-10
    return float((bet[ok] / tot[ok]).mean())


def self_test():
    """Synthetic checks only (no real data): planted site effects are removed on held-out rows; held-out rows cannot influence the fit or each other."""
    rng = np.random.default_rng(0)
    n, p, ns = 400, 60, 5
    site = rng.integers(0, ns, n); age = rng.normal(15, 5, n); fem = rng.integers(0, 2, n).astype(float)
    shift = rng.normal(0, 0.6, (ns, p)); scale = rng.uniform(0.7, 1.5, (ns, p))
    Z = rng.normal(0, 0.2, (n, p)) * scale[site] + shift[site] + 0.01 * (age[:, None] - 15) * np.linspace(-1, 1, p)
    X = np.hstack([np.tanh(Z), site[:, None], age[:, None], fem[:, None]])
    tr = np.arange(n) < 300; te = ~tr
    h = HarmFZ().fit(X[tr])
    before = eta2_site(h.fz_.transform(X[te, :-3]), site[te]); after = eta2_site(h.transform(X[te]), site[te])
    assert after < 0.2 * before, (before, after)
    g0 = h.cp_.gamma_.copy(); out = h.transform(X[te]); assert np.array_equal(g0, h.cp_.gamma_)
    X2 = X[te].copy(); X2[:10, :-3] += 5.0; X2[:10, -3] = (X2[:10, -3] + 1) % ns; X2[:10, -2] += 30
    out2 = h.transform(X2); assert np.array_equal(out[10:], out2[10:]) and not np.array_equal(out[:10], out2[:10])
    h2 = HarmFZ().fit(X[tr][:, :]); assert np.array_equal(h.cp_.gamma_, h2.cp_.gamma_)     # deterministic, uses only given rows
    Xs = X[tr].copy(); Xs[:, -3] = np.where(np.arange(300) < 3, 99, Xs[:, -3])               # tiny site -> left unchanged, no crash
    hs = HarmFZ().fit(Xs); Xt = X[te].copy(); Xt[:5, -3] = 99
    assert np.allclose(hs.transform(Xt)[:5], hs.fz_.transform(Xt[:5, :-3]))                  # unseen/small site: identity
    log(f"self-test PASS: held-out between-site variance share {before:.3f} -> {after:.3f}; test-row perturbation invariance; unseen site left unchanged")


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
    smoke = "--smoke" in sys.argv
    t0 = time.time(); nwarn = [0]
    log("=" * 60 + f"\ncombat_in_fold_v1 start{' (SMOKE)' if smoke else ''}")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    drift = [k for k, v in sha_all().items() if k in pre and pre[k]["sha256"] != v]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}"); assert not drift
    self_test()
    dev, _ = fx.build_dev_set(); y = dev["y"].values; sites = dev["site"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all() and (man["site"].values == sites).all()
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds))
    coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str}); coh["sid"] = coh.subject_id.map(fx.norm_id)
    coh = coh[coh.sid.isin(set(man.sid))].set_index("sid").loc[man.sid]
    ph = pd.read_csv(ROOT / "Phenotypic_V1_0b.csv"); ph["sid"] = ph.SUB_ID.astype(int).astype(str); ph = ph.set_index("sid").loc[man.sid]
    age = coh.age.values.astype(float); fem = (coh.sex.values == 2).astype(float)
    assert np.isfinite(age).all() and np.abs(age - ph.AGE_AT_SCAN.values.astype(float)).max() < 1e-6 and ((ph.SEX.values == 2).astype(float) == fem).all()
    assert ((ph.DX_GROUP.values == 1).astype(int) == y).all() and (ph.SITE_ID.str.upper().values == np.char.upper(sites.astype(str))).all()
    site_code = {s: i for i, s in enumerate(sorted(set(sites)))}
    X = fx.load_fc(dev, "AAL116"); assert X.shape == (600, 6670)
    Xa = np.hstack([X, np.array([site_code[s] for s in sites], float)[:, None], age[:, None], fem[:, None]])
    log(f"600 dev subjects ({int(y.sum())} ASD / {int((1 - y).sum())} TC), {len(site_code)} sites; labels, sites, age, sex aligned with manifest and phenotype file; outer folds identical to E0")
    prim = []; diag = []
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        for seed in SEEDS:
            for fold, (tr, te) in enumerate(folds):
                path = OUT / "checkpoints" / f"outer_seed{seed}_fold{fold}.json"
                ck = fx.load_ck(path)
                if ck is None:
                    if time.time() - t0 > BUDGET_S:
                        log("time budget exceeded; stopping"); return
                    t1 = time.time()
                    gs = GridSearchCV(make_pipe(), GRID, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=seed), n_jobs=6, refit=True, error_score="raise")
                    gs.fit(Xa[tr], y[tr])
                    sc = gs.best_estimator_.predict_proba(Xa[te])[:, 1]
                    ck = {"seed": seed, "fold": fold, "test_idx": [int(i) for i in te], "best": {k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in gs.best_params_.items()},
                          "inner_auc": float(gs.best_score_), "score": [float(v) for v in sc], "seconds": round(time.time() - t1, 1)}
                    fx.save_ck(path, ck)
                log(f"[outer] seed {seed} fold {fold}: best {ck['best']} inner AUC {ck['inner_auc']:.4f} ({ck['seconds']}s)")
                prim.append(ck)
                if smoke:
                    log(f"SMOKE done in {time.time() - t0:.0f}s"); return
    nwarn[0] = len(wl)
    for fold, (tr, te) in enumerate(folds):                      # diagnostic: between-site variance share on TRAINING rows, before vs after
        h = HarmFZ().fit(Xa[tr]); Zr = h.fz_.transform(Xa[tr][:, :-3]); Za = h.transform(Xa[tr])
        diag.append({"fold": fold, "n_train": len(tr), "eta2_site_before": eta2_site(Zr, sites[tr]), "eta2_site_after": eta2_site(Za, sites[tr])})
    report(y, sites, prim, pd.DataFrame(diag), time.time() - t0, pre, nwarn[0], sorted({str(w.category.__name__) for w in wl}))


def report(y, sites, prim, diag, total_s, pre, nwarn, wcats):
    rows, oof = [], []
    for ck in prim:
        te = np.array(ck["test_idx"]); sc = np.array(ck["score"]); pr = (sc > 0.5).astype(int)
        rows.append({"candidate": NAME, "seed": ck["seed"], "fold": ck["fold"], "threshold": 0.5, "selected_C": ck["best"]["clf__C"], "selected_k": ck["best"]["sel__k"], **fx.metrics(y[te], pr, sc)})
        for i, s_, p_ in zip(te, sc, pr):
            oof.append({"candidate": NAME, "seed": ck["seed"], "fold": ck["fold"], "project_id": f"DEV{i:04d}", "site": sites[i], "y_true": int(y[i]), "score": float(s_), "threshold": 0.5, "y_pred": int(p_)})
    fm = pd.DataFrame(rows); fm.to_csv(OUT / "fold_metrics.csv", index=False); pd.DataFrame(oof).to_csv(OUT / "oof_predictions.csv", index=False)
    diag.to_csv(OUT / "site_variance_diagnostic.csv", index=False)
    e0 = pd.read_csv(R / "accuracy_improvement_v1" / "fold_metrics.csv"); e0 = e0[e0.candidate == "E0_control_baseline"]
    comp = []
    for seed in SEEDS:
        h = fm[fm.seed == seed].sort_values("fold"); b = e0[e0.seed == seed].sort_values("fold")
        assert len(h) == 5 and len(b) == 5
        r = {"seed": seed}
        for m in MET:
            r[f"E0_{m}"] = float(b[m].mean()); r[f"H_{m}"] = float(h[m].mean()); d = h[m].values - b[m].values
            r[f"{m}_diff"] = float(d.mean()); r[f"{m}_folds_higher"] = int((d > 0).sum())
        r["rule_met"] = bool(r["accuracy_diff"] >= 0.02 and r["balanced_accuracy_diff"] >= 0.02 and r["accuracy_folds_higher"] >= 4)
        comp.append(r)
    cdf = pd.DataFrame(comp); cdf.to_csv(OUT / "comparison_vs_E0.csv", index=False)
    improves = int(cdf.rule_met.sum()) >= 2
    now = sha_all(); drift = [k for k in pre if k in now and pre[k]["sha256"] != now[k]]
    cfg = {"candidate": NAME, "rule": "mean accuracy and balanced accuracy >= +0.02 vs E0 (same seed), accuracy higher in >=4/5 folds, in >=2 of 3 seeds",
           "seeds_meeting_rule": int(cdf.rule_met.sum()), "improves": improves, "protected_files_changed": len(drift), "runtime_s": round(total_s, 1), "parent_process_warnings": nwarn, "warning_categories": wcats}
    json.dump(cfg, open(OUT / "config.json", "w"), indent=1)
    log(f"Outputs written; protected files changed={len(drift)}; runtime {total_s:.0f}s; warnings in parent process: {nwarn} {wcats}")
    log("\n" + cdf[["seed", "E0_accuracy", "H_accuracy", "accuracy_diff", "accuracy_folds_higher", "E0_balanced_accuracy", "H_balanced_accuracy", "balanced_accuracy_diff", "E0_roc_auc", "H_roc_auc", "rule_met"]].round(4).to_string(index=False))
    log("\n" + diag.round(4).to_string(index=False)); log(f"VERDICT: {'IMPROVES (rule met)' if improves else 'NO IMPROVEMENT (rule not met)'} -- seeds meeting rule: {int(cdf.rule_met.sum())}/3")


if __name__ == "__main__":
    main()
