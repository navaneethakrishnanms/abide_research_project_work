"""Experiment 1 of accuracy_research_v3 (protocol: experiment_protocol.md). Arms B, B2, C, D vs the saved baseline A, on the 600 development
subjects and the five fixed outer folds, inner-CV seeds 42/123/2026. All fitted steps use training rows only. Restricted data, CC200 and
the 780 cohort are never touched."""
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
import psutil
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
R = ROOT / "results"
BASE = R / "fast_accuracy_experiment"
OUT = R / "accuracy_research_v3"
for s in ("checkpoints", "logs", "data"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "demographics_arms.log"
SEEDS = [42, 123, 2026]
NFC = 6670
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
BUDGET_S = 2400
PROC = psutil.Process(); PEAK = {"rss": 0}


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


class FCDemo(ClassifierMixin, BaseEstimator):
    """X = [6,670 raw FC edges | age | female]. Everything is fitted on the rows passed to fit()."""

    def __init__(self, C=1e-3, k=2000, w=1.0):
        self.C, self.k, self.w = C, k, w

    def _demo(self, X):
        return (X[:, NFC:] - self.dmu_) / self.dsd_ * self.w

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fz_ = fx.MeanImputeFisherZ().fit(X[:, :NFC])
        Z = self.fz_.transform(X[:, :NFC])
        self.sc_ = StandardScaler().fit(Z)
        self.sel_ = SelectKBest(f_classif, k=self.k).fit(self.sc_.transform(Z), y)
        self.dmu_ = X[:, NFC:].mean(axis=0); self.dsd_ = X[:, NFC:].std(axis=0) + 1e-12
        F = np.hstack([self.sel_.transform(self.sc_.transform(Z)), self._demo(X)])
        self.lr_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000).fit(F, y)
        return self

    def predict_proba(self, X):
        Z = self.fz_.transform(X[:, :NFC])
        return self.lr_.predict_proba(np.hstack([self.sel_.transform(self.sc_.transform(Z)), self._demo(X)]))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DemoOnly(ClassifierMixin, BaseEstimator):
    """X = demographic (and optionally site one-hot) columns; standardized on training rows; L2 logistic regression."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mu_ = X.mean(axis=0); self.sd_ = X.std(axis=0) + 1e-12
        self.lr_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000).fit((X - self.mu_) / self.sd_, y)
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba((X - self.mu_) / self.sd_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def tuned(est, grid, Xtr, ytr, Xte, seed, nj=6):
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=seed), n_jobs=nj, refit=True, error_score="raise")
    gs.fit(Xtr, ytr)
    return gs.best_estimator_.predict_proba(Xte)[:, 1], {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in gs.best_params_.items()}


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
    t0 = time.time()
    log("=" * 60 + f"\nDEMOGRAPHIC ARMS start (smoke={smoke})")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    drift = [k for k, v in sha_all().items() if k in pre and pre[k]["sha256"] != v]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}"); assert not drift
    dev, _ = fx.build_dev_set(); y = dev["y"].values; site = dev["site"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all() and (man["site"].values == site).all()
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds))
    coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str}); coh["sid"] = coh.subject_id.map(fx.norm_id)
    coh = coh[coh.sid.isin(set(dev.sid))].set_index("sid").loc[dev.sid]
    age = coh.age.values.astype(float); female = (coh.sex.values == 2).astype(float)
    assert np.isfinite(age).all() and set(np.unique(female)) <= {0.0, 1.0}
    X_fc = fx.load_fc(dev, "AAL116")
    X_full = np.hstack([X_fc, age[:, None], female[:, None]])
    X_demo = np.column_stack([age, female])
    onehot = pd.get_dummies(pd.Series(site)).values.astype(float)
    X_demo_site = np.hstack([X_demo, onehot])
    base = pd.read_csv(R / "accuracy_improvement_v1" / "fold_metrics.csv"); base = base[base.candidate == "E0_control_baseline"]
    log(f"Alignment: 600 subjects; age range {age.min():.1f}-{age.max():.1f}; female share {100 * female.mean():.1f}%; feature matrix {X_full.shape}; baseline rows {len(base)}")
    # sanity: arm C pipeline with the demographic columns removed from the weight (w=0) equals the baseline pipeline type? (check on fold 0 that w=0 gives a valid model)
    tr0, te0 = folds[0]
    m = FCDemo(C=1e-3, k=2000, w=1.0).fit(X_full[tr0], y[tr0]); p1 = m.predict_proba(X_full[te0])[:, 1]
    Xj = X_full[te0].copy(); half = np.arange(len(te0)) % 2 == 1; Xj[half] = np.random.default_rng(0).uniform(-0.9, 0.9, Xj[half].shape); Xj[half, NFC] = 99.0
    ok = bool(np.allclose(p1[~half], m.predict_proba(Xj)[:, 1][~half], atol=1e-12)); log(f"Check: test rows do not influence one another or the fit: {'PASS' if ok else 'FAIL'}"); assert ok
    grid_D = {"C": [1e-4, 1e-3, 1e-2, 1e-1], "k": [500, 2000, "all"], "w": [1.0, 10.0]}
    grid_DEMO = {"C": [0.01, 0.1, 1.0, 10.0]}
    # "all" for SelectKBest: map in a thin wrapper
    rows, oof = [], []
    seeds = [42] if smoke else SEEDS
    for seed in seeds:
        for fold, (tr, te) in enumerate(folds):
            if smoke and fold > 0:
                break
            path = OUT / "checkpoints" / f"seed{seed}_fold{fold}.json"
            ck = None if smoke else fx.load_ck(path)
            if ck is None:
                if time.time() - t0 > BUDGET_S:
                    log("time budget exceeded; stopping"); return
                t1 = time.time(); res = {}
                res["B_age_sex"] = tuned(DemoOnly(), grid_DEMO, X_demo[tr], y[tr], X_demo[te], seed)
                res["B2_age_sex_site"] = tuned(DemoOnly(), grid_DEMO, X_demo_site[tr], y[tr], X_demo_site[te], seed)
                res["C_fc_age_sex_fixed"] = (FCDemo(1e-3, 2000, 1.0).fit(X_full[tr], y[tr]).predict_proba(X_full[te])[:, 1], {"C": 1e-3, "k": 2000, "w": 1.0})
                res["D_fc_age_sex_tuned"] = tuned(FCDemo(), {**grid_D, "k": [500, 2000, X_fc.shape[1]]}, X_full[tr], y[tr], X_full[te], seed)
                PEAK["rss"] = max(PEAK["rss"], PROC.memory_info().rss)
                ck = {"seed": seed, "fold": fold, "test_idx": [int(i) for i in te], "res": {k: {"score": [float(v) for v in s], "best": b} for k, (s, b) in res.items()}, "seconds": round(time.time() - t1, 1)}
                if not smoke:
                    fx.save_ck(path, ck)
            te_ = np.array(ck["test_idx"])
            line = []
            for arm, r in ck["res"].items():
                sc = np.array(r["score"]); mm = fx.metrics(y[te_], (sc > 0.5).astype(int), sc)
                rows.append({"arm": arm, "seed": seed, "fold": fold, **mm, "best": json.dumps(r["best"])})
                line.append(f"{arm.split('_')[0]} acc={mm['accuracy']:.3f}/auc={mm['roc_auc']:.3f}")
                for i, s_ in zip(te_, sc):
                    oof.append({"arm": arm, "seed": seed, "fold": fold, "project_id": f"DEV{i:04d}", "site": site[i], "y_true": int(y[i]), "score": float(s_), "y_pred": int(s_ > 0.5)})
            b = base[(base.seed == seed) & (base.fold == fold)].iloc[0]
            log(f"[seed {seed} fold {fold}] " + " | ".join(line) + f" | A acc={b.accuracy:.3f}/auc={b.roc_auc:.3f} ({ck['seconds']}s)")
    if smoke:
        log(f"SMOKE done in {time.time() - t0:.0f}s; peak RSS {PEAK['rss'] / 1e6:.0f} MB"); return
    report(dev, y, site, rows, oof, base, time.time() - t0, pre)


def report(dev, y, site, rows, oof, base, total_s, pre):
    fm = pd.DataFrame(rows)
    baseA = base[["seed", "fold"] + MET + ["tn", "fp", "fn", "tp"]].assign(arm="A_baseline_FC_only", best="")
    allf = pd.concat([baseA, fm], ignore_index=True); allf.to_csv(OUT / "metrics_by_fold.csv", index=False)
    o = pd.DataFrame(oof); o.to_csv(OUT / "oof_predictions.csv", index=False)
    assert all(g.project_id.is_unique and len(g) == 600 for _, g in o.groupby(["arm", "seed"]))
    summ = []
    for arm, g in allf.groupby("arm"):
        for seed, gs in g.groupby("seed"):
            summ.append({"arm": arm, "seed": seed, **{f"{m}_mean": float(gs[m].mean()) for m in MET}, **{f"{m}_sd": float(gs[m].std(ddof=1)) for m in MET}, "tn": int(gs.tn.sum()), "fp": int(gs.fp.sum()), "fn": int(gs.fn.sum()), "tp": int(gs.tp.sum())})
    sdf = pd.DataFrame(summ); sdf.to_csv(OUT / "metrics_by_seed.csv", index=False)
    verdict = {}
    A = allf[allf.arm == "A_baseline_FC_only"]
    for arm in ("C_fc_age_sex_fixed", "D_fc_age_sex_tuned"):
        per = {}
        for seed in SEEDS:
            a = allf[(allf.arm == arm) & (allf.seed == seed)].sort_values("fold"); b = A[A.seed == seed].sort_values("fold")
            d = {m: a[m].values - b[m].values for m in MET}
            per[seed] = {**{f"{m}_diff": float(d[m].mean()) for m in MET}, "folds_acc_improved": int((d["accuracy"] > 0).sum()), "per_fold_acc_diff": [round(float(v), 4) for v in d["accuracy"]],
                         "meets_gain": bool(d["accuracy"].mean() >= 0.02 and d["balanced_accuracy"].mean() >= 0.02 and (d["accuracy"] > 0).sum() >= 4)}
        av = lambda k: float(np.mean([per[s][k] for s in SEEDS]))
        crit = {"1_seed_avg_acc_bacc_ge_0.02": bool(av("accuracy_diff") >= 0.02 and av("balanced_accuracy_diff") >= 0.02), "2_two_seeds_meet_gain_and_4of5_folds": bool(sum(per[s]["meets_gain"] for s in SEEDS) >= 2),
                "3_no_degradation": bool(av("roc_auc_diff") >= -0.01 and av("sensitivity_diff") >= -0.03 and av("specificity_diff") >= -0.03)}
        verdict[arm] = {"criteria": crit, "established": all(crit.values()), "seed_avg": {m: av(f"{m}_diff") for m in MET}, "by_seed": per}
    # confounding diagnostics: pooled AUC vs mean within-site AUC; per-site accuracy
    from sklearn.metrics import roc_auc_score
    conf, per_site = [], []
    arms = ["A_baseline_FC_only"] + [a for a in o.arm.unique()]
    base_oof = pd.read_csv(R / "accuracy_improvement_v1" / "oof_predictions.csv"); base_oof = base_oof[base_oof.candidate == "E0_control_baseline"]
    base_oof = base_oof.assign(arm="A_baseline_FC_only")[["arm", "seed", "fold", "project_id", "site", "y_true", "score", "y_pred"]]
    oo = pd.concat([base_oof, o], ignore_index=True)
    for (arm, seed), g in oo.groupby(["arm", "seed"]):
        within = [roc_auc_score(gs.y_true, gs.score) for _, gs in g.groupby("site") if len(gs) >= 20 and gs.y_true.nunique() == 2]
        conf.append({"arm": arm, "seed": seed, "pooled_auc": float(roc_auc_score(g.y_true, g.score)), "mean_within_site_auc_n_ge_20": float(np.mean(within)), "accuracy": float((g.y_true == g.y_pred).mean()),
                     "majority_class_rate": float(1 - g.y_true.mean()), "predicted_ASD_share": float(g.y_pred.mean())})
        for s, gs in g.groupby("site"):
            per_site.append({"arm": arm, "seed": seed, "site": s, "n": len(gs), "small_site_n_lt_20": len(gs) < 20, "accuracy": float((gs.y_true == gs.y_pred).mean())})
    cdf = pd.DataFrame(conf); cdf.to_csv(OUT / "confounding_diagnostics.csv", index=False)
    psd = pd.DataFrame(per_site); psd.to_csv(OUT / "per_site_accuracy_by_arm.csv", index=False)
    now = sha_all(); drift = [k for k in pre if k in now and pre[k]["sha256"] != now[k]]
    json.dump({"verdict": verdict, "arms": arms, "seeds": SEEDS, "grid_D": {"C": [1e-4, 1e-3, 1e-2, 1e-1], "k": ["500", "2000", "all"], "w": [1.0, 10.0]}, "fixed_C": {"C": 1e-3, "k": 2000, "w": 1.0},
               "outer_cv": "StratifiedKFold(5, shuffle, seed 42) identical to the baseline", "protected_files_changed": len(drift), "runtime_s": round(total_s, 1), "peak_rss_mb": PEAK["rss"] / 1e6}, open(OUT / "config.json", "w"), indent=1, default=float)
    log(f"Outputs written; verdicts: " + json.dumps({a: v["established"] for a, v in verdict.items()}) + f"; protected files changed={len(drift)}; runtime {total_s:.0f}s; peak RSS {PEAK['rss'] / 1e6:.0f} MB")
    log("\n" + sdf[["arm", "seed", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "roc_auc_mean"]].round(4).to_string(index=False))
    log("\n" + cdf.round(4).to_string(index=False))
    log(json.dumps({a: {"criteria": v["criteria"], "seed_avg_acc": v["seed_avg"]["accuracy"], "seed_avg_bacc": v["seed_avg"]["balanced_accuracy"], "seed_avg_auc": v["seed_avg"]["roc_auc"]} for a, v in verdict.items()}, default=float))


if __name__ == "__main__":
    main()
