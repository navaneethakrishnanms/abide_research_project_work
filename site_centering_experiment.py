"""
site_centering_experiment.py -- paired test of TRAINING-FOLD-ONLY site centering (development subjects only).

Reuses fast_accuracy_experiment.py (dev-set definition, cached feature matrices, outer folds, model grids, metrics).
The locked holdout file is never opened. Baselines (uncentered) are read from the completed fast experiment
(identical subjects, folds, grids, code path), results are NOT recomputed or overwritten.

Site centering (SiteFisherZCenter): inside every pipeline fit -- including each inner-training split of the grid
search -- column means are learned on the fitted rows, Fisher-Z is applied, and for every site with >= MIN_SITE_N
fitted rows an offset (site mean - global mean) is stored. transform() subtracts the stored offset for a row's site
(zero offset for sites that are absent from, or too small in, the fitted rows). No statistic is ever computed from
validation/test rows.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
BASE_DIR = ROOT / "results" / "fast_accuracy_experiment"
OUT = ROOT / "results" / "site_centering_experiment"
for s in ("checkpoints", "logs"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "site_centering.log"
MIN_SITE_N = 5
TOTAL_BUDGET_S = 300
CANDIDATES = [("AAL116", "LR_L2"), ("AAL116", "LinearSVM"), ("CC200", "RBF_SVM")]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}\n")
    print(msg, flush=True)


class SiteFisherZCenter(BaseEstimator, TransformerMixin):
    """Input: raw FC features with the integer site code appended as the LAST column."""

    def __init__(self, min_site_n=MIN_SITE_N):
        self.min_site_n = min_site_n

    def _z(self, F):
        F = np.where(np.isnan(F), self.means_[None, :], F)
        return np.arctanh(np.clip(F, -1 + 1e-6, 1 - 1e-6))

    def fit(self, X, y=None):
        F, site = X[:, :-1], X[:, -1].astype(int)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmean(F, axis=0)
        self.means_ = np.where(np.isnan(m), 0.0, m)
        Z = self._z(F)
        gmean = Z.mean(axis=0)
        self.offsets_ = {int(s): Z[site == s].mean(axis=0) - gmean for s in np.unique(site) if (site == s).sum() >= self.min_site_n}
        return self

    def transform(self, X):
        F, site = X[:, :-1], X[:, -1].astype(int)
        Z = self._z(F)
        for s in np.unique(site):
            off = self.offsets_.get(int(s))
            if off is not None:
                Z[site == s] -= off
        return Z


def pipe(clf):
    return Pipeline([("fzc", SiteFisherZCenter()), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)), ("clf", clf)])


def run_fold(atlas, model, fold, Xs, y, tr, te):
    make, grid = fx.MODELS[model]
    gs = GridSearchCV(pipe(make()), grid, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=fx.SEED),
                      n_jobs=6, refit=True, error_score="raise")
    t0 = time.time()
    gs.fit(Xs[tr], y[tr])
    est = gs.best_estimator_
    score = est.predict_proba(Xs[te])[:, 1] if hasattr(est, "predict_proba") else est.decision_function(Xs[te])
    pred = est.predict(Xs[te])
    n_off = len(est.named_steps["fzc"].offsets_)
    return {"atlas": atlas, "model": model, "variant": "site_centered", "fold": fold, "metrics": fx.metrics(y[te], pred, score),
            "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in gs.best_params_.items()},
            "inner_cv_auc": float(gs.best_score_), "sites_with_offsets": n_off, "seconds": round(time.time() - t0, 1)}


def leakage_check(Xs, y, tr, te, site_codes):
    p = pipe(fx.LogisticRegression(class_weight="balanced", max_iter=2000)).set_params(sel__k=500, clf__C=1e-3).fit(Xs[tr], y[tr])
    base = p.predict_proba(Xs[te])[:, 1]
    Xj = Xs[te].copy()
    half = np.arange(len(te)) % 2 == 1
    Xj[half, :-1] = np.random.default_rng(0).uniform(-0.9, 0.9, Xj[half, :-1].shape)   # scramble half of the TEST rows' features
    alt = p.predict_proba(Xj)[:, 1]
    # independent recomputation of one site's offset from TRAINING rows only must equal the fitted offset
    step = p.named_steps["fzc"]
    F = Xs[tr][:, :-1]
    Z = np.arctanh(np.clip(np.where(np.isnan(F), step.means_[None, :], F), -1 + 1e-6, 1 - 1e-6))
    s0 = next(iter(step.offsets_))
    manual = Z[Xs[tr][:, -1].astype(int) == s0].mean(axis=0) - Z.mean(axis=0)
    return bool(np.allclose(base[~half], alt[~half], atol=1e-12) and np.allclose(manual, step.offsets_[s0], atol=1e-12))


def main():
    t_start = time.time()
    log("=" * 60 + "\nSITE-CENTERING EXPERIMENT start")
    dev, _ = fx.build_dev_set()
    y = dev["y"].values
    manifest = pd.read_csv(BASE_DIR / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(manifest["sid"]) == list(dev["sid"]) and (manifest["y"].values == y).all(), "dev set differs from the fast experiment's manifest"
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(manifest["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds)), "fold assignments differ from the fast experiment"
    log(f"Dev set {len(dev)} subjects ({int(y.sum())} ASD / {int((1 - y).sum())} TC), {dev['site'].nunique()} sites; folds identical to fast experiment; holdout file not opened.")
    site_codes = pd.factorize(dev["site"])[0]            # name -> integer id only; no statistics
    base_fr = pd.read_csv(BASE_DIR / "fold_results.csv")
    rows = []
    for atlas, model in CANDIDATES:
        X = fx.load_fc(dev, atlas)
        Xs = np.column_stack([X, site_codes])
        ok = leakage_check(Xs, y, *folds[0], site_codes)
        log(f"[{atlas}/{model}] site-centering leakage check (test rows independent; offsets from train only): {'PASS' if ok else 'FAIL'}")
        assert ok
        for fold, (tr, te) in enumerate(folds):
            path = OUT / "checkpoints" / f"{atlas}__{model}__centered__fold{fold}.json"
            ck = fx.load_ck(path)
            if ck is None:
                if time.time() - t_start > TOTAL_BUDGET_S:
                    log(f"TIME BUDGET {TOTAL_BUDGET_S}s exceeded -- stopping cleanly before {atlas}/{model} fold {fold}")
                    write_outputs(dev, rows, base_fr, incomplete=True, total_s=time.time() - t_start)
                    return
                ck = run_fold(atlas, model, fold, Xs, y, tr, te)
                fx.save_ck(path, ck)
                log(f"[{atlas}/{model}] fold {fold} centered: acc={ck['metrics']['accuracy']:.4f} auc={ck['metrics']['roc_auc']:.4f} "
                    f"(baseline acc={float(base_fr[(base_fr.atlas == atlas) & (base_fr.model == model) & (base_fr.fold == fold)].accuracy.iloc[0]):.4f}) "
                    f"params={ck['best_params']} offsets for {ck['sites_with_offsets']} sites ({ck['seconds']}s)")
            rows.append(ck)
    write_outputs(dev, rows, base_fr, incomplete=False, total_s=time.time() - t_start)


MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]


def write_outputs(dev, rows, base_fr, incomplete, total_s):
    fr = []
    for r in rows:
        fr.append({"atlas": r["atlas"], "model": r["model"], "variant": "site_centered", "fold": r["fold"], **r["metrics"], "inner_cv_auc": r["inner_cv_auc"],
                   "seconds": r["seconds"], "best_params": json.dumps(r["best_params"]), "sites_with_offsets": r["sites_with_offsets"]})
    for atlas, model in CANDIDATES:
        b = base_fr[(base_fr.atlas == atlas) & (base_fr.model == model)]
        for _, x in b.iterrows():
            fr.append({"atlas": atlas, "model": model, "variant": "uncentered_baseline (from fast_accuracy_experiment)", "fold": int(x.fold),
                       **{m: x[m] for m in MET}, **{k: int(x[k]) for k in ("tn", "fp", "fn", "tp")}, "inner_cv_auc": x.inner_cv_auc, "seconds": x.seconds, "best_params": x.best_params})
    fr = pd.DataFrame(fr)
    fr.to_csv(OUT / "fold_results.csv", index=False)
    comp = []
    for atlas, model in CANDIDATES:
        c = fr[(fr.atlas == atlas) & (fr.model == model) & (fr.variant == "site_centered")].sort_values("fold")
        b = fr[(fr.atlas == atlas) & (fr.model == model) & (fr.variant.str.startswith("uncentered"))].sort_values("fold")
        row = {"atlas": atlas, "model": model, "folds_done": len(c), "status": "complete" if len(c) == fx.N_OUTER else "INCOMPLETE"}
        if len(c) == fx.N_OUTER:
            for m in MET:
                d = c[m].values - b[m].values
                row[f"{m}_baseline_mean"], row[f"{m}_centered_mean"] = float(b[m].mean()), float(c[m].mean())
                row[f"{m}_centered_sd"] = float(c[m].std(ddof=1))
                row[f"{m}_paired_diff_mean"], row[f"{m}_paired_diff_sd"] = float(d.mean()), float(d.std(ddof=1))
            d = c["accuracy"].values - b["accuracy"].values
            row["accuracy_diff_per_fold"] = json.dumps([round(float(v), 4) for v in d])
            row["folds_improved_accuracy"] = int((d > 0).sum())
            row["centered_runtime_s"] = float(c["seconds"].sum())
        comp.append(row)
    cd = pd.DataFrame(comp)
    cd.to_csv(OUT / "site_centering_comparison.csv", index=False)
    L = ["# Site-centering experiment -- summary (development-set CV, exploratory)\n",
         f"Development subjects: {len(dev)} ({int(dev['y'].sum())} ASD / {int((1 - dev['y']).sum())} TC), {dev['site'].nunique()} sites. Same outer folds, inner CV and grids as the fast experiment. "
         "Baselines are the uncentered results already saved by that experiment. Site offsets (site mean - global mean) are estimated from the fitted training rows only; sites absent from, or with fewer than "
         f"{MIN_SITE_N} rows in, the fitted data get zero offset. The locked holdout was not opened.\n\n"]
    if incomplete:
        L.append("**INCOMPLETE:** the run stopped at the time budget; see the log.\n\n")
    L.append("| Atlas | Model | Uncentered acc | Centered acc | Paired diff (mean +/- SD) | Folds improved | Uncentered AUC | Centered AUC | AUC diff |\n|---|---|---|---|---|---|---|---|---|\n")
    for _, r in cd.iterrows():
        if r.status == "complete":
            L.append(f"| {r.atlas} | {r.model} | {r.accuracy_baseline_mean:.4f} | {r.accuracy_centered_mean:.4f} +/- {r.accuracy_centered_sd:.4f} | "
                     f"{r.accuracy_paired_diff_mean:+.4f} +/- {r.accuracy_paired_diff_sd:.4f} | {int(r.folds_improved_accuracy)}/5 | {r.roc_auc_baseline_mean:.4f} | {r.roc_auc_centered_mean:.4f} | {r.roc_auc_paired_diff_mean:+.4f} |\n")
    done = cd[cd.status == "complete"]
    if len(done):
        best = done.sort_values("accuracy_centered_mean", ascending=False).iloc[0]
        helped = done[(done.accuracy_paired_diff_mean > 0) & (done.folds_improved_accuracy >= 4) & (done.accuracy_paired_diff_mean > done.accuracy_paired_diff_sd)]
        L.append(f"\nHighest centered accuracy: {best.atlas}/{best.model} at {best.accuracy_centered_mean:.4f}. This is the best of three candidates and is **exploratory**, not an unbiased estimate.\n")
        L.append(f"\nCandidates where centering improved accuracy in >=4/5 folds AND the mean gain exceeded the SD of the paired differences: **{len(helped)} of {len(done)}**.\n")
        L.append(f"\n**85% target reached: {'YES' if best.accuracy_centered_mean >= 0.85 else 'NO'}** (best measured CV accuracy {best.accuracy_centered_mean:.1%}).\n")
    L.append("\nWith 5 outer folds no significance test is performed; differences of a few points are within fold-to-fold variation. These are development cross-validation estimates, not an independent test.\n")
    (OUT / "experiment_summary.md").write_text("".join(L), encoding="utf-8")
    log(f"Outputs written. Total runtime {total_s:.0f}s")
    log("\n" + cd[["atlas", "model", "status"] + [c for c in ("accuracy_baseline_mean", "accuracy_centered_mean", "accuracy_paired_diff_mean", "accuracy_paired_diff_sd", "folds_improved_accuracy", "roc_auc_paired_diff_mean") if c in cd.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
