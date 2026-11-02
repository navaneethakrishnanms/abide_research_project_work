"""
motion_confound_experiment.py -- paired test: does removing the linear mean-FD effect (fitted on TRAINING rows only)
change cross-validated ASD-vs-TC performance? Plus an FD-only control. Development subjects only; the locked holdout
file is never opened.

Adjustment (MotionResidualize), applied to Fisher-Z features z_j after training-row mean imputation:
    z_adj_ij = z_ij - beta_j * (fd_i - mean_fd_fit),   beta_j = OLS slope of z_j on fd over the FITTED rows.
The feature's fitted-row mean is kept (intercept convention); only the FD-linked component is removed. The step is a
Pipeline member, so inside GridSearchCV every inner-training split learns its own beta; validation/test rows never
contribute. Subject-level mean FD is the last input column.

Baselines (unadjusted) are read from the completed fast experiment (same subjects, folds, grids, code path).
Pre-declared reading rule (set before running): hypothesis SUPPORTED if, for >= 2 of the 3 candidates, mean accuracy
drops by >= 3 points and accuracy drops in >= 4/5 folds; WEAKENED if every candidate's mean change is within +/-1.5
points; otherwise UNRESOLVED.
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
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
BASE_DIR = ROOT / "results" / "fast_accuracy_experiment"
OUT = ROOT / "results" / "motion_confound_experiment"
for s in ("checkpoints", "logs"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "motion_confound.log"
TOTAL_BUDGET_S = 300
CANDIDATES = [("AAL116", "LR_L2"), ("AAL116", "LinearSVM"), ("CC200", "RBF_SVM")]
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}\n")
    print(msg, flush=True)


class MotionResidualize(BaseEstimator, TransformerMixin):
    """Input: raw FC features with mean FD appended as the LAST column. Output: adjusted Fisher-Z features."""

    def _z(self, F):
        F = np.where(np.isnan(F), self.means_[None, :], F)
        return np.arctanh(np.clip(F, -1 + 1e-6, 1 - 1e-6))

    def fit(self, X, y=None):
        import warnings
        F, fd = X[:, :-1], X[:, -1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmean(F, axis=0)
        self.means_ = np.where(np.isnan(m), 0.0, m)
        Z = self._z(F)
        self.fd_mean_ = float(fd.mean())
        fdc = fd - self.fd_mean_
        self.beta_ = (fdc @ (Z - Z.mean(axis=0))) / float(fdc @ fdc)          # OLS slope per feature
        return self

    def transform(self, X):
        F, fd = X[:, :-1], X[:, -1]
        return self._z(F) - np.outer(fd - self.fd_mean_, self.beta_)


def pipe(clf):
    return Pipeline([("mot", MotionResidualize()), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)), ("clf", clf)])


def run_fold(atlas, model, fold, Xs, y, tr, te):
    make, grid = fx.MODELS[model]
    gs = GridSearchCV(pipe(make()), grid, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=fx.SEED),
                      n_jobs=6, refit=True, error_score="raise")
    t0 = time.time()
    gs.fit(Xs[tr], y[tr])
    est = gs.best_estimator_
    score = est.predict_proba(Xs[te])[:, 1] if hasattr(est, "predict_proba") else est.decision_function(Xs[te])
    pred = est.predict(Xs[te])
    return {"atlas": atlas, "model": model, "variant": "motion_adjusted", "fold": fold, "metrics": fx.metrics(y[te], pred, score),
            "best_params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in gs.best_params_.items()},
            "inner_cv_auc": float(gs.best_score_), "seconds": round(time.time() - t0, 1), "test_idx": [int(i) for i in te], "score": [float(s) for s in score]}


def run_fd_only(fold, fd, y, tr, te):
    p = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced"))])
    gs = GridSearchCV(p, {"clf__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=fx.SEED), refit=True)
    gs.fit(fd[tr].reshape(-1, 1), y[tr])
    est = gs.best_estimator_
    score = est.predict_proba(fd[te].reshape(-1, 1))[:, 1]
    return {"fold": fold, "metrics": fx.metrics(y[te], est.predict(fd[te].reshape(-1, 1)), score), "best_params": gs.best_params_, "inner_cv_auc": float(gs.best_score_)}


def checks(Xs, y, tr, te):
    p = pipe(LogisticRegression(class_weight="balanced", max_iter=2000)).set_params(sel__k=500, clf__C=1e-3).fit(Xs[tr], y[tr])
    base = p.predict_proba(Xs[te])[:, 1]
    Xj = Xs[te].copy()
    half = np.arange(len(te)) % 2 == 1
    Xj[half] = np.random.default_rng(0).uniform(-0.9, 0.9, Xj[half].shape)           # scramble features AND FD of half the TEST rows
    indep = bool(np.allclose(base[~half], p.predict_proba(Xj)[:, 1][~half], atol=1e-12))
    mot = p.named_steps["mot"]
    Ztr = mot.transform(Xs[tr])
    resid_corr = float(np.max(np.abs([np.corrcoef(Xs[tr][:, -1], Ztr[:, j])[0, 1] for j in range(0, Ztr.shape[1], 97)])))
    F = Xs[tr][:, :-1]; Z = np.arctanh(np.clip(np.where(np.isnan(F), mot.means_[None, :], F), -1 + 1e-6, 1 - 1e-6))
    fdc = Xs[tr][:, -1] - Xs[tr][:, -1].mean()
    j = 123
    manual = np.polyfit(Xs[tr][:, -1], Z[:, j], 1)[0]
    return indep, resid_corr, bool(np.isclose(manual, mot.beta_[j], rtol=1e-8))


def main():
    t_start = time.time()
    log("=" * 60 + "\nMOTION-CONFOUND EXPERIMENT start")
    dev, _ = fx.build_dev_set()
    y = dev["y"].values
    man = pd.read_csv(BASE_DIR / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all(), "dev set differs from fast experiment manifest"
    coh = pd.read_csv(ROOT / "results" / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str})
    coh["sid"] = coh["subject_id"].map(fx.norm_id)
    coh = coh[coh["sid"].isin(set(dev["sid"]))].set_index("sid")                    # development rows only
    fd = coh.loc[dev["sid"], "func_mean_fd"].values.astype(float)
    assert len(fd) == 600 and np.isfinite(fd).all() and (coh.loc[dev["sid"], "diagnosis"].values == dev["diagnosis"].values).all(), "FD missing/misaligned"
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds)), "folds differ from fast experiment"
    log(f"Aligned: 600 subjects ({int(y.sum())} ASD / {int((1 - y).sum())} TC), FD finite for all (mean {fd.mean():.3f}); folds identical to fast experiment; holdout not opened.")
    base_fr = pd.read_csv(BASE_DIR / "fold_results.csv")

    # FD-only control
    fd_rows = []
    for fold, (tr, te) in enumerate(folds):
        p = fx.load_ck(OUT / "checkpoints" / f"fd_only__fold{fold}.json")
        if p is None:
            p = run_fd_only(fold, fd, y, tr, te)
            fx.save_ck(OUT / "checkpoints" / f"fd_only__fold{fold}.json", p)
        fd_rows.append(p)
        log(f"[FD-only] fold {fold}: acc={p['metrics']['accuracy']:.4f} auc={p['metrics']['roc_auc']:.4f}")

    rows = []
    for atlas, model in CANDIDATES:
        X = fx.load_fc(dev, atlas)
        Xs = np.column_stack([X, fd])
        indep, rc, bok = checks(Xs, y, *folds[0])
        log(f"[{atlas}/{model}] checks: test-row independence {'PASS' if indep else 'FAIL'}; max |corr(FD, adjusted train feature)| = {rc:.2e}; fitted beta equals independent OLS: {'PASS' if bok else 'FAIL'}")
        assert indep and bok and rc < 1e-8
        for fold, (tr, te) in enumerate(folds):
            path = OUT / "checkpoints" / f"{atlas}__{model}__adj__fold{fold}.json"
            ck = fx.load_ck(path)
            if ck is None:
                if time.time() - t_start > TOTAL_BUDGET_S:
                    log(f"TIME BUDGET {TOTAL_BUDGET_S}s exceeded -- stopping cleanly before {atlas}/{model} fold {fold}")
                    write_outputs(dev, y, fd, rows, fd_rows, base_fr, True, time.time() - t_start)
                    return
                ck = run_fold(atlas, model, fold, Xs, y, tr, te)
                fx.save_ck(path, ck)
                b = float(base_fr[(base_fr.atlas == atlas) & (base_fr.model == model) & (base_fr.fold == fold)].accuracy.iloc[0])
                log(f"[{atlas}/{model}] fold {fold} adjusted: acc={ck['metrics']['accuracy']:.4f} auc={ck['metrics']['roc_auc']:.4f} (unadjusted acc={b:.4f}) params={ck['best_params']} ({ck['seconds']}s)")
            rows.append(ck)
    write_outputs(dev, y, fd, rows, fd_rows, base_fr, False, time.time() - t_start)


def write_outputs(dev, y, fd, rows, fd_rows, base_fr, incomplete, total_s):
    fr = [{"atlas": r["atlas"], "model": r["model"], "variant": "motion_adjusted", "fold": r["fold"], **r["metrics"], "inner_cv_auc": r["inner_cv_auc"], "seconds": r["seconds"],
           "best_params": json.dumps(r["best_params"])} for r in rows]
    for atlas, model in CANDIDATES:
        for _, x in base_fr[(base_fr.atlas == atlas) & (base_fr.model == model)].iterrows():
            fr.append({"atlas": atlas, "model": model, "variant": "unadjusted (from fast_accuracy_experiment)", "fold": int(x.fold), **{m: x[m] for m in MET},
                       **{k: int(x[k]) for k in ("tn", "fp", "fn", "tp")}, "inner_cv_auc": x.inner_cv_auc, "seconds": x.seconds, "best_params": x.best_params})
    fr = pd.DataFrame(fr)
    fr.to_csv(OUT / "fold_results.csv", index=False)
    fdd = pd.DataFrame([{"fold": p["fold"], **p["metrics"], "inner_cv_auc": p["inner_cv_auc"], "best_params": json.dumps(p["best_params"])} for p in fd_rows])
    fdd.loc[len(fdd)] = {"fold": "mean", **{m: fdd[m].mean() for m in MET}}
    fdd.loc[len(fdd)] = {"fold": "sd", **{m: fdd.iloc[:5][m].std(ddof=1) for m in MET}}
    fdd.to_csv(OUT / "fd_only_baseline.csv", index=False)
    comp = []
    for atlas, model in CANDIDATES:
        c = fr[(fr.atlas == atlas) & (fr.model == model) & (fr.variant == "motion_adjusted")].sort_values("fold")
        b = fr[(fr.atlas == atlas) & (fr.model == model) & (fr.variant.str.startswith("unadjusted"))].sort_values("fold")
        row = {"atlas": atlas, "model": model, "folds_done": len(c), "status": "complete" if len(c) == 5 else "INCOMPLETE"}
        if len(c) == 5:
            for m in MET:
                d = c[m].values - b[m].values
                row.update({f"{m}_unadjusted_mean": float(b[m].mean()), f"{m}_adjusted_mean": float(c[m].mean()), f"{m}_adjusted_sd": float(c[m].std(ddof=1)),
                            f"{m}_paired_diff_mean": float(d.mean()), f"{m}_paired_diff_sd": float(d.std(ddof=1))})
            d = c["accuracy"].values - b["accuracy"].values
            row["accuracy_diff_per_fold"] = json.dumps([round(float(v), 4) for v in d]); row["folds_accuracy_dropped"] = int((d < 0).sum()); row["folds_accuracy_improved"] = int((d > 0).sum())
            dd = c["roc_auc"].values - b["roc_auc"].values
            row["auc_diff_per_fold"] = json.dumps([round(float(v), 4) for v in dd])
            # descriptive: do adjusted OOF scores still track FD?
            sc = np.full(len(y), np.nan)
            for r in rows:
                if r["atlas"] == atlas and r["model"] == model:
                    sc[r["test_idx"]] = r["score"]
            row["spearman_adjusted_score_vs_FD"] = float(stats.spearmanr(sc, fd).correlation)
        comp.append(row)
    cd = pd.DataFrame(comp)
    cd.to_csv(OUT / "motion_adjustment_comparison.csv", index=False)
    done = cd[cd.status == "complete"]
    verdict = "UNRESOLVED"
    if len(done) == 3:
        supp = int(((done.accuracy_paired_diff_mean <= -0.03) & (done.folds_accuracy_dropped >= 4)).sum())
        if supp >= 2:
            verdict = "SUPPORTED"
        elif (done.accuracy_paired_diff_mean.abs() <= 0.015).all():
            verdict = "WEAKENED"
    L = ["# Motion-confound experiment -- summary (development-set CV, exploratory)\n",
         f"600 development subjects (268 ASD / 332 TC), same outer folds, inner CV and grids as the fast experiment. Adjustment: z_adj = z - beta_j*(fd - mean_fd_fit), beta_j = OLS slope on the fitted rows only "
         "(re-estimated in every inner-training split); the feature's fitted mean is kept. Unadjusted results are the saved ones from the fast experiment. The locked holdout was not opened; its independence remains unverified.\n\n"]
    if incomplete:
        L.append("**INCOMPLETE** (time budget).\n\n")
    L.append("| Atlas | Model | Unadjusted acc | Adjusted acc | Paired diff (mean +/- SD) | Per-fold diffs | Unadjusted AUC | Adjusted AUC |\n|---|---|---|---|---|---|---|---|\n")
    for _, r in done.iterrows():
        L.append(f"| {r.atlas} | {r.model} | {r.accuracy_unadjusted_mean:.4f} | {r.accuracy_adjusted_mean:.4f} +/- {r.accuracy_adjusted_sd:.4f} | {r.accuracy_paired_diff_mean:+.4f} +/- {r.accuracy_paired_diff_sd:.4f} | {r.accuracy_diff_per_fold} | {r.roc_auc_unadjusted_mean:.4f} | {r.roc_auc_adjusted_mean:.4f} |\n")
    if len(done):
        L.append("\nAdjusted out-of-fold scores vs FD (Spearman, descriptive): " + ", ".join(f"{r.atlas}/{r.model} {r.spearman_adjusted_score_vs_FD:.2f}" for _, r in done.iterrows()) + " (unadjusted models were 0.20-0.28).\n")
    if len(fd_rows) == 5:
        L.append(f"\n**FD-only control** (one feature, mean FD, logistic regression, same folds): accuracy {fdd.iloc[5].accuracy:.4f} +/- {fdd.iloc[6].accuracy:.4f}, balanced accuracy {fdd.iloc[5].balanced_accuracy:.4f}, AUC {fdd.iloc[5].roc_auc:.4f} +/- {fdd.iloc[6].roc_auc:.4f} (class-prior accuracy is 0.553).\n")
    L.append(f"\n**Pre-declared reading rule outcome: motion-confound hypothesis {verdict}.** (SUPPORTED needs >=2 of 3 candidates with a >=3-point mean accuracy drop and drops in >=4/5 folds; WEAKENED needs every mean change within +/-1.5 points.)\n")
    L.append("\nCaveats: only the linear FD component is removed, so non-linear or site-specific motion effects are not addressed; FD is also related to site and diagnosis, so adjustment can remove real diagnosis-related variance; "
             "5 folds cannot establish significance; correlation and predictive performance do not show causality. No 85% claim: best measured accuracy is in the CSVs.\n")
    (OUT / "experiment_summary.md").write_text("".join(L), encoding="utf-8")
    log(f"Outputs written; total runtime {total_s:.0f}s; verdict {verdict}")
    log("\n" + done[[c for c in ("atlas", "model", "accuracy_unadjusted_mean", "accuracy_adjusted_mean", "accuracy_paired_diff_mean", "accuracy_paired_diff_sd", "folds_accuracy_dropped", "roc_auc_paired_diff_mean", "spearman_adjusted_score_vs_FD") if c in done.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
