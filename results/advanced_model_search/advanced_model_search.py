"""
Advanced model search -- ONE pre-specified experiment: Connectome-Based Predictive Modeling (CPM) on static AAL-116 FC.
600 development subjects, the 5 fixed outer folds of the fast experiment, nested CV.

CPM (training rows only, inside every fit):  mean imputation + Fisher-Z -> for every edge, Pearson correlation with the
0/1 diagnosis and its two-sided t-test p-value -> positive edges (r>0, p<thr) and negative edges (r<0, p<thr) ->
two network-strength features (sum of edge values over the positive set / negative set; 0 if a set is empty) ->
standardize -> L2 logistic regression (C=1, balanced). The ONLY tuned hyper-parameter is the edge p-threshold,
chosen from {0.05, 0.01, 0.005, 0.001} by inner 3-fold ROC-AUC on the outer-training rows.
This is the only new candidate: no choice among methods is made from outer-fold scores, so its outer estimate is not
selection-biased (the method family was chosen by inspecting earlier results, which is a design-process caveat).
Frozen reference: saved AAL116 / LR_L2 results of the fast experiment (read, never recomputed or modified).
Holdout, CC200 and the restricted ID file are not touched.
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
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fast_accuracy_experiment as fx

ROOT = Path(r"C:\abide")
R = ROOT / "results"
BASE = R / "fast_accuracy_experiment"
OUT = R / "advanced_model_search"
for s in ("checkpoints", "logs", "data"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "advanced_search.log"
P_GRID = [0.05, 0.01, 0.005, 0.001]
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
BUDGET_S = 300


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


class CPM(ClassifierMixin, BaseEstimator):
    def __init__(self, p_thresh=0.01):
        self.p_thresh = p_thresh

    def _strength(self, X):
        pos = X[:, self.pos_].sum(axis=1) if self.pos_.size else np.zeros(len(X))
        neg = X[:, self.neg_].sum(axis=1) if self.neg_.size else np.zeros(len(X))
        return np.column_stack([pos, neg])

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n = len(y)
        yc = y - y.mean(); Xc = X - X.mean(axis=0)
        den = np.sqrt((yc ** 2).sum()) * np.sqrt((Xc ** 2).sum(axis=0))
        with np.errstate(all="ignore"):
            r = np.where(den > 0, (yc @ Xc) / den, 0.0)
        with np.errstate(all="ignore"):
            t = r * np.sqrt((n - 2) / np.maximum(1 - r ** 2, 1e-12))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
        self.r_ = r
        self.pos_ = np.where((p < self.p_thresh) & (r > 0))[0]
        self.neg_ = np.where((p < self.p_thresh) & (r < 0))[0]
        F = self._strength(X)
        self.sc_ = StandardScaler().fit(F)
        self.lr_ = LogisticRegression(class_weight="balanced", max_iter=1000).fit(self.sc_.transform(F), y)
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.sc_.transform(self._strength(X)))

    def predict(self, X):
        return self.lr_.predict(self.sc_.transform(self._strength(X)))


def pipe():
    return Pipeline([("fz", fx.MeanImputeFisherZ()), ("cpm", CPM())])


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
    log("=" * 60 + "\nCPM EXPERIMENT start")
    pre = json.load(open(R / "phase10_corrected_nested_cv" / "logs" / "integrity_pre.json"))
    now = sha_all()
    drift = [k for k in pre if now.get(k) != pre[k]["sha256"]]
    log(f"Integrity before: {len(pre)} baseline files, changed={len(drift)}")
    assert not drift
    dev, _ = fx.build_dev_set()
    y = dev["y"].values
    man = pd.read_csv(BASE / "data" / "dev_split_manifest.csv", dtype={"sid": str})
    assert list(man["sid"]) == list(dev["sid"]) and (man["y"].values == y).all()
    folds = list(StratifiedKFold(fx.N_OUTER, shuffle=True, random_state=fx.SEED).split(np.zeros(len(y)), y))
    assert all(set(np.where(man["outer_fold"].values == k)[0]) == set(te) for k, (_, te) in enumerate(folds)), "folds differ from manifest"
    base_fr = pd.read_csv(BASE / "fold_results.csv"); base_fr = base_fr[(base_fr.atlas == "AAL116") & (base_fr.model == "LR_L2")].sort_values("fold")
    base_oof = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str}); base_oof = base_oof[(base_oof.atlas == "AAL116") & (base_oof.model == "LR_L2")]
    j = base_oof.merge(man[["sid", "y", "outer_fold"]], left_on="subject_id", right_on="sid")
    assert len(base_fr) == 5 and len(base_oof) == 600 and (j.y_true == j.y).all() and (j.fold == j.outer_fold).all()
    log("Verified: 600-subject manifest, labels and folds identical to the frozen AAL116 LR_L2 baseline (600 unique saved OOF predictions).")
    X = fx.load_fc(dev, "AAL116")
    log(f"AAL-116 matrix {X.shape} (cached development features, NaN entries {int(np.isnan(X).sum())} imputed inside folds)")

    # leakage / correctness checks on fold 0
    tr0, te0 = folds[0]
    p = pipe().set_params(cpm__p_thresh=0.01).fit(X[tr0], y[tr0])
    base = p.predict_proba(X[te0])[:, 1]
    Xj = X[te0].copy(); half = np.arange(len(te0)) % 2 == 1
    Xj[half] = np.random.default_rng(0).uniform(-0.9, 0.9, Xj[half].shape)
    indep = bool(np.allclose(base[~half], p.predict_proba(Xj)[:, 1][~half], atol=1e-12))
    Zt = p.named_steps["fz"].transform(X[tr0]); cp = p.named_steps["cpm"]
    chk = [float(np.corrcoef(Zt[:, e], y[tr0])[0, 1]) for e in (5, 777, 4000)]
    corr_ok = bool(np.allclose(chk, cp.r_[[5, 777, 4000]], atol=1e-10))
    log(f"Checks: test-row independence {'PASS' if indep else 'FAIL'}; edge correlations equal independent training-row recomputation {'PASS' if corr_ok else 'FAIL'}")
    assert indep and corr_ok

    rows, sets = [], {}
    for fold, (tr, te) in enumerate(folds):
        path = OUT / "checkpoints" / f"CPM__fold{fold}.json"
        ck = fx.load_ck(path)
        if ck is None:
            if time.time() - t0 > BUDGET_S:
                log("time budget exceeded; stopping (incomplete results would not be reported)"); return
            gs = GridSearchCV(pipe(), {"cpm__p_thresh": P_GRID}, scoring="roc_auc", cv=StratifiedKFold(fx.N_INNER, shuffle=True, random_state=fx.SEED), n_jobs=4, refit=True, error_score="raise")
            t1 = time.time(); gs.fit(X[tr], y[tr])
            est = gs.best_estimator_; c = est.named_steps["cpm"]
            sc = est.predict_proba(X[te])[:, 1]; pr = est.predict(X[te])
            ck = {"method": "CPM", "fold": fold, "metrics": fx.metrics(y[te], pr, sc), "inner_cv_auc": float(gs.best_score_), "best_p_thresh": float(gs.best_params_["cpm__p_thresh"]),
                  "n_pos_edges": int(c.pos_.size), "n_neg_edges": int(c.neg_.size), "inner_auc_by_threshold": {str(k): float(v) for k, v in zip(P_GRID, gs.cv_results_["mean_test_score"])},
                  "seconds": round(time.time() - t1, 1), "test_idx": [int(i) for i in te], "score": [float(s) for s in sc], "pred": [int(v) for v in pr],
                  "pos_edges": [int(i) for i in c.pos_], "neg_edges": [int(i) for i in c.neg_]}
            fx.save_ck(path, ck)
        b = base_fr[base_fr.fold == fold].iloc[0]
        log(f"[CPM] fold {fold}: acc={ck['metrics']['accuracy']:.4f} auc={ck['metrics']['roc_auc']:.4f} (baseline acc={b.accuracy:.4f} auc={b.roc_auc:.4f}) "
            f"p_thresh={ck['best_p_thresh']} edges +{ck['n_pos_edges']}/-{ck['n_neg_edges']} inner_auc={ck['inner_cv_auc']:.4f} ({ck['seconds']}s)")
        rows.append(ck)
    report(dev, y, man, rows, base_fr, base_oof, X, time.time() - t0, pre)


def report(dev, y, man, rows, base_fr, base_oof, X, total_s, pre):
    fm = [{"method": "BASELINE_AAL116_LR_L2 (frozen)", "fold": int(b.fold), **{m: b[m] for m in MET}, "inner_cv_auc": b.inner_cv_auc} for _, b in base_fr.iterrows()]
    fm += [{"method": "CPM_AAL116", "fold": r["fold"], **r["metrics"], "inner_cv_auc": r["inner_cv_auc"], "best_p_thresh": r["best_p_thresh"], "n_pos_edges": r["n_pos_edges"], "n_neg_edges": r["n_neg_edges"]} for r in rows]
    fm = pd.DataFrame(fm)
    base = fm[fm.method.str.startswith("BASELINE")].sort_values("fold"); cpm = fm[fm.method == "CPM_AAL116"].sort_values("fold")
    oofr = [{"method": "CPM_AAL116", "fold": r["fold"], "subject_id": dev["sid"].iloc[i], "site": dev["site"].iloc[i], "y_true": int(y[i]), "score": s, "y_pred": p} for r in rows for i, s, p in zip(r["test_idx"], r["score"], r["pred"])]
    oof = pd.DataFrame(oofr); oof.to_csv(OUT / "oof_predictions.csv", index=False)
    assert oof.subject_id.is_unique and len(oof) == 600
    mrows = []
    for name, g in fm.groupby("method"):
        r = {"method": name}
        for m in MET:
            r[f"{m}_mean"], r[f"{m}_sd"] = float(g[m].mean()), float(g[m].std(ddof=1))
        if name == "CPM_AAL116":
            for m in MET:
                d = cpm[m].values - base[m].values
                r[f"{m}_paired_diff_mean"], r[f"{m}_paired_diff_sd"], r[f"{m}_folds_improved"] = float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum())
        mrows.append(r)
    pd.DataFrame(mrows).to_csv(OUT / "metrics_by_method.csv", index=False)
    d_acc = cpm.accuracy.values - base.accuracy.values; d_auc = cpm.roc_auc.values - base.roc_auc.values
    bmap = base.set_index("fold")
    fm2 = fm.copy()
    for m in MET:
        fm2[f"diff_{m}_vs_baseline"] = [float(r[m] - bmap.loc[r.fold, m]) if r.method == "CPM_AAL116" else np.nan for _, r in fm2.iterrows()]
    fm2.to_csv(OUT / "fold_metrics.csv", index=False)
    # edge stability across outer folds
    pos = [set(r["pos_edges"]) for r in rows]; neg = [set(r["neg_edges"]) for r in rows]
    jac = lambda S: float(np.mean([len(a & b) / max(1, len(a | b)) for i, a in enumerate(S) for b in S[i + 1:]]))
    # descriptive generalization checks on CPM OOF
    coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str})
    coh["sid"] = coh.subject_id.map(lambda s: str(int(float(s))))
    coh = coh[coh.sid.isin(set(man.sid))][["sid", "func_mean_fd", "sex", "age"]]            # development rows only
    o = oof.merge(coh, left_on="subject_id", right_on="sid")
    o["err"] = (o.y_true != o.y_pred).astype(int)
    site = o.groupby("site").agg(n=("err", "size"), acc=("err", lambda s: 1 - s.mean())).sort_values("n", ascending=False)
    rho = stats.spearmanr(o.score, o.func_mean_fd)
    sx = o.assign(sexl=o.sex.map({1: "male", 2: "female"})).groupby(["sexl", "y_true"]).agg(n=("err", "size"), acc=("err", lambda s: 1 - s.mean()))
    L = ["# CPM (connectome-based predictive modeling) on static AAL-116 -- nested CV, 600 development subjects\n",
         "Pre-specified single candidate; frozen reference = saved AAL-116 L2 logistic regression (read from `fast_accuracy_experiment`). Edge selection, network strengths, scaling and the logistic model are fitted on training rows only; "
         f"the edge p-threshold ({P_GRID}) is chosen by inner 3-fold ROC-AUC. Same five outer folds. Holdout, CC200 and the restricted ID file not used. Cross-validation estimates on the development cohort, not independent-test results.\n\n",
         "| Method | Accuracy | Bal. acc. | Sens. | Spec. | Prec. | F1 | AUC |\n|---|---|---|---|---|---|---|---|\n"]
    for r in mrows:
        L.append(f"| {r['method']} | {r['accuracy_mean']:.4f} +/- {r['accuracy_sd']:.4f} | {r['balanced_accuracy_mean']:.4f} | {r['sensitivity_mean']:.4f} | {r['specificity_mean']:.4f} | {r['precision_mean']:.4f} | {r['f1_mean']:.4f} | {r['roc_auc_mean']:.4f} +/- {r['roc_auc_sd']:.4f} |\n")
    cm = [r for r in mrows if r["method"] == "CPM_AAL116"][0]
    L.append(f"\nPaired fold-level differences, CPM minus frozen baseline: accuracy {cm['accuracy_paired_diff_mean']:+.4f} +/- {cm['accuracy_paired_diff_sd']:.4f} (per fold {[round(float(v), 3) for v in d_acc]}, improved in {cm['accuracy_folds_improved']}/5); "
             f"AUC {cm['roc_auc_paired_diff_mean']:+.4f} +/- {cm['roc_auc_paired_diff_sd']:.4f} (per fold {[round(float(v), 3) for v in d_auc]}, improved in {cm['roc_auc_folds_improved']}/5); "
             f"balanced accuracy {cm['balanced_accuracy_paired_diff_mean']:+.4f}.\n")
    L.append(f"\nPer-fold CPM: selected p-threshold {[r['best_p_thresh'] for r in rows]}; positive/negative edges {[(r['n_pos_edges'], r['n_neg_edges']) for r in rows]}; inner AUC {[round(r['inner_cv_auc'], 3) for r in rows]} (outer AUC {[round(r['metrics']['roc_auc'], 3) for r in rows]}).\n")
    L.append(f"Edge-set stability across outer folds (mean pairwise Jaccard): positive edges {jac(pos):.3f}, negative edges {jac(neg):.3f}.\n")
    L.append("\n## Generalization checks (descriptive, CPM out-of-fold predictions)\n")
    L.append("Accuracy by site (n<20 flagged noisy): " + ", ".join(f"{s} {r.acc:.2f} (n={int(r.n)}{'*' if r.n < 20 else ''})" for s, r in site.iterrows()) + "\n")
    L.append(f"\nSpearman correlation of CPM scores with mean FD: {rho.correlation:.3f} (p={rho.pvalue:.2g}); accuracy by sex and diagnosis: " + "; ".join(f"{a}/{'ASD' if b else 'TC'} {r.acc:.2f} (n={int(r.n)})" for (a, b), r in sx.iterrows()) + ".\n")
    prom = (cm["accuracy_paired_diff_mean"] >= 0.02 and cm["roc_auc_paired_diff_mean"] >= 0.02 and cm["accuracy_folds_improved"] >= 4)
    worse = cm["accuracy_paired_diff_mean"] <= -0.02 and cm["roc_auc_paired_diff_mean"] <= -0.02
    concl = "CPM IMPROVES on the baseline" if prom else ("CPM PERFORMS WORSE than the baseline" if worse else "NO RELIABLE DIFFERENCE from the baseline (inconclusive)")
    L.append(f"\n**Conclusion (rule fixed before running: improvement needs >= +0.02 mean accuracy AND AUC with >= 4/5 folds improved; worse needs <= -0.02 on both): {concl}.**\n"
             "\nNo significance test is claimed from 5 folds; no independent-test or 98%/85% claim. The frozen baseline (about 64.2% accuracy, AUC 0.694) is unchanged. "
             "Limitations: one pre-specified method, threshold grid of four values, 600 subjects, site and motion effects not adjusted here, method family chosen after inspecting earlier results.\n"
             f"Runtime {total_s:.0f}s.\n")
    (OUT / "experiment_report.md").write_text("".join(L), encoding="utf-8")
    now = sha_all(); drift = [k for k in pre if now.get(k) != pre[k]["sha256"]]
    ck = ["- Subjects, labels, folds identical to the development manifest and the frozen baseline (asserted): PASS", "- AAL-116 only; CC200, holdout and restricted file not used: PASS",
          f"- One OOF prediction per subject for CPM: {len(oof)} rows / {oof.subject_id.nunique()} unique: PASS",
          "- Edge selection, strengths, scaler and logistic model fitted on training rows only; p-threshold chosen by inner 3-fold CV inside each outer-training set: PASS",
          "- Test-row independence perturbation and independent recomputation of edge correlations from training rows (fold 0): PASS",
          f"- Previously hashed files after run (311): changed={len(drift)}: {'PASS' if not drift else 'FAIL'}",
          "- No method selection from outer-fold scores (single pre-specified candidate): PASS", "- Baseline results read from saved files, not modified: PASS"]
    (OUT / "integrity_checks.md").write_text("# Integrity checks\n\n" + "\n".join(ck) + "\n", encoding="utf-8")
    json.dump({"method": "CPM (Pearson edge-label correlation, positive/negative network strengths, L2 logistic regression)", "p_threshold_grid": P_GRID, "inner_cv": "StratifiedKFold(3, seed 42), roc_auc",
               "outer_cv": "StratifiedKFold(5, shuffle, seed 42) identical to fast experiment", "atlas": "AAL116", "n_subjects": 600, "baseline": str(BASE / "fold_results.csv") + " (AAL116/LR_L2)",
               "threads": "OMP/MKL/OPENBLAS=1", "runtime_s": round(total_s, 1)}, open(OUT / "config.json", "w"), indent=1)
    log(f"Report written; {concl}; drift={len(drift)}; runtime {total_s:.0f}s")
    log("\n" + pd.DataFrame(mrows)[["method", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "roc_auc_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
