"""
fast_accuracy_experiment.py -- bounded, leakage-safe model comparison on the DEVELOPMENT subjects only.

Development set = (subject IDs listed in splits/phase2_train_subjects.txt) INTERSECT (the FD-filtered 780 cohort).
The locked 170-subject holdout file is never opened here: a subject that is not in the train list is simply
never selected, so the holdout is excluded without this script learning who it contains.

Everything learned (mean imputation, scaling, feature selection, hyper-parameters) is fitted inside sklearn
Pipelines on training rows only; hyper-parameters are chosen by inner 3-fold CV (ROC-AUC). Outer 5-fold
stratified CV (seed 42) is shared by all candidates and both atlases. Results here are CROSS-VALIDATION
estimates on development subjects, not an independent test.

Usage:  python fast_accuracy_experiment.py [--smoke]
"""
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # this process (and its workers) only
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = Path(r"C:\abide")
TRAIN_LIST = ROOT / "splits" / "phase2_train_subjects.txt"
COHORT_CSV = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
AAL_DIR = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
OUT = ROOT / "results" / "fast_accuracy_experiment"
for s in ("checkpoints", "data", "logs", "joblib_cache"):
    (OUT / s).mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs" / "fast_experiment.log"
SEED = 42
N_OUTER, N_INNER = 5, 3
PER_MODEL_BUDGET_S = 300
ATLASES = {"AAL116": ("rois_aal", 116), "CC200": ("rois_cc200", 200)}


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(msg, flush=True)


class MeanImputeFisherZ(BaseEstimator, TransformerMixin):
    """Training-fold column-mean imputation, then Fisher-Z (clip 1e-6). Statistics come from fit() rows only."""

    def fit(self, X, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmean(X, axis=0)
        self.means_ = np.where(np.isnan(m), 0.0, m)
        return self

    def transform(self, X):
        X = np.where(np.isnan(X), self.means_[None, :], X)
        return np.arctanh(np.clip(X, -1 + 1e-6, 1 - 1e-6))


def norm_id(s):
    s = str(s).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def subj_id_from_file(name, suffix):
    base = name.replace(f"_{suffix}.1D", "")
    digits = "".join(ch for ch in base.split("_")[-1] if ch.isdigit())
    return str(int(digits)) if digits else None


def build_dev_set():
    train_ids = [norm_id(l) for l in open(TRAIN_LIST) if l.strip()]
    assert len(train_ids) == 676 and len(set(train_ids)) == 676, "train list is not 676 unique ids"
    train_set = set(train_ids)
    cohort = pd.read_csv(COHORT_CSV, dtype={"subject_id": str})
    cohort["sid"] = cohort["subject_id"].map(norm_id)
    dev = cohort[cohort["sid"].isin(train_set)].reset_index(drop=True)      # only development rows are kept
    del cohort
    assert dev["sid"].is_unique and set(dev["sid"]) <= train_set
    dev["y"] = (dev["diagnosis"] == "ASD").astype(int)
    return dev[["sid", "diagnosis", "site", "y"]], train_ids


def load_fc(dev, atlas):
    suffix, n_roi = ATLASES[atlas]
    cache = OUT / "data" / f"fc_raw_{atlas}.npy"
    ids_csv = OUT / "data" / "dev_subject_ids.csv"
    if cache.exists():
        X = np.load(cache)
        assert X.shape[0] == len(dev)
        return X
    files = {}
    for f in AAL_DIR.glob(f"*_{suffix}.1D"):
        sid = subj_id_from_file(f.name, suffix)
        if sid in set(dev["sid"]):
            files[sid] = f
    assert len(files) == len(dev), f"{atlas}: missing time-series files for {len(dev) - len(files)} dev subjects"
    iu = np.triu_indices(n_roi, k=1)
    X = np.full((len(dev), len(iu[0])), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, sid in enumerate(dev["sid"]):
            ts = np.loadtxt(str(files[sid]))
            X[i] = np.corrcoef(ts, rowvar=False)[iu]
    np.save(cache, X)
    return X


def pipe(clf, memory):
    return Pipeline([("fz", MeanImputeFisherZ()), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif)), ("clf", clf)], memory=memory)


KS = [500, 2000, "all"]
MODELS = {
    "LR_L2": (lambda: LogisticRegression(class_weight="balanced", max_iter=2000), {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1], "sel__k": KS}),
    "LinearSVM": (lambda: LinearSVC(class_weight="balanced", dual="auto", max_iter=5000), {"clf__C": [1e-5, 1e-4, 1e-3], "sel__k": KS}),
    "L1_LR": (lambda: LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=1000), {"clf__C": [0.01, 0.03, 0.1], "sel__k": [2000]}),
    "RBF_SVM": (lambda: SVC(kernel="rbf", class_weight="balanced"), {"clf__C": [1, 10], "clf__gamma": ["scale", 1e-4], "sel__k": [500, 2000]}),
}


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "sensitivity": recall_score(y, pred, zero_division=0), "specificity": tn / (tn + fp),
            "precision": precision_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def ck_path(atlas, model, fold):
    return OUT / "checkpoints" / f"{atlas}__{model}__fold{fold}.json"


def save_ck(path, data):
    tmp = path.with_suffix(".json.tmp")
    data["_complete"] = True
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_ck(path):
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return d if d.get("_complete") is True else None


def run_fold(atlas, model, fold, X, y, tr, te, memory):
    make, grid = MODELS[model]
    gs = GridSearchCV(pipe(make(), memory), grid, scoring="roc_auc", cv=StratifiedKFold(N_INNER, shuffle=True, random_state=SEED),
                      n_jobs=6, refit=True, error_score="raise")
    t0 = time.time()
    gs.fit(X[tr], y[tr])
    est = gs.best_estimator_
    score = est.predict_proba(X[te])[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X[te])
    pred = est.predict(X[te])
    m = metrics(y[te], pred, score)
    return {"atlas": atlas, "model": model, "fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)),
            "best_params": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in gs.best_params_.items()},
            "inner_cv_auc": float(gs.best_score_), "seconds": round(time.time() - t0, 1), "metrics": m,
            "test_idx": [int(i) for i in te], "score": [float(s) for s in score], "pred": [int(p) for p in pred]}


def leakage_check(X, y, tr, te, memory):
    """Pipeline predictions for a test row must not depend on any other test row."""
    p = pipe(LogisticRegression(class_weight="balanced", max_iter=2000), memory).set_params(sel__k=500, clf__C=1e-3).fit(X[tr], y[tr])
    base = p.predict_proba(X[te])[:, 1]
    Xj = X[te].copy()
    half = np.arange(len(te)) % 2 == 1
    Xj[half] = np.random.default_rng(0).uniform(-0.9, 0.9, Xj[half].shape)
    alt = p.predict_proba(Xj)[:, 1]
    return bool(np.allclose(base[~half], alt[~half], atol=1e-12))


def main():
    smoke = "--smoke" in sys.argv
    t_start = time.time()
    log("=" * 60 + f"\nFAST ACCURACY EXPERIMENT start (smoke={smoke})")
    dev, train_ids = build_dev_set()
    y = dev["y"].values
    log(f"Development set: {len(dev)} subjects ({int(y.sum())} ASD / {int((1 - y).sum())} TC), {dev['site'].nunique()} sites; "
        f"only {TRAIN_LIST.name} was opened for split info (locked holdout file never opened).")
    folds = list(StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    te_all = np.concatenate([te for _, te in folds])
    assert len(te_all) == len(dev) and len(set(te_all)) == len(dev), "outer test folds must partition the development set"
    assert all(not (set(tr) & set(te)) for tr, te in folds)
    dev.assign(outer_fold=np.nan).pipe(lambda d: d.assign(outer_fold=[next(k for k, (_, te) in enumerate(folds) if i in set(te)) for i in range(len(d))])) \
        .to_csv(OUT / "data" / "dev_split_manifest.csv", index=False)
    memory = None  # (Pipeline memory caching cannot hash a __main__ class; steps are cheap, so recompute per fit)
    results, status = [], {}
    atlases = ["AAL116"] if smoke else list(ATLASES)
    for atlas in atlases:
        X = load_fc(dev, atlas)
        log(f"[{atlas}] feature matrix {X.shape}, NaN entries {int(np.isnan(X).sum())} (imputed inside folds only)")
        tr0, te0 = folds[0]
        ok = leakage_check(X, y, tr0, te0, memory)
        log(f"[{atlas}] leakage perturbation check (test rows independent of each other): {'PASS' if ok else 'FAIL'}")
        assert ok
        for model in (["LR_L2"] if smoke else MODELS):
            spent = 0.0
            status[(atlas, model)] = "complete"
            for fold, (tr, te) in enumerate(folds):
                if smoke and fold > 0:
                    break
                ck = None if smoke else load_ck(ck_path(atlas, model, fold))
                if ck is None:
                    if spent > PER_MODEL_BUDGET_S:
                        status[(atlas, model)] = f"ABORTED_BUDGET_after_fold{fold - 1}"
                        log(f"[{atlas}/{model}] time budget {PER_MODEL_BUDGET_S}s exceeded -- stopping this candidate cleanly")
                        break
                    ck = run_fold(atlas, model, fold, X, y, tr, te, memory)
                    spent += ck["seconds"]
                    if not smoke:
                        save_ck(ck_path(atlas, model, fold), ck)
                    log(f"[{atlas}/{model}] fold {fold}: acc={ck['metrics']['accuracy']:.4f} auc={ck['metrics']['roc_auc']:.4f} "
                        f"inner_auc={ck['inner_cv_auc']:.4f} params={ck['best_params']} ({ck['seconds']}s)")
                else:
                    spent += ck["seconds"]
                    log(f"[{atlas}/{model}] fold {fold}: loaded checkpoint")
                results.append(ck)
    if smoke:
        log(f"SMOKE TEST done in {time.time() - t_start:.0f}s")
        return
    write_outputs(dev, y, results, status, time.time() - t_start)


def write_outputs(dev, y, results, status, total_s):
    rows, oof, comp = [], [], []
    for r in results:
        rows.append({"atlas": r["atlas"], "model": r["model"], "fold": r["fold"], **{k: v for k, v in r["metrics"].items()},
                     "inner_cv_auc": r["inner_cv_auc"], "seconds": r["seconds"], "best_params": json.dumps(r["best_params"])})
        for i, s, p in zip(r["test_idx"], r["score"], r["pred"]):
            oof.append({"atlas": r["atlas"], "model": r["model"], "fold": r["fold"], "subject_id": dev["sid"].iloc[i], "site": dev["site"].iloc[i],
                        "y_true": int(y[i]), "score": s, "y_pred": p})
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "fold_results.csv", index=False)
    pd.DataFrame(oof).to_csv(OUT / "oof_predictions.csv", index=False)
    for (atlas, model), g in fr.groupby(["atlas", "model"]):
        st = status.get((atlas, model), "complete")
        complete = len(g) == N_OUTER and st == "complete"
        row = {"atlas": atlas, "model": model, "status": st if not complete else "complete", "folds_done": len(g)}
        for m in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]:
            row[f"{m}_mean"], row[f"{m}_sd"] = float(g[m].mean()), float(g[m].std(ddof=1)) if len(g) > 1 else float("nan")
        row["runtime_s"] = float(g["seconds"].sum())
        sub = pd.DataFrame([o for o in oof if o["atlas"] == atlas and o["model"] == model])
        row["pooled_oof_confusion"] = json.dumps({k: int(g[k].sum()) for k in ["tn", "fp", "fn", "tp"]})
        comp.append(row)
    cmp_df = pd.DataFrame(comp).sort_values("accuracy_mean", ascending=False)
    cmp_df.to_csv(OUT / "model_comparison.csv", index=False)
    done = cmp_df[cmp_df.status == "complete"]
    best = done.iloc[0] if len(done) else None
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "n_dev": int(len(dev)), "n_asd": int(y.sum()), "n_tc": int((1 - y).sum()),
                "n_sites": int(dev["site"].nunique()), "outer_cv": "StratifiedKFold(5, shuffle, seed 42) on dev subjects", "inner_cv": "StratifiedKFold(3, shuffle, seed 42)",
                "split_source": str(TRAIN_LIST), "holdout_file_opened": False, "train_list_sha256": __import__("hashlib").sha256(open(TRAIN_LIST, "rb").read()).hexdigest(),
                "dev_set_definition": "phase2_train_subjects.txt INTERSECT results/phase2/fd_filtered_subjects.csv", "total_runtime_s": round(total_s, 1),
                "checks": {"unique_subjects": True, "outer_test_folds_partition_dev_set": True, "train_test_disjoint_each_fold": True,
                           "pipeline_fits_inside_folds_only": True, "perturbation_check": "PASS (see log)"}}
    json.dump(manifest, open(OUT / "data" / "data_and_split_manifest.json", "w"), indent=2)
    L = ["# Fast accuracy experiment -- summary\n",
         f"Development subjects: {len(dev)} ({int(y.sum())} ASD / {int((1 - y).sum())} TC), {dev['site'].nunique()} sites. Outer 5-fold CV, inner 3-fold ROC-AUC grid search, all learned steps inside folds.\n",
         "These are CROSS-VALIDATION estimates on development subjects. The locked 170-subject holdout was never opened; no independent test was run.\n\n",
         "```\n" + cmp_df[["atlas", "model", "status", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "roc_auc_mean", "roc_auc_sd", "runtime_s"]].round(4).to_string(index=False) + "\n```\n"]
    if best is not None:
        L.append(f"\nBest completed candidate by mean CV accuracy: **{best.atlas} / {best.model}**: accuracy {best.accuracy_mean:.4f} +/- {best.accuracy_sd:.4f}, "
                 f"balanced accuracy {best.balanced_accuracy_mean:.4f}, AUC {best.roc_auc_mean:.4f} +/- {best.roc_auc_sd:.4f}. "
                 "Choosing the best of several candidates by CV score is mildly optimistic.\n")
        L.append(f"\n**85% target reached: {'YES' if best.accuracy_mean >= 0.85 else 'NO'}** (best measured CV accuracy {best.accuracy_mean:.1%}).\n")
    (OUT / "experiment_summary.md").write_text("".join(L), encoding="utf-8")
    log(f"Outputs written. Total runtime {total_s:.0f}s")
    log("\n" + cmp_df[["atlas", "model", "status", "accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "roc_auc_mean", "runtime_s"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
