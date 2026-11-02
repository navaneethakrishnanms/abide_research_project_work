"""CC200 accuracy campaign engine (plan: campaign_plan.md).  CC200 only; the 676 authorized training subjects only; reads the verified feature cache and per-subject raw CC200 files of the
authorized IDs. Never opened: holdout list, RESTRICTED_id_membership.csv, ts_cc200.npy, covs_cc200.npy, any holdout label/feature/prediction. No AAL-116.
Modes: prep | bench | run --exps A,B --seeds 0,1,2
Every fitted step (scaling, selection, PCA, tangent reference, hyperparameters, thresholds) is fitted on the outer-training partition (or inside inner folds for the pipeline steps)."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path(r"C:\abide\cc200_reconstruction\experiments\stage1_2_b0_baseline"); RAW = Path(r"C:\abide\abide_pcp\ABIDE_pcp\cpac\filt_global")
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(BASE)); sys.path.insert(2, r"C:\abide")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from b0_components import DemoFCPrep
from tangent import TangentEmbedding, covariances

for s in ("checkpoints", "logs", "cache", "results"):
    (HERE / s).mkdir(exist_ok=True)
LOG = HERE / "logs" / "campaign.log"
C_GRID, N_OUTER, N_INNER = [0.001, 0.01, 0.1, 1.0, 10.0], 10, 5
G0 = 1.0 / 19902                                            # sklearn gamma='scale' for 19,902 standardized features
HASHES = {"features_cc200_train676.npz": "df6e7f9a3d15c7d659c00c2ed0d24d497d6be58d039856b5fb4ce795d39fd43c", "subject_manifest.csv": "22cb64022ab5947716993b82f19be2814d469f3776241b3a36b5c10b3e93ea7a",
          "seed_metrics.csv": "4211d0305ea384cb77075ab2d0a01d1f8b958899c7b5b27b2e36095d2d4c3c46", "fold_metrics.csv": "bab680c18643666d08ced0c2d30557ccba546058fd0d49b456ceb1fd8c872f2e",
          "oof_predictions.csv": "28dc62c095828f68e76d0d2bbd5496c61b472f742e635207bcce72341e73682e", "stage1_checks.json": "c3f835248dd489221de957994d96f716b3abe06379ae655dd0e5e5a49c94cb55"}
EXPS = {"P1_LR": ("LR", "LW"), "P2_LSVM": ("LSVM", "LW"), "P2_RBF": ("RBF", "LW"), "P2_RF": ("RF", "LW"), "P2_ET": ("ET", "LW"), "P2_XGB": ("XGB", "LW"),
        "P3_PC_LR": ("LR", "PC"), "P3_TS_LR": ("LR", "TS"), "P4_PCA_LR": ("PCA_LR", "LW"), "P4_KBEST_LR": ("KBEST_LR", "LW"), "P4_L1_LR": ("L1_LR", "LW")}


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=5000, solver="lbfgs", class_weight="balanced")


def make(model, memory=None):
    prep = ("prep", DemoFCPrep()); k2000 = ("sel", SelectKBest(f_classif, k=2000))
    if model == "LR": return Pipeline([prep, ("clf", lr())]), {"clf__C": C_GRID}, "proba"
    if model == "LSVM": return Pipeline([prep, ("clf", LinearSVC(class_weight="balanced", dual="auto", max_iter=10000))]), {"clf__C": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]}, "margin"
    if model == "RBF": return Pipeline([prep, ("clf", SVC(kernel="rbf", class_weight="balanced", cache_size=500))]), {"clf__C": [1.0, 10.0], "clf__gamma": [G0 / 3, G0]}, "margin"
    if model == "RF": return Pipeline([prep, k2000, ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=0))]), {"clf__min_samples_leaf": [1, 5]}, "proba"
    if model == "ET": return Pipeline([prep, k2000, ("clf", ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=0))]), {"clf__min_samples_leaf": [1, 5]}, "proba"
    if model == "XGB": return Pipeline([prep, k2000, ("clf", XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, tree_method="hist", n_jobs=1, eval_metric="logloss", random_state=0))]), {"clf__max_depth": [2, 3]}, "proba"
    if model == "PCA_LR": return Pipeline([prep, ("pca", PCA(whiten=True, random_state=0)), ("clf", lr())], memory=memory), {"pca__n_components": [25, 50, 100], "clf__C": C_GRID}, "proba"
    if model == "KBEST_LR": return Pipeline([prep, ("sel", SelectKBest(f_classif)), ("clf", lr())], memory=memory), {"sel__k": [500, 2000, 5000], "clf__C": C_GRID}, "proba"
    if model == "L1_LR": return Pipeline([prep, ("sel", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=1000))), ("clf", lr())], memory=memory), {"sel__estimator__C": [0.05, 0.2], "clf__C": [0.01, 0.1, 1.0]}, "proba"
    raise KeyError(model)


def verify_inputs():
    bad = [k for k, v in HASHES.items() if sha(BASE / k) != v]
    log(f"hash verification of {len(HASHES)} baseline/cache files: {'ALL MATCH' if not bad else 'MISMATCH ' + str(bad)}"); assert not bad, bad
    z = np.load(BASE / "features_cc200_train676.npz", allow_pickle=True)
    tl = {str(int(float(l))) for l in open(r"C:\abide\splits\phase2_train_subjects.txt") if l.strip()}
    sid = z["sid"].astype(str); assert set(sid) == tl and len(tl) == 676 and len(set(sid)) == 676 and json.load(open(BASE / "stage1_checks.json"))["all_pass"]
    log("cache subject IDs == authorized training list (676, unique); Stage 1 all_pass"); return z


def load():
    z = verify_inputs(); y = z["y"]; Xa = np.hstack([z["X"], z["age"][:, None], z["sex"][:, None]]); assert Xa.shape == (676, 19902)
    cov_p, pc_p = HERE / "cache" / "covs_train676.npy", HERE / "cache" / "features_partial_train676.npz"
    prov = json.load(open(HERE / "cache" / "provenance.json")) if (HERE / "cache" / "provenance.json").exists() else None
    covs = Xp = None
    if prov:
        assert sha(cov_p) == prov["covs_sha256"] and sha(pc_p) == prov["partial_sha256"], "cache provenance hash mismatch"
        covs = np.load(cov_p); Xp = np.hstack([np.load(pc_p)["Xp"], Xa[:, -2:]])
    return Xa, y, z["site"].astype(str), z["sid"].astype(str), covs, Xp


# ------------------------------------------------------------------ prep: covariances and partial-correlation features (authorized subjects only), with provenance
def prep():
    z = verify_inputs(); man = pd.read_csv(BASE / "subject_manifest.csv", dtype={"sub_id": str}); assert list(man.sub_id) == list(z["sid"].astype(str))
    ts = [np.loadtxt(RAW / f"{fid}_rois_cc200.1D", dtype=float) for fid in man.file_id]
    covs = covariances(ts); assert covs.shape == (676, 200, 200) and np.isfinite(covs).all()
    Xlw = TangentEmbedding("correlation", precomputed=True).fit_transform(covs); assert np.allclose(Xlw, z["X"], rtol=0, atol=1e-12), "recomputed LW features differ from the verified cache"
    Xp = TangentEmbedding("partial", precomputed=True).fit_transform(covs); assert Xp.shape == (676, 19900) and np.isfinite(Xp).all()
    cond = np.linalg.cond(covs); ev = np.linalg.eigvalsh(covs).min()
    np.save(HERE / "cache" / "covs_train676.npy", covs); np.savez(HERE / "cache" / "features_partial_train676.npz", Xp=Xp)
    json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "source": "per-subject raw CC200 files of the 676 authorized training IDs (manifest file_id order)", "estimator": "tangent.covariances (Ledoit-Wolf, assume_centered) -> TangentEmbedding('partial') = precision-based partial correlation, Fisher-Z, upper triangle x sqrt(2)",
               "covs_sha256": sha(HERE / "cache" / "covs_train676.npy"), "partial_sha256": sha(HERE / "cache" / "features_partial_train676.npz"), "lw_features_match_verified_cache": True, "max_condition_number": float(cond.max()),
               "min_eigenvalue": float(ev), "partial_feature_range": [float(Xp.min()), float(Xp.max())], "subjects_with_constant_rois": int(sum((np.ptp(t, axis=0) == 0).any() for t in ts))},
              open(HERE / "cache" / "provenance.json", "w"), indent=1)
    log(f"prep done: covs (676,200,200) finite; recomputed LW features == verified cache; max cond {cond.max():.1f}, min eigenvalue {ev:.3e}; partial features finite, range [{Xp.min():.3f}, {Xp.max():.3f}]")


# ------------------------------------------------------------------ decision rules
def score_fn(est, X, kind):
    return est.predict_proba(X)[:, 1] if kind == "proba" else est.decision_function(X)


def choose_threshold(y, s, default):
    """threshold maximizing balanced accuracy on (y, s) (inner out-of-fold scores of the outer-training partition); ties -> closest to the default; midpoint between adjacent distinct scores."""
    u = np.unique(s); cand = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]]); P, N = int(y.sum()), int(len(y) - y.sum())
    best, bt = -1.0, default
    for t in cand:
        p = s >= t; b = ((p & (y == 1)).sum() / P + ((~p) & (y == 0)).sum() / N) / 2
        if b > best + 1e-12 or (abs(b - best) <= 1e-12 and abs(t - default) < abs(bt - default)): best, bt = b, t
    return float(bt)


def metrics(y, pred, score):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    m = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=recall_score(y, pred), specificity=tn / (tn + fp), precision=precision_score(y, pred, zero_division=0),
             f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, score), TP=tp, FN=fn, FP=fp, TN=tn)
    assert abs(m["accuracy"] - (tp + tn) / len(y)) < 1e-12 and tp + fn == int(y.sum()); return m


def provider(rep, Xa, Xp, covs):
    if rep == "LW": return lambda tr, te: (Xa[tr], Xa[te])
    if rep == "PC": return lambda tr, te: (Xp[tr], Xp[te])
    demo = Xa[:, -2:]                                                      # TS: tangent reference (geometric mean) fitted on the outer-training covariances only (label-free)

    def f(tr, te):
        t = TangentEmbedding("tangent", precomputed=True).fit(covs[tr]); A, B = t.transform(covs[tr]), t.transform(covs[te])
        assert np.isfinite(A).all() and np.isfinite(B).all(), "non-finite tangent features"
        return np.hstack([A, demo[tr]]), np.hstack([B, demo[te]])
    return f


def run_fold(model, prov_fn, y, tr, te, seed, memory):
    assert not (set(tr.tolist()) & set(te.tolist()))
    Xtr, Xte = prov_fn(tr, te); est, grid, kind = make(model, memory)
    gs = GridSearchCV(est, grid, cv=StratifiedKFold(N_INNER, shuffle=True, random_state=seed), scoring="roc_auc", n_jobs=4, refit=True, error_score=np.nan).fit(Xtr, y[tr])
    best = gs.best_estimator_; s = score_fn(best, Xte, kind)
    io = cross_val_predict(clone(best), Xtr, y[tr], cv=StratifiedKFold(N_INNER, shuffle=True, random_state=seed + 1000), method="predict_proba" if kind == "proba" else "decision_function", n_jobs=4)
    if kind == "proba": io = io[:, 1]
    default = 0.5 if kind == "proba" else 0.0
    return s, io, choose_threshold(y[tr], io, default), default, json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)) for k, v in gs.best_params_.items()}), float(gs.best_score_)


def run_exp(exp, seeds, data):
    model, rep = EXPS[exp]; Xa, y, site_s, sid, covs, Xp = data; prov_fn = provider(rep, Xa, Xp, covs); n = len(y)
    memory = None
    if model in ("PCA_LR", "KBEST_LR", "L1_LR"):
        memory = str(HERE / "cache" / f"joblib_{exp}")
    t_exp = time.time()
    for seed in seeds:
        ck = HERE / "checkpoints" / f"{exp}_seed{seed}.npz"
        if ck.exists(): continue
        score, fold = np.zeros(n), np.zeros(n, int); thr, default, iauc, secs = np.zeros(N_OUTER), np.zeros(N_OUTER), np.zeros(N_OUTER), np.zeros(N_OUTER); io_all = np.full((N_OUTER, n), np.nan); params = []; t0 = time.time()
        for f, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=seed).split(Xa, y)):
            t1 = time.time(); s, io, th, df_, bp, ia = run_fold(model, prov_fn, y, tr, te, seed, memory)
            score[te], fold[te], thr[f], default[f], iauc[f], secs[f] = s, f, th, df_, ia, time.time() - t1; io_all[f, tr] = io; params.append(bp)
            log(f"  {exp} seed {seed} fold {f}: params {bp} inner AUC {ia:.4f} thr {th:.4f} ({secs[f]:.1f}s; seed total {time.time() - t0:.0f}s)")
        tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, score=score, fold=fold, thr=thr, default=default, iauc=iauc, secs=secs, inner_oof=io_all, params=np.array(params)); os.replace(tmp, ck)
    if memory and Path(memory).exists(): shutil.rmtree(memory, ignore_errors=True)
    finalize(exp, seeds, data); log(f"{exp} complete in {time.time() - t_exp:.0f}s")


def finalize(exp, seeds, data):
    Xa, y, site_s, sid, covs, Xp = data; sm, fm, oof = [], [], []
    for seed in seeds:
        z = np.load(HERE / "checkpoints" / f"{exp}_seed{seed}.npz"); sc, fold, thr, dfl = z["score"], z["fold"], z["thr"], z["default"]
        p0 = (sc >= dfl[fold]).astype(int) if True else None; p1 = (sc >= thr[fold]).astype(int)
        for rd, p in (("R0_default", p0), ("R1_nested_threshold", p1)):
            m = metrics(y, p, sc); fa = [accuracy_score(y[fold == f], p[fold == f]) for f in range(N_OUTER)]
            sm.append(dict(experiment=exp, seed=seed, reading=rd, **m, fold_accuracy_mean=np.mean(fa), fold_accuracy_sd=np.std(fa, ddof=1), mean_fold_auc=np.mean([roc_auc_score(y[fold == f], sc[fold == f]) for f in range(N_OUTER)]), runtime_s=float(z["secs"].sum())))
        for f in range(N_OUTER):
            te = fold == f; fm.append(dict(experiment=exp, seed=seed, fold=f, n_test=int(te.sum()), acc_R0=accuracy_score(y[te], p0[te]), acc_R1=accuracy_score(y[te], p1[te]), thr_R1=float(thr[f]), fold_auc=roc_auc_score(y[te], sc[te]),
                                          inner_auc=float(z["iauc"][f]), params=str(z["params"][f]), seconds=float(z["secs"][f])))
        oof += [dict(experiment=exp, seed=seed, fold=int(fold[i]), subject_id=sid[i], site=site_s[i], y_true=int(y[i]), score=float(sc[i]), pred_R0=int(p0[i]), pred_R1=int(p1[i])) for i in range(len(y))]
    pd.DataFrame(sm).to_csv(HERE / "results" / f"{exp}_seed_metrics.csv", index=False); pd.DataFrame(fm).to_csv(HERE / "results" / f"{exp}_fold_metrics.csv", index=False)
    pd.DataFrame(oof).to_csv(HERE / "results" / f"{exp}_oof_predictions_seeds{''.join(map(str, seeds))}.csv", index=False)
    model, rep = EXPS[exp]; est, grid, kind = make(model)
    json.dump({"experiment": exp, "model": model, "representation": rep, "pipeline": str(est), "grid": {k: [str(v) for v in vs] for k, vs in grid.items()}, "score_kind": kind, "outer": "StratifiedKFold(10, shuffle, seed)",
               "inner": "StratifiedKFold(5, shuffle, seed), roc_auc", "threshold_R1": "balanced-accuracy-maximizing threshold on inner 5-fold OOF scores (seed+1000) of the outer-training partition", "seeds": list(seeds),
               "created_utc": datetime.now(timezone.utc).isoformat(), "campaign_py_sha256": sha(HERE / "campaign.py")}, open(HERE / "results" / f"{exp}_config.json", "w"), indent=1)


def baseline_gate(seeds, data):
    """P1_LR must reproduce the saved Arm-B out-of-fold scores and fold assignment of the stage-1/2 baseline exactly (reproduction gate). The R1 threshold rule of this campaign (balanced-accuracy-maximizing
    midpoint on inner OOF scores) is not numerically identical to the stage-2 ROC-Youden threshold, so thresholds are not part of the gate; the R1 accuracy difference is only logged."""
    ok = True
    for s in seeds:
        a, b = np.load(HERE / "checkpoints" / f"P1_LR_seed{s}.npz"), np.load(BASE / "checkpoints" / f"armB_seed{s}.npz")
        d_score = float(np.abs(a["score"] - b["score"]).max()); log(f"  baseline gate seed {s}: max |score diff| {d_score:.2e}; inner AUC identical: {bool(np.allclose(a['iauc'], b['iauc'], atol=1e-12))}")
        ok &= d_score < 1e-8 and (a["fold"] == b["fold"]).all() and bool(np.allclose(a["iauc"], b["iauc"], atol=1e-12))
    log(f"BASELINE REPRODUCTION GATE: {'PASS (exact)' if ok else 'FAIL'}"); return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--exps", default=""); ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args(); warnings.filterwarnings("ignore")
    seeds = tuple(int(s) for s in a.seeds.split(",")); log("=" * 60 + f"\ncampaign start mode={a.mode} exps={a.exps} seeds={seeds}")
    if a.mode == "prep": prep(); return
    data = load()
    if a.mode == "bench":
        Xa, y, site_s, sid, covs, Xp = data; tr, te = next(iter(StratifiedKFold(N_OUTER, shuffle=True, random_state=0).split(Xa, y)))
        for exp in (a.exps.split(",") if a.exps else EXPS):
            model, rep = EXPS[exp]; t1 = time.time(); run_fold(model, provider(rep, Xa, Xp, covs), y, tr, te, 0, None if model not in ("PCA_LR", "KBEST_LR", "L1_LR") else str(HERE / "cache" / "joblib_bench"))
            log(f"BENCH {exp}: one outer fold in {time.time() - t1:.1f}s -> 30 folds ~ {30 * (time.time() - t1) / 60:.1f} min (timing only; scores not inspected)")
        shutil.rmtree(HERE / "cache" / "joblib_bench", ignore_errors=True); return
    for exp in a.exps.split(","):
        t0 = time.time()
        try:
            run_exp(exp, seeds, data)
            if exp == "P1_LR" and seeds == (0, 1, 2): assert baseline_gate(seeds, data), "baseline reproduction failed: stop and investigate"
        except AssertionError:
            raise
        except Exception as e:                                             # a failed variant is recorded and the campaign continues
            log(f"EXPERIMENT FAILED {exp}: {type(e).__name__}: {e}")
        log(f"{exp} finished/failed after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
