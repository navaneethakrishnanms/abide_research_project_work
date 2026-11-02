"""Stage E engine. CC200, 676 authorized subjects. Arms: none (reference reproduction gate), qc (E1), fdres (E2), site (E3), enet (E4).
Every learned step is a Pipeline step (fitted on the training rows of each inner/outer split). Reuses campaign helpers read-only (import only)."""
import os, sys, json, time, hashlib, warnings, argparse
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; BASE = EXP.parent / "stage1_2_b0_baseline"
sys.path[:0] = [str(EXP), str(BASE), r"C:\abide"]
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import Parallel, delayed
from b0_components import DemoFCPrep
import campaign as CP                       # import only: choose_threshold, metrics, C_GRID (campaign.py is not modified; no campaign function that logs is called)
warnings.filterwarnings("ignore")
LOG = HERE / "logs" / "experiments.log"; CK = HERE / "checkpoints"; NF = 19900


def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {m}\n")
    print(m, flush=True)


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    z = np.load(BASE / "features_cc200_train676.npz", allow_pickle=True)
    tl = {str(int(float(l))) for l in open(r"C:\abide\splits\phase2_train_subjects.txt") if l.strip()}
    sid = z["sid"].astype(str); assert set(sid) == tl and len(sid) == 676
    assert sha(BASE / "features_cc200_train676.npz") == CP.HASHES["features_cc200_train676.npz"]
    ph = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv", low_memory=False); ph["sid"] = ph.SUB_ID.astype(int).astype(str)
    ph = ph[ph.sid.isin(tl)].drop_duplicates("sid").set_index("sid").loc[sid]
    y = z["y"]; site = z["site"].astype(str); assert (ph.SITE_ID.values == site).all() and ((ph.DX_GROUP.values == 1).astype(int) == y).all()
    fd = ph.func_mean_fd.values.astype(float); sidx = pd.factorize(site)[0].astype(float)
    X = np.hstack([z["X"], z["age"][:, None], z["sex"][:, None], fd[:, None], sidx[:, None]]); assert X.shape == (676, NF + 4) and np.isfinite(X[:, NF + 2:]).all()
    return X, y, site, sid, fd


class Pre(BaseEstimator, TransformerMixin):
    """X = [FC(19900) | age | sex | FD | site_index]. Returns [FC' | age | sex]; all statistics fitted on the rows passed to fit()."""
    def __init__(self, mode="none", min_site=10): self.mode, self.min_site = mode, min_site

    def fit(self, X, y=None):
        F, fd, s = X[:, :NF], X[:, NF + 2], X[:, NF + 3].astype(int)
        if self.mode == "fdres":
            self.fm_, self.Fm_ = fd.mean(), F.mean(0); d = fd - self.fm_; self.beta_ = (d @ (F - self.Fm_)) / (d @ d)
        if self.mode == "site":
            cnt = np.bincount(s, minlength=int(X[:, NF + 3].max()) + 20); grp = np.where(cnt[s] >= self.min_site, s, -1); self.st_ = {}
            for g in np.unique(grp):
                m = grp == g; sd = F[m].std(0, ddof=1); self.st_[int(g)] = (F[m].mean(0), np.where(sd > 1e-8, sd, 1.0))
            self.cnt_ = cnt
        return self

    def transform(self, X):
        F, fd, s = X[:, :NF], X[:, NF + 2], X[:, NF + 3].astype(int)
        if self.mode == "fdres": F = F - np.outer(fd - self.fm_, self.beta_)
        if self.mode == "site":
            F = F.copy()
            for i, si in enumerate(s):
                if si < len(self.cnt_) and self.cnt_[si] >= self.min_site: g = int(si)
                elif si < len(self.cnt_) and self.cnt_[si] > 0: g = -1           # site seen in training but pooled
                else: continue                                                 # unseen site: unchanged
                if g in self.st_: mu, sd = self.st_[g]; F[i] = (F[i] - mu) / sd
        return np.hstack([F, X[:, NF:NF + 2]])


def make(arm):
    mode = {"none": "none", "qc": "none", "fdres": "fdres", "site": "site", "enet": "none"}[arm]
    if arm == "enet":
        clf = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", max_iter=300, tol=1e-3, random_state=0)
        grid = {"clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.3, 0.7]}
    else:
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", class_weight="balanced"); grid = {"clf__C": CP.C_GRID}
    return Pipeline([("pre", Pre(mode)), ("prep", DemoFCPrep()), ("clf", clf)]), grid


def fold_job(arm, seed, f, tr, te, X, y, keep):
    ck = CK / f"{arm}_seed{seed}_fold{f}.npz"
    if ck.exists():
        z = np.load(ck, allow_pickle=True); return f, dict(z)
    t0 = time.time(); assert not (set(tr.tolist()) & set(te.tolist()))
    tru = tr[keep[tr]] if arm == "qc" else tr; teu = te[keep[te]] if arm == "qc" else te
    est, grid = make(arm)
    gs = GridSearchCV(est, grid, cv=StratifiedKFold(CP.N_INNER, shuffle=True, random_state=seed), scoring="roc_auc", n_jobs=1, refit=True, error_score=np.nan).fit(X[tru], y[tru])
    best = gs.best_estimator_; s = best.predict_proba(X[teu])[:, 1]
    io = cross_val_predict(clone(best), X[tru], y[tru], cv=StratifiedKFold(CP.N_INNER, shuffle=True, random_state=seed + 1000), method="predict_proba", n_jobs=1)[:, 1]
    thr = CP.choose_threshold(y[tru], io, 0.5)
    out = dict(te=teu, score=s, thr=np.float64(thr), params=np.array(json.dumps({k: float(v) for k, v in gs.best_params_.items()})), iauc=np.float64(gs.best_score_), secs=np.float64(time.time() - t0), n_train=np.int64(len(tru)))
    tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, **out); os.replace(tmp, ck); return f, out


def run(arm, seeds, njobs):
    X, y, site, sid, fd = load(); keep = fd <= 0.2
    for seed in seeds:
        final = CK / f"{arm}_seed{seed}.npz"
        if final.exists(): continue
        t0 = time.time(); splits = list(StratifiedKFold(CP.N_OUTER, shuffle=True, random_state=seed).split(X, y))
        res = Parallel(n_jobs=njobs)(delayed(fold_job)(arm, seed, f, tr, te, X, y, keep) for f, (tr, te) in enumerate(splits))
        score = np.full(676, np.nan); fold = np.zeros(676, int); thr = np.zeros(CP.N_OUTER); secs = 0.0
        for f, o in res: score[o["te"]] = o["score"]; fold[splits[f][1]] = f; thr[f] = o["thr"]; secs += float(o["secs"])
        np.savez(final, score=score, fold=fold, thr=thr, secs=secs, params=np.array([str(o["params"]) for f, o in res]))
        log(f"{arm} seed {seed} done: scored {int(np.isfinite(score).sum())} subjects, cpu {secs:.0f}s, wall {time.time() - t0:.0f}s")


def gate(seeds):
    """'none' arm must reproduce the saved reference LR scores (campaign P1_LR for seeds 0-2; next_phase predictions for others)."""
    ref = pd.read_csv(EXP / "next_phase_tangent_audit" / "predictions.csv", dtype={"subject_id": str}); ref = ref[ref.arm == "LR (reference)"]
    X, y, site, sid, fd = load(); ok = True
    for s in seeds:
        z = np.load(CK / f"none_seed{s}.npz"); r = ref[ref.seed == s].set_index("subject_id").loc[sid]
        d = float(np.abs(z["score"] - r.score.values).max()); ok &= d < 1e-8 and (z["fold"] == r.fold.values).all(); log(f"gate seed {s}: max|score diff| {d:.2e}")
    log(f"REFERENCE REPRODUCTION GATE: {'PASS' if ok else 'FAIL'}"); return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--arms", default=""); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--njobs", type=int, default=10); a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.mode == "run":
        for arm in a.arms.split(","): run(arm, seeds, a.njobs)
    if a.mode == "gate": sys.exit(0 if gate(seeds) else 1)
