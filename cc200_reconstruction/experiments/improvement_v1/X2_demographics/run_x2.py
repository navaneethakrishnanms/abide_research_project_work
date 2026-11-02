"""X2: demographic handling, arm D2 (age/sex residualization) vs baseline Arm B -- improvement_plan_v1.md section 5 and x2_definition.md.  CC200 only; 676 authorized training subjects only;
reads only the verified feature cache. Modes: all | replicate (seeds 3-5, only if the frozen rule is provisionally met).
Never opened: holdout list, RESTRICTED_id_membership.csv, ts_cc200.npy, covs_cc200.npy, any holdout label/feature/prediction. No AAL-116. No D1, X3, X4 or LOSO here."""
import argparse
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
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path(r"C:\abide\cc200_reconstruction\experiments\stage1_2_b0_baseline"); X1 = HERE.parent / "X1_site"
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cc200_residualizer import AgeSexResidualizer

for s in ("checkpoints", "logs"):
    (HERE / s).mkdir(exist_ok=True)
LOG = HERE / "logs" / "x2.log"
C_GRID, N_OUTER, N_INNER, SEEDS, REPL_SEEDS = [0.001, 0.01, 0.1, 1.0, 10.0], 10, 5, (0, 1, 2), (3, 4, 5)
HASHES = {"features_cc200_train676.npz": "df6e7f9a3d15c7d659c00c2ed0d24d497d6be58d039856b5fb4ce795d39fd43c", "subject_manifest.csv": "22cb64022ab5947716993b82f19be2814d469f3776241b3a36b5c10b3e93ea7a",
          "seed_metrics.csv": "4211d0305ea384cb77075ab2d0a01d1f8b958899c7b5b27b2e36095d2d4c3c46", "fold_metrics.csv": "bab680c18643666d08ced0c2d30557ccba546058fd0d49b456ceb1fd8c872f2e",
          "oof_predictions.csv": "28dc62c095828f68e76d0d2bbd5496c61b472f742e635207bcce72341e73682e", "stage1_checks.json": "c3f835248dd489221de957994d96f716b3abe06379ae655dd0e5e5a49c94cb55"}


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


def x1_hashes():
    return {p.name: sha(p) for p in sorted(X1.iterdir()) if p.is_file()}


def verify_inputs():
    bad = [k for k, v in HASHES.items() if sha(BASE / k) != v]
    log(f"hash verification of {len(HASHES)} baseline/cache files: {'ALL MATCH' if not bad else 'MISMATCH ' + str(bad)}"); assert not bad, bad
    z = np.load(BASE / "features_cc200_train676.npz", allow_pickle=True)
    tl = {str(int(float(l))) for l in open(r"C:\abide\splits\phase2_train_subjects.txt") if l.strip()}
    assert set(z["sid"].astype(str)) == tl and len(tl) == 676 and json.load(open(BASE / "stage1_checks.json"))["all_pass"]
    log("cache subject IDs == authorized training list (676); Stage 1 all_pass"); return z


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=5000, solver="lbfgs", class_weight="balanced")


def pipe(kind):
    if kind == "base":                                                        # baseline Arm B: demographics appended (DemoFCPrep refitted per inner fold)
        from b0_components import DemoFCPrep
        return Pipeline([("prep", DemoFCPrep()), ("clf", lr())])
    return Pipeline([("resid", AgeSexResidualizer()), ("scale", StandardScaler()), ("clf", lr())])


def metrics(y, pred, score):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    m = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=recall_score(y, pred), specificity=tn / (tn + fp), precision=precision_score(y, pred, zero_division=0),
             f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, score), TP=tp, FN=fn, FP=fp, TN=tn)
    assert abs(m["accuracy"] - (tp + tn) / len(y)) < 1e-12 and tp + fn == int(y.sum()); return m


def self_test():
    rng = np.random.default_rng(2); n, p = 400, 60
    age = rng.normal(15, 5, n); sex = rng.integers(1, 3, n).astype(float); fem = (sex == 2).astype(float)
    F = rng.normal(0, 1, (n, p)) + 0.15 * (age[:, None] - 15) * np.linspace(-1, 1, p) + 0.8 * fem[:, None] * np.linspace(1, -1, p)
    X = np.hstack([F, age[:, None], sex[:, None]]); tr = np.arange(n) < 300; te = ~tr
    r = AgeSexResidualizer().fit(X[tr]); out = r.transform(X[te]); assert out.shape == (te.sum(), p)
    cb = np.abs([np.corrcoef(F[te][:, j], age[te])[0, 1] for j in range(p)]).mean(); ca = np.abs([np.corrcoef(out[:, j], age[te])[0, 1] for j in range(p)]).mean(); assert ca < 0.5 * cb, (cb, ca)
    assert np.allclose(AgeSexResidualizer().fit(X[tr], rng.integers(0, 2, tr.sum())).coef_, r.coef_), "fit depends on y"
    c0 = r.coef_.copy(); X2 = X[te].copy(); X2[:10, :p] += 9.0; X2[:10, -2] += 30; o2 = r.transform(X2)
    assert np.array_equal(r.coef_, c0) and np.array_equal(out[10:], o2[10:]) and not np.array_equal(out[:10], o2[:10]), "test rows influence the fit or each other"
    assert np.array_equal(AgeSexResidualizer().fit(X[tr]).transform(X[te]), out), "not deterministic"
    tr_out = r.transform(X[tr]); assert np.abs(tr_out.mean(0)).max() < 1e-8                         # OLS residuals are centered on the training rows
    log(f"residualizer self-test PASS (synthetic): held-out mean |corr(feature, age)| {cb:.3f} -> {ca:.3f}; fit ignores y; test rows cannot influence the fit or each other; deterministic")


def load():
    z = verify_inputs(); y, sid = z["y"], z["sid"].astype(str)
    Xa = np.hstack([z["X"], z["age"][:, None], z["sex"][:, None]]); assert Xa.shape == (676, 19902) and np.isfinite(Xa).all()
    assert set(np.unique(z["sex"])) == {1.0, 2.0}, "unexpected sex coding"
    return Xa, y, sid, z["site"].astype(str)


def run_cv(kind, seed, X, y, tag):
    ck = HERE / "checkpoints" / f"{tag}_seed{seed}.npz"
    if ck.exists():
        z = np.load(ck); return {k: z[k] for k in z.files}
    n = len(y); score, fold = np.zeros(n), np.zeros(n, int); bc, ia, fs = np.zeros(N_OUTER), np.zeros(N_OUTER), np.zeros(N_OUTER); t0 = time.time()
    for f, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=seed).split(X, y)):
        t1 = time.time()
        gs = GridSearchCV(pipe(kind), {"clf__C": C_GRID}, cv=StratifiedKFold(N_INNER, shuffle=True, random_state=seed), scoring="roc_auc", n_jobs=4, refit=True).fit(X[tr], y[tr])
        score[te], fold[te] = gs.best_estimator_.predict_proba(X[te])[:, 1], f; bc[f], ia[f], fs[f] = gs.best_params_["clf__C"], gs.best_score_, time.time() - t1
        log(f"  {tag} seed {seed} fold {f}: C={bc[f]} inner AUC {ia[f]:.4f} ({fs[f]:.1f}s; seed total {time.time() - t0:.0f}s)")
    out = dict(score=score, fold=fold, bestC=bc, iauc=ia, secs=fs); tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, **out); os.replace(tmp, ck); return out


def baseline_cv(sid, seeds):
    o = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str}); o = o[o.arm == "B"]; out = {}
    for s in seeds:
        d = o[o.seed == s].set_index("subject_id").loc[sid]; out[s] = dict(score=d.score.values, fold=d.fold.values.astype(int), pred=d.pred_thr0_5.values); assert ((d.score.values >= .5).astype(int) == d.pred_thr0_5.values).all()
    return out


def per_fold(y, pred, score, fold):
    return pd.DataFrame([dict(accuracy=accuracy_score(y[fold == f], pred[fold == f]), balanced_accuracy=balanced_accuracy_score(y[fold == f], pred[fold == f]), roc_auc=roc_auc_score(y[fold == f], score[fold == f])) for f in range(N_OUTER)])


def compare(y, R, seeds, base):
    srows, frows = [], []
    for s in seeds:
        bs, bp, bf = base[s]["score"], base[s]["pred"], base[s]["fold"]; hs = R[s]["score"]; hp = (hs >= .5).astype(int); assert (bf == R[s]["fold"]).all(), "outer folds differ from the baseline"
        mb, mh = metrics(y, bp, bs), metrics(y, hp, hs)
        srows.append({"seed": s, **{f"base_{k}": v for k, v in mb.items()}, **{f"D2_{k}": v for k, v in mh.items()}, **{f"d_{k}": mh[k] - mb[k] for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")}})
        fb, fh = per_fold(y, bp, bs, bf), per_fold(y, hp, hs, bf)
        for f in range(N_OUTER):
            frows.append(dict(seed=s, fold=f, n_test=int((bf == f).sum()), base_accuracy=fb.accuracy[f], D2_accuracy=fh.accuracy[f], d_accuracy=fh.accuracy[f] - fb.accuracy[f], base_bacc=fb.balanced_accuracy[f], D2_bacc=fh.balanced_accuracy[f],
                              d_bacc=fh.balanced_accuracy[f] - fb.balanced_accuracy[f], base_auc=fb.roc_auc[f], D2_auc=fh.roc_auc[f], D2_selected_C=float(R[s]["bestC"][f])))
    return pd.DataFrame(srows), pd.DataFrame(frows)


def verdict(SD, FD):                                                         # frozen rule, plan section 4 (criteria 1-4)
    d_acc, d_bacc = SD.d_accuracy.values * 100, SD.d_balanced_accuracy.values * 100; wins = int((FD.d_accuracy > 0).sum())
    c = {"1a mean paired gain >= +1.0 pt accuracy": d_acc.mean() >= 1.0, "1b mean paired gain >= +1.0 pt balanced accuracy": d_bacc.mean() >= 1.0,
         "1c every seed >= +0.5 pt (accuracy AND balanced accuracy; strictest reading)": bool((d_acc >= .5).all() and (d_bacc >= .5).all()), "2 accuracy higher in >= 20 of 30 paired folds": wins >= 20,
         "3a mean AUC not lower by more than 0.005": SD.d_roc_auc.mean() >= -0.005, "3b mean sensitivity not lower by more than 3 pts": SD.d_sensitivity.mean() * 100 >= -3, "3c mean specificity not lower by more than 3 pts": SD.d_specificity.mean() * 100 >= -3}
    return c, dict(mean_d_acc=d_acc.mean(), mean_d_bacc=d_bacc.mean(), wins=wins, ties=int((FD.d_accuracy == 0).sum()), losses=int((FD.d_accuracy < 0).sum()), sign_p=float(stats.binom.sf(wins - 1, 30, .5)),
                   fd_mean=float(FD.d_accuracy.mean()), fd_sd=float(FD.d_accuracy.std(ddof=1)), fd_se=float(FD.d_accuracy.std(ddof=1) / np.sqrt(len(FD))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); a = ap.parse_args(); warnings.filterwarnings("ignore"); t0 = time.time()
    log("=" * 60 + f"\nX2 start, mode={a.mode}"); x1_before = x1_hashes(); self_test(); Xa, y, sid, site_s = load()
    if a.mode == "all":
        for s in SEEDS: run_cv("D2", s, Xa, y, "D2")
        report(y, sid, site_s, Xa, time.time() - t0)
    elif a.mode == "replicate":
        sys.path.insert(1, str(BASE))
        for s in REPL_SEEDS: run_cv("base", s, Xa, y, "rbase"); run_cv("D2", s, Xa, y, "rD2")
        rows = []
        for s in REPL_SEEDS:
            b, d = run_cv("base", s, Xa, y, "rbase"), run_cv("D2", s, Xa, y, "rD2"); rows.append(dict(seed=s, base_acc=accuracy_score(y, (b["score"] >= .5).astype(int)), D2_acc=accuracy_score(y, (d["score"] >= .5).astype(int))))
        R = pd.DataFrame(rows); R["d_acc"] = R.D2_acc - R.base_acc; R.to_csv(HERE / "replication_seeds3_5.csv", index=False); ok = R.d_acc.mean() * 100 >= .5 and int((R.d_acc > 0).sum()) >= 2
        log(f"REPLICATION seeds 3-5: mean gain {R.d_acc.mean() * 100:+.2f} pts, positive in {int((R.d_acc > 0).sum())}/3 -> {'REPLICATED' if ok else 'NOT replicated'}")
    same = x1_hashes() == x1_before; log(f"X1 artifacts unchanged by this run: {same}"); assert same
    log(f"X2 mode {a.mode} finished in {time.time() - t0:.0f}s")


def report(y, sid, site_s, Xa, secs):
    base = baseline_cv(sid, SEEDS); R = {s: run_cv("D2", s, Xa, y, "D2") for s in SEEDS}
    SD, FD = compare(y, R, SEEDS, base); SD.to_csv(HERE / "seed_comparison.csv", index=False); FD.to_csv(HERE / "fold_comparison.csv", index=False); cks, vd = verdict(SD, FD)
    keys = ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1", "TP", "FN", "FP", "TN")
    pd.DataFrame([dict(experiment="X2", arm="baseline_B", seed=int(r.seed), **{k: getattr(r, f"base_{k}") for k in keys}) for r in SD.itertuples()] +
                 [dict(experiment="X2", arm="D2_residualized", seed=int(r.seed), **{k: getattr(r, f"D2_{k}") for k in keys}) for r in SD.itertuples()]).to_csv(HERE / "experiment_results.csv", index=False)
    pd.DataFrame([{"seed": s, "fold": int(R[s]["fold"][i]), "subject_id": sid[i], "site": site_s[i], "y_true": int(y[i]), "score": float(R[s]["score"][i]), "pred_thr0_5": int(R[s]["score"][i] >= .5)} for s in SEEDS for i in range(len(y))]).to_csv(HERE / "oof_predictions.csv", index=False)
    diag = []                                                                 # seed-0 outer partitions: how much feature variance age/sex explain, and association with age before/after on held-out rows
    for f, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=0).split(Xa, y)):
        r = AgeSexResidualizer().fit(Xa[tr]); F, res = Xa[te][:, :-2], r.transform(Xa[te]); Ftr, rtr = Xa[tr][:, :-2], r.transform(Xa[tr]); age = Xa[te][:, -2]
        r2 = 1 - ((rtr - rtr.mean(0)) ** 2).sum(0) / ((Ftr - Ftr.mean(0)) ** 2).sum(0)
        cor = lambda M: float(np.nanmean(np.abs([np.corrcoef(M[:, j], age)[0, 1] for j in range(0, M.shape[1], 20)])))
        diag.append(dict(fold=f, mean_R2_age_sex_train=float(np.nanmean(r2)), mean_abs_corr_age_heldout_before=cor(F), mean_abs_corr_age_heldout_after=cor(res)))
    DG = pd.DataFrame(diag); DG.to_csv(HERE / "residualization_diagnostic.csv", index=False)
    prov = all(cks.values())
    json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "definition": "x2_definition.md; improvement_plan_v1.md section 5 arm D2", "reference": "baseline Arm B, fixed 0.5, saved predictions", "seeds": list(SEEDS), "grid_C": C_GRID, "hashes_verified": HASHES,
               "criteria": {k: bool(v) for k, v in cks.items()}, "provisional_pass_before_replication": bool(prov), "details": vd, "runtime_s": round(secs, 1)}, open(HERE / "config.json", "w"), indent=1)
    pc = lambda v: f"{v * 100:.2f}%"
    st = "\n".join(f"| {int(r.seed)} | {pc(r.base_accuracy)} -> {pc(r.D2_accuracy)} ({r.d_accuracy * 100:+.2f}) | {pc(r.base_balanced_accuracy)} -> {pc(r.D2_balanced_accuracy)} ({r.d_balanced_accuracy * 100:+.2f}) | {r.base_roc_auc:.4f} -> {r.D2_roc_auc:.4f} | {pc(r.base_sensitivity)} -> {pc(r.D2_sensitivity)} | {pc(r.base_specificity)} -> {pc(r.D2_specificity)} | {pc(r.base_precision)} -> {pc(r.D2_precision)} | {pc(r.base_f1)} -> {pc(r.D2_f1)} |" for r in SD.itertuples())
    ct = "\n".join(f"| {a} | seed {int(r.seed)} | {int(getattr(r, a2 + '_TP'))} | {int(getattr(r, a2 + '_FN'))} | {int(getattr(r, a2 + '_FP'))} | {int(getattr(r, a2 + '_TN'))} |" for r in SD.itertuples() for a, a2 in (("baseline", "base"), ("D2 residualized", "D2")))
    ft = "\n".join(f"| {int(r.seed)} | {int(r.fold)} | {pc(r.base_accuracy)} | {pc(r.D2_accuracy)} | {r.d_accuracy * 100:+.2f} | {r.D2_selected_C} |" for r in FD.itertuples())
    kt = "\n".join(f"| {'PASS' if v else 'FAIL'} | {k} |" for k, v in cks.items())
    txt = f"""# X2 report: demographic handling (baseline Arm B vs D2 = in-fold age/sex residualization)
Total runtime {secs / 60:.1f} min. Development estimates on the 676 authorized training subjects (CC200 only); holdout and combined arrays not touched. Frozen rule: `improvement_plan_v1.md` section 4.

## 1. 10-fold x 3-seed CV, fixed threshold 0.5 (same outer folds; baseline = saved Arm B predictions)
| Seed | Accuracy base -> D2 (diff pts) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|
{st}
Mean over seeds: accuracy {pc(SD.base_accuracy.mean())} -> {pc(SD.D2_accuracy.mean())} ({vd['mean_d_acc']:+.2f} pts); balanced accuracy {pc(SD.base_balanced_accuracy.mean())} -> {pc(SD.D2_balanced_accuracy.mean())} ({vd['mean_d_bacc']:+.2f} pts); AUC {SD.base_roc_auc.mean():.4f} -> {SD.D2_roc_auc.mean():.4f};
sensitivity {pc(SD.base_sensitivity.mean())} -> {pc(SD.D2_sensitivity.mean())}; specificity {pc(SD.base_specificity.mean())} -> {pc(SD.D2_specificity.mean())}.
Paired outer folds (30): D2 higher in {vd['wins']}, equal in {vd['ties']}, lower in {vd['losses']}; paired difference mean {vd['fd_mean'] * 100:+.2f} pts, SD {vd['fd_sd'] * 100:.2f}, SE {vd['fd_se'] * 100:.2f}; one-sided sign-test p for >= {vd['wins']} wins: {vd['sign_p']:.3f}.
Confusion matrices (pooled out-of-fold predictions, 313 ASD / 363 TC):
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
{ct}
Fold-level comparison:
| Seed | Fold | Baseline acc. | D2 acc. | Diff (pts) | Selected C (D2) |
|---|---|---|---|---|---|
{ft}

## 2. Residualization diagnostic (seed-0 outer-training sets)
Mean per-feature R^2 of age + sex on training rows: {DG.mean_R2_age_sex_train.mean():.4f}. Mean |correlation(feature, age)| on held-out rows: {DG.mean_abs_corr_age_heldout_before.mean():.4f} before -> {DG.mean_abs_corr_age_heldout_after.mean():.4f} after (subsample of every 20th feature).

## 3. Frozen acceptance rule applied
| Result | Criterion |
|---|---|
{kt}
Provisional outcome (criteria 1-4): **{'ALL MET -> replication on seeds 3-5 is required before calling it validated' if prov else 'NOT MET -> variant not retained; rule not changed'}**.
"""
    (HERE / "report.md").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
