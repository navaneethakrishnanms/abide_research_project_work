"""X1: site robustness (Candidate E) -- improvement_plan_v1.md, section 5.  CC200 only; the 676 authorized training subjects only; reads only the verified feature cache.
Modes: smoke | cv | loso | report | replicate.   (replicate = seeds 3-5, run only if the frozen rule is provisionally met)
Never opened: holdout list, RESTRICTED_id_membership.csv, ts_cc200.npy, covs_cc200.npy, any holdout label/feature/prediction. No AAL-116."""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path(r"C:\abide\cc200_reconstruction\experiments\stage1_2_b0_baseline")
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(BASE))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from b0_components import DemoFCPrep                    # baseline preprocessing, read-only import
from cc200_harmonizer import CC200Harmonizer

for s in ("checkpoints", "logs"):
    (HERE / s).mkdir(exist_ok=True)
LOG = HERE / "logs" / "x1.log"
C_GRID, N_OUTER, N_INNER = [0.001, 0.01, 0.1, 1.0, 10.0], 10, 5
SEEDS, REPL_SEEDS, LOSO_INNER_SEED = (0, 1, 2), (3, 4, 5), 0
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
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs():
    bad = [k for k, v in HASHES.items() if sha(BASE / k) != v]
    log(f"hash verification of {len(HASHES)} baseline/cache files: {'ALL MATCH' if not bad else 'MISMATCH ' + str(bad)}"); assert not bad, bad
    z = np.load(BASE / "features_cc200_train676.npz", allow_pickle=True)
    tl = {str(int(float(l))) for l in open(r"C:\abide\splits\phase2_train_subjects.txt") if l.strip()}
    assert set(z["sid"].astype(str)) == tl and len(tl) == 676, "cache subject IDs differ from the authorized training list"
    assert json.load(open(BASE / "stage1_checks.json"))["all_pass"]
    log("cache subject IDs == authorized training list (676); Stage 1 all_pass")
    return z


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=5000, solver="lbfgs", class_weight="balanced")


def pipe(harm):
    return Pipeline(([("harm", CC200Harmonizer())] if harm else []) + [("prep", DemoFCPrep()), ("clf", lr())])


def metrics(y, pred, score):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    m = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=recall_score(y, pred), specificity=tn / (tn + fp), precision=precision_score(y, pred, zero_division=0),
             f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, score), TP=tp, FN=fn, FP=fp, TN=tn)
    assert abs(m["accuracy"] - (tp + tn) / len(y)) < 1e-12 and tp + fn == int(y.sum())
    return m


def eta2(Z, site):
    mu = Z.mean(0); tot = ((Z - mu) ** 2).sum(0); bet = sum((site == g).sum() * (Z[site == g].mean(0) - mu) ** 2 for g in np.unique(site)); ok = tot > 1e-10
    return float((bet[ok] / tot[ok]).mean())


# ---------------------------------------------------------------- validity self-test (synthetic data only)
def self_test():
    rng = np.random.default_rng(1); n, p, ns = 420, 80, 5
    site = rng.integers(0, ns, n); age = rng.normal(15, 5, n); fem = rng.integers(0, 2, n).astype(float)
    shift, scale = rng.normal(0, 0.6, (ns, p)), rng.uniform(0.7, 1.5, (ns, p))
    Z = rng.normal(0, 0.3, (n, p)) * scale[site] + shift[site] + 0.02 * (age[:, None] - 15) * np.linspace(-1, 1, p)
    X = np.hstack([Z, age[:, None], fem[:, None], site[:, None]]); tr = np.arange(n) < 300; te = ~tr
    h = CC200Harmonizer().fit(X[tr]); out = h.transform(X[te])
    b, a = eta2(X[te][:, :p], site[te]), eta2(out[:, :p], site[te]); assert a < 0.2 * b, (b, a)
    assert out.shape == (te.sum(), p + 2) and np.array_equal(out[:, -2:], X[te][:, -3:-1])
    h_y = CC200Harmonizer().fit(X[tr], rng.integers(0, 2, tr.sum())); assert np.array_equal(h.gamma_, h_y.gamma_), "fit depends on y"
    g0 = h.gamma_.copy(); X2 = X[te].copy(); X2[:15, :p] += 7.0; X2[:15, -1] = (X2[:15, -1] + 1) % ns; X2[:15, -3] += 25
    o2 = h.transform(X2); assert np.array_equal(h.gamma_, g0) and np.array_equal(out[15:], o2[15:]) and not np.array_equal(out[:15], o2[:15]), "test rows influence each other or the fit"
    X3 = X[te].copy(); X3[:, -1] = 99; assert np.array_equal(h.transform(X3)[:, :p], X3[:, :p]), "unseen site was altered"
    Xs = X[tr].copy(); Xs[:2, -1] = 77; Xs[2:4, -1] = 78                                           # two sites with 2 subjects each (4 < 5): not estimated
    hs = CC200Harmonizer().fit(Xs); Xq = X[te].copy(); Xq[:6, -1] = 77
    assert np.array_equal(hs.transform(Xq)[:6, :p], Xq[:6, :p]), "tiny site must pass through unchanged"
    assert np.array_equal(CC200Harmonizer().fit(X[tr]).transform(X[te]), out), "not deterministic"
    log(f"harmonizer self-test PASS (synthetic): held-out between-site variance share {b:.3f} -> {a:.3f}; fit ignores y; test rows cannot influence the fit or each other; unseen/tiny sites unchanged; deterministic")


# ---------------------------------------------------------------- data
def load():
    z = verify_inputs(); y, age, sex = z["y"], z["age"], z["sex"]
    site_s = z["site"].astype(str); codes = {s: i for i, s in enumerate(sorted(set(site_s)))}; site = np.array([codes[s] for s in site_s])
    Xa = np.hstack([z["X"], age[:, None], sex[:, None]]); Xs = np.hstack([Xa, site[:, None].astype(float)])
    assert Xa.shape == (676, 19902) and Xs.shape == (676, 19903)
    return Xa, Xs, y, site_s, site, z["sid"].astype(str)


def fit_predict(harm, X, y, tr, te, inner_seed):
    gs = GridSearchCV(pipe(harm), {"clf__C": C_GRID}, cv=StratifiedKFold(N_INNER, shuffle=True, random_state=inner_seed), scoring="roc_auc", n_jobs=4, refit=True).fit(X[tr], y[tr])
    return gs.best_estimator_.predict_proba(X[te])[:, 1], float(gs.best_params_["clf__C"]), float(gs.best_score_)


def run_cv(arm, seed, X, y, tag):
    ck = HERE / "checkpoints" / f"{tag}_seed{seed}.npz"
    if ck.exists():
        z = np.load(ck); return {k: z[k] for k in z.files}
    n = len(y); score, fold = np.zeros(n), np.zeros(n, int); bc, ia, fs = np.zeros(N_OUTER), np.zeros(N_OUTER), np.zeros(N_OUTER); t0 = time.time()
    for f, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=seed).split(X, y)):
        t1 = time.time(); s, c, a = fit_predict(arm == "H", X, y, tr, te, seed); score[te], fold[te] = s, f; bc[f], ia[f], fs[f] = c, a, time.time() - t1
        log(f"  {tag} seed {seed} fold {f}: C={c} inner AUC {a:.4f} ({fs[f]:.1f}s; seed total {time.time() - t0:.0f}s)")
    out = dict(score=score, fold=fold, bestC=bc, iauc=ia, secs=fs); tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, **out); os.replace(tmp, ck); return out


def run_loso(arm, X, y, site_s, Xs_for_check=None):
    ck = HERE / "checkpoints" / f"loso_{arm}.npz"
    if ck.exists():
        z = np.load(ck, allow_pickle=True); return {k: z[k] for k in z.files}
    n = len(y); score = np.zeros(n); rows = []; t0 = time.time()
    for s in sorted(set(site_s), key=lambda q: -(site_s == q).sum()):
        te = np.where(site_s == s)[0]; tr = np.where(site_s != s)[0]; t1 = time.time()
        assert not (set(site_s[tr]) & set(site_s[te])) and len(set(y[tr])) == 2
        sc, c, a = fit_predict(arm == "H", X, y, tr, te, LOSO_INNER_SEED); score[te] = sc
        unchanged = np.nan
        if arm == "H":                                                           # the held-out (unseen) site must pass through the harmonizer unchanged
            h = CC200Harmonizer().fit(X[tr]); unchanged = float(np.abs(h.transform(X[te])[:, :-2] - X[te][:, :-3]).max())
            assert unchanged == 0.0, f"unseen site {s} was modified ({unchanged})"
        rows.append(dict(site=s, n=len(te), bestC=c, iauc=a, secs=time.time() - t1, unseen_max_abs_change=unchanged))
        log(f"  loso {arm} {s:9s} n={len(te):3d}: C={c} inner AUC {a:.4f} ({time.time() - t1:.1f}s; total {time.time() - t0:.0f}s)")
    out = dict(score=score, meta=pd.DataFrame(rows).to_json()); tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, **out); os.replace(tmp, ck); return out


# ---------------------------------------------------------------- baseline (saved) results and comparison
def baseline_cv(sid, seeds):
    o = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str}); o = o[o.arm == "B"]; out = {}
    for s in seeds:
        d = o[o.seed == s].set_index("subject_id").loc[sid]; out[s] = dict(score=d.score.values, fold=d.fold.values.astype(int), pred=d.pred_thr0_5.values)
        assert ((d.score.values >= 0.5).astype(int) == d.pred_thr0_5.values).all()
    return out


def per_fold(y, pred, score, fold):
    return pd.DataFrame([dict(fold=f, n=int((fold == f).sum()), accuracy=accuracy_score(y[fold == f], pred[fold == f]), balanced_accuracy=balanced_accuracy_score(y[fold == f], pred[fold == f]),
                              sensitivity=recall_score(y[fold == f], pred[fold == f]), specificity=((pred[fold == f] == 0) & (y[fold == f] == 0)).sum() / (y[fold == f] == 0).sum(),
                              roc_auc=roc_auc_score(y[fold == f], score[fold == f])) for f in range(N_OUTER)])


def compare_cv(y, sid, H, seeds, base):
    seed_rows, fold_rows = [], []
    for s in seeds:
        bs, bp, bf = base[s]["score"], base[s]["pred"], base[s]["fold"]; hs = H[s]["score"]; hp = (hs >= 0.5).astype(int); hf = H[s]["fold"]
        assert (bf == hf).all(), "outer folds differ from the baseline"
        mb, mh = metrics(y, bp, bs), metrics(y, hp, hs)
        seed_rows.append({"seed": s, **{f"base_{k}": v for k, v in mb.items()}, **{f"H_{k}": v for k, v in mh.items()}, **{f"d_{k}": mh[k] - mb[k] for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")}})
        fb, fh = per_fold(y, bp, bs, bf), per_fold(y, hp, hs, hf)
        for f in range(N_OUTER):
            fold_rows.append(dict(seed=s, fold=f, n_test=int(fb.n[f]), base_accuracy=fb.accuracy[f], H_accuracy=fh.accuracy[f], d_accuracy=fh.accuracy[f] - fb.accuracy[f], base_bacc=fb.balanced_accuracy[f],
                                  H_bacc=fh.balanced_accuracy[f], d_bacc=fh.balanced_accuracy[f] - fb.balanced_accuracy[f], base_auc=fb.roc_auc[f], H_auc=fh.roc_auc[f], H_selected_C=float(H[s]["bestC"][f])))
    return pd.DataFrame(seed_rows), pd.DataFrame(fold_rows)


def site_table(y, site_s, score_b, score_h):
    rows = []
    for s in sorted(set(site_s), key=lambda q: -(site_s == q).sum()):
        m = site_s == s; r = dict(site=s, n=int(m.sum()), n_asd=int(y[m].sum()), n_tc=int((1 - y[m]).sum()))
        for tag, sc in (("base", score_b), ("H", score_h)):
            p = (sc[m] >= 0.5).astype(int); both = len(set(y[m])) == 2
            r.update({f"{tag}_accuracy": accuracy_score(y[m], p), f"{tag}_balanced_accuracy": balanced_accuracy_score(y[m], p) if both else np.nan, f"{tag}_sensitivity": recall_score(y[m], p) if y[m].sum() else np.nan,
                      f"{tag}_specificity": ((p == 0) & (y[m] == 0)).sum() / max((y[m] == 0).sum(), 1), f"{tag}_roc_auc": roc_auc_score(y[m], sc[m]) if both else np.nan})
        r["d_accuracy"] = r["H_accuracy"] - r["base_accuracy"]; rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- frozen acceptance rule (improvement_plan_v1.md, section 4)
def verdict_cv(SD, FD):
    d_acc, d_bacc = SD.d_accuracy.values * 100, SD.d_balanced_accuracy.values * 100; wins = int((FD.d_accuracy > 0).sum()); ties = int((FD.d_accuracy == 0).sum())
    c = {"1a mean paired gain >= +1.0 pt accuracy": d_acc.mean() >= 1.0, "1b mean paired gain >= +1.0 pt balanced accuracy": d_bacc.mean() >= 1.0,
         "1c every seed >= +0.5 pt (accuracy AND balanced accuracy; strictest reading)": bool((d_acc >= 0.5).all() and (d_bacc >= 0.5).all()),
         "2 accuracy higher in >= 20 of 30 paired folds": wins >= 20, "3a mean AUC not lower by more than 0.005": SD.d_roc_auc.mean() >= -0.005,
         "3b mean sensitivity not lower by more than 3 pts": SD.d_sensitivity.mean() * 100 >= -3, "3c mean specificity not lower by more than 3 pts": SD.d_specificity.mean() * 100 >= -3}
    return c, dict(mean_d_acc=d_acc.mean(), mean_d_bacc=d_bacc.mean(), d_acc_by_seed=d_acc.tolist(), d_bacc_by_seed=d_bacc.tolist(), wins=wins, ties=ties, losses=int((FD.d_accuracy < 0).sum()),
                   sign_test_p_ge_wins=float(stats.binom.sf(wins - 1, 30, .5)), fold_diff_mean=float(FD.d_accuracy.mean()), fold_diff_sd=float(FD.d_accuracy.std(ddof=1)), fold_diff_se=float(FD.d_accuracy.std(ddof=1) / np.sqrt(len(FD))))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); a = ap.parse_args(); warnings.filterwarnings("ignore"); t0 = time.time()
    log("=" * 60 + f"\nX1 start, mode={a.mode}"); self_test(); Xa, Xs, y, site_s, site, sid = load()
    if a.mode == "smoke":
        tr, te = next(iter(StratifiedKFold(N_OUTER, shuffle=True, random_state=0).split(Xs, y)))
        for arm, X in (("base", Xa), ("H", Xs)):
            t1 = time.time(); s, c, ia = fit_predict(arm == "H", X, y, tr, te, 0); log(f"SMOKE {arm}: one outer fold (seed 0 fold 0) C={c} inner AUC {ia:.4f} acc@0.5 {accuracy_score(y[te], (s >= .5).astype(int)):.4f} in {time.time() - t1:.1f}s")
        te2 = np.where(site_s == "USM")[0]; tr2 = np.where(site_s != "USM")[0]; t1 = time.time(); fit_predict(True, Xs, y, tr2, te2, 0); log(f"SMOKE H: one leave-one-site-out fit (USM) in {time.time() - t1:.1f}s"); return
    if a.mode in ("cv", "all"):
        for s in SEEDS: run_cv("H", s, Xs, y, "H")
    if a.mode in ("loso", "all"):
        run_loso("base", Xa, y, site_s); run_loso("H", Xs, y, site_s)
    if a.mode in ("report", "all"): report(y, site_s, sid, Xs, time.time() - t0)
    if a.mode == "replicate":
        for s in REPL_SEEDS: run_cv("base", s, Xa, y, "rbase"); run_cv("H", s, Xs, y, "rH")
        replicate_report(y, sid)
    log(f"X1 mode {a.mode} finished in {time.time() - t0:.0f}s")


def report(y, site_s, sid, Xs, secs):
    base = baseline_cv(sid, SEEDS); H = {s: run_cv("H", s, Xs, y, "H") for s in SEEDS}
    SD, FD = compare_cv(y, sid, H, SEEDS, base); SD.to_csv(HERE / "seed_comparison.csv", index=False); FD.to_csv(HERE / "fold_comparison.csv", index=False)
    cks, vd = verdict_cv(SD, FD)
    rows = [dict(experiment="X1", arm="baseline_B", design="10-fold x seed", seed=int(r.seed), **{k: getattr(r, f"base_{k}") for k in ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1", "TP", "FN", "FP", "TN")}) for r in SD.itertuples()]
    rows += [dict(experiment="X1", arm="H_harmonized", design="10-fold x seed", seed=int(r.seed), **{k: getattr(r, f"H_{k}") for k in ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1", "TP", "FN", "FP", "TN")}) for r in SD.itertuples()]
    pd.DataFrame([{"seed": s, "fold": int(H[s]["fold"][i]), "subject_id": sid[i], "site": site_s[i], "y_true": int(y[i]), "score": float(H[s]["score"][i]), "pred_thr0_5": int(H[s]["score"][i] >= .5)} for s in SEEDS for i in range(len(y))]).to_csv(HERE / "oof_predictions.csv", index=False)
    # leave-one-site-out
    lb, lh = run_loso("base", None, y, site_s), run_loso("H", None, y, site_s); sb, sh_ = lb["score"], lh["score"]
    ST = site_table(y, site_s, sb, sh_); ST.to_csv(HERE / "site_metrics.csv", index=False)
    pooled = {t: metrics(y, (s >= .5).astype(int), s) for t, s in (("base", sb), ("H", sh_))}
    for t, nm in (("base", "baseline_B"), ("H", "H_harmonized")): rows.append(dict(experiment="X1", arm=nm, design="leave-one-site-out (pooled; inner seed 0)", seed=np.nan, **{k: pooled[t][k] for k in ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1", "TP", "FN", "FP", "TN")}))
    pd.DataFrame(rows).to_csv(HERE / "experiment_results.csv", index=False)
    pd.DataFrame([{"arm": t, "subject_id": sid[i], "site": site_s[i], "y_true": int(y[i]), "score": float(s[i]), "pred_thr0_5": int(s[i] >= .5)} for t, s in (("base", sb), ("H", sh_)) for i in range(len(y))]).to_csv(HERE / "leave_site_out_predictions.csv", index=False)
    loso_ok = (pooled["H"]["accuracy"] - pooled["base"]["accuracy"]) * 100 >= -0.5; cks["LOSO co-requirement: pooled accuracy not worse by more than 0.5 pt"] = bool(loso_ok)
    meta = pd.read_json(StringIO(str(lh["meta"]))); metab = pd.read_json(StringIO(str(lb["meta"])))
    # diagnostic: between-site variance share before/after (seed 0 outer-training sets)
    diag = []
    for f, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=0).split(Xs, y)):
        h = CC200Harmonizer().fit(Xs[tr]); diag.append(dict(fold=f, eta2_before=eta2(Xs[tr][:, :-3], site_s[tr]), eta2_after=eta2(h.transform(Xs[tr])[:, :-2], site_s[tr])))
    pd.DataFrame(diag).to_csv(HERE / "harmonization_diagnostic.csv", index=False); DG = pd.DataFrame(diag)
    prov = all(v for k, v in cks.items() if not k.startswith("LOSO")) and loso_ok
    json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "design": "see improvement_plan_v1.md section 5 (X1) and section 4 (frozen rule)", "reference": "baseline Arm B, fixed 0.5, saved predictions", "seeds": list(SEEDS), "loso_inner_seed": LOSO_INNER_SEED,
               "grid_C": C_GRID, "hashes_verified": HASHES, "harmonizer": {"type": "ComBat-style parametric EB, label-free", "covariates": ["age (z)", "female"], "min_batch": 5, "unseen_sites": "unchanged"},
               "criteria": {k: bool(v) for k, v in cks.items()}, "provisional_pass_before_replication": bool(prov), "details": vd, "loso_pooled": {t: {k: float(v) for k, v in m.items()} for t, m in pooled.items()},
               "runtime_s": round(secs, 1), "unseen_site_max_abs_change_all_sites": float(meta.unseen_max_abs_change.max())}, open(HERE / "config.json", "w"), indent=1)
    pc = lambda v: f"{v * 100:.2f}%"
    seedtab = "\n".join(f"| {int(r.seed)} | {pc(r.base_accuracy)} -> {pc(r.H_accuracy)} ({r.d_accuracy * 100:+.2f}) | {pc(r.base_balanced_accuracy)} -> {pc(r.H_balanced_accuracy)} ({r.d_balanced_accuracy * 100:+.2f}) | {r.base_roc_auc:.4f} -> {r.H_roc_auc:.4f} | {pc(r.base_sensitivity)} -> {pc(r.H_sensitivity)} | {pc(r.base_specificity)} -> {pc(r.H_specificity)} | {pc(r.base_precision)} -> {pc(r.H_precision)} | {pc(r.base_f1)} -> {pc(r.H_f1)} |" for r in SD.itertuples())
    cmtab = "\n".join(f"| {a} | seed {int(r.seed)} | {int(getattr(r, a2 + '_TP'))} | {int(getattr(r, a2 + '_FN'))} | {int(getattr(r, a2 + '_FP'))} | {int(getattr(r, a2 + '_TN'))} |" for r in SD.itertuples() for a, a2 in (("baseline", "base"), ("harmonized", "H")))
    foldtab = "\n".join(f"| {int(r.seed)} | {int(r.fold)} | {pc(r.base_accuracy)} | {pc(r.H_accuracy)} | {r.d_accuracy * 100:+.2f} | {r.H_selected_C} |" for r in FD.itertuples())
    sitetab = "\n".join(f"| {r.site} | {r.n} | {r.n_asd}/{r.n_tc} | {pc(r.base_accuracy)} | {pc(r.H_accuracy)} | {r.d_accuracy * 100:+.1f} | {r.base_roc_auc:.3f} | {r.H_roc_auc:.3f} | {pc(r.base_sensitivity)} / {pc(r.base_specificity)} | {pc(r.H_sensitivity)} / {pc(r.H_specificity)} |" for r in ST.itertuples())
    cktab = "\n".join(f"| {'PASS' if v else 'FAIL'} | {k} |" for k, v in cks.items())
    txt = f"""# X1 report: site robustness (baseline Arm B vs in-fold label-free harmonization H)
Runtime of this report step included in total {secs / 60:.1f} min. All numbers are development estimates on the 676 authorized training subjects (CC200 only); holdout and combined arrays not touched. Rule frozen in `improvement_plan_v1.md`.

## 1. 10-fold x 3-seed CV, fixed threshold 0.5 (same outer folds; baseline = saved Arm B predictions)
| Seed | Accuracy base -> H (diff pts) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|
{seedtab}
Mean over seeds: accuracy {pc(SD.base_accuracy.mean())} -> {pc(SD.H_accuracy.mean())} ({vd['mean_d_acc']:+.2f} pts); balanced accuracy {pc(SD.base_balanced_accuracy.mean())} -> {pc(SD.H_balanced_accuracy.mean())} ({vd['mean_d_bacc']:+.2f} pts); AUC {SD.base_roc_auc.mean():.4f} -> {SD.H_roc_auc.mean():.4f}.
Paired outer folds (30): H higher in {vd['wins']}, equal in {vd['ties']}, lower in {vd['losses']}; paired difference mean {vd['fold_diff_mean'] * 100:+.2f} pts, SD {vd['fold_diff_sd'] * 100:.2f}, SE {vd['fold_diff_se'] * 100:.2f}; one-sided sign-test p for >= {vd['wins']} wins: {vd['sign_test_p_ge_wins']:.3f}.
Confusion matrices (pooled out-of-fold predictions, 313 ASD / 363 TC):
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
{cmtab}
Fold-level comparison:
| Seed | Fold | Baseline acc. | H acc. | Diff (pts) | Selected C (H) |
|---|---|---|---|---|---|
{foldtab}

## 2. Leave-one-site-out (19 sites; fixed 0.5; inner CV seed 0)
Pooled: baseline accuracy {pc(pooled['base']['accuracy'])}, balanced {pc(pooled['base']['balanced_accuracy'])}, sens {pc(pooled['base']['sensitivity'])}, spec {pc(pooled['base']['specificity'])}, F1 {pc(pooled['base']['f1'])}, pooled AUC {pooled['base']['roc_auc']:.4f} (mixes 19 site-specific models);
H accuracy {pc(pooled['H']['accuracy'])}, balanced {pc(pooled['H']['balanced_accuracy'])}, sens {pc(pooled['H']['sensitivity'])}, spec {pc(pooled['H']['specificity'])}, F1 {pc(pooled['H']['f1'])}, pooled AUC {pooled['H']['roc_auc']:.4f}.
Pooled confusion (TP/FN/FP/TN): baseline {pooled['base']['TP']}/{pooled['base']['FN']}/{pooled['base']['FP']}/{pooled['base']['TN']}; H {pooled['H']['TP']}/{pooled['H']['FN']}/{pooled['H']['FP']}/{pooled['H']['TN']}.
Mean per-site AUC: baseline {ST.base_roc_auc.mean():.3f}, H {ST.H_roc_auc.mean():.3f}. Unseen-site check: maximum absolute change of the held-out site's features by the harmonizer = {meta.unseen_max_abs_change.max():.1f} in all 19 sites (unseen sites pass through unchanged).
| Site | n | ASD/TC | Base acc. | H acc. | Diff (pts) | Base AUC | H AUC | Base sens/spec | H sens/spec |
|---|---|---|---|---|---|---|---|---|---|
{sitetab}
Per-site values for sites with fewer than about 30 subjects are very noisy.

## 3. Harmonization diagnostic (seed-0 outer-training sets)
Mean per-feature between-site variance share: {DG.eta2_before.mean():.4f} before -> {DG.eta2_after.mean():.4f} after.

## 4. Frozen acceptance rule applied
| Result | Criterion |
|---|---|
{cktab}
Provisional outcome before replication (criteria 1-4 and the LOSO co-requirement): **{'ALL MET -> replication on seeds 3-5 is required before calling it validated' if prov else 'NOT MET -> candidate not retained; rule not changed'}**.
"""
    (HERE / "report.md").write_text(txt, encoding="utf-8"); print(txt)


def replicate_report(y, sid):
    rb = {s: run_cv("base", s, None, y, "rbase") for s in REPL_SEEDS}; rh = {s: run_cv("H", s, None, y, "rH") for s in REPL_SEEDS}
    rows = []
    for s in REPL_SEEDS:
        mb = metrics(y, (rb[s]["score"] >= .5).astype(int), rb[s]["score"]); mh = metrics(y, (rh[s]["score"] >= .5).astype(int), rh[s]["score"]); rows.append(dict(seed=s, base_acc=mb["accuracy"], H_acc=mh["accuracy"], d_acc=mh["accuracy"] - mb["accuracy"]))
    R = pd.DataFrame(rows); R.to_csv(HERE / "replication_seeds3_5.csv", index=False); ok = R.d_acc.mean() * 100 >= 0.5 and int((R.d_acc > 0).sum()) >= 2
    log(f"REPLICATION seeds 3-5: mean gain {R.d_acc.mean() * 100:+.2f} pts, positive in {int((R.d_acc > 0).sum())}/3 -> {'REPLICATED' if ok else 'NOT replicated'}")


if __name__ == "__main__":
    main()
