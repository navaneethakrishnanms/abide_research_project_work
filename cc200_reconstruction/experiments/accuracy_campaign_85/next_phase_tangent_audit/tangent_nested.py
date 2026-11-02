"""Strictly nested tangent-space evaluation (plan: experiment_plan.md; method: methodology.md).  CC200 only; the 676 authorized training subjects only.
mode "nested": the tangent reference (geometric mean) is refitted inside every inner training split and applied, never refitted, to that split's validation rows; the final outer model refits it on the outer-training rows only.
mode "old"   : reference fitted once on the outer-training rows and reused by all inner fits (the pre-correction behaviour) -- used only to prove that this evaluator reproduces the saved P3_TS_LR results.
Modes of this script: tests | repro | run | lr | hashes.   Never opened: holdout list, RESTRICTED_id_membership.csv, ts_cc200.npy, covs_cc200.npy, any holdout label/feature/prediction. No AAL-116."""
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
HERE = Path(__file__).resolve().parent; CAMP = HERE.parent
BASE = Path(r"C:\abide\cc200_reconstruction\experiments\stage1_2_b0_baseline")
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(BASE)); sys.path.insert(2, r"C:\abide")
for s in ("checkpoints", "logs"):
    (HERE / s).mkdir(exist_ok=True)
LOGF = HERE / "logs" / "tangent_nested.log"

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from b0_components import DemoFCPrep
from tangent import TangentEmbedding, spd_inv_sqrt, spd_log

CS = [0.001, 0.01, 0.1, 1.0, 10.0]; N_OUTER = 10; N_INNER = 5; COVS = CAMP / "cache" / "covs_train676.npy"


def log(m):
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def lr(C):
    return LogisticRegression(C=C, max_iter=5000, solver="lbfgs", class_weight="balanced")


def choose_threshold(y, s, default):                       # identical to campaign.choose_threshold
    u = np.unique(s); cand = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]]); P, N = int(y.sum()), int(len(y) - y.sum()); best, bt = -1.0, default
    for t in cand:
        p = s >= t; b = ((p & (y == 1)).sum() / P + ((~p) & (y == 0)).sum() / N) / 2
        if b > best + 1e-12 or (abs(b - best) <= 1e-12 and abs(t - default) < abs(bt - default)): best, bt = b, t
    return float(bt)


def make_features(cov_fit, demo_fit, cov_app, demo_app):
    """Fit the tangent reference on cov_fit ONLY; transform cov_fit and cov_app with that fixed reference (no refit on cov_app)."""
    t = TangentEmbedding("tangent", precomputed=True).fit(cov_fit)
    return np.hstack([t.transform(cov_fit), demo_fit]), np.hstack([t.transform(cov_app), demo_app]), t


def auc_per_C(A, yA, B, yB):
    prep = DemoFCPrep().fit(A); Ap, Bp = prep.transform(A), prep.transform(B)
    return np.array([roc_auc_score(yB, lr(C).fit(Ap, yA).decision_function(Bp)) for C in CS])


def proba(A, yA, B, C):
    prep = DemoFCPrep().fit(A); return lr(C).fit(prep.transform(A), yA).predict_proba(prep.transform(B))[:, 1]


def run_fold_core(ctr, dtr, ytr, cte, dte, seed, mode):
    """One outer fold. Receives NO outer-test labels. Returns outer-test scores, training-only inner OOF scores, threshold, selected C, diagnostics."""
    t0 = time.time(); A_out, B_out, t_out = make_features(ctr, dtr, cte, dte)                   # final (outer-training) reference; applied to outer-test rows
    if mode == "old": split = lambda itr, iva: (A_out[itr], A_out[iva], None)
    else:
        def split(itr, iva):
            a, b, t = make_features(ctr[itr], dtr[itr], ctr[iva], dtr[iva]); return a, b, t
    aucs, diag = [], []
    for itr, iva in StratifiedKFold(N_INNER, shuffle=True, random_state=seed).split(ctr, ytr):  # inner CV for C (seed = outer seed)
        a, b, t = split(itr, iva); aucs.append(auc_per_C(a, ytr[itr], b, ytr[iva]))
        if t is not None:                                                                       # diagnostic: how much the old (outer-fitted) reference differs from the strictly inner one
            Gk, Gf = t.reference_, t_out.reference_; W = spd_inv_sqrt(Gf[None])[0]
            ref_dist = float(np.linalg.norm(spd_log((W @ Gk @ W)[None])[0])); f_old = t_out.transform(ctr[iva]); f_new = b[:, :-2]
            diag.append((ref_dist, float(np.linalg.norm(f_new - f_old) / np.linalg.norm(f_old))))
    mean_auc = np.mean(aucs, axis=0); bestC = CS[int(np.argmax(mean_auc))]
    io = np.zeros(len(ytr))
    for itr, iva in StratifiedKFold(N_INNER, shuffle=True, random_state=seed + 1000).split(ctr, ytr):   # threshold scores (seed + 1000), best C
        a, b, _ = split(itr, iva); io[iva] = proba(a, ytr[itr], b, bestC)
    thr = choose_threshold(ytr, io, 0.5); score = proba(A_out, ytr, B_out, bestC)
    return dict(score=score, io=io, thr=thr, bestC=bestC, inner_auc=float(mean_auc.max()), aucs=mean_auc, ref_dist=float(np.mean([d[0] for d in diag])) if diag else np.nan,
                feat_rel_change=float(np.mean([d[1] for d in diag])) if diag else np.nan, secs=time.time() - t0)


def fold_job(seed, f, tr, te, mode, y, demo, tag):
    ck = HERE / "checkpoints" / f"{tag}_{mode}_seed{seed}_fold{f}.npz"
    if ck.exists(): return str(ck)
    covs = np.load(COVS, mmap_mode="r"); r = run_fold_core(np.asarray(covs[tr]), demo[tr], y[tr], np.asarray(covs[te]), demo[te], seed, mode)
    tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, **r, tr=tr, te=te); os.replace(tmp, ck); return str(ck)


# ------------------------------------------------------------------ data and hashes
def load_data():
    import json as _j
    z = np.load(BASE / "features_cc200_train676.npz", allow_pickle=True); prov = _j.load(open(CAMP / "cache" / "provenance.json"))
    assert sha(COVS) == prov["covs_sha256"], "covariance cache hash mismatch"
    HS = {"features_cc200_train676.npz": "df6e7f9a3d15c7d659c00c2ed0d24d497d6be58d039856b5fb4ce795d39fd43c", "subject_manifest.csv": "22cb64022ab5947716993b82f19be2814d469f3776241b3a36b5c10b3e93ea7a",
          "seed_metrics.csv": "4211d0305ea384cb77075ab2d0a01d1f8b958899c7b5b27b2e36095d2d4c3c46", "fold_metrics.csv": "bab680c18643666d08ced0c2d30557ccba546058fd0d49b456ceb1fd8c872f2e",
          "oof_predictions.csv": "28dc62c095828f68e76d0d2bbd5496c61b472f742e635207bcce72341e73682e", "stage1_checks.json": "c3f835248dd489221de957994d96f716b3abe06379ae655dd0e5e5a49c94cb55"}
    assert all(sha(BASE / k) == v for k, v in HS.items()), "baseline/cache hash mismatch"
    tl = {str(int(float(l))) for l in open(r"C:\abide\splits\phase2_train_subjects.txt") if l.strip()}; assert set(z["sid"].astype(str)) == tl and len(tl) == 676
    log("inputs verified: baseline/cache hashes, covariance cache hash, 676 authorized IDs"); return z["y"], np.column_stack([z["age"], z["sex"]]), z["sid"].astype(str), z["site"].astype(str)


def inventory():
    files = {"covs_cache": COVS, "provenance": CAMP / "cache" / "provenance.json", "campaign.py": CAMP / "campaign.py", "selection_rule.py": CAMP / "selection_rule.py", "final_report.md": CAMP / "final_report.md",
             "audit_report.md": CAMP / "independent_audit" / "audit_report.md", "confirmation_report.md": CAMP / "fresh_seed_confirmation" / "report.md", "source_and_config_hashes.json": CAMP / "source_and_config_hashes.json"}
    inv = {k: sha(v) for k, v in files.items()}
    for d in ("checkpoints", "results"):
        inv.update({f"campaign/{d}/{p.name}": sha(p) for p in sorted((CAMP / d).glob("*")) if p.is_file()})
    inv.update({f"confirmation/{p.name}": sha(p) for p in sorted((CAMP / "fresh_seed_confirmation").rglob("*")) if p.is_file()})
    inv.update({f"baseline/{p.name}": sha(p) for p in sorted(BASE.glob("*")) if p.is_file()}); return inv


# ------------------------------------------------------------------ synthetic validity tests
def synth(n=90, p=10, seed=0):
    rng = np.random.default_rng(seed); y = (rng.random(n) < 0.45).astype(int); covs = np.zeros((n, p, p))
    for i in range(n):
        M = rng.normal(size=(p, p + 5)) * 0.5; M[0, :3] += 0.8 * y[i]; covs[i] = M @ M.T / (p + 5) + 0.3 * np.eye(p)
    demo = np.column_stack([rng.normal(15, 4, n), rng.integers(1, 3, n).astype(float)]); return covs, demo, y


def tests():
    covs, demo, y = synth(); tr = np.arange(72); te = np.arange(72, 90); ok = []
    def chk(name, c, d=""): ok.append(bool(c)); log(f"  [{'PASS' if c else 'FAIL'}] {name} {d}")
    # T1: strict nesting of the inner reference: perturbing inner-VALIDATION covariances must not change the reference fitted on the inner TRAINING split
    itr, iva = np.arange(0, 57), np.arange(57, 72); a1, b1, t1 = make_features(covs[tr][itr], demo[tr][itr], covs[tr][iva], demo[tr][iva])
    pert = covs[tr].copy(); rng = np.random.default_rng(5); X = rng.normal(size=(15, 10, 14)); pert[iva] = X @ np.swapaxes(X, 1, 2) / 14 + 0.3 * np.eye(10)
    a2, b2, t2 = make_features(pert[itr], demo[tr][itr], pert[iva], demo[tr][iva])
    chk("T1 inner reference unchanged when inner-validation covariances change", np.array_equal(t1.reference_, t2.reference_) and np.array_equal(a1, a2) and not np.array_equal(b1, b2))
    # T2: outer-test rows never influence any fit: altering them leaves C, threshold, inner OOF scores and the other test rows' scores unchanged
    r1 = run_fold_core(covs[tr], demo[tr], y[tr], covs[te], demo[te], 3, "nested"); ct = covs[te].copy(); X = rng.normal(size=(6, 10, 14)); ct[:6] = X @ np.swapaxes(X, 1, 2) / 14 + 0.3 * np.eye(10); dt = demo[te].copy(); dt[:6, 0] += 30
    r2 = run_fold_core(covs[tr], demo[tr], y[tr], ct, dt, 3, "nested")
    chk("T2 outer-test rows do not influence C, threshold, inner OOF scores or other test rows", r1["bestC"] == r2["bestC"] and r1["thr"] == r2["thr"] and np.array_equal(r1["io"], r2["io"]) and np.array_equal(r1["score"][6:], r2["score"][6:]) and not np.array_equal(r1["score"][:6], r2["score"][:6]))
    import inspect
    chk("T3 run_fold_core receives no outer-test labels (signature check)", "yte" not in inspect.signature(run_fold_core).parameters and list(inspect.signature(run_fold_core).parameters)[:5] == ["ctr", "dtr", "ytr", "cte", "dte"])
    r3 = run_fold_core(covs[tr], demo[tr], y[tr], covs[te], demo[te], 3, "old"); chk("T4 nested and old modes differ in features (reference distance > 0) but share the same code path otherwise", r1["ref_dist"] > 0 and np.isnan(r3["ref_dist"]), f"mean inner-vs-outer reference distance {r1['ref_dist']:.3f}, relative feature change {r1['feat_rel_change']:.3f}")
    chk("T5 determinism", np.array_equal(r1["score"], run_fold_core(covs[tr], demo[tr], y[tr], covs[te], demo[te], 3, "nested")["score"]))
    log(f"synthetic validity tests: {sum(ok)}/{len(ok)} passed"); return all(ok)


def repro():
    """old mode must reproduce the saved P3_TS_LR results (scores, selected C, threshold, inner OOF) for seed 0, folds 0 and 1."""
    from sklearn.model_selection import StratifiedKFold as SKF
    y, demo, sid, site = load_data(); st = np.load(CAMP / "checkpoints" / "P3_TS_LR_seed0.npz"); folds = list(SKF(N_OUTER, shuffle=True, random_state=0).split(np.zeros(len(y)), y)); res = []
    jobs = [(0, f, folds[f][0], folds[f][1]) for f in (0, 1)]
    outs = Parallel(n_jobs=2)(delayed(fold_job)(s, f, tr, te, "old", y, demo, "repro") for s, f, tr, te in jobs); ok = True
    for (s, f, tr, te), p in zip(jobs, outs):
        r = np.load(p); ds = float(np.abs(r["score"] - st["score"][te]).max()); dt = abs(float(r["thr"]) - float(st["thr"][f])); dio = float(np.abs(r["io"] - st["inner_oof"][f][tr]).max())
        sc = json.loads(str(st["params"][f]))["clf__C"]; good = ds < 1e-9 and dt < 1e-9 and dio < 1e-9 and float(r["bestC"]) == sc; ok &= good
        log(f"  repro seed 0 fold {f}: max|score diff| {ds:.2e}, |thr diff| {dt:.2e}, max|inner-OOF diff| {dio:.2e}, selected C {float(r['bestC'])} vs saved {sc}: {'EXACT' if good else 'MISMATCH'}")
    log(f"OLD-MODE REPRODUCTION OF SAVED P3_TS_LR: {'PASS' if ok else 'FAIL'}"); return ok


def assemble(seed, y, tag, mode):
    n = len(y); score, fold = np.zeros(n), np.zeros(n, int); thr, iauc, secs, bc, rd, fr = (np.zeros(N_OUTER) for _ in range(6)); io = np.full((N_OUTER, n), np.nan); params = []
    for f in range(N_OUTER):
        r = np.load(HERE / "checkpoints" / f"{tag}_{mode}_seed{seed}_fold{f}.npz"); score[r["te"]] = r["score"]; fold[r["te"]] = f; io[f, r["tr"]] = r["io"]; thr[f], iauc[f], secs[f], bc[f], rd[f], fr[f] = (float(r[k]) for k in ("thr", "inner_auc", "secs", "bestC", "ref_dist", "feat_rel_change"))
        params.append(json.dumps({"clf__C": float(r["bestC"])}))
    out = dict(score=score, fold=fold, thr=thr, default=np.full(N_OUTER, 0.5), iauc=iauc, secs=secs, inner_oof=io, params=np.array(params), ref_dist=rd, feat_rel_change=fr)
    np.savez(HERE / "checkpoints" / f"{tag}_{mode}_seed{seed}.npz", **out); return out


def run(seeds, mode, tag, njobs):
    from sklearn.model_selection import StratifiedKFold as SKF
    y, demo, sid, site = load_data(); inv0 = inventory(); json.dump(inv0, open(HERE / f"hashes_before_{tag}.json", "w"), indent=1); t0 = time.time(); jobs = []
    for s in seeds:
        for f, (tr, te) in enumerate(SKF(N_OUTER, shuffle=True, random_state=s).split(np.zeros(len(y)), y)):
            if not (HERE / "checkpoints" / f"{tag}_{mode}_seed{s}_fold{f}.npz").exists(): jobs.append((s, f, tr, te))
    log(f"{tag}/{mode}: {len(jobs)} outer folds to run on {njobs} workers (seeds {list(seeds)})")
    Parallel(n_jobs=njobs, verbose=5)(delayed(fold_job)(s, f, tr, te, mode, y, demo, tag) for s, f, tr, te in jobs)
    for s in seeds: assemble(s, y, tag, mode)
    inv1 = inventory(); same = inv0 == inv1; json.dump(inv1, open(HERE / f"hashes_after_{tag}.json", "w"), indent=1)
    log(f"{tag}/{mode} finished in {time.time() - t0:.0f}s; existing artifacts unchanged by this stage: {same}"); assert same


def lr_reference(seeds):
    """Reference LR for fresh seeds via the campaign engine (read-only import; its logger is redirected); checkpoints go to THIS folder."""
    sys.path.insert(3, str(CAMP)); import campaign as CP; CP.log = log
    Xa, y, site_s, sid, covs, Xp = CP.load(); from sklearn.model_selection import StratifiedKFold as SKF; prov = CP.provider("LW", Xa, Xp, covs)
    for seed in seeds:
        ck = HERE / "checkpoints" / f"LR_seed{seed}.npz"
        if ck.exists(): continue
        n = len(y); score, fold = np.zeros(n), np.zeros(n, int); thr, dfl, ia, secs = (np.zeros(N_OUTER) for _ in range(4)); io_all = np.full((N_OUTER, n), np.nan); params = []; t0 = time.time()
        for f, (tr, te) in enumerate(SKF(N_OUTER, shuffle=True, random_state=seed).split(Xa, y)):
            t1 = time.time(); s, io, th, df_, bp, a = CP.run_fold("LR", prov, y, tr, te, seed, None); score[te], fold[te], thr[f], dfl[f], ia[f], secs[f] = s, f, th, df_, a, time.time() - t1; io_all[f, tr] = io; params.append(bp)
        np.savez(ck, score=score, fold=fold, thr=thr, default=dfl, iauc=ia, secs=secs, inner_oof=io_all, params=np.array(params)); log(f"reference LR seed {seed} done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--tag", default="exploratory"); ap.add_argument("--njobs", type=int, default=15); ap.add_argument("--evalmode", default="nested"); a = ap.parse_args()
    warnings.filterwarnings("ignore"); seeds = tuple(int(s) for s in a.seeds.split(","))
    if a.mode == "tests": sys.exit(0 if tests() else 1)
    if a.mode == "repro": sys.exit(0 if repro() else 1)
    if a.mode == "run": run(seeds, a.evalmode, a.tag, a.njobs)
    if a.mode == "lr": lr_reference(seeds)


if __name__ == "__main__":
    main()
