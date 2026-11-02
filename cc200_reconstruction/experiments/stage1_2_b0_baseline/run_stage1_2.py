"""CC200 reconstruction, Stages 1-2 (see protocol.md).  Usage:  python run_stage1_2.py --stage 1   |   python run_stage1_2.py --stage 2
Stage 1: integrity check of the 676 authorized training subjects (per-subject raw CC200 files only). Stage 2: baseline B0 (arms A and B) with nested CV.
Never opened: ts_cc200.npy, covs_cc200.npy, the holdout list, RESTRICTED_id_membership.csv, holdout_guard.py, any holdout label/feature/prediction. No AAL-116."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\abide"); sys.path.insert(1, str(HERE))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from tangent import TangentEmbedding, covariances      # historical feature code, imported read-only
from b0_components import DemoFCPrep

ROOT = Path(r"C:\abide"); RAW = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
TRAIN_LIST = ROOT / "splits" / "phase2_train_subjects.txt"; PHENO = ROOT / "abide_pcp" / "phenotypic_qc.csv"
for s in ("checkpoints", "logs"):
    (HERE / s).mkdir(exist_ok=True)
LOG = HERE / "logs" / "stage_run.log"
SEEDS, C_GRID, N_OUTER, N_INNER = (0, 1, 2), [0.001, 0.01, 0.1, 1.0, 10.0], 10, 5
NROI, MIN_T = 200, 100
EXPECTED = {"n": 676, "asd": 313, "tc": 363}          # from the earlier independent audit of the training list


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def norm_id(s):
    return str(int(float(str(s).strip())))


# ======================================================================== Stage 1
def stage1():
    t0 = time.time(); checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": str(detail)}); log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}"); return ok

    log("=" * 60 + "\nSTAGE 1 start")
    ids = [norm_id(l) for l in open(TRAIN_LIST) if l.strip()]; train = set(ids)
    check("training list has 676 unique IDs", len(ids) == EXPECTED["n"] and len(train) == len(ids), f"{len(ids)} lines, {len(train)} unique")
    files = {}
    for p in RAW.glob("*_rois_cc200.1D"):                       # names only; non-training names are dropped immediately
        m = re.search(r"_(\d{7})_rois_cc200\.1D$", p.name)
        if m and norm_id(m.group(1)) in train:
            files.setdefault(norm_id(m.group(1)), []).append(p)
    check("every training ID has a raw CC200 file", train <= set(files), f"missing {len(train - set(files))}")
    check("no training ID has more than one raw file", all(len(v) == 1 for v in files.values()), f"{sum(len(v) > 1 for v in files.values())} duplicated")
    ph = pd.read_csv(PHENO); ph["sid"] = ph.SUB_ID.astype(int).astype(str)
    ph = ph[ph.sid.isin(train)].reset_index(drop=True)          # all other rows discarded unread
    check("one unique phenotype row per training ID", len(ph) == 676 and ph.sid.is_unique and set(ph.sid) == train, f"{len(ph)} rows, unique={ph.sid.is_unique}")
    fid_ok = all(files[r.sid][0].name == f"{r.FILE_ID}_rois_cc200.1D" for r in ph.itertuples())
    check("phenotype FILE_ID equals the raw file name for every subject", fid_ok, "676/676" if fid_ok else "mismatch")
    nz = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
    site_ok = all(nz(str(r.FILE_ID).rsplit("_", 1)[0]).startswith(nz(r.SITE_ID)) for r in ph.itertuples())
    check("site in FILE_ID prefix agrees with SITE_ID", site_ok, f"{ph.SITE_ID.nunique()} sites")
    check("labels are DX_GROUP in {1,2}", set(ph.DX_GROUP.unique()) <= {1, 2}, sorted(ph.DX_GROUP.unique().tolist()))
    y = (ph.DX_GROUP.values == 1).astype(int)
    check("class counts match the earlier independent audit (313 ASD / 363 TC)", int(y.sum()) == EXPECTED["asd"] and int((1 - y).sum()) == EXPECTED["tc"], f"{int(y.sum())} / {int((1 - y).sum())}")
    age = pd.to_numeric(ph.AGE_AT_SCAN, errors="coerce").replace(-9999, np.nan).values; sex = pd.to_numeric(ph.SEX, errors="coerce").replace(-9999, np.nan).values
    check("age and sex complete and valid", np.isfinite(age).all() and set(np.unique(sex)) <= {1.0, 2.0}, f"age {age.min():.1f}-{age.max():.1f}, sex values {sorted(np.unique(sex))}, missing 0" if np.isfinite(age).all() else "missing values")
    # ---- time series
    ts, rows = [], []
    for r in ph.itertuples():
        a = np.loadtxt(files[r.sid][0], dtype=float)
        ts.append(a); rows.append({"sub_id": r.sid, "file_id": r.FILE_ID, "site": r.SITE_ID, "y": int(r.DX_GROUP == 1), "age": float(r.AGE_AT_SCAN), "sex": float(r.SEX), "n_timepoints": a.shape[0],
                                    "n_rois": a.shape[1] if a.ndim == 2 else -1, "finite": bool(np.isfinite(a).all()), "n_constant_rois": int((np.ptp(a, axis=0) == 0).sum()) if a.ndim == 2 else -1,
                                    "ts_sha1": hashlib.sha1(a.tobytes()).hexdigest()})
    man = pd.DataFrame(rows)
    check("every series is (T x 200)", bool((man.n_rois == NROI).all() and all(a.ndim == 2 for a in ts)), f"ROI counts {sorted(man.n_rois.unique())}")
    check("every series has >= 100 timepoints", bool((man.n_timepoints >= MIN_T).all()), f"T range {man.n_timepoints.min()}-{man.n_timepoints.max()}")
    check("all time-series values are finite", bool(man.finite.all()), "no NaN/Inf")
    check("no two subjects share an identical series", man.ts_sha1.is_unique, f"{man.ts_sha1.nunique()} distinct")
    cs = man[man.n_constant_rois > 0]
    log(f"  [INFO] subjects with at least one constant ROI: {len(cs)} (max {int(man.n_constant_rois.max())} per subject); handled by the 1e-8 term, giving ~zero correlations, not NaN")
    # ---- features through the historical code
    covs = covariances(ts)
    check("covariance array is (676, 200, 200)", covs.shape == (676, NROI, NROI), covs.shape)
    check("covariances symmetric (max asymmetry < 1e-10)", float(np.abs(covs - covs.transpose(0, 2, 1)).max()) < 1e-10, f"{float(np.abs(covs - covs.transpose(0, 2, 1)).max()):.2e}")
    ev = np.linalg.eigvalsh(covs).min()
    check("covariances positive definite", ev > 0, f"min eigenvalue {ev:.3e}")
    tang = TangentEmbedding("correlation", precomputed=True).fit(covs); X = tang.transform(covs)
    d = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1)); corr = covs / (d[:, :, None] * d[:, None, :] + 1e-12)
    check("correlations in [-1, 1] with unit diagonal", float(np.abs(corr).max()) <= 1 + 1e-6 and float(np.abs(np.diagonal(corr, axis1=-2, axis2=-1) - 1).max()) < 1e-6, f"max|r| {float(np.abs(corr).max()):.6f}")
    check("feature matrix is (676, 19900) and finite", X.shape == (676, 19900) and bool(np.isfinite(X).all()), X.shape)
    iu = np.triu_indices(NROI, 1)
    worst = 0.0
    for i in range(0, 676, 34):                                  # independent recomputation on 20 subjects
        x = (ts[i] - ts[i].mean(0)) / (ts[i].std(0) + 1e-8)
        c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_; c = (c + c.T) / 2
        dd = np.sqrt(np.diag(c)); r = c / (dd[:, None] * dd[None, :] + 1e-12)
        f = np.arctanh(np.clip(r, -0.999999, 0.999999))[iu] * np.sqrt(2.0)
        worst = max(worst, float(np.abs(f - X[i]).max()))
    check("independent recomputation (own Ledoit-Wolf/arctanh/triu code) equals tangent.py on 20 subjects", worst < 1e-9, f"max abs diff {worst:.2e}")
    X2 = TangentEmbedding("correlation", precomputed=True).fit_transform(covariances(ts[:5]))
    check("feature generation is deterministic (recomputed for 5 subjects)", np.array_equal(X2, X[:5]), "identical")
    man.to_csv(HERE / "subject_manifest.csv", index=False)
    np.savez(HERE / "features_cc200_train676.npz", X=X, y=y, age=age, sex=sex, sid=ph.sid.values.astype(str), site=ph.SITE_ID.astype(str).values)
    ok = all(c["result"] == "PASS" for c in checks)
    json.dump({"all_pass": ok, "n_checks": len(checks), "n_fail": sum(c["result"] == "FAIL" for c in checks), "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks}, open(HERE / "stage1_checks.json", "w"), indent=1)
    md = ["# Stage 1 audit log: CC200 training-data integrity (676 authorized Phase-2 training subjects)\n",
          f"Run {datetime.now(timezone.utc).isoformat(timespec='seconds')} | runtime {time.time() - t0:.0f}s | **{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}** ({sum(c['result'] == 'PASS' for c in checks)}/{len(checks)})\n",
          "Scope: per-subject raw CC200 files of the 676 training-list IDs only; `ts_cc200.npy`, `covs_cc200.npy`, the holdout list, `RESTRICTED_id_membership.csv`, `holdout_guard.py` and every holdout label/feature/prediction were not opened.\n",
          "| Result | Check | Detail |\n|---|---|---|"] + [f"| {c['result']} | {c['check']} | {c['detail']} |" for c in checks]
    md += ["", f"Information: {len(cs)} of 676 subjects have at least one constant ROI (max {int(man.n_constant_rois.max())}); T range {man.n_timepoints.min()}-{man.n_timepoints.max()}; 19 sites; ASD {int(y.sum())} / TC {int((1 - y).sum())}.",
           "Not done on purpose: comparison of raw-file features with the rows of `covs_cc200.npy` (it needs the combined array that contains restricted holdout rows). The end-to-end check against the recorded phase 2h result replaces it.",
           "Output: `subject_manifest.csv` (IDs of the training list only), `features_cc200_train676.npz` (cache of the 676 x 19,900 features)."]
    (HERE / "stage1_audit_log.md").write_text("\n".join(md), encoding="utf-8")
    log(f"STAGE 1 {'PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f}s"); return ok


# ======================================================================== Stage 2
def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s); return float(thr[np.argmax(tpr - fpr)])


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel(); tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    m = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=recall_score(y, pred), specificity=tn / (tn + fp), precision=precision_score(y, pred, zero_division=0),
             f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, score), TP=tp, FN=fn, FP=fp, TN=tn)
    assert abs(m["accuracy"] - (tp + tn) / len(y)) < 1e-12 and abs(m["f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12 and tp + fn == int(y.sum())
    return m


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=5000, solver="lbfgs", class_weight="balanced")


def run_seed(arm, seed, Xa, y, t0):
    ck = HERE / "checkpoints" / f"arm{arm}_seed{seed}.npz"
    if ck.exists():
        z = np.load(ck); return {k: z[k] for k in z.files}
    n = len(y); score = np.zeros(n); fold = np.zeros(n, int); bestC = np.zeros(N_OUTER); ntau = np.zeros(N_OUTER); iauc = np.zeros(N_OUTER); fsec = np.zeros(N_OUTER)
    for f, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=seed).split(Xa, y)):
        t1 = time.time(); inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed); nest = StratifiedKFold(N_INNER, shuffle=True, random_state=seed + 1000)
        if arm == "A":                                  # preprocessing fitted on the outer-training rows, then inner CV on the scaled matrix (as in phase 2c)
            prep = DemoFCPrep().fit(Xa[tr]); Xtr, Xte = prep.transform(Xa[tr]), prep.transform(Xa[te])
            gs = GridSearchCV(lr(), {"C": C_GRID}, cv=inner, scoring="roc_auc", n_jobs=4, refit=True).fit(Xtr, y[tr])
            s = gs.best_estimator_.predict_proba(Xte)[:, 1]
            io = cross_val_predict(lr(gs.best_params_["C"]), Xtr, y[tr], cv=nest, method="predict_proba")[:, 1]; bc = gs.best_params_["C"]
        else:                                           # strict: preprocessing refitted inside every inner fold
            pipe = Pipeline([("prep", DemoFCPrep()), ("clf", lr())])
            gs = GridSearchCV(pipe, {"clf__C": C_GRID}, cv=inner, scoring="roc_auc", n_jobs=4, refit=True).fit(Xa[tr], y[tr])
            s = gs.best_estimator_.predict_proba(Xa[te])[:, 1]; bc = gs.best_params_["clf__C"]
            io = cross_val_predict(Pipeline([("prep", DemoFCPrep()), ("clf", lr(bc))]), Xa[tr], y[tr], cv=nest, method="predict_proba")[:, 1]
        score[te], fold[te] = s, f; bestC[f], ntau[f], iauc[f], fsec[f] = bc, youden(y[tr], io), gs.best_score_, time.time() - t1
        log(f"  arm {arm} seed {seed} fold {f}: C={bc} inner AUC {gs.best_score_:.4f} nested tau {ntau[f]:.4f} ({fsec[f]:.1f}s; total {time.time() - t0:.0f}s)")
    out = dict(score=score, fold=fold, bestC=bestC, ntau=ntau, iauc=iauc, fsec=fsec)
    tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, **out); os.replace(tmp, ck)
    return out


def stage2():
    t0 = time.time(); log("=" * 60 + "\nSTAGE 2 start")
    s1 = json.load(open(HERE / "stage1_checks.json")); assert s1["all_pass"], "Stage 1 did not pass; Stage 2 refuses to run"
    z = np.load(HERE / "features_cc200_train676.npz", allow_pickle=True); Xfc, y, age, sex, sid, site = z["X"], z["y"], z["age"], z["sex"], z["sid"], z["site"]
    Xa = np.hstack([Xfc, age[:, None], sex[:, None]]); assert Xa.shape == (676, 19902)
    log(f"features {Xa.shape} (19,900 connectivity + age + sex); ASD {int(y.sum())} / TC {int((1 - y).sum())}")
    fm, sm, oofrows, detail = [], [], [], {}
    for arm in ("A", "B"):
        for seed in SEEDS:
            r = run_seed(arm, seed, Xa, y, t0); detail[(arm, seed)] = r
            score, fold = r["score"], r["fold"]; tau_h = youden(y, score)
            taus = {"thr0.5": np.full(len(y), 0.5), "youden_oof_historical": np.full(len(y), tau_h), "nested_youden": r["ntau"][fold]}
            for rd, thr in taus.items():
                pred = (score >= thr).astype(int); m = metrics(y, pred, score)
                fa = [accuracy_score(y[fold == f], pred[fold == f]) for f in range(N_OUTER)]; fb = [balanced_accuracy_score(y[fold == f], pred[fold == f]) for f in range(N_OUTER)]
                sm.append(dict(arm=arm, seed=seed, reading=rd, threshold=float(thr[0]) if rd != "nested_youden" else float(np.mean(r["ntau"])), **m, fold_accuracy_mean=np.mean(fa), fold_accuracy_sd=np.std(fa),
                               fold_bacc_mean=np.mean(fb), fold_bacc_sd=np.std(fb), fold_auc_mean=np.mean([roc_auc_score(y[fold == f], score[fold == f]) for f in range(N_OUTER)]),
                               chosen_C=str([float(c) for c in r["bestC"]])))
                if rd == "thr0.5": p05 = pred
                elif rd == "youden_oof_historical": ph_ = pred
                else: pn = pred
            for f in range(N_OUTER):
                te = np.where(fold == f)[0]
                fm.append(dict(arm=arm, seed=seed, fold=f, n_test=len(te), test_asd=int(y[te].sum()), test_tc=int((1 - y[te]).sum()), best_C=float(r["bestC"][f]), inner_best_auc=float(r["iauc"][f]),
                               nested_tau=float(r["ntau"][f]), fold_auc=roc_auc_score(y[te], score[te]), acc_at_0_5=accuracy_score(y[te], p05[te]), acc_at_hist_tau=accuracy_score(y[te], ph_[te]),
                               acc_at_nested_tau=accuracy_score(y[te], pn[te]), seconds=float(r["fsec"][f])))
            for i in range(len(y)):
                oofrows.append(dict(arm=arm, seed=seed, fold=int(fold[i]), subject_id=sid[i], site=site[i], y_true=int(y[i]), score=float(score[i]), pred_thr0_5=int(p05[i]), pred_hist_youden=int(ph_[i]), pred_nested_youden=int(pn[i])))
    FM, SM = pd.DataFrame(fm), pd.DataFrame(sm)
    FM.to_csv(HERE / "fold_metrics.csv", index=False); SM.to_csv(HERE / "seed_metrics.csv", index=False); pd.DataFrame(oofrows).to_csv(HERE / "oof_predictions.csv", index=False)
    cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    agg = SM.groupby(["arm", "reading"])[cols].agg(["mean", "std"]); agg.columns = [f"{a}_{b}" for a, b in agg.columns]; agg.reset_index().to_csv(HERE / "summary_metrics.csv", index=False)
    # ---- comparison with phase 2h/2c (recorded)
    rec = pd.read_csv(ROOT / "results" / "phase2c_baseline_results.csv"); recs = rec[rec.seed.isin(["0", "1", "2"])].copy()
    comp = []
    for _, rr in recs.iterrows():
        s = int(rr.seed); mine = SM[(SM.arm == "A") & (SM.seed == s) & (SM.reading == "youden_oof_historical")].iloc[0]
        cm_rec = np.array(json.loads(rr.confusion_matrix)); cm_my = np.array([[mine.TN, mine.FP], [mine.FN, mine.TP]])
        c_rec = json.loads(rr.chosen_C_per_fold); c_my = [float(c) for c in detail[("A", s)]["bestC"]]
        comp.append(dict(seed=s, rec_accuracy=float(rr.oof_accuracy), my_accuracy=mine.accuracy, diff_accuracy=mine.accuracy - float(rr.oof_accuracy), rec_tau=float(rr.tau_youden), my_tau=mine.threshold, rec_auc=float(rr.oof_roc_auc),
                         my_auc=mine.roc_auc, diff_auc=mine.roc_auc - float(rr.oof_roc_auc), rec_f1=float(rr.oof_f1), my_f1=mine.f1, confusion_identical=bool((cm_rec == cm_my).all()), rec_cm=str(cm_rec.tolist()), my_cm=str(cm_my.tolist()),
                         chosen_C_identical_folds=int(sum(a == b for a, b in zip(c_rec, c_my))), chosen_C_folds=len(c_my)))
    CP = pd.DataFrame(comp); CP.to_csv(HERE / "comparison_vs_phase2h.csv", index=False)
    log(f"STAGE 2 done in {time.time() - t0:.0f}s"); make_report(SM, FM, CP, agg.reset_index(), time.time() - t0)


def make_report(SM, FM, CP, agg, secs):
    pc = lambda v: f"{v * 100:.2f}%"
    g = lambda arm, rd, c: agg[(agg.arm == arm) & (agg.reading == rd)][f"{c}_mean"].iloc[0]; sd = lambda arm, rd, c: agg[(agg.arm == arm) & (agg.reading == rd)][f"{c}_std"].iloc[0]
    rows = []
    for arm in ("A", "B"):
        for rd, lab in (("youden_oof_historical", "(ii) Youden on pooled OOF (historical rule)"), ("thr0.5", "(i) fixed 0.5"), ("nested_youden", "(iii) nested training-only Youden")):
            rows.append(f"| {arm} | {lab} | " + " | ".join(f"{pc(g(arm, rd, c))} +/- {sd(arm, rd, c) * 100:.2f}" for c in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1")) + f" | {g(arm, rd, 'roc_auc'):.4f} +/- {sd(arm, rd, 'roc_auc'):.4f} |")
    cm = SM[(SM.arm == "A") & (SM.reading == "youden_oof_historical")]
    cmrows = "\n".join(f"| A | seed {int(r.seed)} | (ii) | {int(r.TP)} | {int(r.FN)} | {int(r.FP)} | {int(r.TN)} | {pc(r.accuracy)} |" for r in cm.itertuples())
    cmB = SM[(SM.arm == "B") & (SM.reading == "thr0.5")]; cmrows += "\n" + "\n".join(f"| B | seed {int(r.seed)} | (i) | {int(r.TP)} | {int(r.FN)} | {int(r.FP)} | {int(r.TN)} | {pc(r.accuracy)} |" for r in cmB.itertuples())
    cp = "\n".join(f"| {int(r.seed)} | {pc(r.rec_accuracy)} | {pc(r.my_accuracy)} | {r.diff_accuracy * 100:+.2f} | {r.rec_auc:.4f} | {r.my_auc:.4f} | {r.chosen_C_identical_folds}/{r.chosen_C_folds} | {'yes' if r.confusion_identical else 'no'} |" for r in CP.itertuples())
    mine_mean = CP.my_accuracy.mean(); rec_mean = CP.rec_accuracy.mean()
    txt = f"""# Stage 2 report: CC200 baseline B0 (development data only)

Data: the 676 authorized training subjects (313 ASD / 363 TC, 19 sites); features 19,902 = 19,900 Ledoit-Wolf correlation/Fisher-Z edges + age + sex. 10-fold outer CV x seeds 0, 1, 2; inner 5-fold ROC-AUC over C in {{0.001, 0.01, 0.1, 1, 10}}.
Runtime {secs / 60:.1f} min. Restricted holdout, combined arrays, leave-one-site-out and variants V1-V6 were not touched or run. All numbers are development cross-validation estimates (mean +/- SD over the 3 seeds), not independent-test performance.

## Metrics (arm A = historical-faithful preprocessing; arm B = strict, preprocessing refitted in every inner fold)
| Arm | Threshold reading | Accuracy | Balanced acc. | Sensitivity | Specificity | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Fold-to-fold SD of accuracy (arm A, reading ii): {SM[(SM.arm == 'A') & (SM.reading == 'youden_oof_historical')].fold_accuracy_sd.mean() * 100:.2f} points (mean over seeds); per-fold values in `fold_metrics.csv`.

## Confusion matrices from the genuine out-of-fold predictions (676 subjects: 313 ASD / 363 TC)
| Arm | Seed | Threshold | TP | FN | FP | TN | Accuracy |
|---|---|---|---|---|---|---|---|
{cmrows}
All readings and arms: `seed_metrics.csv`; predictions: `oof_predictions.csv`.

## Comparison with the recorded phase 2h result (arm A, historical threshold rule)
| Seed | Recorded accuracy | Reconstructed | Diff (points) | Recorded AUC | Reconstructed AUC | Same C in folds | Identical confusion matrix |
|---|---|---|---|---|---|---|---|
{cp}

Mean accuracy: recorded {pc(rec_mean)} (phase 2h: 67.21%), reconstructed {pc(mine_mean)}. Nothing was tuned to move this number.
"""
    (HERE / "final_report.md").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, required=True); a = ap.parse_args()
    warnings.filterwarnings("ignore")
    if a.stage == 1:
        sys.exit(0 if stage1() else 1)
    stage2()
