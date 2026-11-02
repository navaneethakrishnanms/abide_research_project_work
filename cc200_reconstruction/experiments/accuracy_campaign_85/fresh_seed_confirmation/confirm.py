"""Pre-registered fresh-seed confirmation (see preregistration.md): reference P1_LR vs the saved 3-member soft vote, seeds 3, 4, 5.  CC200 only; 676 authorized training subjects only.
Imports the campaign engine read-only (campaign.py is not edited; its log is redirected here); all checkpoints/outputs go to THIS folder. Existing campaign outputs are only read.
Modes: run | report.   Never opened: holdout list, RESTRICTED_id_membership.csv, ts_cc200.npy, covs_cc200.npy, any holdout label/feature/prediction. No AAL-116."""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent; CAMP = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(CAMP))
for s in ("checkpoints", "logs"):
    (HERE / s).mkdir(exist_ok=True)
LOGF = HERE / "logs" / "confirm.log"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import campaign as CP                                                          # engine imported read-only


def log(m):
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


CP.log = log                                                                   # redirect the engine's logging so no campaign log is appended to
SEEDS = (3, 4, 5); OLD = (0, 1, 2); MEMBERS = ["P3_TS_LR", "P2_LSVM", "P1_LR"]; N_OUTER = 10
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
pc = lambda v: f"{v * 100:.2f}%"


def build_e1(ck, y, fold_by_seed, seeds):
    """Soft vote exactly as phase5.py: Platt (C=1e6) on training-only inner OOF scores, mean of calibrated probabilities, threshold by CP.choose_threshold(y[tr], ensemble inner scores, 0.5)."""
    out = {}
    for s in seeds:
        n = len(y); score, thr = np.zeros(n), np.zeros(N_OUTER); fo = fold_by_seed[s]
        for f in range(N_OUTER):
            te = fo == f; tr = ~te
            for m in MEMBERS: assert np.array_equal(~np.isnan(ck[(m, s)]["inner_oof"][f]), tr), f"{m}: inner-OOF rows differ from the outer-training rows"
            Zin = np.column_stack([ck[(m, s)]["inner_oof"][f][tr] for m in MEMBERS]); Zte = np.column_stack([ck[(m, s)]["score"][te] for m in MEMBERS])
            cal = [LogisticRegression(C=1e6).fit(Zin[:, j:j + 1], y[tr]) for j in range(len(MEMBERS))]
            sin = np.mean([c.predict_proba(Zin[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0); score[te] = np.mean([c.predict_proba(Zte[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0); thr[f] = CP.choose_threshold(y[tr], sin, 0.5)
        out[s] = dict(score=score, thr=thr, fold=fo, pred_R1=(score >= thr[fo]).astype(int), pred_R0=(score >= 0.5).astype(int))
    return out


def gate_and_load():
    z = CP.verify_inputs(); data = CP.load(); Xa, y, site_s, sid, covs, Xp = data
    assert json.load(open(CAMP / "results" / "P5_E1_SOFTVOTE_config.json"))["members"] == MEMBERS, "saved ensemble members differ from the pre-registered ones"
    hs = json.load(open(CAMP / "source_and_config_hashes.json")); assert sha(CAMP / "campaign.py") == hs["campaign.py"] and sha(CAMP / "selection_rule.py") == hs["selection_rule.py"], "campaign code or rule changed"
    # reconstruction gate: the builder must reproduce the saved seed 0-2 soft-vote predictions exactly
    ck_old = {(m, s): np.load(CAMP / "checkpoints" / f"{m}_seed{s}.npz") for m in MEMBERS for s in OLD}; fo_old = {s: ck_old[("P1_LR", s)]["fold"] for s in OLD}; e1 = build_e1(ck_old, y, fo_old, OLD)
    saved = pd.read_csv(CAMP / "results" / "P5_E1_SOFTVOTE_oof_predictions_seeds012.csv", dtype={"subject_id": str}); ok = True
    for s in OLD: ok &= bool((saved[saved.seed == s].set_index("subject_id").loc[sid].pred_R1.values == e1[s]["pred_R1"]).all())
    log(f"RECONSTRUCTION GATE: rebuilt soft vote reproduces the saved seed 0-2 predictions: {'EXACT' if ok else 'MISMATCH'}"); assert ok, "ensemble cannot be reconstructed exactly: stop"
    return data


def fold_validity(y):
    old = {s: np.zeros(len(y), int) for s in OLD}; new = {s: np.zeros(len(y), int) for s in SEEDS}
    for d, seeds in ((old, OLD), (new, SEEDS)):
        for s in seeds:
            for f, (_, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=s).split(np.zeros(len(y)), y)): d[s][te] = f
    rows = []
    for s in SEEDS:
        sizes = np.bincount(new[s]); asd = np.array([y[new[s] == f].sum() for f in range(N_OUTER)])
        ok = bool((np.sort(np.unique(new[s])) == np.arange(N_OUTER)).all() and sizes.sum() == len(y) and sizes.min() >= 67 and sizes.max() <= 68 and asd.min() >= 31 and asd.max() <= 32)
        rows.append(dict(seed=s, fold_sizes=f"{sizes.min()}-{sizes.max()}", asd_per_fold=f"{asd.min()}-{asd.max()}", every_subject_in_exactly_one_test_fold=True, valid=ok, differs_from_seeds_0_2=all(not (new[s] == old[o]).all() for o in OLD)))
    log("fold validity: " + "; ".join(f"seed {r['seed']}: sizes {r['fold_sizes']}, ASD/fold {r['asd_per_fold']}, valid {r['valid']}, new partition {r['differs_from_seeds_0_2']}" for r in rows)); assert all(r["valid"] and r["differs_from_seeds_0_2"] for r in rows); return new


def run_member(m, seed, data, prov_fns):
    ck = HERE / "checkpoints" / f"{m}_seed{seed}.npz"
    if ck.exists(): return
    Xa, y, site_s, sid, covs, Xp = data; model, rep = CP.EXPS[m]; n = len(y); score, fold = np.zeros(n), np.zeros(n, int); thr, dfl, ia, secs = (np.zeros(N_OUTER) for _ in range(4)); io_all = np.full((N_OUTER, n), np.nan); params = []; t0 = time.time()
    for f, (tr, te) in enumerate(StratifiedKFold(N_OUTER, shuffle=True, random_state=seed).split(Xa, y)):
        t1 = time.time(); s, io, th, df_, bp, a = CP.run_fold(model, prov_fns[rep], y, tr, te, seed, None); score[te], fold[te], thr[f], dfl[f], ia[f], secs[f] = s, f, th, df_, a, time.time() - t1; io_all[f, tr] = io; params.append(bp)
        log(f"  {m} seed {seed} fold {f}: params {bp} inner AUC {a:.4f} thr {th:.4f} ({secs[f]:.1f}s; seed total {time.time() - t0:.0f}s)")
    tmp = ck.with_name(ck.stem + "_tmp.npz"); np.savez(tmp, score=score, fold=fold, thr=thr, default=dfl, iauc=ia, secs=secs, inner_oof=io_all, params=np.array(params)); os.replace(tmp, ck)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); a = ap.parse_args(); warnings.filterwarnings("ignore"); t0 = time.time(); log("=" * 60 + f"\nconfirmation start mode={a.mode}")
    data = gate_and_load(); Xa, y, site_s, sid, covs, Xp = data; folds_new = fold_validity(y)
    if a.mode == "run":
        prov_fns = {rep: CP.provider(rep, Xa, Xp, covs) for rep in ("LW", "TS")}
        for seed in SEEDS:
            for m in ["P1_LR", "P2_LSVM", "P3_TS_LR"]:                                   # sequential; failures stop the run (no silent substitution)
                run_member(m, seed, data, prov_fns); log(f"{m} seed {seed} done")
        log(f"all members complete in {time.time() - t0:.0f}s")
    report(data, folds_new, time.time() - t0)


def report(data, folds_new, secs):
    Xa, y, site_s, sid, covs, Xp = data; ck = {(m, s): np.load(HERE / "checkpoints" / f"{m}_seed{s}.npz") for m in MEMBERS for s in SEEDS}
    for m in MEMBERS:
        for s in SEEDS: assert (ck[(m, s)]["fold"] == folds_new[s]).all(), "outer folds differ between members"
    e1 = build_e1(ck, y, folds_new, SEEDS); rows, frows, oof, runtime = [], [], [], {m: sum(float(ck[(m, s)]["secs"].sum()) for s in SEEDS) for m in MEMBERS}
    def arms(s):
        ref = dict(score=ck[("P1_LR", s)]["score"], fold=folds_new[s]); ref["pred_R1"] = (ref["score"] >= ck[("P1_LR", s)]["thr"][ref["fold"]]).astype(int); ref["pred_R0"] = (ref["score"] >= 0.5).astype(int)
        mem = {m: dict(score=ck[(m, s)]["score"], pred_R1=(ck[(m, s)]["score"] >= ck[(m, s)]["thr"][folds_new[s]]).astype(int), pred_R0=(ck[(m, s)]["score"] >= ck[(m, s)]["default"][folds_new[s]]).astype(int)) for m in MEMBERS}
        return {"reference_P1_LR": ref, "soft_vote_E1": e1[s], **{f"member_{m}": mem[m] for m in MEMBERS if m != "P1_LR"}}
    for s in SEEDS:
        A = arms(s)
        for arm, d in A.items():
            for rd, col in (("R1", "pred_R1"), ("R0", "pred_R0")):
                m = CP.metrics(y, d[col], d["score"]); fa = [accuracy_score(y[folds_new[s] == f], d[col][folds_new[s] == f]) for f in range(N_OUTER)]
                rows.append(dict(arm=arm, seed=s, reading=rd, **m, fold_accuracy_mean=np.mean(fa), fold_accuracy_sd=np.std(fa, ddof=1)))
        for f in range(N_OUTER):
            te = folds_new[s] == f; frows.append(dict(seed=s, fold=f, n_test=int(te.sum()), ref_acc_R1=accuracy_score(y[te], A["reference_P1_LR"]["pred_R1"][te]), ens_acc_R1=accuracy_score(y[te], e1[s]["pred_R1"][te]),
                                                    ref_acc_R0=accuracy_score(y[te], A["reference_P1_LR"]["pred_R0"][te]), ens_acc_R0=accuracy_score(y[te], e1[s]["pred_R0"][te]), ens_thr=float(e1[s]["thr"][f]), ref_thr=float(ck[("P1_LR", s)]["thr"][f])))
        for arm in ("reference_P1_LR", "soft_vote_E1"):
            d = A[arm]; oof += [dict(arm=arm, seed=s, fold=int(folds_new[s][i]), subject_id=sid[i], site=site_s[i], y_true=int(y[i]), score=float(d["score"][i]), pred_R0=int(d["pred_R0"][i]), pred_R1=int(d["pred_R1"][i])) for i in range(len(y))]
    SM, FM = pd.DataFrame(rows), pd.DataFrame(frows); SM.to_csv(HERE / "seed_metrics.csv", index=False); FM.to_csv(HERE / "fold_comparison.csv", index=False); pd.DataFrame(oof).to_csv(HERE / "oof_predictions.csv", index=False)
    g = lambda arm, rd, s, k: float(SM[(SM.arm == arm) & (SM.reading == rd) & (SM.seed == s)][k].iloc[0]); res = {}
    for rd in ("R1", "R0"):
        da = np.array([g("soft_vote_E1", rd, s, "accuracy") - g("reference_P1_LR", rd, s, "accuracy") for s in SEEDS]) * 100; dau = np.array([g("soft_vote_E1", rd, s, "roc_auc") - g("reference_P1_LR", rd, s, "roc_auc") for s in SEEDS])
        db = np.array([g("soft_vote_E1", rd, s, "balanced_accuracy") - g("reference_P1_LR", rd, s, "balanced_accuracy") for s in SEEDS]) * 100
        diff = (FM[f"ens_acc_{rd}"] - FM[f"ref_acc_{rd}"]).values * 100
        res[rd] = dict(d_acc_by_seed=da.tolist(), d_bacc_by_seed=db.tolist(), d_auc_by_seed=dau.tolist(), mean_d_acc=float(da.mean()), sd_d_acc_seeds=float(da.std(ddof=1)), mean_d_bacc=float(db.mean()), mean_d_auc=float(dau.mean()),
                       n_seeds_positive=int((da > 0).sum()), fold_wins=int((diff > 0).sum()), fold_ties=int((diff == 0).sum()), fold_losses=int((diff < 0).sum()), fold_diff_mean=float(diff.mean()), fold_diff_sd=float(diff.std(ddof=1)), fold_diff_se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                       sign_test_p=float(stats.binom.sf(int((diff > 0).sum()) - 1, int((diff != 0).sum()) if False else len(diff), .5)))
    r = res["R1"]; c1, c2, c3 = r["mean_d_acc"] >= 0.5, r["n_seeds_positive"] >= 2, r["mean_d_auc"] >= 0.01; passed = bool(c1 and c2 and c3)
    hashes = {"preregistration.md": sha(HERE / "preregistration.md"), "confirm.py": sha(HERE / "confirm.py"), "campaign.py": sha(CAMP / "campaign.py"), "selection_rule.py": sha(CAMP / "selection_rule.py"), "phase5.py": sha(CAMP / "phase5.py"),
              "audit_report.md": sha(CAMP / "independent_audit" / "audit_report.md"), "P5_E1_config": sha(CAMP / "results" / "P5_E1_SOFTVOTE_config.json"), "cache_provenance": sha(CAMP / "cache" / "provenance.json"),
              "baseline_cache_files": {k: sha(CP.BASE / k) for k in CP.HASHES}, "checkpoints": {p.name: sha(p) for p in sorted((HERE / "checkpoints").glob("*.npz"))}}
    json.dump(hashes, open(HERE / "hashes.json", "w"), indent=1)
    json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "preregistered_criteria": {"mean_accuracy_gain_pts>=": 0.5, "seeds_with_positive_gain>=": 2, "mean_auc_gain>=": 0.01, "reading": "R1"}, "results": res, "criteria_met": {"1_mean_acc_gain": bool(c1), "2_positive_in_2_of_3_seeds": bool(c2), "3_mean_auc_gain": bool(c3)}, "PASS": passed,
               "members": MEMBERS, "seeds": list(SEEDS), "runtime_s": round(secs, 1), "member_runtime_s": runtime, "protected_holdout_accessed": False}, open(HERE / "config.json", "w"), indent=1)
    def trow(arm, rd):
        return "\n".join(f"| {arm} | {s} | {pc(g(arm, rd, s, 'accuracy'))} | {pc(g(arm, rd, s, 'balanced_accuracy'))} | {g(arm, rd, s, 'roc_auc'):.4f} | {pc(g(arm, rd, s, 'sensitivity'))} | {pc(g(arm, rd, s, 'specificity'))} | {pc(g(arm, rd, s, 'precision'))} | {pc(g(arm, rd, s, 'f1'))} | {pc(float(SM[(SM.arm == arm) & (SM.reading == rd) & (SM.seed == s)].fold_accuracy_sd.iloc[0]))} |" for s in SEEDS)
    def cmrow(rd):
        return "\n".join(f"| {arm} | {s} | {int(g(arm, rd, s, 'TP'))} | {int(g(arm, rd, s, 'FN'))} | {int(g(arm, rd, s, 'FP'))} | {int(g(arm, rd, s, 'TN'))} |" for s in SEEDS for arm in ("reference_P1_LR", "soft_vote_E1"))
    mean_row = lambda arm, rd, k: np.mean([g(arm, rd, s, k) for s in SEEDS])
    members_tab = "\n".join(f"| {arm} | {pc(mean_row(arm, 'R1', 'accuracy'))} | {mean_row(arm, 'R1', 'roc_auc'):.4f} | {pc(mean_row(arm, 'R0', 'accuracy'))} |" for arm in ("reference_P1_LR", "member_P2_LSVM", "member_P3_TS_LR", "soft_vote_E1"))
    fold_tab = "\n".join(f"| {int(x.seed)} | {int(x.fold)} | {pc(x.ref_acc_R1)} | {pc(x.ens_acc_R1)} | {(x.ens_acc_R1 - x.ref_acc_R1) * 100:+.2f} |" for x in FM.itertuples())
    verdict = ("**PASS: the ensemble is supported by this confirmation (all three pre-registered conditions hold).**" if passed else "**FAIL: the ensemble is NOT supported by this confirmation (at least one pre-registered condition does not hold).**")
    txt = f"""# Pre-registered fresh-seed confirmation: reference P1_LR vs soft vote of (P3_TS_LR, P2_LSVM, P1_LR)
Seeds 3, 4, 5 (new outer partitions of the same 676 authorized training subjects; identical folds for all arms); CC200 only; development cross-validation, not independent-test performance. The protected holdout was **not accessed**.
Pre-registration: `preregistration.md` (criteria fixed before running; evaluated under R1 = threshold chosen from training-only inner scores; R0 = default thresholds is secondary).
Members and settings were read from the saved artifacts; the rebuilt soft vote reproduced the saved seed 0-2 predictions exactly before the fresh seeds were run. Runtime {secs / 60:.1f} min (members, summed fold time: {', '.join(f'{m} {t / 60:.1f} min' for m, t in runtime.items())}); no failures.

## 1. Decision against the pre-registered criteria (R1)
| Criterion | Required | Observed | Met |
|---|---|---|---|
| 1. Mean accuracy gain over seeds 3-5 | >= +0.5 pt | {r['mean_d_acc']:+.2f} pt | {'yes' if c1 else 'NO'} |
| 2. Seeds with positive accuracy gain | >= 2 of 3 | {r['n_seeds_positive']} of 3 (gains {', '.join(f'{v:+.2f}' for v in r['d_acc_by_seed'])} pt) | {'yes' if c2 else 'NO'} |
| 3. Mean ROC-AUC gain | >= +0.01 | {r['mean_d_auc']:+.4f} (by seed {', '.join(f'{v:+.4f}' for v in r['d_auc_by_seed'])}) | {'yes' if c3 else 'NO'} |
{verdict}
This confirmation does not alter the campaign verdict: no candidate passed the frozen rule, and 85% was not achieved.

## 2. Per-seed results, R1 (primary)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
{trow('reference_P1_LR', 'R1')}
{trow('soft_vote_E1', 'R1')}
Paired differences (ensemble minus reference): accuracy {', '.join(f'{v:+.2f}' for v in r['d_acc_by_seed'])} pt (mean {r['mean_d_acc']:+.2f}, SD over seeds {r['sd_d_acc_seeds']:.2f}); balanced accuracy mean {r['mean_d_bacc']:+.2f} pt; AUC mean {r['mean_d_auc']:+.4f}.
Paired outer folds (30): ensemble higher in {r['fold_wins']}, equal in {r['fold_ties']}, lower in {r['fold_losses']}; mean fold difference {r['fold_diff_mean']:+.2f} pt, SD {r['fold_diff_sd']:.2f}, SE {r['fold_diff_se']:.2f}; one-sided sign-test p = {r['sign_test_p']:.3f} (descriptive only).
Confusion matrices, R1 (pooled out-of-fold, 313 ASD / 363 TC):
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
{cmrow('R1')}

## 3. Secondary: default thresholds (R0; not part of the decision)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
{trow('reference_P1_LR', 'R0')}
{trow('soft_vote_E1', 'R0')}
R0 accuracy gain by seed: {', '.join(f'{v:+.2f}' for v in res['R0']['d_acc_by_seed'])} pt (mean {res['R0']['mean_d_acc']:+.2f}); mean AUC gain {res['R0']['mean_d_auc']:+.4f}.

## 4. Members for context (mean over seeds 3-5)
| Arm | Accuracy R1 | ROC-AUC | Accuracy R0 |
|---|---|---|---|
{members_tab}
Per-seed member metrics: `seed_metrics.csv`.

## 5. Fold-level comparison (R1)
| Seed | Fold | Reference acc. | Ensemble acc. | Diff (pt) |
|---|---|---|---|---|
{fold_tab}

## 6. Integrity
Hashes of inputs, source code, configurations and checkpoints: `hashes.json` (campaign code and selection rule unchanged, equal to the recorded values). Valid subject-level outer folds for seeds 3-5 (each subject in exactly one test fold; sizes 67-68; 31-32 ASD per fold; partitions differ from seeds 0-2). Campaign outputs were only read; the campaign log was not appended to. Holdout, restricted membership file and combined arrays: not accessed.
Limitations: same 676 subjects as in seeds 0-2 (new partitions, not new subjects); 30 paired folds are not independent; accuracy SE about 1.8 pt; the conclusion concerns a small ensembling effect on this cohort only.
"""
    (HERE / "report.md").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
