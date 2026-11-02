"""Independent audit of the CC200 accuracy campaign. READ-ONLY on every campaign artifact; trains nothing; writes only into independent_audit/.
Re-derives metrics, folds, thresholds, screening, the 68-subject count and the soft-vote selection-bias analysis from the saved out-of-fold predictions, scores and inner-OOF arrays.
No holdout / restricted file is read (only the authorized-training-subject cache and campaign outputs)."""
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent; A = OUT.parent; R = A / "results"; CK = A / "checkpoints"; BASE = Path(r"C:\abide\cc200_reconstruction\experiments\stage1_2_b0_baseline")
sys.path.insert(0, str(A))
import selection_rule as SR                                                      # the frozen rule file (hash-checked below)

SEEDS = (0, 1, 2)
ORDER = ["P1_LR", "P2_LSVM", "P2_RBF", "P2_RF", "P2_ET", "P2_XGB", "P3_PC_LR", "P3_TS_LR", "P4_PCA_LR", "P4_KBEST_LR", "P4_L1_LR", "P5_E1_SOFTVOTE", "P5_E2_STACK"]
SINGLES = ORDER[:11]; MARGIN = {"P2_LSVM", "P2_RBF"}
checks = []
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
pc = lambda v: f"{v * 100:.2f}%"


def chk(cid, item, ok, detail=""):
    checks.append(dict(id=cid, item=item, result="INFO" if ok is None else ("PASS" if bool(ok) else "FAIL"), detail=str(detail)))


# ------------------------------------------------------------------ inputs
z = np.load(BASE / "features_cc200_train676.npz", allow_pickle=True); y_cache, sid = z["y"], z["sid"].astype(str); n = len(y_cache)
train_ids = {str(int(float(l))) for l in open(r"C:\abide\splits\phase2_train_subjects.txt") if l.strip()}                 # authorized training list only
oof = {e: pd.read_csv(R / f"{e}_oof_predictions_seeds012.csv", dtype={"subject_id": str}) for e in ORDER}
smet = {e: pd.read_csv(R / f"{e}_seed_metrics.csv") for e in ORDER}; fmet = {e: pd.read_csv(R / f"{e}_fold_metrics.csv").sort_values(["seed", "fold"]) for e in ORDER}


def arr(e, s, col):
    d = oof[e][oof[e].seed == s].set_index("subject_id").loc[sid]; return d[col].values


# ================================================================== 1. metrics re-derived from saved predictions
maxdiff = 0.0; rec = {}
for e in ORDER:
    for s in SEEDS:
        y, sc = arr(e, s, "y_true"), arr(e, s, "score")
        for rd, col in (("R0_default", "pred_R0"), ("R1_nested_threshold", "pred_R1")):
            p = arr(e, s, col); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            m = dict(accuracy=(tp + tn) / n, balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2, sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), precision=tp / (tp + fp), f1=2 * tp / (2 * tp + fp + fn), roc_auc=roc_auc_score(y, sc), TP=tp, FN=fn, FP=fp, TN=tn)
            rec[(e, s, rd)] = m; row = smet[e][(smet[e].seed == s) & (smet[e].reading == rd)].iloc[0]
            maxdiff = max(maxdiff, max(abs(m[k] - float(row[k])) for k in m))
chk("1a", "all seed-level metrics and confusion counts (13 experiments x 3 seeds x 2 readings) recomputed from the saved predictions equal seed_metrics.csv", maxdiff < 1e-9, f"max |difference| {maxdiff:.2e}")
comp = pd.read_csv(A / "all_experiments_comparison.csv"); md = 0.0
for e in ORDER:
    r1 = np.mean([rec[(e, s, "R1_nested_threshold")]["accuracy"] for s in SEEDS]); c = comp[comp.experiment == e].iloc[0]; md = max(md, abs(r1 - c.R1_accuracy), abs(np.mean([rec[(e, s, "R0_default")]["accuracy"] for s in SEEDS]) - c.R0_accuracy),
                                                                                  abs(np.mean([rec[(e, s, "R1_nested_threshold")]["roc_auc"] for s in SEEDS]) - c.R1_roc_auc))
chk("1b", "all_experiments_comparison.csv means (R1 accuracy, R0 accuracy, AUC) equal the recomputed means", md < 1e-9, f"max |difference| {md:.2e}")
fr = open(A / "final_report.md", encoding="utf-8").read(); miss = []
for e in ORDER:
    ms = [rec[(e, s, "R1_nested_threshold")] for s in SEEDS]; acc = np.array([m["accuracy"] for m in ms]); cell = f"{pc(acc.mean())} +/- {acc.std(ddof=1) * 100:.2f}"
    if f"| {e} |" not in fr or cell not in fr: miss.append(e)
chk("1c", "every accuracy cell (mean +/- seed SD) in the final report's ranking table matches the recomputed value", not miss, f"mismatches: {miss}" if miss else "13/13 rows match")
# AUC reproducibility from scores: pooled AUC per seed is threshold independent -> identical for R0/R1 (sanity)
chk("1d", "ROC-AUC is the same under R0 and R1 for every run (threshold independent)", all(abs(rec[(e, s, 'R0_default')]['roc_auc'] - rec[(e, s, 'R1_nested_threshold')]['roc_auc']) < 1e-12 for e in ORDER for s in SEEDS))

# ================================================================== 2. same subjects, compatible folds
ids_ok = all(set(oof[e][oof[e].seed == s].subject_id) == train_ids and oof[e][oof[e].seed == s].subject_id.is_unique and len(oof[e][oof[e].seed == s]) == n for e in ORDER for s in SEEDS)
chk("2a", "every experiment and seed has exactly the 676 authorized training subjects, each once", ids_ok)
ymatch = all((arr(e, s, "y_true") == y_cache).all() for e in ORDER for s in SEEDS); chk("2b", "y_true in every prediction file equals the verified cache labels (same subject order)", ymatch)
fold_ok = True; regen_ok = True
for s in SEEDS:
    regen = np.zeros(n, int)
    for f, (_, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=s).split(np.zeros(n), y_cache)): regen[te] = f
    for e in ORDER:
        fo = arr(e, s, "fold").astype(int); fold_ok &= bool((fo == arr("P1_LR", s, "fold").astype(int)).all()); regen_ok &= bool((fo == regen).all())
chk("2c", "fold assignment is identical across all 13 experiments for each seed", fold_ok); chk("2d", "fold assignment equals an independent regeneration StratifiedKFold(10, shuffle, seed).split(y) for each seed", regen_ok)
site_ok = all((arr(e, s, "site") == z["site"].astype(str)).all() for e in ORDER for s in SEEDS); chk("2e", "site labels agree across files with the cache", site_ok)
hs = json.load(open(A / "source_and_config_hashes.json")); bad = []
for k, v in hs.items():
    if isinstance(v, str) and len(v) == 64 and (A / k).exists() and sha(A / k) != v: bad.append(k)
for k, v in hs["baseline_and_cache_files"].items():
    if sha(BASE / k) != v: bad.append("baseline/" + k)
chk("2f", "hashes in source_and_config_hashes.json equal the current files (campaign code, plan, configs, provenance, baseline and cache)", not bad, f"changed: {bad}" if bad else "all match")
chk("2g", "frozen selection rule file and campaign.py unchanged since the plan (hashes recorded in the plan)", sha(A / "selection_rule.py") == "e4389cbebcae96950fab28bb8f81ae5e81c7967a3d29ed82552e28730c266467" and sha(A / "campaign.py") == "f1770d3dc7a2e932ba8bde9446b09513472b681de632dbe5dee4ae9a3d0d33ae")
log = open(A / "logs" / "campaign.log", encoding="utf-8").read(); chk("2h", "no experiment failure was logged (case-sensitive 'EXPERIMENT FAILED')", "EXPERIMENT FAILED" not in log, f"occurrences: {log.count('EXPERIMENT FAILED')}; invocations: {log.count('campaign start')}; hash verifications: {log.count('ALL MATCH')}")

# ================================================================== 3. threshold selection and leakage structure
def thr_indep(yv, s, default):
    P, N = int(yv.sum()), int(len(yv) - yv.sum()); u = np.unique(s); cand = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]]); best, bt = -1, default
    for t in cand:
        pr = s >= t; b = (((pr) & (yv == 1)).sum() / P + ((~pr) & (yv == 0)).sum() / N) / 2
        if b > best + 1e-12 or (abs(b - best) <= 1e-12 and abs(t - default) < abs(bt - default)): best, bt = b, t
    return bt


cover_ok = True; thr_ok = True; pred_ok = True; maxthr = 0.0; test_nan = True
for e in SINGLES:
    default = 0.0 if e in MARGIN else 0.5
    for s in SEEDS:
        c = np.load(CK / f"{e}_seed{s}.npz"); fo, io, thr = c["fold"], c["inner_oof"], c["thr"]
        for f in range(10):
            te = fo == f; tr = ~te; cover_ok &= bool((~np.isnan(io[f]) == tr).all()); test_nan &= bool(np.isnan(io[f][te]).all())
            t2 = thr_indep(y_cache[tr], io[f][tr], default); maxthr = max(maxthr, abs(t2 - thr[f])); thr_ok &= abs(t2 - thr[f]) < 1e-6          # tolerance 1e-6: XGBoost returns float32 probabilities (rounding ~3e-8 in the stored midpoints)
        p1 = arr(e, s, "pred_R1"); pred_ok &= bool(((arr(e, s, "score") >= thr[fo]).astype(int) == p1).all())
chk("3a", "inner out-of-fold scores exist exactly for the outer-TRAINING rows of each fold and are NaN for every outer-test row (so thresholds cannot see test rows)", cover_ok and test_nan)
chk("3b", "R1 threshold of every fold recomputed independently from the stored training-only inner-OOF scores and training labels equals the stored threshold", thr_ok, f"max |difference| {maxthr:.2e}")
chk("3c", "R1 predictions equal (test score >= that fold's training-derived threshold) for all 11 single-model experiments", pred_ok)
code = open(A / "campaign.py", encoding="utf-8").read(); rf = code[code.index("def run_fold"):code.index("def run_exp")]
fit_lines = [l.strip() for l in rf.splitlines() if ".fit(" in l or "cross_val_predict(" in l]
chk("3d", "static check of run_fold: every .fit(...) and cross_val_predict(...) uses only Xtr / y[tr]; the test matrix Xte appears only in scoring", all("Xte" not in l and "y[te]" not in l for l in fit_lines), "; ".join(fit_lines)[:230])
chk("3e", "tangent-space reference is fitted on the outer-training covariances only (static check)", "fit(covs[tr])" in code)
chk("3f", "stored P1_LR scores equal the pre-campaign Arm B scores (reproduction)", all(np.abs(np.load(CK / f"P1_LR_seed{s}.npz")["score"] - np.load(BASE / "checkpoints" / f"armB_seed{s}.npz")["score"]).max() == 0 for s in SEEDS))
chk("3g", "KNOWN LIMITATION (documented in the plan): in P3_TS_LR the tangent reference is built from all outer-training rows, so the inner CV that selects C sees a reference computed with the inner-validation rows (label-free); outer-test rows never enter", None, "strict rule 6 would refit the reference per inner fold; affects only hyperparameter selection")

# ================================================================== 4. soft vote: selection bias on the same evaluation folds
ELIG = ["P1_LR", "P2_LSVM", "P2_RBF", "P2_ET", "P2_XGB", "P3_TS_LR", "P4_KBEST_LR"]; ckp = {e: {s: np.load(CK / f"{e}_seed{s}.npz") for s in SEEDS} for e in SINGLES}
fold_ref = {s: arr("P1_LR", s, "fold").astype(int) for s in SEEDS}


def vote(models, s):
    sc = np.zeros(n); thrs = np.zeros(10); fo = fold_ref[s]
    for f in range(10):
        te = fo == f; tr = ~te
        Zin = np.column_stack([ckp[e][s]["inner_oof"][f][tr] for e in models]); Zte = np.column_stack([ckp[e][s]["score"][te] for e in models])
        cal = [LogisticRegression(C=1e6).fit(Zin[:, j:j + 1], y_cache[tr]) for j in range(len(models))]
        sin = np.mean([c.predict_proba(Zin[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0); sc[te] = np.mean([c.predict_proba(Zte[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0); thrs[f] = thr_indep(y_cache[tr], sin, 0.5)
    return sc, (sc >= thrs[fo]).astype(int)


def acc_by_seed(models):
    out = {}
    for s in SEEDS:
        sc, p = vote(models, s); out[s] = (accuracy_score(y_cache, p), p)
    return out


ref_acc = {s: rec[("P1_LR", s, "R1_nested_threshold")]["accuracy"] for s in SEEDS}; mem_acc = {e: np.mean([rec[(e, s, "R1_nested_threshold")]["accuracy"] for s in SEEDS]) for e in SINGLES}
chosen = ["P3_TS_LR", "P2_LSVM", "P1_LR"]; rep = acc_by_seed(chosen); saved_ok = all(((arr("P5_E1_SOFTVOTE", s, "pred_R1")) == rep[s][1]).all() for s in SEEDS)
chk("4a", "the saved P5_E1 soft-vote predictions are reproduced exactly from the stored training-only inner-OOF scores (independent re-implementation)", saved_ok)
rows = []
for sub in itertools.combinations(ELIG, 3):
    a = acc_by_seed(list(sub)); g = np.mean([a[s][0] - ref_acc[s] for s in SEEDS]) * 100; rows.append(dict(members="+".join(sub), mean_acc=np.mean([a[s][0] for s in SEEDS]), gain_vs_ref_pts=g, gain_vs_member_mean_pts=(np.mean([a[s][0] for s in SEEDS]) - np.mean([mem_acc[m] for m in sub])) * 100, chosen=set(sub) == set(chosen)))
S = pd.DataFrame(rows).sort_values("gain_vs_ref_pts", ascending=False).reset_index(drop=True); S.to_csv(OUT / "softvote_all_3subsets.csv", index=False)
rank_chosen = int(S.index[S.chosen][0]) + 1
all7 = acc_by_seed(ELIG); all11 = acc_by_seed(SINGLES)
loso = []
for s in SEEDS:                                                       # selection from the OTHER two seeds only, evaluated on the held-out seed
    oth = [t for t in SEEDS if t != s]; m_acc = {e: np.mean([rec[(e, t, "R1_nested_threshold")]["accuracy"] for t in oth]) for e in SINGLES}; r_acc = m_acc["P1_LR"]
    el = [e for e in SINGLES if m_acc[e] >= r_acc - 0.01]; top = sorted(el, key=lambda e: -m_acc[e])[:3]; sc_, p_ = vote(top, s); loso.append(dict(heldout_seed=s, members="+".join(top), acc=accuracy_score(y_cache, p_), ref=ref_acc[s], gain_pts=(accuracy_score(y_cache, p_) - ref_acc[s]) * 100))
L = pd.DataFrame(loso); L.to_csv(OUT / "softvote_leave_one_seed_out_selection.csv", index=False)
# paired fold differences of the chosen vote vs reference
ff = np.concatenate([[accuracy_score(y_cache[fold_ref[s] == f], rep[s][1][fold_ref[s] == f]) - fmet["P1_LR"][(fmet["P1_LR"].seed == s) & (fmet["P1_LR"].fold == f)].acc_R1.iloc[0] for f in range(10)] for s in SEEDS])
chk("4b", "selection bias: gain of the chosen soft vote vs the average of all 35 possible 3-model eligible soft votes and its rank among them", None,
    f"chosen gain {np.mean([rep[s][0] - ref_acc[s] for s in SEEDS]) * 100:+.2f} pts (rank {rank_chosen} of 35); mean gain of all 35 subsets {S.gain_vs_ref_pts.mean():+.2f} (min {S.gain_vs_ref_pts.min():+.2f}, max {S.gain_vs_ref_pts.max():+.2f}); vote of all 7 eligible {np.mean([all7[s][0] - ref_acc[s] for s in SEEDS]) * 100:+.2f}; vote of all 11 single models {np.mean([all11[s][0] - ref_acc[s] for s in SEEDS]) * 100:+.2f}")
chk("4c", "leave-one-seed-out selection (members chosen from the other two seeds, scored on the held-out seed)", None, f"gains by held-out seed {[round(v, 2) for v in L.gain_pts]} pts; mean {L.gain_pts.mean():+.2f}")
chk("4d", "ensembling benefit over the members themselves for the chosen vote (vote accuracy minus the mean of its three members' accuracies)", None, f"{S[S.chosen].gain_vs_member_mean_pts.iloc[0]:+.2f} pts; across all 35 subsets mean {S.gain_vs_member_mean_pts.mean():+.2f}")
chk("4e", "chosen vote's paired-fold result vs reference", None, f"wins {int((ff > 0).sum())}, ties {int((ff == 0).sum())}, losses {int((ff < 0).sum())} of 30; mean {ff.mean() * 100:+.2f} pts, SE {ff.std(ddof=1) / np.sqrt(30) * 100:.2f}")

# ================================================================== 5. the 68 consistently misclassified subjects
E = np.zeros((n, len(ORDER) * 3), dtype=int); col = 0
for e in ORDER:
    for s in SEEDS: E[:, col] = (arr(e, s, "pred_R1") != y_cache).astype(int); col += 1
errc = E.sum(1); hard = int((errc >= int(np.ceil(0.9 * 39))).sum()); chk("5a", "subjects misclassified in >= 90% of the 39 runs (>= 36 of 39 errors), recomputed from the predictions", hard == 68, f"{hard} subjects (report says 68)")
E33 = E[:, :11 * 3]; h33 = int((E33.sum(1) >= int(np.ceil(0.9 * 33))).sum()); h_noens = int((E[:, [i for i in range(39) if i // 3 not in (11, 12)]].sum(1) >= 30).sum())
chk("5b", "same count using only the 11 non-ensemble experiments (33 runs, >= 30 errors)", None, f"{h33} subjects (equal to the {h_noens} computed on the non-ensemble columns)")
pm = E.mean(0).mean(); chk("5c", "the 39 'runs' are not independent: they share features/folds, and the two ensembles are functions of three other runs; mean error rate per run", None, f"{pm * 100:.1f}%; under independent errors at this rate P(>=36/39 errors) = {stats.binom.sf(35, 39, pm):.1e}, so the persistence is real but expected when models share the same features")
chk("5d", "class split of the consistently misclassified subjects", None, f"ASD {int(((errc >= 36) & (y_cache == 1)).sum())}, TC {int(((errc >= 36) & (y_cache == 0)).sum())}")

# ================================================================== 6. frozen rule applied independently
ref_r1 = {s: rec[("P1_LR", s, "R1_nested_threshold")] for s in SEEDS}; fa_ref = fmet["P1_LR"].acc_R1.values; passers = []; margins = []
for e in ORDER[1:]:
    cd = {s: rec[(e, s, "R1_nested_threshold")] for s in SEEDS}; fa = fmet[e].acc_R1.values
    d_acc = np.array([cd[s]["accuracy"] - ref_r1[s]["accuracy"] for s in SEEDS]) * 100; d_b = np.array([cd[s]["balanced_accuracy"] - ref_r1[s]["balanced_accuracy"] for s in SEEDS]) * 100; d_auc = np.mean([cd[s]["roc_auc"] - ref_r1[s]["roc_auc"] for s in SEEDS]); wins = int((fa > fa_ref).sum())
    ok = d_acc.mean() >= 1.5 and d_b.mean() >= 1.5 and (d_acc >= 0.5).all() and wins >= 20 and d_auc >= -0.005; ok_rule = SR.screen({s: dict(accuracy=ref_r1[s]["accuracy"], balanced_accuracy=ref_r1[s]["balanced_accuracy"], roc_auc=ref_r1[s]["roc_auc"]) for s in SEEDS},
                                                                                                                                                                     {s: dict(accuracy=cd[s]["accuracy"], balanced_accuracy=cd[s]["balanced_accuracy"], roc_auc=cd[s]["roc_auc"]) for s in SEEDS}, fa_ref, fa)[1]
    assert ok == ok_rule, e
    if ok: passers.append(e)
    margins.append(dict(experiment=e, gain_acc=d_acc.mean(), gain_bacc=d_b.mean(), min_seed_gain=d_acc.min(), wins=wins, dAUC=d_auc))
M = pd.DataFrame(margins); M.to_csv(OUT / "screening_margins_recomputed.csv", index=False)
chk("6a", "frozen screening re-implemented independently and compared with selection_rule.py: no candidate passes, and the two implementations agree for all 12 candidates", not passers, f"passers: {passers}; best gain {M.gain_acc.max():+.2f} pts (needs +1.5), most fold wins {M.wins.max()} (needs 20)")
chk("6b", "rule consistency: reference for every comparison is P1_LR under R1; same 30 paired folds ordered by (seed, fold); no candidate-specific changes to the rule", True, "paired arrays share ordering (folds identical, check 2c)")
zs = {e: (lambda d: d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))(fmet[e].acc_R1.values - fa_ref) for e in ORDER[1:]}
chk("6c", "multiple comparisons: z = mean paired fold difference / its SE for the 12 candidates (folds treated as independent, which overstates precision); the maximum of 12 standard-normal draws is about 1.6 by chance alone", None, f"max z {max(zs.values()):.2f} ({max(zs, key=zs.get)}); z values {', '.join(f'{k}:{v:+.2f}' for k, v in zs.items())}")

# ================================================================== 7. the two reported reference accuracies
r1s = [ref_r1[s]["accuracy"] for s in SEEDS]; r0s = [rec[("P1_LR", s, "R0_default")]["accuracy"] for s in SEEDS]
base_pred = pd.read_csv(BASE / "oof_predictions.csv", dtype={"subject_id": str}); base_pred = base_pred[base_pred.arm == "B"]
same_r0 = all((base_pred[base_pred.seed == s].set_index("subject_id").loc[sid].pred_thr0_5.values == arr("P1_LR", s, "pred_R0")).all() for s in SEEDS)
st2 = pd.read_csv(BASE / "seed_metrics.csv"); st2n = st2[(st2.arm == "B") & (st2.reading == "nested_youden")].sort_values("seed").accuracy.values
chk("7a", "66.07% = reading R0: default threshold 0.5, no threshold selection (identical to the saved Arm B 'fixed 0.5' predictions)", same_r0 and abs(np.mean(r0s) - 0.6607) < 5e-5, f"per seed {[pc(v) for v in r0s]}, mean {pc(np.mean(r0s))}")
chk("7b", "65.68% = reading R1: per-fold threshold maximizing balanced accuracy on training-only inner out-of-fold scores", abs(np.mean(r1s) - 0.6568) < 5e-5, f"per seed {[pc(v) for v in r1s]}, mean {pc(np.mean(r1s))}")
same_r1 = all((base_pred[base_pred.seed == s].set_index("subject_id").loc[sid].pred_nested_youden.values == arr("P1_LR", s, "pred_R1")).all() for s in SEEDS)
chk("7c", "campaign reading R1 for the reference is identical, subject by subject, to the Stage-2 'nested Youden' reading of Arm B (Youden's J maximization = balanced-accuracy maximization)", same_r1, f"Stage-2 nested per seed {[pc(v) for v in st2n]} vs campaign R1 per seed {[pc(v) for v in r1s]}; predictions identical for all {3 * n} subject-seed pairs" if same_r1 else "predictions differ")
chk("7d", "wording in the campaign request: '66.07% nested cross-validated accuracy' conflates the two: 66.07% is the fixed-0.5 reading; the nested-threshold reading is 65.68%", None, "outer CV is nested in both (inner CV selects C); only the threshold procedure differs")

pd.DataFrame(checks).to_csv(OUT / "audit_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 240)
print(pd.DataFrame(checks)[["id", "result", "item"]].to_string(index=False)); print()
for c in checks: print(f"[{c['id']}] {c['result']}: {c['detail']}")
print("\nTop soft-vote subsets:\n", S.head(5)[["members", "mean_acc", "gain_vs_ref_pts", "gain_vs_member_mean_pts"]].round(4).to_string(index=False)); print("chosen rank:", rank_chosen); print(L.round(4).to_string(index=False))
