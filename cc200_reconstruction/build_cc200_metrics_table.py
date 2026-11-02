"""Builds metrics_table.csv for the CC200 reconstruction audit. READ-ONLY on historical files; writes only inside C:\\abide\\cc200_reconstruction.
Inputs are the historical result tables/logs (fc_v2_results.csv, fc_improved_results.csv, fc_results.csv, results/phase2h_rbf_vs_logreg_summary.csv, fc_full3.log).
No CC200 array, holdout list, label or feature file is opened; nothing is trained or re-evaluated. Counts that were not stored are DERIVED from the recorded
rates and class sizes and are accepted only if every recorded metric is reproduced (otherwise the row carries no counts)."""
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(r"C:\abide"); OUT = ROOT / "cc200_reconstruction"
POOL_ASD, POOL_TC = 391, 455          # 846-subject CC200 pool (fc_quick.log / fc_full3.log)
rows = []


def metrics_from_counts(tp, fn, fp, tn):
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else np.nan
    return dict(accuracy=(tp + tn) / (tp + fn + fp + tn), balanced_accuracy=(sens + spec) / 2, sensitivity=sens, specificity=spec, precision=prec, f1=2 * tp / (2 * tp + fp + fn))


def derive_counts(n_total, sens, spec, ho=True):
    """Find (n_asd, TP, TN) so that the recorded sensitivity and specificity are exact fractions; returns None unless unique."""
    sols = []
    for n_asd_ho in range(60, 100):                                   # the 20% stratified holdout holds ~78-79 ASD
        n_tc_ho = 170 - n_asd_ho
        n_asd, n_tc = (n_asd_ho, n_tc_ho) if ho else (POOL_ASD - n_asd_ho, POOL_TC - n_tc_ho)
        tp, tn = round(sens * n_asd), round(spec * n_tc)
        if abs(tp / n_asd - sens) < 1e-9 and abs(tn / n_tc - spec) < 1e-9:
            sols.append((n_asd, n_tc, tp, tn))
    return sols[0] if len(sols) == 1 else None


def add(rid, desc, src, protocol, n, rep, counts=None, origin="not stored", sd=np.nan, status="", auc=np.nan, n_asd=np.nan, n_tc=np.nan, features="CC200 Ledoit-Wolf correlation + Fisher-Z (19,900) + age + sex",
        classifier="L2 logistic regression"):
    r = dict(result_id=rid, description=desc, source_file=src, atlas="CC200", features=features, classifier=classifier, evaluation_protocol=protocol, n=n, n_asd=n_asd, n_tc=n_tc,
             accuracy=rep.get("accuracy", np.nan), balanced_accuracy=rep.get("balanced_accuracy", np.nan), sensitivity=rep.get("sensitivity", np.nan), specificity=rep.get("specificity", np.nan),
             precision=rep.get("precision", np.nan), f1=rep.get("f1", np.nan), roc_auc=auc, TP=np.nan, FN=np.nan, FP=np.nan, TN=np.nan, accuracy_sd=sd, counts_origin=origin, consistency_check="n/a (no counts)",
             validation_status=status)
    if counts:
        tp, fn, fp, tn = counts; r.update(TP=tp, FN=fn, FP=fp, TN=tn); calc = metrics_from_counts(tp, fn, fp, tn)
        bad = [k for k, v in calc.items() if k in rep and not (abs(v - rep[k]) < 2e-4)]
        r["consistency_check"] = "PASS: " + ", ".join(k for k in calc if k in rep) if not bad else "FAIL: " + ", ".join(bad)
        r["precision"] = calc["precision"] if np.isnan(r["precision"]) else r["precision"]            # precision only available via the counts when derived
        if np.isnan(r["f1"]): r["f1"] = calc["f1"]
        r["n_asd"], r["n_tc"] = tp + fn, fp + tn
    rows.append(r)


HOLD = "Old seed-42 holdout (n=170, label-stratified), REUSED across scripts; one fit on 676 train/val, threshold from OOF Youden"
STAT_H = "Recorded historical metric; reused holdout; NOT independently validated"
STAT_C = "Recorded development CV; Youden threshold chosen on the same OOF scores (mild optimism)"

# ---- 1) fc_v2_results.csv (the 74.12% row)
v2 = pd.read_csv(ROOT / "fc_v2_results.csv").iloc[0]
ho = derive_counts(170, v2.ho_sens, v2.ho_spec, ho=True)
if ho:
    n_a, n_t, tp, tn = ho; hc = (tp, n_a - tp, n_t - tn, tn)
    add("H1", "fc_v2 EXP-2A LogReg-L2, holdout (the reported 74.12%)", "fc_v2_results.csv", HOLD + "; --quick run inferred (5-fold, 1 seed)", 170,
        dict(accuracy=v2.ho_acc, balanced_accuracy=v2.ho_bacc, sensitivity=v2.ho_sens, specificity=v2.ho_spec, f1=v2.ho_f1), counts=hc, origin="derived from recorded sensitivity/specificity and class sizes",
        status=STAT_H, auc=v2.ho_auc)
    cv = derive_counts(676, v2.oof_sens, v2.oof_spec, ho=False)
    if cv:
        n_a, n_t, tp, tn = cv
        add("C1", "fc_v2 EXP-2A LogReg-L2, out-of-fold on 676 train/val", "fc_v2_results.csv", "5-fold stratified CV, 1 seed (quick run inferred); OOF Youden threshold", 676,
            dict(accuracy=v2.oof_acc, balanced_accuracy=v2.oof_bacc, sensitivity=v2.oof_sens, specificity=v2.oof_spec, f1=v2.oof_f1), counts=(tp, n_a - tp, n_t - tn, tn),
            origin="derived from recorded sensitivity/specificity and class sizes", status=STAT_C, auc=v2.oof_auc)

# ---- 2) fc_improved_results.csv
imp = pd.read_csv(ROOT / "fc_improved_results.csv")
for i, r in imp.iterrows():
    d = derive_counts(170, r.ho_sens, r.ho_spec, ho=True)
    hc = (d[2], d[0] - d[2], d[1] - d[3], d[3]) if d else None
    add(f"H{i+2}", f"fc_improved holdout: {r.config}", "fc_improved_results.csv", HOLD, 170,
        dict(accuracy=r.ho_acc, balanced_accuracy=r.ho_bacc, sensitivity=r.ho_sens, specificity=r.ho_spec, f1=r.ho_f1), counts=hc, origin="derived from recorded sensitivity/specificity and class sizes" if hc else "not stored",
        status=STAT_H + ("; includes FIQ (diagnosis-related confound)" if "FIQ" in r.config else ""), auc=r.ho_roc_auc,
        features=("CC200 tangent-space" if "tang" in r.config else "CC200 Ledoit-Wolf correlation + Fisher-Z") + (" (19,900 edges)" if True else "") + ("" if "FC only" in r.config else " + demographics"))
    add(f"C{i+2}", f"fc_improved CV: {r.config}", "fc_improved_results.csv", "10-fold x 3 seeds on 676 train/val; seed-mean of per-seed Youden-threshold metrics", 676,
        dict(accuracy=r.cv_acc_mean, balanced_accuracy=r.cv_bacc_mean, sensitivity=r.oof_sens, specificity=r.oof_spec, f1=r.oof_f1), sd=r.cv_acc_std, status=STAT_C, auc=r.cv_auc_mean,
        features=("CC200 tangent-space" if "tang" in r.config else "CC200 Ledoit-Wolf correlation + Fisher-Z") + ("" if "FC only" in r.config else " + demographics"))

# ---- 3) fc_results.csv (846 subjects, no holdout separation)
fr = pd.read_csv(ROOT / "fc_results.csv", index_col=0)
for j, (name, r) in enumerate(fr.iterrows()):
    add(f"F{j+1}", f"train_fc.py all-846 CV: {name}", "fc_results.csv", "10-fold x 3 seeds on all 846 subjects (includes the later-reused holdout subjects; no holdout separation)", 846,
        dict(accuracy=r.accuracy_mean, sensitivity=r.sens_mean, specificity=r.spec_mean), sd=r.accuracy_std, auc=r.auc_mean,
        status="Recorded CV on the full pool (confounds-only baseline reported alongside); not a holdout result", features="CC200 " + name.split(" + ")[0] + " (no age/sex)",
        classifier=name.split(" + ", 1)[1] if " + " in name else name)

# ---- 4) leave-one-site-out pooled, from fc_full3.log
log = (ROOT / "fc_full3.log").read_text(encoding="utf-8", errors="replace")
blk = log.split("pooled across held-out sites")[1].split("====")[0]
g = lambda pat: float(re.search(pat, blk).group(1))
add("L1", "train_fc.py leave-one-site-out, best pipeline (correlation + L2), pooled over held-out sites", "fc_full3.log", "Leave-one-site-out on 17 sites with >= 21 subjects (N=820 of 846)", 820,
    dict(accuracy=g(r"accuracy\s+([0-9.]+)%") / 100, sensitivity=g(r"sensitivity\s+([0-9.]+)%") / 100, specificity=g(r"specificity\s+([0-9.]+)%") / 100, f1=g(r"F1\s+([0-9.]+)")),
    auc=g(r"ROC-AUC\s+([0-9.]+)"), status="Recorded site-aware CV on the full pool (includes later-reused holdout subjects); pooled over 17 site-specific models; 95% CI of accuracy [65.2, 71.7]",
    features="CC200 Ledoit-Wolf correlation + Fisher-Z (no age/sex)")

# ---- 5) phase 2h development-only CC200 logistic regression (stored confusion matrices)
p2h = pd.read_csv(ROOT / "results" / "phase2h_rbf_vs_logreg_summary.csv"); lg = p2h[p2h.classifier == "LogReg_L2"]
for _, r in lg[lg.seed.isin(["0", "1", "2", 0, 1, 2])].iterrows():
    tn, fp, fn, tp = np.array(ast.literal_eval(r.confusion_matrix)).ravel()
    add(f"D{int(r.seed)+1}", f"phase 2h LogReg-L2 (+age+sex), development CV, seed {int(r.seed)}", "results/phase2h_rbf_vs_logreg_summary.csv",
        "10-fold stratified CV on the 676 phase-2 training subjects; seed-level OOF Youden threshold (holdout never accessed)", 676,
        {k: float(r[c]) for k, c in dict(accuracy="oof_accuracy", balanced_accuracy="oof_balanced_accuracy", sensitivity="oof_sensitivity", specificity="oof_specificity", precision="oof_precision", f1="oof_f1").items()},
        counts=(int(tp), int(fn), int(fp), int(tn)), origin="stored confusion matrix [[TN,FP],[FN,TP]]", sd=float(r.fold_accuracy_std), status=STAT_C, auc=float(r.oof_roc_auc))
mean = lg[lg.seed == "MEAN_SD"].iloc[0]
pm = lambda s: tuple(float(x) for x in s.split("+/-"))
add("D4", "phase 2h LogReg-L2 (+age+sex), development CV, mean +/- SD over 3 seeds", "results/phase2h_rbf_vs_logreg_summary.csv", "mean over the 3 seed-level values above", 676,
    dict(accuracy=pm(mean.oof_accuracy)[0], balanced_accuracy=pm(mean.oof_balanced_accuracy)[0], sensitivity=pm(mean.oof_sensitivity)[0], specificity=pm(mean.oof_specificity)[0], f1=pm(mean.oof_f1)[0]),
    sd=pm(mean.oof_accuracy)[1], status=STAT_C, auc=pm(mean.oof_roc_auc)[0])

df = pd.DataFrame(rows)
df.round(6).to_csv(OUT / "metrics_table.csv", index=False)
fails = df[df.consistency_check.str.startswith("FAIL")]
print(df[["result_id", "description", "n", "accuracy", "balanced_accuracy", "roc_auc", "TP", "FN", "FP", "TN", "consistency_check"]].to_string(index=False, max_colwidth=58))
print("\nrows:", len(df), "| consistency FAILs:", len(fails)); assert len(fails) == 0
