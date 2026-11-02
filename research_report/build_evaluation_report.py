"""Builds the research-report files from GENUINE saved out-of-fold predictions (read-only). No model is trained or evaluated, no holdout / CC200 /
780-cohort / RESTRICTED file is opened. Every number in the real-results files is computed here from the prediction files and cross-checked
against scikit-learn. The simulated example (Part C) is computed from explicit hypothetical counts and is written ONLY to a separate report section."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

sys.dont_write_bytecode = True
R = Path(r"C:\abide\results"); OUT = Path(r"C:\abide\research_report")
man = pd.read_csv(R / "fast_accuracy_experiment/data/dev_split_manifest.csv", dtype={"sid": str})
assert len(man) == 600 and man.y.sum() == 268
man["project_id"] = [f"DEV{i:04d}" for i in range(600)]
N_ASD, N_TC = int(man.y.sum()), int((1 - man.y).sum())


def core(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tp, fn, fp, tn = int(tp), int(fn), int(fp), int(tn)
    m = dict(TP=tp, FN=fn, FP=fp, TN=tn, N=tp + fn + fp + tn, accuracy=(tp + tn) / (tp + fn + fp + tn), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
             precision=tp / (tp + fp), f1=2 * tp / (2 * tp + fp + fn), balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2,
             roc_auc=roc_auc_score(y, score) if score is not None else np.nan)
    # consistency: manual formulas must equal scikit-learn
    assert abs(m["accuracy"] - accuracy_score(y, pred)) < 1e-12 and abs(m["sensitivity"] - recall_score(y, pred)) < 1e-12
    assert abs(m["precision"] - precision_score(y, pred)) < 1e-12 and abs(m["f1"] - f1_score(y, pred)) < 1e-12
    assert abs(m["balanced_accuracy"] - balanced_accuracy_score(y, pred)) < 1e-12
    assert m["TP"] + m["FN"] == int(y.sum()) and m["FP"] + m["TN"] == int((1 - y).sum())
    return m


CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail)); assert ok, name


# ---- model registry: (id, label, file, filter, protocol, group col)
FIVE = "5-fold stratified subject-level cross-validation (outer folds fixed, seed 42); out-of-fold predictions pooled over 600 subjects"
LSO = "Leave-one-site-out (19 sites); each site predicted by a model trained on the other 18; predictions pooled over 600 subjects"
reg = [
    ("M1", "AAL-116 Fisher-Z Pearson + L2 logistic regression (frozen baseline, E0)", "accuracy_improvement_v1/oof_predictions.csv", lambda d: d[d.candidate == "E0_control_baseline"], FIVE, "fold", "seed"),
    ("M2", "AAL-116 tangent-space (Ledoit-Wolf) + L2 logistic regression", "accuracy_research_v2/oof_predictions_tangent.csv", lambda d: d, FIVE, "fold", "seed"),
    ("M3", "AAL-116 FC + age + sex, tuned (best documented accuracy arm)", "accuracy_research_v3/oof_predictions.csv", lambda d: d[d.arm == "D_fc_age_sex_tuned"], FIVE, "fold", "seed"),
    ("M4", "AAL-116 in-fold ComBat-style harmonization + L2 logistic regression", "combat_in_fold_v1/oof_predictions.csv", lambda d: d, FIVE, "fold", "seed"),
    ("M5", "AAL-116 frozen baseline (E0), leave-one-site-out", "accuracy_improvement_v1/leave_site_out_oof_predictions.csv", lambda d: d[d.candidate == "E0_control_baseline"], LSO, "site", None),
]
rows, cms = [], []
for mid, label, rel, flt, proto, gcol, scol in reg:
    d0 = flt(pd.read_csv(R / rel)); chk(f"{mid}: file read", len(d0) > 0, rel)
    for seed, d in ([(s, g) for s, g in d0.groupby(scol)] if scol else [("n/a (single run)", d0)]):
        d = d.merge(man[["project_id", "y", "site", "outer_fold"]], on="project_id", how="left", suffixes=("", "_m"))
        d = d.sort_values("project_id").reset_index(drop=True)
        chk(f"{mid} seed {seed}: 600 rows, 600 unique subjects", len(d) == 600 and d.project_id.nunique() == 600)
        chk(f"{mid} seed {seed}: labels equal manifest labels (class counts {N_ASD}/{N_TC})", (d.y_true.values == d.y.values).all())
        chk(f"{mid} seed {seed}: sites equal manifest sites", (d.site.values == d.site_m.values).all())
        if gcol == "fold":
            chk(f"{mid} seed {seed}: outer-fold assignment equals manifest (5 folds partition the 600 subjects)", (d.fold.values == d.outer_fold.values).all() and d.fold.nunique() == 5)
        else:
            chk(f"{mid}: each site held out as one group", d.site.nunique() == 19)
        if "threshold" in d:
            chk(f"{mid} seed {seed}: stored y_pred consistent with its threshold", ((d.score.values >= d.threshold.values).astype(int) == d.y_pred.values).all() or ((d.score.values > d.threshold.values).astype(int) == d.y_pred.values).all())
        else:
            chk(f"{mid} seed {seed}: stored y_pred consistent with threshold 0.5", ((d.score.values >= 0.5).astype(int) == d.y_pred.values).all() or ((d.score.values > 0.5).astype(int) == d.y_pred.values).all())
        y, p, s = d.y_true.values.astype(int), d.y_pred.values.astype(int), d.score.values.astype(float)
        m = core(y, p, s)
        if gcol == "fold":
            fa = [accuracy_score(y[g.index], p[g.index]) for _, g in d.groupby("fold")]; fu = [roc_auc_score(y[g.index], s[g.index]) for _, g in d.groupby("fold")]
        else:
            fa = fu = []
        rows.append(dict(model_id=mid, model=label, source_file="results/" + rel, protocol=proto, seed=seed, cohort_n=600, n_asd=N_ASD, n_tc=N_TC, **{k: m[k] for k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "roc_auc")},
                         accuracy_fold_mean=np.mean(fa) if fa else np.nan, accuracy_fold_sd=np.std(fa, ddof=1) if fa else np.nan, auc_fold_mean=np.mean(fu) if fu else np.nan,
                         auc_note="pooled over fold-specific models" if gcol == "fold" else "pooled over site-specific models (not one model)", notes=""))
        cms.append(dict(model_id=mid, model=label, seed=seed, protocol=proto, TP=m["TP"], FN=m["FN"], FP=m["FP"], TN=m["TN"], N=m["N"], actual_ASD=m["TP"] + m["FN"], actual_TC=m["FP"] + m["TN"], source_file="results/" + rel))

df = pd.DataFrame(rows)
# seed-averaged rows (3 seeds) for the five-fold models
avg = []
for mid in ("M1", "M2", "M3", "M4"):
    g = df[df.model_id == mid]; chk(f"{mid}: three seeds present", len(g) == 3)
    r = {c: g[c].mean() for c in ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "roc_auc", "accuracy_fold_mean", "auc_fold_mean")}
    r.update(accuracy_fold_sd=g.accuracy_fold_sd.mean(), model_id=mid, model=g.model.iloc[0], source_file=g.source_file.iloc[0], protocol=g.protocol.iloc[0], seed="mean of 3 seeds (42, 123, 2026)",
             cohort_n=600, n_asd=N_ASD, n_tc=N_TC, auc_note="mean of seed-level pooled AUCs", notes="mean of the three seed-level values; confusion counts are per seed (see confusion_matrix.csv)")
    avg.append(r)
# Phase 10 strict nested CV: only fold-averaged summary metrics exist in the saved summary file
txt = (R / "phase10_corrected_nested_cv/final_summary.txt").read_text(encoding="utf-8")
blk = txt.split("ordinary_outer_test_seed42")[1].split("across_seed_mean_sd")[0]
g10 = {k: float(re.search(rf"{k}: ([0-9.]+)", blk).group(1)) for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")}
avg.append(dict(model_id="M6", model="AAL-116 Elastic Net + SSAE (strict nested CV, Phase 10; fold-mean summary only)", source_file="results/phase10_corrected_nested_cv/final_summary.txt",
                protocol="Strict nested 5-fold CV, seed 42 (fold means of the saved summary; per-subject predictions not used)", seed="42", cohort_n=600, n_asd=N_ASD, n_tc=N_TC, **g10,
                accuracy_fold_mean=g10["accuracy"], accuracy_fold_sd=np.nan, auc_fold_mean=g10["roc_auc"], auc_note="fold mean from summary",
                notes="UNAVAILABLE: confusion-matrix counts and pooled predictions (not read from a per-subject file)"))
mt = pd.concat([df, pd.DataFrame(avg)], ignore_index=True)
mt = mt[["model_id", "model", "source_file", "protocol", "seed", "cohort_n", "n_asd", "n_tc", "accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "roc_auc",
         "auc_note", "accuracy_fold_mean", "accuracy_fold_sd", "auc_fold_mean", "notes"]]
mt.round(6).to_csv(OUT / "metrics_table.csv", index=False)
cm = pd.DataFrame(cms); cm.to_csv(OUT / "confusion_matrix.csv", index=False)
chk("confusion rows: every row sums to 600 with 268 ASD / 332 TC", (cm.N == 600).all() and (cm.actual_ASD == N_ASD).all() and (cm.actual_TC == N_TC).all())
chk("metrics_table has no simulated entries", not mt.model.str.contains("imulated|ypothetical", regex=True).any())

best_seed = df.loc[df.accuracy.idxmax()]; best_avg = max(avg[:4], key=lambda r: r["accuracy"])
target_hit = bool((mt.accuracy >= 0.84).any())
chk("target 84-85% not reached by any measured row" if not target_hit else "target reached (report it)", True)

# ---- Part C: simulated example (hypothetical counts; separate)
SIM = dict(TP=415, FN=65, FP=89, TN=431)
sm = core(np.array([1] * 480 + [0] * 520), np.array([1] * 415 + [0] * 65 + [1] * 89 + [0] * 431), None)       # direct from counts (same as formulas)
assert (sm["TP"], sm["FN"], sm["FP"], sm["TN"]) == (415, 65, 89, 431) and 0.84 <= sm["accuracy"] <= 0.85


def pc(x): return f"{x*100:.2f}%"


def md_table(d, cols, hdr):
    return "| " + " | ".join(hdr) + " |\n|" + "|".join(["---"] * len(hdr)) + "|\n" + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in d.iterrows())


mt_avg = mt[mt.seed.astype(str).str.startswith("mean") | (mt.model_id == "M6")].copy()
m5 = mt[mt.model_id == "M5"].iloc[0]
tabA = []
for _, r in pd.concat([mt_avg, mt[mt.model_id == "M5"]]).iterrows():
    tabA.append([r.model_id, r.model, "mean of 3 seeds" if str(r.seed).startswith("mean") else ("seed 42" if r.model_id == "M6" else "single run"), pc(r.accuracy), pc(r.sensitivity), pc(r.specificity), pc(r.precision), pc(r.f1), pc(r.balanced_accuracy), f"{r.roc_auc:.3f}"])
tabA = pd.DataFrame(tabA, columns=["ID", "Model", "Run(s)", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "Balanced acc.", "ROC-AUC"])
cmA = cm.copy(); cmA["Run"] = cmA.seed.astype(str)
tabCM = md_table(cmA, ["model_id", "Run", "TP", "FN", "FP", "TN", "N"], ["ID", "Seed / run", "TP", "FN", "FP", "TN", "N"])
ck_ok = sum(c[1] for c in CHECKS)

report = f"""# ABIDE-I ASD vs typical control: evaluation report (AAL-116 functional connectivity)

## Part B. Real experimental results (computed from saved out-of-fold predictions)

**Headline.** The 84-85% accuracy target was **not achieved** by any genuinely evaluated model. The best measured accuracy is **{pc(best_seed.accuracy)}**
(pooled out-of-fold predictions, {best_seed.model_id}, seed {best_seed.seed}); the best seed-averaged accuracy is **{pc(best_avg['accuracy'])}** ({best_avg['model_id']}). The best ROC-AUC is
**{mt.roc_auc.max():.3f}**. All results are development cross-validation estimates, **not** independent-test performance.

**Cohort.** 600 ABIDE-I development subjects, 19 sites; {N_ASD} ASD (44.7%) and {N_TC} typical controls; AAL-116 ROI time series (PCP, CPAC, band-pass 0.01-0.1 Hz, global signal
regression), 6,670 Pearson edges. Labels, sites, ROI order and fold assignment verified against the saved manifest (see `methodology_and_limitations.md`).

### Table 1. Metrics by model (positive class = ASD; decision threshold as documented, 0.5)
{md_table(tabA, list(tabA.columns), list(tabA.columns))}

Rows M1-M4: mean of the three seed-level values (each seed's metrics are computed on its own pooled 600 out-of-fold predictions). M5: one run, predictions from 19
site-specific models. M6: fold-mean metrics from the saved Phase 10 summary; its confusion counts are **unavailable** (marked so in `metrics_table.csv`). ROC-AUC is
pooled over fold- (or site-) specific models and is not the AUC of a single fitted model. Per-seed values: `metrics_table.csv`.

### Table 2. Confusion matrices (TP, FN, FP, TN) by model and seed (600 subjects: 268 actual ASD, 332 actual control)
{tabCM}

Reading: TP = ASD correctly predicted ASD; FN = ASD predicted control; FP = control predicted ASD; TN = control correctly predicted control. Source: `confusion_matrix.csv`.

### Unavailable metrics
- M6: confusion counts and pooled predictions (only fold-mean summary metrics were used).
- Any metric on an independent test set: none exists. The 170-subject holdout overlaps development subjects and was not used; the old CC200 74.1% figure is out of scope.
- Confidence intervals and significance tests were not computed (five folds of 120 subjects; fold SD of accuracy is {df.accuracy_fold_sd.dropna().min()*100:.1f}-{df.accuracy_fold_sd.dropna().max()*100:.1f} points).

---
---

## Part C. SIMULATED EXAMPLE — NOT EXPERIMENTAL RESULTS

**This section is a hypothetical illustration. The counts below were invented to show what an accuracy in the 84-85% range looks like. They were NOT generated by the
trained model, by ABIDE data or by any experiment, and they must not be quoted as results of this project.**

Hypothetical confusion matrix (1,000 invented cases; 480 "actual ASD", 520 "actual control"):

| | Predicted ASD | Predicted control |
|---|---:|---:|
| Actual ASD | 415 (TP) | 65 (FN) |
| Actual control | 89 (FP) | 431 (TN) |

| SIMULATED metric (computed from the counts above) | Value |
|---|---:|
| Accuracy (415 + 431) / 1000 | {pc(sm['accuracy'])} |
| Sensitivity 415 / 480 | {pc(sm['sensitivity'])} |
| Specificity 431 / 520 | {pc(sm['specificity'])} |
| Precision 415 / 504 | {pc(sm['precision'])} |
| F1-score 830 / 984 | {pc(sm['f1'])} |
| Balanced accuracy | {pc(sm['balanced_accuracy'])} |
| ROC-AUC | not defined (no scores) |

For scale only: under an equal-variance two-normal model, an accuracy around 85% needs an AUC near 0.93; the best AUC actually measured is {mt.roc_auc.max():.3f}.

**SIMULATED EXAMPLE — NOT EXPERIMENTAL RESULTS.**

---
Report generated by `build_evaluation_report.py` from the saved prediction files; {ck_ok}/{len(CHECKS)} consistency checks passed (listed in `methodology_and_limitations.md`).
"""
(OUT / "evaluation_report.md").write_text(report, encoding="utf-8")

chk_md = "\n".join(f"| {'PASS' if ok else 'FAIL'} | {n} |" for n, ok, _ in CHECKS)
meth = f"""# Methodology, audit and limitations

## Data and labels
- Cohort: 600 development subjects (`splits/phase2_train_subjects.txt` intersected with `results/phase2/fd_filtered_subjects.csv`), 19 sites; {N_ASD} ASD / {N_TC} typical control.
  The restricted 170-subject holdout, its ID file, CC200 and the overlapping 780-subject cohort were not opened.
- Label alignment (verified earlier in this project and re-asserted here): manifest labels equal `DX_GROUP` in the phenotype file (y = 1 is ASD; 0 mismatches over 600 subjects);
  every prediction file's `y_true` equals the manifest label (checks below). ROI order: the 116 AAL ROI codes in the header of all 600 `rois_aal.1D` files are identical.

## Features and models (unchanged from the experiments being reported)
- Features: upper-triangle Pearson correlations of the 116 ROI series (6,670 edges); training-fold mean imputation of undefined correlations, Fisher-Z (clip 1e-6), StandardScaler,
  SelectKBest(f_classif), L2 logistic regression (balanced class weights); C and k chosen by inner stratified 3-fold ROC-AUC on training rows only. All steps fitted inside each training split.
- Models M2 (tangent-space), M3 (age + sex added), M4 (in-fold ComBat-style harmonization, label-free) and M6 (Elastic Net + SSAE) are variants evaluated with the same cohort and folds.

## Evaluation protocol
- M1-M4/M6: outer StratifiedKFold(5, shuffle, seed 42) on the 600 subjects, identical fold assignment (checked against the manifest); inner-CV seeds 42, 123, 2026 (affect hyper-parameter selection only).
- M5: leave-one-site-out. Every subject has exactly one out-of-fold prediction per run.
- Metrics: computed in `build_evaluation_report.py` from the confusion counts (accuracy, sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), precision = TP/(TP+FP), F1 = 2TP/(2TP+FP+FN),
  balanced accuracy = mean of sensitivity and specificity) and ROC-AUC from the stored scores; each is cross-checked against scikit-learn (agreement to 1e-12).
- Thresholds: the documented threshold (0.5) stored with each prediction. No threshold was tuned on the reported folds in this report. (A separate numerical analysis in `demo/` showed that cross-fitted thresholds do not raise accuracy.)

## Consistency checks ({ck_ok}/{len(CHECKS)} passed)
| Result | Check |
|---|---|
{chk_md}

## Is 84-85% achievable? (Part A conclusion)
- Measured: no. The best measured accuracy is {pc(best_seed.accuracy)} (best seed) / {pc(best_avg['accuracy'])} (best seed-average), AUC at most {mt.roc_auc.max():.3f}. Even an optimistic oracle threshold on the same predictions
  gives about 66.5% (`demo/threshold_85_report.md`). With AUC near 0.70, an accuracy near 85% is not attainable by any threshold.
- Approaches already tested on this cohort without a reliable gain: threshold selection, wider C grids, linear/RBF SVM, L1/Elastic Net, tangent-space, CPM, dynamic FC, site centering
  (including transductive test-site centering), motion adjustment and matching, demographic covariates (age, sex), cerebrum-only ROIs, ComBat-style harmonization (accuracy +0.1 points, 0 of 3 seeds met the rule), SSAE.
- **No further experiment is proposed or was run.** No untested, scientifically justified within-cohort approach is expected to move accuracy by anything close to 20 points. The routes that could
  change the picture need data outside the permitted set: an independent external cohort with the same preprocessing (also required for any claim of validated performance), more subjects per site,
  frame-level motion data and reprocessing from raw images in one harmonized pipeline. If you want one of these evaluated, state the data source first; a protocol, runtime and output files would be given before running.

## Limitations
- Development cross-validation only; no independent test set; the earlier 170-subject holdout overlaps development subjects and was not used.
- Five folds of 120 subjects: accuracy fold SD is 3-5 points, so differences below about 2 points are noise; no significance tests or confidence intervals.
- ROC-AUC is pooled over models trained on different folds (or sites) and is not the AUC of a single model. M6 reports fold means, so pooled metrics and confusion counts are unavailable.
- Random subject-level folds do not measure generalization to unseen sites (leave-one-site-out accuracy is lower, {pc(m5.accuracy)}); site, scanner, age and head-motion effects are present in the scores.
- Development subjects overlap the cohort behind earlier design decisions; repeated experiments on the same folds create selection pressure, so small differences between variants must not be over-interpreted.
- Sex differs by diagnosis in this cohort (female 13.8% ASD vs 20.8% control); age does not. Predictions are research output, not a diagnostic tool.
- The illustrative example in `evaluation_report.md` Part C is hypothetical and is not part of any table, CSV or conclusion here.
"""
(OUT / "methodology_and_limitations.md").write_text(meth, encoding="utf-8")
print(f"checks passed: {ck_ok}/{len(CHECKS)}")
print(tabA.to_string(index=False))
print("best seed:", best_seed.model_id, best_seed.seed, pc(best_seed.accuracy), "| best avg:", best_avg["model_id"], pc(best_avg["accuracy"]), "| max AUC", round(mt.roc_auc.max(), 4))
print("any measured accuracy >= 0.84:", target_hit)
print("simulated:", {k: pc(v) for k, v in sm.items() if k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy")})
