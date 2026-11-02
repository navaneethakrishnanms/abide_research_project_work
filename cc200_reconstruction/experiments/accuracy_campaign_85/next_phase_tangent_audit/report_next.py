"""Consolidated report for the strictly-nested tangent-space workflow: results.csv, predictions.csv, final_report.md (reads saved checkpoints only; applies the criteria frozen in experiment_plan.md).
Stages: exploratory (seeds 0-2), confirmation (seeds 6-8, formal), supplementary (seeds 3-5, not independent for this hypothesis). Missing stages are reported as not run."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent; CAMP = HERE.parent
sys.path.insert(0, str(HERE))
import tangent_nested as TN

STAGES = {"exploratory": dict(seeds=(0, 1, 2), lr=lambda s: CAMP / "checkpoints" / f"P1_LR_seed{s}.npz", old=lambda s: CAMP / "checkpoints" / f"P3_TS_LR_seed{s}.npz", label="Stage A: exploratory (seeds 0-2; seeds used earlier for model comparison)"),
          "confirmation": dict(seeds=(6, 7, 8), lr=lambda s: HERE / "checkpoints" / f"LR_seed{s}.npz", old=None, label="Stage B: formal confirmation (seeds 6-8; never evaluated before)"),
          "supplementary": dict(seeds=(3, 4, 5), lr=lambda s: CAMP / "fresh_seed_confirmation" / "checkpoints" / f"P1_LR_seed{s}.npz", old=lambda s: CAMP / "fresh_seed_confirmation" / "checkpoints" / f"P3_TS_LR_seed{s}.npz",
                              label="Stage C: supplementary (seeds 3-5; the old tangent model was already seen there, so NOT independent for this hypothesis)")}
pc = lambda v: f"{v * 100:.2f}%"
y_all, demo_all, sid, site = TN.load_data()


def metrics(y, pred, score):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=recall_score(y, pred), specificity=tn / (tn + fp), precision=precision_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, score), TP=tp, FN=fn, FP=fp, TN=tn)


def soft_vote(a, b, fold):                                      # same construction as the saved ensemble code: Platt on training-only inner OOF scores, mean, threshold from the ensemble's inner scores
    n = len(y_all); score, thr = np.zeros(n), np.zeros(10)
    for f in range(10):
        te = fold == f; tr = ~te
        for ck in (a, b): assert np.array_equal(~np.isnan(ck["inner_oof"][f]), tr)
        Zin = np.column_stack([ck["inner_oof"][f][tr] for ck in (a, b)]); Zte = np.column_stack([ck["score"][te] for ck in (a, b)]); cal = [LogisticRegression(C=1e6).fit(Zin[:, j:j + 1], y_all[tr]) for j in range(2)]
        sin = np.mean([c.predict_proba(Zin[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0); score[te] = np.mean([c.predict_proba(Zte[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0); thr[f] = TN.choose_threshold(y_all[tr], sin, 0.5)
    return dict(score=score, thr=thr, default=np.full(10, 0.5), fold=fold)


def arms_for(stage, s):
    cfg = STAGES[stage]; p = HERE / "checkpoints" / f"{stage}_nested_seed{s}.npz"
    if not p.exists(): return None
    ts = dict(np.load(p)); lr = dict(np.load(cfg["lr"](s))); assert (ts["fold"] == lr["fold"]).all(); arms = {"LR (reference)": lr, "TS-N (strictly nested tangent LR)": ts}
    if cfg["old"] is not None: arms["TS-old (reference fitted on all outer-training rows)"] = dict(np.load(cfg["old"](s)))
    arms["H3: soft vote LR + TS-N"] = soft_vote(lr, ts, lr["fold"]); return arms


def preds(d, rd):
    return (d["score"] >= (d["thr"][d["fold"]] if rd == "R1" else 0.5)).astype(int)


def main():
    rows, prow, stage_res, ran = [], [], {}, []
    for stage, cfg in STAGES.items():
        if not all((HERE / "checkpoints" / f"{stage}_nested_seed{s}.npz").exists() for s in cfg["seeds"]): continue
        ran.append(stage); per = {s: arms_for(stage, s) for s in cfg["seeds"]}; res = {}
        for s in cfg["seeds"]:
            for arm, d in per[s].items():
                for rd in ("R1", "R0"):
                    m = metrics(y_all, preds(d, rd), d["score"]); fa = [accuracy_score(y_all[d["fold"] == f], preds(d, rd)[d["fold"] == f]) for f in range(10)]
                    rows.append(dict(stage=stage, seed=s, arm=arm, reading=rd, **m, fold_accuracy_mean=np.mean(fa), fold_accuracy_sd=np.std(fa, ddof=1))); res[(s, arm, rd)] = (m, np.array(fa))
            ref = per[s]["LR (reference)"]
            for arm, d in per[s].items():
                prow.append(pd.DataFrame(dict(stage=stage, seed=s, arm=arm, subject_id=sid, site=site, y_true=y_all, fold=d["fold"], score=d["score"], pred_R1=preds(d, "R1"), pred_R0=preds(d, "R0"))))
        stage_res[stage] = res
    R = pd.DataFrame(rows); R.to_csv(HERE / "results.csv", index=False); pd.concat(prow).to_csv(HERE / "predictions.csv", index=False)

    def gain(stage, arm, base, rd="R1", key="accuracy"):
        seeds = STAGES[stage]["seeds"]; return np.array([stage_res[stage][(s, arm, rd)][0][key] - stage_res[stage][(s, base, rd)][0][key] for s in seeds])

    def fold_diff(stage, arm, base, rd="R1"):
        return np.concatenate([stage_res[stage][(s, arm, rd)][1] - stage_res[stage][(s, base, rd)][1] for s in STAGES[stage]["seeds"]])

    TS, LR, OLD, H3 = "TS-N (strictly nested tangent LR)", "LR (reference)", "TS-old (reference fitted on all outer-training rows)", "H3: soft vote LR + TS-N"
    sections, summary_lines = [], []
    for stage in ran:
        cfg = STAGES[stage]; seeds = cfg["seeds"]; arms = [a for a in (LR, OLD, TS, H3) if (seeds[0], a, "R1") in stage_res[stage]]
        def mtab(rd):
            return "\n".join(f"| {a.split(' (')[0].split(':')[0] if a != H3 else 'H3 soft vote LR+TS-N'} | {s} | " + " | ".join(pc(stage_res[stage][(s, a, rd)][0][k]) for k in ("accuracy", "balanced_accuracy")) + f" | {stage_res[stage][(s, a, rd)][0]['roc_auc']:.4f} | " +
                             " | ".join(pc(stage_res[stage][(s, a, rd)][0][k]) for k in ("sensitivity", "specificity", "precision", "f1")) + f" | {pc(stage_res[stage][(s, a, rd)][1].std(ddof=1))} |" for a in arms for s in seeds)
        def cmtab(rd):
            return "\n".join(f"| {a.split(' (')[0] if a != H3 else 'H3'} | {s} | {stage_res[stage][(s, a, rd)][0]['TP']} | {stage_res[stage][(s, a, rd)][0]['FN']} | {stage_res[stage][(s, a, rd)][0]['FP']} | {stage_res[stage][(s, a, rd)][0]['TN']} |" for a in arms for s in seeds)
        comps = [(TS, LR, "H1: TS-N minus LR")] + ([(TS, OLD, "Correction effect: TS-N minus TS-old")] if OLD in arms else []) + [(H3, LR, "H3: soft vote minus LR"), (H3, TS, "H3 minus TS-N (informational)")]
        ctab = []
        for a, b, name in comps:
            ga, gb, gu = gain(stage, a, b) * 100, gain(stage, a, b, key="balanced_accuracy") * 100, gain(stage, a, b, key="roc_auc"); fd = fold_diff(stage, a, b) * 100; w, t, l = int((fd > 0).sum()), int((fd == 0).sum()), int((fd < 0).sum())
            r0 = gain(stage, a, b, "R0") * 100
            ctab.append(f"| {name} | {', '.join(f'{v:+.2f}' for v in ga)} | {ga.mean():+.2f} (SD {ga.std(ddof=1):.2f}) | {gb.mean():+.2f} | {gu.mean():+.4f} | {w}/{t}/{l} | {fd.mean():+.2f} (SE {fd.std(ddof=1) / np.sqrt(len(fd)):.2f}) | {r0.mean():+.2f} |")
        g1, u1 = gain(stage, TS, LR) * 100, gain(stage, TS, LR, key="roc_auc"); g3, u3 = gain(stage, H3, LR) * 100, gain(stage, H3, LR, key="roc_auc")
        h1 = (g1.mean() >= 0.5, int((g1 > 0).sum()) >= 2, u1.mean() >= -0.005); h3 = (g3.mean() >= 0.5, int((g3 > 0).sum()) >= 2, u3.mean() >= 0.01)
        decisive = stage == "confirmation"
        crit = (f"| H1 (TS-N vs LR) | mean accuracy gain >= +0.5 pt | {g1.mean():+.2f} | {'yes' if h1[0] else 'NO'} |\n| | positive in >= 2 of 3 seeds | {int((g1 > 0).sum())} of 3 | {'yes' if h1[1] else 'NO'} |\n| | mean AUC change >= -0.005 | {u1.mean():+.4f} | {'yes' if h1[2] else 'NO'} |\n"
                f"| H3 (soft vote vs LR) | mean accuracy gain >= +0.5 pt | {g3.mean():+.2f} | {'yes' if h3[0] else 'NO'} |\n| | positive in >= 2 of 3 seeds | {int((g3 > 0).sum())} of 3 | {'yes' if h3[1] else 'NO'} |\n| | mean AUC gain >= +0.01 | {u3.mean():+.4f} | {'yes' if h3[2] else 'NO'} |")
        verdict = f"H1: **{'PASS' if all(h1) else 'FAIL'}**; H3: **{'PASS' if all(h3) else 'FAIL'}**" + ("" if decisive else " (descriptive only: this stage is not the pre-registered decision stage)")
        diag = []
        for s in seeds:
            d = np.load(HERE / "checkpoints" / f"{stage}_nested_seed{s}.npz"); diag.append((float(np.mean(d["ref_dist"])), float(np.mean(d["feat_rel_change"])), float(np.sum(d["secs"]))))
        diag = np.array(diag); summary_lines.append((stage, h1, h3, g1, g3, u1, u3))
        sections.append(f"""## {cfg['label']}
### Accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, precision, F1 (R1 = threshold from training-only inner scores; primary)
| Arm | Seed | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|
{mtab('R1')}
Secondary (R0, default thresholds): see `results.csv`; paired R0 accuracy gains are in the last column below.
### Confusion matrices (R1; pooled out-of-fold, 313 ASD / 363 TC)
| Arm | Seed | TP | FN | FP | TN |
|---|---|---|---|---|---|
{cmtab('R1')}
### Paired comparisons (R1; gains in percentage points; fold W/T/L out of 30)
| Comparison | Accuracy gain by seed | Mean accuracy gain | Mean balanced-acc. gain | Mean AUC gain | Fold W/T/L | Mean fold diff (SE) | R0 mean acc. gain |
|---|---|---|---|---|---|---|---|
{chr(10).join(ctab)}
### Frozen criteria evaluated on this stage
| Hypothesis | Condition | Observed | Met |
|---|---|---|---|
{crit}
{verdict}
Strict-nesting diagnostics (means over folds): affine-invariant distance between an inner reference and the outer reference {diag[:, 0].mean():.3f}; relative change of the validation features {diag[:, 1].mean() * 100:.1f}%; CPU time for the TS-N fits {diag[:, 2].sum() / 60:.0f} min.
""")
    # ---- integrity
    inv_now = TN.inventory(); before = json.load(open(HERE / "hashes_before_exploratory.json")) if (HERE / "hashes_before_exploratory.json").exists() else None
    same = (inv_now == before) if before else None
    diffs = [k for k in (before or {}) if before.get(k) != inv_now.get(k)] + [k for k in inv_now if before and k not in before]
    # ---- consolidated conclusions (absolute levels, descriptive pooling over all stages, decisions as frozen)
    def lvl(stage, arm, key, rd="R1"):
        return float(np.mean([stage_res[stage][(s, arm, rd)][0][key] for s in STAGES[stage]["seeds"]]))
    lvl_rows = "\n".join(f"| {STAGES[st]['label'].split(':')[0]} | {a.split(' (')[0] if a != H3 else 'H3 soft vote LR+TS-N'} | {pc(lvl(st, a, 'accuracy'))} | {pc(lvl(st, a, 'balanced_accuracy'))} | {lvl(st, a, 'roc_auc'):.4f} | {pc(lvl(st, a, 'sensitivity'))} | {pc(lvl(st, a, 'specificity'))} | {pc(lvl(st, a, 'accuracy', 'R0'))} |"
                         for st in ran for a in (LR, OLD, TS, H3) if (STAGES[st]["seeds"][0], a, "R1") in stage_res[st])
    allg1 = np.concatenate([gain(st, TS, LR) * 100 for st in ran]); allg3 = np.concatenate([gain(st, H3, LR) * 100 for st in ran]); allu1 = np.concatenate([gain(st, TS, LR, key="roc_auc") for st in ran]); allu3 = np.concatenate([gain(st, H3, LR, key="roc_auc") for st in ran])
    conf = "confirmation" in ran; cs = next((x for x in summary_lines if x[0] == "confirmation"), None)
    if cs:
        _, h1c, h3c, g1c, g3c, u1c, u3c = cs; h1p, h3p = all(h1c), all(h3c)
    concl = f"""## 2. Conclusions
### Absolute levels (R1 primary; mean over each stage's three seeds)
| Stage | Arm | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Accuracy R0 |
|---|---|---|---|---|---|---|---|
{lvl_rows}
### Formal decisions (frozen criteria, confirmation seeds 6-8 only)
{('- **H1, strictly nested tangent-space LR vs reference LR: ' + ('PASS' if h1p else 'FAIL') + '.** Mean accuracy gain %+.2f pt (needs >= +0.5): %s; positive in %d of 3 seeds (needs 2): %s; mean AUC change %+.4f (needs >= -0.005): %s.' % (g1c.mean(), 'met' if h1c[0] else 'not met', int((g1c > 0).sum()), 'met' if h1c[1] else 'not met', u1c.mean(), 'met' if h1c[2] else 'not met')) if cs else '- Stage B not run.'}
{('- **H3, soft vote of LR and strictly nested tangent LR vs reference LR: ' + ('PASS' if h3p else 'FAIL') + '.** Mean accuracy gain %+.2f pt (needs >= +0.5): %s; positive in %d of 3 seeds (needs 2): %s; mean AUC gain %+.4f (needs >= +0.01): %s.' % (g3c.mean(), 'met' if h3c[0] else 'not met', int((g3c > 0).sum()), 'met' if h3c[1] else 'not met', u3c.mean(), 'met' if h3c[2] else 'not met')) if cs else ''}
- **Effect of the nesting correction itself:** TS-N minus the old non-nested tangent model: {', '.join(f'{gain(st, TS, OLD).mean() * 100:+.2f} pt ({STAGES[st]["label"].split(":")[0]})' for st in ran if OLD in [a for a in (LR, OLD, TS, H3) if (STAGES[st]["seeds"][0], a, 'R1') in stage_res[st]])} (most folds identical; relative feature change about 2%). The correction removed a methodological doubt; it did not change the results materially.
### Descriptive pooling over all nine seeds (A+B+C; NOT a decision: stages A and C were seen before or are not independent for H1, and every seed uses the same 676 subjects)
- TS-N minus LR: mean accuracy gain {allg1.mean():+.2f} pt, positive in {int((allg1 > 0).sum())} of {len(allg1)} seeds (zero in {int((allg1 == 0).sum())}), mean AUC change {allu1.mean():+.4f}.
- H3 soft vote minus LR: mean accuracy gain {allg3.mean():+.2f} pt, positive in {int((allg3 > 0).sum())} of {len(allg3)} seeds, mean AUC gain {allu3.mean():+.4f}.
### Answers
1. **Does the corrected tangent-space LR improve on the baseline?** On average it is about {allg1.mean():+.1f} pt more accurate than the reference LR over nine seeds, but the gain is not consistent: it failed its pre-registered confirmation (positive in only {int((g1c > 0).sum()) if cs else '?'} of 3 confirmation seeds and AUC slightly lower), and its gain by seed ranges from {allg1.min():+.2f} to {allg1.max():+.2f} pt. It is not established as an improvement.
2. **Consistent across folds and seeds?** No for the tangent model alone (paired-fold W/T/L close to even, SE about 0.7-1.1 pt in every stage). The two-model soft vote is more consistent (gain positive in {int((allg3 > 0).sum())} of {len(allg3)} seeds, AUC +{allu3.mean():.3f}) and passed the formal confirmation, but its mean accuracy gain is about {allg3.mean():+.1f} pt, and its paired-fold SE is 0.6-0.8 pt.
3. **Best-performing model:** the soft vote of the reference LR and the strictly nested tangent LR (H3): the only candidate that passed a formal confirmation in this workflow, with R1 accuracy {pc(lvl('confirmation', H3, 'accuracy')) if conf else 'n/a'} and AUC {lvl('confirmation', H3, 'roc_auc') if conf else float('nan'):.3f} on the confirmation seeds versus {pc(lvl('confirmation', LR, 'accuracy')) if conf else 'n/a'} and {lvl('confirmation', LR, 'roc_auc') if conf else float('nan'):.3f} for the reference LR. This is a small gain, not the 85% target.
4. **Is further experimentation on this cohort justified?** Not for accuracy: after nine seeds, the nesting correction with its validation, and a pre-registered confirmation, the ceiling of the connectivity features on these 676 subjects is about 66-68% accuracy (AUC about 0.72-0.74), and the effects being discussed are about one point. What remains valuable is data, not modelling: an independent external cohort with the same preprocessing, more subjects per site, frame-level motion data and harmonized reprocessing; a final evaluation on the protected holdout would need a frozen model and your separate authorization.
"""
    stage_status = "\n".join(f"- {cfg['label']}: {'RUN' if st in ran else 'NOT RUN'}" for st, cfg in STAGES.items())
    best = R[(R.reading == "R1") & (R.stage.isin(ran))].groupby(["stage", "arm"])[["accuracy", "balanced_accuracy", "roc_auc"]].mean().round(4)
    txt = f"""# Strictly nested tangent-space workflow: final report
CC200 only; 676 authorized training subjects (313 ASD / 363 TC); development cross-validation (10 folds x 3 seeds per stage); the protected holdout was **not accessed**. 85% is the project's target and was **not achieved**; the best accuracies below are in the mid-60s to about 67%.
Plan and frozen criteria: `experiment_plan.md`; method and validity evidence: `methodology.md`; numbers: `results.csv`, `predictions.csv`.

## 0. Stage status
{stage_status}

## 1. Priority 1: the tangent-space reference was corrected and the correction was validated
- The old pipeline fitted the tangent reference once on all outer-training rows, so inner-validation subjects had contributed to the reference that transformed them. The corrected evaluator refits the reference inside every inner training split and never on validation or test rows.
- Validation: with the reference scope switched back to the old behaviour, the new evaluator reproduces the saved `P3_TS_LR` results **exactly** (zero difference in scores, selected C, thresholds and inner out-of-fold scores on two real folds); 5/5 synthetic tests show that perturbing inner-validation or outer-test rows changes no fitted reference or any other row's output, and that no outer-test labels reach the evaluator.
- Effect size (see diagnostics under each stage): the strictly inner references differ only slightly from the outer one, so the correction is expected to change results little; the stage tables measure that directly (TS-N minus TS-old).

{chr(10).join(sections)}

{concl}
## 3. Integrity
- Hash inventory (baseline files, covariance cache, campaign code, frozen selection rule, final/audit/confirmation reports, all existing campaign checkpoints and results, confirmation outputs): identical before the first corrected run and now: **{same}**{'' if same else '; differing entries: ' + ', '.join(diffs[:10])}. Per-stage inventories: `hashes_before_*.json`, `hashes_after_*.json`.
- Subject-level outer folds identical for all arms within a seed (asserted); all fits use only the outer-training rows (or inner splits of them); holdout, `RESTRICTED_id_membership.csv` and the combined arrays were not opened.

## 4. Limitations
Result types: Stage A is exploratory, Stage B is the pre-registered confirmation, Stage C is supplementary and not independent for H1; **no independent external validation exists**. All stages use the same 676 subjects (new partitions are not new subjects); folds are not independent; accuracy SE is about 1.8 points; exploratory results decide nothing; no independent external validation was possible or performed. Nothing here supports the 85% target.
"""
    (HERE / "final_report.md").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
