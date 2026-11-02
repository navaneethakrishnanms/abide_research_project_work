"""Phase report generator for the CC200 accuracy campaign (read-only on results; writes phase{N}_comparison.csv and phase{N}_report.md).  Usage: python report_phase.py --phase 2 --exps P1_LR,P2_LSVM,...
Applies the frozen selection rule (selection_rule.py) exactly; reference = P1_LR (the campaign's reproduction of the baseline). No holdout or restricted file is read."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import selection_rule as SR

R = HERE / "results"; SEEDS = (0, 1, 2)
pc = lambda v: f"{v * 100:.2f}%"


def load(exp):
    sm, fm = pd.read_csv(R / f"{exp}_seed_metrics.csv"), pd.read_csv(R / f"{exp}_fold_metrics.csv")
    return sm[sm.seed.isin(SEEDS)], fm[fm.seed.isin(SEEDS)].sort_values(["seed", "fold"])


def seed_dict(sm, reading):
    return {int(r.seed): dict(accuracy=r.accuracy, balanced_accuracy=r.balanced_accuracy, roc_auc=r.roc_auc) for r in sm[sm.reading == reading].itertuples()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True); ap.add_argument("--exps", required=True); a = ap.parse_args()
    exps = [e for e in a.exps.split(",")]; ref_sm, ref_fm = load("P1_LR"); ref_r1 = seed_dict(ref_sm, "R1_nested_threshold"); ref_fold = ref_fm.acc_R1.values
    rows, cmrows, seedrows, missing = [], [], [], []
    for e in exps:
        if not (R / f"{e}_seed_metrics.csv").exists():
            missing.append(e); continue
        sm, fm = load(e)
        for rd, lab in (("R1_nested_threshold", "R1"), ("R0_default", "R0")):
            s = sm[sm.reading == rd]; row = dict(experiment=e, reading=lab)
            for k in ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1"): row[k] = s[k].mean(); row[k + "_sd_seeds"] = s[k].std(ddof=1)
            row["fold_accuracy_sd"] = s.fold_accuracy_sd.mean(); row["runtime_min"] = s.runtime_s.sum() / 60 / 2 if lab == "R1" else np.nan
            if lab == "R1":
                crit, ok, info = SR.screen(ref_r1, seed_dict(sm, rd), ref_fold, fm.acc_R1.values)
                row.update(passes_screening=ok, **{k: v for k, v in info.items() if k not in ("gain_acc_by_seed",)}, gain_acc_by_seed=str([round(x, 2) for x in info["gain_acc_by_seed"]]),
                           failed_criteria="; ".join(k for k, v in crit.items() if not v) or "none")
            rows.append(row)
        for r in sm[sm.reading == "R1_nested_threshold"].itertuples():
            cmrows.append(f"| {e} | {int(r.seed)} | {int(r.TP)} | {int(r.FN)} | {int(r.FP)} | {int(r.TN)} | {pc(r.accuracy)} | {pc(r.balanced_accuracy)} | {r.roc_auc:.4f} |")
        for r in sm.itertuples():
            seedrows.append(f"| {e} | {int(r.seed)} | {r.reading[:2]} | {pc(r.accuracy)} | {pc(r.balanced_accuracy)} | {r.roc_auc:.4f} | {pc(r.sensitivity)} | {pc(r.specificity)} | {pc(r.precision)} | {pc(r.f1)} | {pc(r.fold_accuracy_sd)} |")
    C = pd.DataFrame(rows); C.to_csv(R / f"phase{a.phase}_comparison.csv", index=False)
    r1 = C[C.reading == "R1"].sort_values("accuracy", ascending=False); r0 = C[C.reading == "R0"].set_index("experiment")
    t1 = "\n".join(f"| {i + 1} | {r.experiment} | {pc(r.accuracy)} +/- {r.accuracy_sd_seeds * 100:.2f} | {pc(r.balanced_accuracy)} | {r.roc_auc:.4f} +/- {r.roc_auc_sd_seeds:.4f} | {pc(r.sensitivity)} | {pc(r.specificity)} | {pc(r.precision)} | {pc(r.f1)} | {r.fold_accuracy_sd * 100:.2f} | {pc(r0.loc[r.experiment, 'accuracy'])} | {r.runtime_min:.1f} |" for i, r in enumerate(r1.itertuples()))
    t2 = "\n".join(f"| {r.experiment} | {r.mean_gain_acc:+.2f} | {r.mean_gain_bacc:+.2f} | {r.gain_acc_by_seed} | {r.fold_wins}/{r.fold_ties}/{r.fold_losses} | {r.sign_test_p:.3f} | {r.fold_diff_se:.2f} | {r.mean_d_auc:+.4f} | {'PASS' if r.passes_screening else 'no'} | {r.failed_criteria} |" for r in r1.itertuples() if r.experiment != "P1_LR")
    best = r1.iloc[0]; passers = r1[(r1.passes_screening == True) & (r1.experiment != "P1_LR")]
    verdict = (f"Screening passers: {', '.join(passers.experiment)} (replication on seeds 3-5 required)." if len(passers) else "No candidate passed the frozen screening rule; the reference logistic regression (P1_LR) remains the best defensible model of this phase.")
    stab = "\n".join(f"- {r.experiment}: SD of the three seed-level accuracies {r.accuracy_sd_seeds * 100:.2f} pts; mean fold-to-fold SD {r.fold_accuracy_sd * 100:.2f} pts." for r in r1.itertuples())
    txt = f"""# Phase {a.phase} report: CC200 accuracy campaign (reference = P1_LR, campaign reproduction of the baseline; exact match to the saved baseline verified)
Data: 676 authorized training subjects (313 ASD / 363 TC); outer folds 10 x seeds 0, 1, 2 (identical for all rows); all numbers are development cross-validation estimates (mean over 3 seeds), not independent-test performance. Holdout untouched.
Readings: **R1** = threshold chosen on inner out-of-fold scores of the outer-training partition (primary); R0 = default threshold (0.5 probability / 0 margin). Note: the reference under R0 is the saved baseline (66.07%); under R1 it is {pc(ref_sm[ref_sm.reading == 'R1_nested_threshold'].accuracy.mean())}.
{'Not available (failed or not run): ' + ', '.join(missing) if missing else 'All requested experiments completed.'}

## 1. Ranking by mean R1 accuracy (primary), with stability
| Rank | Experiment | Accuracy R1 (mean +/- SD over seeds) | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD (pts) | Accuracy R0 | Runtime (min per 3 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|---|
{t1}
Stability:
{stab}

## 2. Paired comparison with the reference and frozen screening (R1; folds wins/ties/losses out of 30)
| Experiment | Mean gain acc. (pts) | Mean gain bacc. (pts) | Gain by seed (pts) | Fold W/T/L | Sign-test p | Paired-fold SE (pts) | Mean dAUC | Screening | Criteria failed |
|---|---|---|---|---|---|---|---|---|---|
{t2}
**Outcome:** {verdict}  (screening needs +1.5 pts accuracy and balanced accuracy, +0.5 in every seed, >= 20/30 fold wins and AUC within 0.005.) The highest mean R1 accuracy is {best.experiment} ({pc(best.accuracy)}); with the paired-fold SE shown above, differences below about 1 point are not distinguishable from noise.

## 3. Per-seed results (both readings)
| Experiment | Seed | Reading | Accuracy | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold-acc SD |
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(seedrows)}

## 4. Confusion matrices (pooled out-of-fold predictions, R1; 313 ASD / 363 TC)
| Experiment | Seed | TP | FN | FP | TN | Accuracy | Balanced acc. | AUC |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(cmrows)}
Per-fold values, selected hyperparameters, thresholds and runtimes: `results/<experiment>_fold_metrics.csv`; per-subject predictions: `results/<experiment>_oof_predictions_seeds012.csv`; configurations: `results/<experiment>_config.json`.

## 5. Protocol verification (code-level and run-time checks)
- Every learned step is inside an sklearn `Pipeline` refitted by `GridSearchCV` on each inner-training split (demographic imputation and scaling, `StandardScaler`, in-fold `SelectKBest` for the tree models, PCA, L1 selection, classifier); the outer-test fold is only transformed and scored.
- Hyperparameters: chosen by inner 5-fold ROC-AUC on the outer-training rows only. Thresholds (R1): from `cross_val_predict` on the outer-training rows only; the outer-test scores never enter their choice.
- The engine asserts disjoint train/test indices for every fold; subject IDs are unique and no two series are identical (Stage 1); the reference run reproduced the saved baseline's scores with zero difference.
- Tangent-space (Phase 3): the geometric-mean reference is fitted on the outer-training covariances (label-free); the inner CV of that pipeline sees features built with that reference (unsupervised; no labels, no outer-test rows).
- Tree models use an in-fold k = 2000 univariate filter inside the pipeline (runtime control, predeclared); selection never sees outer-test rows.
"""
    (HERE / f"phase{a.phase}_report.md").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
