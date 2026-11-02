"""Phase 6: final campaign report (reads saved results only; applies the frozen selection rule; writes final_report.md, all_experiments_comparison.csv, source_and_config_hashes.json).
No holdout or restricted file is read."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import selection_rule as SR

R = HERE / "results"; SEEDS = (0, 1, 2)
ORDER = ["P1_LR", "P2_LSVM", "P2_RBF", "P2_RF", "P2_ET", "P2_XGB", "P3_PC_LR", "P3_TS_LR", "P4_PCA_LR", "P4_KBEST_LR", "P4_L1_LR", "P5_E1_SOFTVOTE", "P5_E2_STACK"]
DESC = {"P1_LR": "Reference: LW correlation + Fisher-Z + age/sex, balanced L2 logistic regression", "P2_LSVM": "Linear SVM (balanced)", "P2_RBF": "RBF SVM (balanced)", "P2_RF": "Random forest (in-fold k=2000 filter)",
        "P2_ET": "Extra trees (in-fold k=2000 filter)", "P2_XGB": "XGBoost (in-fold k=2000 filter)", "P3_PC_LR": "Partial correlation (LW precision) + L2 LR", "P3_TS_LR": "Tangent-space (in-fold reference) + L2 LR",
        "P4_PCA_LR": "In-fold PCA (25/50/100) + L2 LR", "P4_KBEST_LR": "In-fold SelectKBest (500/2000/5000) + L2 LR", "P4_L1_LR": "In-fold L1 selection + L2 LR",
        "P5_E1_SOFTVOTE": "Soft vote of TS_LR, LSVM, LR (Platt-calibrated, training-only inner OOF)", "P5_E2_STACK": "Stacking of TS_LR, LSVM, LR (meta-LR on training-only inner OOF)"}
pc = lambda v: f"{v * 100:.2f}%"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def sd(sm, rd):
    return {int(r.seed): dict(accuracy=r.accuracy, balanced_accuracy=r.balanced_accuracy, roc_auc=r.roc_auc) for r in sm[sm.reading == rd].itertuples()}


def main():
    sms = {e: pd.read_csv(R / f"{e}_seed_metrics.csv") for e in ORDER}; fms = {e: pd.read_csv(R / f"{e}_fold_metrics.csv").sort_values(["seed", "fold"]) for e in ORDER}
    ref = sd(sms["P1_LR"], "R1_nested_threshold"); reff = fms["P1_LR"].acc_R1.values; rows = []
    for e in ORDER:
        r1 = sms[e][sms[e].reading == "R1_nested_threshold"]; r0 = sms[e][sms[e].reading == "R0_default"]
        row = dict(experiment=e, description=DESC[e], **{f"R1_{k}": r1[k].mean() for k in ("accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1")}, R1_accuracy_sd_seeds=r1.accuracy.std(ddof=1),
                   R1_fold_accuracy_sd=r1.fold_accuracy_sd.mean(), R0_accuracy=r0.accuracy.mean(), R0_balanced_accuracy=r0.balanced_accuracy.mean(), runtime_min=r1.runtime_s.sum() / 2 / 60)
        if e != "P1_LR":
            crit, ok, info = SR.screen(ref, sd(sms[e], "R1_nested_threshold"), reff, fms[e].acc_R1.values); row.update(passes_screening=ok, mean_gain_acc_pts=info["mean_gain_acc"], mean_gain_bacc_pts=info["mean_gain_bacc"],
                                                                                                                 fold_wins=info["fold_wins"], sign_test_p=info["sign_test_p"], paired_fold_se_pts=info["fold_diff_se"], suspicious=info["suspicious"])
        rows.append(row)
    C = pd.DataFrame(rows); C.to_csv(HERE / "all_experiments_comparison.csv", index=False); cand = C[C.experiment != "P1_LR"]; npass = int(cand.passes_screening.sum()); rank = C.sort_values("R1_accuracy", ascending=False).reset_index(drop=True)
    t_rank = "\n".join(f"| {i + 1} | {r.experiment} | {r.description} | {pc(r.R1_accuracy)} +/- {r.R1_accuracy_sd_seeds * 100:.2f} | {pc(r.R1_balanced_accuracy)} | {r.R1_roc_auc:.4f} | {pc(r.R1_sensitivity)} | {pc(r.R1_specificity)} | {pc(r.R1_precision)} | {pc(r.R1_f1)} | {r.R1_fold_accuracy_sd * 100:.2f} | {pc(r.R0_accuracy)} |" for i, r in rank.iterrows())
    t_scr = "\n".join(f"| {r.experiment} | {r.mean_gain_acc_pts:+.2f} | {r.mean_gain_bacc_pts:+.2f} | {r.fold_wins}/30 | {r.sign_test_p:.3f} | {r.paired_fold_se_pts:.2f} | {'PASS' if r.passes_screening else 'no'} |" for r in cand.sort_values("mean_gain_acc_pts", ascending=False).itertuples())
    # ---- site-wise (R1) for the reference and the top-ranked candidate
    top = rank[rank.experiment != "P1_LR"].iloc[0].experiment; sid_order = None; site_tabs = []
    oofs = {e: pd.read_csv(R / f"{e}_oof_predictions_seeds012.csv", dtype={"subject_id": str}) for e in ORDER}
    def site_acc(e):
        o = oofs[e]; o = o.assign(ok=(o.pred_R1 == o.y_true).astype(int)); return o.groupby(["site", "seed"]).ok.mean().groupby("site").mean()
    sa_ref, sa_top = site_acc("P1_LR"), site_acc(top); o0 = oofs["P1_LR"][oofs["P1_LR"].seed == 0]; info = o0.groupby("site").y_true.agg(["size", "mean"])
    t_site = "\n".join(f"| {s} | {int(info.loc[s, 'size'])} | {info.loc[s, 'mean'] * 100:.0f}% | {pc(sa_ref[s])} | {pc(sa_top[s])} | {(sa_top[s] - sa_ref[s]) * 100:+.1f} |" for s in info.sort_values("size", ascending=False).index)
    # ---- failure cases: how often each subject is misclassified across all 13 experiments x 3 seeds (R1)
    allp = pd.concat([oofs[e].assign(ok=lambda d: (d.pred_R1 == d.y_true).astype(int)) for e in ORDER]); subj = allp.groupby(["subject_id", "site", "y_true"]).ok.mean().reset_index(); subj["err_rate"] = 1 - subj.ok
    hard = subj[subj.err_rate >= 0.9]; easy = subj[subj.err_rate <= 0.1]
    hs = hard.groupby("site").size().sort_values(ascending=False).head(5)
    fail = (f"{len(hard)} of 676 subjects ({len(hard) / 676 * 100:.1f}%; {int((hard.y_true == 1).sum())} ASD, {int((hard.y_true == 0).sum())} TC) are misclassified in at least 90% of the 39 runs (13 experiments x 3 seeds); "
            f"{len(easy)} subjects ({len(easy) / 676 * 100:.1f}%) are correct in at least 90%. Sites contributing most consistently misclassified subjects: " + ", ".join(f"{s} ({n})" for s, n in hs.items()) +
            f". Median per-subject error rate {subj.err_rate.median() * 100:.0f}%; ASD subjects have a mean error rate of {subj[subj.y_true == 1].err_rate.mean() * 100:.1f}% against {subj[subj.y_true == 0].err_rate.mean() * 100:.1f}% for controls.")
    pd.DataFrame(subj).to_csv(HERE / "results" / "subject_error_rates_all_experiments.csv", index=False)
    # ---- hashes
    files = ["campaign.py", "selection_rule.py", "report_phase.py", "phase5.py", "final_report.py", "campaign_plan.md"] + [f"results/{e}_config.json" for e in ORDER] + ["cache/provenance.json"]
    H = {f: sha(HERE / f) for f in files}; H["plan_time_campaign_py_sha256_recorded_in_plan"] = "f1770d3dc7a2e932ba8bde9446b09513472b681de632dbe5dee4ae9a3d0d33ae"; H["campaign_py_unchanged_since_plan"] = bool(H["campaign.py"] == H["plan_time_campaign_py_sha256_recorded_in_plan"])
    H["selection_rule_unchanged_since_plan"] = bool(H["selection_rule.py"] == "e4389cbebcae96950fab28bb8f81ae5e81c7967a3d29ed82552e28730c266467"); H["baseline_and_cache_files"] = {k: sha(Path(r"C:\abide\cc200_reconstruction\experiments\stage1_2_b0_baseline") / k) for k in ("features_cc200_train676.npz", "subject_manifest.csv", "seed_metrics.csv", "fold_metrics.csv", "oof_predictions.csv", "stage1_checks.json")}
    json.dump(H, open(HERE / "source_and_config_hashes.json", "w"), indent=1)
    best = rank.iloc[0]; ref_row = C[C.experiment == "P1_LR"].iloc[0]; maxauc = C.R1_roc_auc.max()
    txt = f"""# CC200 accuracy campaign (target 85%): final report
Atlas: CC200 only. Data: 676 authorized training subjects (313 ASD / 363 TC, 19 sites); development cross-validation (10 folds x seeds 0, 1, 2; identical folds for every experiment). The restricted holdout, its IDs/labels/features/predictions and `RESTRICTED_id_membership.csv` were never opened.
Baseline, X1, X2, their reports, the cached features and hashes were not modified (hashes re-verified before every run). The selection rule (`selection_rule.py`) is unchanged since the plan: `{H['selection_rule_unchanged_since_plan']}`; `campaign.py` unchanged since the plan: `{H['campaign_py_unchanged_since_plan']}`.

## 1. Answer
**85% was NOT achieved, and nothing in this campaign supports it.** The highest mean accuracy under a valid evaluation is **{pc(best.R1_accuracy)}** ({best.experiment}); the highest ROC-AUC is {maxauc:.4f}. The reference logistic regression reaches {pc(ref_row.R1_accuracy)} (nested threshold, R1) or 66.07% (default 0.5 threshold, R0).
**No candidate passed the frozen selection rule** ({npass} of {len(cand)} candidates), so no replication on fresh seeds was triggered and no gain approached the 5-point leakage-investigation trigger (largest gain: {cand.mean_gain_acc_pts.max():+.2f} points). Twelve candidates were examined; the largest gain is within what chance would produce given paired-fold standard errors of 0.5-1.2 points.

## 2. All experiments ranked by mean R1 accuracy (the primary metric), with stability
R1 = threshold chosen on inner out-of-fold scores of the outer-training partition only; R0 = the model's default threshold. Means over 3 seeds; the +/- is the SD of the three seed-level accuracies; "fold SD" is the mean fold-to-fold SD of accuracy.
| Rank | Experiment | Description | Accuracy R1 | Balanced acc. | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Fold SD (pts) | Accuracy R0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
{t_rank}

## 3. Frozen screening against the reference (R1; thresholds: +1.5 pts accuracy and balanced accuracy, +0.5 in every seed, >= 20/30 fold wins, AUC within 0.005)
| Experiment | Mean gain acc. (pts) | Mean gain bacc. (pts) | Fold wins | Sign-test p | Paired-fold SE (pts) | Screening |
|---|---|---|---|---|---|---|
{t_scr}
Per-phase detail (per-seed metrics, confusion matrices, criteria failed): `phase2_report.md`, `phase3_report.md`, `phase4_report.md`, `phase5_report.md`; per-fold tables and hyperparameters: `results/<experiment>_fold_metrics.csv`; per-subject predictions: `results/<experiment>_oof_predictions_seeds012.csv`; configs and hashes: `results/<experiment>_config.json`, `source_and_config_hashes.json`.

## 4. What each phase showed (negative results included)
- **Phase 1 (reproduction):** the campaign pipeline reproduced the saved baseline's out-of-fold scores and inner AUCs exactly (difference 0.0). Reference: {pc(ref_row.R1_accuracy)} (R1) / 66.07% (R0), AUC {ref_row.R1_roc_auc:.4f}.
- **Phase 2 (classifiers):** linear SVM ~ logistic regression (+0.39); RBF SVM, XGBoost, extra trees and random forest were equal or worse (-0.39 to -1.68), and the tree models had clearly lower AUC (0.69-0.71). Pairs of models disagree on 7.5-28% of subjects, but none is better.
- **Phase 3 (representations):** partial correlation was clearly worse (-4.98 points, AUC 0.667): inverting 200 x 200 covariances from about 540 training rows amplifies noise. Tangent-space was +0.59 (not distinguishable from noise; AUC slightly lower).
- **Phase 4 (feature reduction):** SelectKBest -0.10, PCA -1.58, L1 selection -2.37; every reduction lost AUC, which suggests that the strongly regularized full-feature model is not leaving usable signal on the table (an inference, not a tested explanation).
- **Phase 5 (ensembles):** the complementarity condition was met (smallest error phi 0.37), so soft voting and stacking of the top-3 eligible models were built from training-only inner out-of-fold scores. Soft vote: {pc(C[C.experiment == 'P5_E1_SOFTVOTE'].R1_accuracy.iloc[0])} (+0.74, wins 17/30 folds, AUC +0.013); stacking +0.15. Neither passed. Caveat: the members were chosen as the best models on these same outer folds, so even this small gain is optimistic. The AUC gain is the one consistent signal (about +0.013 in every seed) and the only reason to look at ensembles again, but AUC was not the criterion and it did not translate into accuracy.

## 5. Site-wise performance (R1; mean of 3 seeds; reference vs top-ranked candidate {top})
| Site | n | ASD share | Reference accuracy | {top} accuracy | Diff (pts) |
|---|---|---|---|---|---|
{t_site}
Per-site values for sites with fewer than about 30 subjects are very noisy. Accuracy differs strongly by site (from about 43-46% at the weakest sites to about 84% at the strongest), consistent with the leave-one-site-out results of experiment X1. Removing site mean and scale effects from the features (X1) did not change accuracy, so these differences are not obviously due to such effects; plausible contributors are site-level sample composition and noise at small n (not tested here).

## 6. Failure cases
{fail}  (`results/subject_error_rates_all_experiments.csv`.) A sizeable block of subjects is misclassified by nearly every model and representation, which suggests that the limiting factor is the information in these connectivity features at this sample size rather than the choice of classifier (an inference from consistent failures, not a tested cause).

## 7. Best defensible CC200 model and the evidence
**The reference model:** Ledoit-Wolf covariance -> correlation -> Fisher-Z (19,900 edges) + age + sex -> standardization -> balanced L2 logistic regression, C chosen by inner 5-fold AUC (typically C = 0.001). Evidence: reproduced exactly; {pc(ref_row.R1_accuracy)}-66.07% accuracy depending on the threshold reading, balanced accuracy {pc(ref_row.R1_balanced_accuracy)}, AUC {ref_row.R1_roc_auc:.3f}; 12 alternatives (five classifiers, two other representations, three feature-reduction schemes, two ensembles) and, earlier, site harmonization and age/sex residualization all failed to beat it by the predeclared margin.
This is a development cross-validation estimate: there is no independent test, and the earlier holdout overlaps development at the subject level. No model here is validated for deployment.

## 8. Limitations and recommendations
- 676 subjects, 19 sites, 10 folds: accuracy SE about 1.8 points, paired-fold SE 0.5-1.2; differences under 1 point cannot be resolved. Multiple candidates were examined (12), which the +1.5 floor and the replication step were designed to absorb.
- The tree models used an in-fold k = 2000 filter for runtime; the tangent reference is fitted on the outer-training rows (label-free); stability selection and graphical-lasso partial correlations were not run (cost; the Ledoit-Wolf precision is already regularized).
- The route to substantially higher accuracy is not further tuning on this cohort: an independent external cohort with the same preprocessing, many more subjects per site, frame-level motion data, severity/subtype information, and reprocessing from raw images. Any final evaluation on the protected holdout needs a frozen model and your separate authorization.
"""
    (HERE / "final_report.md").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
