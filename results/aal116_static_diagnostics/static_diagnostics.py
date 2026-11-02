"""Read-only diagnostics of the static AAL-116 results on the 600 development subjects (saved OOF predictions + manifests).
No classifier is trained. One descriptive filter computation (univariate F-score top-k per outer fold, TRAINING rows only)
is used for feature-stability description because selected feature indices were never saved; it selects/tunes nothing.
Holdout data, CC200 and the restricted ID file are not touched."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, r"C:\abide")
R = Path(r"C:\abide\results")
FX = R / "fast_accuracy_experiment"
OUT = R / "aal116_static_diagnostics"
OUT.mkdir(exist_ok=True)
md = []
AAL_MODELS = ["LR_L2", "LinearSVM", "L1_LR", "RBF_SVM"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


man = pd.read_csv(FX / "data" / "dev_split_manifest.csv", dtype={"sid": str})
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str})
coh["sid"] = coh.subject_id.map(lambda s: str(int(float(s))))
coh = coh[coh.sid.isin(set(man.sid))]
man = man.merge(coh[["sid", "age", "sex", "func_mean_fd"]], on="sid")
oof = pd.read_csv(FX / "oof_predictions.csv", dtype={"subject_id": str})
oof = oof[(oof.atlas == "AAL116") & oof.model.isin(AAL_MODELS)].merge(man[["sid", "age", "sex", "func_mean_fd"]], left_on="subject_id", right_on="sid")
oof["err"] = (oof.y_true != oof.y_pred).astype(int)
fr = pd.read_csv(FX / "fold_results.csv"); fr = fr[(fr.atlas == "AAL116") & fr.model.isin(AAL_MODELS)]

md.append("# Static AAL-116 diagnostics (600 development subjects; saved predictions; no model trained)\n")
md.append("All associations below are descriptive; none establishes a cause.\n")

# ---- 1. error consistency
md.append("\n## 1. Error consistency across the four AAL-116 models\n")
subj = oof.groupby("subject_id").agg(n_wrong=("err", "sum"), y=("y_true", "first"), site=("site", "first"), fd=("func_mean_fd", "first"), age=("age", "first"), sex=("sex", "first"))
p = oof.err.mean()
exp = {k: len(subj) * stats.binom.pmf(k, 4, p) for k in range(5)}
dist = subj.n_wrong.value_counts().sort_index()
md.append(f"Mean error rate {p:.3f}. Subjects wrong in 0/1/2/3/4 of 4 models: {dist.reindex(range(5), fill_value=0).tolist()} "
          f"(expected if errors were independent: {[round(exp[k], 1) for k in range(5)]}). Wrong in all 4: **{int((subj.n_wrong == 4).sum())}** "
          f"(ASD share {subj[subj.n_wrong == 4].y.mean():.2f}); right in all 4: {int((subj.n_wrong == 0).sum())} (ASD share {subj[subj.n_wrong == 0].y.mean():.2f}; overall {subj.y.mean():.2f}).\n")
for c in (1, 0):
    s = subj[subj.y == c]
    md.append(f"- {'ASD' if c else 'TC'}: n={len(s)}; wrong in all 4 models: {int((s.n_wrong == 4).sum())} ({100 * (s.n_wrong == 4).mean():.0f}%); right in all 4: {int((s.n_wrong == 0).sum())} ({100 * (s.n_wrong == 0).mean():.0f}%)\n")

# ---- 2. site
md.append("\n## 2. Performance by site (small groups flagged)\n")
rows = []
for label, g in [("LR_L2", oof[oof.model == "LR_L2"]), ("mean_of_4_AAL_models", oof)]:
    for site, gs in g.groupby("site"):
        n = gs.subject_id.nunique(); acc = 1 - gs.err.mean(); lo, hi = wilson(round(acc * len(gs)), len(gs))
        a, t = gs[gs.y_true == 1], gs[gs.y_true == 0]
        rows.append({"model": label, "site": site, "n_subjects": n, "n_asd": a.subject_id.nunique(), "n_tc": t.subject_id.nunique(), "accuracy": acc, "acc_ci95_lo": lo, "acc_ci95_hi": hi,
                     "sensitivity": 1 - a.err.mean() if len(a) else np.nan, "specificity": 1 - t.err.mean() if len(t) else np.nan, "small_group_n_lt_20": n < 20})
SE = pd.DataFrame(rows); SE.to_csv(OUT / "site_error_summary.csv", index=False)
v = SE[SE.model == "mean_of_4_AAL_models"].sort_values("n_subjects", ascending=False)
md.append("```\n" + v[["site", "n_subjects", "accuracy", "acc_ci95_lo", "acc_ci95_hi", "sensitivity", "specificity", "small_group_n_lt_20"]].round(3).to_string(index=False) + "\n```\n")
big = v[v.n_subjects >= 20]
es = oof[oof.site.isin(big.site)].groupby("site").agg(e=("err", "sum"), n=("err", "size"))
het = stats.chi2_contingency(np.c_[es.e, es.n - es.e])
md.append(f"Sites with n>=20 (k={len(big)}): accuracy range {big.accuracy.min():.2f}-{big.accuracy.max():.2f}; heterogeneity of error rate across them chi2={het[0]:.1f}, dof={het[2]}, p={het[1]:.3g} (the four models are not independent, so p is anti-conservative; individual-site estimates are wide, e.g. CIs about +/-0.10-0.17).\n")

# ---- 3. motion
md.append("\n## 3. Association of scores and errors with mean FD\n")
mrows = []
for m, g in oof.groupby("model"):
    r_all = stats.spearmanr(g.score, g.func_mean_fd); r_a = stats.spearmanr(g[g.y_true == 1].score, g[g.y_true == 1].func_mean_fd); r_t = stats.spearmanr(g[g.y_true == 0].score, g[g.y_true == 0].func_mean_fd)
    fd_err = g[g.err == 1].func_mean_fd; fd_ok = g[g.err == 0].func_mean_fd
    terc = pd.qcut(g.func_mean_fd, 3, labels=["low FD", "mid FD", "high FD"])
    acc_t = (1 - g.groupby(terc).err.mean())
    pred_asd_t = g.groupby(terc).y_pred.mean()
    mrows.append({"model": m, "spearman_score_vs_FD_all": r_all.correlation, "p_all": r_all.pvalue, "spearman_within_ASD": r_a.correlation, "spearman_within_TC": r_t.correlation,
                  "mean_FD_misclassified": fd_err.mean(), "mean_FD_correct": fd_ok.mean(), "mannwhitney_p_FD_err_vs_correct": stats.mannwhitneyu(fd_err, fd_ok).pvalue,
                  "accuracy_low_FD": acc_t["low FD"], "accuracy_mid_FD": acc_t["mid FD"], "accuracy_high_FD": acc_t["high FD"],
                  "share_predicted_ASD_low_FD": pred_asd_t["low FD"], "share_predicted_ASD_high_FD": pred_asd_t["high FD"]})
ME = pd.DataFrame(mrows); ME.to_csv(OUT / "motion_error_summary.csv", index=False)
md.append("```\n" + ME.round(3).T.to_string(header=False) + "\n```\n")
a_fd = man.loc[man.y == 1, "func_mean_fd"]; t_fd = man.loc[man.y == 0, "func_mean_fd"]
auc_fd = stats.mannwhitneyu(a_fd, t_fd).statistic / (len(a_fd) * len(t_fd))
md.append(f"Mean FD: ASD {a_fd.mean():.3f}, TC {t_fd.mean():.3f}; AUC-equivalent of FD alone for the diagnosis labels = {auc_fd:.3f} (descriptive rank statistic, all 600 subjects). "
          "Predicted-ASD share is higher in the high-FD tercile for the models above, i.e. predictions track motion; whether the models use motion or share a correlated signal cannot be told from these data.\n")

# ---- 4. demographics
md.append("\n## 4. Demographic balance (development subjects)\n")
sx = pd.crosstab(man.y, man.sex); chi = stats.chi2_contingency(sx)
md.append(f"Sex (1 male, 2 female) by diagnosis (rows TC, ASD): {sx.values.tolist()}; female share TC {sx.loc[0, 2] / sx.loc[0].sum():.3f}, ASD {sx.loc[1, 2] / sx.loc[1].sum():.3f}; chi2 p={chi[1]:.3g}. "
          f"Age: ASD {man[man.y == 1].age.mean():.1f}, TC {man[man.y == 0].age.mean():.1f} (Mann-Whitney p={stats.mannwhitneyu(man[man.y == 1].age, man[man.y == 0].age).pvalue:.3g}).\n")
g = oof[oof.model == "LR_L2"].assign(sexl=lambda d: d.sex.map({1: "male", 2: "female"}))
tb = g.groupby(["sexl", "y_true"]).agg(n=("err", "size"), accuracy=("err", lambda s: 1 - s.mean())).round(3)
md.append("LR_L2 accuracy by sex and diagnosis:\n```\n" + tb.to_string() + "\n```\n")

# ---- 5. calibration and thresholds (LR_L2 probabilities)
md.append("\n## 5. Calibration and class behavior (LR_L2 out-of-fold probabilities)\n")
g = oof[oof.model == "LR_L2"]
bins = pd.cut(g.score, [0, .3, .4, .5, .6, .7, 1.0])
cal = g.groupby(bins).agg(n=("y_true", "size"), mean_pred=("score", "mean"), observed_ASD_rate=("y_true", "mean")).round(3)
brier = float(((g.score - g.y_true) ** 2).mean()); prior = g.y_true.mean(); brier_prior = float(((prior - g.y_true) ** 2).mean())
md.append(f"Brier score {brier:.4f} (always predicting the prevalence {prior:.3f}: {brier_prior:.4f}). Mean predicted ASD probability: ASD {g[g.y_true == 1].score.mean():.3f}, TC {g[g.y_true == 0].score.mean():.3f}.\n```\n" + cal.to_string() + "\n```\n")
trow = []
for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    pr = (g.score >= t).astype(int); tp = ((pr == 1) & (g.y_true == 1)).sum(); fn = ((pr == 0) & (g.y_true == 1)).sum(); tn = ((pr == 0) & (g.y_true == 0)).sum(); fp = ((pr == 1) & (g.y_true == 0)).sum()
    trow.append({"threshold": t, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "accuracy": (tp + tn) / len(g), "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2})
md.append("Threshold behavior of the saved scores (descriptive; not for choosing a threshold, which would need inner CV):\n```\n" + pd.DataFrame(trow).round(3).to_string(index=False) + "\n```\n")
md.append("Per-fold AUC (all four models): " + "; ".join(f"{m}: " + " ".join(f"{x:.3f}" for x in fr[fr.model == m].sort_values('fold').roc_auc) for m in AAL_MODELS) + "\n")

# ---- 6. data quality
md.append("\n## 6. Data quality (AAL-116, development subjects)\n")
X = np.load(FX / "data" / "fc_raw_AAL116.npy")
nn = np.isnan(X).sum(1)
sdv = man.assign(n_nan=nn)
md.append(f"Matrix {X.shape}; undefined (NaN) correlation entries {int(np.isnan(X).sum())} ({100 * np.isnan(X).mean():.2f}%); subjects with any undefined entries {int((nn > 0).sum())}; "
          f"features with any NaN {int(np.isnan(X).any(0).sum())} of {X.shape[1]}; max per subject {int(nn.max())}. Undefined entries arise from constant (zero) ROIs (Phase 4.1). Imputation is the fold-scoped training-column mean inside the Pipeline (verified in the script).\n")
per_site = sdv.groupby("site").agg(n=("n_nan", "size"), subjects_with_nan=("n_nan", lambda s: int((s > 0).sum()))).query("subjects_with_nan>0").sort_values("subjects_with_nan", ascending=False)
md.append("Subjects with undefined entries by site:\n```\n" + per_site.to_string() + "\n```\n")
g4 = subj.join(sdv.set_index("sid").n_nan)
has = g4.n_nan > 0
md.append(f"Mean number of the 4 models wrong: subjects with undefined entries {g4[has].n_wrong.mean():.2f} (n={int(has.sum())}) vs without {g4[~has].n_wrong.mean():.2f} (n={int((~has).sum())}); Mann-Whitney p={stats.mannwhitneyu(g4[has].n_wrong, g4[~has].n_wrong).pvalue:.3g}.\n")

# ---- 7. feature stability (new descriptive filter computation, training rows only)
md.append("\n## 7. Feature-selection stability\nSelected-feature indices were not saved for the 600-subject runs, so existing artifacts only give the chosen k per fold. "
          "Descriptive substitute: in each outer fold, on TRAINING rows only (fold-scoped mean imputation, Fisher-Z, standardization), rank the 6,670 features by univariate F-score and compare the top-k sets across folds. This is not a model and nothing is selected or tuned from it.\n")
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import fast_accuracy_experiment as fx
from sklearn.model_selection import StratifiedKFold
y = man.y.values
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(600), y))
assert all(set(np.where(man.outer_fold.values == k)[0]) == set(te) for k, (_, te) in enumerate(folds))
tops = {500: [], 2000: []}
for tr, te in folds:
    t = fx.MeanImputeFisherZ().fit(X[tr]); Z = StandardScaler().fit_transform(t.transform(X[tr]))
    F = np.nan_to_num(f_classif(Z, y[tr])[0]); order = np.argsort(-F)
    for k in tops:
        tops[k].append(set(order[:k].tolist()))
srows = []
p_feat = X.shape[1]
for k, sets in tops.items():
    jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    cnt = np.zeros(p_feat, int)
    for s in sets:
        cnt[list(s)] += 1
    exp_j = (k * k / p_feat) / (2 * k - k * k / p_feat)
    srows.append({"top_k": k, "mean_pairwise_jaccard_across_folds": float(np.mean(jac)), "min_jaccard": float(np.min(jac)), "max_jaccard": float(np.max(jac)),
                  "jaccard_expected_if_random_sets": float(exp_j), "features_in_top_k_of_all_5_folds": int((cnt == 5).sum()), "features_in_top_k_of_>=4_folds": int((cnt >= 4).sum()),
                  "features_in_top_k_of_>=1_fold": int((cnt >= 1).sum()), "note": "univariate F filter, training rows only; training sets of any two folds overlap by 75% by construction"})
FS = pd.DataFrame(srows); FS.to_csv(OUT / "feature_stability_summary.csv", index=False)
md.append("```\n" + FS.drop(columns="note").round(3).to_string(index=False) + "\n```\n")
md.append("Historical context (Phase 6, 780-subject cohort that overlaps the designated holdout; Elastic Net, all five folds): 1,286 features selected at least once, only 50 in all five folds.\n")
(OUT / "diagnostic_report.md").write_text("".join(md), encoding="utf-8")
print("".join(md))
