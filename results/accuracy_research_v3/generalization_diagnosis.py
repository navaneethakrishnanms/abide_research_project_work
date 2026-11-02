"""Experiment 2 (descriptive, saved predictions only): which variables are associated with errors of the baseline and tangent OOF predictions (seed 42),
and within-site discrimination vs between-site score distributions. No model is trained, no subject is removed, no rule is changed."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.simplefilter("ignore")
sys.path.insert(0, r"C:\abide")
import fast_accuracy_experiment as fx

R = Path(r"C:\abide\results")
OUT = R / "accuracy_research_v3"
man = pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", dtype={"sid": str})
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str}); coh["sid"] = coh.subject_id.map(fx.norm_id)
coh = coh[coh.sid.isin(set(man.sid))].set_index("sid").loc[man.sid]
meta = pd.DataFrame({"project_id": [f"DEV{i:04d}" for i in range(600)], "age": coh.age.values.astype(float), "female": (coh.sex.values == 2).astype(int), "fd": coh.func_mean_fd.values.astype(float)})
bo = pd.read_csv(R / "accuracy_improvement_v1" / "oof_predictions.csv"); bo = bo[(bo.candidate == "E0_control_baseline") & (bo.seed == 42)]
to = pd.read_csv(R / "accuracy_research_v2" / "oof_predictions_tangent.csv"); to = to[to.seed == 42]
out = []
md = []
for name, d in (("BASELINE (Fisher-Z Pearson)", bo), ("TANGENT (Ledoit-Wolf)", to)):
    d = d.merge(meta, on="project_id"); d["err"] = (d.y_true != d.y_pred).astype(int)
    md.append(f"\n## {name}  (seed 42, five-fold OOF, n={len(d)})\n")
    md.append(f"Accuracy {1 - d.err.mean():.3f}; sensitivity {1 - d[d.y_true == 1].err.mean():.3f}; specificity {1 - d[d.y_true == 0].err.mean():.3f}; predicted-ASD share {d.y_pred.mean():.3f} (true {d.y_true.mean():.3f}).\n")
    # by sex x class
    t = d.assign(sex=d.female.map({0: "male", 1: "female"})).groupby(["sex", "y_true"]).agg(n=("err", "size"), accuracy=("err", lambda s: 1 - s.mean()))
    md.append("By sex and class:\n```\n" + t.round(3).to_string() + "\n```\n")
    p_sex = stats.chi2_contingency(pd.crosstab(d.female, d.err))[1]
    d["age_t"] = pd.qcut(d.age, 3, labels=["young", "mid", "old"]); d["fd_t"] = pd.qcut(d.fd, 3, labels=["low FD", "mid FD", "high FD"])
    ta = d.groupby(["age_t", "y_true"]).agg(n=("err", "size"), accuracy=("err", lambda s: 1 - s.mean()))
    tf = d.groupby(["fd_t", "y_true"]).agg(n=("err", "size"), accuracy=("err", lambda s: 1 - s.mean()))
    md.append("By age tercile and class:\n```\n" + ta.round(3).to_string() + "\n```\nBy mean-FD tercile and class:\n```\n" + tf.round(3).to_string() + "\n```\n")
    md.append(f"Error vs sex chi2 p={p_sex:.3f}; Spearman(error, age) rho={stats.spearmanr(d.err, d.age).correlation:.3f} (p={stats.spearmanr(d.err, d.age).pvalue:.3f}); Spearman(error, FD) rho={stats.spearmanr(d.err, d.fd).correlation:.3f} (p={stats.spearmanr(d.err, d.fd).pvalue:.3f}).\n")
    # within-site vs between-site
    g_ok = [(s, g) for s, g in d.groupby("site") if len(g) >= 20 and g.y_true.nunique() == 2]
    within = np.array([roc_auc_score(g.y_true, g.score) for _, g in g_ok])
    pooled = roc_auc_score(d.y_true, d.score)
    # variance of scores explained by site vs class
    mu = d.score.mean(); sst = ((d.score - mu) ** 2).sum()
    eta_site = sum(len(g) * (g.score.mean() - mu) ** 2 for _, g in d.groupby("site")) / sst
    eta_cls = sum(len(g) * (g.score.mean() - mu) ** 2 for _, g in d.groupby("y_true")) / sst
    site_tab = d.groupby("site").agg(n=("err", "size"), true_asd=("y_true", "mean"), pred_asd=("y_pred", "mean"), mean_score=("score", "mean"), mean_age=("age", "mean"), mean_fd=("fd", "mean"), female=("female", "mean"))
    site_tab = site_tab[site_tab.n >= 20]
    rho_prev = stats.spearmanr(site_tab.mean_score, site_tab.true_asd)
    rho_fd = stats.spearmanr(site_tab.mean_score, site_tab.mean_fd); rho_age = stats.spearmanr(site_tab.mean_score, site_tab.mean_age)
    md.append(f"Pooled AUC {pooled:.3f}; mean within-site AUC over {len(within)} sites with n >= 20: {within.mean():.3f} (range {within.min():.2f}-{within.max():.2f}); share of score variance explained by site {eta_site:.3f} vs by true class {eta_cls:.3f}. "
              f"Across the {len(site_tab)} larger sites, the site mean score correlates with the site's true ASD share (Spearman {rho_prev.correlation:.2f}, p={rho_prev.pvalue:.2f}), with site mean FD ({rho_fd.correlation:.2f}, p={rho_fd.pvalue:.2f}) and with site mean age ({rho_age.correlation:.2f}, p={rho_age.pvalue:.2f}).\n")
    md.append("Site table (n >= 20): true vs predicted ASD share and mean score:\n```\n" + site_tab.round(3).to_string() + "\n```\n")
    out.append({"model": name, "accuracy": 1 - d.err.mean(), "sensitivity": 1 - d[d.y_true == 1].err.mean(), "specificity": 1 - d[d.y_true == 0].err.mean(), "pooled_auc": pooled, "mean_within_site_auc": within.mean(),
                "score_variance_share_site": eta_site, "score_variance_share_class": eta_cls, "error_vs_sex_p": p_sex, "spearman_error_age": stats.spearmanr(d.err, d.age).correlation, "spearman_error_fd": stats.spearmanr(d.err, d.fd).correlation,
                "site_mean_score_vs_true_prevalence_rho": rho_prev.correlation, "site_mean_score_vs_site_fd_rho": rho_fd.correlation, "site_mean_score_vs_site_age_rho": rho_age.correlation})
pd.DataFrame(out).to_csv(OUT / "generalization_diagnosis_summary.csv", index=False)
(OUT / "generalization_diagnosis.md").write_text("# Experiment 2: generalization diagnosis (descriptive; saved predictions; seed 42)\n" + "".join(md), encoding="utf-8")
print("".join(md))
