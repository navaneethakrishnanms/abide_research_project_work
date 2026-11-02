"""Read-only diagnostics on SAVED artifacts of the completed fast experiment (600 development subjects).
No model is fitted. The locked holdout file is never opened. Outputs go to results/error_diagnostics only.
Feature-matrix checks below are UNSUPERVISED (no labels used) and are diagnostics, not feature selection."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
R = Path(r"C:\abide\results")
FX = R / "fast_accuracy_experiment"
SC = R / "site_centering_experiment"
OUT = R / "error_diagnostics"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
md = []


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


man = pd.read_csv(FX / "data" / "dev_split_manifest.csv", dtype={"sid": str})
oof = pd.read_csv(FX / "oof_predictions.csv", dtype={"subject_id": str})
fr = pd.read_csv(FX / "fold_results.csv")
cohort = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str})
cohort["sid"] = cohort.subject_id.map(lambda s: str(int(float(s))))
cohort = cohort[cohort.sid.isin(set(man.sid))]                      # development rows only
man = man.merge(cohort[["sid", "age", "sex", "func_mean_fd"]], on="sid", how="left")

# ---------------------------------------------------------------- 1. integrity
md.append("# Data integrity diagnostics (existing 600-subject development cohort; no new models)\n")
md.append("## 1. IDs and labels\n")
checks = {
    "unique subject IDs in manifest": man.sid.is_unique,
    "n subjects": len(man),
    "labels consistent with diagnosis (y == ASD)": bool((man.y == (man.diagnosis == "ASD").astype(int)).all()),
    "ASD / TC": f"{int(man.y.sum())} / {int((1 - man.y).sum())}",
    "sites": man.site.nunique(),
    "age/sex/FD found for all dev subjects": bool(man[["age", "sex", "func_mean_fd"]].notna().all().all()),
}
models = oof.groupby(["atlas", "model"])
per_model_ok = []
for (a, m), g in models:
    j = g.merge(man[["sid", "y", "outer_fold"]], left_on="subject_id", right_on="sid")
    per_model_ok.append(len(g) == 600 and g.subject_id.is_unique and (j.y_true == j.y).all() and (j.fold == j.outer_fold).all())
checks["each of 8 candidates: 600 unique OOF subjects, labels and folds equal the manifest"] = bool(all(per_model_ok))
for k, v in checks.items():
    md.append(f"- {k}: **{v}**\n")

# ---------------------------------------------------------------- 2. counts
md.append("\n## 2. Class counts by site, fold, source\nDataset source = ABIDE acquisition site (19 in the development set).\n\n")
st = man.groupby("site").agg(n=("y", "size"), asd=("y", "sum")).assign(tc=lambda d: d.n - d.asd, asd_pct=lambda d: 100 * d.asd / d.n).sort_values("n", ascending=False)
st["flag"] = np.where(st.n < 20, "small (n<20)", "")
st.loc[(st[["asd", "tc"]].min(axis=1) < 5), "flag"] += " minority class <5"
md.append("```\n" + st.round(1).to_string() + "\n```\n")
ct = pd.crosstab(man.site, man.y)
chi = stats.chi2_contingency(ct)
md.append(f"ASD prevalence differs across sites: chi2={chi[0]:.1f}, dof={chi[2]}, p={chi[1]:.3g} (expected counts <5 in {(chi[3] < 5).sum()} cells, so the p-value is approximate). Range {st.asd_pct.min():.0f}%-{st.asd_pct.max():.0f}%.\n")
fc = man.groupby("outer_fold").agg(n=("y", "size"), asd=("y", "sum")).assign(asd_pct=lambda d: 100 * d.asd / d.n)
md.append("\nOuter folds:\n```\n" + fc.round(1).to_string() + "\n```\n")
sf = pd.crosstab(man.site, man.outer_fold)
md.append(f"Smallest site-by-fold cell: {int(sf.values.min())}; sites missing from some test folds: {int((sf == 0).any(axis=1).sum())}.\n")
md.append("\n### Possible confounds (descriptive, development subjects, ASD vs TC)\n")
rows = []
for var in ("age", "func_mean_fd"):
    a, t = man.loc[man.y == 1, var], man.loc[man.y == 0, var]
    rows.append(f"- {var}: ASD mean {a.mean():.2f} (SD {a.std():.2f}), TC mean {t.mean():.2f} (SD {t.std():.2f}); Mann-Whitney p={stats.mannwhitneyu(a, t).pvalue:.3g}")
sx = pd.crosstab(man.y, man.sex)
rows.append(f"- sex counts (rows TC/ASD): {sx.values.tolist()}; chi2 p={stats.chi2_contingency(sx)[1]:.3g}")
md.append("\n".join(rows) + "\n")

# ---------------------------------------------------------------- 3. confusion / per-fold
fold_rows, summ = [], []
for (a, m), g in oof.groupby(["atlas", "model"]):
    for scope, gg in [("overall", g)] + [(f"fold{k}", g[g.fold == k]) for k in range(5)]:
        tn = int(((gg.y_true == 0) & (gg.y_pred == 0)).sum()); fp = int(((gg.y_true == 0) & (gg.y_pred == 1)).sum())
        fn = int(((gg.y_true == 1) & (gg.y_pred == 0)).sum()); tp = int(((gg.y_true == 1) & (gg.y_pred == 1)).sum())
        n = len(gg); acc = (tn + tp) / n; sens = tp / (tp + fn); spec = tn / (tn + fp)
        lo, hi = wilson(tn + tp, n)
        if scope == "overall":
            auc = float(fr[(fr.atlas == a) & (fr.model == m)].roc_auc.mean()); auc_note = "mean of 5 per-fold AUCs"
        else:
            auc = float(fr[(fr.atlas == a) & (fr.model == m) & (fr.fold == int(scope[-1]))].roc_auc.iloc[0]); auc_note = "per-fold AUC"
        summ.append({"atlas": a, "model": m, "scope": scope, "n": n, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": acc, "acc_ci95_lo": lo, "acc_ci95_hi": hi,
                     "sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2, "auc": auc, "auc_note": auc_note})
pd.DataFrame(summ).to_csv(OUT / "oof_error_summary.csv", index=False)
S = pd.DataFrame(summ)
md.append("\n## 3. Out-of-fold performance (saved predictions)\n")
md.append("```\n" + S[S.scope == "overall"][["atlas", "model", "tn", "fp", "fn", "tp", "accuracy", "acc_ci95_lo", "acc_ci95_hi", "sensitivity", "specificity", "balanced_accuracy", "auc"]].round(3).to_string(index=False) + "\n```\n")
pf = S[S.scope != "overall"].pivot_table(index=["atlas", "model"], columns="scope", values="accuracy")
md.append("Per-fold accuracy:\n```\n" + pf.round(3).to_string() + "\n```\n")
gap = fr.groupby(["atlas", "model"]).agg(inner_auc=("inner_cv_auc", "mean"), outer_auc=("roc_auc", "mean"))
gap["optimism"] = gap.inner_auc - gap.outer_auc
md.append("Inner-CV AUC (selected setting) vs outer AUC:\n```\n" + gap.round(3).to_string() + "\n```\n")
md.append(f"Fold means over all 8 candidates: " + ", ".join(f"fold{k}={oof[oof.fold == k].eval('y_true==y_pred').mean():.3f}" for k in range(5)) + "\n")

# ---------------------------------------------------------------- 4/5. site x class errors
rows = []
oof["err"] = (oof.y_true != oof.y_pred).astype(int)
oof = oof.merge(man[["sid", "age", "func_mean_fd"]], left_on="subject_id", right_on="sid", how="left")
for (a, m), g in list(oof.groupby(["atlas", "model"])) + [((a, "ALL_4_MODELS_mean"), g) for a, g in oof.groupby("atlas")]:
    for site, gs in g.groupby("site"):
        n_s = gs.subject_id.nunique()
        acc = 1 - gs.err.mean(); lo, hi = wilson(round(acc * len(gs)), len(gs))
        a1, t1 = gs[gs.y_true == 1], gs[gs.y_true == 0]
        rows.append({"group_type": "site", "atlas": a, "model": m, "group": site, "n_subjects": n_s, "n_asd": int(a1.subject_id.nunique()), "n_tc": int(t1.subject_id.nunique()),
                     "accuracy": acc, "acc_ci95_lo": lo, "acc_ci95_hi": hi, "sensitivity": 1 - a1.err.mean() if len(a1) else np.nan,
                     "specificity": 1 - t1.err.mean() if len(t1) else np.nan, "small_group": n_s < 20})
    for k, gs in g.groupby("fold"):
        a1, t1 = gs[gs.y_true == 1], gs[gs.y_true == 0]
        rows.append({"group_type": "fold", "atlas": a, "model": m, "group": f"fold{k}", "n_subjects": gs.subject_id.nunique(), "n_asd": int(a1.subject_id.nunique()), "n_tc": int(t1.subject_id.nunique()),
                     "accuracy": 1 - gs.err.mean(), "acc_ci95_lo": wilson(round((1 - gs.err.mean()) * len(gs)), len(gs))[0], "acc_ci95_hi": wilson(round((1 - gs.err.mean()) * len(gs)), len(gs))[1],
                     "sensitivity": 1 - a1.err.mean(), "specificity": 1 - t1.err.mean(), "small_group": False})
SF = pd.DataFrame(rows)
SF.to_csv(OUT / "site_and_fold_error_summary.csv", index=False)
md.append("\n## 4. Errors by site (average over the 4 models of each atlas; sites with n<20 are noisy)\n")
tab = SF[(SF.group_type == "site") & (SF.model == "ALL_4_MODELS_mean")].pivot_table(index="group", columns="atlas", values=["accuracy", "n_subjects"])
tab = pd.DataFrame({"n": tab[("n_subjects", "AAL116")], "acc_AAL116": tab[("accuracy", "AAL116")], "acc_CC200": tab[("accuracy", "CC200")]}).sort_values("n", ascending=False)
md.append("```\n" + tab.round(3).to_string() + "\n```\n")
big = tab[tab.n >= 20]
rho = stats.spearmanr(big.acc_AAL116, big.acc_CC200)
md.append(f"Consistency across atlases (sites with n>=20, k={len(big)}): Spearman rho of per-site accuracy = {rho.correlation:.2f} (p={rho.pvalue:.2g}).\n")
fold_a = oof.groupby(["atlas", "fold"]).err.mean().unstack(0)
md.append(f"Fold-level error rate, AAL116 vs CC200: Spearman rho = {stats.spearmanr(fold_a.AAL116, fold_a.CC200).correlation:.2f} (k=5, very low power).\n")
e_site = oof.groupby("site").agg(err=("err", "sum"), n=("err", "size"))
e_site = e_site[e_site.n >= 80]                                       # pooled over 8 models, sites with >=10 subjects
het = stats.chi2_contingency(np.c_[e_site.err, e_site.n - e_site.err])
md.append(f"Do error rates differ by site beyond chance? (sites with >=10 subjects, 8 model-runs pooled; not independent, so p is anti-conservative) chi2={het[0]:.1f}, dof={het[2]}, p={het[1]:.3g}.\n")
# shared hard subjects
subj = oof.groupby("subject_id").agg(n_wrong=("err", "sum"), y=("y_true", "first"), site=("site", "first"), age=("age", "first"), fd=("func_mean_fd", "first"))
p_err = oof.err.mean()
exp_all8 = len(subj) * p_err ** 8
dist = subj.n_wrong.value_counts().sort_index()
md.append(f"\n## 5. Do the models fail on the same subjects?\nMean error rate {p_err:.3f}. Subjects misclassified by all 8 candidate runs: **{int((subj.n_wrong == 8).sum())}** (expected {exp_all8:.1f} if errors were independent); by none: **{int((subj.n_wrong == 0).sum())}**.\n")
md.append("Distribution of 'number of runs (of 8) that misclassify the subject':\n```\n" + dist.to_string() + "\n```\n")
hard = subj[subj.n_wrong >= 7]; easy = subj[subj.n_wrong <= 1]
md.append(f"Consistently wrong (>=7/8) n={len(hard)}: ASD share {hard.y.mean():.2f}, mean age {hard.age.mean():.1f}, mean FD {hard.fd.mean():.3f}. Consistently right (<=1/8) n={len(easy)}: ASD share {easy.y.mean():.2f}, mean age {easy.age.mean():.1f}, mean FD {easy.fd.mean():.3f}. "
          f"Age Mann-Whitney p={stats.mannwhitneyu(hard.age, easy.age).pvalue:.3g}; FD p={stats.mannwhitneyu(hard.fd, easy.fd).pvalue:.3g}.\n")
subj.assign(error_fraction=subj.n_wrong / 8).to_csv(OUT / "subject_error_counts.csv")
# motion: descriptive rank statistic (equals Mann-Whitney U / (n1*n2); no model, not cross-validated) and score-vs-motion correlation
U = stats.mannwhitneyu(man.loc[man.y == 1, "func_mean_fd"], man.loc[man.y == 0, "func_mean_fd"]).statistic
auc_fd = U / ((man.y == 1).sum() * (man.y == 0).sum())
auc_age = stats.mannwhitneyu(man.loc[man.y == 1, "age"], man.loc[man.y == 0, "age"]).statistic / ((man.y == 1).sum() * (man.y == 0).sum())
md.append(f"\n### Motion and age as descriptive single-variable rank statistics (all 600 subjects, not cross-validated, no model)\n"
          f"AUC-equivalent of mean framewise displacement for ASD vs TC = {auc_fd:.3f}; of age = {auc_age:.3f} (0.5 = no separation). "
          "Compare with the models' out-of-fold AUC of about 0.65-0.70.\n")
cr = []
for (a, m), g in oof.groupby(["atlas", "model"]):
    g = g.merge(man[["sid", "func_mean_fd"]].rename(columns={"func_mean_fd": "fd2"}), left_on="subject_id", right_on="sid")
    cr.append((a, m, stats.spearmanr(g.score, g.fd2).correlation, stats.spearmanr(g[g.y_true == 1].score, g[g.y_true == 1].fd2).correlation,
               stats.spearmanr(g[g.y_true == 0].score, g[g.y_true == 0].fd2).correlation))
crd = pd.DataFrame(cr, columns=["atlas", "model", "rho_all", "rho_within_ASD", "rho_within_TC"])
md.append("Spearman correlation of saved out-of-fold scores with mean FD (scores are per-model scales; within-fold models differ):\n```\n" + crd.round(3).to_string(index=False) + "\n```\n")
crd.to_csv(OUT / "score_vs_motion_correlation.csv", index=False)

# ---------------------------------------------------------------- 6. features (unsupervised) and saved selections
md.append("\n## 6. Features and saved feature selection\n")
ks = fr.assign(k=fr.best_params.map(lambda s: json.loads(s).get("sel__k"))).groupby(["atlas", "model"]).k.apply(lambda s: s.value_counts().to_dict())
md.append("Selected k (SelectKBest) per outer fold, as saved:\n```\n" + fr.assign(k=fr.best_params.map(lambda s: json.loads(s).get("sel__k"))).pivot_table(index=["atlas", "model"], columns="fold", values="k", aggfunc="first").to_string() + "\n```\n")
try:
    p6 = pd.read_csv(R / "phase6_strategyC_all_nonzero" / "feature_stability" / "stability_summary.csv")
    md.append("Phase 6 (780-cohort) Elastic Net stability, as saved:\n```\n" + p6.to_string(index=False) + "\n```\n")
except Exception as e:
    md.append(f"(Phase 6 stability file not readable: {e})\n")
for atlas in ("AAL116", "CC200"):
    X = np.load(FX / "data" / f"fc_raw_{atlas}.npy")
    nan_cols = int(np.isnan(X).any(axis=0).sum())
    mu = np.nanmean(X, axis=0); mu = np.where(np.isnan(mu), 0, mu)
    Z = np.arctanh(np.clip(np.where(np.isnan(X), mu, X), -1 + 1e-6, 1 - 1e-6))
    sd = Z.std(axis=0); Zs = (Z - Z.mean(0)) / np.where(sd == 0, 1, sd)
    sv = np.linalg.svd(Zs, compute_uv=False); ev = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    idx = rng.choice(Z.shape[1], 1500, replace=False); C = np.corrcoef(Zs[:, idx].T); iu = np.triu_indices(1500, 1); ac = np.abs(C[iu])
    md.append(f"- {atlas}: matrix {X.shape}, features with any NaN {nan_cols} ({100 * nan_cols / X.shape[1]:.1f}%), near-constant features {(sd < 1e-8).sum()}; "
              f"components for 50/80/90% variance (unsupervised, all 600 rows): {int(np.searchsorted(ev, .5)) + 1}/{int(np.searchsorted(ev, .8)) + 1}/{int(np.searchsorted(ev, .9)) + 1} (n=600 subjects, max rank {len(sv)}); "
              f"among {len(ac):,} random feature pairs, |r|>0.5: {100 * (ac > .5).mean():.2f}%, |r|>0.8: {100 * (ac > .8).mean():.3f}%.\n")
md.append("These matrix checks use no labels and only describe redundancy; no feature was selected.\n")

# ---------------------------------------------------------------- 7. audits
md.append("\n## 7. Leakage / independence review (from recorded files only)\n")
md.append("- Fast experiment and site-centering audits record: dev set = train list (676) INTERSECT 780 cohort = 600; folds identical across runs; all learned steps inside Pipelines; 0 of 311 earlier-phase hashes changed.\n"
          "- **The unopened holdout file does not by itself prove independence.** Recorded facts: (a) 780 - 600 = 180 cohort subjects are not in the train list; the holdout has 170 subjects, so most of those 180 could be holdout subjects. Phases 6-10 evaluated and tuned on all 780, so the earlier phases may have used holdout subjects' data, and their design decisions (imputation, Elastic Net settings, architectures) may have been shaped partly by them. I did not read the holdout IDs to check, so this is **unverified**.\n"
          "- (b) `holdout_guard.py` documents that an older 170-subject holdout was evaluated repeatedly in early development (a holdout-reuse problem); the current split was created afterwards (2026-09-09) as a fresh split. Whether current development choices were informed by holdout data is not demonstrable from these files.\n"
          "- (c) The dev-set CV scores here are estimates on development subjects, with candidate selection not nested; they are not independent test results.\n")
(OUT / "data_integrity_diagnostics.md").write_text("".join(md), encoding="utf-8")
print("\n".join(md)[:20000])
