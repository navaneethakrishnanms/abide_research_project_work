"""Stage D diagnostics. CC200, 676 authorized subjects only. Read-only on all earlier artifacts.
Outputs: diagnostics_*.csv/json in this folder. Uses saved out-of-fold predictions (no refit) for D1-D2 and light, fixed-hyperparameter fits for D3-D7."""
import json, hashlib, sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
HERE = Path(__file__).parent; EXP = HERE.parent
BASE = EXP.parent / "stage1_2_b0_baseline"; PRED = EXP / "next_phase_tangent_audit" / "predictions.csv"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
z = np.load(BASE / "features_cc200_train676.npz", allow_pickle=True)
tl = {str(int(float(l))) for l in open(r"C:\abide\splits\phase2_train_subjects.txt") if l.strip()}
sid = z["sid"].astype(str); assert set(sid) == tl and len(sid) == 676
y = z["y"]; X = z["X"]; age = z["age"]; sex = z["sex"]; site = z["site"].astype(str)
ph = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv", low_memory=False); ph["sid"] = ph.SUB_ID.astype(int).astype(str)
ph = ph[ph.sid.isin(tl)].drop_duplicates("sid").set_index("sid").loc[sid]      # filtered to the authorized list on load
assert len(ph) == 676 and (ph.SITE_ID.values == site).all(), "site mismatch"
assert ((ph.DX_GROUP.values == 1).astype(int) == y).all(), "label mismatch (1=ASD)"
out = {}

# ---- D1 data-quality table
q = ph[["func_mean_fd", "func_perc_fd", "func_num_fd", "func_dvars", "func_outlier", "func_quality", "func_fwhm", "func_efc", "func_gsr"]].copy()
q["age"] = age; q["sex"] = sex; q["FIQ"] = ph.FIQ.where(ph.FIQ > 0); q["y"] = y
rows = []
for c in q.columns[:-1]:
    a, b = q.loc[q.y == 1, c].dropna(), q.loc[q.y == 0, c].dropna()
    from scipy.stats import mannwhitneyu
    p = mannwhitneyu(a, b).pvalue
    rows.append(dict(variable=c, n_asd=len(a), n_tc=len(b), median_asd=a.median(), median_tc=b.median(), auc_asd_vs_tc=roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]), mannwhitney_p=p, missing=int(q[c].isna().sum())))
d1 = pd.DataFrame(rows); d1.to_csv(HERE / "diagnostics_D1_variables.csv", index=False)
fd = ph.func_mean_fd.values
crit = {"mean_FD>0.2mm": fd > 0.2, "mean_FD>0.5mm": fd > 0.5, "perc_FD>20%": ph.func_perc_fd.values > 20, "perc_FD>50%": ph.func_perc_fd.values > 50, "num_valid_vol_proxy": ph.func_num_fd.values > 50}
rows = []
for k, m in crit.items():
    rows.append(dict(criterion=k, n_flagged=int(m.sum()), asd_flagged=int((m & (y == 1)).sum()), tc_flagged=int((m & (y == 0)).sum()), sites_affected=int(pd.Series(site[m]).nunique()), top_site=(pd.Series(site[m]).value_counts().index[0] if m.any() else ""), top_site_n=(int(pd.Series(site[m]).value_counts().iloc[0]) if m.any() else 0)))
pd.DataFrame(rows).to_csv(HERE / "diagnostics_D1_qc_criteria.csv", index=False)
st = pd.DataFrame(dict(site=site, y=y, fd=fd, age=age, sex=sex)).groupby("site").agg(n=("y", "size"), n_asd=("y", "sum"), asd_frac=("y", "mean"), median_fd=("fd", "median"), median_age=("age", "median"), male_frac=("sex", "mean"))
st["n_tc"] = st.n - st.n_asd; st = st.sort_values("n", ascending=False); st.to_csv(HERE / "diagnostics_D1_sites.csv")
out["D1"] = dict(n=676, n_asd=int(y.sum()), n_sites=int(len(st)), sites_n_lt_20=int((st.n < 20).sum()), asd_frac_range=[float(st.asd_frac.min()), float(st.asd_frac.max())],
                 fd_asd_median=float(np.median(fd[y == 1])), fd_tc_median=float(np.median(fd[y == 0])))

# ---- D2 performance by subgroup from saved OOF predictions (reference LR and 2-member vote)
P = pd.read_csv(PRED); P["sid"] = P.subject_id.astype(str)
P = P[P.arm.isin(["LR (reference)", "H3: soft vote LR + TS-N"])]
P["fd_q"] = pd.qcut(pd.Series(fd, index=sid).loc[P.sid].values, 3, labels=["FD low", "FD mid", "FD high"])
P["age_g"] = pd.cut(pd.Series(age, index=sid).loc[P.sid].values, [0, 12, 18, 100], labels=["<=12", "12-18", ">18"])
P["ok"] = (P.pred_R1 == P.y_true).astype(int)
rows = []
for arm, g in P.groupby("arm"):
    for key in ["site", "fd_q", "age_g"]:
        for lv, h in g.groupby(key, observed=True):
            per = h.groupby("seed").ok.mean()
            rows.append(dict(arm=arm, grouping=key, level=lv, n_subjects=h.sid.nunique(), asd=int(h.drop_duplicates("sid").y_true.sum()), acc_mean_over_seeds=per.mean(), acc_sd_over_seeds=per.std(), n_seeds=int(h.seed.nunique())))
pd.DataFrame(rows).to_csv(HERE / "diagnostics_D2_subgroup_accuracy.csv", index=False)

# ---- D3 confound-only baselines: site one-hot, FD, age, sex, combinations; 10-fold stratified CV, seeds 0-2, LR balanced
Sd = pd.get_dummies(pd.Series(site)).values.astype(float)
sets = {"site": Sd, "mean_FD": fd[:, None], "age": age[:, None], "sex": sex[:, None], "site+FD+age+sex": np.c_[Sd, fd, age, sex], "FD+age+sex": np.c_[fd, age, sex]}
def cvconf(A, seed):
    s = np.zeros(676)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(A, y):
        sc = StandardScaler().fit(A[tr]); m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(A[tr]), y[tr]); s[te] = m.predict_proba(sc.transform(A[te]))[:, 1]
    return ((s >= .5) == y).mean(), roc_auc_score(y, s)
rows = [dict(confounds=k, acc=np.mean([cvconf(A, s)[0] for s in (0, 1, 2)]), auc=np.mean([cvconf(A, s)[1] for s in (0, 1, 2)])) for k, A in sets.items()]
pd.DataFrame(rows).to_csv(HERE / "diagnostics_D3_confound_only.csv", index=False)

# ---- D4 feature-space structure: effective dimension from PCA (centered, standardized features), and correlation with confounds
Xs = StandardScaler().fit_transform(X)
ev = PCA(n_components=min(676, 300), svd_solver="randomized", random_state=0).fit(Xs).explained_variance_ratio_
tot = 1.0
out["D4"] = dict(var_top1=float(ev[0]), var_top10=float(ev[:10].sum()), var_top50=float(ev[:50].sum()), var_top300=float(ev.sum()),
                 n_pc_for_50pct=int(np.searchsorted(np.cumsum(ev), .5) + 1) if ev.sum() >= .5 else None, n_features=int(X.shape[1]))
# association of first PCs with site and FD (descriptive, whole cohort, no labels)
pcs = PCA(n_components=10, svd_solver="randomized", random_state=0).fit_transform(Xs)
from scipy.stats import f_oneway, spearmanr
out["D4"]["pc_vs_site_eta2"] = [float(1 - sum(((pcs[site == s, i] - pcs[site == s, i].mean()) ** 2).sum() for s in np.unique(site)) / ((pcs[:, i] - pcs[:, i].mean()) ** 2).sum()) for i in range(10)]
out["D4"]["pc_vs_FD_spearman"] = [float(spearmanr(pcs[:, i], fd).correlation) for i in range(10)]
out["D4"]["pc_vs_label_auc"] = [float(roc_auc_score(y, pcs[:, i])) for i in range(10)]

# ---- D5 learning curve / D6 permutation / D7 leave-one-site-out, fixed LR (balanced, C=0.001, standardized), fixed protocol
Xd = np.c_[X, age, sex]
def fit_score(tr, te, yy, C=0.001):
    sc = StandardScaler().fit(Xd[tr]); m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(Xd[tr]), yy[tr]); return m.decision_function(sc.transform(Xd[te]))
def lc(frac, seed):
    r = np.random.RandomState(1000 + seed); accs, aucs = [], []
    s = np.zeros(676)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(Xd, y):
        n = int(round(frac * len(tr))); sub = np.concatenate([r.permutation(tr[y[tr] == c])[: int(round(n * (y[tr] == c).mean()))] for c in (0, 1)])
        s[te] = fit_score(sub, te, y)
    return frac, seed, ((s >= 0) == y).mean(), roc_auc_score(y, s)
jobs = [(f, s) for f in (0.25, 0.5, 0.75, 1.0) for s in (0, 1, 2)]
res = Parallel(n_jobs=12)(delayed(lc)(f, s) for f, s in jobs)
pd.DataFrame(res, columns=["train_fraction", "seed", "acc_R0", "auc"]).groupby("train_fraction").agg(["mean", "std"]).to_csv(HERE / "diagnostics_D5_learning_curve.csv")
def perm(seed):
    yy = np.random.RandomState(500 + seed).permutation(y); s = np.zeros(676)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(Xd, yy): s[te] = fit_score(tr, te, yy)
    return ((s >= 0) == yy).mean(), roc_auc_score(yy, s)
pr = Parallel(n_jobs=12)(delayed(perm)(s) for s in range(6)); out["D6_permutation"] = dict(acc_mean=float(np.mean([a for a, _ in pr])), auc_mean=float(np.mean([b for _, b in pr])), accs=[float(a) for a, _ in pr], aucs=[float(b) for _, b in pr])
def loso(site_):
    te = np.where(site == site_)[0]; tr = np.where(site != site_)[0]
    return site_, te, fit_score(tr, te, y)
rr = Parallel(n_jobs=12)(delayed(loso)(s_) for s_ in np.unique(site))
sc_ = np.zeros(676); rows = []
for s_, te, sco in rr:
    sc_[te] = sco; ok = ((sco >= 0) == y[te]).mean(); a = roc_auc_score(y[te], sco) if len(set(y[te])) == 2 and len(te) >= 15 else np.nan
    rows.append(dict(site=s_, n=len(te), asd_frac=float(y[te].mean()), acc=float(ok), within_site_auc=a, pred_asd_frac=float((sco >= 0).mean())))
pd.DataFrame(rows).to_csv(HERE / "diagnostics_D7_loso_perSite.csv", index=False)
out["D7_loso_pooled"] = dict(acc=float(((sc_ >= 0) == y).mean()), mean_within_site_auc_n_ge_15=float(np.nanmean([r["within_site_auc"] for r in rows])))
# matched random-fold reference (same fixed LR)
rf = []
for seed in (0, 1, 2):
    s = np.zeros(676)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(Xd, y): s[te] = fit_score(tr, te, y)
    rf.append((((s >= 0) == y).mean(), roc_auc_score(y, s)))
out["D7_random10fold_fixedLR"] = dict(acc=float(np.mean([a for a, _ in rf])), auc=float(np.mean([b for _, b in rf])))
out["inputs"] = dict(features_sha=sha(BASE / "features_cc200_train676.npz"), predictions_sha=sha(PRED))
json.dump(out, open(HERE / "diagnostics_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
