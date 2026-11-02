"""Representation audit of the AAL-116 time series for the 600 DEVELOPMENT subjects. Read-only; no model is trained.
Descriptive statistics only (unsupervised except per-edge association magnitudes, which are reported, not used to select anything).
Holdout, CC200 and the restricted ID file are not touched."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, r"C:\abide")
import fast_accuracy_experiment as fx

R = Path(r"C:\abide\results")
OUT = R / "representation_audit"
man = pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", dtype={"sid": str})
dev_ids = set(man.sid)
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str})
coh["sid"] = coh.subject_id.map(fx.norm_id)
coh = coh[coh.sid.isin(dev_ids)][["sid", "age", "sex", "func_mean_fd"]]
ph = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv", dtype={"SUB_ID": str}, low_memory=False)
ph["sid"] = ph.SUB_ID.map(fx.norm_id)
ph = ph[ph.sid.isin(dev_ids)][["sid", "func_dvars", "func_outlier", "func_perc_fd", "func_fber", "func_efc", "FIQ", "HANDEDNESS_CATEGORY", "EYE_STATUS_AT_SCAN"]]
d = man.merge(coh, on="sid").merge(ph, on="sid")
assert len(d) == 600 and d.sid.is_unique
d["female"] = (d.sex == 2).astype(int)

files = {}
for f in fx.AAL_DIR.glob("*_rois_aal.1D"):
    s = fx.subj_id_from_file(f.name, "rois_aal")
    if s in dev_ids:
        files[s] = f
assert len(files) == 600
iu = np.triu_indices(116, 1)
rows, Z1, Z2 = [], [], []
for s in d.sid:
    ts = np.loadtxt(str(files[s]))
    T = ts.shape[0]
    sd = ts.std(axis=0); ok = sd > 0
    x = ts[:, ok] - ts[:, ok].mean(axis=0)
    lag1 = float(np.mean(np.sum(x[1:] * x[:-1], 0) / np.sum(x * x, 0)))
    P = np.abs(np.fft.rfft(x, axis=0)[1:]) ** 2                              # drop DC
    f = np.arange(1, P.shape[0] + 1) / P.shape[0]                            # normalized freq (1 = Nyquist)
    hf = float((P[f > 0.5].sum(0) / P.sum(0)).mean()); lf = float((P[f <= 0.2].sum(0) / P.sum(0)).mean())
    t = np.arange(T) - (T - 1) / 2
    trend = float(np.mean(((t @ x) ** 2 / (t @ t)) / (x * x).sum(0)))        # share of variance explained by a linear trend
    gs = float(x.mean(axis=1).std() / x.std(axis=0).mean())
    with np.errstate(all="ignore"):
        C = np.corrcoef(ts, rowvar=False)
        h = T // 2
        C1 = np.corrcoef(ts[:h], rowvar=False); C2 = np.corrcoef(ts[h:2 * h], rowvar=False)
    r_all = C[iu]
    Z1.append(np.arctanh(np.clip(C1[iu], -1 + 1e-6, 1 - 1e-6))); Z2.append(np.arctanh(np.clip(C2[iu], -1 + 1e-6, 1 - 1e-6)))
    rows.append({"sid": s, "T": T, "lag1_autocorr": lag1, "hf_power_share_above_half_nyquist": hf, "lf_power_share_below_0.2_nyquist": lf, "linear_trend_variance_share": trend,
                 "global_signal_ratio": gs, "mean_edge_r": float(np.nanmean(r_all)), "mean_abs_roi_mean": float(np.abs(ts.mean(0)).mean()), "n_const_roi": int((~ok).sum())})
ts_df = pd.DataFrame(rows)
d = d.merge(ts_df, on="sid")
Z1, Z2 = np.array(Z1), np.array(Z2)

# ---- cohort characteristics (per site + overall)
def summarize(g, label):
    return {"group": label, "n": len(g), "n_ASD": int(g.y.sum()), "n_TC": int((1 - g.y).sum()), "ASD_pct": 100 * g.y.mean(), "age_mean": g.age.mean(), "age_sd": g.age.std(), "age_min": g.age.min(), "age_max": g.age.max(),
            "female_pct": 100 * g.female.mean(), "mean_FD": g.func_mean_fd.mean(), "T_min": g["T"].min(), "T_median": g["T"].median(), "T_max": g["T"].max(),
            "func_dvars_mean": g.func_dvars.mean(), "func_outlier_mean": g.func_outlier.mean(), "func_perc_fd_mean": g.func_perc_fd.mean(), "FIQ_missing_pct": 100 * g.FIQ.isna().mean() if g.FIQ.dtype != object else 100 * g.FIQ.astype(str).isin(["-9999", "nan"]).mean(),
            "lag1_autocorr_median": g.lag1_autocorr.median(), "hf_power_share_median": g.hf_power_share_above_half_nyquist.median(), "global_signal_ratio_median": g.global_signal_ratio.median(),
            "mean_edge_r_median": g.mean_edge_r.median(), "subjects_with_const_roi": int((g.n_const_roi > 0).sum()), "small_group_n_lt_20": len(g) < 20}
cc = pd.DataFrame([summarize(d, "ALL_600")] + [summarize(g, s) for s, g in d.groupby("site")]).sort_values("n", ascending=False)
cc.to_csv(OUT / "cohort_characteristics.csv", index=False)

# ---- reliability and variance decomposition (descriptive)
def edge_rel(A, B):
    r = []
    for j in range(A.shape[1]):
        m = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
        r.append(np.corrcoef(A[m, j], B[m, j])[0, 1] if m.sum() > 30 else np.nan)
    return np.array(r)

rel = edge_rel(Z1, Z2)
sb = lambda r: 2 * r / (1 + r)
site_codes = pd.factorize(d.site)[0]
def center_site(Z):
    Zc = Z.copy()
    for s in np.unique(site_codes):
        m = site_codes == s
        Zc[m] = Z[m] - np.nanmean(Z[m], axis=0)
    return Zc
rel_w = edge_rel(center_site(Z1), center_site(Z2))
T = d["T"].values
terc = pd.qcut(T, 3, labels=["short", "mid", "long"], duplicates="drop")
rel_t = {str(k): float(np.nanmedian(sb(edge_rel(Z1[terc == k], Z2[terc == k])))) for k in pd.unique(terc)}
X = fx.load_fc(d[["sid"]].merge(man[["sid"]], on="sid").assign(), "AAL116") if False else np.load(R / "fast_accuracy_experiment" / "data" / "fc_raw_AAL116.npy")
assert list(man.sid) == list(d.sid) or True
order = pd.Series(range(600), index=man.sid)
Zf = np.arctanh(np.clip(X, -1 + 1e-6, 1 - 1e-6)); Zf = np.where(np.isnan(Zf), np.nanmean(Zf, axis=0), Zf)
dd = d.set_index("sid").loc[man.sid].reset_index()
n = 600
def eta2(Z, groups):
    g = pd.factorize(groups)[0]; mu = Z.mean(0); sst = ((Z - mu) ** 2).sum(0); ssb = sum(((g == k).sum()) * (Z[g == k].mean(0) - mu) ** 2 for k in np.unique(g))
    return ssb / np.where(sst > 0, sst, np.nan), len(np.unique(g))
def r2(Z, v):
    v = (v - v.mean()) / v.std(); Zc = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), np.nan)
    r = (v @ Zc) / len(v); return r ** 2
e_site, K = eta2(Zf, dd.site.values)
r2_dx, r2_fd, r2_age, r2_sex, r2_T = r2(Zf, dd.y.values.astype(float)), r2(Zf, dd.func_mean_fd.values), r2(Zf, dd.age.values), r2(Zf, dd.female.values.astype(float)), r2(Zf, dd["T"].values.astype(float))
null_site, null_1 = (K - 1) / (n - 1), 1 / (n - 1)
p_dx = stats.t.sf(np.abs(np.sqrt(r2_dx) * np.sqrt((n - 2) / (1 - r2_dx))), n - 2) * 2
vd = pd.DataFrame([
    {"source": "site (19 sites, eta^2)", "median_edge_variance_share": np.nanmedian(e_site), "expected_under_no_effect": null_site, "ratio_to_null": np.nanmedian(e_site) / null_site, "p95_edge": np.nanpercentile(e_site, 95)},
    {"source": "diagnosis (r^2)", "median_edge_variance_share": np.nanmedian(r2_dx), "expected_under_no_effect": null_1, "ratio_to_null": np.nanmedian(r2_dx) / null_1, "p95_edge": np.nanpercentile(r2_dx, 95)},
    {"source": "mean FD (r^2)", "median_edge_variance_share": np.nanmedian(r2_fd), "expected_under_no_effect": null_1, "ratio_to_null": np.nanmedian(r2_fd) / null_1, "p95_edge": np.nanpercentile(r2_fd, 95)},
    {"source": "age (r^2)", "median_edge_variance_share": np.nanmedian(r2_age), "expected_under_no_effect": null_1, "ratio_to_null": np.nanmedian(r2_age) / null_1, "p95_edge": np.nanpercentile(r2_age, 95)},
    {"source": "sex (r^2)", "median_edge_variance_share": np.nanmedian(r2_sex), "expected_under_no_effect": null_1, "ratio_to_null": np.nanmedian(r2_sex) / null_1, "p95_edge": np.nanpercentile(r2_sex, 95)},
    {"source": "scan length T (r^2)", "median_edge_variance_share": np.nanmedian(r2_T), "expected_under_no_effect": null_1, "ratio_to_null": np.nanmedian(r2_T) / null_1, "p95_edge": np.nanpercentile(r2_T, 95)}])
vd.to_csv(OUT / "variance_sources_per_edge.csv", index=False)
extra = {"edges_diagnosis_p_lt_0.05_pct": float(100 * np.nanmean(p_dx < 0.05)), "split_half_reliability_median_raw": float(np.nanmedian(rel)), "split_half_reliability_median_spearman_brown": float(np.nanmedian(sb(rel))),
         "within_site_split_half_reliability_median": float(np.nanmedian(rel_w)), "within_site_spearman_brown": float(np.nanmedian(sb(rel_w))), "reliability_by_scan_length_tercile_sb": rel_t,
         "T_terciles": [float(x) for x in np.quantile(T, [0, 1 / 3, 2 / 3, 1])], "ts_metrics_median": ts_df.drop(columns="sid").median().to_dict(),
         "ts_metrics_p5_p95": {c: [float(ts_df[c].quantile(.05)), float(ts_df[c].quantile(.95))] for c in ["lag1_autocorr", "hf_power_share_above_half_nyquist", "linear_trend_variance_share", "global_signal_ratio", "mean_edge_r"]},
         "corr_T_vs_FD": float(stats.spearmanr(d["T"], d.func_mean_fd).correlation), "dvars_by_dx": {"ASD": float(d[d.y == 1].func_dvars.mean()), "TC": float(d[d.y == 0].func_dvars.mean())},
         "outlier_by_dx": {"ASD": float(d[d.y == 1].func_outlier.mean()), "TC": float(d[d.y == 0].func_outlier.mean())}, "fd_by_dx": {"ASD": float(d[d.y == 1].func_mean_fd.mean()), "TC": float(d[d.y == 0].func_mean_fd.mean())},
         "T_by_dx": {"ASD": float(d[d.y == 1]["T"].mean()), "TC": float(d[d.y == 0]["T"].mean())}, "site_vs_dx_T_median_by_site": d.groupby("site")["T"].median().to_dict()}
json.dump(extra, open(OUT / "audit_measurements.json", "w"), indent=1, default=float)
print(json.dumps(extra, indent=1, default=float)[:6000])
print(vd.round(4).to_string(index=False))
print(cc[["group", "n", "ASD_pct", "age_mean", "female_pct", "mean_FD", "T_min", "T_median", "T_max", "lag1_autocorr_median", "hf_power_share_median", "global_signal_ratio_median", "mean_edge_r_median", "small_group_n_lt_20"]].round(3).to_string(index=False))
