"""Read-only feature/data-quality audit of the AAL-116 FC features of the 600 DEVELOPMENT subjects. No model is trained.
Labels are used only for the descriptive QC-confounding associations (section E). Holdout, restricted ID file and CC200 are not touched."""
import hashlib
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
OUT = R / "feature_quality_audit"
FX = R / "fast_accuracy_experiment"
rows = []


def add(section, metric, value, note=""):
    rows.append({"section": section, "metric": metric, "value": value, "note": note})


man = pd.read_csv(FX / "data" / "dev_split_manifest.csv", dtype={"sid": str})
dev_ids = set(man.sid)
X = np.load(FX / "data" / "fc_raw_AAL116.npy")
n, p = X.shape

# ---------------- A. dimensions and ordering
add("A dimensions", "matrix shape (subjects x features)", f"{n} x {p}", "expected 600 x 6670 = 116*115/2")
add("A dimensions", "dtype", str(X.dtype))
iu = np.triu_indices(116, 1)
idx = pd.read_csv(R / "phase5a_fc_diagnostic" / "fc_feature_index.csv")
order_ok = bool((idx.roi_i.values - 1 == iu[0]).all() and (idx.roi_j.values - 1 == iu[1]).all() and len(idx) == p and (idx.feature_index.values == np.arange(1, p + 1)).all())
add("A dimensions", "feature order equals Phase 5A feature index (np.triu_indices(116,1), 1-indexed, i<j)", order_ok, "6670 rows compared")
files = {}
for f in fx.AAL_DIR.glob("*_rois_aal.1D"):
    s = fx.subj_id_from_file(f.name, "rois_aal")
    if s in dev_ids:
        files[s] = f
add("A dimensions", "AAL time-series files found for development subjects", len(files), "expected 600")
sample = list(np.random.default_rng(0).choice(n, 40, replace=False)) + [0, n - 1]
maxdiff, shapes = 0.0, set()
for i in sample:
    ts = np.loadtxt(str(files[man.sid[i]]))
    shapes.add(ts.shape[1])
    with np.errstate(all="ignore"):
        C = np.corrcoef(ts, rowvar=False)[iu]
    both = np.isfinite(C) & np.isfinite(X[i])
    assert (np.isfinite(C) == np.isfinite(X[i])).all()
    maxdiff = max(maxdiff, float(np.abs(C[both] - X[i][both]).max()))
add("A dimensions", "cached matrix vs recomputation from raw .1D for 42 subjects (40 random + first + last): max abs difference and NaN pattern identical", f"{maxdiff:.2e}", f"ROI counts in files: {sorted(shapes)}")

# ---------------- B. feature values
fin = np.isfinite(X)
add("B values", "NaN entries", int(np.isnan(X).sum()), f"{100 * np.isnan(X).mean():.3f}% of 4,002,000 entries")
add("B values", "infinite entries", int(np.isinf(X).sum()))
add("B values", "features with any NaN", int(np.isnan(X).any(0).sum()), "undefined correlations from constant (zero) ROIs")
add("B values", "subjects with any NaN", int(np.isnan(X).any(1).sum()))
add("B values", "max NaN entries in one subject", int(np.isnan(X).sum(1).max()))
add("B values", "features with NaN in more than 5% of subjects", int((np.isnan(X).mean(0) > 0.05).sum()))
add("B values", "min / max correlation", f"{np.nanmin(X):.4f} / {np.nanmax(X):.4f}")
add("B values", "entries with |r| >= 0.99", int((np.abs(X[fin]) >= 0.99).sum()), "would need the 1e-6 clip only if |r| = 1 exactly")
add("B values", "entries with |r| = 1 exactly", int((np.abs(X[fin]) >= 1.0).sum()))
Z = np.arctanh(np.clip(X, -1 + 1e-6, 1 - 1e-6))
mu = np.nanmean(Z, axis=0); Zf = np.where(np.isnan(Z), mu, Z)
sd = Zf.std(0)
add("B values", "Fisher-Z range", f"{np.nanmin(Z):.3f} / {np.nanmax(Z):.3f}")
add("B values", "constant features (SD = 0)", int((sd == 0).sum()))
for thr in (0.02, 0.05):
    add("B values", f"near-constant features (SD of Fisher-Z < {thr})", int((sd < thr).sum()), f"median feature SD {np.median(sd):.3f}")
# duplicates
h = [hashlib.md5(np.round(Zf[:, j], 8).tobytes()).hexdigest() for j in range(p)]
add("B values", "exactly duplicated feature columns (rounded 1e-8)", int(p - len(set(h))))
sub = np.random.default_rng(1).choice(p, 2500, replace=False)
Cc = np.corrcoef(Zf[:, sub].T); ac = np.abs(Cc[np.triu_indices(2500, 1)])
add("B values", "random feature pairs with |r| > 0.95 (2,500 features, 3.1M pairs)", f"{100 * (ac > 0.95).mean():.4f}%", f"|r|>0.8: {100 * (ac > 0.8).mean():.4f}%")
sk = stats.skew(Zf, axis=0)
add("B values", "features with |skewness| > 1", int((np.abs(sk) > 1).sum()), f"median |skew| {np.median(np.abs(sk)):.3f}")
med = np.median(Zf, 0); mad = stats.median_abs_deviation(Zf, axis=0, scale="normal"); mad = np.where(mad == 0, np.nan, mad)
rz = (Zf - med) / mad
ext = np.abs(rz) > 6
add("B values", "entries with robust |z| > 6", int(np.nansum(ext)), f"{100 * np.nansum(ext) / ext.size:.3f}% (Gaussian expectation about 1e-7%)")
add("B values", "features with at least one robust |z| > 6", int(ext.any(0).sum()))
sub_ext = ext.mean(1)
add("B values", "subjects with > 1% of features at robust |z| > 6", int((sub_ext > 0.01).sum()), f"max share {100 * sub_ext.max():.2f}%")
# flagged features
flag = pd.DataFrame({"feature_index": np.arange(1, p + 1), "roi_i": idx.roi_i, "roi_j": idx.roi_j, "roi_i_name": idx.roi_i_name, "roi_j_name": idx.roi_j_name, "n_nan_subjects": np.isnan(X).sum(0), "sd_fisher_z": sd})
flag = flag[(flag.n_nan_subjects > 0) | (flag.sd_fisher_z < 0.05)]
flag.to_csv(OUT / "flagged_features.csv", index=False)
add("B values", "rows in flagged_features.csv (any NaN or SD < 0.05)", len(flag))

# ---------------- C. subject-level
ph = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv", dtype={"SUB_ID": str}, low_memory=False)
ph["sid"] = ph.SUB_ID.map(fx.norm_id); ph = ph[ph.sid.isin(dev_ids)].set_index("sid").loc[man.sid]            # development rows only
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str}); coh["sid"] = coh.subject_id.map(fx.norm_id)
coh = coh[coh.sid.isin(dev_ids)].set_index("sid").loc[man.sid]
subj = pd.DataFrame({"site": man.site.values, "y": man.y.values, "n_nan": np.isnan(X).sum(1), "ext_share": sub_ext, "mean_r": np.nanmean(X, 1)}, index=man.sid)
add("C subjects", "subjects with mean correlation outside [-0.05, 0.05]", int(((subj.mean_r < -0.05) | (subj.mean_r > 0.05)).sum()), "global-signal-regressed data are expected near 0")
add("C subjects", "class counts", f"{int(man.y.sum())} ASD / {int((1 - man.y).sum())} TC", f"{man.site.nunique()} sites")

# ---------------- D. site-level quality and feature shift
Tlen = {s: np.loadtxt(str(files[s])).shape[0] for s in man.sid}
subj["T"] = [Tlen[s] for s in subj.index]
gm = Zf.mean(0); gsd = Zf.std(0); good = gsd > 0
site_rows = []
for s, g in subj.groupby("site"):
    m = (man.site.values == s)
    shift = np.sqrt(np.mean(((Zf[m][:, good].mean(0) - gm[good]) / gsd[good]) ** 2))
    site_rows.append({"site": s, "n": int(m.sum()), "n_ASD": int(man.y[m].sum()), "n_TC": int((1 - man.y[m]).sum()), "small_site_n_lt_20": bool(m.sum() < 20),
                      "T_min": int(g["T"].min()), "T_median": float(g["T"].median()), "T_max": int(g["T"].max()),
                      "mean_FD": float(coh.func_mean_fd[m].mean()), "func_dvars_mean": float(ph.func_dvars[m].astype(float).mean()), "func_outlier_mean": float(ph.func_outlier[m].astype(float).mean()),
                      "func_perc_fd_mean": float(ph.func_perc_fd[m].astype(float).mean()), "func_fber_mean": float(ph.func_fber[m].astype(float).mean()), "func_efc_mean": float(ph.func_efc[m].astype(float).mean()),
                      "subjects_with_NaN_edges": int((g.n_nan > 0).sum()), "mean_edge_r": float(g.mean_r.mean()), "share_features_robust_z_gt6_mean": float(g.ext_share.mean()),
                      "feature_shift_rms_z": float(shift), "feature_shift_expected_if_no_site_effect": float(np.sqrt(1.0 / m.sum())), "shift_ratio_to_expected": float(shift / np.sqrt(1.0 / m.sum()))})
sq = pd.DataFrame(site_rows).sort_values("n", ascending=False); sq.to_csv(OUT / "site_quality_summary.csv", index=False)
add("D sites", "median ratio of observed site mean-FC shift to the shift expected with no site effect", f"{sq.shift_ratio_to_expected.median():.2f}", f"range {sq.shift_ratio_to_expected.min():.2f}-{sq.shift_ratio_to_expected.max():.2f}; 1.0 = no site effect")

# ---------------- E. QC metadata availability and confounding
qc_cols = ["func_mean_fd", "func_num_fd", "func_perc_fd", "func_dvars", "func_outlier", "func_quality", "func_efc", "func_fber", "func_gsr", "func_fwhm", "qc_rater_1", "qc_func_rater_2", "qc_func_rater_3"]
qrows = []
sitecode = pd.factorize(man.site)[0]
for c in qc_cols:
    v = pd.to_numeric(ph[c], errors="coerce") if c != "func_mean_fd" else coh.func_mean_fd.astype(float)
    ok = v.notna() & (v != -9999)
    r = {"variable": c, "n_available": int(ok.sum()), "n_unique": int(v[ok].nunique())}
    if r["n_unique"] > 2 or c in ("qc_func_rater_2",):
        a, t = v[ok][man.y.values[ok.values] == 1], v[ok][man.y.values[ok.values] == 0]
        if len(a) and len(t) and v[ok].nunique() > 1:
            r["AUC_equivalent_ASD_vs_TC"] = float(stats.mannwhitneyu(a, t).statistic / (len(a) * len(t))); r["mannwhitney_p_diagnosis"] = float(stats.mannwhitneyu(a, t).pvalue)
            groups = [v[ok][sitecode[ok.values] == k].values for k in np.unique(sitecode)]
            H, pk = stats.kruskal(*[g for g in groups if len(g) > 1])
            r["kruskal_p_site"] = float(pk); r["epsilon2_site"] = float((H - len(groups) + 1) / (ok.sum() - len(groups)))
            wsum, wt = 0.0, 0.0
            for k in np.unique(sitecode):
                m = ok.values & (sitecode == k); yy = man.y.values[m]; vv = v[m].values
                if len(np.unique(yy)) == 2 and len(vv) >= 6:
                    aa, tt = vv[yy == 1], vv[yy == 0]; w = len(aa) * len(tt); wsum += w * stats.mannwhitneyu(aa, tt).statistic / w; wt += w
            r["within_site_AUC_equivalent"] = float(wsum / wt) if wt else np.nan
    else:
        r["note"] = "constant or binary: no usable variation" if r["n_unique"] <= 1 else "binary rating"
    qrows.append(r)
qdf = pd.DataFrame(qrows); qdf.to_csv(OUT / "qc_metadata_summary.csv", index=False)
add("E QC metadata", "available QC variables with variation", ", ".join(qdf[qdf.n_unique > 1].variable), "ratings with a single value carry no information")
add("E QC metadata", "per-frame FD / TR / scan-level time-series QC files", "NOT AVAILABLE", "only subject-level summaries exist")
json.dump({"A_order_ok": order_ok, "maxdiff": maxdiff}, open(OUT / "audit_facts.json", "w"))
pd.DataFrame(rows).to_csv(OUT / "feature_quality_summary.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False, max_colwidth=110))
print("\nQC metadata:\n", qdf.round(4).to_string(index=False))
print("\nSite quality:\n", sq[["site", "n", "T_median", "mean_FD", "func_dvars_mean", "func_outlier_mean", "func_fber_mean", "subjects_with_NaN_edges", "shift_ratio_to_expected"]].round(3).to_string(index=False))
