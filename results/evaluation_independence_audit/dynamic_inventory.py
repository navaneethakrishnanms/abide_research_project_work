"""Inventory of ROI time series for the 600 DEVELOPMENT subjects only (holdout subjects' files are never opened).
Read-only; no feature is regenerated and no model is trained."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(r"C:\abide\results")
OUT = R / "evaluation_independence_audit"
D = Path(r"C:\abide\abide_pcp\ABIDE_pcp\cpac\filt_global")
man = pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", dtype={"sid": str})
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str})
coh["sid"] = coh.subject_id.map(lambda s: str(int(float(s))))
coh = coh[coh.sid.isin(set(man.sid))].set_index("sid")                      # development rows only


def sid_of(name, suffix):
    d = "".join(ch for ch in name.replace(f"_{suffix}.1D", "").split("_")[-1] if ch.isdigit())
    return str(int(d)) if d else None


files = {a: {} for a in ("rois_aal", "rois_cc200")}
dev = set(man.sid)
for a in files:
    for f in D.glob(f"*_{a}.1D"):
        s = sid_of(f.name, a)
        if s in dev:
            files[a][s] = f
rows = []
for i, r in man.iterrows():
    row = {"sid": r.sid, "site": r.site, "outer_fold": r.outer_fold, "func_mean_fd": float(coh.loc[r.sid, "func_mean_fd"])}
    for a, nroi in (("rois_aal", 116), ("rois_cc200", 200)):
        p = files[a].get(r.sid)
        if p is None:
            row.update({f"{a}_file": False})
            continue
        ts = np.loadtxt(str(p))
        row.update({f"{a}_file": True, f"{a}_T": ts.shape[0], f"{a}_nROI": ts.shape[1], f"{a}_nan": int(np.isnan(ts).sum()),
                    f"{a}_const_roi": int((ts.std(axis=0) == 0).sum()), f"{a}_bytes": p.stat().st_size})
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "dynamic_data_inventory.csv", index=False)

summ = {}
for a, nroi in (("rois_aal", 116), ("rois_cc200", 200)):
    ok = df[f"{a}_file"].all()
    T = df[f"{a}_T"]
    summ[a] = {"files_found": int(df[f"{a}_file"].sum()), "all_600_present": bool(ok), "n_rois_values": sorted(df[f"{a}_nROI"].unique().tolist()),
               "T_min": int(T.min()), "T_median": float(T.median()), "T_mean": float(T.mean()), "T_max": int(T.max()), "n_T_lt_100": int((T < 100).sum()), "n_T_lt_120": int((T < 120).sum()),
               "nan_total": int(df[f"{a}_nan"].sum()), "subjects_with_const_roi": int((df[f"{a}_const_roi"] > 0).sum()), "const_roi_total": int(df[f"{a}_const_roi"].sum()),
               "disk_MB": round(df[f"{a}_bytes"].sum() / 1e6, 1)}
al = {"subjects": len(df), "ids_unique": bool(df.sid.is_unique), "fd_finite": bool(np.isfinite(df.func_mean_fd).all()), "fd_range": [round(df.func_mean_fd.min(), 3), round(df.func_mean_fd.max(), 3)],
      "ts_length_identical_across_atlases": bool((df.rois_aal_T == df.rois_cc200_T).all()),
      "folds_counts": df.outer_fold.value_counts().sort_index().to_dict(), "sites": int(df.site.nunique())}
# dynamic-feature cost estimates (sliding window; window W, step S)
est = []
for W, S in ((50, 10), (60, 20), (40, 10)):
    for a, nroi in (("rois_aal", 116), ("rois_cc200", 200)):
        T = df[f"{a}_T"].values
        nwin = np.maximum(0, (T - W) // S + 1)
        nfeat = nroi * (nroi - 1) // 2
        tot_win = int(nwin.sum())
        est.append({"window_TR": W, "step_TR": S, "atlas": a, "windows_min": int(nwin.min()), "windows_mean": round(float(nwin.mean()), 1), "windows_max": int(nwin.max()), "subjects_with_lt_3_windows": int((nwin < 3).sum()),
                    "total_windows": tot_win, "edges_per_window": nfeat, "float32_storage_GB_all_windows": round(tot_win * nfeat * 4 / 1e9, 2),
                    "per_subject_summary_features(var/mean-abs-change per edge)": 2 * nfeat})
est = pd.DataFrame(est)
est.to_csv(OUT / "dynamic_cost_estimates.csv", index=False)
json.dump({"summary": summ, "alignment": al}, open(OUT / "dynamic_inventory_summary.json", "w"), indent=1, default=str)
print(json.dumps({"summary": summ, "alignment": al}, indent=1, default=str))
print(est.to_string(index=False))
print("TR per file types in folder (derivatives found for dev subjects): only rois_aal and rois_cc200 time series; no per-frame FD / confound files.")
