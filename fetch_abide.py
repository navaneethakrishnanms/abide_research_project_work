"""Download ABIDE Preprocessed (PCP) ROI time series for all sites.

Retries around transient Windows file-locking errors (antivirus or a sync
client grabbing the .part file mid-move); already-downloaded subjects are
skipped, so re-running resumes rather than restarting.
"""
import glob, os, time
import numpy as np, pandas as pd
from nilearn.datasets import fetch_abide_pcp

DATA = r"C:\abide\abide_pcp"
KW = dict(data_dir=DATA, pipeline="cpac", band_pass_filtering=True,
          global_signal_regression=True, derivatives=["rois_cc200"],
          quality_checked=True, verbose=0)

d = None
for attempt in range(1, 61):
    # Clear stale .part files left behind by an interrupted move.
    for p in glob.glob(os.path.join(DATA, "**", "*.part"), recursive=True):
        try: os.remove(p)
        except OSError: pass
    try:
        d = fetch_abide_pcp(**KW)
        break
    except Exception as exc:
        n = len(glob.glob(os.path.join(DATA, "**", "*_rois_cc200.1D"), recursive=True))
        print(f"  attempt {attempt}: {type(exc).__name__} after {n} files - retrying", flush=True)
        time.sleep(3)

if d is None:
    raise SystemExit("download did not complete after 60 attempts")

ph = pd.DataFrame(d.phenotypic)
print("subjects:", len(d.rois_cc200))
print("sites:", ph["SITE_ID"].nunique())
print(ph["SITE_ID"].value_counts().to_string())
print("DX_GROUP (1=ASD,2=TC):", ph["DX_GROUP"].value_counts().to_dict())
shapes = [t.shape for t in d.rois_cc200]
print("regions:", shapes[0][1], "| timepoints:", min(s[0] for s in shapes), "-", max(s[0] for s in shapes))

ph.to_csv(os.path.join(DATA, "phenotypic_qc.csv"), index=False)
np.save(os.path.join(DATA, "ts_cc200.npy"), np.array(d.rois_cc200, dtype=object), allow_pickle=True)
print("saved.")
