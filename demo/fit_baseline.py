"""Fit the frozen AAL-116 baseline ONCE on the 600-subject development cohort and save it for the demo.
No hyperparameter search: C=1e-3, k=2000 are the values the baseline selected in 11 of its 15 stored fits
(results/accuracy_improvement_v1/fold_metrics.csv, E0). Holdout / RESTRICTED files are never opened."""
import hashlib, json, sys, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn
sys.path.insert(0, r"C:\abide"); sys.path.insert(1, r"C:\abide\demo")
import fast_accuracy_experiment as fx
from model_defs import MeanImputeFisherZ
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"C:\abide"); DEMO = ROOT / "demo"
dev, ids = fx.build_dev_set()
man = pd.read_csv(ROOT / "results/fast_accuracy_experiment/data/dev_split_manifest.csv", dtype={"sid": str})
assert len(dev) == 600 and dev.sid.tolist() == man.sid.tolist() and dev.y.tolist() == man.y.tolist() and dev.site.tolist() == man.site.tolist()
sha = hashlib.sha256(open(fx.TRAIN_LIST, "rb").read()).hexdigest()
assert sha == "7ff90fc7e4657dc273d0dcc1b8fac8855d58c5176fada67f6340c46e81740b2a"
X = np.load(ROOT / "results/fast_accuracy_experiment/data/fc_raw_AAL116.npy"); y = dev.y.values
assert X.shape == (600, 6670) and y.sum() == 268

# AAL header (ROI order) must be identical in all 600 development files
files = {}
for f in fx.AAL_DIR.glob("*_rois_aal.1D"):
    s = fx.subj_id_from_file(f.name, "rois_aal")
    if s in set(dev.sid): files[s] = f
assert len(files) == 600
headers = {open(files[s]).readline().strip() for s in dev.sid}
assert len(headers) == 1; header = headers.pop(); roi_labels = header.replace("#", "").split()
assert len(roi_labels) == 116
iu = np.triu_indices(116, k=1)

model = Pipeline([("fz", MeanImputeFisherZ()), ("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=2000)),
                  ("clf", LogisticRegression(C=1e-3, class_weight="balanced", max_iter=2000))])
model.fit(X, y)                       # one fit; no search, no evaluation on these subjects
assert model.named_steps["sel"].get_support().sum() == 2000

joblib.dump(model, DEMO / "baseline_lr.joblib")
meta = {
 "created_utc": datetime.now(timezone.utc).isoformat(),
 "purpose": "Demonstration artifact. NOT a validated diagnostic tool.",
 "training_cohort": {"n": 600, "n_asd": 268, "n_tc": 332, "n_sites": 19,
   "definition": "splits/phase2_train_subjects.txt INTERSECT results/phase2/fd_filtered_subjects.csv",
   "train_list_sha256": sha, "order_and_labels_match": "results/fast_accuracy_experiment/data/dev_split_manifest.csv",
   "holdout_or_restricted_files_opened": False},
 "input": {"atlas": "AAL-116", "source": "ABIDE PCP, CPAC pipeline, filt_global (band-pass 0.01-0.1 Hz, global signal regression), file suffix rois_aal.1D",
   "roi_labels_in_order": roi_labels, "min_timepoints": 100, "n_features": 6670,
   "feature_order": "np.triu_indices(116, k=1) of np.corrcoef(timeseries, rowvar=False)  (row-major upper triangle)"},
 "pipeline": ["MeanImputeFisherZ (training-column means for undefined correlations, arctanh with clip 1e-6)", "StandardScaler",
              "SelectKBest(f_classif, k=2000)", "LogisticRegression(L2, C=1e-3, class_weight='balanced', max_iter=2000)"],
 "hyperparameters": {"C": 1e-3, "k": 2000, "selection": "fixed; modal choice (11 of 15 fits) of the frozen baseline, no new search"},
 "positive_class": "ASD (label 1); decision threshold 0.5",
 "historical_estimates_not_measured_on_this_estimator": {
   "accuracy_cv": "64.7% (seed-averaged 64.2/64.7/65.3)", "auc_cv": 0.694, "balanced_accuracy_cv": 0.643, "sensitivity_cv": 0.609, "specificity_cv": 0.678,
   "protocol": "5 outer folds x 3 inner seeds, preprocessing and selection inside training folds, 600 development subjects",
   "source": "results/accuracy_improvement_v1 (control E0)",
   "caveat": "Cross-validation on the development set; not an independent test; leave-one-site-out accuracy was 62.2%"},
 "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
 "custom_class_required_to_load": "model_defs.MeanImputeFisherZ (demo/model_defs.py, same code as fast_accuracy_experiment.MeanImputeFisherZ)",
}
json.dump(meta, open(DEMO / "baseline_lr_metadata.json", "w"), indent=1)
print("saved", (DEMO / "baseline_lr.joblib").stat().st_size, "bytes")
