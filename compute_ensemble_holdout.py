import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix, precision_score

warnings.filterwarnings("ignore")

DATA = r"C:\abide\abide_pcp"
TS_PATH = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH = os.path.join(DATA, "phenotypic_qc.csv")
MIN_TIMEPOINTS = 100
SEED = 42

ts = np.load(TS_PATH, allow_pickle=True)
ph = pd.read_csv(PHENO_PATH)
keep = np.array([t is not None and t.shape[0] >= MIN_TIMEPOINTS for t in ts])
ts = ts[keep]; ph = ph.loc[keep].reset_index(drop=True)
y = (ph["DX_GROUP"].values == 1).astype(int)
sites = ph["SITE_ID"].astype(str).values

num = lambda c: pd.to_numeric(ph.get(c), errors="coerce").replace(-9999, np.nan).values
age = num("AGE_AT_SCAN").astype(float)
sex = num("SEX").astype(float)
demo = np.column_stack([age, sex])
covs = np.load(COVS_PATH)

all_idx = np.arange(len(y))
tv_idx, ho_idx = train_test_split(all_idx, test_size=0.20, stratify=y, random_state=SEED)
covs_tv, covs_ho = covs[tv_idx], covs[ho_idx]
y_tv, y_ho = y[tv_idx], y[ho_idx]
demo_tv, demo_ho = demo[tv_idx], demo[ho_idx]
sites_tv, sites_ho = sites[tv_idx], sites[ho_idx]

from tangent import TangentEmbedding
tang = TangentEmbedding("correlation", precomputed=True)
X_tv = tang.fit_transform(covs_tv)
X_ho = tang.transform(covs_ho)

# Demographics
d_tv = demo_tv.copy(); d_ho = demo_ho.copy()
for col in range(d_tv.shape[1]):
    med = np.nanmedian(d_tv[:, col])
    d_tv[np.isnan(d_tv[:, col]), col] = med
    d_ho[np.isnan(d_ho[:, col]), col] = med
ds = StandardScaler()
d_tv = ds.fit_transform(d_tv)
d_ho = ds.transform(d_ho)
X_tv = np.hstack([X_tv, d_tv])
X_ho = np.hstack([X_ho, d_ho])

fs = StandardScaler()
X_tv = fs.fit_transform(X_tv)
X_ho = fs.transform(X_ho)

# Site Harmonized
X_tv_h = X_tv.copy(); X_ho_h = X_ho.copy()
for s in np.unique(sites_tv):
    m_tr = (sites_tv == s)
    s_mean = X_tv_h[m_tr].mean(axis=0)
    X_tv_h[m_tr] -= s_mean
    m_ho = (sites_ho == s)
    if m_ho.any():
        X_ho_h[m_ho] -= s_mean

# PCA
pca200 = PCA(n_components=200, random_state=SEED)
X_tv_pca200 = pca200.fit_transform(X_tv)
X_ho_pca200 = pca200.transform(X_ho)

# 1. LogReg-L2
m1 = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs", class_weight="balanced", random_state=SEED)
m1.fit(X_tv, y_tv)
p1 = m1.predict_proba(X_ho)[:, 1]

# 2. LinearSVC (calibrated)
m2 = CalibratedClassifierCV(LinearSVC(C=0.01, max_iter=5000, class_weight="balanced", random_state=SEED), cv=3)
m2.fit(X_tv, y_tv)
p2 = m2.predict_proba(X_ho)[:, 1]

# 3. LogReg-L2 + SiteHarmonize
m3 = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs", class_weight="balanced", random_state=SEED)
m3.fit(X_tv_h, y_tv)
p3 = m3.predict_proba(X_ho_h)[:, 1]

# 4. PCA(200)+RBF-SVM
m4 = SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
m4.fit(X_tv_pca200, y_tv)
p4 = m4.predict_proba(X_ho_pca200)[:, 1]

# Top-3 Ensemble (LogReg-L2 + LinearSVC + SiteHarmonized LogReg)
p_ens = (p1 + p2 + p3) / 3.0
pred_ens = (p_ens >= 0.5).astype(int)

acc = accuracy_score(y_ho, pred_ens)
auc = roc_auc_score(y_ho, p_ens)
bacc = balanced_accuracy_score(y_ho, pred_ens)
cm = confusion_matrix(y_ho, pred_ens)
tn, fp, fn, tp = cm.ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
f1 = f1_score(y_ho, pred_ens)

print(f"TOP-3 ENSEMBLE HOLDOUT ACCURACY : {acc:.4f}")
print(f"TOP-3 ENSEMBLE HOLDOUT ACCURACY : {acc*100:.2f}%")
print(f"IMPROVEMENT OVER 72.93% : {acc*100 - 72.93:+.2f} percentage points")
print(f"Holdout ROC-AUC : {auc:.4f}")
print(f"Holdout Balanced Acc : {bacc*100:.2f}%")
print(f"Sensitivity : {sens*100:.2f}%")
print(f"Specificity : {spec*100:.2f}%")
print(f"F1-Score : {f1:.4f}")
print("Confusion Matrix:")
print(cm)
