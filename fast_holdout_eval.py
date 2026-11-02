"""
fast_holdout_eval.py -- Fast holdout evaluation on 20 CPU threads
Using exact same data, split (seed=42), preprocessing, and models as train_fc_v2.py.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, precision_score
import xgboost as xgb
from tangent import TangentEmbedding

warnings.filterwarnings("ignore")

DATA        = r"C:\abide\abide_pcp"
TS_PATH     = os.path.join(DATA, "ts_cc200.npy")
COVS_PATH   = os.path.join(DATA, "covs_cc200.npy")
PHENO_PATH  = os.path.join(DATA, "phenotypic_qc.csv")
MIN_TIMEPOINTS = 100
SEED           = 42

def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[np.argmax(tpr - fpr)])

def full_metrics(y_true, scores, threshold, label=""):
    y_pred = (scores >= threshold).astype(int)
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    return {
        "label": label, "n": n, "threshold": threshold,
        "accuracy": acc, "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": prec, "sensitivity": sens, "specificity": spec, "cm": cm,
    }

def score_clf(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)

print("Loading data...")
ts   = np.load(TS_PATH, allow_pickle=True)
ph   = pd.read_csv(PHENO_PATH)
keep = np.array([t is not None and t.shape[0] >= MIN_TIMEPOINTS for t in ts])
ts   = ts[keep]; ph = ph.loc[keep].reset_index(drop=True)
y    = (ph["DX_GROUP"].values == 1).astype(int)
sites = ph["SITE_ID"].astype(str).values

num  = lambda c: pd.to_numeric(ph.get(c), errors="coerce").replace(-9999, np.nan).values
age  = num("AGE_AT_SCAN").astype(float)
sex  = num("SEX").astype(float)
demo = np.column_stack([age, sex])
covs = np.load(COVS_PATH)

all_idx = np.arange(len(y))
tv_idx, ho_idx = train_test_split(all_idx, test_size=0.20, stratify=y, random_state=SEED)
covs_tv, covs_ho = covs[tv_idx], covs[ho_idx]
y_tv,    y_ho    = y[tv_idx],    y[ho_idx]
demo_tv, demo_ho = demo[tv_idx], demo[ho_idx]
sites_tv, sites_ho = sites[tv_idx], sites[ho_idx]

print(f"Train+Val: {len(tv_idx)} | Holdout: {len(ho_idx)}")

# 1. Feature extraction (exact same as train_fc_v2.py)
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

# Site Harmonized version
X_tv_h = X_tv.copy(); X_ho_h = X_ho.copy()
for site in np.unique(sites_tv):
    s_mean = X_tv_h[sites_tv == site].mean(axis=0)
    X_tv_h[sites_tv == site] -= s_mean
    m_ho = (sites_ho == site)
    if m_ho.any():
        X_ho_h[m_ho] -= s_mean

# PCA versions
pca200 = PCA(n_components=200, random_state=SEED)
X_tv_pca200 = pca200.fit_transform(X_tv)
X_ho_pca200 = pca200.transform(X_ho)

pca100 = PCA(n_components=100, random_state=SEED)
X_tv_pca100 = pca100.fit_transform(X_tv)
X_ho_pca100 = pca100.transform(X_ho)

# Models to evaluate
models = {
    "LogReg-L2 (balanced)": (
        LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
        {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        X_tv, X_ho
    ),
    "LogReg-L2 + SiteHarmonize": (
        LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
        {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        X_tv_h, X_ho_h
    ),
    "LinearSVC (balanced)": (
        CalibratedClassifierCV(LinearSVC(max_iter=5000, class_weight="balanced"), cv=3),
        {"estimator__C": [0.001, 0.01, 0.1, 1.0]},
        X_tv, X_ho
    ),
    "PCA(200)+RBF-SVM": (
        SVC(probability=True, class_weight="balanced"),
        {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.001, 0.01]},
        X_tv_pca200, X_ho_pca200
    ),
    "PCA(100)+XGBoost": (
        xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, scale_pos_weight=1.0, random_state=SEED, n_jobs=1),
        {"n_estimators": [200, 500], "max_depth": [3, 5], "learning_rate": [0.01, 0.05], "subsample": [0.8], "colsample_bytree": [0.8]},
        X_tv_pca100, X_ho_pca100
    ),
}

inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof_scores = {}
ho_scores = {}

for name, (clf, grid, xtr, xho) in models.items():
    t0 = time.time()
    search = GridSearchCV(clf, grid, cv=inner, scoring="roc_auc", n_jobs=16, refit=True)
    search.fit(xtr, y_tv)
    
    # Compute CV OOF threshold
    cv_preds = np.zeros(len(y_tv))
    for tr_k, val_k in inner.split(xtr, y_tv):
        f_clf = search.best_estimator_
        f_clf.fit(xtr[tr_k], y_tv[tr_k])
        cv_preds[val_k] = score_clf(f_clf, xtr[val_k])
    
    tau = youden_threshold(y_tv, cv_preds)
    p_ho = score_clf(search.best_estimator_, xho)
    
    oof_scores[name] = cv_preds
    ho_scores[name] = p_ho
    
    m_ho = full_metrics(y_ho, p_ho, tau, label=name)
    print(f"[{name}] (elapsed {time.time()-t0:.1f}s) | Holdout Acc: {m_ho['accuracy']*100:.2f}% | Holdout AUC: {m_ho['roc_auc']:.4f} | tau: {tau:.4f}")

# Top-3 Ensemble
sorted_by_auc = sorted(models.keys(), key=lambda k: roc_auc_score(y_tv, oof_scores[k]), reverse=True)
top3 = sorted_by_auc[:3]
print(f"\nTop-3 Models by CV AUC: {top3}")
ens_oof = np.mean([oof_scores[k] for k in top3], axis=0)
ens_tau = youden_threshold(y_tv, ens_oof)
ens_ho  = np.mean([ho_scores[k] for k in top3], axis=0)

m_ens = full_metrics(y_ho, ens_ho, ens_tau, label="Ensemble (top-3 soft-vote)")
print("\n" + "="*70)
print("  FINAL ENSEMBLE HOLDOUT EVALUATION")
print("="*70)
print(f"NEW FINAL HOLDOUT ACCURACY : {m_ens['accuracy']:.4f}")
print(f"NEW FINAL HOLDOUT ACCURACY : {m_ens['accuracy']*100:.2f}%")
print(f"Holdout ROC-AUC            : {m_ens['roc_auc']:.4f}")
print(f"Holdout Balanced Acc       : {m_ens['balanced_acc']*100:.2f}%")
print(f"Sensitivity (Recall)       : {m_ens['sensitivity']*100:.2f}%")
print(f"Specificity                : {m_ens['specificity']*100:.2f}%")
print(f"Precision                  : {m_ens['precision']*100:.2f}%")
print(f"F1-Score                   : {m_ens['f1']:.4f}")
print(f"Confusion Matrix:\n{m_ens['cm']}")
print(f"IMPROVEMENT : {m_ens['accuracy']*100 - 72.93:+.2f} percentage points")
print("="*70)
