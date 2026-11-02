import sys, json, numpy as np, pandas as pd
sys.argv=['x']; import experiments as E
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
X,y,site,sid,fd=E.load(); rows=[]
q=pd.qcut(fd,3,labels=False)
for s in (0,1,2,6,7,8):
    sc=np.load(E.CK/f"none_seed{s}.npz")["score"]
    r=dict(seed=s,auc_all=roc_auc_score(y,sc),rho_score_FD=spearmanr(sc,fd).correlation)
    for t in range(3): r[f"auc_FD_tertile{t}"]=roc_auc_score(y[q==t],sc[q==t])
    # within-site-centered score vs label (site-adjusted AUC): mean of per-site AUC weighted by n for sites n>=20 with both classes
    ws=[(roc_auc_score(y[site==k],sc[site==k]),(site==k).sum()) for k in np.unique(site) if (site==k).sum()>=20]
    r["weighted_within_site_auc"]=sum(a*n for a,n in ws)/sum(n for _,n in ws); rows.append(r)
d=pd.DataFrame(rows); d.to_csv("diagnostics_D8_score_vs_motion.csv",index=False); print(d.round(3).to_string()); print(d.mean().round(3).to_dict())
# learning-curve extrapolation (descriptive): AUC vs log2(n_train)
lc=pd.read_csv("diagnostics_D5_learning_curve.csv",header=[0,1],index_col=0)
