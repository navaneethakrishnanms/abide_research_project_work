"""Leakage-safe GPU (PyTorch) experiment on NUMERICAL AAL-116 connectivity features (600 dev subjects, outer folds = saved manifest).
Models: torch logistic regression (L2) and a small torch MLP, both on Fisher-Z FC (+ age, sex).
All fitting (imputation, scaling, ANOVA top-k selection, hyper-parameters, early stopping) uses TRAINING rows of each outer fold only;
hyper-parameters are chosen by inner stratified 3-fold ROC-AUC. Holdout files are never opened. Reports whatever accuracy is achieved."""
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, f1_score

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent  # folder that contains data/ and code/
D = HERE / "data"; OUT = HERE / "outputs" / "cc200_baseline"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
man = pd.read_csv(D / "dev_split_manifest.csv", dtype={"sid": str})
X_raw = np.load(D / "fc_raw_CC200.npy"); y = man.y.values.astype(int); assert X_raw.shape[0] == 600 == len(y)
ph = pd.read_csv(D / "Phenotypic_V1_0b.csv", low_memory=False)
ph["sid"] = ph.SUB_ID.astype(float).astype(int).astype(str)
cov = man[["sid"]].merge(ph[["sid", "AGE_AT_SCAN", "SEX"]], on="sid", how="left")
assert cov.AGE_AT_SCAN.notna().all() and cov.SEX.notna().all()
COV = np.c_[cov.AGE_AT_SCAN.values, (cov.SEX.values == 1).astype(float)]
folds = [(np.where(man.outer_fold != k)[0], np.where(man.outer_fold == k)[0]) for k in range(5)]


def prep(Xtr, Xte, Ctr, Cte, k, ytr):
    """Fit on train only: mean-impute, Fisher-Z, standardise, ANOVA top-k, append standardised covariates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); m = np.nanmean(Xtr, 0)
    m = np.where(np.isnan(m), 0, m)
    f = lambda A: np.arctanh(np.clip(np.where(np.isnan(A), m, A), -1 + 1e-6, 1 - 1e-6))
    A, B = f(Xtr), f(Xte); mu, sd = A.mean(0), A.std(0) + 1e-8; A, B = (A - mu) / sd, (B - mu) / sd
    if k < A.shape[1]:
        a, b = A[ytr == 1], A[ytr == 0]
        t = np.abs(a.mean(0) - b.mean(0)) / np.sqrt(a.var(0) / len(a) + b.var(0) / len(b) + 1e-12); idx = np.argsort(-t)[:k]; A, B = A[:, idx], B[:, idx]
    cm, cs = Ctr.mean(0), Ctr.std(0) + 1e-8
    return np.c_[A, (Ctr - cm) / cs].astype(np.float32), np.c_[B, (Cte - cm) / cs].astype(np.float32)


def fit_predict(A, ya, B, kind, wd, seed, epochs=150):
    torch.manual_seed(seed); np.random.seed(seed)
    d = A.shape[1]
    net = (nn.Linear(d, 1) if kind == "lr" else nn.Sequential(nn.Dropout(0.5), nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    At, yt, Bt = torch.tensor(A, device=dev), torch.tensor(ya, dtype=torch.float32, device=dev), torch.tensor(B, device=dev)
    pw = torch.tensor((ya == 0).sum() / max((ya == 1).sum(), 1), dtype=torch.float32, device=dev); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        net.train(); opt.zero_grad(); lf(net(At).squeeze(1), yt).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(Bt).squeeze(1)).cpu().numpy()


GRID = [(kind, k, wd) for kind in ("lr", "mlp") for k in (500, 2000, 19900) for wd in (1e-2, 1e-1, 1.0)]


def inner_select(tr):
    sk = StratifiedKFold(3, shuffle=True, random_state=42); sc = {g: [] for g in GRID}
    for itr, iva in sk.split(tr, y[tr]):
        a, b = tr[itr], tr[iva]
        for k in (500, 2000, 19900):
            A, B = prep(X_raw[a], X_raw[b], COV[a], COV[b], k, y[a])
            for g in GRID:
                if g[1] == k: sc[g].append(roc_auc_score(y[b], fit_predict(A, y[a], B, g[0], g[2], 0)))
    return max(GRID, key=lambda g: np.mean(sc[g])), {str(g): float(np.mean(v)) for g, v in sc.items()}


def main():
    t0 = time.time(); oof = np.zeros(600); rows = []; sel_log = []
    for fi, (tr, te) in enumerate(folds):
        assert not set(tr) & set(te)
        g, scores = inner_select(tr); kind, k, wd = g
        A, B = prep(X_raw[tr], X_raw[te], COV[tr], COV[te], k, y[tr])
        p = np.mean([fit_predict(A, y[tr], B, kind, wd, s) for s in range(5)], 0); oof[te] = p
        rows.append(dict(fold=fi, selected=str(g), inner_auc=scores[str(g)], acc=accuracy_score(y[te], p > .5), auc=roc_auc_score(y[te], p)))
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    pred = (oof > .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    res = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
               precision=tp / (tp + fp), f1=f1_score(y, pred), roc_auc=roc_auc_score(y, oof), TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn), n=600,
               fold_acc_mean=float(np.mean([r["acc"] for r in rows])), runtime_s=round(time.time() - t0))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame([res]).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(dict(sid=man.sid, y_true=y, score=oof, y_pred=pred, fold=man.outer_fold)).to_csv(OUT / "oof_predictions.csv", index=False)
    json.dump(res, open(OUT / "result.json", "w"), indent=1); print(res)


if __name__ == "__main__":
    main()
