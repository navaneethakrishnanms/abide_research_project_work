"""CC200 + in-fold ComBat harmonization + ensemble (torch GPU). Leakage-safe: ComBat (label-free), imputation, scaling, ANOVA top-k, hyper-parameters,
ensemble membership are all learned from TRAINING rows of each fold only; hyper-parameters chosen by inner stratified 3-fold ROC-AUC.
Outer folds = saved manifest (600 dev subjects). Holdout never opened."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, f1_score

sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\abide\results\combat_in_fold_v1")
from combat_harmonizer import HarmFZ  # noqa: E402

ROOT = Path(r"C:\abide"); D = ROOT / "results" / "fast_accuracy_experiment" / "data"; OUT = ROOT / "results" / "torch_gpu_cc200_combat_v1"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
man = pd.read_csv(D / "dev_split_manifest.csv", dtype={"sid": str})
X_raw = np.load(D / "fc_raw_CC200.npy"); y = man.y.values.astype(int); assert X_raw.shape[0] == 600
ph = pd.read_csv(ROOT / "Phenotypic_V1_0b.csv", low_memory=False); ph["sid"] = ph.SUB_ID.astype(float).astype(int).astype(str)
cov = man[["sid"]].merge(ph[["sid", "AGE_AT_SCAN", "SEX"]], on="sid", how="left")
site_code = man.site.astype("category").cat.codes.values.astype(float)
H_IN = np.c_[X_raw, site_code, cov.AGE_AT_SCAN.values, (cov.SEX.values == 1).astype(float)]   # [FC | site | age | female]
COV = np.c_[cov.AGE_AT_SCAN.values, (cov.SEX.values == 1).astype(float)]
folds = [(np.where(man.outer_fold != k)[0], np.where(man.outer_fold == k)[0]) for k in range(5)]
NF = X_raw.shape[1]


def harmonize(tr, te):
    h = HarmFZ().fit(H_IN[tr]); A, B = h.transform(H_IN[tr]), h.transform(H_IN[te])
    mu, sd = A.mean(0), A.std(0) + 1e-8
    return (A - mu) / sd, (B - mu) / sd


def select(A, B, ytr, Ctr, Cte, k):
    if k < A.shape[1]:
        a, b = A[ytr == 1], A[ytr == 0]
        t = np.abs(a.mean(0) - b.mean(0)) / np.sqrt(a.var(0) / len(a) + b.var(0) / len(b) + 1e-12); i = np.argsort(-t)[:k]; A, B = A[:, i], B[:, i]
    cm, cs = Ctr.mean(0), Ctr.std(0) + 1e-8
    return np.c_[A, (Ctr - cm) / cs].astype(np.float32), np.c_[B, (Cte - cm) / cs].astype(np.float32)


def fit_predict(A, ya, B, kind, wd, seed, epochs=150):
    torch.manual_seed(seed)
    d = A.shape[1]
    net = (nn.Linear(d, 1) if kind == "lr" else nn.Sequential(nn.Dropout(0.5), nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    At, yt, Bt = torch.tensor(A, device=dev), torch.tensor(ya, dtype=torch.float32, device=dev), torch.tensor(B, device=dev)
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ya == 0).sum() / max((ya == 1).sum(), 1), dtype=torch.float32, device=dev))
    for _ in range(epochs):
        net.train(); opt.zero_grad(); lf(net(At).squeeze(1), yt).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(Bt).squeeze(1)).cpu().numpy()


KS = (2000, NF)
GRID = [(kind, k, wd) for kind in ("lr", "mlp") for k in KS for wd in (1e-2, 1e-1, 1.0)]
TOPN = 3


def inner_rank(tr):
    sc = {g: [] for g in GRID}
    for itr, iva in StratifiedKFold(3, shuffle=True, random_state=42).split(tr, y[tr]):
        a, b = tr[itr], tr[iva]; A0, B0 = harmonize(a, b)
        for k in KS:
            A, B = select(A0, B0, y[a], COV[a], COV[b], k)
            for g in GRID:
                if g[1] == k: sc[g].append(roc_auc_score(y[b], fit_predict(A, y[a], B, g[0], g[2], 0)))
    return sorted(GRID, key=lambda g: -np.mean(sc[g])), {str(g): float(np.mean(v)) for g, v in sc.items()}


def main():
    t0 = time.time(); oof = np.zeros(600); rows = []
    for fi, (tr, te) in enumerate(folds):
        assert not set(tr) & set(te)
        rank, scores = inner_rank(tr); top = rank[:TOPN]; A0, B0 = harmonize(tr, te); ps = []
        for g in top:
            A, B = select(A0, B0, y[tr], COV[tr], COV[te], g[1])
            ps += [fit_predict(A, y[tr], B, g[0], g[2], s) for s in range(5)]
        p = np.mean(ps, 0); oof[te] = p
        rows.append(dict(fold=fi, ensemble=str([str(g) for g in top]), acc=accuracy_score(y[te], p > .5), auc=roc_auc_score(y[te], p)))
        print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    pred = (oof > .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    res = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
               precision=tp / (tp + fp), f1=f1_score(y, pred), roc_auc=roc_auc_score(y, oof), TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn), n=600, runtime_s=round(time.time() - t0))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv", index=False); pd.DataFrame([res]).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(dict(sid=man.sid, y_true=y, score=oof, y_pred=pred, fold=man.outer_fold)).to_csv(OUT / "oof_predictions.csv", index=False)
    print(res)


if __name__ == "__main__":
    main()
