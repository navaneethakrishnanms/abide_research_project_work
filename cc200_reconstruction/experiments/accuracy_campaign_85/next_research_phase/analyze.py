"""Computes per-arm metrics, paired comparison with the reference ('none') on identical subjects/folds, frozen-criteria decisions.
Usage: analyze.py STAGE ARM1,ARM2 SEEDS   -> appends to experiment_comparison.csv / predictions.csv (rewritten from checkpoints each call)."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import experiments as E
from sklearn.metrics import roc_auc_score
X, y, site, sid, fd = E.load(); CK = E.CK
CRIT = dict(gain=0.5, pos=2, auc=-0.005)          # frozen in research_plan.md Addendum A


def load_arm(arm, seed):
    z = np.load(CK / f"{arm}_seed{seed}.npz"); sc, fold, thr = z["score"], z["fold"], z["thr"]; m = np.isfinite(sc)
    return sc, fold, thr, m, (sc >= thr[fold]).astype(int), (sc >= 0.5).astype(int)


def met(m, p1, p0, sc):
    yy = y[m]; a = lambda p: float((p[m] == yy).mean())
    P, N = yy.sum(), (1 - yy).sum(); tp = int(((p1[m] == 1) & (yy == 1)).sum()); tn = int(((p1[m] == 0) & (yy == 0)).sum())
    return dict(n=int(m.sum()), acc_R1=a(p1), acc_R0=a(p0), bacc_R1=(tp / P + tn / N) / 2, sens_R1=tp / P, spec_R1=tn / N, prec_R1=tp / max(1, tp + int(((p1[m] == 1) & (yy == 0)).sum())),
                f1_R1=2 * tp / max(1, 2 * tp + int(((p1[m] == 1) & (yy == 0)).sum()) + int(((p1[m] == 0) & (yy == 1)).sum())), auc=float(roc_auc_score(yy, sc[m])), TP=tp, FN=int(P - tp), FP=int(N - tn), TN=tn)


def run(stage, arms, seeds, nonin=()):
    rows, preds = [], []
    for arm in arms:
        gains = []
        for s in seeds:
            sc, fold, thr, m, p1, p0 = load_arm(arm, s); rsc, _, _, rm, rp1, rp0 = load_arm("none", s); assert (rm | True).all()
            mm = m                                                    # evaluation set = subjects the arm scored (all 676, or the retained subjects for qc)
            a, r = met(mm, p1, p0, sc), met(mm, rp1, rp0, rsc)       # reference restricted to the same subjects, same folds
            rows.append(dict(stage=stage, arm=arm, seed=s, **{k: a[k] for k in a}, ref_acc_R1=r["acc_R1"], ref_auc=r["auc"], ref_acc_R0=r["acc_R0"], gain_acc_R1_pts=100 * (a["acc_R1"] - r["acc_R1"]), gain_acc_R0_pts=100 * (a["acc_R0"] - r["acc_R0"]), delta_auc=a["auc"] - r["auc"],
                             fold_wins=int(sum((p1[mm & (fold == f)] == y[mm & (fold == f)]).mean() > (rp1[mm & (fold == f)] == y[mm & (fold == f)]).mean() for f in range(10))), fold_losses=int(sum((p1[mm & (fold == f)] == y[mm & (fold == f)]).mean() < (rp1[mm & (fold == f)] == y[mm & (fold == f)]).mean() for f in range(10)))))
            preds += [dict(stage=stage, arm=arm, seed=s, subject_id=sid[i], site=site[i], y_true=int(y[i]), fold=int(fold[i]), score=float(sc[i]), pred_R1=int(p1[i]), pred_R0=int(p0[i])) for i in np.where(mm)[0]]
    return pd.DataFrame(rows), pd.DataFrame(preds)


def decide(df, arm, nonin=False):
    d = df[df.arm == arm]; g, au = d.gain_acc_R1_pts.mean(), d.delta_auc.mean(); pos = int((d.gain_acc_R1_pts > 0).sum())
    if nonin: ok = (g >= -0.5) and (au >= CRIT["auc"]); return dict(arm=arm, rule="non-inferiority", mean_gain=g, seeds_positive=pos, mean_delta_auc=au, passed=bool(ok))
    ok = g >= CRIT["gain"] and pos >= 2 and au >= CRIT["auc"]; return dict(arm=arm, rule="superiority", mean_gain=g, seeds_positive=pos, mean_delta_auc=au, passed=bool(ok))


if __name__ == "__main__":
    stage, arms, seeds = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")]
    df, pr = run(stage, arms, seeds); pd.set_option("display.width", 250)
    print(df[["arm", "seed", "n", "acc_R1", "ref_acc_R1", "gain_acc_R1_pts", "acc_R0", "ref_acc_R0", "auc", "ref_auc", "delta_auc", "fold_wins", "fold_losses"]].round(4).to_string())
    print(df.groupby("arm")[["acc_R1", "ref_acc_R1", "gain_acc_R1_pts", "gain_acc_R0_pts", "auc", "delta_auc"]].mean().round(4).to_string())
    out = HERE / f"stage_{stage}.csv"; df.to_csv(out, index=False); pr.to_csv(HERE / f"predictions_{stage}.csv", index=False)
    print([decide(df, a, nonin=(a == "fdres")) for a in arms])
