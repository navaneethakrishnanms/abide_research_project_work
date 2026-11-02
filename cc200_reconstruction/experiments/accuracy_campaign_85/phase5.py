"""Phase 5 (conditional ensembles) exactly as predeclared in campaign_plan.md section 3.
Eligibility: mean R1 accuracy >= reference - 1.0 pt. Condition: at least one eligible pair with error-indicator phi <= 0.70 AND disagreement >= 8% (R1 predictions pooled over seeds 0-2).
If met: E1 soft vote and E2 stacking of the top-3 eligible models by mean R1 accuracy, built ONLY from training-only inner out-of-fold scores (stored in the checkpoints).
Reads saved results only; no holdout or restricted file; no model is refitted on subjects it has seen."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign as CP

R = HERE / "results"; SEEDS = (0, 1, 2); CANDS = ["P1_LR", "P2_LSVM", "P2_RBF", "P2_RF", "P2_ET", "P2_XGB", "P3_PC_LR", "P3_TS_LR", "P4_PCA_LR", "P4_KBEST_LR", "P4_L1_LR"]
pc = lambda v: f"{v * 100:.2f}%"


def main():
    Xa, y, site_s, sid, covs, Xp = CP.load(); n = len(y); ycat = np.concatenate([y] * 3)
    acc, pred = {}, {}
    for e in CANDS:
        sm = pd.read_csv(R / f"{e}_seed_metrics.csv"); acc[e] = sm[sm.reading == "R1_nested_threshold"].accuracy.mean()
        o = pd.read_csv(R / f"{e}_oof_predictions_seeds012.csv", dtype={"subject_id": str})
        pred[e] = np.concatenate([o[o.seed == s].set_index("subject_id").loc[sid].pred_R1.values for s in SEEDS])
    ref = acc["P1_LR"]; elig = [e for e in CANDS if acc[e] >= ref - 0.01]
    rows = []
    for a, b in itertools.combinations(elig, 2):
        ea, eb = (pred[a] != ycat).astype(float), (pred[b] != ycat).astype(float)
        rows.append(dict(a=a, b=b, error_phi=float(np.corrcoef(ea, eb)[0, 1]), disagreement=float((pred[a] != pred[b]).mean()), both_wrong=float(((ea == 1) & (eb == 1)).mean()), oracle_acc=float(1 - ((ea == 1) & (eb == 1)).mean())))
    P = pd.DataFrame(rows).sort_values("error_phi"); P.to_csv(R / "phase5_pairwise_complementarity.csv", index=False)
    met = bool(((P.error_phi <= 0.70) & (P.disagreement >= 0.08)).any())
    ptxt = "\n".join(f"| {r.a} | {r.b} | {r.error_phi:.3f} | {pc(r.disagreement)} | {pc(r.both_wrong)} | {pc(r.oracle_acc)} |" for r in P.itertuples())
    head = f"""# Phase 5 report: ensembles (conditional, predeclared)
Eligible models (mean R1 accuracy >= reference {pc(ref)} minus 1.0 pt): {', '.join(f'{e} ({pc(acc[e])})' for e in elig)}.
Criterion: at least one eligible pair with error phi <= 0.70 and disagreement >= 8% (R1 predictions pooled over seeds 0-2, 2,028 predictions per model).
| Model A | Model B | Error phi | Disagreement | Both wrong | Oracle accuracy (either right) |
|---|---|---|---|---|---|
{ptxt}
Criterion met: **{met}** (smallest error phi {P.error_phi.min():.3f}, largest disagreement {pc(P.disagreement.max())}).
"""
    if not met:
        (HERE / "phase5_report.md").write_text(head + "\n**Outcome: the complementarity condition is not met, so no ensemble was built (as predeclared).** Stacking or voting models whose errors are this correlated cannot add information.\n", encoding="utf-8"); print(head); return
    top3 = sorted(elig, key=lambda e: -acc[e])[:3]; out = {}
    ck = {e: {s: np.load(HERE / "checkpoints" / f"{e}_seed{s}.npz") for s in SEEDS} for e in top3}
    for name in ("P5_E1_SOFTVOTE", "P5_E2_STACK"):
        srows, frows, oof = [], [], []
        for s in SEEDS:
            score, fold, thr = np.zeros(n), np.zeros(n, int), np.zeros(10)
            for f, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=s).split(Xa, y)):
                assert all(np.array_equal(np.where(~np.isnan(ck[e][s]["inner_oof"][f]))[0], tr) for e in top3), "inner-OOF training rows differ from the outer-training rows"
                Zin = np.column_stack([ck[e][s]["inner_oof"][f, tr] for e in top3]); Zte = np.column_stack([ck[e][s]["score"][te] for e in top3])
                if name.endswith("SOFTVOTE"):
                    cal = [LogisticRegression(C=1e6).fit(Zin[:, j:j + 1], y[tr]) for j in range(len(top3))]                      # Platt calibrators on inner OOF scores (training rows only)
                    sin = np.mean([c.predict_proba(Zin[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0); ste = np.mean([c.predict_proba(Zte[:, j:j + 1])[:, 1] for j, c in enumerate(cal)], axis=0)
                else:
                    meta = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced")); sin = cross_val_predict(meta, Zin, y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=s + 2000), method="predict_proba")[:, 1]
                    ste = meta.fit(Zin, y[tr]).predict_proba(Zte)[:, 1]
                thr[f] = CP.choose_threshold(y[tr], sin, 0.5); score[te], fold[te] = ste, f
            p0 = (score >= 0.5).astype(int); p1 = (score >= thr[fold]).astype(int)
            for rd, p in (("R0_default", p0), ("R1_nested_threshold", p1)):
                m = CP.metrics(y, p, score); fa = [accuracy_score(y[fold == f], p[fold == f]) for f in range(10)]
                srows.append(dict(experiment=name, seed=s, reading=rd, **m, fold_accuracy_mean=np.mean(fa), fold_accuracy_sd=np.std(fa, ddof=1), mean_fold_auc=np.mean([roc_auc_score(y[fold == f], score[fold == f]) for f in range(10)]), runtime_s=0.0))
            frows += [dict(experiment=name, seed=s, fold=f, n_test=int((fold == f).sum()), acc_R0=accuracy_score(y[fold == f], p0[fold == f]), acc_R1=accuracy_score(y[fold == f], p1[fold == f]), thr_R1=float(thr[f]),
                           fold_auc=roc_auc_score(y[fold == f], score[fold == f]), inner_auc=np.nan, params=f"members={top3}", seconds=0.0) for f in range(10)]
            oof += [dict(experiment=name, seed=s, fold=int(fold[i]), subject_id=sid[i], site=site_s[i], y_true=int(y[i]), score=float(score[i]), pred_R0=int(p0[i]), pred_R1=int(p1[i])) for i in range(n)]
        pd.DataFrame(srows).to_csv(R / f"{name}_seed_metrics.csv", index=False); pd.DataFrame(frows).to_csv(R / f"{name}_fold_metrics.csv", index=False); pd.DataFrame(oof).to_csv(R / f"{name}_oof_predictions_seeds012.csv", index=False)
        json.dump({"experiment": name, "members": top3, "construction": "training-only inner OOF scores; Platt calibration (E1) / balanced L2 LR meta-model (E2); R1 threshold from the same inner OOF scores"}, open(R / f"{name}_config.json", "w"), indent=1)
        out[name] = pd.DataFrame(srows)
    (HERE / "phase5_report.md").write_text(head + f"\nEnsembles built from {top3}. Metrics and screening: see `phase5_comparison.csv` / `phase5_ensemble_report.md` (generated by report_phase.py).\n", encoding="utf-8"); print(head)


if __name__ == "__main__":
    main()
