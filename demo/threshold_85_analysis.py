"""Numerical analysis of SAVED AAL-116 out-of-fold predictions (600 development subjects). No training, no holdout, no new scores.
For every prediction group:
  documented  : accuracy etc. from the stored y_pred (the experiment's own threshold)
  cross-fitted: the threshold for group g (outer fold, or site for leave-site-out files) is chosen ONLY from the predictions of the OTHER
                groups (maximize balanced accuracy), then applied to g. Subjects in g never influence their own threshold.
  oracle      : best accuracy over all thresholds on the same predictions. OPTIMISTIC UPPER BOUND, not an estimate.
Caveat: scores of the other groups came from models that had seen group g's subjects in training, so the cross-fitted number is still
slightly optimistic (documented, not removable without retraining)."""
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

R = r"C:\abide\results"
OUT = r"C:\abide\demo"


def metrics(y, pred, score):
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1); f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=(sens + spec) / 2, sensitivity=sens, specificity=spec, f1=f1, auc=roc_auc_score(y, score))


def best_thr(y, s):
    """threshold maximizing balanced accuracy on (y, s); candidates = midpoints between sorted unique scores"""
    u = np.unique(s); cand = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])
    best, bt = -1, 0.5
    for t in cand:
        p = (s >= t).astype(int)
        b = (p[y == 1].mean() + (1 - p[y == 0].mean())) / 2
        if b > best + 1e-12 or (abs(b - best) <= 1e-12 and abs(t) < abs(bt)):
            best, bt = b, t
    return bt


def oracle_acc(y, s):
    u = np.unique(s); cand = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])
    return max(((s >= t).astype(int) == y).mean() for t in cand)


def groups():
    def ld(rel, **kw):
        return pd.read_csv(os.path.join(R, rel), **kw)
    d = ld(r"accuracy_improvement_v1\oof_predictions.csv")
    for (c, s), g in d.groupby(["candidate", "seed"]): yield f"accuracy_improvement_v1 | {c} | seed {s}", g, "fold", "accuracy_improvement_v1/oof_predictions.csv"
    d = ld(r"accuracy_improvement_v1\leave_site_out_oof_predictions.csv")
    for c, g in d.groupby("candidate"): yield f"accuracy_improvement_v1 LSO | {c}", g, "site", "accuracy_improvement_v1/leave_site_out_oof_predictions.csv"
    d = ld(r"accuracy_research_v2\oof_predictions_tangent.csv")
    for s, g in d.groupby("seed"): yield f"accuracy_research_v2 tangent | seed {s}", g, "fold", "accuracy_research_v2/oof_predictions_tangent.csv"
    yield "accuracy_research_v2 tangent LSO", ld(r"accuracy_research_v2\leave_site_out_oof_predictions_tangent.csv"), "site", "accuracy_research_v2/leave_site_out_oof_predictions_tangent.csv"
    d = ld(r"accuracy_research_v3\oof_predictions.csv")
    for (a, s), g in d.groupby(["arm", "seed"]):
        if a.startswith(("C_", "D_")): yield f"accuracy_research_v3 | {a} | seed {s}", g, "fold", "accuracy_research_v3/oof_predictions.csv"
    d = ld(r"cerebrum_only_experiment\oof_predictions.csv")
    for (r, s), g in d.groupby(["representation", "seed"]): yield f"cerebrum_only | {r} | seed {s}", g, "fold", "cerebrum_only_experiment/oof_predictions.csv"
    d = ld(r"fast_accuracy_experiment\oof_predictions.csv"); d = d[d.atlas == "AAL116"]
    for m, g in d.groupby("model"): yield f"fast_accuracy_experiment AAL116 | {m} | seed 42", g, "fold", "fast_accuracy_experiment/oof_predictions.csv"
    d = ld(r"aal116_dynamic_fast\oof_predictions_dynamic.csv")
    for m, g in d.groupby("method"): yield f"aal116_dynamic_fast | {m}", g, "fold", "aal116_dynamic_fast/oof_predictions_dynamic.csv"
    d = ld(r"advanced_model_search\oof_predictions.csv")
    for m, g in d.groupby("method"): yield f"advanced_model_search | {m}", g, "fold", "advanced_model_search/oof_predictions.csv"
    d = ld(r"leave_site_out_baseline\oof_predictions.csv"); yield "leave_site_out_baseline", d, "site", "leave_site_out_baseline/oof_predictions.csv"
    d = ld(r"test_site_centering_experiment\five_fold_control_oof_predictions.csv"); yield "test_site_centering five-fold control", d, "fold", "test_site_centering_experiment/five_fold_control_oof_predictions.csv"
    d = ld(r"test_site_centering_experiment\oof_predictions.csv")
    for tag in ("centered", "uncentered_same_pipeline", "frozen_baseline"):
        g = d.rename(columns={f"score_{tag}" if tag != "centered" else "score_centered": "score", f"pred_{tag}" if tag != "centered" else "pred_centered": "y_pred"})
        yield f"test_site_centering LSO | {tag}", g, "site", "test_site_centering_experiment/oof_predictions.csv"


rows = []
for name, g, gcol, src in groups():
    g = g.reset_index(drop=True); y = g.y_true.values.astype(int); s = g.score.values.astype(float)
    assert len(g) == 600 and g.groupby(gcol).size().sum() == 600, (name, len(g))
    id_col = "project_id" if "project_id" in g else "subject_id"
    assert g[id_col].is_unique, name                   # exactly one prediction per subject
    doc = metrics(y, g.y_pred.values.astype(int), s)
    cf = np.zeros(600, int); thr_used = []
    for k in g[gcol].unique():
        te = (g[gcol] == k).values
        t = best_thr(y[~te], s[~te]); thr_used.append(t); cf[te] = (s[te] >= t).astype(int)
    cfm = metrics(y, cf, s)
    rows.append(dict(group=name, source=src, n=600, fold_unit=gcol, n_units=len(thr_used),
                     **{f"doc_{k}": v for k, v in doc.items()}, **{f"cf_{k}": v for k, v in cfm.items() if k != "auc"},
                     cf_thr_min=min(thr_used), cf_thr_max=max(thr_used), oracle_accuracy_OPTIMISTIC=oracle_acc(y, s)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "threshold_85_analysis_results.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df[["group", "doc_accuracy", "doc_balanced_accuracy", "doc_auc", "cf_accuracy", "cf_balanced_accuracy", "oracle_accuracy_OPTIMISTIC"]].round(3).to_string(index=False))
print("\nmax documented accuracy: %.4f | max cross-fitted accuracy: %.4f | max oracle (optimistic): %.4f | max AUC: %.4f"
      % (df.doc_accuracy.max(), df.cf_accuracy.max(), df.oracle_accuracy_OPTIMISTIC.max(), df.doc_auc.max()))
print("groups with doc/cross-fitted accuracy >= 0.85:", int((df.doc_accuracy >= .85).sum()), int((df.cf_accuracy >= .85).sum()))

# ILLUSTRATIVE ONLY (binormal, equal-variance assumption; not an experimental result)
z = stats.norm.ppf
auc_needed = stats.norm.cdf((2 * z(0.85)) / np.sqrt(2))
auc_best = df.doc_auc.max(); acc_at_best = stats.norm.cdf(np.sqrt(2) * z(auc_best) / 2)
print("\nILLUSTRATIVE (binormal, equal variance, balanced classes): 85%% accuracy needs AUC ~ %.3f; best observed AUC %.3f implies ceiling ~ %.3f"
      % (auc_needed, auc_best, acc_at_best))
pd.Series(dict(auc_needed_for_85=auc_needed, best_observed_auc=auc_best, binormal_accuracy_ceiling_at_best_auc=acc_at_best)).to_csv(os.path.join(OUT, "threshold_85_illustration.csv"))
