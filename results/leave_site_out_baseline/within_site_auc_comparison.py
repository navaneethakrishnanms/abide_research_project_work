"""Descriptive, read-only: within-site AUC of the saved five-fold OOF scores vs the leave-site-out scores (same subjects, same pipeline).
No model is trained; project IDs are row order of the development manifest."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
R = Path(r"C:\abide\results")
man = pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", dtype={"sid": str})
f5 = pd.read_csv(R / "fast_accuracy_experiment" / "oof_predictions.csv", dtype={"subject_id": str})
f5 = f5[(f5.atlas == "AAL116") & (f5.model == "LR_L2")].set_index("subject_id").loc[man.sid].reset_index()
f5["project_id"] = [f"DEV{i:04d}" for i in range(600)]
lso = pd.read_csv(R / "leave_site_out_baseline" / "oof_predictions.csv")
d = lso.merge(f5[["project_id", "score", "y_pred"]].rename(columns={"score": "score_5", "y_pred": "pred_5"}), on="project_id")
rows = []
for s, g in d.groupby("site"):
    if len(g) < 20 or g.y_true.nunique() < 2:
        continue
    rows.append({"site": s, "n": len(g), "within_site_auc_5fold": roc_auc_score(g.y_true, g.score_5), "within_site_auc_leave_site_out": roc_auc_score(g.y_true, g.score),
                 "pred_ASD_share_5fold": g.pred_5.mean(), "pred_ASD_share_lso": g.y_pred.mean(), "true_ASD_share": g.y_true.mean(),
                 "mean_score_5fold": g.score_5.mean(), "mean_score_lso": g.score.mean()})
t = pd.DataFrame(rows); t["auc_diff"] = t.within_site_auc_leave_site_out - t.within_site_auc_5fold
t["abs_gap_predicted_vs_true_ASD_share_5fold"] = (t.pred_ASD_share_5fold - t.true_ASD_share).abs()
t["abs_gap_predicted_vs_true_ASD_share_lso"] = (t.pred_ASD_share_lso - t.true_ASD_share).abs()
t.to_csv(R / "leave_site_out_baseline" / "within_site_auc_and_prediction_share.csv", index=False)
print(t.round(3).to_string(index=False))
print("\nmean within-site AUC (sites n>=20): 5-fold %.4f | leave-site-out %.4f | mean diff %.4f; sites where LSO AUC lower: %d of %d" % (t.within_site_auc_5fold.mean(), t.within_site_auc_leave_site_out.mean(), t.auc_diff.mean(), (t.auc_diff < 0).sum(), len(t)))
print("mean |predicted ASD share - true ASD share| per site: 5-fold %.3f | LSO %.3f" % (t.abs_gap_predicted_vs_true_ASD_share_5fold.mean(), t.abs_gap_predicted_vs_true_ASD_share_lso.mean()))
