"""Descriptive, saved predictions only (seed 42): association of out-of-fold scores with age, sex and mean FD, overall, within class and within site
(site-mean-removed). No model is trained."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")
sys.path.insert(0, r"C:\abide")
import fast_accuracy_experiment as fx

R = Path(r"C:\abide\results")
OUT = R / "accuracy_research_v3"
man = pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", dtype={"sid": str})
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str}); coh["sid"] = coh.subject_id.map(fx.norm_id)
coh = coh[coh.sid.isin(set(man.sid))].set_index("sid").loc[man.sid]
meta = pd.DataFrame({"project_id": [f"DEV{i:04d}" for i in range(600)], "age": coh.age.values.astype(float), "female": (coh.sex.values == 2).astype(float), "fd": coh.func_mean_fd.values.astype(float)})
bo = pd.read_csv(R / "accuracy_improvement_v1" / "oof_predictions.csv"); bo = bo[(bo.candidate == "E0_control_baseline") & (bo.seed == 42)]
to = pd.read_csv(R / "accuracy_research_v2" / "oof_predictions_tangent.csv"); to = to[to.seed == 42]
rows = []
for name, d in (("baseline", bo), ("tangent", to)):
    d = d.merge(meta, on="project_id")
    for v in ("age", "female", "fd"):
        dc = d.assign(score_c=d.score - d.groupby("site").score.transform("mean"), v_c=d[v] - d.groupby("site")[v].transform("mean"))
        for scope, g, a, b in (("all", d, d.score, d[v]), ("ASD only", d[d.y_true == 1], d[d.y_true == 1].score, d[d.y_true == 1][v]), ("TC only", d[d.y_true == 0], d[d.y_true == 0].score, d[d.y_true == 0][v]),
                               ("within-site (site means removed)", dc, dc.score_c, dc.v_c)):
            r = stats.spearmanr(a, b)
            rows.append({"model": name, "variable": v, "scope": scope, "spearman_rho_score_vs_variable": r.correlation, "p": r.pvalue, "n": len(a)})
t = pd.DataFrame(rows); t.to_csv(OUT / "score_covariate_association.csv", index=False)
print(t.pivot_table(index=["model", "variable"], columns="scope", values="spearman_rho_score_vs_variable").round(3).to_string())
print("\np-values (all):\n", t[t.scope == "all"][["model", "variable", "p"]].round(4).to_string(index=False))
