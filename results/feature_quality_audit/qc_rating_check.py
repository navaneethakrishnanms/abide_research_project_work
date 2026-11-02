"""Read-only: relation of the rater-2 'maybe' flag (development subjects only) to undefined-correlation subjects, site and diagnosis."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
sys.path.insert(0, r"C:\abide")
import fast_accuracy_experiment as fx
R = Path(r"C:\abide\results")
man = pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", dtype={"sid": str})
ids = set(man.sid)
ph = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv", dtype={"SUB_ID": str}, low_memory=False)
ph["sid"] = ph.SUB_ID.map(fx.norm_id); ph = ph[ph.sid.isin(ids)].set_index("sid").loc[man.sid]
X = np.load(R / "fast_accuracy_experiment" / "data" / "fc_raw_AAL116.npy")
d = pd.DataFrame({"site": man.site.values, "y": man.y.values, "maybe": (ph.qc_func_rater_2.values == "maybe"), "has_nan": np.isnan(X).any(1), "note": ph.qc_func_notes_rater_2.fillna("").values})
t = pd.crosstab(d.maybe, d.has_nan)
print("rater-2 'maybe' x subjects with undefined correlations:\n", t)
print("Fisher exact p:", stats.fisher_exact(t.values)[1])
print("NaN subjects flagged 'maybe': %d of %d; 'maybe' subjects with NaN: %d of %d" % (t.loc[True, True], t[True].sum(), t.loc[True, True], t.loc[True].sum()))
print("\nshare 'maybe' by diagnosis: ASD %.3f TC %.3f; chi2 p=%.3f" % (d[d.y == 1].maybe.mean(), d[d.y == 0].maybe.mean(), stats.chi2_contingency(pd.crosstab(d.y, d.maybe))[1]))
s = d.groupby("site").agg(n=("maybe", "size"), maybe=("maybe", "sum"), nan_subj=("has_nan", "sum")); s["maybe_share"] = (s.maybe / s.n).round(2)
print("\nby site:\n", s.sort_values("n", ascending=False).to_string())
print("\nnote categories among 'maybe':\n", d[d.maybe].note.value_counts().head(6).to_string())
s.sort_values("n", ascending=False).rename(columns={"maybe": "rater2_maybe", "nan_subj": "subjects_with_undefined_edges"}).to_csv(R / "feature_quality_audit" / "qc_rater2_flag_by_site.csv")
