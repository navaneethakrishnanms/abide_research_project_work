"""Read-only audit of age and sex for the 600 DEVELOPMENT subjects (two independent sources, alignment, missingness, distributions,
association with diagnosis overall and within sites). No model is trained. Restricted data, CC200 and the 780 cohort are not used;
non-development rows of the phenotype files are discarded immediately after reading."""
import json
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
ids = set(man.sid)
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str}); coh["sid"] = coh.subject_id.map(fx.norm_id)
coh = coh[coh.sid.isin(ids)].set_index("sid").loc[man.sid]
ph = pd.read_csv(r"C:\abide\Phenotypic_V1_0b.csv", dtype={"SUB_ID": str}, low_memory=False); ph["sid"] = ph.SUB_ID.map(fx.norm_id)
ph = ph[ph.sid.isin(ids)].set_index("sid").loc[man.sid]
facts = {}
facts["n"] = len(man)
facts["age_missing_cohort_csv"] = int(coh.age.isna().sum()); facts["sex_missing_cohort_csv"] = int(coh.sex.isna().sum())
age_ph = pd.to_numeric(ph.AGE_AT_SCAN, errors="coerce"); sex_ph = pd.to_numeric(ph.SEX, errors="coerce")
facts["age_missing_or_-9999_phenotype"] = int((age_ph.isna() | (age_ph < 0)).sum()); facts["sex_missing_phenotype"] = int(sex_ph.isna().sum())
facts["age_sources_equal_max_abs_diff"] = float(np.abs(age_ph.values - coh.age.values).max())
facts["sex_sources_equal_all"] = bool((sex_ph.values == coh.sex.values).all())
facts["sex_values"] = sorted(int(v) for v in coh.sex.unique())
facts["age_range"] = [float(coh.age.min()), float(coh.age.max())]
facts["dx_from_phenotype_matches_manifest"] = bool(((pd.to_numeric(ph.DX_GROUP) == 1).astype(int).values == man.y.values).all())
d = pd.DataFrame({"site": man.site.values, "y": man.y.values, "age": coh.age.values.astype(float), "female": (coh.sex.values == 2).astype(int)})
for lab, g in (("ASD", d[d.y == 1]), ("TC", d[d.y == 0])):
    facts[f"age_{lab}_mean_sd_median"] = [float(g.age.mean()), float(g.age.std()), float(g.age.median())]; facts[f"female_pct_{lab}"] = float(100 * g.female.mean())
facts["age_dx_mannwhitney_p"] = float(stats.mannwhitneyu(d[d.y == 1].age, d[d.y == 0].age).pvalue)
facts["sex_dx_chi2_p"] = float(stats.chi2_contingency(pd.crosstab(d.y, d.female))[1])
# Mantel-Haenszel common odds ratio for female vs diagnosis, stratified by site (sites with both cells informative)
num = den = 0.0
for s, g in d.groupby("site"):
    a = ((g.y == 1) & (g.female == 1)).sum(); b = ((g.y == 1) & (g.female == 0)).sum(); c = ((g.y == 0) & (g.female == 1)).sum(); dd = ((g.y == 0) & (g.female == 0)).sum(); n = len(g)
    num += a * dd / n; den += b * c / n
facts["female_vs_ASD_MH_common_odds_ratio_within_sites"] = float(num / den) if den else None
facts["female_vs_ASD_crude_odds_ratio"] = float(((d.y == 1) & (d.female == 1)).sum() * ((d.y == 0) & (d.female == 0)).sum() / (((d.y == 1) & (d.female == 0)).sum() * ((d.y == 0) & (d.female == 1)).sum()))
# within-site age difference (ASD minus TC), weighted by n; Spearman of age with site-mean-removed labels
dd2 = d.copy(); dd2["age_c"] = dd2.age - dd2.groupby("site").age.transform("mean")
facts["age_within_site_AUC_equivalent_ASD_vs_TC"] = float(stats.mannwhitneyu(dd2[dd2.y == 1].age_c, dd2[dd2.y == 0].age_c).statistic / ((dd2.y == 1).sum() * (dd2.y == 0).sum()))
facts["age_within_site_p"] = float(stats.mannwhitneyu(dd2[dd2.y == 1].age_c, dd2[dd2.y == 0].age_c).pvalue)
# how much of age/sex is explained by site
facts["age_variance_share_explained_by_site_eta2"] = float(1 - ((dd2.age_c) ** 2).sum() / ((d.age - d.age.mean()) ** 2).sum())
ct = pd.crosstab(d.site, d.female); facts["female_by_site_chi2_p"] = float(stats.chi2_contingency(ct)[1])
site_tab = d.groupby("site").agg(n=("y", "size"), asd_pct=("y", lambda s: 100 * s.mean()), age_mean=("age", "mean"), age_min=("age", "min"), age_max=("age", "max"), female_pct=("female", lambda s: 100 * s.mean()))
site_tab["small_site_n_lt_20"] = site_tab.n < 20
site_tab.sort_values("n", ascending=False).round(2).to_csv(OUT / "demographics_by_site.csv")
json.dump(facts, open(OUT / "demographics_audit_facts.json", "w"), indent=1)
print(json.dumps(facts, indent=1))
print(site_tab.sort_values("n", ascending=False).round(1).to_string())
