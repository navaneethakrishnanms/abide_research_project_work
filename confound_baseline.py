"""
confound_baseline.py - How much of ABIDE is predictable without any brain data?

Full ABIDE has real group differences in FIQ (d = -0.35, p < 1e-4) and sex
(p = 0.02), and the ASD/TC ratio varies across sites by 34.5 percentage points.
A classifier can score on all three without learning anything neurological.

This script establishes the number a connectivity model has to beat. If
functional connectivity lands near the confounds-only score, the imaging is not
contributing - and that is worth knowing before claiming a brain-based result.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import evaluation as ev

warnings.filterwarnings("ignore")

DATA = r"C:\abide\abide_pcp"


def main():
    ph = pd.read_csv(os.path.join(DATA, "phenotypic_qc.csv"))
    ts = np.load(os.path.join(DATA, "ts_cc200.npy"), allow_pickle=True)

    keep = np.array([t is not None and t.shape[0] >= 100 for t in ts])
    ph = ph.loc[keep].reset_index(drop=True)
    y = (ph["DX_GROUP"].values == 1).astype(int)

    num = lambda c: pd.to_numeric(ph.get(c), errors="coerce").replace(-9999, np.nan)
    age, fiq, sex = num("AGE_AT_SCAN"), num("FIQ"), num("SEX")
    site = pd.get_dummies(ph["SITE_ID"].astype(str), prefix="site").astype(float)

    # Median-impute FIQ so missingness doesn't silently drop subjects.
    fiq = fiq.fillna(fiq.median())
    age = age.fillna(age.median())
    sex = sex.fillna(sex.median())

    blocks = {
        "FIQ only": fiq.values.reshape(-1, 1),
        "sex only": sex.values.reshape(-1, 1),
        "age only": age.values.reshape(-1, 1),
        "site only": site.values,
        "FIQ + sex + age": np.column_stack([fiq, sex, age]),
        "FIQ + sex + age + site": np.column_stack([fiq, sex, age, site.values]),
    }

    print(f"N = {len(y)} | ASD {int(y.sum())} | TC {int((1-y).sum())} "
          f"| majority baseline {max(y.mean(), 1-y.mean())*100:.2f}%")
    print("\n" + "=" * 74)
    print("  What is predictable WITHOUT any imaging data")
    print("=" * 74)

    rows = {}
    for name, X in blocks.items():
        pipe = Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(max_iter=5000))])
        _, agg = ev.nested_cv(pipe, X, y, param_grid={"c__C": [0.01, 0.1, 1.0, 10.0]},
                              outer_splits=10, seeds=(0, 1, 2), label=name)
        rows[name] = agg

    # A non-linear model can exploit interactions a linear one misses.
    rf = Pipeline([("s", StandardScaler()),
                   ("c", RandomForestClassifier(n_estimators=400, random_state=0))])
    _, agg = ev.nested_cv(rf, blocks["FIQ + sex + age + site"], y,
                          param_grid={"c__max_depth": [3, 5, None]},
                          outer_splits=10, seeds=(0, 1, 2),
                          label="all confounds, random forest")
    rows["all confounds, random forest"] = agg

    best = max(rows, key=lambda k: rows[k]["auc_mean"])
    print("\n" + "-" * 74)
    print(f"  Confounds-only ceiling: {rows[best]['auc_mean']:.3f} AUC "
          f"({rows[best]['accuracy_mean']*100:.2f}% accuracy) via '{best}'")
    print("  Any imaging model must beat this to be making a brain-based claim.")

    pd.DataFrame(rows).T.to_csv(r"C:\abide\confound_results.csv")
    print("\nSaved to C:\\abide\\confound_results.csv")


if __name__ == "__main__":
    main()
