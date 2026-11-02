"""
train_anat.py - Structural morphometry classifier on the UCLA_1 cohort.

Evaluates the features produced by preprocess_anat.py using the leakage-free
harness in evaluation.py. This is the honest replacement for train_mil.py: the
same question, asked in a way the answer can be trusted.

Run preprocess_anat.py first.
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
from sklearn.svm import SVC

import evaluation as ev

warnings.filterwarnings("ignore")

MORPH = r"C:\abide\derivatives\morphometry.csv"
PHENO = r"C:\abide\Phenotypic_V1_0b.csv"

# Deterministic histogram description plus global head size. The mixture-based
# tissue fractions are deliberately excluded: they fail on ~7% of subjects (see
# intensity_histogram in preprocess_anat.py), and dropping those subjects would
# mean excluding on a criterion correlated with scan quality.
HIST = (
    [f"q{p:02d}" for p in (5, 10, 25, 50, 75, 90, 95)]
    + [f"bin{i:02d}" for i in range(12)]
    + ["hist_skew", "hist_kurtosis", "hist_cov", "hist_iqr"]
)
FEATURES = HIST + ["head_size_index"]


def load(feature_set=None):
    features = feature_set or FEATURES
    m = pd.read_csv(MORPH)
    m = m[m["error"].isna() | (m["error"] == "")].copy()
    m["SUB_ID"] = m["subject"].astype(int)

    ph = pd.read_csv(PHENO)
    df = m.merge(
        ph[["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ", "SITE_ID"]],
        on="SUB_ID", how="inner",
    )

    # 1 = autism, 2 = control -> positive class is autism.
    df["y"] = (df["DX_GROUP"] == 1).astype(int)
    df = df.dropna(subset=features)
    n_seg_ok = int(df["seg_ok"].sum()) if "seg_ok" in df else -1
    print(f"Cohort: {len(df)} subjects | ASD {int(df.y.sum())} | TC {int((1-df.y).sum())}")
    if n_seg_ok >= 0:
        print(f"  (mixture segmentation passed QC on {n_seg_ok}/{len(df)}; not used as features)")
    return df


def main():
    if not os.path.exists(MORPH):
        raise SystemExit("Run preprocess_anat.py first - no morphometry.csv found.")

    df = load()
    X = df[FEATURES].values
    y = df["y"].values

    print("\n" + "=" * 74)
    print("  Univariate group differences")
    print("=" * 74)
    ev.confound_report(
        y,
        age=df["AGE_AT_SCAN"].values,
        FIQ=df["FIQ"].replace(-9999, np.nan).values,
        **{f: df[f].values for f in ("q25", "q50", "q75", "hist_skew",
                                     "hist_cov", "hist_iqr", "head_size_index")},
    )

    print("\n" + "=" * 74)
    print("  Nested cross-validation")
    print("=" * 74)

    models = {
        "logistic (L2)": (
            Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=5000))]),
            {"c__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "linear SVM": (
            Pipeline([("s", StandardScaler()), ("c", SVC(kernel="linear", probability=True))]),
            {"c__C": [0.01, 0.1, 1.0]},
        ),
        "RBF SVM": (
            Pipeline([("s", StandardScaler()), ("c", SVC(probability=True))]),
            {"c__C": [0.1, 1.0, 10.0], "c__gamma": ["scale", 0.01]},
        ),
        "random forest": (
            Pipeline([("s", StandardScaler()),
                      ("c", RandomForestClassifier(n_estimators=500, random_state=0))]),
            {"c__max_depth": [2, 3, None], "c__min_samples_leaf": [1, 3, 5]},
        ),
    }

    results = {}
    for name, (pipe, grid) in models.items():
        _, agg = ev.nested_cv(pipe, X, y, param_grid=grid, outer_splits=10,
                              seeds=(0, 1, 2, 3, 4), label=name)
        results[name] = agg

    best = max(results, key=lambda k: results[k]["auc_mean"])
    print("\n" + "-" * 74)
    print(f"  Best by AUC: {best} ({results[best]['auc_mean']:.3f})")

    print("\n" + "=" * 74)
    print("  Permutation test (best model)")
    print("=" * 74)
    ev.permutation_test(models[best][0], X, y, n_permutations=500)

    pd.DataFrame(results).T.to_csv(r"C:\abide\anat_results.csv")
    print("\nSaved to C:\\abide\\anat_results.csv")


if __name__ == "__main__":
    main()
