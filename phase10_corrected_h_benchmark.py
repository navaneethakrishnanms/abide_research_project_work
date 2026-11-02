"""Controlled BLAS-thread benchmark for Experiment H (ordinary outer fold 0, inner fold 0 + 1). Implementation-only:
same data, same preprocessing, same PCA/LR specification and random_state as stage_H; only thread limits differ
(set via environment BEFORE numpy is imported). Reads no outer-test data. Writes logs/h_thread_benchmark_<tag>.json
and logs/h_bench_<tag>.npz (outputs for cross-setting comparison)."""
import json
import os
import sys
import time

tag = sys.argv[1]
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np  # noqa: E402

import phase10_corrected_nested_cv as m  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

fc_raw, y, site, _ = m.load_data()
tr, _te = m.make_outer_splits("ordinary", fc_raw, y, site)[0]       # outer test indices are never used below
inner = m.build_inner(fc_raw, y, tr, m.make_inner_splits("ordinary", y[tr], site[tr]))
out, timing = {}, {}
for fd in inner:
    reps = {"full": None, **{f"pca_{n}": n for n in m.PCA_COMPS}}
    for name, n in reps.items():
        t0 = time.time()
        if n is None:
            A, B, extra = fd["Xtr"], fd["Xva"], None
        else:
            p = PCA(n_components=n, random_state=m.SEED).fit(fd["Xtr"])
            A, B, extra = p.transform(fd["Xtr"]), p.transform(fd["Xva"]), p.explained_variance_ratio_
        t_pca = time.time() - t0
        t1 = time.time()
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=m.SEED).fit(A, fd["ytr"])
        prob = lr.predict_proba(B)[:, 1]
        t_lr = time.time() - t1
        k = f"f{fd['i']}_{name}"
        out[k + "_Xva"], out[k + "_prob"], out[k + "_coef"] = B, prob, lr.coef_
        if extra is not None:
            out[k + "_evr"] = extra
        out[k + "_auc"] = np.array([roc_auc_score(fd["yva"], prob)])
        timing[k] = {"pca_s": round(t_pca, 2), "lr_s": round(t_lr, 2), "n_iter": int(np.max(lr.n_iter_))}
total = sum(v["pca_s"] + v["lr_s"] for v in timing.values())
np.savez(m.OUT / "logs" / f"h_bench_{tag}.npz", **out)
json.dump({"tag": tag, "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
           "total_compute_seconds": round(total, 2), "timing": timing}, open(m.OUT / "logs" / f"h_thread_benchmark_{tag}.json", "w"), indent=1)
print(tag, "total", round(total, 1), "s")
