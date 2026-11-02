"""FD-matched sensitivity analysis of the SAVED static AAL-116 LR_L2 out-of-fold predictions (600 development subjects).
No model is trained. Predictions, scores and errors are never used to choose matches or the protocol.

PREDECLARED PROTOCOL
  model            : AAL116 / LR_L2 saved OOF predictions (probability score, saved 0.5-threshold prediction)
  matching         : greedy 1:1 nearest-neighbour on subject-level mean FD, without replacement;
                     each ASD subject (processed in a seeded random order, seed 0) takes the nearest unused TC within the caliper
  caliper          : 0.2 * SD(mean FD over all 600 development subjects)
  bootstrap        : 2000 resamples of matched PAIRS (both members), seed 1
  random reference : 2000 subsets each, seed 2: (a) same total size, simple random; (b) class-balanced (n_pairs ASD + n_pairs TC)
Holdout, CC200 and the restricted ID file are not used."""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

warnings.simplefilter("ignore")
R = Path(r"C:\abide\results")
FX = R / "fast_accuracy_experiment"
OUT = R / "fd_matched_sensitivity"
OUT.mkdir(exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
LOG = OUT / "logs" / "fd_matching.log"
N_BOOT = N_RAND = 2000
SEED_MATCH, SEED_BOOT, SEED_RAND = 0, 1, 2
CAL_MULT = 0.2


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}\n")
    print(m, flush=True)


def metrics(y, pred, score):
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    return {"roc_auc": roc_auc_score(y, score), "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "f1": f1_score(y, pred, zero_division=0)}


MET = ["roc_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"]
log("=" * 60 + "\nFD-MATCHED SENSITIVITY start")
man = pd.read_csv(FX / "data" / "dev_split_manifest.csv", dtype={"sid": str})
coh = pd.read_csv(R / "phase2" / "fd_filtered_subjects.csv", dtype={"subject_id": str})
coh["sid"] = coh.subject_id.map(lambda s: str(int(float(s))))
coh = coh[coh.sid.isin(set(man.sid))]                                    # development rows only
oof = pd.read_csv(FX / "oof_predictions.csv", dtype={"subject_id": str})
oof = oof[(oof.atlas == "AAL116") & (oof.model == "LR_L2")]

# ---- verify inputs
assert len(man) == 600 and man.sid.is_unique and len(oof) == 600 and oof.subject_id.is_unique and set(oof.subject_id) == set(man.sid)
d = man.merge(oof[["subject_id", "y_true", "score", "y_pred", "fold"]], left_on="sid", right_on="subject_id").merge(coh[["sid", "func_mean_fd", "diagnosis"]].rename(columns={"diagnosis": "diagnosis_cohort"}), on="sid")
assert len(d) == 600 and (d.y == d.y_true).all() and (d.outer_fold == d.fold).all() and (d.y == (d.diagnosis_cohort == "ASD").astype(int)).all() and (d.diagnosis == d.diagnosis_cohort).all()
assert np.isfinite(d.func_mean_fd).all() and np.isfinite(d.score).all()
log(f"Inputs verified: 600 subjects, one LR_L2 AAL116 OOF prediction each; labels, folds, diagnosis and FD aligned (FD range {d.func_mean_fd.min():.3f}-{d.func_mean_fd.max():.3f}).")

# ---- matching (uses only FD and diagnosis)
fd = d.func_mean_fd.values; y = d.y.values
caliper = CAL_MULT * fd.std(ddof=1)
rng = np.random.default_rng(SEED_MATCH)
asd = np.where(y == 1)[0]; tc = list(np.where(y == 0)[0])
order = rng.permutation(asd)
pairs, used = [], set()
for a in order:
    cand = [t for t in tc if t not in used]
    dist = np.abs(fd[cand] - fd[a]); j = int(np.argmin(dist))
    if dist[j] <= caliper:
        pairs.append((a, cand[j])); used.add(cand[j])
pairs = np.array(pairs)
n_pairs = len(pairs)
sel = np.r_[pairs[:, 0], pairs[:, 1]]
unmatched = np.setdiff1d(np.arange(600), sel)
log(f"Caliper = 0.2*SD(FD) = {caliper:.4f}; matched pairs = {n_pairs}; unmatched ASD {268 - n_pairs}, unmatched TC {332 - n_pairs}")


def bal(label, a_fd, t_fd):
    pooled = np.sqrt((a_fd.var(ddof=1) + t_fd.var(ddof=1)) / 2)
    return {"sample": label, "n_ASD": len(a_fd), "n_TC": len(t_fd), "FD_mean_ASD": a_fd.mean(), "FD_mean_TC": t_fd.mean(), "FD_SD_ASD": a_fd.std(ddof=1), "FD_SD_TC": t_fd.std(ddof=1),
            "FD_median_ASD": np.median(a_fd), "FD_median_TC": np.median(t_fd), "std_mean_difference": (a_fd.mean() - t_fd.mean()) / pooled,
            "mannwhitney_p": stats.mannwhitneyu(a_fd, t_fd).pvalue, "ks_p": stats.ks_2samp(a_fd, t_fd).pvalue}


balance = pd.DataFrame([bal("full development cohort (before matching)", fd[y == 1], fd[y == 0]), bal("matched subset (after matching)", fd[pairs[:, 0]], fd[pairs[:, 1]]),
                        bal("unmatched remainder", fd[unmatched][y[unmatched] == 1], fd[unmatched][y[unmatched] == 0])])
balance["caliper"] = caliper
balance.to_csv(OUT / "fd_balance.csv", index=False)
log("FD balance (std. mean difference ASD-TC): before {:.3f}, after {:.3f}".format(balance.iloc[0].std_mean_difference, balance.iloc[1].std_mean_difference))

# ---- metrics
sc, pr = d.score.values, d.y_pred.values
full = metrics(y, pr, sc)
mat = metrics(y[sel], pr[sel], sc[sel])
# bootstrap over pairs
rb = np.random.default_rng(SEED_BOOT)
boot = {m: [] for m in MET}
for _ in range(N_BOOT):
    idx = rb.integers(0, n_pairs, n_pairs)
    s = np.r_[pairs[idx, 0], pairs[idx, 1]]
    mm = metrics(y[s], pr[s], sc[s])
    for m in MET:
        boot[m].append(mm[m])
bi = pd.DataFrame([{"metric": m, "matched_estimate": mat[m], "boot_mean": np.mean(boot[m]), "ci95_lo": np.percentile(boot[m], 2.5), "ci95_hi": np.percentile(boot[m], 97.5), "n_boot": N_BOOT,
                    "unit_resampled": "matched pairs", "note": "descriptive; ignores model-training uncertainty"} for m in MET])
bi.to_csv(OUT / "bootstrap_intervals.csv", index=False)
# random subsets
rr = np.random.default_rng(SEED_RAND)
size = 2 * n_pairs
ra, rbal = {m: [] for m in MET}, {m: [] for m in MET}
for _ in range(N_RAND):
    s = rr.choice(600, size, replace=False)
    if len(np.unique(y[s])) < 2:
        continue
    mm = metrics(y[s], pr[s], sc[s])
    for m in MET:
        ra[m].append(mm[m])
    s2 = np.r_[rr.choice(asd, n_pairs, replace=False), rr.choice(np.where(y == 0)[0], n_pairs, replace=False)]
    mm = metrics(y[s2], pr[s2], sc[s2])
    for m in MET:
        rbal[m].append(mm[m])
rows = []
for name, dist in (("random same-size subsets (unstratified)", ra), ("random class-balanced subsets", rbal)):
    for m in MET:
        v = np.array(dist[m]); est = mat[m]
        rows.append({"reference": name, "metric": m, "n_subsets": len(v), "subset_size": size, "ref_mean": v.mean(), "ref_sd": v.std(ddof=1), "ref_p2.5": np.percentile(v, 2.5), "ref_p97.5": np.percentile(v, 97.5),
                     "matched_estimate": est, "full_cohort": full[m], "matched_percentile_in_reference": float((v < est).mean() * 100),
                     "two_sided_empirical_p": float(2 * min((v <= est).mean(), (v >= est).mean())), "within_ref_95_range": bool(np.percentile(v, 2.5) <= est <= np.percentile(v, 97.5))})
rc = pd.DataFrame(rows); rc.to_csv(OUT / "random_subset_comparison.csv", index=False)
mm_df = pd.DataFrame([{"sample": "full development cohort (saved OOF, pooled)", "n": 600, **full}, {"sample": f"FD-matched subset ({n_pairs} ASD + {n_pairs} TC)", "n": size, **mat}])
mm_df.to_csv(OUT / "matched_metrics.csv", index=False)
json.dump({"caliper": caliper, "n_pairs": n_pairs, "seeds": {"match": SEED_MATCH, "bootstrap": SEED_BOOT, "random": SEED_RAND}, "n_boot": N_BOOT, "n_random": N_RAND, "full": full, "matched": mat}, open(OUT / "results_summary.json", "w"), indent=1)
# matched pairs file (IDs of development subjects only)
pd.DataFrame({"asd_sid": d.sid.values[pairs[:, 0]], "tc_sid": d.sid.values[pairs[:, 1]], "asd_fd": fd[pairs[:, 0]], "tc_fd": fd[pairs[:, 1]]}).to_csv(OUT / "matched_pairs.csv", index=False)
log("\n" + mm_df.round(4).to_string(index=False))
log("\n" + bi[["metric", "matched_estimate", "ci95_lo", "ci95_hi"]].round(4).to_string(index=False))
log("\n" + rc[["reference", "metric", "ref_mean", "ref_p2.5", "ref_p97.5", "matched_estimate", "matched_percentile_in_reference", "within_ref_95_range"]].round(4).to_string(index=False))
log("\n" + balance.round(4).to_string(index=False))
