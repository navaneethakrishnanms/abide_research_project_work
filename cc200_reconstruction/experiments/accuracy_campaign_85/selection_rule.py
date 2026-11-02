"""FROZEN selection rule for the accuracy campaign (separate from, and not replacing, the frozen X1/X2 rule). Written and hashed before any campaign result exists.
Primary metric: pooled out-of-fold ACCURACY under reading R1 (decision threshold chosen on inner out-of-fold scores of the outer-training partition only), mean over seeds 0-2.
Reference: the campaign's own reproduction of the baseline (P1_LR, same outer folds). Inputs per candidate: dict seed -> metrics dict (accuracy, balanced_accuracy, roc_auc) under R1,
and the 30 paired per-fold accuracies (candidate, reference)."""
import numpy as np
from scipy import stats

GAIN_ACC_PTS = 1.5          # mean paired gain in accuracy AND balanced accuracy over seeds (percentage points)
GAIN_EVERY_SEED_PTS = 0.5   # gain in every seed (accuracy)
MIN_FOLD_WINS = 20          # of 30 paired outer folds (one-sided sign test p = 0.049)
AUC_TOLERANCE = 0.005       # mean AUC may not be lower than the reference by more than this
REPL_GAIN_PTS = 1.0         # replication on fresh seeds 3-5: mean accuracy gain
REPL_MIN_POSITIVE = 2       # ... and positive in at least this many of the 3 fresh seeds
SUSPICIOUS_GAIN_PTS = 5.0   # a gain this large triggers a mandatory leakage / data-quality investigation before anything else


def screen(ref, cand, ref_fold_acc, cand_fold_acc):
    seeds = sorted(ref)
    d_acc = np.array([cand[s]["accuracy"] - ref[s]["accuracy"] for s in seeds]) * 100
    d_bacc = np.array([cand[s]["balanced_accuracy"] - ref[s]["balanced_accuracy"] for s in seeds]) * 100
    d_auc = np.array([cand[s]["roc_auc"] - ref[s]["roc_auc"] for s in seeds])
    diff = np.asarray(cand_fold_acc) - np.asarray(ref_fold_acc); wins = int((diff > 0).sum())
    crit = {"a_mean_gain_accuracy>=1.5pt": d_acc.mean() >= GAIN_ACC_PTS, "a_mean_gain_balanced_accuracy>=1.5pt": d_bacc.mean() >= GAIN_ACC_PTS,
            "b_every_seed_accuracy_gain>=0.5pt": bool((d_acc >= GAIN_EVERY_SEED_PTS).all()), "c_fold_wins>=20_of_30": wins >= MIN_FOLD_WINS,
            "d_mean_auc_not_lower_by_more_than_0.005": d_auc.mean() >= -AUC_TOLERANCE}
    info = dict(mean_gain_acc=float(d_acc.mean()), mean_gain_bacc=float(d_bacc.mean()), gain_acc_by_seed=d_acc.tolist(), mean_d_auc=float(d_auc.mean()), fold_wins=wins, fold_ties=int((diff == 0).sum()),
                fold_losses=int((diff < 0).sum()), sign_test_p=float(stats.binom.sf(wins - 1, len(diff), .5)), fold_diff_se=float(diff.std(ddof=1) / np.sqrt(len(diff))) * 100,
                suspicious=bool(d_acc.mean() >= SUSPICIOUS_GAIN_PTS))
    return crit, all(crit.values()), info


def replication_ok(d_acc_pts_by_seed):
    d = np.asarray(d_acc_pts_by_seed)
    return bool(d.mean() >= REPL_GAIN_PTS and int((d > 0).sum()) >= REPL_MIN_POSITIVE)
