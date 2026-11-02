"""
phase10_corrected_nested_cv.py -- Phase 10 (CORRECTED, STRICT nested CV).

Supersedes the interrupted run in results/phase10_advanced_accuracy/ (preserved as historical evidence,
never read as a result and never modified). Fixes two audited defects of that run:

  1. Global majority-vote hyperparameter selection across outer folds  ->  REMOVED. Every outer fold
     selects its OWN configuration from inner CV on its OWN outer-training data only.
  2. Inner-fold imputation/scaling fitted on the whole outer-training set  ->  FIXED. For every inner
     split, imputation + scaler are fitted on INNER-TRAIN only and applied to INNER-VALIDATION.

Design (per outer fold, per track):
  stage EN -> ARCH -> WD -> DROPOUT -> LR -> ES  (greedy, sequential, each stage = 2-fold inner CV)
  then refit preprocessing + Elastic Net + SSAE on the ENTIRE outer-training data and evaluate ONCE on
  the outer-test fold (seed 42; seeds 123/2026 re-run the final refit only, with that fold's own
  selection). Ordinary tracks additionally run comparator experiments B (feature stability, inner-train
  subsampling), H (representation/PCA) and CLF (classifier choice by inner CV, then one outer evaluation).
  Site-grouped track repeats the whole selection pipeline with StratifiedGroupKFold(site) outer AND inner.

Inner validation data is never used for fitting anything: SSAE early stopping / best-epoch restore uses a
15% hold-out carved out of the INNER-TRAIN data (stratified), so inner-validation scores are not
contaminated by epoch selection either.

Selection functions receive only inner-split matrices (never te_idx / X_te); this is machine-checked by
phase10_corrected_selftest.py.  Phases 1-9 and the old Phase 10 run are read-only (SHA-256 verified).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
AAL_DIR = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
PHASE5A_DIR = ROOT / "results" / "phase5a_fc_diagnostic"
PHASE6 = ROOT / "results" / "phase6_strategyC_all_nonzero"
PHASE9 = ROOT / "results" / "phase9_accuracy_optimization"
OLD10 = ROOT / "results" / "phase10_advanced_accuracy"
OUT = ROOT / "results" / "phase10_corrected_nested_cv"
SUBDIRS = ["checkpoints", "logs", "nested_cv", "elastic_net", "architecture", "regularization", "learning_rate",
           "early_stopping", "representation", "classifier", "final", "site_aware", "reports"]
for _s in SUBDIRS:
    (OUT / _s).mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT / "checkpoints"

GUARD_DIRS = [ROOT / "results" / d for d in [
    "phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification", "phase3_6_aal_download",
    "phase4", "phase4_1_constant_roi_audit", "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic",
    "phase5b_missing_fc_analysis", "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero",
    "phase6_strategyC_all_nonzero", "phase7_synthesis", "phase8_final_audit", "phase9_accuracy_optimization",
    "phase10_advanced_accuracy",  # old run: preserved historical evidence
]]

SEED = 42
SEEDS = [42, 123, 2026]
N_OUTER = 5
N_INNER = 2
TOL = 0.005
ES_HOLDOUT = 0.15
FISHER_Z_CLIP = 1e-6
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
EN_CONFIGS = [{"C": c, "l1_ratio": 0.5} for c in C_GRID] + [{"C": 0.1, "l1_ratio": r} for r in (0.25, 0.75)]
ARCHS = [("64x32", 64, 32), ("128x32", 128, 32), ("128x64", 128, 64), ("256x64", 256, 64), ("256x128", 256, 128)]
WDS = [1e-4, 1e-3, 1e-2, 5e-2]
DROPOUTS = [0.1, 0.2, 0.3, 0.5]
LRS = [1e-4, 3e-4, 1e-3, 3e-3]
PATIENCES = [5, 10, 15, None]
STAB_THRESH = [0.5, 0.67, 0.83, 1.0]
N_STAB_SUB = 6
PCA_COMPS = [64, 128, 256]
BASE = {"dropout": 0.0, "weight_decay": 1e-4, "lr": 1e-3, "batch_size": 32, "pretrain_epochs": 40,
        "finetune_epochs": 80, "sparsity_weight": 1e-4, "early_stopping_patience": None, "class_weight": "balanced"}
STAGES = ["EN", "ARCH", "WD", "DROPOUT", "LR", "ES"]

AUDIT_LOG = OUT / "logs" / "phase10_corrected_audit.log"


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(msg, flush=True)


def py(o):
    """Recursively convert numpy / pandas scalars to plain python for JSON."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, np.ndarray):
        return py(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ───────────────────────────── checkpointing (atomic, resume-safe) ─────────────────────────────
def ckpt_path(unit_id, d=None):
    return (d or CKPT_DIR) / f"{unit_id}.json"


def save_checkpoint(unit_id, payload, d=None):
    path = ckpt_path(unit_id, d)
    tmp = path.with_suffix(".json.tmp")
    data = {"payload": py(payload), "_complete": True, "_unit_id": unit_id,
            "_saved_utc": datetime.now(timezone.utc).isoformat()}
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(unit_id, d=None):
    path = ckpt_path(unit_id, d)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        log(f"  [checkpoint] {unit_id}: malformed -> NOT complete, will recompute")
        return None
    if (not isinstance(data, dict) or data.get("_complete") is not True or data.get("_unit_id") != unit_id
            or not isinstance(data.get("payload"), dict) or not data["payload"]):
        log(f"  [checkpoint] {unit_id}: invalid/incomplete -> NOT complete, will recompute")
        return None
    return data


def checkpointed(unit_id, fn, d=None):
    ck = load_checkpoint(unit_id, d)
    if ck is not None:
        log(f"  [checkpoint] {unit_id}: SKIP (done {ck.get('_saved_utc')})")
        return ck["payload"], True
    t0 = time.time()
    res = fn()
    res = py(res)
    res["_runtime_sec"] = round(time.time() - t0, 1)
    save_checkpoint(unit_id, res, d)
    return res, False


# ───────────────────────────── integrity (SHA-256) ─────────────────────────────
def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_files():
    files = []
    for d in GUARD_DIRS:
        if d.exists():
            files += [p for p in d.rglob("*") if p.is_file()]
    files += [p for p in ROOT.glob("phase*.py") if not p.name.startswith("phase10")]
    files += [ROOT / "phase10_advanced_accuracy.py"]  # old script, preserved
    return sorted(set(files))


def integrity_snapshot():
    return {str(p): {"size": p.stat().st_size, "mtime": p.stat().st_mtime, "sha256": _sha(p)} for p in guard_files()}


def integrity_compare(pre, post):
    changed = [k for k in pre if k not in post or pre[k]["sha256"] != post[k]["sha256"]]
    added = [k for k in post if k not in pre]
    return changed, added


# ───────────────────────────── data / preprocessing ─────────────────────────────
def normalize_id(raw):
    s = str(raw).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def extract_subject_id(name):
    base = name.replace("_rois_aal.1D", "")
    digits = "".join(ch for ch in base.split("_")[-1] if ch.isdigit())
    return str(int(digits)) if digits else None


def load_data():
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    cohort["subject_id_norm"] = cohort["subject_id"].map(normalize_id)
    assert len(pd.read_csv(PHASE5A_DIR / "fc_feature_index.csv")) == 6670
    id_to_file = {}
    for f in AAL_DIR.glob("*_rois_aal.1D"):
        sid = extract_subject_id(f.name)
        if sid:
            id_to_file[sid] = f
    subj_order = list(cohort["subject_id_norm"])
    subject_files = {sid: id_to_file[sid] for sid in subj_order if sid in id_to_file}
    assert len(subject_files) == 780
    diag = cohort.set_index("subject_id_norm").loc[subj_order, "diagnosis"].values
    site = cohort.set_index("subject_id_norm").loc[subj_order, "site"].values
    y = (diag == "ASD").astype(int)
    assert int(y.sum()) == 351 and len(y) - int(y.sum()) == 429 and len(set(site)) == 20
    fc_raw = np.full((780, 6670), np.nan, dtype=float)
    iu = np.triu_indices(116, k=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, sid in enumerate(subj_order):
            arr = np.loadtxt(str(subject_files[sid]))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            fc_raw[i] = np.corrcoef(arr, rowvar=False)[iu]
    assert int(np.isnan(fc_raw).sum()) == 14025
    return fc_raw, y, site, subj_order


def _fz(x):
    return np.arctanh(np.clip(x, -1 + FISHER_Z_CLIP, 1 - FISHER_Z_CLIP))


def fit_prep(X_raw_train):
    """Strategy C: column-mean imputation + Fisher-Z + StandardScaler, fitted on the GIVEN training rows only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        col_means = np.nanmean(X_raw_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    z = _fz(np.where(np.isnan(X_raw_train), col_means[None, :], X_raw_train))
    return {"col_means": col_means, "scaler": StandardScaler().fit(z)}


def apply_prep(prep, X_raw):
    imp = np.where(np.isnan(X_raw), prep["col_means"][None, :], X_raw)
    return prep["scaler"].transform(_fz(imp))


def build_inner(fc_raw, y, tr_idx, inner_splits):
    """Per inner split: preprocessing fitted on INNER-TRAIN rows only; applied to inner-train and inner-validation."""
    out = []
    for i, (a, b) in enumerate(inner_splits):
        itr, iva = tr_idx[a], tr_idx[b]
        prep = fit_prep(fc_raw[itr])
        out.append({"i": i, "Xtr": apply_prep(prep, fc_raw[itr]), "Xva": apply_prep(prep, fc_raw[iva]),
                    "ytr": y[itr], "yva": y[iva]})
    return out


def make_inner_splits(track, y_tr, groups_tr):
    z = np.zeros(len(y_tr))
    if track == "ordinary":
        return list(StratifiedKFold(N_INNER, shuffle=True, random_state=SEED).split(z, y_tr))
    return list(StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED).split(z, y_tr, groups=groups_tr))


def make_outer_splits(track, fc_raw, y, site):
    if track == "ordinary":
        return list(StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED).split(fc_raw, y))
    return list(StratifiedGroupKFold(N_OUTER, shuffle=True, random_state=SEED).split(fc_raw, y, groups=site))


# ───────────────────────────── models ─────────────────────────────
def fit_en(X_tr, y_tr, C, l1_ratio):
    en = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio, max_iter=3000,
                            tol=1e-4, class_weight="balanced", random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        en.fit(X_tr, y_tr)
    sel = np.where(en.coef_[0] != 0)[0]
    if len(sel) == 0:
        sel = np.argsort(-np.abs(en.coef_[0]))[:1]
    return sel


_EN_CACHE = {}


def get_en(key, X, y, en):
    k = (key, en["C"], en["l1_ratio"])
    if k not in _EN_CACHE:
        _EN_CACHE[k] = fit_en(X, y, en["C"], en["l1_ratio"])
    return _EN_CACHE[k]


N_JOBS = 10  # 12 physical cores / 24 threads; saga is single-threaded, so independent fits run as separate loky processes


def prefetch_en(key, inner, ens):
    """Speed-only: fit independent Elastic Net configs in parallel and fill _EN_CACHE. Each fit is exactly
    fit_en(inner-train matrix, inner-train labels, C, l1) -- deterministic (saga, random_state=SEED) -- so the
    cached selections are identical to serial computation (verified in selftest against stored checkpoints)."""
    from joblib import Parallel, delayed
    todo = [(fd, en) for fd in inner for en in ens if ((key, fd["i"]), en["C"], en["l1_ratio"]) not in _EN_CACHE]
    if not todo:
        return
    res = Parallel(n_jobs=min(N_JOBS, len(todo)), backend="loky")(
        delayed(fit_en)(fd["Xtr"], fd["ytr"], en["C"], en["l1_ratio"]) for fd, en in todo)
    for (fd, en), sel in zip(todo, res):
        _EN_CACHE[((key, fd["i"]), en["C"], en["l1_ratio"])] = sel


def _torch():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return torch, nn, optim, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_ssae(n_in, h1, h2, dropout, device, nn):
    class SSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1, self.dec1 = nn.Linear(n_in, h1), nn.Linear(h1, n_in)
            self.enc2, self.dec2 = nn.Linear(h1, h2), nn.Linear(h2, h1)
            self.classifier = nn.Linear(h2, 2)
            self.act1, self.act2 = nn.ReLU(), nn.ReLU()
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def encode1(self, x):
            return self.act1(self.enc1(x))

        def encode2(self, h):
            return self.act2(self.enc2(h))

        def forward(self, x):
            return self.classifier(self.drop(self.encode2(self.encode1(x))))
    return SSAE().to(device)


def train_ssae(X_fit, y_fit, X_es, y_es, cfg, seed):
    """Same SSAE recipe as Phase 6/9/old-10. (X_es, y_es) = early-stopping hold-out drawn from the TRAINING data."""
    torch, nn, optim, device = _torch()
    torch.manual_seed(seed)
    model = build_ssae(X_fit.shape[1], cfg["h1"], cfg["h2"], cfg["dropout"], device, nn)
    Xt = torch.tensor(X_fit, dtype=torch.float32, device=device)
    yt = torch.tensor(y_fit, dtype=torch.long, device=device)
    Xe = torch.tensor(X_es, dtype=torch.float32, device=device)
    ye = torch.tensor(y_es, dtype=torch.long, device=device)
    bs = cfg["batch_size"]
    opt1 = optim.Adam(list(model.enc1.parameters()) + list(model.dec1.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["pretrain_epochs"]):
        perm = torch.randperm(Xt.shape[0])
        for i in range(0, len(perm), bs):
            xb = Xt[perm[i:i + bs]]
            h1 = model.encode1(xb)
            loss = nn.functional.mse_loss(model.dec1(h1), xb) + cfg["sparsity_weight"] * h1.abs().mean()
            opt1.zero_grad(); loss.backward(); opt1.step()
    with torch.no_grad():
        H1 = model.encode1(Xt)
    opt2 = optim.Adam(list(model.enc2.parameters()) + list(model.dec2.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["pretrain_epochs"]):
        perm = torch.randperm(H1.shape[0])
        for i in range(0, len(perm), bs):
            hb = H1[perm[i:i + bs]]
            h2 = model.encode2(hb)
            loss = nn.functional.mse_loss(model.dec2(h2), hb) + cfg["sparsity_weight"] * h2.abs().mean()
            opt2.zero_grad(); loss.backward(); opt2.step()
    classes, counts = np.unique(y_fit, return_counts=True)
    weight = torch.tensor([len(y_fit) / (len(classes) * c) for c in counts], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    patience = cfg["early_stopping_patience"]
    best, best_state, ctr, stop_epoch, tl, vl = float("inf"), None, 0, cfg["finetune_epochs"], [], []
    for epoch in range(cfg["finetune_epochs"]):
        model.train()
        perm = torch.randperm(Xt.shape[0])
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = crit(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        tl.append(tot / Xt.shape[0])
        model.eval()
        with torch.no_grad():
            v = crit(model(Xe), ye).item()
        vl.append(v)
        if patience is not None:
            if v < best - 1e-6:
                best, best_state, ctr = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                ctr += 1
                if ctr >= patience:
                    stop_epoch = epoch + 1
                    break
        elif v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"final_train_loss": tl[-1], "final_es_loss": vl[-1], "best_es_loss": best, "stop_epoch": stop_epoch}


def eval_torch(model, X, y_true):
    torch, _, _, device = _torch()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
    return metrics_dict(y_true, preds, probs)


def metrics_dict(y_true, preds, probs):
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, preds), "balanced_accuracy": balanced_accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0), "sensitivity": recall_score(y_true, preds, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"), "f1": f1_score(y_true, preds, zero_division=0),
            "roc_auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def make_cfg(sel):
    return {**BASE, "h1": sel["h1"], "h2": sel["h2"], "weight_decay": sel["weight_decay"], "dropout": sel["dropout"],
            "lr": sel["lr"], "early_stopping_patience": sel["patience"]}


def es_split(y):
    return train_test_split(np.arange(len(y)), test_size=ES_HOLDOUT, stratify=y, random_state=SEED)


# ───────────────────────────── SELECTION (inner data only; no outer-test access) ─────────────────────────────
def eval_candidate(inner, key, en, cfg):
    per = []
    for fd in inner:
        sel = get_en((key, fd["i"]), fd["Xtr"], fd["ytr"], en)
        a, b = es_split(fd["ytr"])
        model, h = train_ssae(fd["Xtr"][a][:, sel], fd["ytr"][a], fd["Xtr"][b][:, sel], fd["ytr"][b], cfg, SEED)
        vm = eval_torch(model, fd["Xva"][:, sel], fd["yva"])
        per.append({"inner_fold": fd["i"], "n_features": len(sel), "val_roc_auc": vm["roc_auc"], "val_balanced_accuracy": vm["balanced_accuracy"],
                    "val_accuracy": vm["accuracy"], "train_loss": h["final_train_loss"], "es_loss": h["final_es_loss"], "stop_epoch": h["stop_epoch"]})
    return summarize(per), per


def summarize(per):
    return {"mean_N": float(np.mean([p["n_features"] for p in per])), "mean_inner_roc_auc": float(np.nanmean([p["val_roc_auc"] for p in per])),
            "mean_inner_balanced_accuracy": float(np.mean([p["val_balanced_accuracy"] for p in per])),
            "mean_inner_accuracy": float(np.mean([p["val_accuracy"] for p in per])),
            "mean_loss_gap": float(np.mean([p["es_loss"] - p["train_loss"] for p in per]))}


def pick(rows, complexity):
    best_auc = max(r["mean_inner_roc_auc"] for r in rows)
    tied = [r for r in rows if r["mean_inner_roc_auc"] >= best_auc - TOL]
    return min(tied, key=lambda r: (complexity(r), -r["mean_inner_roc_auc"]))


def stage_EN(inner, key):
    rows, per_all = [], []
    prefetch_en(key, inner, EN_CONFIGS)  # speed-only parallel pre-fit of the same 22 independent EN fits
    for en in EN_CONFIGS:
        s, per = eval_candidate(inner, key, en, {**BASE, "h1": 500, "h2": 100})
        rows.append({"C": en["C"], "l1_ratio": en["l1_ratio"], **s})
        per_all += [{"C": en["C"], "l1_ratio": en["l1_ratio"], **p} for p in per]
    b = pick(rows, lambda r: r["mean_N"])
    return {"rows": rows, "inner": per_all, "best": {"C": b["C"], "l1_ratio": b["l1_ratio"]}, "best_row": b}


def stage_ARCH(inner, key, en):
    rows, per_all = [], []
    for name, h1, h2 in ARCHS:
        s, per = eval_candidate(inner, key, en, {**BASE, "h1": h1, "h2": h2})
        rows.append({"architecture": name, "h1": h1, "h2": h2, "approx_params": int(s["mean_N"] * h1 + h1 * h2 + h2 * 2), **s})
        per_all += [{"architecture": name, **p} for p in per]
    b = pick(rows, lambda r: r["approx_params"])
    return {"rows": rows, "inner": per_all, "best": {"h1": b["h1"], "h2": b["h2"], "name": b["architecture"]}, "best_row": b}


def _sweep(inner, key, en, base_sel, field, values, label, complexity):
    rows, per_all = [], []
    for v in values:
        sel = {**base_sel, field: v}
        s, per = eval_candidate(inner, key, en, make_cfg(sel))
        rows.append({label: v, **s})
        per_all += [{label: v, **p} for p in per]
    return rows, per_all


def stage_WD(inner, key, en, arch):
    sel = {"h1": arch["h1"], "h2": arch["h2"], "weight_decay": 1e-4, "dropout": 0.0, "lr": 1e-3, "patience": None}
    rows, per_all = _sweep(inner, key, en, sel, "weight_decay", WDS, "weight_decay", None)
    b = pick(rows, lambda r: -r["weight_decay"])  # tie-break: stronger regularization
    return {"rows": rows, "inner": per_all, "best": {"weight_decay": b["weight_decay"]}, "best_row": b}


def stage_DROPOUT(inner, key, en, arch, wd, wd_best_row):
    sel = {"h1": arch["h1"], "h2": arch["h2"], "weight_decay": wd["weight_decay"], "dropout": 0.0, "lr": 1e-3, "patience": None}
    rows, per_all = _sweep(inner, key, en, sel, "dropout", DROPOUTS, "dropout", None)
    rows.append({"dropout": 0.0, **{k: v for k, v in wd_best_row.items() if k != "weight_decay"}, "carried_from_WD_stage": True})
    b = pick(rows, lambda r: -r["dropout"])
    return {"rows": rows, "inner": per_all, "best": {"dropout": b["dropout"]}, "best_row": b}


def stage_LR(inner, key, en, arch, wd, do):
    sel = {"h1": arch["h1"], "h2": arch["h2"], "weight_decay": wd["weight_decay"], "dropout": do["dropout"], "lr": 1e-3, "patience": None}
    rows, per_all = _sweep(inner, key, en, sel, "lr", LRS, "learning_rate_value", None)
    for r in rows:
        r["lr"] = r.pop("learning_rate_value")
    b = pick(rows, lambda r: r["lr"])  # tie-break: smaller LR
    return {"rows": rows, "inner": per_all, "best": {"lr": b["lr"]}, "best_row": b}


def stage_ES(inner, key, en, arch, wd, do, lr):
    sel = {"h1": arch["h1"], "h2": arch["h2"], "weight_decay": wd["weight_decay"], "dropout": do["dropout"], "lr": lr["lr"], "patience": None}
    rows, per_all = _sweep(inner, key, en, sel, "patience", PATIENCES, "patience", None)
    b = pick(rows, lambda r: 999 if r["patience"] is None else r["patience"])  # tie-break: earlier stopping
    return {"rows": rows, "inner": per_all, "best": {"patience": b["patience"]}, "best_row": b}


def stage_B(inner, key, en):
    """Feature-stability threshold: frequency from EN fits on random 50% subsamples of INNER-TRAIN only; LR probe on inner-val."""
    from joblib import Parallel, delayed
    rows = []
    subs = {fd["i"]: [sa for sa, _ in StratifiedShuffleSplit(n_splits=N_STAB_SUB, train_size=0.5, random_state=SEED).split(fd["Xtr"], fd["ytr"])] for fd in inner}
    jobs = [(fd, sa) for fd in inner for sa in subs[fd["i"]]]
    sels = Parallel(n_jobs=min(N_JOBS, len(jobs)), backend="loky")(delayed(fd_en)(fd, sa, en) for fd, sa in jobs)  # speed-only parallelism
    sel_by_fold = {fd["i"]: [] for fd in inner}
    for (fd, _), s in zip(jobs, sels):
        sel_by_fold[fd["i"]].append(s)
    for fd in inner:
        counts = np.zeros(fd["Xtr"].shape[1])
        for s in sel_by_fold[fd["i"]]:
            counts[s] += 1
        freq = counts / N_STAB_SUB
        for th in STAB_THRESH:
            stable = np.where(freq >= th - 1e-9)[0]
            if len(stable) == 0:
                continue
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED).fit(fd["Xtr"][:, stable], fd["ytr"])
            auc = roc_auc_score(fd["yva"], lr.predict_proba(fd["Xva"][:, stable])[:, 1])
            rows.append({"inner_fold": fd["i"], "stability_threshold": th, "n_stable_features": int(len(stable)), "inner_roc_auc_LR_probe": float(auc)})
    df = pd.DataFrame(rows)
    summ = df.groupby("stability_threshold").agg(n_stable_features=("n_stable_features", "mean"), mean_inner_roc_auc_LR_probe=("inner_roc_auc_LR_probe", "mean")).reset_index()
    return {"rows": summ.to_dict(orient="records"), "inner": rows,
            "best": {"stability_threshold": float(summ.sort_values("mean_inner_roc_auc_LR_probe", ascending=False).iloc[0].stability_threshold)}}


def fd_en(fd, sub_idx, en):
    return fit_en(fd["Xtr"][sub_idx], fd["ytr"][sub_idx], en["C"], en["l1_ratio"])


def stage_H(inner):
    """Representation check (LR probe). PCA fitted on INNER-TRAIN only per inner split."""
    rows = []
    for fd in inner:
        reps = {"standardized_fisher_z_full": (fd["Xtr"], fd["Xva"])}
        for n in PCA_COMPS:
            p = PCA(n_components=n, random_state=SEED).fit(fd["Xtr"])
            reps[f"pca_{n}"] = (p.transform(fd["Xtr"]), p.transform(fd["Xva"]))
        for name, (A, B) in reps.items():
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=SEED).fit(A, fd["ytr"])
            rows.append({"inner_fold": fd["i"], "representation": name, "n_components": A.shape[1],
                         "inner_roc_auc": float(roc_auc_score(fd["yva"], lr.predict_proba(B)[:, 1]))})
    summ = pd.DataFrame(rows).groupby("representation").agg(n_components=("n_components", "first"), mean_inner_roc_auc=("inner_roc_auc", "mean")).reset_index()
    return {"rows": summ.to_dict(orient="records"), "inner": rows}


CLF_ORDER = ["LogisticRegression", "LinearSVM", "RBF_SVM", "MLP_small", "SSAE"]


def new_clf(name):
    return {"LogisticRegression": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED),
            "LinearSVM": lambda: SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED, C=1.0),
            "RBF_SVM": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED, C=1.0),
            "MLP_small": lambda: MLPClassifier(hidden_layer_sizes=(64, 16), alpha=1e-2, max_iter=500, random_state=SEED)}[name]()


def stage_CLF(inner, key, en, sel):
    per = []
    for fd in inner:
        s = get_en((key, fd["i"]), fd["Xtr"], fd["ytr"], en)
        Xi, Xv = fd["Xtr"][:, s], fd["Xva"][:, s]
        for name in CLF_ORDER[:-1]:
            clf = new_clf(name).fit(Xi, fd["ytr"])
            m = metrics_dict(fd["yva"], clf.predict(Xv), clf.predict_proba(Xv)[:, 1])
            per.append({"inner_fold": fd["i"], "classifier": name, "val_roc_auc": m["roc_auc"], "val_balanced_accuracy": m["balanced_accuracy"], "val_accuracy": m["accuracy"]})
        a, b = es_split(fd["ytr"])
        model, _ = train_ssae(Xi[a], fd["ytr"][a], Xi[b], fd["ytr"][b], make_cfg(sel), SEED)
        m = eval_torch(model, Xv, fd["yva"])
        per.append({"inner_fold": fd["i"], "classifier": "SSAE", "val_roc_auc": m["roc_auc"], "val_balanced_accuracy": m["balanced_accuracy"], "val_accuracy": m["accuracy"]})
    df = pd.DataFrame(per).groupby("classifier").agg(mean_inner_roc_auc=("val_roc_auc", "mean"), mean_inner_balanced_accuracy=("val_balanced_accuracy", "mean"),
                                                     mean_inner_accuracy=("val_accuracy", "mean")).reset_index()
    rows = df.to_dict(orient="records")
    b = pick(rows, lambda r: CLF_ORDER.index(r["classifier"]))
    return {"rows": rows, "inner": per, "best": {"classifier": b["classifier"]}}


# ───────────────────────────── FINAL (outer train refit, outer test touched once per unit) ─────────────────────────────
_OUTER_CACHE = {}


def outer_prepared(track, k, fc_raw, y, tr_idx, te_idx, en):
    ck = (track, k, en["C"], en["l1_ratio"])
    if ck not in _OUTER_CACHE:
        prep = fit_prep(fc_raw[tr_idx])                       # ENTIRE outer train only
        Xtr, Xte = apply_prep(prep, fc_raw[tr_idx]), apply_prep(prep, fc_raw[te_idx])
        sel = fit_en(Xtr, y[tr_idx], en["C"], en["l1_ratio"])  # ENTIRE outer train only
        _OUTER_CACHE.clear()
        _OUTER_CACHE[ck] = (Xtr, Xte, sel)
    return _OUTER_CACHE[ck]


def final_unit(track, k, seed, selection, inner_best_row, fc_raw, y, site, tr_idx, te_idx):
    en = selection["EN"]
    Xtr, Xte, sel = outer_prepared(track, k, fc_raw, y, tr_idx, te_idx, en)
    ytr, yte = y[tr_idx], y[te_idx]
    cfg = make_cfg(selection)
    a, b = es_split(ytr)
    model, h = train_ssae(Xtr[a][:, sel], ytr[a], Xtr[b][:, sel], ytr[b], cfg, seed)
    m = eval_torch(model, Xte[:, sel], yte)
    out = {"track": track, "outer_fold": k, "seed": seed, "selected_configuration": selection, "n_selected_features": int(len(sel)),
           "metrics": m, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)), "train_loss": h["final_train_loss"],
           "es_holdout_loss": h["final_es_loss"], "stop_epoch": h["stop_epoch"],
           "inner_validation_of_selected_config": inner_best_row}
    if track == "site":
        out["n_sites_train"], out["n_sites_test"] = int(len(set(site[tr_idx]))), int(len(set(site[te_idx])))
        out["site_overlap_train_test"] = int(len(set(site[tr_idx]) & set(site[te_idx])))
    return out


def clf_outer_unit(k, chosen, selection, fc_raw, y, site, tr_idx, te_idx, ssae_final):
    if chosen == "SSAE":
        return {"classifier": "SSAE", "metrics": ssae_final["metrics"], "note": "inner CV chose the SSAE; outer result = seed-42 final unit"}
    Xtr, Xte, sel = outer_prepared("ordinary", k, fc_raw, y, tr_idx, te_idx, selection["EN"])
    clf = new_clf(chosen).fit(Xtr[:, sel], y[tr_idx])
    return {"classifier": chosen, "metrics": metrics_dict(y[te_idx], clf.predict(Xte[:, sel]), clf.predict_proba(Xte[:, sel])[:, 1])}


# ───────────────────────────── unit bookkeeping ─────────────────────────────
def uid(track, k, name):
    p = "ord" if track == "ordinary" else "site"
    return f"{p}_outer{k}_{name}"


def final_uid(track, k, seed):
    if track == "site":
        return f"site_outer_fold_{k}_final"
    return f"outer_fold_{k}_final" if seed == 42 else f"outer_fold_{k}_final_seed{seed}"


def planned_units():
    units = []
    for k in range(N_OUTER):
        units += [uid("ordinary", k, s) for s in STAGES] + [final_uid("ordinary", k, s) for s in SEEDS]
        units += [uid("ordinary", k, n) for n in ("B", "H", "CLF")]
    for k in range(N_OUTER):
        units += [uid("site", k, s) for s in STAGES] + [final_uid("site", k, 42)]
    return units


def progress():
    units = planned_units()
    done = [u for u in units if load_checkpoint_quiet(u)]
    return done, units


def load_checkpoint_quiet(u):
    p = ckpt_path(u)
    if not p.exists():
        return False
    try:
        d = json.load(open(p))
        return d.get("_complete") is True and d.get("_unit_id") == u and bool(d.get("payload"))
    except Exception:
        return False


def write_progress(extra=""):
    done, units = progress()
    with open(OUT / "logs" / "progress.json", "w") as f:
        json.dump({"completed": len(done), "planned": len(units), "remaining": [u for u in units if u not in done][:5],
                   "updated_utc": datetime.now(timezone.utc).isoformat(), "note": extra}, f, indent=1)
    return len(done), len(units)


def run_fold(track, k, fc_raw, y, site, outer_splits):
    tr_idx, te_idx = outer_splits[k]
    log(f"\n=== {track.upper()} outer fold {k}: train={len(tr_idx)} test={len(te_idx)} ===")
    if track == "site":
        assert not (set(site[tr_idx]) & set(site[te_idx])), "site leakage between outer train/test"
    inner_splits = make_inner_splits(track, y[tr_idx], site[tr_idx])
    inner = build_inner(fc_raw, y, tr_idx, inner_splits)   # preprocessing fitted per inner-train
    key = (track, k)
    R = {}

    def run(stage, fn):
        res, cached = checkpointed(uid(track, k, stage), fn)
        R[stage] = res
        if not cached:
            n, t = write_progress()
            log(f"  {track} fold {k} stage {stage}: best={res.get('best', '(comparator, no selection)')}  [PROGRESS {n}/{t}]")
        return res

    en = run("EN", lambda: stage_EN(inner, key))["best"]
    arch = run("ARCH", lambda: stage_ARCH(inner, key, en))["best"]
    wd = run("WD", lambda: stage_WD(inner, key, en, arch))
    do = run("DROPOUT", lambda: stage_DROPOUT(inner, key, en, arch, wd["best"], wd["best_row"]))["best"]
    lr = run("LR", lambda: stage_LR(inner, key, en, arch, wd["best"], do))["best"]
    es = run("ES", lambda: stage_ES(inner, key, en, arch, wd["best"], do, lr))
    selection = {"EN": en, "h1": arch["h1"], "h2": arch["h2"], "arch_name": arch["name"], "weight_decay": wd["best"]["weight_decay"],
                 "dropout": do["dropout"], "lr": lr["lr"], "patience": es["best"]["patience"]}
    log(f"  {track} fold {k} SELECTED (inner CV only): {selection}")
    finals = {}
    for seed in (SEEDS if track == "ordinary" else [42]):
        def _f(seed=seed):
            return final_unit(track, k, seed, selection, R["ES"]["best_row"], fc_raw, y, site, tr_idx, te_idx)
        res, cached = checkpointed(final_uid(track, k, seed), _f)
        finals[seed] = res
        if not cached:
            n, t = write_progress()
            log(f"  {track} fold {k} seed {seed} OUTER TEST: acc={res['metrics']['accuracy']:.4f} auc={res['metrics']['roc_auc']:.4f}  [PROGRESS {n}/{t}]")
    if track == "ordinary":
        run("B", lambda: stage_B(inner, key, en))
        run("H", lambda: stage_H(inner))

        def _clf():
            r = stage_CLF(inner, key, en, selection)
            r["outer_test_of_inner_selected_classifier"] = clf_outer_unit(k, r["best"]["classifier"], selection, fc_raw, y, site, tr_idx, te_idx, finals[42])
            return r
        run("CLF", _clf)


# ───────────────────────────── reporting (from checkpoints only) ─────────────────────────────
def P(u):
    d = load_checkpoint(u) if load_checkpoint_quiet(u) else None
    return d["payload"] if d else None


def collect():
    tabs = {"inner": [], "cand": [], "sel": [], "final": [], "clf": [], "repr": [], "stab": [], "site": []}
    for track in ("ordinary", "site"):
        for k in range(N_OUTER):
            for st in STAGES + (["B", "H", "CLF"] if track == "ordinary" else []):
                p = P(uid(track, k, st))
                if not p:
                    continue
                tag = {"track": track, "outer_fold": k, "stage": st}
                tabs["cand"] += [{**tag, **r} for r in p["rows"]]
                tabs["inner"] += [{**tag, **r} for r in p["inner"]]
                if st == "B":
                    tabs["stab"] += [{**tag, **r} for r in p["rows"]]
                if st == "H":
                    tabs["repr"] += [{**tag, **r} for r in p["rows"]]
                if st == "CLF":
                    o = p["outer_test_of_inner_selected_classifier"]
                    tabs["clf"] += [{**tag, **r, "inner_selected": r["classifier"] == p["best"]["classifier"]} for r in p["rows"]]
                    tabs["clf"].append({**tag, "classifier": o["classifier"], "OUTER_TEST_of_inner_selected": True, **{f"outer_{m}": v for m, v in o["metrics"].items()}})
            for s in (SEEDS if track == "ordinary" else [42]):
                p = P(final_uid(track, k, s))
                if p:
                    row = {"track": track, "outer_fold": k, "seed": s, **p["metrics"], "n_selected_features": p["n_selected_features"],
                           "n_train": p["n_train"], "n_test": p["n_test"], "train_loss": p["train_loss"], "es_holdout_loss": p["es_holdout_loss"],
                           "stop_epoch": p["stop_epoch"], **{f"inner_val_{m}": v for m, v in p["inner_validation_of_selected_config"].items() if m.startswith("mean_inner")}}
                    for extra in ("n_sites_train", "n_sites_test", "site_overlap_train_test"):
                        if extra in p:
                            row[extra] = p[extra]
                    tabs["final"].append(row)
                    if s == 42:
                        tabs["sel"].append({"track": track, "outer_fold": k, "seed": s, **{f"{a}": b for a, b in p["selected_configuration"].items() if a != "EN"},
                                            "elastic_net_C": p["selected_configuration"]["EN"]["C"], "elastic_net_l1_ratio": p["selected_configuration"]["EN"]["l1_ratio"],
                                            "n_selected_features": p["n_selected_features"]})
    return {k: pd.DataFrame(v) for k, v in tabs.items()}


MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]


def msd(df, col):
    return float(df[col].mean()), float(df[col].std(ddof=1)) if len(df) > 1 else float("nan")


def write_reports(status_extra=None, unit_tests=None, integrity=None):
    T = collect()
    done, units = progress()
    complete_units = len(done) == len(units)
    wr = lambda df, rel: df.to_csv(OUT / rel, index=False) if len(df) else None
    wr(T["inner"], "inner_cv_results.csv")
    wr(T["cand"], "hyperparameter_results.csv")
    wr(T["sel"], "selected_configurations_by_outer_fold.csv")
    wr(T["clf"], "classifier_results.csv")
    stage_dirs = {"EN": "elastic_net", "ARCH": "architecture", "WD": "regularization", "DROPOUT": "regularization", "LR": "learning_rate", "ES": "early_stopping"}
    for tr, tag in (("ordinary", ""), ("site", "site_")):
        for st, d in stage_dirs.items():
            sub = T["cand"][(T["cand"].track == tr) & (T["cand"].stage == st)] if len(T["cand"]) else T["cand"]
            wr(sub, f"{d}/{tag}{st}_results.csv")
    if len(T["repr"]):
        wr(T["repr"], "representation/representation_results.csv")
    if len(T["stab"]):
        wr(T["stab"], "representation/feature_stability_results.csv")
    fin = T["final"]
    ordf = fin[(fin.track == "ordinary") & (fin.seed == 42)].sort_values("outer_fold") if len(fin) else fin
    sitef = fin[fin.track == "site"].sort_values("outer_fold") if len(fin) else fin
    seedf = fin[fin.track == "ordinary"].sort_values(["seed", "outer_fold"]) if len(fin) else fin
    wr(ordf, "outer_fold_results.csv"); wr(ordf, "final/outer_fold_results_ordinary_seed42.csv")
    wr(seedf, "seed_results.csv"); wr(sitef, "site_aware_results.csv"); wr(sitef, "site_aware/site_grouped_outer_results.csv")
    wr(T["sel"], "nested_cv/selected_configurations_by_outer_fold.csv")

    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "units_completed": len(done), "units_planned": len(units)}
    # comparison
    p6 = pd.read_csv(PHASE6 / "evaluation" / "overall_metrics.csv")
    p9 = pd.read_csv(PHASE9 / "final_test" / "locked_test_results.csv")
    p9 = p9[p9.random_seed == 42].sort_values("fold").copy()
    p9["precision"] = p9.test_tp / (p9.test_tp + p9.test_fp)
    p9map = {"accuracy": "test_accuracy", "balanced_accuracy": "test_balanced_accuracy", "sensitivity": "test_sensitivity", "specificity": "test_specificity",
             "precision": "precision", "f1": "test_F1", "roc_auc": "test_ROC_AUC"}
    p6map = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "sensitivity": "sensitivity_recall", "specificity": "specificity",
             "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}
    comp = []
    for m in MET:
        r = {"metric": m, "phase6_mean": float(p6.loc[p6.statistic == "mean", p6map[m]].iloc[0]), "phase6_sd": float(p6.loc[p6.statistic == "std", p6map[m]].iloc[0]),
             "phase9_mean": float(p9[p9map[m]].mean()), "phase9_sd": float(p9[p9map[m]].std(ddof=1))}
        if len(ordf) == N_OUTER:
            r["phase10_corrected_ordinary_mean"], r["phase10_corrected_ordinary_sd"] = msd(ordf, m)
            r["p10_minus_p9"] = r["phase10_corrected_ordinary_mean"] - r["phase9_mean"]
            r["p10_minus_p6"] = r["phase10_corrected_ordinary_mean"] - r["phase6_mean"]
        comp.append(r)
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "comparison_phase6_phase9_phase10.csv", index=False)
    if len(ordf) == N_OUTER:
        summary["ordinary_outer_test_seed42"] = {m: dict(zip(("mean", "sd"), msd(ordf, m))) for m in MET}
    if len(seedf) == N_OUTER * len(SEEDS):
        summary["per_seed_mean"] = {int(s): {m: float(g[m].mean()) for m in MET} for s, g in seedf.groupby("seed")}
        sm = seedf.groupby("seed")[MET].mean()
        summary["across_seed_mean_sd"] = {m: {"mean": float(sm[m].mean()), "sd": float(sm[m].std(ddof=1))} for m in MET}
    if len(sitef) == N_OUTER:
        summary["site_grouped_outer_test_seed42"] = {m: dict(zip(("mean", "sd"), msd(sitef, m))) for m in MET}

    # validation-test gap
    gap = ["# Validation-test gap (corrected Phase 10)\n",
           "Inner validation = mean over the 2 inner folds of the SELECTED configuration (max over candidates, so mildly optimistic by selection). "
           "Inner-validation rows are scored on data never used for imputation/scaling/Elastic Net/SSAE fitting/early stopping.\n",
           "Phase 9 reference (seed 42): validation accuracy ~95.5%, validation AUC ~0.994, outer-test accuracy 64.9%, outer-test AUC 0.688.\n"]
    for name, df in (("ORDINARY nested CV (seed 42)", ordf), ("SITE-GROUPED nested CV", sitef)):
        if len(df):
            g = df[["outer_fold", "inner_val_mean_inner_accuracy", "accuracy", "inner_val_mean_inner_roc_auc", "roc_auc"]].copy()
            g["acc_gap"] = g.inner_val_mean_inner_accuracy - g.accuracy
            g["auc_gap"] = g.inner_val_mean_inner_roc_auc - g.roc_auc
            gap.append(f"\n## {name}  ({len(df)}/5 folds)\n\n```\n" + g.round(4).to_string(index=False) + "\n```\n")
            gap.append(f"\nMean inner-val accuracy {g.inner_val_mean_inner_accuracy.mean():.4f} vs outer-test accuracy {g.accuracy.mean():.4f} (gap {g.acc_gap.mean():+.4f}); "
                       f"mean inner-val AUC {g.inner_val_mean_inner_roc_auc.mean():.4f} vs outer-test AUC {g.roc_auc.mean():.4f} (gap {g.auc_gap.mean():+.4f}).\n")
            summary[f"gap_{'ordinary' if 'ORD' in name else 'site'}"] = {"inner_val_acc": float(g.inner_val_mean_inner_accuracy.mean()), "outer_test_acc": float(g.accuracy.mean()),
                                                                          "inner_val_auc": float(g.inner_val_mean_inner_roc_auc.mean()), "outer_test_auc": float(g.roc_auc.mean())}
    if len(ordf) == 0:
        gap.append("\nNo outer-test results yet.\n")
    (OUT / "validation_test_gap.md").write_text("".join(gap), encoding="utf-8")

    # leakage audit
    ut = unit_tests or (json.load(open(OUT / "logs" / "unit_tests.json")) if (OUT / "logs" / "unit_tests.json").exists() else None)
    ut_ok = bool(ut and ut.get("all_passed"))
    ev = lambda n: ("PASS" if ut and ut.get("tests", {}).get(n) else "UNCLEAR (selftest not passed)")
    rows = [("Imputation (inner)", ev("inner_prep_isolation"), "fit_prep() on inner-train rows only (build_inner); perturbing inner-val/outer-test rows leaves train output bit-identical."),
            ("Imputation/scaling (outer)", ev("outer_prep_isolation"), "fit_prep() on entire outer-train only (outer_prepared); perturbing outer-test rows leaves train matrix + EN selection identical."),
            ("Scaling (inner)", ev("inner_prep_isolation"), "StandardScaler inside fit_prep, inner-train only."),
            ("Fisher-Z", "PASS", "_fz(): element-wise arctanh with 1e-6 clip, nothing learned."),
            ("Elastic Net / feature selection (inner)", ev("inner_prep_isolation"), "get_en()/fit_en() on fd['Xtr'] (inner-train) only."),
            ("Elastic Net / feature selection (final)", ev("outer_prep_isolation"), "outer_prepared(): EN fit on entire outer-train only; applied once to outer-test."),
            ("PCA (Experiment H)", ev("inner_prep_isolation"), "stage_H(): PCA.fit(fd['Xtr']) per inner split; transform on inner-val."),
            ("Architecture / regularization / LR / early-stopping selection", ev("selection_has_no_test_access"), "stage_* functions take only the inner list; source scan finds no te_idx/X_te/y_te."),
            ("Hyperparameter selection scope", "PASS", "Per outer fold from that fold's own outer-train inner CV; no cross-fold majority vote anywhere (run_fold has no cross-fold state)."),
            ("Classifier selection", ev("selection_has_no_test_access"), "stage_CLF() inner CV only; outer test touched once for the inner-selected classifier."),
            ("Early stopping", "PASS", "Uses a 15% stratified hold-out of the TRAINING data (es_split); inner-val/outer-test never used for epoch choice."),
            ("Seed handling", "PASS", "Seeds 42/123/2026 predefined; all reported; headline = seed 42 fixed a priori; selection done once (seed 42) per fold."),
            ("Outer-test evaluation", ev("selection_has_no_test_access"), "Only final_unit()/clf_outer_unit() read te_idx; one evaluation per (fold, seed) plus one for the inner-selected classifier."),
            ("Site-grouped evaluation", ev("partitions"), "StratifiedGroupKFold(site) for outer AND inner splits; run_fold asserts no outer site overlap; own selection per site fold."),
            ("Fold partitions", ev("partitions"), "Inner subsets disjoint from outer test; group folds share no sites.")]
    md = ["# Leakage audit -- corrected Phase 10\n", f"Self-test (phase10_corrected_selftest.py) all_passed = {ut_ok}\n\n", "| Item | Result | Evidence |\n|---|---|---|\n"]
    md += [f"| {a} | {b} | {c} |\n" for a, b, c in rows]
    md.append("\n## Residual notes (not strict-nested failures)\n- Greedy sequential stage search (EN->ARCH->WD->DROPOUT->LR->ES) per fold; interactions between stages are not searched (scope reduction, disclosed).\n"
              "- Inner CV has k=2 (small training sets); inner scores are noisy. Tie tolerance 0.005 AUC with complexity tie-break.\n"
              "- Selected-config inner scores are a max over candidates and therefore slightly optimistic.\n"
              "- Final refit uses the same 15% training hold-out recipe for early stopping; that hold-out was inside Elastic Net's training data (affects only epoch choice, not test).\n"
              "- Experiments B/H are comparators (LR probe) and do not feed the main model; CLF chooses a classifier by inner CV and evaluates only that choice on outer test.\n")
    (OUT / "leakage_audit.md").write_text("".join(md), encoding="utf-8")

    (OUT / "methodology.md").write_text(
        "# Methodology (corrected Phase 10)\nCohort: 780 subjects (351 ASD / 429 TC), 20 sites, AAL-116, 6,670 FC features (unchanged).\n"
        "Outer: 5-fold (ordinary StratifiedKFold seed 42; site-grouped StratifiedGroupKFold seed 42). Inner: 2-fold (same type) inside each outer-train.\n"
        "Preprocessing (Strategy C mean imputation -> Fisher-Z (clip 1e-6) -> StandardScaler) fitted on the training rows of each split only.\n"
        f"Search per outer fold: EN {len(EN_CONFIGS)} configs (C grid x l1 0.5 plus l1 0.25/0.75 at C=0.1) with 500x100 SSAE; ARCH {[a[0] for a in ARCHS]}; WD {WDS}; dropout {DROPOUTS} (+0); LR {LRS}; patience {PATIENCES}.\n"
        "Selection: max mean inner ROC-AUC; ties within 0.005 -> simpler (fewer features / fewer params / stronger regularization / smaller LR / earlier stopping).\n"
        "Final: refit preprocessing + Elastic Net + SSAE on entire outer-train, one outer-test evaluation. Seeds 42/123/2026 vary SSAE init/shuffle only.\n"
        "Comparators (ordinary only): B stability (6 x 50% inner-train subsamples), H representation/PCA (LR probe), CLF classifier choice by inner CV.\n", encoding="utf-8")
    (OUT / "limitations.md").write_text(
        "# Limitations\n- Computational classification/reproduction study; no selected feature is claimed as a biomarker, causal connection or clinical/disease mechanism.\n"
        "- Not an exact reproduction of the published paper (Strategy C imputation, Elastic Net instead of SVM-RFE, variable input SSAE); no equivalence to its 98.2% is claimed.\n"
        "- 5 outer folds / 2 inner folds: limited power, noisy selection; greedy stage-wise search not a full grid; tie tolerance 0.005.\n"
        "- Multi-seed runs reuse each fold's seed-42 selection (selection itself is not repeated per seed).\n- Site-grouped track is a separate evaluation; results must not be pooled with the ordinary track.\n"
        "- Constant-ROI / imputed entries concentrated at SBL and PITT (Phase 4/7) remain a confound; 47 subjects carry imputed values.\n- No external validation cohort.\n", encoding="utf-8")

    complete = complete_units and ut_ok and bool(integrity and integrity.get("unchanged"))
    summary["leakage_selftest_passed"] = ut_ok
    summary["integrity"] = integrity
    summary["status"] = "COMPLETE" if complete else f"IN PROGRESS ({len(done)}/{len(units)} units)"
    with open(OUT / "final_summary.json", "w") as f:
        json.dump(py(summary), f, indent=2)
    lines = [f"PHASE 10 CORRECTED NESTED CV -- {summary['status']}", f"units: {len(done)}/{len(units)}", ""]
    for key in ("ordinary_outer_test_seed42", "across_seed_mean_sd", "site_grouped_outer_test_seed42"):
        if key in summary:
            lines.append(key)
            lines += [f"  {m}: {v.get('mean'):.4f} +/- {v.get('sd'):.4f}" for m, v in summary[key].items()]
    if "per_seed_mean" in summary:
        lines.append("per_seed_mean")
        lines += [f"  seed {s}: acc {v['accuracy']:.4f} auc {v['roc_auc']:.4f}" for s, v in summary["per_seed_mean"].items()]
    lines.append(f"\ncomparison table: comparison_phase6_phase9_phase10.csv\nintegrity: {integrity}")
    (OUT / "final_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return summary


# ───────────────────────────── main ─────────────────────────────
def main():
    log("=" * 70)
    log("PHASE 10 CORRECTED (strict nested CV) -- start/resume")
    pre_path = OUT / "logs" / "integrity_pre.json"
    if pre_path.exists():
        pre = json.load(open(pre_path))
        log("Verifying Phases 1-9 + old Phase 10 against stored SHA-256 baseline...")
    else:
        pre = integrity_snapshot()
        json.dump(pre, open(pre_path, "w"))
        log(f"Baseline SHA-256 manifest written ({len(pre)} files).")
    ch, ad = integrity_compare(pre, integrity_snapshot())
    if ch or ad:
        log(f"INTEGRITY FAIL: changed={ch[:10]} added={ad[:10]}")
        sys.exit(2)
    log("Integrity before run: PASS (all hashes identical to baseline).")
    ut_path = OUT / "logs" / "unit_tests.json"
    if not (ut_path.exists() and json.load(open(ut_path)).get("all_passed")):
        log("Self-test has not passed -- refusing to train. Run phase10_corrected_selftest.py first.")
        sys.exit(3)

    fc_raw, y, site, _ = load_data()
    log("Data loaded: 780 subjects (351/429), 20 sites, 6670 features, 14025 undefined FC entries.")
    splits = {t: make_outer_splits(t, fc_raw, y, site) for t in ("ordinary", "site")}
    for t, sp in splits.items():
        rows = [{"subject_index": int(i), "site": site[i], "diagnosis": int(y[i]), "outer_fold": k, "role": r}
                for k, (a, b) in enumerate(sp) for r, idx in (("train", a), ("test", b)) for i in idx]
        pd.DataFrame(rows).to_csv(OUT / "nested_cv" / f"fold_assignments_{t}.csv", index=False)
    n, t = write_progress()
    log(f"RESUME CHECK: {n}/{t} units complete, {t - n} remaining.")
    for track in ("ordinary", "site"):
        for k in range(N_OUTER):
            run_fold(track, k, fc_raw, y, site, splits[track])
    n, t = write_progress()
    changed, added = integrity_compare(pre, integrity_snapshot())
    integ = {"unchanged": not changed and not added, "changed": changed, "added": added, "files_checked": len(pre), "method": "SHA-256"}
    log(f"Integrity after run: {'PASS' if integ['unchanged'] else 'FAIL'} {integ}")
    s = write_reports(integrity=integ)
    log(f"FINAL STATUS: {s['status']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        pre = json.load(open(OUT / "logs" / "integrity_pre.json"))
        ch, ad = integrity_compare(pre, integrity_snapshot())
        print(write_reports(integrity={"unchanged": not ch and not ad, "changed": ch, "added": ad, "files_checked": len(pre), "method": "SHA-256"})["status"])
    else:
        main()
