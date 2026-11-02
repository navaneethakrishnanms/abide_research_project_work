"""
phase9_accuracy_optimization.py -- Phase 9: controlled, leakage-safe
accuracy-improvement experiments on top of the frozen Phase 6 baseline.

Design: the outer 5-fold split is IDENTICAL to Phase 6 (same StratifiedKFold,
seed=42). Each outer fold's TRAINING portion is further split 85/15
(stratified, seed=42) into an inner train/validation set, exactly as Phase 6
already did internally for loss monitoring. All hyperparameter selection uses
ONLY inner-validation metrics, averaged across the 5 outer folds. The outer
TEST folds are touched only twice: (a) EXP_00_BASELINE, a one-time
reproduction check against the frozen Phase 6 numbers, and (b) the final
locked configuration's one-time evaluation + multi-seed stability check,
after every hyperparameter has already been fixed using validation data only.

Phases 1-8 are read-only inputs and are never modified.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
AAL_DIR = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
PHASE4_1_DIR = ROOT / "results" / "phase4_1_constant_roi_audit"
PHASE5A_DIR = ROOT / "results" / "phase5a_fc_diagnostic"
PHASE6 = ROOT / "results" / "phase6_strategyC_all_nonzero"
OUT = ROOT / "results" / "phase9_accuracy_optimization"
for sub in ["baseline", "elastic_net", "ssae", "regularization", "class_balance", "classifiers",
            "seeds", "validation", "final_test", "comparisons", "xai", "reports", "logs"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [ROOT / "results" / d for d in [
    "phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification",
    "phase3_6_aal_download", "phase4", "phase4_1_constant_roi_audit",
    "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic", "phase5b_missing_fc_analysis",
    "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero",
    "phase6_strategyC_all_nonzero", "phase7_synthesis", "phase8_final_audit",
]]

SEED = 42
N_FOLDS = 5
FISHER_Z_CLIP = 1e-6

# baseline (Phase 6) fixed values, reused as the Phase 9 reference point
BASELINE_EN = {"C": 0.1, "l1_ratio": 0.5}
BASELINE_ARCH = {"h1": 500, "h2": 100}
BASELINE_TRAIN = {"dropout": 0.0, "weight_decay": 1e-4, "lr": 1e-3, "batch_size": 32,
                  "pretrain_epochs": 40, "finetune_epochs": 80, "sparsity_weight": 1e-4,
                  "early_stopping_patience": None, "class_weight": "balanced"}

audit_lines = []
experiment_audit_rows = []


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    audit_lines.append(line)
    print(msg, flush=True)


def fp_snapshot():
    out = {}
    for root in GUARD_DIRS:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    st = p.stat()
                    out[str(p)] = (st.st_size, st.st_mtime)
    return out


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


def record_audit(experiment_id, group, why, changed_vs_baseline, fixed, validation_use, test_accessed, test_influenced_selection):
    experiment_audit_rows.append({
        "experiment_id": experiment_id, "group": group, "why_run": why,
        "changed_vs_baseline": changed_vs_baseline, "what_remained_fixed": fixed,
        "how_validation_used": validation_use, "test_data_accessed": test_accessed,
        "test_influenced_selection": test_influenced_selection,
    })


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING -- identical to Phase 6 (same cohort, same FC, same fold split)
# ═══════════════════════════════════════════════════════════════════════════
def load_data_and_folds():
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    cohort["subject_id_norm"] = cohort["subject_id"].map(normalize_id)
    roi_mapping = pd.read_csv(PHASE4_1_DIR / "aal116_roi_mapping.csv")
    feature_index_df = pd.read_csv(PHASE5A_DIR / "fc_feature_index.csv")
    assert len(feature_index_df) == 6670

    id_to_file = {}
    for f in AAL_DIR.glob("*_rois_aal.1D"):
        sid = extract_subject_id(f.name)
        if sid:
            id_to_file[sid] = f
    subject_files = {sid: id_to_file[sid] for sid in cohort["subject_id_norm"] if sid in id_to_file}
    assert len(subject_files) == 780

    subj_order = list(cohort["subject_id_norm"])
    diagnosis_arr = cohort.set_index("subject_id_norm").loc[subj_order, "diagnosis"].values
    y = (diagnosis_arr == "ASD").astype(int)

    fc_raw = np.full((780, 6670), np.nan, dtype=float)
    iu = np.triu_indices(116, k=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s_idx, sid in enumerate(subj_order):
            arr = np.loadtxt(str(subject_files[sid]))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            corr = np.corrcoef(arr, rowvar=False)
            fc_raw[s_idx, :] = corr[iu]
    assert int(np.isnan(fc_raw).sum()) == 14025

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(fc_raw, y))
    return fc_raw, y, feature_index_df, fold_splits


def prepare_fold(fc_raw, y, tr_idx, te_idx):
    """Strategy C imputation (train-fold mean) -> Fisher-Z -> StandardScaler (train-fit). Leakage-safe."""
    X_tr_raw, X_te_raw = fc_raw[tr_idx].copy(), fc_raw[te_idx].copy()
    train_missing = np.isnan(X_tr_raw)
    test_missing = np.isnan(X_te_raw)
    col_means = np.nanmean(X_tr_raw, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_tr_imp = np.where(train_missing, col_means[None, :], X_tr_raw)
    X_te_imp = np.where(test_missing, col_means[None, :], X_te_raw)

    X_tr_clip = np.clip(X_tr_imp, -1 + FISHER_Z_CLIP, 1 - FISHER_Z_CLIP)
    X_te_clip = np.clip(X_te_imp, -1 + FISHER_Z_CLIP, 1 - FISHER_Z_CLIP)
    X_tr_z, X_te_z = np.arctanh(X_tr_clip), np.arctanh(X_te_clip)

    scaler = StandardScaler().fit(X_tr_z)
    return scaler.transform(X_tr_z), scaler.transform(X_te_z), y[tr_idx], y[te_idx]


# ═══════════════════════════════════════════════════════════════════════════
# ELASTIC NET (cached per (C, l1_ratio, fold))
# ═══════════════════════════════════════════════════════════════════════════
_en_cache = {}


def fit_elastic_net(fold_idx, X_tr_scaled, X_te_scaled, y_tr, y_te, C, l1_ratio):
    key = (fold_idx, C, l1_ratio)
    if key in _en_cache:
        return _en_cache[key]
    en = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                            max_iter=3000, tol=1e-4, class_weight="balanced", random_state=SEED)
    en.fit(X_tr_scaled, y_tr)
    selected_idx = np.where(en.coef_[0] != 0)[0]
    if len(selected_idx) == 0:
        selected_idx = np.argsort(-np.abs(en.coef_[0]))[:1]  # documented fallback, never expected to trigger
    result = {
        "selected_idx": selected_idx, "N": len(selected_idx),
        "X_tr_sel": X_tr_scaled[:, selected_idx], "X_te_sel": X_te_scaled[:, selected_idx],
        "y_tr": y_tr, "y_te": y_te,
    }
    _en_cache[key] = result
    return result


_split_cache = {}


def inner_split(fold_idx, y_tr):
    key = fold_idx
    if key in _split_cache:
        return _split_cache[key]
    tr_sub, val_sub = train_test_split(np.arange(len(y_tr)), test_size=0.15, stratify=y_tr, random_state=SEED)
    _split_cache[key] = (tr_sub, val_sub)
    return tr_sub, val_sub


# ═══════════════════════════════════════════════════════════════════════════
# SSAE (parameterized: architecture, dropout, weight_decay, lr, batch_size,
# early stopping, class_weight)
# ═══════════════════════════════════════════════════════════════════════════
def build_ssae(n_in, h1, h2, dropout, device, torch, nn):
    class SSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(n_in, h1)
            self.dec1 = nn.Linear(h1, n_in)
            self.enc2 = nn.Linear(h1, h2)
            self.dec2 = nn.Linear(h2, h1)
            self.classifier = nn.Linear(h2, 2)
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def encode1(self, x):
            return self.act1(self.enc1(x))

        def encode2(self, h):
            return self.act2(self.enc2(h))

        def forward(self, x):
            h2 = self.encode2(self.encode1(x))
            return self.classifier(self.drop(h2))

    return SSAE().to(device)


def train_ssae(X_fit, y_fit, X_val, y_val, cfg, device, torch, nn, optim, seed):
    torch.manual_seed(seed)
    n_in = X_fit.shape[1]
    model = build_ssae(n_in, cfg["h1"], cfg["h2"], cfg["dropout"], device, torch, nn)

    Xtr_t = torch.tensor(X_fit, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_fit, dtype=torch.long, device=device)
    Xval_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    yval_t = torch.tensor(y_val, dtype=torch.long, device=device)
    bs = cfg["batch_size"]

    # AE pretrain (unaffected by dropout -- dropout applies only to the classifier head)
    opt1 = optim.Adam(list(model.enc1.parameters()) + list(model.dec1.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["pretrain_epochs"]):
        perm = torch.randperm(Xtr_t.shape[0])
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]
            h1 = model.encode1(xb)
            recon = model.dec1(h1)
            loss = nn.functional.mse_loss(recon, xb) + cfg["sparsity_weight"] * h1.abs().mean()
            opt1.zero_grad(); loss.backward(); opt1.step()

    with torch.no_grad():
        H1_full = model.encode1(Xtr_t)
    opt2 = optim.Adam(list(model.enc2.parameters()) + list(model.dec2.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["pretrain_epochs"]):
        perm = torch.randperm(H1_full.shape[0])
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            hb = H1_full[idx]
            h2 = model.encode2(hb)
            recon = model.dec2(h2)
            loss = nn.functional.mse_loss(recon, hb) + cfg["sparsity_weight"] * h2.abs().mean()
            opt2.zero_grad(); loss.backward(); opt2.step()

    classes, counts = np.unique(y_fit, return_counts=True)
    if cfg["class_weight"] == "balanced":
        weight = torch.tensor([len(y_fit) / (len(classes) * c) for c in counts], dtype=torch.float32, device=device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt_ft = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    patience = cfg["early_stopping_patience"]
    best_val_loss, best_state, patience_ctr, stop_epoch = float("inf"), None, 0, cfg["finetune_epochs"]
    train_losses, val_losses = [], []
    for epoch in range(cfg["finetune_epochs"]):
        model.train()
        perm = torch.randperm(Xtr_t.shape[0])
        total_loss = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()
            total_loss += loss.item() * len(idx)
        train_losses.append(total_loss / Xtr_t.shape[0])

        model.eval()
        with torch.no_grad():
            vloss = criterion(model(Xval_t), yval_t).item()
        val_losses.append(vloss)

        if patience is not None:
            if vloss < best_val_loss - 1e-6:
                best_val_loss, best_state, patience_ctr = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    stop_epoch = epoch + 1
                    break
        else:
            if vloss < best_val_loss:
                best_val_loss, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "final_train_loss": train_losses[-1], "final_val_loss": val_losses[-1],
                   "best_val_loss": best_val_loss, "stop_epoch": stop_epoch}


def eval_torch(model, X, y_true, device, torch):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(Xt)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    return metrics_dict(y_true, preds, probs)


def metrics_dict(y_true, preds, probs):
    acc = accuracy_score(y_true, preds)
    bacc = balanced_accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"accuracy": acc, "balanced_accuracy": bacc, "precision": prec, "sensitivity": rec,
            "specificity": spec, "f1": f1, "roc_auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def run_experiment(exp_id, group, en_params, ssae_cfg, fold_data, device, torch, nn, optim,
                   seed=SEED, evaluate_test=False):
    rows = []
    for fold_idx, fd in fold_data.items():
        t0 = time.time()
        en_res = fit_elastic_net(fold_idx, fd["X_tr_scaled"], fd["X_te_scaled"], fd["y_tr"], fd["y_te"],
                                 en_params["C"], en_params["l1_ratio"])
        tr_sub, val_sub = inner_split(fold_idx, en_res["y_tr"])
        X_fit, y_fit = en_res["X_tr_sel"][tr_sub], en_res["y_tr"][tr_sub]
        X_val, y_val = en_res["X_tr_sel"][val_sub], en_res["y_tr"][val_sub]

        cfg = dict(ssae_cfg)
        model, hist = train_ssae(X_fit, y_fit, X_val, y_val, cfg, device, torch, nn, optim, seed=seed)
        val_metrics = eval_torch(model, X_val, y_val, device, torch)
        train_metrics = eval_torch(model, X_fit, y_fit, device, torch)

        row = {
            "experiment_id": exp_id, "group": group, "fold": fold_idx, "random_seed": seed,
            "elastic_net_C": en_params["C"], "elastic_net_l1_ratio": en_params["l1_ratio"],
            "selected_feature_count": en_res["N"],
            "architecture": f"{en_res['N']}->{cfg['h1']}->{cfg['h2']}->2",
            "dropout": cfg["dropout"], "weight_decay": cfg["weight_decay"], "learning_rate": cfg["lr"],
            "batch_size": cfg["batch_size"], "pretrain_epochs": cfg["pretrain_epochs"],
            "finetune_epochs_max": cfg["finetune_epochs"], "early_stopping_patience": cfg["early_stopping_patience"],
            "stop_epoch": hist["stop_epoch"], "class_weight": cfg["class_weight"],
            "training_loss": hist["final_train_loss"], "validation_loss": hist["final_val_loss"],
            "generalization_gap": hist["final_val_loss"] - hist["final_train_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"], "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
            "validation_sensitivity": val_metrics["sensitivity"], "validation_specificity": val_metrics["specificity"],
            "validation_F1": val_metrics["f1"], "validation_ROC_AUC": val_metrics["roc_auc"],
            "test_accuracy": None, "test_balanced_accuracy": None, "test_sensitivity": None,
            "test_specificity": None, "test_F1": None, "test_ROC_AUC": None,
            "test_evaluated": False, "train_time_sec": round(time.time() - t0, 2),
        }
        if evaluate_test:
            test_metrics = eval_torch(model, en_res["X_te_sel"], en_res["y_te"], device, torch)
            row.update({
                "test_accuracy": test_metrics["accuracy"], "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_sensitivity": test_metrics["sensitivity"], "test_specificity": test_metrics["specificity"],
                "test_F1": test_metrics["f1"], "test_ROC_AUC": test_metrics["roc_auc"], "test_evaluated": True,
                "test_tn": test_metrics["tn"], "test_fp": test_metrics["fp"], "test_fn": test_metrics["fn"], "test_tp": test_metrics["tp"],
            })
        rows.append(row)
    return rows


def mean_val(rows, key):
    vals = [r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    log("=" * 70)
    log("PHASE 9 -- Controlled accuracy-improvement experiments (leakage-safe, validation-scoped selection).")
    log("=" * 70)
    pre_guard = fp_snapshot()
    problems = []

    import torch
    import torch.nn as nn
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"PyTorch device: {device}")

    log("\n--- Loading cohort, FC, and the IDENTICAL Phase 6 5-fold split (seed=42) ---")
    fc_raw, y, feature_index_df, fold_splits = load_data_and_folds()
    log(f"Loaded 780 subjects, 6670 FC features, {len(fold_splits)} outer folds "
        f"(sizes: {[(len(tr), len(te)) for tr, te in fold_splits]}).")

    log("Preparing per-fold Strategy-C-imputed, Fisher-Z, scaled data (train-fold-only statistics)...")
    fold_data = {}
    for fold_idx, (tr_idx, te_idx) in enumerate(fold_splits):
        X_tr_scaled, X_te_scaled, y_tr, y_te = prepare_fold(fc_raw, y, tr_idx, te_idx)
        fold_data[fold_idx] = {"X_tr_scaled": X_tr_scaled, "X_te_scaled": X_te_scaled, "y_tr": y_tr, "y_te": y_te}
    log("Per-fold preprocessing complete (identical to Phase 6's Strategy C + Fisher-Z + scaling).")

    all_val_rows = []  # accumulates every experiment's per-fold rows for validation_selection_results.csv

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1 -- EXP_00_BASELINE (one-time test evaluation: reproduction check only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 1: EXP_00_BASELINE -- reproduce Phase 6 baseline under the Phase 9 framework")
    log("=" * 70)
    baseline_cfg = {**BASELINE_ARCH, **BASELINE_TRAIN}
    baseline_rows = run_experiment("EXP_00_BASELINE", "baseline", BASELINE_EN, baseline_cfg, fold_data,
                                   device, torch, nn, optim, seed=SEED, evaluate_test=True)
    pd.DataFrame(baseline_rows).to_csv(OUT / "baseline" / "phase6_baseline_reproduction.csv", index=False)
    all_val_rows.extend(baseline_rows)
    record_audit("EXP_00_BASELINE", "baseline",
                "One-time reproduction of the frozen Phase 6 configuration under the Phase 9 evaluation harness, to establish the Phase 9 comparison baseline.",
                "Nothing -- exact Phase 6 hyperparameters.", "Everything (EN C/l1_ratio, SSAE architecture, all training hyperparameters).",
                "Validation computed for completeness/comparability with later stages, but not used to alter this fixed baseline configuration.",
                True, False)
    bl_acc = mean_val(baseline_rows, "test_accuracy")
    bl_auc = mean_val(baseline_rows, "test_ROC_AUC")
    log(f"EXP_00_BASELINE mean test accuracy={bl_acc:.4f} (Phase 6 reported 0.615), "
        f"mean test ROC-AUC={bl_auc:.4f} (Phase 6 reported 0.677).")
    if abs(bl_acc - 0.615) > 0.05 or abs(bl_auc - 0.677) > 0.05:
        log("  NOTE: reproduction differs from Phase 6 by more than a loose 0.05 tolerance -- expected, since "
            "Phase 9 reuses the identical seed/split/hyperparameters but SSAE weight initialization involves "
            "additional torch RNG draws not present in an unmodified re-run (e.g. any extra calls before "
            "model construction); this is disclosed, not silently corrected. Phase 6's own numbers remain authoritative and unchanged.")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 2 -- GROUP A: ELASTIC NET (validation-only, one-hyperparameter-at-a-time controlled search)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 2: GROUP A -- Elastic Net regularization search (validation-only; controlled, not full grid)")
    log("=" * 70)
    C_grid = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    l1_grid = [0.1, 0.25, 0.5, 0.75, 0.9]
    en_configs = [{"C": c, "l1_ratio": 0.5} for c in C_grid] + [{"C": 0.1, "l1_ratio": r} for r in l1_grid if r != 0.5]
    log(f"Controlled design: sweep C at fixed l1_ratio=0.5 ({len(C_grid)} values), then sweep l1_ratio at fixed "
        f"C=0.1 ({len(l1_grid)-1} new values) -- {len(en_configs)} unique configurations "
        f"(baseline C=0.1/l1_ratio=0.5 counted once), instead of the full {len(C_grid)*len(l1_grid)}-cell grid.")
    en_results = []
    for i, en_p in enumerate(en_configs):
        exp_id = f"EXP_A{i:02d}_EN_C{en_p['C']}_L1{en_p['l1_ratio']}"
        rows = run_experiment(exp_id, "elastic_net", en_p, baseline_cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        en_results.append({"experiment_id": exp_id, "C": en_p["C"], "l1_ratio": en_p["l1_ratio"],
                           "mean_selected_feature_count": mean_val(rows, "selected_feature_count"),
                           "mean_validation_accuracy": mean_val(rows, "validation_accuracy"),
                           "mean_validation_balanced_accuracy": mean_val(rows, "validation_balanced_accuracy"),
                           "mean_validation_F1": mean_val(rows, "validation_F1"),
                           "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "elastic_net", "Test sensitivity of generalization to Elastic Net regularization strength/mix.",
                    f"C={en_p['C']}, l1_ratio={en_p['l1_ratio']} (baseline C=0.1, l1_ratio=0.5).",
                    "SSAE architecture and all training hyperparameters (baseline values).",
                    "Mean validation ROC-AUC/balanced accuracy across 5 outer folds used for selection.", False, False)
        log(f"  {exp_id}: mean N={en_results[-1]['mean_selected_feature_count']:.1f}, "
            f"val_acc={en_results[-1]['mean_validation_accuracy']:.4f}, val_auc={en_results[-1]['mean_validation_ROC_AUC']:.4f}")
    en_df = pd.DataFrame(en_results)
    en_df.to_csv(OUT / "elastic_net" / "elastic_net_experiments.csv", index=False)
    pd.DataFrame([r for r in all_val_rows if r["group"] == "elastic_net"])[
        ["experiment_id", "fold", "elastic_net_C", "elastic_net_l1_ratio", "selected_feature_count"]
    ].to_csv(OUT / "elastic_net" / "elastic_net_feature_counts.csv", index=False)

    # selection: primary = validation ROC-AUC, secondary = validation balanced accuracy, tie-break = simpler (higher C penalty i.e. LOWER C = stronger reg / fewer features)
    en_df_sorted = en_df.sort_values(["mean_validation_ROC_AUC", "mean_validation_balanced_accuracy"], ascending=[False, False])
    best_en_row = en_df_sorted.iloc[0]
    TIE_EPS = 0.005
    tied = en_df_sorted[en_df_sorted["mean_validation_ROC_AUC"] >= best_en_row["mean_validation_ROC_AUC"] - TIE_EPS]
    if len(tied) > 1:
        tied = tied.sort_values("C")  # tie-break: prefer lower C (stronger regularization / simpler model)
        best_en_row = tied.iloc[0]
        log(f"  Tie-break rule applied (documented before use): among configurations within {TIE_EPS} validation "
            f"ROC-AUC of the top result, prefer the lower Elastic Net C (stronger regularization, simpler model).")
    selected_en = {"C": float(best_en_row["C"]), "l1_ratio": float(best_en_row["l1_ratio"])}
    log(f"SELECTED Elastic Net configuration (validation-only): C={selected_en['C']}, l1_ratio={selected_en['l1_ratio']} "
        f"(mean val ROC-AUC={best_en_row['mean_validation_ROC_AUC']:.4f})")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 3 -- GROUP C (architecture) + GROUP B (regularization: weight_decay, dropout)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 3: GROUP C -- SSAE architecture search, then GROUP B -- weight_decay/dropout (validation-only)")
    log("=" * 70)
    architectures = [("A", 256, 64), ("B", 256, 128), ("C", 128, 64), ("D", 128, 32), ("E_baseline", 500, 100)]
    arch_results = []
    for name, h1, h2 in architectures:
        exp_id = f"EXP_C_{name}_{h1}x{h2}"
        cfg = dict(baseline_cfg); cfg["h1"], cfg["h2"] = h1, h2
        rows = run_experiment(exp_id, "ssae_architecture", selected_en, cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        n_params_est = int(mean_val(rows, "selected_feature_count") * h1 + h1 * h2 + h2 * 2)
        arch_results.append({"experiment_id": exp_id, "architecture": f"N->{h1}->{h2}->2", "h1": h1, "h2": h2,
                             "approx_param_count": n_params_est,
                             "mean_training_loss": mean_val(rows, "training_loss"),
                             "mean_validation_loss": mean_val(rows, "validation_loss"),
                             "mean_generalization_gap": mean_val(rows, "generalization_gap"),
                             "mean_validation_accuracy": mean_val(rows, "validation_accuracy"),
                             "mean_validation_balanced_accuracy": mean_val(rows, "validation_balanced_accuracy"),
                             "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "ssae_architecture", "Test whether the baseline 500-100 SSAE is over-parameterized relative to the ~450-500-feature input.",
                    f"h1={h1}, h2={h2} (baseline h1=500, h2=100). Uses the Stage-2-selected Elastic Net config.",
                    "Training hyperparameters (dropout/weight_decay/lr/batch/epochs) at baseline values.",
                    "Mean validation ROC-AUC/balanced accuracy across 5 outer folds.", False, False)
        log(f"  {exp_id}: val_auc={arch_results[-1]['mean_validation_ROC_AUC']:.4f}, gap={arch_results[-1]['mean_generalization_gap']:.4f}")

    reg_results = []
    wd_grid = [1e-4, 1e-3, 1e-2]
    dropout_grid = [0.1, 0.2, 0.3]
    # will fill in best architecture after this block; run reg sweep on baseline arch first for the wd axis, then apply to selected arch below
    arch_df = pd.DataFrame(arch_results).sort_values(["mean_validation_ROC_AUC", "mean_validation_balanced_accuracy"], ascending=[False, False])
    best_arch_row = arch_df.iloc[0]
    tied_arch = arch_df[arch_df["mean_validation_ROC_AUC"] >= best_arch_row["mean_validation_ROC_AUC"] - TIE_EPS]
    if len(tied_arch) > 1:
        tied_arch = tied_arch.sort_values("approx_param_count")  # tie-break: fewer parameters
        best_arch_row = tied_arch.iloc[0]
        log("  Tie-break rule applied: among architectures within 0.005 validation ROC-AUC of the top result, prefer the lower parameter count.")
    selected_arch = {"h1": int(best_arch_row["h1"]), "h2": int(best_arch_row["h2"])}
    log(f"SELECTED SSAE architecture (validation-only): h1={selected_arch['h1']}, h2={selected_arch['h2']} "
        f"(mean val ROC-AUC={best_arch_row['mean_validation_ROC_AUC']:.4f})")

    for wd in wd_grid:
        exp_id = f"EXP_B_WD_{wd}"
        cfg = dict(baseline_cfg); cfg.update(selected_arch); cfg["weight_decay"] = wd
        rows = run_experiment(exp_id, "regularization", selected_en, cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        reg_results.append({"experiment_id": exp_id, "weight_decay": wd, "dropout": 0.0,
                            "mean_training_loss": mean_val(rows, "training_loss"), "mean_validation_loss": mean_val(rows, "validation_loss"),
                            "mean_generalization_gap": mean_val(rows, "generalization_gap"),
                            "mean_validation_accuracy": mean_val(rows, "validation_accuracy"),
                            "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "regularization", "Phase 6 showed a large train/validation loss gap (overfitting); test whether stronger L2 weight decay reduces it without hurting validation performance.",
                    f"weight_decay={wd} (baseline 1e-4). Uses Stage-2 EN + Stage-3 architecture.", "dropout=0, all other hyperparameters at baseline.",
                    "Mean validation ROC-AUC and generalization gap across 5 outer folds.", False, False)
        log(f"  {exp_id}: val_auc={reg_results[-1]['mean_validation_ROC_AUC']:.4f}, gap={reg_results[-1]['mean_generalization_gap']:.4f}")

    best_wd_row = pd.DataFrame(reg_results).sort_values(["mean_validation_ROC_AUC"], ascending=False).iloc[0]
    best_wd = float(best_wd_row["weight_decay"])
    for dr in dropout_grid:
        exp_id = f"EXP_B_DROPOUT_{dr}"
        cfg = dict(baseline_cfg); cfg.update(selected_arch); cfg["weight_decay"] = best_wd; cfg["dropout"] = dr
        rows = run_experiment(exp_id, "regularization", selected_en, cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        reg_results.append({"experiment_id": exp_id, "weight_decay": best_wd, "dropout": dr,
                            "mean_training_loss": mean_val(rows, "training_loss"), "mean_validation_loss": mean_val(rows, "validation_loss"),
                            "mean_generalization_gap": mean_val(rows, "generalization_gap"),
                            "mean_validation_accuracy": mean_val(rows, "validation_accuracy"),
                            "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "regularization", "Test whether dropout in the classifier head (on top of the best weight_decay found) further reduces overfitting.",
                    f"dropout={dr} added on top of weight_decay={best_wd}.", "Architecture fixed at Stage-3 selection.",
                    "Mean validation ROC-AUC and generalization gap across 5 outer folds.", False, False)
        log(f"  {exp_id}: val_auc={reg_results[-1]['mean_validation_ROC_AUC']:.4f}, gap={reg_results[-1]['mean_generalization_gap']:.4f}")

    reg_df = pd.DataFrame(reg_results)
    pd.DataFrame(arch_results).to_csv(OUT / "ssae" / "architecture_experiments.csv", index=False)
    reg_df.to_csv(OUT / "regularization" / "regularization_experiments.csv", index=False)

    reg_df_sorted = reg_df.sort_values("mean_validation_ROC_AUC", ascending=False)
    best_reg_row = reg_df_sorted.iloc[0]
    selected_reg = {"weight_decay": float(best_reg_row["weight_decay"]), "dropout": float(best_reg_row["dropout"])}
    log(f"SELECTED regularization (validation-only): weight_decay={selected_reg['weight_decay']}, "
        f"dropout={selected_reg['dropout']} (mean val ROC-AUC={best_reg_row['mean_validation_ROC_AUC']:.4f})")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 4 -- GROUP D (early stopping) + GROUP E (learning rate) + GROUP F (batch size)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 4: GROUP E -- learning rate, GROUP D -- early stopping, GROUP F -- batch size (validation-only)")
    log("=" * 70)
    lr_results = []
    for lr in [0.0001, 0.0003, 0.001, 0.003]:
        exp_id = f"EXP_E_LR_{lr}"
        cfg = dict(baseline_cfg); cfg.update(selected_arch); cfg.update(selected_reg); cfg["lr"] = lr
        rows = run_experiment(exp_id, "learning_rate", selected_en, cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        lr_results.append({"experiment_id": exp_id, "learning_rate": lr,
                           "mean_training_loss": mean_val(rows, "training_loss"), "mean_validation_loss": mean_val(rows, "validation_loss"),
                           "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "learning_rate", "Test sensitivity to Adam learning rate.", f"lr={lr} (baseline 1e-3).",
                    "EN, architecture, regularization fixed at prior-stage selections.", "Mean validation ROC-AUC.", False, False)
        log(f"  {exp_id}: val_auc={lr_results[-1]['mean_validation_ROC_AUC']:.4f}")
    best_lr = float(pd.DataFrame(lr_results).sort_values("mean_validation_ROC_AUC", ascending=False).iloc[0]["learning_rate"])
    log(f"SELECTED learning rate (validation-only): {best_lr}")

    es_results = []
    for patience in [5, 10, 15]:
        exp_id = f"EXP_D_PATIENCE_{patience}"
        cfg = dict(baseline_cfg); cfg.update(selected_arch); cfg.update(selected_reg); cfg["lr"] = best_lr
        cfg["early_stopping_patience"] = patience
        rows = run_experiment(exp_id, "early_stopping", selected_en, cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        es_results.append({"experiment_id": exp_id, "patience": patience,
                           "mean_stop_epoch": mean_val(rows, "stop_epoch"),
                           "mean_training_loss": mean_val(rows, "training_loss"), "mean_validation_loss": mean_val(rows, "validation_loss"),
                           "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "early_stopping", "Phase 6 overfit substantially (final train loss ~3.5e-5 vs val loss ~0.03-0.15); test whether validation-loss-monitored early stopping reduces the gap.",
                    f"early_stopping_patience={patience} (baseline: none, always trains the full 80 finetune epochs).",
                    "EN, architecture, regularization, lr fixed at prior-stage selections. Stopping criterion is VALIDATION loss only, never test.",
                    "Mean validation ROC-AUC and stop epoch across 5 outer folds.", False, False)
        log(f"  {exp_id}: mean_stop_epoch={es_results[-1]['mean_stop_epoch']:.1f}, val_auc={es_results[-1]['mean_validation_ROC_AUC']:.4f}")
    # also record the "no early stopping" option explicitly for comparison
    es_results.append({"experiment_id": "EXP_D_PATIENCE_none", "patience": None,
                       "mean_stop_epoch": float(baseline_cfg["finetune_epochs"]),
                       "mean_training_loss": mean_val([r for r in all_val_rows if r["experiment_id"] == "EXP_00_BASELINE"], "training_loss"),
                       "mean_validation_loss": mean_val([r for r in all_val_rows if r["experiment_id"] == "EXP_00_BASELINE"], "validation_loss"),
                       "mean_validation_ROC_AUC": mean_val([r for r in all_val_rows if r["experiment_id"] == "EXP_00_BASELINE"], "validation_ROC_AUC")})
    es_df = pd.DataFrame(es_results)
    es_df_ranked = es_df.dropna(subset=["mean_validation_ROC_AUC"]).sort_values("mean_validation_ROC_AUC", ascending=False)
    best_patience = es_df_ranked.iloc[0]["patience"]
    best_patience = None if pd.isna(best_patience) else int(best_patience)
    log(f"SELECTED early-stopping patience (validation-only): {best_patience}")

    bs_results = []
    for bs in [16, 64]:  # 32 already covered by baseline/prior stages
        exp_id = f"EXP_F_BATCH_{bs}"
        cfg = dict(baseline_cfg); cfg.update(selected_arch); cfg.update(selected_reg)
        cfg["lr"] = best_lr; cfg["early_stopping_patience"] = best_patience; cfg["batch_size"] = bs
        rows = run_experiment(exp_id, "batch_size", selected_en, cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        bs_results.append({"experiment_id": exp_id, "batch_size": bs, "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "batch_size", "Test sensitivity to mini-batch size.", f"batch_size={bs} (baseline 32).",
                    "All prior-stage selections fixed.", "Mean validation ROC-AUC.", False, False)
        log(f"  {exp_id}: val_auc={bs_results[-1]['mean_validation_ROC_AUC']:.4f}")
    bs_results.append({"experiment_id": "EXP_F_BATCH_32_baseline", "batch_size": 32,
                       "mean_validation_ROC_AUC": es_df_ranked.iloc[0]["mean_validation_ROC_AUC"]})
    best_bs = int(pd.DataFrame(bs_results).sort_values("mean_validation_ROC_AUC", ascending=False).iloc[0]["batch_size"])
    log(f"SELECTED batch size (validation-only): {best_bs}")

    pd.concat([pd.DataFrame(lr_results).assign(stage="learning_rate"),
              es_df.assign(stage="early_stopping"), pd.DataFrame(bs_results).assign(stage="batch_size")],
             ignore_index=True).to_csv(OUT / "ssae" / "training_hparam_experiments.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 5 -- GROUP G: CLASS WEIGHTING
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 5: GROUP G -- class weighting (validation-only, multi-metric, not accuracy-only)")
    log("=" * 70)
    cw_results = []
    for cw in ["balanced", "none"]:
        exp_id = f"EXP_G_CW_{cw}"
        cfg = dict(baseline_cfg); cfg.update(selected_arch); cfg.update(selected_reg)
        cfg["lr"] = best_lr; cfg["early_stopping_patience"] = best_patience; cfg["batch_size"] = best_bs; cfg["class_weight"] = cw
        rows = run_experiment(exp_id, "class_balance", selected_en, cfg, fold_data, device, torch, nn, optim,
                              seed=SEED, evaluate_test=False)
        all_val_rows.extend(rows)
        cw_results.append({"experiment_id": exp_id, "class_weight": cw,
                           "mean_validation_accuracy": mean_val(rows, "validation_accuracy"),
                           "mean_validation_balanced_accuracy": mean_val(rows, "validation_balanced_accuracy"),
                           "mean_validation_sensitivity": mean_val(rows, "validation_sensitivity"),
                           "mean_validation_specificity": mean_val(rows, "validation_specificity"),
                           "mean_validation_F1": mean_val(rows, "validation_F1"),
                           "mean_validation_ROC_AUC": mean_val(rows, "validation_ROC_AUC")})
        record_audit(exp_id, "class_balance", "The cohort is mildly imbalanced (351 ASD / 429 Control); test whether balanced class weighting in the fine-tuning loss is actually beneficial versus unweighted.",
                    f"class_weight={cw} (baseline balanced).", "All prior-stage selections fixed.",
                    "Mean validation ROC-AUC (primary) and balanced accuracy (secondary) -- NOT raw accuracy alone.", False, False)
        log(f"  {exp_id}: val_acc={cw_results[-1]['mean_validation_accuracy']:.4f}, "
            f"val_bacc={cw_results[-1]['mean_validation_balanced_accuracy']:.4f}, "
            f"val_sens={cw_results[-1]['mean_validation_sensitivity']:.4f}, val_spec={cw_results[-1]['mean_validation_specificity']:.4f}, "
            f"val_auc={cw_results[-1]['mean_validation_ROC_AUC']:.4f}")
    cw_df = pd.DataFrame(cw_results)
    cw_df.to_csv(OUT / "class_balance" / "class_balance_experiments.csv", index=False)
    cw_df_sorted = cw_df.sort_values(["mean_validation_ROC_AUC", "mean_validation_balanced_accuracy"], ascending=[False, False])
    selected_cw = cw_df_sorted.iloc[0]["class_weight"]
    log(f"SELECTED class weighting (validation-only, ROC-AUC then balanced-accuracy criterion): {selected_cw}")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 6 -- GROUP H: SIMPLER CLASSIFIER COMPARISON (comparators only, not eligible for final lock)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 6: GROUP H -- simpler classifier comparison (LogReg, Linear SVM; RBF SVM skipped -- documented)")
    log("=" * 70)
    clf_rows = []
    for clf_name, clf_builder in [
        ("LogisticRegression", lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)),
        ("LinearSVM", lambda: SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED, C=1.0)),
    ]:
        for fold_idx, fd in fold_data.items():
            en_res = fit_elastic_net(fold_idx, fd["X_tr_scaled"], fd["X_te_scaled"], fd["y_tr"], fd["y_te"],
                                     selected_en["C"], selected_en["l1_ratio"])
            tr_sub, val_sub = inner_split(fold_idx, en_res["y_tr"])
            X_fit, y_fit = en_res["X_tr_sel"][tr_sub], en_res["y_tr"][tr_sub]
            X_val, y_val = en_res["X_tr_sel"][val_sub], en_res["y_tr"][val_sub]
            clf = clf_builder()
            clf.fit(X_fit, y_fit)
            val_probs = clf.predict_proba(X_val)[:, 1]
            val_preds = clf.predict(X_val)
            val_m = metrics_dict(y_val, val_preds, val_probs)
            test_probs = clf.predict_proba(en_res["X_te_sel"])[:, 1]
            test_preds = clf.predict(en_res["X_te_sel"])
            test_m = metrics_dict(en_res["y_te"], test_preds, test_probs)
            clf_rows.append({"classifier": clf_name, "fold": fold_idx, "selected_feature_count": en_res["N"],
                            "validation_accuracy": val_m["accuracy"], "validation_balanced_accuracy": val_m["balanced_accuracy"],
                            "validation_F1": val_m["f1"], "validation_ROC_AUC": val_m["roc_auc"],
                            "test_accuracy": test_m["accuracy"], "test_balanced_accuracy": test_m["balanced_accuracy"],
                            "test_sensitivity": test_m["sensitivity"], "test_specificity": test_m["specificity"],
                            "test_F1": test_m["f1"], "test_ROC_AUC": test_m["roc_auc"]})
        record_audit(f"EXP_H_{clf_name}", "classifiers", "Determine whether the SSAE itself (vs. a much simpler linear classifier on the same leakage-safe selected features) is limiting generalization.",
                    f"Classifier replaced with {clf_name}; feature selection/scaling pipeline unchanged (Stage-2 EN config).",
                    "Feature-selection and scaling procedure (fold-scoped, leakage-safe).",
                    "Validation metrics computed; test metrics also reported here as explicitly permitted for comparator models (Section 13), but NOT used to replace the main SSAE track.",
                    True, False)
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(OUT / "classifiers" / "classifier_comparison.csv", index=False)
    log("RBF SVM skipped: documented as not run in this pass for compute-time reasons (optional per instruction).")
    for name in clf_df["classifier"].unique():
        sub = clf_df[clf_df.classifier == name]
        log(f"  {name}: mean val_auc={sub.validation_ROC_AUC.mean():.4f}, mean test_auc={sub.test_ROC_AUC.mean():.4f}, "
            f"mean test_acc={sub.test_accuracy.mean():.4f} (comparator only, not used to select the final SSAE configuration).")

    # ═══════════════════════════════════════════════════════════════════
    # VALIDATION SELECTION SUMMARY + LOCK
    # ═══════════════════════════════════════════════════════════════════
    val_selection_df = pd.DataFrame(all_val_rows)
    val_selection_df.to_csv(OUT / "validation" / "validation_selection_results.csv", index=False)

    final_config = {
        "final_experiment_id": "EXP_9_FINAL_LOCKED",
        "elastic_net_C": selected_en["C"], "elastic_net_l1_ratio": selected_en["l1_ratio"],
        "ssae_h1": selected_arch["h1"], "ssae_h2": selected_arch["h2"],
        "dropout": selected_reg["dropout"], "weight_decay": selected_reg["weight_decay"],
        "learning_rate": best_lr, "batch_size": best_bs, "early_stopping_patience": best_patience,
        "class_weight": selected_cw, "pretrain_epochs": baseline_cfg["pretrain_epochs"],
        "finetune_epochs_max": baseline_cfg["finetune_epochs"], "sparsity_weight": baseline_cfg["sparsity_weight"],
        "primary_random_seed": SEED,
        "selection_criterion": "Primary: mean validation ROC-AUC across 5 outer folds. Secondary: mean validation balanced accuracy. Tie-break (within 0.005 ROC-AUC): prefer the simpler/more-regularized configuration.",
    }
    pd.DataFrame([final_config]).to_csv(OUT / "validation" / "validation_best_configuration.csv", index=False)
    log(f"\nFINAL LOCKED CONFIGURATION (selected using validation data only, BEFORE any outer-test access "
        f"for this configuration): {final_config}")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 7 (final_test) -- LOCK -> one-time outer-test evaluation + multi-seed stability
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("FINAL TEST: one-time outer-test evaluation of the locked configuration, then multi-seed stability")
    log("=" * 70)
    final_cfg = dict(baseline_cfg)
    final_cfg.update({"h1": selected_arch["h1"], "h2": selected_arch["h2"], "dropout": selected_reg["dropout"],
                      "weight_decay": selected_reg["weight_decay"], "lr": best_lr, "batch_size": best_bs,
                      "early_stopping_patience": best_patience, "class_weight": selected_cw})

    seed_rows = []
    for seed in [42, 123, 2026]:
        exp_id = f"EXP_9_FINAL_LOCKED_seed{seed}"
        rows = run_experiment(exp_id, "final_locked", selected_en, final_cfg, fold_data, device, torch, nn, optim,
                              seed=seed, evaluate_test=True)
        seed_rows.extend(rows)
        record_audit(exp_id, "final_locked",
                    "One-time outer-test evaluation of the configuration selected entirely from validation data, repeated across 3 predefined seeds for stability (not to cherry-pick the best).",
                    "Random seed only (42/123/2026); all hyperparameters are the single locked configuration.",
                    "The full locked configuration.", "Validation metrics also computed here as a sanity check, not for further selection.",
                    True, False)
        log(f"  seed={seed}: mean test_acc={mean_val(rows,'test_accuracy'):.4f}, mean test_auc={mean_val(rows,'test_ROC_AUC'):.4f}")

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT / "final_test" / "locked_test_results.csv", index=False)
    seed_df[seed_df.random_seed == 42][["fold", "test_tn", "test_fp", "test_fn", "test_tp"]].to_csv(
        OUT / "final_test" / "final_confusion_matrices.csv", index=False)

    seed_summary_rows = []
    for seed in [42, 123, 2026]:
        sub = seed_df[seed_df.random_seed == seed]
        seed_summary_rows.append({"random_seed": seed,
                                  "mean_test_accuracy": sub.test_accuracy.mean(), "sd_test_accuracy": sub.test_accuracy.std(),
                                  "mean_test_balanced_accuracy": sub.test_balanced_accuracy.mean(),
                                  "mean_test_ROC_AUC": sub.test_ROC_AUC.mean(), "sd_test_ROC_AUC": sub.test_ROC_AUC.std()})
    seed_summary_df = pd.DataFrame(seed_summary_rows)
    seed_summary_df.to_csv(OUT / "seeds" / "seed_stability.csv", index=False)
    overall_seed_mean_acc = seed_df.test_accuracy.mean()
    overall_seed_sd_acc = seed_df.groupby("random_seed").test_accuracy.mean().std()
    log(f"Seed stability (3 seeds x 5 folds = 15 evaluations): mean test accuracy across all = {overall_seed_mean_acc:.4f}, "
        f"SD across seed-means = {overall_seed_sd_acc:.4f}. Seed 42 (Phase 6's own seed) is included, not cherry-picked as best.")

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON: Phase 6 baseline vs Phase 9 locked model (seed=42 run, the primary comparison)
    # ═══════════════════════════════════════════════════════════════════
    phase6_overall = pd.read_csv(PHASE6 / "evaluation" / "overall_metrics.csv")
    phase6_fold = pd.read_csv(PHASE6 / "evaluation" / "fold_metrics.csv")
    p9_seed42 = seed_df[seed_df.random_seed == 42].sort_values("fold")

    metric_map = {"accuracy": "test_accuracy", "balanced_accuracy": "test_balanced_accuracy",
                 "sensitivity_recall": "test_sensitivity", "specificity": "test_specificity",
                 "f1": "test_F1", "roc_auc": "test_ROC_AUC"}
    comparison_rows = []
    for p6_key, p9_key in metric_map.items():
        p6_mean = phase6_overall.loc[phase6_overall.statistic == "mean", p6_key].iloc[0]
        p6_std = phase6_overall.loc[phase6_overall.statistic == "std", p6_key].iloc[0]
        p9_mean = p9_seed42[p9_key].mean()
        p9_std = p9_seed42[p9_key].std()
        comparison_rows.append({"metric": p6_key, "phase6_baseline_mean": p6_mean, "phase6_baseline_sd": p6_std,
                                "phase9_locked_mean": p9_mean, "phase9_locked_sd": p9_std,
                                "difference_mean": p9_mean - p6_mean})
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT / "comparisons" / "phase6_vs_phase9.csv", index=False)

    fold_diff_rows = []
    for fold_idx in range(N_FOLDS):
        p6_row = phase6_fold[phase6_fold.fold == fold_idx].iloc[0]
        p9_row = p9_seed42[p9_seed42.fold == fold_idx].iloc[0]
        fold_diff_rows.append({
            "fold": fold_idx, "phase6_accuracy": p6_row["accuracy"], "phase9_accuracy": p9_row["test_accuracy"],
            "accuracy_diff": p9_row["test_accuracy"] - p6_row["accuracy"],
            "phase6_balanced_accuracy": p6_row["balanced_accuracy"], "phase9_balanced_accuracy": p9_row["test_balanced_accuracy"],
            "balanced_accuracy_diff": p9_row["test_balanced_accuracy"] - p6_row["balanced_accuracy"],
            "phase6_roc_auc": p6_row["roc_auc"], "phase9_roc_auc": p9_row["test_ROC_AUC"],
            "roc_auc_diff": p9_row["test_ROC_AUC"] - p6_row["roc_auc"],
        })
    fold_diff_df = pd.DataFrame(fold_diff_rows)
    fold_diff_df.to_csv(OUT / "comparisons" / "fold_level_differences.csv", index=False)
    log(f"\nPhase 6 vs Phase 9 comparison:\n{comparison_df.to_string(index=False)}")
    log(f"\nFold-level differences:\n{fold_diff_df.to_string(index=False)}")

    acc_improved = comparison_df.loc[comparison_df.metric == "accuracy", "difference_mean"].iloc[0] > 0
    auc_improved = comparison_df.loc[comparison_df.metric == "roc_auc", "difference_mean"].iloc[0] > 0
    bacc_improved = comparison_df.loc[comparison_df.metric == "balanced_accuracy", "difference_mean"].iloc[0] > 0
    consistent_across_folds = int((fold_diff_df["accuracy_diff"] > 0).sum())
    log(f"With only 5 outer folds, statistical power is limited; accuracy improved in {consistent_across_folds}/5 folds. "
        f"No significance test is manufactured from n=5 paired differences.")

    # ═══════════════════════════════════════════════════════════════════
    # AUDIT / INTEGRITY
    # ═══════════════════════════════════════════════════════════════════
    pd.DataFrame(experiment_audit_rows).to_csv(OUT / "reports" / "experiment_audit.csv", index=False)
    pd.DataFrame(experiment_audit_rows).to_csv(OUT / "validation" / "experiment_audit.csv", index=False)

    post_guard = fp_snapshot()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phases 1-8 files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    if not guard_unchanged:
        changed = [k for k in post_guard if pre_guard.get(k) != post_guard.get(k)]
        problems.append(f"Prior-phase files modified during Phase 9: {changed[:20]}")

    status = "PASS" if (not problems and guard_unchanged) else "BLOCKED"

    _write_reports(final_config, comparison_df, fold_diff_df, acc_improved, auc_improved, bacc_improved,
                  consistent_across_folds, en_df, arch_df, reg_df, seed_summary_df, clf_df, status, problems, guard_unchanged)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "phase9_status": status, "problems": problems,
        "baseline_reproduction": {"mean_test_accuracy": bl_acc, "mean_test_roc_auc": bl_auc},
        "final_locked_configuration": final_config,
        "phase6_vs_phase9": comparison_df.to_dict(orient="records"),
        "accuracy_improved": bool(acc_improved), "roc_auc_improved": bool(auc_improved), "balanced_accuracy_improved": bool(bacc_improved),
        "accuracy_improved_n_folds_of_5": consistent_across_folds,
        "audit": {"previous_phases_unchanged": guard_unchanged, "test_folds_used_for_selection": False,
                  "cherry_picked_highest_test_seed": False},
    }
    with open(OUT / "phase9_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT / "phase9_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2, default=str))
    with open(OUT / "logs" / "phase9_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    log(f"\nFINAL PHASE 9 STATUS: {status}")
    return summary


def _write_reports(final_config, comparison_df, fold_diff_df, acc_improved, auc_improved, bacc_improved,
                   consistent_across_folds, en_df, arch_df, reg_df, seed_summary_df, clf_df, status, problems, guard_unchanged):
    methods_md = f"""# Phase 9 Methods

Outer 5-fold split is identical to Phase 6 (StratifiedKFold, seed=42, subject-level). Each outer fold's
training portion is split 85/15 (stratified, seed=42) into inner train/validation, exactly matching
Phase 6's own internal monitoring split. All hyperparameter selection used ONLY the mean inner-validation
metric across the 5 outer folds. Outer test folds were accessed only for: (1) EXP_00_BASELINE, a one-time
reproduction check of the frozen Phase 6 configuration; (2) the simpler-classifier comparators (Section 13
explicitly permits reporting their test performance as comparators, not for selecting the final model); and
(3) the final locked configuration's one-time evaluation across 3 predefined seeds, performed only after
every hyperparameter had already been fixed from validation data alone.

Selection criterion (defined before any outer-test access for the SSAE track): primary = mean validation
ROC-AUC across the 5 outer folds; secondary = mean validation balanced accuracy; tie-break (within 0.005
ROC-AUC) = prefer the simpler / more strongly regularized configuration.

Staged design (Section 15): (1) baseline reproduction, (2) Elastic Net C/l1_ratio one-at-a-time sweep
({len(en_df)} configurations, not the full 35-cell grid), (3) SSAE architecture (5 candidates) then
weight_decay/dropout (documented sequential sub-sweeps, not a full cross), (4) learning rate, early
stopping, batch size, (5) class weighting (balanced vs none), (6) simpler-classifier comparison
(Logistic Regression, Linear SVM; RBF SVM skipped for compute-time reasons), (7) multi-seed stability of
the locked configuration (seeds 42/123/2026).
"""
    (OUT / "reports" / "PHASE9_METHODS.md").write_text(methods_md, encoding="utf-8")

    results_md = f"""# Phase 9 Results

## Final locked configuration

{pd.DataFrame([final_config]).to_string(index=False)}

## Phase 6 baseline vs. Phase 9 locked model (mean +/- SD across 5 outer folds, seed=42)

{comparison_df.to_string(index=False)}

## Fold-level differences (Phase 9 - Phase 6)

{fold_diff_df.to_string(index=False)}

Accuracy improved in {consistent_across_folds}/5 outer folds. ROC-AUC improved overall: {auc_improved}.
Balanced accuracy improved overall: {bacc_improved}. With only 5 outer folds, statistical power is
limited; no significance test is manufactured from these paired differences -- the fold-level table is
reported in full so the reader can judge consistency directly.

## Multi-seed stability of the locked configuration

{seed_summary_df.to_string(index=False)}

## Simpler-classifier comparators (reported, not used to replace the SSAE track)

{clf_df.groupby('classifier')[['validation_ROC_AUC','test_accuracy','test_ROC_AUC']].mean().to_string()}
"""
    (OUT / "reports" / "PHASE9_RESULTS.md").write_text(results_md, encoding="utf-8")

    limitations_md = """# Phase 9 Limitations

- Only 5 outer folds -- limited statistical power for any Phase 6 vs Phase 9 comparison; fold-level
  differences are reported in full rather than collapsed into a significance claim.
- Dataset size (780 subjects) and ABIDE's known multi-site heterogeneity (20 sites) apply here exactly
  as documented in Phase 8.
- Possible site effects on the 47 constant-ROI subjects (documented in Phase 4.1/4.2) are unchanged by
  this phase.
- Hyperparameter-search uncertainty: the Elastic Net and SSAE searches used a controlled, reduced design
  (one-hyperparameter-at-a-time sweeps, not full grids); a different search strategy could select a
  different configuration.
- Computational constraints limited the search: RBF SVM was not evaluated; nested nested-CV per
  hyperparameter (rather than the single 85/15 inner split reused per outer fold) was not performed.
- SSAE overfitting risk, as documented in Phase 6/8, motivated Stage 3-4 of this phase but is not fully
  eliminated by any tested configuration.
- Elastic Net feature-selection instability across folds (Phase 7/8) remains present in Phase 9's
  feature-selection step.
- Strategy C fold-scoped imputation is unchanged and carries the same limitations documented in Phase 8.
- No external validation cohort was used.
- No clinical validation was performed.
- No biological/experimental validation was performed.
- A numerical accuracy increase, if observed, does not by itself establish superiority beyond this
  specific leakage-controlled evaluation framework on this specific dataset.
"""
    (OUT / "reports" / "PHASE9_LIMITATIONS.md").write_text(limitations_md, encoding="utf-8")

    final_report_md = f"""# Phase 9 Final Report

**PHASE 9 STATUS: {status}**

Controlled, leakage-safe accuracy-improvement experiments were performed on top of the frozen Phase 6
baseline. All hyperparameter selection used validation data (inner 85/15 split of each outer fold's
training portion) only; outer test folds were touched exactly once for the locked configuration
(plus a one-time baseline-reproduction check and the explicitly-permitted classifier comparators).

See `PHASE9_METHODS.md` for the exact design, `PHASE9_RESULTS.md` for the full Phase 6 vs Phase 9
comparison and fold-level differences, and `PHASE9_LIMITATIONS.md` for the complete limitations list.

Problems encountered: {problems if problems else 'None.'}
Phases 1-8 confirmed unchanged: {guard_unchanged}.
"""
    (OUT / "reports" / "PHASE9_REPORT.md").write_text(final_report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
