"""
phase10_advanced_accuracy.py -- Phase 10: nested-CV accuracy-improvement
experiments, addressing the Phase 9 finding that a single 85/15 inner
validation split gave near-ceiling (~0.99 AUC) validation scores that did not
predict outer-test performance (~0.69 AUC). Phase 10 replaces that single
split with a genuine inner K-fold (StratifiedKFold, k=2, seed=42) nested
inside each outer training fold for the two experiment groups most directly
tied to that gap (Elastic Net regularization, SSAE architecture/model size).
Remaining axes use a single controlled global sweep (documented explicitly),
consistent with the project's established practice of disclosing every scope
reduction rather than silently running a smaller design.

Outer test folds are touched exactly twice: once for the final locked
configuration's one-time evaluation, and once for the separate site-grouped
secondary evaluation (which reuses the locked configuration -- no new
hyperparameter selection happens there).

Phases 1-9 are read-only inputs and are never modified.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"C:\abide")
PHASE2_FILTERED = ROOT / "results" / "phase2" / "fd_filtered_subjects.csv"
AAL_DIR = ROOT / "abide_pcp" / "ABIDE_pcp" / "cpac" / "filt_global"
PHASE4_1_DIR = ROOT / "results" / "phase4_1_constant_roi_audit"
PHASE5A_DIR = ROOT / "results" / "phase5a_fc_diagnostic"
PHASE6 = ROOT / "results" / "phase6_strategyC_all_nonzero"
PHASE9 = ROOT / "results" / "phase9_accuracy_optimization"
OUT = ROOT / "results" / "phase10_advanced_accuracy"
for sub in ["nested_cv", "elastic_net", "feature_stability", "classifiers", "ssae", "regularization",
            "representation", "site_aware", "seeds", "final", "comparisons", "reports", "logs"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

GUARD_DIRS = [ROOT / "results" / d for d in [
    "phase1", "phase2", "phase3", "phase3_5", "phase3_6", "phase3_6_verification",
    "phase3_6_aal_download", "phase4", "phase4_1_constant_roi_audit",
    "phase4_2_methodological_sensitivity", "phase5a_fc_diagnostic", "phase5b_missing_fc_analysis",
    "phase5c_elastic_net_comparison", "phase5d_elastic_net_all_nonzero",
    "phase6_strategyC_all_nonzero", "phase7_synthesis", "phase8_final_audit", "phase9_accuracy_optimization",
]]

SEED = 42
N_OUTER = 5
N_INNER = 2  # inner K-fold within each outer training fold -- reduced from a larger k for compute tractability, documented
FISHER_Z_CLIP = 1e-6

BASELINE_TRAIN = {"dropout": 0.0, "weight_decay": 1e-4, "lr": 1e-3, "batch_size": 32,
                  "pretrain_epochs": 40, "finetune_epochs": 80, "sparsity_weight": 1e-4,
                  "early_stopping_patience": None, "class_weight": "balanced"}

audit_lines = []
manifest_rows = []
leakage_audit_rows = []


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


def record_leakage(stage, test_data_accessed, test_label_accessed, imputation_train_only,
                   scaling_train_only, pca_train_only, feature_selection_train_only, hparam_inner_cv_only, notes=""):
    leakage_audit_rows.append({
        "stage": stage, "test_data_accessed": test_data_accessed, "test_label_accessed": test_label_accessed,
        "imputation_fitted_train_only": imputation_train_only, "scaling_fitted_train_only": scaling_train_only,
        "pca_fitted_train_only": pca_train_only, "feature_selection_fitted_train_only": feature_selection_train_only,
        "hyperparameter_selection_inner_cv_only": hparam_inner_cv_only, "notes": notes,
    })


def manifest_row(**kwargs):
    row = {k: kwargs.get(k, "N/A") for k in [
        "experiment_id", "outer_fold", "inner_fold", "model", "elastic_net_C", "elastic_net_l1_ratio",
        "feature_selection_method", "feature_count", "architecture", "dropout", "weight_decay",
        "learning_rate", "batch_size", "early_stopping", "pca_setting", "site_aware_setting", "seed",
        "selection_metric", "validation_roc_auc", "validation_balanced_accuracy",
    ]}
    manifest_rows.append(row)


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINTING -- interruption-safe persistence, one file per completed unit.
# A unit is "complete" only if its JSON parses and contains "_complete": True;
# anything else (missing, truncated, malformed) is treated as NOT done and
# recomputed. Writes are atomic: write to a .tmp file, fsync, then os.replace
# (atomic on the same filesystem) onto the final name -- so a kill mid-write
# never leaves a checkpoint that could be mistaken for a completed unit.
# ═══════════════════════════════════════════════════════════════════════════
CKPT_DIR = OUT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def ckpt_path(unit_id):
    safe = unit_id.replace("/", "_").replace("\\", "_")
    return CKPT_DIR / f"{safe}.json"


def save_checkpoint(unit_id, payload):
    path = ckpt_path(unit_id)
    tmp = path.with_suffix(".json.tmp")
    data = dict(payload)
    data["_complete"] = True
    data["_unit_id"] = unit_id
    data["_saved_utc"] = datetime.now(timezone.utc).isoformat()
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic rename on the same filesystem


def load_checkpoint(unit_id):
    path = ckpt_path(unit_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        log(f"  [checkpoint] {unit_id}: file exists but is malformed/unreadable -- treating as NOT complete, will rerun.")
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        log(f"  [checkpoint] {unit_id}: file exists but is missing '_complete: true' -- treating as NOT complete, will rerun.")
        return None
    return data


def checkpointed(unit_id, compute_fn):
    """Run compute_fn() unless unit_id's checkpoint already holds a complete result; persist the result either way."""
    ck = load_checkpoint(unit_id)
    if ck is not None:
        log(f"  [checkpoint] {unit_id}: SKIP (already completed at {ck.get('_saved_utc', '?')})")
        return ck["payload"], True
    result = compute_fn()
    save_checkpoint(unit_id, {"payload": result})
    return result, False


def count_checkpoints(prefix):
    return len(list(CKPT_DIR.glob(f"{prefix}*.json")))


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING -- identical cohort/FC/outer-split convention as Phase 6/9
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    cohort = pd.read_csv(PHASE2_FILTERED, dtype={"subject_id": str})
    cohort["subject_id_norm"] = cohort["subject_id"].map(normalize_id)
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
    site_arr = cohort.set_index("subject_id_norm").loc[subj_order, "site"].values
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

    outer_skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    outer_splits = list(outer_skf.split(fc_raw, y))
    return fc_raw, y, site_arr, feature_index_df, outer_splits, subj_order


def prepare_split(fc_raw, y, tr_idx, te_idx):
    X_tr_raw, X_te_raw = fc_raw[tr_idx].copy(), fc_raw[te_idx].copy()
    train_missing, test_missing = np.isnan(X_tr_raw), np.isnan(X_te_raw)
    col_means = np.nanmean(X_tr_raw, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    X_tr_imp = np.where(train_missing, col_means[None, :], X_tr_raw)
    X_te_imp = np.where(test_missing, col_means[None, :], X_te_raw)
    X_tr_clip = np.clip(X_tr_imp, -1 + FISHER_Z_CLIP, 1 - FISHER_Z_CLIP)
    X_te_clip = np.clip(X_te_imp, -1 + FISHER_Z_CLIP, 1 - FISHER_Z_CLIP)
    X_tr_z, X_te_z = np.arctanh(X_tr_clip), np.arctanh(X_te_clip)
    scaler = StandardScaler().fit(X_tr_z)
    return scaler.transform(X_tr_z), scaler.transform(X_te_z), y[tr_idx], y[te_idx]


def fit_en(X_tr, y_tr, C, l1_ratio):
    en = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                            max_iter=3000, tol=1e-4, class_weight="balanced", random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        en.fit(X_tr, y_tr)
    selected_idx = np.where(en.coef_[0] != 0)[0]
    if len(selected_idx) == 0:
        selected_idx = np.argsort(-np.abs(en.coef_[0]))[:1]
    return selected_idx


# ═══════════════════════════════════════════════════════════════════════════
# SSAE (same parameterized builder as Phase 9)
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
    weight = (torch.tensor([len(y_fit) / (len(classes) * c) for c in counts], dtype=torch.float32, device=device)
             if cfg["class_weight"] == "balanced" else None)
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
                   "final_train_loss": train_losses[-1], "final_val_loss": val_losses[-1], "stop_epoch": stop_epoch}


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


def inner_folds_for(y_tr_outer, n_splits=N_INNER):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros_like(y_tr_outer), y_tr_outer))


def main():
    log("=" * 70)
    log("PHASE 10 -- Nested-CV accuracy-improvement experiments (genuine inner-CV hyperparameter selection).")
    log("=" * 70)
    pre_guard = fp_snapshot()
    problems = []

    import torch
    import torch.nn as nn
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"PyTorch device: {device}")

    log("\n--- Loading cohort/FC and building the outer 5-fold split (identical convention to Phase 6/9) ---")
    fc_raw, y, site_arr, feature_index_df, outer_splits, subj_order = load_data()
    log(f"780 subjects, 6670 features, {N_OUTER} outer folds, {N_INNER} inner folds per outer-training fold "
        f"(reduced from a larger k for compute tractability -- documented).")

    fold_assign_rows = []
    for outer_idx, (tr_idx, te_idx) in enumerate(outer_splits):
        for i in tr_idx:
            fold_assign_rows.append({"subject_index": int(i), "site": site_arr[i], "diagnosis": int(y[i]),
                                     "outer_fold": outer_idx, "outer_role": "train"})
        for i in te_idx:
            fold_assign_rows.append({"subject_index": int(i), "site": site_arr[i], "diagnosis": int(y[i]),
                                     "outer_fold": outer_idx, "outer_role": "test"})
    pd.DataFrame(fold_assign_rows).to_csv(OUT / "nested_cv" / "fold_assignments.csv", index=False)

    # ── Section 4: site-imbalance measurement (descriptive, no modeling) ──
    log("\n--- Site-imbalance measurement across the ordinary stratified outer folds ---")
    site_imbalance_rows = []
    overall_site_counts = pd.Series(site_arr).value_counts(normalize=True)
    for outer_idx, (tr_idx, te_idx) in enumerate(outer_splits):
        te_site_counts = pd.Series(site_arr[te_idx]).value_counts(normalize=True)
        max_dev = float((te_site_counts.reindex(overall_site_counts.index).fillna(0) - overall_site_counts).abs().max())
        n_sites_in_test = pd.Series(site_arr[te_idx]).nunique()
        site_imbalance_rows.append({"outer_fold": outer_idx, "n_sites_in_test_fold": n_sites_in_test,
                                    "n_sites_total": pd.Series(site_arr).nunique(),
                                    "max_site_proportion_deviation_vs_overall": max_dev})
    site_imbalance_df = pd.DataFrame(site_imbalance_rows)
    site_imbalance_df.to_csv(OUT / "site_aware" / "site_imbalance_measurement.csv", index=False)
    log(f"Site-imbalance measurement (ordinary StratifiedKFold, stratified by diagnosis only, NOT by site): "
        f"{site_imbalance_df.to_dict(orient='records')}")
    log("This is a MEASUREMENT, not an assumption of leakage -- ordinary stratification by diagnosis does not "
        "guarantee balanced site representation, and the table above quantifies the actual deviation observed.")

    # ═══════════════════════════════════════════════════════════════════
    # Precompute outer-level prepared data (Strategy C + Fisher-Z + scaling, train-fold-only stats)
    # ═══════════════════════════════════════════════════════════════════
    outer_data = {}
    for outer_idx, (tr_idx, te_idx) in enumerate(outer_splits):
        X_tr, X_te, y_tr, y_te = prepare_split(fc_raw, y, tr_idx, te_idx)
        outer_data[outer_idx] = {"X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
                                 "tr_idx": tr_idx, "te_idx": te_idx}
    record_leakage("outer_preprocessing", False, False, True, True, "N/A (no PCA at this stage)", "N/A (pre-feature-selection)", "N/A",
                   "Strategy C imputation statistic and StandardScaler fit on outer-training data only, exactly as Phase 6/9.")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE A -- NESTED ELASTIC NET (genuine per-outer-fold inner CV)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT A: Nested Elastic Net search (per-outer-fold inner CV, baseline arch 500x100)")
    log("=" * 70)
    C_grid = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    l1_grid_extra = [0.25, 0.75]  # extra values beyond the C-sweep's fixed l1_ratio=0.5; 0.1/0.9 dropped for compute tractability (documented)
    en_configs = [{"C": c, "l1_ratio": 0.5} for c in C_grid] + [{"C": 0.1, "l1_ratio": r} for r in l1_grid_extra]
    log(f"Controlled EN grid: {len(C_grid)} C values at l1_ratio=0.5, plus {len(l1_grid_extra)} extra l1_ratio "
        f"values at C=0.1 -- {len(en_configs)} unique configurations per outer fold (not the full "
        f"{len(C_grid)*5}-cell grid), each evaluated via {N_INNER}-fold inner CV.")

    n_planned_A = N_OUTER
    n_done_A = count_checkpoints("experiment_A_outer_fold_")
    log(f"PHASE 10 RESUME CHECK (Experiment A): Completed checkpoints: {n_done_A} / Remaining: {n_planned_A - n_done_A}")

    en_nested_rows = []
    per_outer_best_en = {}
    for outer_idx in range(N_OUTER):
        def _compute_A(outer_idx=outer_idx):
            od = outer_data[outer_idx]
            inner_splits = inner_folds_for(od["y_tr"])
            fold_rows, fold_manifest = [], []
            for en_p in en_configs:
                inner_metrics = []
                for inner_idx, (itr, ival) in enumerate(inner_splits):
                    X_itr, y_itr = od["X_tr"][itr], od["y_tr"][itr]
                    X_ival, y_ival = od["X_tr"][ival], od["y_tr"][ival]
                    sel_idx = fit_en(X_itr, y_itr, en_p["C"], en_p["l1_ratio"])
                    cfg = {**BASELINE_TRAIN, "h1": 500, "h2": 100}
                    model, hist = train_ssae(X_itr[:, sel_idx], y_itr, X_ival[:, sel_idx], y_ival, cfg, device, torch, nn, optim, seed=SEED)
                    vm = eval_torch(model, X_ival[:, sel_idx], y_ival, device, torch)
                    inner_metrics.append({"roc_auc": vm["roc_auc"], "balanced_accuracy": vm["balanced_accuracy"], "N": len(sel_idx)})
                    fold_manifest.append(dict(experiment_id=f"A_outer{outer_idx}_C{en_p['C']}_L1{en_p['l1_ratio']}_inner{inner_idx}",
                                outer_fold=outer_idx, inner_fold=inner_idx, model="SSAE_500x100",
                                elastic_net_C=en_p["C"], elastic_net_l1_ratio=en_p["l1_ratio"],
                                feature_selection_method="ElasticNet", feature_count=len(sel_idx),
                                architecture="500->100", dropout=0.0, weight_decay=1e-4, learning_rate=1e-3,
                                batch_size=32, early_stopping="None", pca_setting="none", site_aware_setting="ordinary",
                                seed=SEED, selection_metric="validation_ROC_AUC",
                                validation_roc_auc=vm["roc_auc"], validation_balanced_accuracy=vm["balanced_accuracy"]))
                mean_auc = float(np.mean([m["roc_auc"] for m in inner_metrics]))
                mean_bacc = float(np.mean([m["balanced_accuracy"] for m in inner_metrics]))
                mean_N = float(np.mean([m["N"] for m in inner_metrics]))
                fold_rows.append({"outer_fold": outer_idx, "C": en_p["C"], "l1_ratio": en_p["l1_ratio"],
                                  "mean_inner_feature_count": mean_N, "mean_inner_ROC_AUC": mean_auc,
                                  "mean_inner_balanced_accuracy": mean_bacc})
            outer_en_df = pd.DataFrame(fold_rows)
            best = outer_en_df.sort_values(["mean_inner_ROC_AUC", "mean_inner_balanced_accuracy"], ascending=[False, False]).iloc[0]
            tied = outer_en_df[outer_en_df.mean_inner_ROC_AUC >= best.mean_inner_ROC_AUC - 0.005]
            if len(tied) > 1:
                best = tied.sort_values("C").iloc[0]
            best_en = {"C": float(best.C), "l1_ratio": float(best.l1_ratio)}
            return {"rows": fold_rows, "manifest": fold_manifest, "best": best_en}

        result, was_cached = checkpointed(f"experiment_A_outer_fold_{outer_idx}", _compute_A)
        en_nested_rows.extend(result["rows"])
        manifest_rows.extend(result["manifest"])
        per_outer_best_en[outer_idx] = result["best"]
        if not was_cached:
            log(f"  Outer fold {outer_idx}: nested-CV-selected EN config C={result['best']['C']}, l1_ratio={result['best']['l1_ratio']}")

    en_nested_df = pd.DataFrame(en_nested_rows)
    en_nested_df.to_csv(OUT / "elastic_net" / "elastic_net_nested_results.csv", index=False)
    record_leakage("elastic_net_nested", False, False, True, True, "N/A", True, True,
                   "Elastic Net fit only on each inner-training split; selection uses only inner-validation ROC-AUC/balanced accuracy, never outer test.")

    from collections import Counter
    en_mode = Counter([(v["C"], v["l1_ratio"]) for v in per_outer_best_en.values()]).most_common(1)[0][0]
    final_en = {"C": en_mode[0], "l1_ratio": en_mode[1]}
    log(f"FINAL (majority-vote across 5 outer folds' nested selections) Elastic Net configuration for "
        f"the single locked model: C={final_en['C']}, l1_ratio={final_en['l1_ratio']}. "
        f"Per-outer-fold selections: {per_outer_best_en}")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE D -- NESTED SSAE ARCHITECTURE (per-outer-fold inner CV, using that fold's nested EN selection)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT D: Nested SSAE architecture / model-size search (per-outer-fold inner CV)")
    log("=" * 70)
    architectures = [("64x32", 64, 32), ("128x32", 128, 32), ("128x64", 128, 64), ("256x64", 256, 64), ("256x128", 256, 128)]
    n_planned_D = N_OUTER
    n_done_D = count_checkpoints("experiment_D_outer_fold_")
    log(f"PHASE 10 RESUME CHECK (Experiment D): Completed checkpoints: {n_done_D} / Remaining: {n_planned_D - n_done_D}")

    arch_nested_rows = []
    per_outer_best_arch = {}
    for outer_idx in range(N_OUTER):
        def _compute_D(outer_idx=outer_idx):
            od = outer_data[outer_idx]
            inner_splits = inner_folds_for(od["y_tr"])
            en_p = per_outer_best_en[outer_idx]
            fold_rows, fold_manifest = [], []
            for name, h1, h2 in architectures:
                inner_metrics = []
                for inner_idx, (itr, ival) in enumerate(inner_splits):
                    X_itr, y_itr = od["X_tr"][itr], od["y_tr"][itr]
                    X_ival, y_ival = od["X_tr"][ival], od["y_tr"][ival]
                    sel_idx = fit_en(X_itr, y_itr, en_p["C"], en_p["l1_ratio"])
                    cfg = {**BASELINE_TRAIN, "h1": h1, "h2": h2}
                    model, hist = train_ssae(X_itr[:, sel_idx], y_itr, X_ival[:, sel_idx], y_ival, cfg, device, torch, nn, optim, seed=SEED)
                    vm = eval_torch(model, X_ival[:, sel_idx], y_ival, device, torch)
                    inner_metrics.append({"roc_auc": vm["roc_auc"], "balanced_accuracy": vm["balanced_accuracy"],
                                          "gap": hist["final_val_loss"] - hist["final_train_loss"], "N": len(sel_idx)})
                    fold_manifest.append(dict(experiment_id=f"D_outer{outer_idx}_{name}_inner{inner_idx}", outer_fold=outer_idx,
                                inner_fold=inner_idx, model=f"SSAE_{name}", elastic_net_C=en_p["C"], elastic_net_l1_ratio=en_p["l1_ratio"],
                                feature_selection_method="ElasticNet", feature_count=len(sel_idx), architecture=f"{h1}->{h2}",
                                dropout=0.0, weight_decay=1e-4, learning_rate=1e-3, batch_size=32, early_stopping="None",
                                pca_setting="none", site_aware_setting="ordinary", seed=SEED, selection_metric="validation_ROC_AUC",
                                validation_roc_auc=vm["roc_auc"], validation_balanced_accuracy=vm["balanced_accuracy"]))
                approx_params = int(np.mean([m["N"] for m in inner_metrics]) * h1 + h1 * h2 + h2 * 2)
                fold_rows.append({"outer_fold": outer_idx, "architecture": name, "h1": h1, "h2": h2,
                                  "approx_param_count": approx_params,
                                  "mean_inner_ROC_AUC": float(np.mean([m["roc_auc"] for m in inner_metrics])),
                                  "mean_inner_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in inner_metrics])),
                                  "mean_inner_generalization_gap": float(np.mean([m["gap"] for m in inner_metrics]))})
            outer_arch_df = pd.DataFrame(fold_rows)
            best = outer_arch_df.sort_values(["mean_inner_ROC_AUC", "mean_inner_balanced_accuracy"], ascending=[False, False]).iloc[0]
            tied = outer_arch_df[outer_arch_df.mean_inner_ROC_AUC >= best.mean_inner_ROC_AUC - 0.005]
            if len(tied) > 1:
                best = tied.sort_values("approx_param_count").iloc[0]
            best_arch = {"h1": int(best.h1), "h2": int(best.h2), "name": best.architecture}
            return {"rows": fold_rows, "manifest": fold_manifest, "best": best_arch}

        result, was_cached = checkpointed(f"experiment_D_outer_fold_{outer_idx}", _compute_D)
        arch_nested_rows.extend(result["rows"])
        manifest_rows.extend(result["manifest"])
        per_outer_best_arch[outer_idx] = result["best"]
        if not was_cached:
            log(f"  Outer fold {outer_idx}: nested-CV-selected architecture {result['best']['name']}")

    arch_nested_df = pd.DataFrame(arch_nested_rows)
    arch_nested_df.to_csv(OUT / "ssae" / "ssae_architecture_results.csv", index=False)
    record_leakage("ssae_architecture_nested", False, False, True, True, "N/A", True, True,
                   "Architecture selection uses only inner-validation ROC-AUC/balanced accuracy per outer fold; outer test never accessed.")

    arch_mode = Counter([v["name"] for v in per_outer_best_arch.values()]).most_common(1)[0][0]
    final_arch_h1h2 = next((v["h1"], v["h2"]) for v in per_outer_best_arch.values() if v["name"] == arch_mode)
    final_arch = {"h1": final_arch_h1h2[0], "h2": final_arch_h1h2[1]}
    log(f"FINAL (majority-vote) architecture: {final_arch}. Per-outer-fold selections: {per_outer_best_arch}")

    # ═══════════════════════════════════════════════════════════════════
    # EXPERIMENT B -- FEATURE STABILITY THRESHOLD (using inner-fold EN fits, fast LR evaluator)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT B: Feature-selection stability threshold (inner-CV only; evaluated with Logistic Regression for compute tractability)")
    log("=" * 70)
    n_done_B = count_checkpoints("experiment_B_outer_fold_")
    log(f"PHASE 10 RESUME CHECK (Experiment B): Completed checkpoints: {n_done_B} / Remaining: {N_OUTER - n_done_B}")

    stability_rows = []
    per_outer_best_thresh = {}
    for outer_idx in range(N_OUTER):
        def _compute_B(outer_idx=outer_idx):
            od = outer_data[outer_idx]
            inner_splits = inner_folds_for(od["y_tr"])
            en_p = per_outer_best_en[outer_idx]
            inner_sel_sets = []
            for itr, ival in inner_splits:
                X_itr, y_itr = od["X_tr"][itr], od["y_tr"][itr]
                sel_idx = fit_en(X_itr, y_itr, en_p["C"], en_p["l1_ratio"])
                inner_sel_sets.append(set(sel_idx.tolist()))
            all_feats = sorted(set.union(*inner_sel_sets)) if inner_sel_sets else []
            freq = {f: sum(f in s for s in inner_sel_sets) / len(inner_sel_sets) for f in all_feats}
            fold_rows = []
            for thresh in [0.5, 0.6, 0.7, 0.8]:
                stable_feats = np.array(sorted([f for f, fr in freq.items() if fr >= thresh]))
                if len(stable_feats) == 0:
                    continue
                inner_aucs = []
                for itr, ival in inner_splits:
                    X_itr, y_itr = od["X_tr"][itr][:, stable_feats], od["y_tr"][itr]
                    X_ival, y_ival = od["X_tr"][ival][:, stable_feats], od["y_tr"][ival]
                    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
                    lr.fit(X_itr, y_itr)
                    probs = lr.predict_proba(X_ival)[:, 1]
                    try:
                        auc = roc_auc_score(y_ival, probs)
                    except ValueError:
                        auc = float("nan")
                    inner_aucs.append(auc)
                fold_rows.append({"outer_fold": outer_idx, "stability_threshold": thresh,
                                  "n_stable_features": len(stable_feats), "mean_inner_ROC_AUC_LR_probe": float(np.nanmean(inner_aucs))})
            best_thresh = None
            if fold_rows:
                outer_stab_df = pd.DataFrame(fold_rows)
                best_t = outer_stab_df.sort_values("mean_inner_ROC_AUC_LR_probe", ascending=False).iloc[0]
                best_thresh = float(best_t.stability_threshold)
            return {"rows": fold_rows, "best_thresh": best_thresh}

        result, was_cached = checkpointed(f"experiment_B_outer_fold_{outer_idx}", _compute_B)
        stability_rows.extend(result["rows"])
        if result["best_thresh"] is not None:
            per_outer_best_thresh[outer_idx] = result["best_thresh"]
        if not was_cached:
            log(f"  Outer fold {outer_idx}: best stability threshold (LR probe, inner CV) = {result['best_thresh']}")
    pd.DataFrame(stability_rows).to_csv(OUT / "feature_stability" / "feature_stability_results.csv", index=False)
    record_leakage("feature_stability", False, False, True, True, "N/A", True, True,
                   "Stability frequency computed from inner-fold EN fits only; threshold selected via inner-CV LR-probe ROC-AUC, not outer test. Evaluated with Logistic Regression rather than the full SSAE for compute tractability -- documented, not hidden.")
    log("NOTE: this stability-selected representation is reported as a comparator, not substituted into the main nested EN+SSAE track, "
        "since its selection criterion (LR-probe AUC) differs from the primary track's SSAE-based criterion.")

    # ═══════════════════════════════════════════════════════════════════
    # GLOBAL SWEEPS (regularization, LR, early stopping) -- single inner split per outer fold, reduced scope, documented
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT E/F/G: regularization, early stopping, learning rate (global controlled sweep, single inner split per outer fold -- reduced from full nested-per-config for compute tractability)")
    log("=" * 70)

    def global_sweep_eval(cfg_overrides, tag):
        def _compute():
            rows, manifest = [], []
            for outer_idx in range(N_OUTER):
                od = outer_data[outer_idx]
                itr, ival = train_test_split(np.arange(len(od["y_tr"])), test_size=1 / N_INNER, stratify=od["y_tr"], random_state=SEED)
                X_itr, y_itr = od["X_tr"][itr], od["y_tr"][itr]
                X_ival, y_ival = od["X_tr"][ival], od["y_tr"][ival]
                sel_idx = fit_en(X_itr, y_itr, final_en["C"], final_en["l1_ratio"])
                cfg = {**BASELINE_TRAIN, "h1": final_arch["h1"], "h2": final_arch["h2"]}
                cfg.update(cfg_overrides)
                model, hist = train_ssae(X_itr[:, sel_idx], y_itr, X_ival[:, sel_idx], y_ival, cfg, device, torch, nn, optim, seed=SEED)
                vm = eval_torch(model, X_ival[:, sel_idx], y_ival, device, torch)
                rows.append({"outer_fold": outer_idx, **cfg_overrides, "validation_ROC_AUC": vm["roc_auc"],
                            "validation_balanced_accuracy": vm["balanced_accuracy"],
                            "generalization_gap": hist["final_val_loss"] - hist["final_train_loss"], "stop_epoch": hist["stop_epoch"]})
                manifest.append(dict(experiment_id=f"{tag}_outer{outer_idx}", outer_fold=outer_idx, inner_fold=0,
                            model="SSAE_final_arch", elastic_net_C=final_en["C"], elastic_net_l1_ratio=final_en["l1_ratio"],
                            feature_selection_method="ElasticNet", feature_count=len(sel_idx),
                            architecture=f"{final_arch['h1']}->{final_arch['h2']}", dropout=cfg["dropout"],
                            weight_decay=cfg["weight_decay"], learning_rate=cfg["lr"], batch_size=cfg["batch_size"],
                            early_stopping=cfg["early_stopping_patience"], pca_setting="none", site_aware_setting="ordinary",
                            seed=SEED, selection_metric="validation_ROC_AUC", validation_roc_auc=vm["roc_auc"],
                            validation_balanced_accuracy=vm["balanced_accuracy"]))
            return {"rows": rows, "manifest": manifest}

        result, was_cached = checkpointed(f"experiment_{tag}", _compute)
        manifest_rows.extend(result["manifest"])
        if was_cached:
            log(f"  {tag}: using checkpointed result")
        return pd.DataFrame(result["rows"])

    reg_rows = []
    for wd in [1e-4, 1e-3, 1e-2, 5e-2]:
        df = global_sweep_eval({"weight_decay": wd}, f"E_WD_{wd}")
        reg_rows.append({"weight_decay": wd, "dropout": 0.0, "mean_validation_ROC_AUC": df.validation_ROC_AUC.mean(),
                         "mean_generalization_gap": df.generalization_gap.mean()})
        log(f"  weight_decay={wd}: val_auc={reg_rows[-1]['mean_validation_ROC_AUC']:.4f}, gap={reg_rows[-1]['mean_generalization_gap']:.4f}")
    best_wd = pd.DataFrame(reg_rows).sort_values("mean_validation_ROC_AUC", ascending=False).iloc[0]["weight_decay"]
    for dr in [0.1, 0.2, 0.3, 0.5]:
        df = global_sweep_eval({"weight_decay": best_wd, "dropout": dr}, f"E_DROPOUT_{dr}")
        reg_rows.append({"weight_decay": best_wd, "dropout": dr, "mean_validation_ROC_AUC": df.validation_ROC_AUC.mean(),
                         "mean_generalization_gap": df.generalization_gap.mean()})
        log(f"  dropout={dr} (wd={best_wd}): val_auc={reg_rows[-1]['mean_validation_ROC_AUC']:.4f}, gap={reg_rows[-1]['mean_generalization_gap']:.4f}")
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(OUT / "regularization" / "regularization_results.csv", index=False)
    best_reg = reg_df.sort_values("mean_validation_ROC_AUC", ascending=False).iloc[0]
    final_reg = {"weight_decay": float(best_reg.weight_decay), "dropout": float(best_reg.dropout)}
    log(f"SELECTED regularization: {final_reg}")

    lr_rows = []
    for lr in [0.0001, 0.0003, 0.001, 0.003]:
        df = global_sweep_eval({**final_reg, "lr": lr}, f"G_LR_{lr}")
        lr_rows.append({"learning_rate": lr, "mean_validation_ROC_AUC": df.validation_ROC_AUC.mean()})
        log(f"  lr={lr}: val_auc={lr_rows[-1]['mean_validation_ROC_AUC']:.4f}")
    lr_df = pd.DataFrame(lr_rows)
    best_lr = float(lr_df.sort_values("mean_validation_ROC_AUC", ascending=False).iloc[0]["learning_rate"])
    log(f"SELECTED learning rate: {best_lr}")

    es_rows = []
    for patience in [5, 10, 15, None]:
        df = global_sweep_eval({**final_reg, "lr": best_lr, "early_stopping_patience": patience}, f"F_PATIENCE_{patience}")
        es_rows.append({"patience": patience, "mean_validation_ROC_AUC": df.validation_ROC_AUC.mean(),
                        "mean_stop_epoch": df.stop_epoch.mean()})
        log(f"  patience={patience}: val_auc={es_rows[-1]['mean_validation_ROC_AUC']:.4f}, mean_stop_epoch={es_rows[-1]['mean_stop_epoch']:.1f}")
    es_df = pd.DataFrame(es_rows)
    es_df.to_csv(OUT / "regularization" / "early_stopping_results.csv", index=False)
    lr_df.to_csv(OUT / "regularization" / "learning_rate_results.csv", index=False)
    best_patience_row = es_df.sort_values("mean_validation_ROC_AUC", ascending=False).iloc[0]
    final_patience = best_patience_row["patience"]
    final_patience = None if (final_patience is None or (isinstance(final_patience, float) and np.isnan(final_patience))) else int(final_patience)
    log(f"SELECTED early-stopping patience: {final_patience}")
    record_leakage("regularization_lr_earlystopping_global_sweep", False, False, True, True, "N/A", True, True,
                   "Global controlled sweep (single inner split per outer fold, not full nested-per-config) reusing the majority-vote EN+architecture; selection uses only that inner-validation split, never outer test. Reduced scope documented for compute tractability.")

    # ═══════════════════════════════════════════════════════════════════
    # EXPERIMENT H -- REPRESENTATION CHECK (Fisher-Z / scaled Fisher-Z / PCA, fast LR probe, inner-CV only)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT H: Representation check (Fisher-Z vs standardized Fisher-Z vs train-fold-only PCA; Logistic Regression probe, inner-CV only)")
    log("=" * 70)
    n_done_H = count_checkpoints("experiment_H_outer_fold_")
    log(f"PHASE 10 RESUME CHECK (Experiment H): Completed checkpoints: {n_done_H} / Remaining: {N_OUTER - n_done_H}")

    repr_rows = []
    for outer_idx in range(N_OUTER):
        def _compute_H(outer_idx=outer_idx):
            od = outer_data[outer_idx]
            inner_splits = inner_folds_for(od["y_tr"])
            fold_rows = []
            for inner_idx, (itr, ival) in enumerate(inner_splits):
                X_itr_scaled, y_itr = od["X_tr"][itr], od["y_tr"][itr]
                X_ival_scaled, y_ival = od["X_tr"][ival], od["y_tr"][ival]

                lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=SEED)
                lr.fit(X_itr_scaled, y_itr)
                auc = roc_auc_score(y_ival, lr.predict_proba(X_ival_scaled)[:, 1])
                fold_rows.append({"outer_fold": outer_idx, "inner_fold": inner_idx, "representation": "standardized_fisher_z_full",
                                  "n_components": 6670, "inner_ROC_AUC": auc})

                for n_comp in [64, 128, 256]:
                    pca = PCA(n_components=n_comp, random_state=SEED)
                    X_itr_pca = pca.fit_transform(X_itr_scaled)
                    X_ival_pca = pca.transform(X_ival_scaled)
                    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=SEED)
                    lr.fit(X_itr_pca, y_itr)
                    auc = roc_auc_score(y_ival, lr.predict_proba(X_ival_pca)[:, 1])
                    fold_rows.append({"outer_fold": outer_idx, "inner_fold": inner_idx, "representation": f"pca_{n_comp}",
                                      "n_components": n_comp, "inner_ROC_AUC": auc})
            return {"rows": fold_rows}

        result, was_cached = checkpointed(f"experiment_H_outer_fold_{outer_idx}", _compute_H)
        repr_rows.extend(result["rows"])
    repr_df = pd.DataFrame(repr_rows)
    repr_df.to_csv(OUT / "representation" / "representation_results.csv", index=False)
    repr_summary = repr_df.groupby("representation")["inner_ROC_AUC"].agg(["mean", "std"]).reset_index()
    log(f"Representation comparison (mean inner-CV ROC-AUC, Logistic Regression probe):\n{repr_summary.to_string(index=False)}")
    record_leakage("representation_check", False, False, True, True, True, "N/A (PCA is the representation under test)", True,
                   "PCA fit on inner-training data only per inner fold, applied to inner-validation only; never fit on pooled/outer/test data. Evaluated with Logistic Regression, not the full SSAE, for compute tractability.")

    # ═══════════════════════════════════════════════════════════════════
    # EXPERIMENT C -- CLASSIFIER COMPARISON (inner-CV only; outer test intentionally NOT touched for comparators in Phase 10)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("EXPERIMENT C: Classifier comparison (LogReg, Linear SVM, RBF SVM, small MLP, SSAE) -- inner-CV only, outer test NOT accessed")
    log("=" * 70)
    n_done_C = count_checkpoints("experiment_C_outer_fold_")
    log(f"PHASE 10 RESUME CHECK (Experiment C): Completed checkpoints: {n_done_C} / Remaining: {N_OUTER - n_done_C}")

    clf_rows = []
    for outer_idx in range(N_OUTER):
        def _compute_C(outer_idx=outer_idx):
            od = outer_data[outer_idx]
            inner_splits = inner_folds_for(od["y_tr"])
            en_p = per_outer_best_en[outer_idx]
            fold_rows = []
            for inner_idx, (itr, ival) in enumerate(inner_splits):
                X_itr, y_itr = od["X_tr"][itr], od["y_tr"][itr]
                X_ival, y_ival = od["X_tr"][ival], od["y_tr"][ival]
                sel_idx = fit_en(X_itr, y_itr, en_p["C"], en_p["l1_ratio"])
                Xi, Xv = X_itr[:, sel_idx], X_ival[:, sel_idx]
                for clf_name, builder in [
                    ("LogisticRegression", lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)),
                    ("LinearSVM", lambda: SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED, C=1.0)),
                    ("RBF_SVM", lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED, C=1.0)),
                    ("MLP_small", lambda: MLPClassifier(hidden_layer_sizes=(64, 16), alpha=1e-2, max_iter=500, random_state=SEED)),
                ]:
                    clf = builder()
                    clf.fit(Xi, y_itr)
                    probs = clf.predict_proba(Xv)[:, 1]
                    preds = clf.predict(Xv)
                    m = metrics_dict(y_ival, preds, probs)
                    fold_rows.append({"outer_fold": outer_idx, "inner_fold": inner_idx, "classifier": clf_name,
                                    "validation_accuracy": m["accuracy"], "validation_balanced_accuracy": m["balanced_accuracy"],
                                    "validation_ROC_AUC": m["roc_auc"]})
                cfg = {**BASELINE_TRAIN, "h1": final_arch["h1"], "h2": final_arch["h2"], **final_reg}
                model, hist = train_ssae(Xi, y_itr, Xv, y_ival, cfg, device, torch, nn, optim, seed=SEED)
                vm = eval_torch(model, Xv, y_ival, device, torch)
                fold_rows.append({"outer_fold": outer_idx, "inner_fold": inner_idx, "classifier": "SSAE_locked_arch",
                                "validation_accuracy": vm["accuracy"], "validation_balanced_accuracy": vm["balanced_accuracy"],
                                "validation_ROC_AUC": vm["roc_auc"]})
            return {"rows": fold_rows}

        result, was_cached = checkpointed(f"experiment_C_outer_fold_{outer_idx}", _compute_C)
        clf_rows.extend(result["rows"])
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(OUT / "classifiers" / "classifier_comparison.csv", index=False)
    clf_summary = clf_df.groupby("classifier")[["validation_accuracy", "validation_balanced_accuracy", "validation_ROC_AUC"]].mean()
    log(f"Classifier comparison (mean inner-CV validation metrics, outer test NOT accessed for any of these):\n{clf_summary.to_string()}")
    record_leakage("classifier_comparison", False, False, True, True, "N/A", True, True,
                   "All classifiers evaluated on inner-validation splits only. Unlike Phase 9, Phase 10 does not report outer-test performance for comparator classifiers, to strictly minimize outer-test contact given this phase's stronger emphasis on protecting the test set.")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL LOCK
    # ═══════════════════════════════════════════════════════════════════
    final_config = {
        "elastic_net_C": final_en["C"], "elastic_net_l1_ratio": final_en["l1_ratio"],
        "architecture_h1": final_arch["h1"], "architecture_h2": final_arch["h2"],
        "weight_decay": final_reg["weight_decay"], "dropout": final_reg["dropout"],
        "learning_rate": best_lr, "early_stopping_patience": final_patience,
        "batch_size": 32, "class_weight": "balanced", "pretrain_epochs": 40, "finetune_epochs_max": 80,
        "sparsity_weight": 1e-4, "primary_random_seed": SEED,
        "selection_method": "Elastic Net and architecture selected via genuine per-outer-fold inner CV (majority vote across 5 outer folds for the single deployed configuration); regularization/LR/early-stopping selected via a global controlled sweep (single inner split per outer fold, reusing the majority-vote EN+architecture).",
        "per_outer_fold_nested_en_selections": per_outer_best_en, "per_outer_fold_nested_arch_selections": per_outer_best_arch,
    }
    with open(OUT / "final" / "locked_configuration.json", "w") as f:
        json.dump(final_config, f, indent=2, default=str)
    log(f"\nFINAL LOCKED CONFIGURATION: {final_config}")

    log("\n" + "=" * 70)
    log("FINAL OUTER-TEST EVALUATION (first and only touch of outer test for this exact configuration)")
    log("=" * 70)
    n_planned_final = 3 * N_OUTER
    n_done_final = count_checkpoints("final_lock_seed")
    log(f"PHASE 10 RESUME CHECK (Final lock evaluation): Completed checkpoints: {n_done_final} / Remaining: {n_planned_final - n_done_final}")

    final_rows = []
    for seed in [42, 123, 2026]:
        for outer_idx in range(N_OUTER):
            def _compute_final(seed=seed, outer_idx=outer_idx):
                od = outer_data[outer_idx]
                sel_idx = fit_en(od["X_tr"], od["y_tr"], final_en["C"], final_en["l1_ratio"])
                cfg = {**BASELINE_TRAIN, "h1": final_arch["h1"], "h2": final_arch["h2"], **final_reg,
                      "lr": best_lr, "early_stopping_patience": final_patience}
                itr, ival = train_test_split(np.arange(len(od["y_tr"])), test_size=0.15, stratify=od["y_tr"], random_state=SEED)
                X_fit, y_fit = od["X_tr"][itr][:, sel_idx], od["y_tr"][itr]
                X_val, y_val = od["X_tr"][ival][:, sel_idx], od["y_tr"][ival]
                model, hist = train_ssae(X_fit, y_fit, X_val, y_val, cfg, device, torch, nn, optim, seed=seed)
                test_m = eval_torch(model, od["X_te"][:, sel_idx], od["y_te"], device, torch)
                row = {"seed": seed, "outer_fold": outer_idx, "selected_feature_count": len(sel_idx),
                      "test_accuracy": test_m["accuracy"], "test_balanced_accuracy": test_m["balanced_accuracy"],
                      "test_sensitivity": test_m["sensitivity"], "test_specificity": test_m["specificity"],
                      "test_precision": test_m["precision"], "test_F1": test_m["f1"], "test_ROC_AUC": test_m["roc_auc"],
                      "test_tn": test_m["tn"], "test_fp": test_m["fp"], "test_fn": test_m["fn"], "test_tp": test_m["tp"],
                      "training_loss": hist["final_train_loss"], "validation_loss": hist["final_val_loss"]}
                return {"row": row}

            result, was_cached = checkpointed(f"final_lock_seed{seed}_outer{outer_idx}", _compute_final)
            final_rows.append(result["row"])
            record_leakage(f"final_lock_seed{seed}_outer{outer_idx}", True, True, True, True, "N/A", True, True,
                           "Outer test accessed ONLY here, after the configuration was fully locked from inner-CV results above.")
        log(f"  seed={seed}: mean test_acc={np.mean([r['test_accuracy'] for r in final_rows if r['seed']==seed]):.4f}, "
            f"mean test_auc={np.mean([r['test_ROC_AUC'] for r in final_rows if r['seed']==seed]):.4f}")
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUT / "final" / "final_outer_test_results.csv", index=False)
    final_df[final_df.seed == 42][["outer_fold", "test_tn", "test_fp", "test_fn", "test_tp"]].to_csv(
        OUT / "final" / "final_confusion_matrices.csv", index=False)

    seed_summary = final_df.groupby("seed")[["test_accuracy", "test_balanced_accuracy", "test_ROC_AUC"]].mean().reset_index()
    seed_summary.to_csv(OUT / "seeds" / "seed_stability.csv", index=False)
    log(f"\nMulti-seed stability:\n{seed_summary.to_string(index=False)}")

    p10_seed42 = final_df[final_df.seed == 42].sort_values("outer_fold")
    p10_mean = {k: p10_seed42[v].mean() for k, v in {
        "accuracy": "test_accuracy", "balanced_accuracy": "test_balanced_accuracy", "sensitivity": "test_sensitivity",
        "specificity": "test_specificity", "f1": "test_F1", "roc_auc": "test_ROC_AUC"}.items()}
    p10_std = {k: p10_seed42[v].std() for k, v in {
        "accuracy": "test_accuracy", "balanced_accuracy": "test_balanced_accuracy", "sensitivity": "test_sensitivity",
        "specificity": "test_specificity", "f1": "test_F1", "roc_auc": "test_ROC_AUC"}.items()}

    # ═══════════════════════════════════════════════════════════════════
    # SITE-AWARE SECONDARY EXPERIMENT (locked config only, no new selection)
    # ═══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("SITE-AWARE SECONDARY EXPERIMENT: StratifiedGroupKFold(site) outer split, locked configuration only")
    log("=" * 70)
    sgkf = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    site_outer_splits = list(sgkf.split(fc_raw, y, groups=site_arr))
    n_done_site = count_checkpoints("site_aware_outer_fold_")
    log(f"PHASE 10 RESUME CHECK (Site-aware): Completed checkpoints: {n_done_site} / Remaining: {N_OUTER - n_done_site}")

    site_rows = []
    for outer_idx, (tr_idx, te_idx) in enumerate(site_outer_splits):
        def _compute_site(outer_idx=outer_idx, tr_idx=tr_idx, te_idx=te_idx):
            X_tr, X_te, y_tr, y_te = prepare_split(fc_raw, y, tr_idx, te_idx)
            sel_idx = fit_en(X_tr, y_tr, final_en["C"], final_en["l1_ratio"])
            cfg = {**BASELINE_TRAIN, "h1": final_arch["h1"], "h2": final_arch["h2"], **final_reg,
                  "lr": best_lr, "early_stopping_patience": final_patience}
            itr, ival = train_test_split(np.arange(len(y_tr)), test_size=0.15, stratify=y_tr, random_state=SEED)
            model, hist = train_ssae(X_tr[itr][:, sel_idx], y_tr[itr], X_tr[ival][:, sel_idx], y_tr[ival], cfg, device, torch, nn, optim, seed=SEED)
            test_m = eval_torch(model, X_te[:, sel_idx], y_te, device, torch)
            n_sites_train, n_sites_test = pd.Series(site_arr[tr_idx]).nunique(), pd.Series(site_arr[te_idx]).nunique()
            overlap = len(set(site_arr[tr_idx]) & set(site_arr[te_idx]))
            row = {"outer_fold": outer_idx, "n_sites_train": n_sites_train, "n_sites_test": n_sites_test,
                  "site_overlap_train_test": overlap, "selected_feature_count": len(sel_idx),
                  "test_accuracy": test_m["accuracy"], "test_balanced_accuracy": test_m["balanced_accuracy"],
                  "test_ROC_AUC": test_m["roc_auc"], "test_F1": test_m["f1"]}
            return {"row": row}

        result, was_cached = checkpointed(f"site_aware_outer_fold_{outer_idx}", _compute_site)
        site_rows.append(result["row"])
        record_leakage(f"site_aware_outer{outer_idx}", True, True, True, True, "N/A", True, "N/A (reuses locked configuration, no new selection)",
                       "Secondary evaluation only -- reuses the already-locked configuration; no hyperparameter selection performed under this alternative split.")
    site_df = pd.DataFrame(site_rows)
    site_df.to_csv(OUT / "site_aware" / "site_aware_results.csv", index=False)
    log(f"Site-grouped evaluation (locked configuration, no re-selection):\n{site_df.to_string(index=False)}")
    log(f"Ordinary nested-CV mean test accuracy={p10_mean['accuracy']:.4f} vs. site-grouped mean test accuracy="
        f"{site_df.test_accuracy.mean():.4f}. These measure different things: ordinary CV assumes new SUBJECTS from "
        f"already-seen sites are representative of generalization, while site-grouped CV tests generalization to "
        f"ENTIRELY UNSEEN acquisition sites -- neither is unconditionally 'better', they answer different questions.")

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISONS: Phase 9 vs Phase 10 (and Phase 6 for context)
    # ═══════════════════════════════════════════════════════════════════
    p9_locked = pd.read_csv(PHASE9 / "final_test" / "locked_test_results.csv")
    p9_seed42 = p9_locked[p9_locked.random_seed == 42].sort_values("fold")
    phase6_overall = pd.read_csv(PHASE6 / "evaluation" / "overall_metrics.csv")

    metric_pairs = {"accuracy": ("test_accuracy", "accuracy"), "balanced_accuracy": ("test_balanced_accuracy", "balanced_accuracy"),
                    "sensitivity": ("test_sensitivity", "sensitivity_recall"), "specificity": ("test_specificity", "specificity"),
                    "f1": ("test_F1", "f1"), "roc_auc": ("test_ROC_AUC", "roc_auc")}
    comparison_rows = []
    for metric, (p9_key, p6_key) in metric_pairs.items():
        p6_mean = phase6_overall.loc[phase6_overall.statistic == "mean", p6_key].iloc[0]
        p9_mean, p9_sd = p9_seed42[p9_key].mean(), p9_seed42[p9_key].std()
        p10_m, p10_s = p10_mean[metric], p10_std[metric]
        comparison_rows.append({"metric": metric, "phase6_mean": p6_mean, "phase9_mean": p9_mean, "phase9_sd": p9_sd,
                                "phase10_mean": p10_m, "phase10_sd": p10_s, "phase10_minus_phase9": p10_m - p9_mean})
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT / "comparisons" / "phase9_vs_phase10.csv", index=False)
    log(f"\nPhase 6 / Phase 9 / Phase 10 comparison:\n{comparison_df.to_string(index=False)}")
    log("NOTE: Phase 9 used a single 85/15 inner-validation split for model selection; Phase 10 used genuine "
        "inner K-fold CV for the EN/architecture axes. These are different validation structures, disclosed here explicitly "
        "as instructed, not treated as a strictly apples-to-apples comparison beyond the shared outer-test evaluation protocol.")

    fold_comp_rows = []
    for outer_idx in range(N_OUTER):
        p9_row = p9_seed42[p9_seed42.fold == outer_idx].iloc[0]
        p10_row = p10_seed42[p10_seed42.outer_fold == outer_idx].iloc[0]
        fold_comp_rows.append({"outer_fold": outer_idx, "phase9_accuracy": p9_row["test_accuracy"], "phase10_accuracy": p10_row["test_accuracy"],
                               "accuracy_diff": p10_row["test_accuracy"] - p9_row["test_accuracy"],
                               "phase9_roc_auc": p9_row["test_ROC_AUC"], "phase10_roc_auc": p10_row["test_ROC_AUC"],
                               "roc_auc_diff": p10_row["test_ROC_AUC"] - p9_row["test_ROC_AUC"]})
    fold_comp_df = pd.DataFrame(fold_comp_rows)
    fold_comp_df.to_csv(OUT / "comparisons" / "fold_level_comparison.csv", index=False)
    log(f"\nFold-level Phase 9 vs Phase 10 differences:\n{fold_comp_df.to_string(index=False)}")
    n_folds_improved = int((fold_comp_df.accuracy_diff > 0).sum())
    log(f"Accuracy improved (Phase 10 vs Phase 9) in {n_folds_improved}/5 outer folds. With n=5 paired folds, "
        f"statistical power is very limited -- no significance test is manufactured from this.")

    # ═══════════════════════════════════════════════════════════════════
    # SAVE MANIFEST / LEAKAGE AUDIT / AUDIT
    # ═══════════════════════════════════════════════════════════════════
    pd.DataFrame(manifest_rows).to_csv(OUT / "nested_cv" / "experiment_manifest.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT / "nested_cv" / "nested_cv_results.csv", index=False)
    pd.DataFrame(leakage_audit_rows).to_csv(OUT / "nested_cv" / "leakage_audit.csv", index=False)

    post_guard = fp_snapshot()
    guard_unchanged = (pre_guard == post_guard)
    log(f"\nAUDIT: Phases 1-9 files unchanged: {'PASS' if guard_unchanged else 'FAIL'}")
    if not guard_unchanged:
        changed = [k for k in post_guard if pre_guard.get(k) != post_guard.get(k)]
        problems.append(f"Prior-phase files modified during Phase 10: {changed[:20]}")

    status = "PASS" if (not problems and guard_unchanged) else "BLOCKED"

    _write_reports(final_config, comparison_df, fold_comp_df, n_folds_improved, en_nested_df, arch_nested_df,
                  reg_df, es_df, lr_df, repr_summary, clf_summary, site_df, site_imbalance_df, seed_summary,
                  p10_mean, p10_std, status, problems, guard_unchanged)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "phase10_status": status, "problems": problems,
        "final_locked_configuration": final_config,
        "phase10_test_mean": p10_mean, "phase10_test_sd": p10_std,
        "phase9_vs_phase10": comparison_df.to_dict(orient="records"),
        "accuracy_improved_vs_phase9_n_folds_of_5": n_folds_improved,
        "site_aware_mean_test_accuracy": float(site_df.test_accuracy.mean()),
        "audit": {"previous_phases_unchanged": guard_unchanged, "outer_test_used_for_selection": False},
    }
    with open(OUT / "phase10_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(OUT / "phase10_summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2, default=str))
    with open(OUT / "logs" / "phase10_audit.log", "w") as f:
        f.write("\n".join(audit_lines))
    log(f"\nFINAL PHASE 10 STATUS: {status}")
    return summary


def _write_reports(final_config, comparison_df, fold_comp_df, n_folds_improved, en_nested_df, arch_nested_df,
                   reg_df, es_df, lr_df, repr_summary, clf_summary, site_df, site_imbalance_df, seed_summary,
                   p10_mean, p10_std, status, problems, guard_unchanged):
    methods_md = f"""# Phase 10 Methods

## Design actually implemented (disclosed scope relative to the idealized full-nested-grid design)

The outer 5-fold split is identical in convention to Phase 6/9 (StratifiedKFold on diagnosis, seed=42).
Within each outer fold's TRAINING portion, a genuine {N_INNER}-fold inner StratifiedKFold (seed=42) was
used for the two experiment groups most directly tied to the Phase 9 validation/test gap: Elastic Net
regularization (Experiment A) and SSAE architecture/model size (Experiment D). Each outer fold selects
its own best configuration from its own inner CV only (true nested CV, no pooling across outer folds for
selection); the single configuration used for the final locked/deployed model is the majority vote across
the 5 outer folds' independent nested selections (documented convention, not itself chosen from test data).

For compute tractability, the inner fold count was set to {N_INNER} (not a larger k), the Elastic Net grid
used 9 C values x 1 fixed l1_ratio plus 2 extra l1_ratio values (11 unique configs) rather than the full
9x5 grid, and the remaining axes (regularization, learning rate, early stopping) used a single controlled
global sweep (one inner split per outer fold, reusing the majority-vote EN+architecture) rather than a
full nested search repeated per configuration. The feature-stability-threshold and representation-check
(PCA) experiments were evaluated with a fast Logistic Regression probe rather than the full SSAE, since
their purpose is to compare feature-selection/representation strategies, not to retune the deep model —
this is disclosed explicitly, not hidden. The classifier-comparison experiment (Experiment C) intentionally
does NOT touch the outer test set at all in Phase 10 (unlike Phase 9, which reported test metrics for
comparators) — Phase 10's leakage rules place stronger emphasis on minimizing outer-test contact, so
comparators here are evaluated on inner-validation only.

Outer test folds were touched exactly twice: (1) the final locked configuration's one-time evaluation
across 3 predefined seeds, performed only after every hyperparameter was fixed from inner-CV/majority-vote
results; and (2) the separate site-grouped (StratifiedGroupKFold on site) secondary evaluation, which
reuses the already-locked configuration with no new hyperparameter selection.

## Selection criterion (defined before any outer-test access)

Primary: mean inner-CV ROC-AUC. Secondary: mean inner-CV balanced accuracy. Tie-break (within 0.005
ROC-AUC): prefer lower Elastic Net C / lower architecture parameter count (simpler model).
"""
    (OUT / "reports" / "PHASE10_METHODS.md").write_text(methods_md, encoding="utf-8")

    results_md = f"""# Phase 10 Results

## Final locked configuration

{json.dumps(final_config, indent=2, default=str)}

## Nested Elastic Net search (per-outer-fold, top rows)

{en_nested_df.sort_values('mean_inner_ROC_AUC', ascending=False).head(10).to_string(index=False)}

## Nested SSAE architecture search (per-outer-fold)

{arch_nested_df.to_string(index=False)}

## Regularization / learning-rate / early-stopping global sweep

{reg_df.to_string(index=False)}

{lr_df.to_string(index=False)}

{es_df.to_string(index=False)}

## Representation check (mean inner-CV ROC-AUC, Logistic Regression probe)

{repr_summary.to_string(index=False)}

## Classifier comparison (mean inner-CV validation metrics; outer test NOT accessed)

{clf_summary.to_string()}

## Site-imbalance measurement (ordinary stratified-by-diagnosis outer folds)

{site_imbalance_df.to_string(index=False)}

## Site-grouped secondary evaluation (locked configuration, no re-selection)

{site_df.to_string(index=False)}

## Final outer-test performance (mean +/- SD across 5 outer folds, seed=42)

Accuracy: {p10_mean['accuracy']:.4f} +/- {p10_std['accuracy']:.4f}
Balanced accuracy: {p10_mean['balanced_accuracy']:.4f} +/- {p10_std['balanced_accuracy']:.4f}
Sensitivity: {p10_mean['sensitivity']:.4f} +/- {p10_std['sensitivity']:.4f}
Specificity: {p10_mean['specificity']:.4f} +/- {p10_std['specificity']:.4f}
F1: {p10_mean['f1']:.4f} +/- {p10_std['f1']:.4f}
ROC-AUC: {p10_mean['roc_auc']:.4f} +/- {p10_std['roc_auc']:.4f}

## Multi-seed stability

{seed_summary.to_string(index=False)}

## Phase 6 / Phase 9 / Phase 10 comparison

{comparison_df.to_string(index=False)}

## Fold-level Phase 9 vs Phase 10 differences

{fold_comp_df.to_string(index=False)}

Accuracy improved (Phase 10 vs Phase 9) in {n_folds_improved}/5 outer folds. With only 5 paired folds,
this is reported descriptively, not as a statistically significant result.
"""
    (OUT / "reports" / "PHASE10_RESULTS.md").write_text(results_md, encoding="utf-8")

    limitations_md = """# Phase 10 Limitations

- Nested CV used k=2 inner folds (not a larger k) for compute tractability -- a coarser inner-CV
  estimate than a typical 5-fold inner CV would give.
- The Elastic Net and architecture grids were reduced/controlled searches, not exhaustive grids.
- Regularization, learning rate, and early-stopping selection used a single global controlled sweep
  (not full nested CV repeated per configuration), reusing the majority-vote EN+architecture choice.
- The feature-stability-threshold and representation/PCA experiments were evaluated with a fast
  Logistic Regression probe rather than the full SSAE, so their conclusions describe the
  feature-representation axis specifically, not the deep model's behavior on those representations.
- The final locked configuration is a majority-vote convention across 5 independently nested-selected
  per-outer-fold configurations, not itself chosen via an additional CV layer.
- Phase 9 and Phase 10 used different validation structures (single 85/15 split vs. genuine inner
  K-fold CV); the Phase 9 vs Phase 10 comparison is limited to the shared outer-test evaluation
  protocol, not the internal selection process.
- Only 5 outer folds are available for the final Phase 9 vs Phase 10 comparison -- very limited
  statistical power; no significance test is manufactured from n=5 paired differences.
- The site-grouped secondary evaluation reuses the primary track's locked configuration without any
  site-grouped hyperparameter re-selection, so it measures generalization of a configuration tuned
  under ordinary (non-site-grouped) nested CV, not a jointly-optimized site-robust model.
- Multi-site heterogeneity (20 acquisition sites), the site-concentrated constant-ROI subjects
  documented in Phase 4.1/4.2, and Strategy C fold-scoped imputation all remain unchanged limitations
  carried forward from Phase 6/8/9.
- No external validation cohort, no clinical validation, and no biological/experimental validation
  were performed.
- Any accuracy difference observed here reflects this specific leakage-controlled nested-CV framework
  on this specific dataset; it does not establish general superiority, clinical usefulness, or
  biological validity.
"""
    (OUT / "reports" / "PHASE10_LIMITATIONS.md").write_text(limitations_md, encoding="utf-8")

    final_report_md = f"""# Phase 10 Final Report

**PHASE 10 STATUS: {status}**

Genuine nested cross-validation (inner K-fold within each outer training fold) was used to select
Elastic Net regularization and SSAE architecture, directly addressing the Phase 9 finding that a
single 85/15 validation split gave near-ceiling validation scores unrelated to outer-test performance.
Remaining hyperparameters used a controlled global sweep. A separate site-grouped secondary evaluation
tested generalization to unseen acquisition sites using the already-locked configuration.

See `PHASE10_METHODS.md` for the full disclosed design (including every scope reduction),
`PHASE10_RESULTS.md` for the complete numeric results, and `PHASE10_LIMITATIONS.md` for the
limitations list.

Problems encountered: {problems if problems else 'None.'}
Phases 1-9 confirmed unchanged: {guard_unchanged}.
"""
    (OUT / "reports" / "PHASE10_FINAL_REPORT.md").write_text(final_report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
