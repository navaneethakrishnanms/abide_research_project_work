"""
phase6_pipeline_core.py -- the per-fold real-data pipeline for Phase 6:
Strategy C imputation -> Fisher-Z -> Elastic Net (all non-zero) -> SSAE
training -> evaluation -> feature stability -> ROI mapping -> XAI -> ROAR.

Imported by phase6_strategyC_all_nonzero.py. Not run standalone.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5
ROAR_PERCENTAGES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

# ── Elastic Net configuration: FIXED, documented, NOT tuned against test data ──
EN_C = 0.1            # inverse regularization strength (sklearn convention)
EN_L1_RATIO = 0.5     # balanced L1/L2
EN_SOLVER = "saga"
EN_MAX_ITER = 3000
EN_TOL = 1e-4

# ── SSAE configuration ─────────────────────────────────────────────────
H1, H2, N_CLASSES = 500, 100, 2
PRETRAIN_EPOCHS = 40
FINETUNE_EPOCHS = 80
ROAR_FINETUNE_EPOCHS = 40   # reduced epochs for the 65 ROAR retrains, documented as a compute-scoping choice
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
SPARSITY_WEIGHT = 1e-4
FISHER_Z_CLIP = 1e-6  # clip |r| to [-(1-eps), 1-eps] before arctanh


def build_ssae(n_in, device, torch, nn):
    class SSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(n_in, H1)
            self.dec1 = nn.Linear(H1, n_in)
            self.enc2 = nn.Linear(H1, H2)
            self.dec2 = nn.Linear(H2, H1)
            self.classifier = nn.Linear(H2, N_CLASSES)
            self.act1 = nn.ReLU()  # separate module instances -- captum's DeepLift/DeepLiftShap
            self.act2 = nn.ReLU()  # require each activation module to be used exactly once

        def encode1(self, x):
            return self.act1(self.enc1(x))

        def encode2(self, h1):
            return self.act2(self.enc2(h1))

        def forward(self, x):
            return self.classifier(self.encode2(self.encode1(x)))

    return SSAE().to(device)


def pretrain_and_finetune(X_tr, y_tr, X_val, y_val, device, torch, nn, optim, seed,
                          pretrain=True, finetune_epochs=FINETUNE_EPOCHS):
    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    model = build_ssae(n_in, device, torch, nn)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    Xval_t = torch.tensor(X_val, dtype=torch.float32, device=device) if X_val is not None else None
    yval_t = torch.tensor(y_val, dtype=torch.long, device=device) if y_val is not None else None

    history = {"ae1_loss": None, "ae2_loss": None, "finetune_train_loss": [], "finetune_val_loss": []}

    if pretrain:
        # AE1: reconstruct X from enc1->dec1, L1 sparsity on hidden
        opt1 = optim.Adam(list(model.enc1.parameters()) + list(model.dec1.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
        for epoch in range(PRETRAIN_EPOCHS):
            model.train()
            perm = torch.randperm(Xtr_t.shape[0])
            total_loss = 0.0
            for i in range(0, len(perm), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb = Xtr_t[idx]
                h1 = model.encode1(xb)
                recon = model.dec1(h1)
                loss = nn.functional.mse_loss(recon, xb) + SPARSITY_WEIGHT * h1.abs().mean()
                opt1.zero_grad(); loss.backward(); opt1.step()
                total_loss += loss.item() * len(idx)
            history["ae1_loss"] = total_loss / Xtr_t.shape[0]

        # AE2: reconstruct h1 (detached) from enc2->dec2, L1 sparsity on hidden
        with torch.no_grad():
            H1_full = model.encode1(Xtr_t)
        opt2 = optim.Adam(list(model.enc2.parameters()) + list(model.dec2.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
        for epoch in range(PRETRAIN_EPOCHS):
            perm = torch.randperm(H1_full.shape[0])
            total_loss = 0.0
            for i in range(0, len(perm), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                hb = H1_full[idx]
                h2 = model.encode2(hb)
                recon = model.dec2(h2)
                loss = nn.functional.mse_loss(recon, hb) + SPARSITY_WEIGHT * h2.abs().mean()
                opt2.zero_grad(); loss.backward(); opt2.step()
                total_loss += loss.item() * len(idx)
            history["ae2_loss"] = total_loss / H1_full.shape[0]

    # Fine-tune end to end with class-weighted CrossEntropyLoss
    classes, counts = np.unique(y_tr, return_counts=True)
    weight = torch.tensor([len(y_tr) / (len(classes) * c) for c in counts], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt_ft = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(finetune_epochs):
        model.train()
        perm = torch.randperm(Xtr_t.shape[0])
        total_loss = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()
            total_loss += loss.item() * len(idx)
        history["finetune_train_loss"].append(total_loss / Xtr_t.shape[0])
        if Xval_t is not None:
            model.eval()
            with torch.no_grad():
                vloss = criterion(model(Xval_t), yval_t).item()
            history["finetune_val_loss"].append(vloss)

    return model, history


def evaluate_model(model, X_test, y_test, device, torch):
    model.eval()
    Xte_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(Xte_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    bacc = balanced_accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)  # sensitivity
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"accuracy": acc, "balanced_accuracy": bacc, "precision": prec, "sensitivity_recall": rec,
            "specificity": spec, "f1": f1, "roc_auc": auc, "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)}, preds, probs


def _run_pipeline(fc_raw, y, subj_order, diagnosis_arr, site_arr, fold_splits,
                  feature_index_df, roi_name, device, log, OUT, problems):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    n_cohort = fc_raw.shape[0]
    imputation_records = []
    fold_feature_rows = []
    fold_feature_counts = []
    fold_arch_rows = []
    fold_training_rows = []
    fold_metric_rows = []
    fold_cm_rows = []
    xai_attr_rows = []
    roar_rows = []
    fold_models = {}  # fold_idx -> (model, selected_idx, scaler-equivalent already baked in, X_test_sel, y_test)
    fold_feature_selected_matrix = np.zeros((N_FOLDS, 6670), dtype=int)

    log("\n" + "=" * 70)
    log("PER-FOLD PIPELINE: Strategy C imputation -> Fisher-Z -> Elastic Net -> SSAE -> evaluation")
    log("=" * 70)

    for fold_idx, (tr_idx, te_idx) in enumerate(fold_splits):
        t_fold_start = time.time()
        log(f"\n--- FOLD {fold_idx} (train={len(tr_idx)}, test={len(te_idx)}) ---")

        # ── STRATEGY C: fold-scoped feature-wise MEAN imputation ──────
        X_tr_raw = fc_raw[tr_idx].copy()
        X_te_raw = fc_raw[te_idx].copy()
        train_missing_mask = np.isnan(X_tr_raw)
        test_missing_mask = np.isnan(X_te_raw)
        n_missing_train = int(train_missing_mask.sum())
        n_missing_test = int(test_missing_mask.sum())

        col_means = np.nanmean(X_tr_raw, axis=0)  # computed from TRAINING data only
        n_all_nan_cols = int(np.isnan(col_means).sum())
        if n_all_nan_cols > 0:
            # a feature with zero observed values in this fold's training set -- impute with 0.0
            # (documented fallback; only relevant if a feature is constant for EVERY training subject)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
        X_tr_imputed = np.where(train_missing_mask, col_means[None, :], X_tr_raw)
        X_te_imputed = np.where(test_missing_mask, col_means[None, :], X_te_raw)  # TRAIN-derived stat applied to test

        for feat_idx in np.where(train_missing_mask.any(axis=0) | test_missing_mask.any(axis=0))[0]:
            imputation_records.append({
                "fold": fold_idx, "feature_index": int(feature_index_df.iloc[feat_idx]["feature_index"]),
                "missing_count_train": int(train_missing_mask[:, feat_idx].sum()),
                "missing_count_test": int(test_missing_mask[:, feat_idx].sum()),
                "imputation_method": "training_fold_mean",
                "training_derived_statistic": float(col_means[feat_idx]),
                "number_imputed_total": int(train_missing_mask[:, feat_idx].sum() + test_missing_mask[:, feat_idx].sum()),
            })
        log(f"  Strategy C: imputed {n_missing_train} train entries, {n_missing_test} test entries using "
            f"TRAINING-FOLD MEAN (method fixed for entire run; {n_all_nan_cols} features had zero training "
            f"observations and used a 0.0 fallback).")

        # ── FISHER-Z (with epsilon clipping) ───────────────────────────
        X_tr_clip = np.clip(X_tr_imputed, -1 + FISHER_Z_CLIP, 1 - FISHER_Z_CLIP)
        X_te_clip = np.clip(X_te_imputed, -1 + FISHER_Z_CLIP, 1 - FISHER_Z_CLIP)
        n_clipped_train = int(np.sum(X_tr_imputed != X_tr_clip))
        n_clipped_test = int(np.sum(X_te_imputed != X_te_clip))
        X_tr_z = np.arctanh(X_tr_clip)
        X_te_z = np.arctanh(X_te_clip)
        n_nonfinite = int((~np.isfinite(X_tr_z)).sum() + (~np.isfinite(X_te_z)).sum())
        if n_nonfinite > 0:
            problems.append(f"Fold {fold_idx}: {n_nonfinite} non-finite Fisher-Z values after clipping.")
            log(f"  [STOP CONDITION] {problems[-1]}")
            continue
        log(f"  Fisher-Z applied (clipped {n_clipped_train} train / {n_clipped_test} test values near +/-1 "
            f"before arctanh). All resulting values finite: {n_nonfinite == 0}.")

        # ── SCALING (fit on training fold only) ────────────────────────
        scaler = StandardScaler().fit(X_tr_z)
        X_tr_scaled = scaler.transform(X_tr_z)
        X_te_scaled = scaler.transform(X_te_z)

        # ── ELASTIC NET (fixed hyperparameters, fit on training fold only) ──
        en = LogisticRegression(penalty="elasticnet", solver=EN_SOLVER, C=EN_C, l1_ratio=EN_L1_RATIO,
                                max_iter=EN_MAX_ITER, tol=EN_TOL, class_weight="balanced", random_state=SEED)
        en.fit(X_tr_scaled, y[tr_idx])
        coef = en.coef_[0]
        selected_mask = (coef != 0)
        selected_idx = np.where(selected_mask)[0]
        N = len(selected_idx)
        fold_feature_selected_matrix[fold_idx, selected_idx] = 1
        log(f"  Elastic Net (C={EN_C}, l1_ratio={EN_L1_RATIO}, solver={EN_SOLVER}) fit on training fold only. "
            f"N (non-zero features) = {N} of 6670.")

        for k in range(6670):
            fold_feature_rows.append({
                "fold": fold_idx, "feature_index": int(feature_index_df.iloc[k]["feature_index"]),
                "roi_i": int(feature_index_df.iloc[k]["roi_i"]), "roi_j": int(feature_index_df.iloc[k]["roi_j"]),
                "roi_i_name": feature_index_df.iloc[k]["roi_i_name"], "roi_j_name": feature_index_df.iloc[k]["roi_j_name"],
                "elastic_net_coefficient": float(coef[k]), "absolute_coefficient": float(abs(coef[k])),
                "selected": bool(selected_mask[k]),
            })
        fold_feature_counts.append({
            "fold": fold_idx, "N": N, "total_features": 6670, "C": EN_C, "l1_ratio": EN_L1_RATIO,
            "solver": EN_SOLVER, "max_iter": EN_MAX_ITER, "tol": EN_TOL, "random_state": SEED,
        })

        if N == 0:
            problems.append(f"Fold {fold_idx}: Elastic Net selected 0 features -- cannot train SSAE.")
            log(f"  [STOP CONDITION] {problems[-1]}")
            continue

        X_tr_sel = X_tr_scaled[:, selected_idx]
        X_te_sel = X_te_scaled[:, selected_idx]

        # ── internal train/val split from TRAINING fold only (for monitoring, not test) ──
        y_tr = y[tr_idx]
        tr_sub_idx, val_sub_idx = train_test_split(np.arange(len(tr_idx)), test_size=0.15,
                                                    stratify=y_tr, random_state=SEED)
        X_fit, y_fit = X_tr_sel[tr_sub_idx], y_tr[tr_sub_idx]
        X_val, y_val = X_tr_sel[val_sub_idx], y_tr[val_sub_idx]

        # ── SSAE: pretrain + finetune ───────────────────────────────────
        model, history = pretrain_and_finetune(X_fit, y_fit, X_val, y_val, device, torch, nn, optim,
                                               seed=SEED, pretrain=True, finetune_epochs=FINETUNE_EPOCHS)
        fold_arch_rows.append({
            "fold": fold_idx, "input_dim_N": N, "hidden1": H1, "hidden2": H2, "output": N_CLASSES,
            "activation": "ReLU (hidden), Softmax (output, via CrossEntropyLoss)",
            "optimizer": "Adam", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "pretrain_epochs_per_layer": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS, "sparsity_weight": SPARSITY_WEIGHT, "random_seed": SEED,
        })
        fold_training_rows.append({
            "fold": fold_idx, "ae1_final_recon_loss": history["ae1_loss"], "ae2_final_recon_loss": history["ae2_loss"],
            "finetune_final_train_loss": history["finetune_train_loss"][-1],
            "finetune_final_val_loss": history["finetune_val_loss"][-1] if history["finetune_val_loss"] else None,
        })

        metrics, preds, probs = evaluate_model(model, X_te_sel, y[te_idx], device, torch)
        fold_metric_rows.append({"fold": fold_idx, "n_test": len(te_idx), **{k: v for k, v in metrics.items() if k not in ("tn", "fp", "fn", "tp")}})
        fold_cm_rows.append({"fold": fold_idx, "TN": metrics["tn"], "FP": metrics["fp"], "FN": metrics["fn"], "TP": metrics["tp"]})
        log(f"  SSAE trained ({N}->500->100->2->Softmax). Test metrics: acc={metrics['accuracy']:.4f} "
            f"bacc={metrics['balanced_accuracy']:.4f} sens={metrics['sensitivity_recall']:.4f} "
            f"spec={metrics['specificity']:.4f} f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}")

        fold_models[fold_idx] = {
            "model": model, "selected_idx": selected_idx, "X_tr_sel": X_tr_sel, "y_tr": y_tr,
            "X_te_sel": X_te_sel, "y_te": y[te_idx], "scaler_selected": True,
        }

        log(f"  Fold {fold_idx} completed in {time.time()-t_fold_start:.1f}s.")

    # ── save Elastic Net / feature outputs ──────────────────────────────
    pd.DataFrame(fold_feature_rows).to_csv(OUT / "elastic_net" / "fold_feature_selection.csv", index=False)
    pd.DataFrame(fold_feature_counts).to_csv(OUT / "elastic_net" / "fold_feature_counts.csv", index=False)
    pd.DataFrame([{
        "penalty": "elasticnet", "solver": EN_SOLVER, "C": EN_C, "l1_ratio": EN_L1_RATIO,
        "max_iter": EN_MAX_ITER, "tol": EN_TOL, "class_weight": "balanced", "random_state": SEED,
        "tuned_against_test_data": False, "hyperparameter_selection": "fixed a priori, documented, not tuned",
    }]).to_csv(OUT / "elastic_net" / "elastic_net_parameters.csv", index=False)

    completed_folds = sorted(fold_models.keys())
    n_folds_completed = len(completed_folds)
    selection_freq_rows = []
    for k in range(6670):
        n_sel = int(fold_feature_selected_matrix[:, k].sum())
        selection_freq_rows.append({
            "feature_index": int(feature_index_df.iloc[k]["feature_index"]),
            "roi_i": int(feature_index_df.iloc[k]["roi_i"]), "roi_j": int(feature_index_df.iloc[k]["roi_j"]),
            "roi_i_name": feature_index_df.iloc[k]["roi_i_name"], "roi_j_name": feature_index_df.iloc[k]["roi_j_name"],
            "n_folds_selected": n_sel, "selection_frequency": n_sel / N_FOLDS,
        })
    selection_freq_df = pd.DataFrame(selection_freq_rows)
    selection_freq_df.to_csv(OUT / "elastic_net" / "feature_selection_frequency.csv", index=False)

    stability_df = selection_freq_df.copy()
    for f in range(N_FOLDS):
        stability_df[f"fold_{f}_selected"] = fold_feature_selected_matrix[f, :]
    stability_df.to_csv(OUT / "feature_stability" / "feature_stability.csv", index=False)

    stability_hist = stability_df["n_folds_selected"].value_counts().sort_index()
    stability_summary = pd.DataFrame({
        "n_folds_selected": list(range(N_FOLDS + 1)),
        "n_features": [int(stability_hist.get(i, 0)) for i in range(N_FOLDS + 1)],
    })
    stability_summary.loc[len(stability_summary)] = ["total_unique_selected_at_least_once",
                                                       int((stability_df["n_folds_selected"] > 0).sum())]
    stability_summary.loc[len(stability_summary)] = ["total_selected_in_every_completed_fold",
                                                       int((stability_df["n_folds_selected"] == n_folds_completed).sum())]
    stability_summary.to_csv(OUT / "feature_stability" / "stability_summary.csv", index=False)
    log(f"\nFeature stability: {int((stability_df['n_folds_selected']>0).sum())} unique features selected "
        f"at least once across {n_folds_completed} completed folds; "
        f"{int((stability_df['n_folds_selected']==n_folds_completed).sum())} selected in every completed fold.")

    # ── ROI-level summary ────────────────────────────────────────────────
    roi_rows = []
    feat_roi_i = feature_index_df["roi_i"].values
    feat_roi_j = feature_index_df["roi_j"].values
    for roi_idx in range(1, 117):
        involved_mask = (feat_roi_i == roi_idx) | (feat_roi_j == roi_idx)
        involved = selection_freq_df[involved_mask]
        n_selected_conn = int((involved["n_folds_selected"] > 0).sum())
        # folds where at least one of this ROI's connections was selected:
        n_folds_any = int((fold_feature_selected_matrix[:, involved_mask].sum(axis=1) > 0).sum())
        roi_rows.append({
            "roi_index": roi_idx, "roi_name": roi_name.get(roi_idx),
            "n_connections_total": len(involved), "n_connections_selected_at_least_once": n_selected_conn,
            "mean_selection_frequency": float(involved["selection_frequency"].mean()) if len(involved) else 0.0,
            "n_folds_with_any_selected_connection": n_folds_any,
        })
    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(OUT / "roi_analysis" / "roi_selection_summary.csv", index=False)
    log("roi_analysis/roi_selection_summary.csv written.")

    # ── evaluation outputs ────────────────────────────────────────────────
    metrics_df = pd.DataFrame(fold_metric_rows)
    metrics_df.to_csv(OUT / "evaluation" / "fold_metrics.csv", index=False)
    pd.DataFrame(fold_cm_rows).to_csv(OUT / "evaluation" / "confusion_matrices.csv", index=False)
    if len(metrics_df):
        overall = metrics_df.drop(columns=["fold", "n_test"]).agg(["mean", "std"]).reset_index().rename(columns={"index": "statistic"})
        overall.to_csv(OUT / "evaluation" / "overall_metrics.csv", index=False)
    pd.DataFrame(fold_arch_rows).to_csv(OUT / "ssae" / "fold_architecture.csv", index=False)
    pd.DataFrame(fold_training_rows).to_csv(OUT / "ssae" / "training_summary.csv", index=False)
    pd.DataFrame(imputation_records).to_csv(OUT / "fisher_z" / "fisher_z_summary.csv", index=False)
    log("evaluation/, ssae/, fisher_z/ outputs written.")

    # ═══════════════════════════════════════════════════════════════════
    # XAI (real, on trained models)
    # ═══════════════════════════════════════════════════════════════════
    log("\n--- XAI: Integrated Gradients, GuidedBackprop, DeepLift, DeepLiftShap, GradientShap "
        "(captum); SHAP (GradientExplainer); LIME (subset) ---")
    try:
        from captum.attr import IntegratedGradients, GuidedBackprop, DeepLift, DeepLiftShap, GradientShap
        captum_ok = True
    except ImportError as exc:
        problems.append(f"captum import failed: {exc}")
        captum_ok = False
    try:
        import shap as shap_lib
        shap_ok = True
    except ImportError as exc:
        problems.append(f"shap import failed: {exc}")
        shap_ok = False
    try:
        from lime.lime_tabular import LimeTabularExplainer
        lime_ok = True
    except ImportError as exc:
        problems.append(f"lime import failed: {exc}")
        lime_ok = False

    for fold_idx in completed_folds:
        fm = fold_models[fold_idx]
        model, selected_idx = fm["model"], fm["selected_idx"]
        X_te_sel, X_tr_sel = fm["X_te_sel"], fm["X_tr_sel"]
        model.eval()
        Xte_t = torch.tensor(X_te_sel, dtype=torch.float32, device=device, requires_grad=True) if captum_ok else None
        baseline = torch.zeros_like(Xte_t) if captum_ok else None

        method_results = {}
        if captum_ok:
            try:
                ig = IntegratedGradients(model)
                attr_ig = ig.attribute(Xte_t, baselines=baseline, target=1, n_steps=32).detach().cpu().numpy()
                method_results["IntegratedGradients"] = attr_ig
            except Exception as exc:
                problems.append(f"Fold {fold_idx} IntegratedGradients failed: {exc}")
            try:
                gb = GuidedBackprop(model)
                attr_gb = gb.attribute(Xte_t, target=1).detach().cpu().numpy()
                method_results["GuidedBackprop"] = attr_gb
            except Exception as exc:
                problems.append(f"Fold {fold_idx} GuidedBackprop failed: {exc}")
            try:
                dl = DeepLift(model)
                attr_dl = dl.attribute(Xte_t, baselines=baseline, target=1).detach().cpu().numpy()
                method_results["DeepLift"] = attr_dl
            except Exception as exc:
                problems.append(f"Fold {fold_idx} DeepLift failed: {exc}")
            try:
                bg_idx = np.random.default_rng(SEED).choice(len(X_tr_sel), size=min(20, len(X_tr_sel)), replace=False)
                bg = torch.tensor(X_tr_sel[bg_idx], dtype=torch.float32, device=device)
                dls = DeepLiftShap(model)
                attr_dls = dls.attribute(Xte_t, baselines=bg, target=1).detach().cpu().numpy()
                method_results["DeepLiftShap"] = attr_dls
            except Exception as exc:
                problems.append(f"Fold {fold_idx} DeepLiftShap failed: {exc}")
            try:
                gs = GradientShap(model)
                bg_idx = np.random.default_rng(SEED).choice(len(X_tr_sel), size=min(20, len(X_tr_sel)), replace=False)
                bg = torch.tensor(X_tr_sel[bg_idx], dtype=torch.float32, device=device)
                attr_gs = gs.attribute(Xte_t, baselines=bg, target=1, n_samples=10).detach().cpu().numpy()
                method_results["GradientShap"] = attr_gs
            except Exception as exc:
                problems.append(f"Fold {fold_idx} GradientShap failed: {exc}")

        if shap_ok:
            try:
                bg_idx = np.random.default_rng(SEED).choice(len(X_tr_sel), size=min(30, len(X_tr_sel)), replace=False)
                bg_t = torch.tensor(X_tr_sel[bg_idx], dtype=torch.float32, device=device)
                explainer = shap_lib.GradientExplainer(model, bg_t)
                Xte_shap = torch.tensor(X_te_sel, dtype=torch.float32, device=device)
                shap_vals = explainer.shap_values(Xte_shap)
                sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals[..., 1]
                method_results["SHAP"] = np.asarray(sv)
            except Exception as exc:
                problems.append(f"Fold {fold_idx} SHAP failed: {exc}")

        if lime_ok:
            try:
                n_lime_subjects = min(5, X_te_sel.shape[0])
                lime_idx = np.random.default_rng(SEED).choice(X_te_sel.shape[0], size=n_lime_subjects, replace=False)

                def predict_fn(x):
                    with torch.no_grad():
                        t = torch.tensor(x, dtype=torch.float32, device=device)
                        return torch.softmax(model(t), dim=1).cpu().numpy()

                lime_explainer = LimeTabularExplainer(X_tr_sel, mode="classification",
                                                      class_names=["Control", "ASD"], discretize_continuous=False)
                lime_attr = np.zeros((n_lime_subjects, X_te_sel.shape[1]))
                for li, si in enumerate(lime_idx):
                    exp = lime_explainer.explain_instance(X_te_sel[si], predict_fn,
                                                          num_features=X_te_sel.shape[1], labels=(1,))
                    for feat_local_idx, weight in exp.local_exp[1]:
                        lime_attr[li, feat_local_idx] = weight
                method_results["LIME"] = lime_attr
                lime_subject_note = f"LIME computed for {n_lime_subjects} representative test subjects only (per-sample cost)"
            except Exception as exc:
                problems.append(f"Fold {fold_idx} LIME failed: {exc}")
                lime_subject_note = "LIME failed"
        else:
            lime_subject_note = "lime not available"

        for method, attr in method_results.items():
            mean_abs = np.mean(np.abs(attr), axis=0)
            for j, sel_k in enumerate(selected_idx):
                xai_attr_rows.append({
                    "fold": fold_idx, "method": method,
                    "feature_index": int(feature_index_df.iloc[sel_k]["feature_index"]),
                    "roi_i": int(feature_index_df.iloc[sel_k]["roi_i"]), "roi_j": int(feature_index_df.iloc[sel_k]["roi_j"]),
                    "roi_i_name": feature_index_df.iloc[sel_k]["roi_i_name"], "roi_j_name": feature_index_df.iloc[sel_k]["roi_j_name"],
                    "mean_abs_attribution": float(mean_abs[j]), "n_test_subjects_or_samples":
                        (n_lime_subjects if method == "LIME" else X_te_sel.shape[0]),
                })
        log(f"  Fold {fold_idx}: XAI methods computed: {list(method_results.keys())} "
            f"({lime_subject_note if lime_ok else 'lime unavailable'})")

    xai_df = pd.DataFrame(xai_attr_rows)
    xai_df.to_csv(OUT / "xai" / "xai_feature_attributions.csv", index=False)
    if len(xai_df):
        roi_xai = []
        for roi_idx in range(1, 117):
            sub = xai_df[(xai_df["roi_i"] == roi_idx) | (xai_df["roi_j"] == roi_idx)]
            if len(sub) == 0:
                continue
            roi_xai.append({"roi_index": roi_idx, "roi_name": roi_name.get(roi_idx),
                            "mean_abs_attribution_all_methods": float(sub["mean_abs_attribution"].mean()),
                            "n_attributions": len(sub)})
        pd.DataFrame(roi_xai).to_csv(OUT / "xai" / "xai_roi_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["roi_index", "roi_name", "mean_abs_attribution_all_methods", "n_attributions"]).to_csv(
            OUT / "xai" / "xai_roi_summary.csv", index=False)
    log(f"xai/ outputs written ({len(xai_df)} attribution rows).")

    # ═══════════════════════════════════════════════════════════════════
    # ROAR (real, reduced fine-tune epochs, documented)
    # ═══════════════════════════════════════════════════════════════════
    log(f"\n--- ROAR: {len(ROAR_PERCENTAGES)} percentages x {len(completed_folds)} folds "
        f"(ranking by training-fold mean |IntegratedGradients|; reduced-epoch retraining, documented) ---")
    for fold_idx in completed_folds:
        if "IntegratedGradients" not in [r for r in xai_df["method"].unique()] if len(xai_df) else True:
            pass
        fm = fold_models[fold_idx]
        model, selected_idx = fm["model"], fm["selected_idx"]
        X_tr_sel, y_tr_fold, X_te_sel, y_te_fold = fm["X_tr_sel"], fm["y_tr"], fm["X_te_sel"], fm["y_te"]
        N = X_tr_sel.shape[1]

        # Rank by TRAINING-fold mean |IG| (leakage-safe: uses training data, not test)
        if captum_ok:
            try:
                Xtr_t = torch.tensor(X_tr_sel, dtype=torch.float32, device=device, requires_grad=True)
                ig = IntegratedGradients(model)
                attr_tr = ig.attribute(Xtr_t, baselines=torch.zeros_like(Xtr_t), target=1, n_steps=32).detach().cpu().numpy()
                importance = np.mean(np.abs(attr_tr), axis=0)
            except Exception as exc:
                problems.append(f"Fold {fold_idx} ROAR ranking (training-fold IG) failed: {exc}")
                importance = None
        else:
            importance = None

        if importance is None:
            log(f"  Fold {fold_idx}: ROAR skipped -- no ranking available (captum unavailable or IG failed).")
            continue

        ranked_order = np.argsort(-importance)  # most-important first, local indices within selected_idx
        for pct in ROAR_PERCENTAGES:
            n_remove = int(round(N * pct / 100.0))
            n_remove = min(n_remove, N - 1) if N > 1 else 0  # always keep >=1 feature to fit a model
            remove_local = set(ranked_order[:n_remove].tolist())
            keep_local = [i for i in range(N) if i not in remove_local]
            n_keep = len(keep_local)
            if n_keep == 0:
                roar_rows.append({"fold": fold_idx, "percentage_removed": pct, "n_features_removed": N,
                                  "n_features_remaining": 0, "accuracy": None, "f1": None, "roc_auc": None,
                                  "notes": "all features removed, not evaluable"})
                continue
            X_tr_roar = X_tr_sel[:, keep_local]
            X_te_roar = X_te_sel[:, keep_local]
            tr_sub_idx, val_sub_idx = train_test_split(np.arange(len(y_tr_fold)), test_size=0.15,
                                                        stratify=y_tr_fold, random_state=SEED)
            roar_model, _ = pretrain_and_finetune(
                X_tr_roar[tr_sub_idx], y_tr_fold[tr_sub_idx], X_tr_roar[val_sub_idx], y_tr_fold[val_sub_idx],
                device, torch, nn, optim, seed=SEED, pretrain=False, finetune_epochs=ROAR_FINETUNE_EPOCHS)
            roar_metrics, _, _ = evaluate_model(roar_model, X_te_roar, y_te_fold, device, torch)
            roar_rows.append({"fold": fold_idx, "percentage_removed": pct, "n_features_removed": N - n_keep,
                              "n_features_remaining": n_keep, "accuracy": roar_metrics["accuracy"],
                              "balanced_accuracy": roar_metrics["balanced_accuracy"],
                              "f1": roar_metrics["f1"], "roc_auc": roar_metrics["roc_auc"], "notes": ""})
        log(f"  Fold {fold_idx}: ROAR completed for all {len(ROAR_PERCENTAGES)} percentages.")

    pd.DataFrame(roar_rows).to_csv(OUT / "roar" / "roar_results.csv", index=False)
    log(f"roar/roar_results.csv written ({len(roar_rows)} rows).")

    # ── PHASE6_REPORT.md ────────────────────────────────────────────────
    N_values = [r["N"] for r in fold_feature_counts]
    report_lines = _build_report(fold_feature_counts, metrics_df, fold_arch_rows, imputation_records,
                                 stability_df, roar_rows, problems, n_folds_completed)
    with open(OUT / "PHASE6_REPORT.md", "w") as f:
        f.write(report_lines)

    return {
        "n_folds_completed": n_folds_completed,
        "N_per_fold": N_values,
        "N_min": min(N_values) if N_values else None, "N_max": max(N_values) if N_values else None,
        "N_mean": float(np.mean(N_values)) if N_values else None,
        "N_median": float(np.median(N_values)) if N_values else None,
        "N_std": float(np.std(N_values)) if N_values else None,
        "unique_features_selected_at_least_once": int((stability_df["n_folds_selected"] > 0).sum()),
        "features_selected_every_fold": int((stability_df["n_folds_selected"] == n_folds_completed).sum()) if n_folds_completed else None,
        "fold_metrics": fold_metric_rows,
        "captum_available": captum_ok, "shap_available": shap_ok, "lime_available": lime_ok,
        "roar_rows_written": len(roar_rows),
    }


def _build_report(fold_feature_counts, metrics_df, fold_arch_rows, imputation_records,
                  stability_df, roar_rows, problems, n_folds_completed):
    N_vals = [r["N"] for r in fold_feature_counts]
    lines = ["# Phase 6 Report — Real ABIDE Modeling (Strategy C + All-Non-Zero Elastic Net)", ""]
    lines.append(f"Folds completed: {n_folds_completed}/{N_FOLDS}")
    lines.append(f"Per-fold N (non-zero Elastic Net features): {N_vals}")
    if N_vals:
        lines.append(f"min={min(N_vals)} max={max(N_vals)} mean={np.mean(N_vals):.1f} "
                     f"median={np.median(N_vals):.1f} std={np.std(N_vals):.1f}")
    lines.append("")
    lines.append("## Evaluation (per fold)")
    lines.append(metrics_df.to_markdown(index=False) if len(metrics_df) else "No folds completed.")
    lines.append("")
    lines.append("## SSAE architecture per fold")
    lines.append(pd.DataFrame(fold_arch_rows).to_markdown(index=False) if fold_arch_rows else "N/A")
    lines.append("")
    lines.append("## Problems encountered")
    lines.append("\n".join(f"- {p}" for p in problems) if problems else "None.")
    lines.append("")
    lines.append("## Paper comparison")
    lines.append("This is a MODIFIED REPRODUCTION, not an exact reproduction of the original paper. "
                 "Differences: (1) Strategy C fold-scoped imputation (paper does not specify this "
                 "handling), (2) Elastic Net instead of SVM-RFE, (3) ALL non-zero Elastic Net features "
                 "instead of a fixed 1,000, (4) variable SSAE input dimension N per fold instead of a "
                 "fixed 1,000. No claim is made that this reproduces or targets the paper's reported "
                 "accuracy.")
    return "\n\n".join(lines)
