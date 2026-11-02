"""Read-only verification and tabulation of the completed fast_safe_ssae run. Trains nothing, runs nothing from the project.
Reads: results/fast_safe_ssae (checkpoints, logs, CSVs), the development manifest and the frozen baseline's saved fold results.
Writes: only into results/fast_safe_ssae/final_report."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(r"C:\abide\results")
S = R / "fast_safe_ssae"
OUT = S / "final_report"
MET = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
man = pd.read_csv(R / "fast_accuracy_experiment" / "data" / "dev_split_manifest.csv", dtype={"sid": str})
exp_n = man.groupby("outer_fold").size().to_dict(); exp_asd = man.groupby("outer_fold").y.sum().to_dict()
summary = {"folds_found": [], "problems": []}

cks = {}
for k in range(5):
    p = S / "checkpoints" / f"fold{k}.json"
    if not p.exists():
        summary["problems"].append(f"fold{k}: checkpoint missing"); continue
    try:
        d = json.load(open(p))
    except Exception as e:
        summary["problems"].append(f"fold{k}: unreadable ({e})"); continue
    if d.get("_complete") is not True:
        summary["problems"].append(f"fold{k}: not marked complete"); continue
    cks[k] = d; summary["folds_found"].append(k)

# ---- traceability: confusion matrices vs manifest, metrics recomputed from confusion counts
def from_cm(m):
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    sens, spec = tp / (tp + fn), tn / (tn + fp); prec = tp / (tp + fp) if tp + fp else 0.0
    return {"accuracy": (tn + tp) / (tn + fp + fn + tp), "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec, "precision": prec,
            "f1": 2 * prec * sens / (prec + sens) if prec + sens else 0.0}
trace = []
for k, d in cks.items():
    entries = [("EN classifier", d["en_only_metrics"])] + [(f"EN+SSAE seed {s}", v["metrics"]) for s, v in d["ssae"].items()]
    for name, m in entries:
        n_ok = (m["tn"] + m["fp"] + m["fn"] + m["tp"]) == exp_n[k]
        asd_ok = (m["fn"] + m["tp"]) == exp_asd[k]
        rec = from_cm(m); diff = max(abs(rec[x] - m[x]) for x in rec)
        trace.append({"fold": k, "method": name, "n_equals_manifest_fold_size": bool(n_ok), "ASD_count_equals_manifest": bool(asd_ok), "max_abs_metric_diff_vs_recomputed_from_confusion": diff})
tr = pd.DataFrame(trace)
summary["trace_all_consistent"] = bool(len(tr) and tr.n_equals_manifest_fold_size.all() and tr.ASD_count_equals_manifest.all() and (tr.max_abs_metric_diff_vs_recomputed_from_confusion < 1e-9).all())
summary["subject_level_predictions_saved"] = False   # checkpoints hold confusion counts and metrics only

# ---- tables
rows = []
for k, d in cks.items():
    for s, v in d["ssae"].items():
        rows.append({"method": "EN+SSAE (selected config)", "seed": int(s), "fold": k, "selected_C": d["C_selected"], "selected_ssae_config": d["ssae_config_selected"], "n_selected_features": d["n_selected_features"],
                     "final_train_loss": v["final_train_loss"], "es_holdout_loss": v["final_es_loss"], "stop_epoch": v["stop_epoch"], "seconds": v["seconds"], **v["metrics"]})
    rows.append({"method": "Elastic Net classifier (no SSAE)", "seed": np.nan, "fold": k, "selected_C": d["C_selected"], "n_selected_features": d["n_selected_features"], **d["en_only_metrics"]})
base = pd.read_csv(R / "fast_accuracy_experiment" / "fold_results.csv"); base = base[(base.atlas == "AAL116") & (base.model == "LR_L2")].sort_values("fold")
for _, b in base.iterrows():
    rows.append({"method": "Frozen baseline AAL-116 L2 logistic regression", "seed": np.nan, "fold": int(b.fold), **{m: b[m] for m in MET}, "tn": b.tn, "fp": b.fp, "fn": b.fn, "tp": b.tp})
fl = pd.DataFrame(rows); fl.to_csv(OUT / "fold_level_results.csv", index=False)

complete = len(cks) == 5
mc = []
def agg(name, g, note):
    r = {"model": name, "completed_folds": g.fold.nunique(), "subjects_evaluated": int(g.groupby("fold").apply(lambda x: (x.tn + x.fp + x.fn + x.tp).iloc[0]).sum()),
         "seeds_completed": note}
    for m in MET:
        r[f"{m}_mean"], r[f"{m}_sd"] = float(g[m].mean()), float(g[m].std(ddof=1)) if g.fold.nunique() > 1 else np.nan
    return r
b_g = fl[fl.method.str.startswith("Frozen")]; e_g = fl[fl.method.str.startswith("Elastic")]
mc.append({**agg("Frozen baseline AAL-116 L2 logistic regression", b_g, "n/a (deterministic, saved)"), "status": "complete (read from saved fast-experiment results)", "runtime_s": float(base.seconds.sum())})
mc.append({**agg("Elastic Net classifier (no SSAE)", e_g, "n/a (deterministic)"), "status": "complete" if complete else "INCOMPLETE", "runtime_s": np.nan})
for seed in (42, 123, 2026):
    g = fl[(fl.method == "EN+SSAE (selected config)") & (fl.seed == seed)]
    mc.append({**agg(f"EN+SSAE (SSAE config selected per fold by inner CV), seed {seed}" + (" [predeclared headline]" if seed == 42 else ""), g, str(seed)), "status": "complete" if len(g) == 5 else "INCOMPLETE",
               "runtime_s": float(g.seconds.sum()) if len(g) else np.nan})
mc = pd.DataFrame(mc)
# per-config rows: only inner-validation evidence exists
cfgrows = []
for name, lab in (("S1", "SSAE original (Phase 6) configuration"), ("S2", "SSAE regularized configuration"), ("S3", "SSAE regularized + stronger sparsity configuration")):
    a = [next(e for e in d["ssae_stage"] if e["config"] == name)["mean_inner_auc"] for d in cks.values()]
    sel = sum(1 for d in cks.values() if d["ssae_config_selected"] == name)
    cfgrows.append({"model": lab + " (not separately evaluated on outer test)", "completed_folds": len(cks), "status": "OUTER-TEST METRICS UNAVAILABLE: only inner-validation AUC was computed for this configuration",
                    "inner_validation_auc_mean": float(np.mean(a)), "inner_validation_auc_sd": float(np.std(a, ddof=1)) if len(a) > 1 else np.nan, "folds_selected": sel})
cfg = pd.DataFrame(cfgrows)
pd.concat([mc, cfg], ignore_index=True).to_csv(OUT / "model_comparison.csv", index=False)

# ---- paired differences
def paired(a, b):
    a = a.sort_values("fold"); b = b.sort_values("fold")
    return {m: (float((a[m].values - b[m].values).mean()), float((a[m].values - b[m].values).std(ddof=1)), int(((a[m].values - b[m].values) > 0).sum())) for m in ("accuracy", "balanced_accuracy", "roc_auc", "f1")}
s42 = fl[(fl.method == "EN+SSAE (selected config)") & (fl.seed == 42)]
summary["paired_ssae42_vs_baseline"] = paired(s42, b_g) if complete else None
summary["paired_ssae42_vs_en"] = paired(s42, e_g) if complete else None
summary["paired_en_vs_baseline"] = paired(e_g, b_g) if complete else None
if complete:
    summary["paired_by_seed_vs_baseline"] = {str(s): paired(fl[(fl.method == "EN+SSAE (selected config)") & (fl.seed == s)], b_g) for s in (42, 123, 2026)}
    summary["seed_means"] = {str(s): {m: float(fl[(fl.method == "EN+SSAE (selected config)") & (fl.seed == s)][m].mean()) for m in ("accuracy", "roc_auc", "balanced_accuracy")} for s in (42, 123, 2026)}
    summary["config_selection_counts"] = pd.Series([d["ssae_config_selected"] for d in cks.values()]).value_counts().to_dict()
    summary["C_selected"] = [d["C_selected"] for d in cks.values()]
    summary["features"] = [d["n_selected_features"] for d in cks.values()]
    summary["inner_auc_by_fold_selected_config"] = [max(e["mean_inner_auc"] for e in d["ssae_stage"]) for d in cks.values()]
    summary["outer_auc_seed42"] = [float(x) for x in s42.sort_values("fold").roc_auc]
    summary["train_loss_vs_es_loss_seed42"] = [(float(r.final_train_loss), float(r.es_holdout_loss), int(r.stop_epoch)) for _, r in s42.sort_values("fold").iterrows()]
    summary["params_ssae_incl_decoders"] = [d["n_params_ssae"] for d in cks.values()]
    summary["selection_seconds"] = [d["selection_seconds"] for d in cks.values()]
    summary["fold_seconds"] = [d["seconds"] for d in cks.values()]
    fs = [set(d["selected_features"]) for d in cks.values()]
    summary["feature_jaccard_mean"] = float(np.mean([len(a & b) / len(a | b) for i, a in enumerate(fs) for b in fs[i + 1:]]))
summary["trace_table"] = tr.round(12).to_dict(orient="records")
# existing run outputs present?
for f in ("fold_metrics.csv", "baseline_comparison.csv", "training_log.csv", "integrity_report.md", "final_report.md", "config.json", "implementation_notes.md"):
    summary[f"run_output_{f}"] = (S / f).exists()
log = (S / "logs" / "fast_safe_ssae.log").read_text(encoding="utf-8")
summary["log_has_reports_written"] = "Reports written" in log
summary["log_has_time_budget_stop"] = "TIME BUDGET" in log
summary["log_perturbation_check_pass_count"] = log.count("independent of outer-test rows: PASS")
summary["log_integrity_before_changed0"] = "changed=0" in log
json.dump(summary, open(OUT / "verification_summary.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if k != "trace_table"}, indent=1, default=str))
print(pd.concat([mc, cfg], ignore_index=True)[["model", "completed_folds", "status"] + [c for c in ["accuracy_mean", "accuracy_sd", "balanced_accuracy_mean", "roc_auc_mean", "roc_auc_sd", "inner_validation_auc_mean"] if c in pd.concat([mc, cfg]).columns]].round(4).to_string(index=False))
