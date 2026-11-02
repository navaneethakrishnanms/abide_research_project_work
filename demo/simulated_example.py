"""HYPOTHETICAL confusion matrix for illustration. SIMULATED, NOT MEASURED: constructed by hand for demonstration; not produced by ABIDE data,
the trained model or any experiment. Nothing here is read from or written to any evaluation file.
Every metric is computed from the four integer counts below (no hardcoded percentages)."""
TP, FN, FP, TN = 4300, 700, 744, 4256  # 10,000 hypothetical cases: 5,000 actual ASD, 5,000 actual control; 8,556 correct predictions


# Four hypothetical entries for the comparison table. Each has 10,000 invented cases (4,800 actual ASD, 5,200 actual control) so that every
# accuracy is exact to two decimals. The counts are explicit; names are generic on purpose (they are NOT real models and were not run).
HYPOTHETICAL_MODELS = [
    # name,                  TP,   FN,  FP,   TN
    ("Hypothetical model A", 3900, 900, 966, 4234),
    ("Hypothetical model B", 4000, 800, 828, 4372),
    ("Hypothetical model C", 4112, 688, 856, 4344),
    ("Hypothetical model D", 3950, 850, 859, 4341),
]


def hypothetical_entries():
    out = []
    for name, tp, fn, fp, tn in HYPOTHETICAL_MODELS:
        assert tp + fn == 4800 and fp + tn == 5200
        out.append({"name": name, "tp": tp, "fn": fn, "fp": fp, "tn": tn, **simulated_metrics(tp, fn, fp, tn)})
    return out


def simulated_metrics(tp=TP, fn=FN, fp=FP, tn=TN):
    n = tp + fn + fp + tn
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    return {
        "n": n,
        "accuracy": (tp + tn) / n,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "balanced_accuracy": (sens + spec) / 2,
    }
