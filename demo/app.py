"""Streamlit demo: ABIDE-I ASD vs typical control, AAL-116 functional connectivity + L2 logistic regression.
Research demonstration only. Run:  streamlit run C:\\abide\\demo\\app.py"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_defs  # noqa: F401,E402  (needed to unpickle MeanImputeFisherZ)
import validate  # noqa: E402
import simulated_example as sim  # noqa: E402  (single source of every simulated figure)

SIM_LABEL = "SIMULATED EXAMPLE — NOT REAL MODEL PERFORMANCE."
SIM_STATUS = "SIMULATED — NOT REAL MODEL PERFORMANCE"
SIM_WARNING = SIM_LABEL + " These values are illustrative and were not produced by the trained classifier or a real experiment."
m = sim.simulated_metrics()
SIM_ACC = f"{m['accuracy']*100:.2f}%"

st.set_page_config(page_title="ABIDE-I FC demo", layout="wide")


@st.cache_resource
def load():
    meta = json.loads((HERE / "baseline_lr_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(HERE / "baseline_lr.joblib")
    return model, meta


model, meta = load()
hist = meta["historical_estimates_not_measured_on_this_estimator"]
expected_header = list(meta["input"]["roi_labels_in_order"])

st.title("ABIDE-I: ASD vs typical control from resting-state connectivity")
st.warning("Research demonstration. Not a diagnostic tool. Outputs must not be used for any clinical or personal decision.")

tab_over, tab_res, tab_sim, tab_lim, tab_pred = st.tabs(["Overview", "Existing evaluation results", "Simulated example (NOT real)", "Limitations", "Prediction (format-checked)"])

with tab_over:
    c = meta["training_cohort"]
    st.markdown(f"""
**Data.** ABIDE-I (Preprocessed Connectomes Project, CPAC pipeline, band-pass 0.01-0.1 Hz, global signal regression), AAL-116 ROI time series.
**Cohort used to fit this model.** {c['n']} development subjects ({c['n_asd']} ASD, {c['n_tc']} typical control, {c['n_sites']} sites).
**Features.** Pearson correlation between all ROI pairs: 116 x 115 / 2 = {meta['input']['n_features']} edges (upper triangle, fixed order).
**Pipeline.** {' -> '.join(meta['pipeline'])}.
**Fitting.** One fit with fixed hyperparameters (C = 1e-3, k = 2000); no search, no evaluation on the training subjects.
""")
    st.caption(f"Artifact created {meta['created_utc']}. Environment at fit time: {meta['versions']}.")

    def _html_table(headers, rows, aligns, row_style="", border=""):
        th = "padding:8px 12px;border-bottom:2px solid rgba(128,128,128,.5);"
        td = f"padding:8px 12px;border-bottom:1px solid rgba(128,128,128,.25);{border}"
        head = "".join(f"<th style='{th}text-align:{a};'>{h}</th>" for h, a in zip(headers, aligns))
        body = "".join(f"<tr style='{row_style}'>" + "".join(f"<td style='{td}text-align:{a};font-variant-numeric:tabular-nums;'>{c}</td>" for c, a in zip(r, aligns)) + "</tr>"
                       for r in rows)
        return f"<table style='width:100%;border-collapse:collapse;'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    st.header("Accuracy Comparison")
    st.subheader("Actual experimental results (measured)")
    st.markdown(_html_table(
        ["Model / Experiment", "Accuracy", "AUC", "Status"],
        [("AAL-116 + Fisher-Z + L2 Logistic Regression", "64.7%", "—", "Actual cross-validation"),
         ("Best documented development result (FC + age + sex)", "66.0%", "—", "Actual saved result"),
         ("ComBat harmonization", "64.8%", "—", "Actual cross-validation; acceptance rule not met"),
         ("Best single fold", "70.8%", "—", "Single fold, not overall accuracy"),
         ("Best documented AUC (any saved AAL-116 group)", "—", "0.708", "Actual saved result")],
        ["left", "right", "right", "left"]), unsafe_allow_html=True)

    st.subheader("Simulated comparison (hypothetical, NOT measured)")
    st.error("ILLUSTRATIVE ONLY: the entries below are invented. They are not models that were trained or run, and not results from the trained classifier.")
    _hyp = sim.hypothetical_entries()
    st.markdown(_html_table(
        ["Hypothetical entry", "Accuracy", "Sensitivity", "Specificity", "Hypothetical counts (10,000 cases)", "Status"],
        [(e["name"], f"{e['accuracy']*100:.2f}%", f"{e['sensitivity']*100:.2f}%", f"{e['specificity']*100:.2f}%",
          f"TP {e['tp']:,} · FN {e['fn']:,} · FP {e['fp']:,} · TN {e['tn']:,}", "SIMULATED — NOT MEASURED") for e in _hyp],
        ["left", "right", "right", "right", "left", "left"],
        row_style="background:rgba(220,38,38,.12);", border="border-left:0;"), unsafe_allow_html=True)
    st.caption("Accuracy, sensitivity and specificity are computed from the listed counts (`simulated_example.py`). One further worked example with all metrics "
               f"({SIM_ACC}, {m['n']:,} invented cases) is on the *Simulated example (NOT real)* tab.")
    st.info("The best documented development accuracy is 66.0%. Simulated values are illustrative, were constructed from hypothetical confusion-matrix counts "
            "and are not results from the trained classifier.")

with tab_res:
    st.subheader("Real AAL-116 results (saved out-of-fold predictions, 600 development subjects)")
    st.table(pd.DataFrame([
        ["Best documented accuracy (any saved AAL-116 group)", "66.0%", "`threshold_85_report.md`; FC + age + sex tuned, seed 2026"],
        ["Best AUC (any saved AAL-116 group)", "0.708", "test-site centering, leave-one-site-out"],
        ["Historical baseline accuracy (cross-validation)", "64.7%", "`accuracy_improvement_v1`, control E0 (AUC 0.694)"],
    ], columns=["Metric", "Value", "Source"]))
    st.caption("No saved AAL-116 result reaches 85%; the best accuracy under a threshold chosen without the held-out fold was 65.8%.")
    st.subheader("Historical cross-validation estimates (NOT measured on this fitted estimator)")
    st.table(pd.DataFrame([
        ["Accuracy", str(hist["accuracy_cv"])], ["ROC AUC", str(hist["auc_cv"])], ["Balanced accuracy", str(hist["balanced_accuracy_cv"])],
        ["Sensitivity / specificity", f"{hist['sensitivity_cv']} / {hist['specificity_cv']}"],
    ], columns=["Metric", "Value"]))
    st.write(f"Protocol: {hist['protocol']}. Source: `{hist['source']}`. {hist['caveat']}.")
    st.subheader("Other recorded experiments (development cross-validation, 5-fold)")
    st.table(pd.DataFrame([
        ["Fisher-Z Pearson + L2 logistic regression (this model's recipe)", "AAL-116, 600", "64.7%", "0.694"],
        ["Tangent-space + Ledoit-Wolf", "AAL-116, 600", "65.2%", "0.711"],
        ["Elastic Net + SSAE (Phase 10, strict nested CV)", "AAL-116, 600", "62.7%", "0.680"],
        ["Elastic Net + SSAE (Phase 10, site-grouped folds)", "AAL-116, 600", "66.3%", "0.713"],
        ["Leave-one-site-out, baseline", "AAL-116, 600", "62.2%", "-"],
        ["Older CC200 logistic regression (cross-validation, with age and sex)", "CC200, 676", "67.0-68.6%", "0.71-0.75"],
    ], columns=["Experiment", "Atlas, subjects", "Accuracy", "AUC"]))
    st.info("An older CC200 run recorded 74.1% on a 170-subject holdout that was reused across configurations. "
            "It is not independently verified and is not shown as a result of this demo. This demo's model is AAL-116, a different pipeline.")

with tab_sim:
    st.error(f"### {SIM_WARNING}")
    st.subheader(f"Simulated Evaluation Metrics — {SIM_ACC} Accuracy")
    st.markdown(f"""
These numbers were **constructed for demonstration** from hypothetical counts for {m['n']:,} invented cases. They were **not produced by the trained
classifier, the ABIDE data or any real experiment**, and nothing on this tab is read from or written to the real evaluation files. Every percentage below is
computed from the four counts. The trained model's real performance is in the *Existing evaluation results* tab (best documented accuracy 66.0%,
best AUC 0.708, historical baseline 64.7%).
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (SIMULATED)", f"{m['accuracy']*100:.2f}%")
    c2.metric("Sensitivity (SIMULATED)", f"{m['sensitivity']*100:.2f}%")
    c3.metric("Specificity (SIMULATED)", f"{m['specificity']*100:.2f}%")
    st.markdown("**Hypothetical confusion matrix (SIMULATED, NOT MEASURED)**")
    st.markdown(_html_table(
        ["Actual / Predicted", "Predicted ASD", "Predicted Control"],
        [("Actual ASD", f"TP = {sim.TP:,}", f"FN = {sim.FN:,}"), ("Actual Control", f"FP = {sim.FP:,}", f"TN = {sim.TN:,}")],
        ["left", "right", "right"], row_style="background:rgba(220,38,38,.10);"), unsafe_allow_html=True)
    st.markdown("**Simulated evaluation metrics (computed from the counts above)**")
    st.markdown(_html_table(
        ["Metric", "Formula", "Result"],
        [("Accuracy", "(TP + TN) / Total", f"{m['accuracy']*100:.2f}%"),
         ("Sensitivity / Recall", "TP / (TP + FN)", f"{m['sensitivity']*100:.2f}%"),
         ("Specificity", "TN / (TN + FP)", f"{m['specificity']*100:.2f}%"),
         ("Precision", "TP / (TP + FP)", f"{m['precision']*100:.2f}%"),
         ("F1-score", "2 × Precision × Recall / (Precision + Recall)", f"{m['f1']*100:.2f}%"),
         ("Balanced accuracy", "(Sensitivity + Specificity) / 2", f"{m['balanced_accuracy']*100:.2f}%")],
        ["left", "left", "right"], row_style="background:rgba(220,38,38,.10);"), unsafe_allow_html=True)
    st.caption(f"Rounding: percentages are rounded to two decimals. Exact values: precision = {sim.TP:,}/{sim.TP + sim.FP:,} = {m['precision']*100:.4f}%, "
               f"F1 = {2*sim.TP:,}/{2*sim.TP + sim.FP + sim.FN:,} = {m['f1']*100:.4f}%; accuracy, sensitivity, specificity and balanced accuracy are exact at two decimals.")
    st.caption(f"Under a simple equal-variance two-normal model, an accuracy like the simulated {SIM_ACC} would require an AUC of about 0.93; the best saved AUC is 0.708. "
               "This is arithmetic illustration, not an experimental finding.")
    st.error(SIM_WARNING)

with tab_lim:
    st.markdown("""
- Accuracy is modest (about 65% in cross-validation, AUC about 0.69), far from clinical usefulness. Many attempted improvements gave no reliable gain.
- All numbers are **development cross-validation estimates**; there is **no independent test set**. The earlier holdout overlaps development data and is not used here.
- Performance on unseen sites is lower (62.2% leave-one-site-out). Site, scanner, age and head-motion effects are present in the scores.
- Training-set performance of this fitted model is not reported because it would be optimistic and is not test accuracy.
- The model learned from 600 mostly pediatric/adolescent subjects from 19 sites; it should not be expected to generalize to other populations or acquisitions.
- A prediction is a statistical score for a research dataset, not a diagnosis. Synthetic or random input produces numbers but says nothing about ASD classification.
""")

with tab_pred:
    st.markdown("""
Predictions are enabled only for **AAL-116 ROI time-series files in the ABIDE PCP `rois_aal.1D` layout**:
a `#`-prefixed header with the exact 116 ROI codes in training order, then numeric rows (>= 100 timepoints, 116 columns).
The app checks format, header, shape, finiteness and variance. It **cannot verify** that the series went through the
CPAC pipeline (filtering 0.01-0.1 Hz, global signal regression); you must confirm that.
""")
    if sklearn.__version__ != meta["versions"]["sklearn"]:
        st.error(f"scikit-learn {sklearn.__version__} differs from the version used at fit time ({meta['versions']['sklearn']}). Predictions are disabled.")
        st.stop()
    up = st.file_uploader("ROI time-series file (.1D / .txt)", type=["1D", "txt"])
    attest = st.checkbox("I confirm this file comes from the ABIDE PCP CPAC filt_global AAL-116 output (or an identically processed series).")
    if up is not None:
        try:
            ts = validate.parse_timeseries(up.getvalue(), expected_header)
        except validate.InputError as e:
            st.error(f"Input rejected: {e}")
        else:
            st.success(f"Format check passed: {ts.shape[0]} timepoints x {ts.shape[1]} ROIs.")
            if not attest:
                st.info("Tick the confirmation above to enable the prediction.")
            elif st.button("Compute model score"):
                p = float(model.predict_proba(validate.build_features(ts))[0, 1])
                st.metric("Model score for ASD (probability-like, threshold 0.5)", f"{p:.3f}")
                st.write("Model output label: **" + ("ASD" if p >= 0.5 else "typical control") + "**")
                st.caption("Research output with ~65% historical cross-validated accuracy; not a diagnosis.")
