"""PulmoVision AI - Model Evaluation & Performance Analytics Page

Presents clinical diagnostic validation metrics on the held-out test set,
confusion matrix analysis, ROC and Precision-Recall curves.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

from config import (
    BEST_MODEL_PATH,
    DATA_DIR,
    EVALUATION_METRICS_PATH,
    OUTPUTS_DIR,
)
from src.evaluate import evaluate_model_on_test_set
from src.model import load_trained_model
from src.ui_components import apply_custom_theme, render_disclaimer, render_header
from src.utils import load_json

st.set_page_config(
    page_title="Model Evaluation - PulmoVision AI",
    page_icon="🧪",
    layout="wide",
)

apply_custom_theme()

render_header(
    title="Held-Out Test Set Model Evaluation",
    subtitle="Clinical Sensitivity, Specificity, ROC-AUC, and Diagnostic Error Analysis",
    icon="🧪",
)

render_disclaimer()

# Check Model Checkpoint
if not BEST_MODEL_PATH.exists():
    st.warning("⚠️ No trained model checkpoint detected. Please train the model before evaluating.")
    st.info("Visit the **Model Training** page or run: `python scripts/train_model.py`")
    st.stop()

# Re-evaluation trigger button
col_header, col_btn = st.columns([3, 1])
with col_header:
    st.markdown("### 📋 Clinical Performance Dashboard")
with col_btn:
    run_eval_clicked = st.button("🔄 Run Test Evaluation", type="primary", use_container_width=True)

if run_eval_clicked:
    with st.spinner("Evaluating model on 624 held-out test radiographs..."):
        try:
            model = load_trained_model(BEST_MODEL_PATH)
            eval_data = evaluate_model_on_test_set(
                model=model,
                data_dir=DATA_DIR,
                outputs_dir=OUTPUTS_DIR,
                batch_size=16,
            )
            st.success("Test set evaluation completed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

eval_data = load_json(EVALUATION_METRICS_PATH)

if not eval_data or "metrics" not in eval_data:
    st.info("Evaluation has not been performed on the test set yet. Click **'Run Test Evaluation'** above to generate metrics.")
    st.stop()

metrics = eval_data["metrics"]
cm = eval_data["confusion_matrix"]

st.caption(f"Last evaluated: {eval_data.get('evaluation_timestamp', 'Recent')} · Held-out sample size: {eval_data.get('total_test_samples', 624)} radiographs")

# Primary Metric Cards (Medical Triage Focus: Sensitivity & Specificity)
m1, m2, m3, m4 = st.columns(4)

with m1:
    with st.container(border=True):
        st.metric(label="Overall Accuracy", value=f"{metrics['accuracy']:.2%}")
        st.caption("Total correct classifications")

with m2:
    with st.container(border=True):
        st.metric(label="Sensitivity / Recall", value=f"{metrics['sensitivity_recall']:.2%}")
        st.caption("Pneumonia detection rate (TP / (TP+FN))")

with m3:
    with st.container(border=True):
        st.metric(label="Clinical Specificity", value=f"{metrics['specificity']:.2%}")
        st.caption("Normal identification rate (TN / (TN+FP))")

with m4:
    with st.container(border=True):
        st.metric(label="ROC-AUC Score", value=f"{metrics['roc_auc']:.4f}")
        st.caption("Discriminative separation capability")

# Secondary Metric Cards
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Precision (PPV)", f"{metrics['precision']:.2%}")
with s2:
    st.metric("Negative Pred Value (NPV)", f"{metrics['negative_predictive_value']:.2%}")
with s3:
    st.metric("F1-Score (Harmonic Mean)", f"{metrics['f1_score']:.4f}")
with s4:
    st.metric("PR-AUC Score", f"{metrics['pr_auc']:.4f}")

# Diagnostic Error Analysis & Confusion Matrix
st.markdown("### 🔲 Diagnostic Confusion Matrix")
cm_col_text, cm_col_img = st.columns([1.2, 1.8])

with cm_col_text:
    with st.container(border=True):
        st.markdown("#### Clinical Error Breakdown")
        st.markdown(
            f"""
            - **True Positives (TP):** `{cm['true_positive']}` *(Pneumonias correctly flagged)*
            - **True Negatives (TN):** `{cm['true_negative']}` *(Healthy lungs correctly identified)*
            - **False Positives (FP):** `{cm['false_positive']}` *(Type I error: Normal misclassified as Pneumonia)*
            - **False Negatives (FN):** `{cm['false_negative']}` *(Type II error: Critical missed Pneumonia cases)*
            """
        )
        st.info(
            "💡 **Clinical Screening Priority:** False negatives (FN) are significantly more dangerous in acute triage than false positives (FP). High sensitivity (95.9%) ensures patients with active infection are promptly identified."
        )

with cm_col_img:
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix on Held-Out Test Set", use_container_width=True)

# Performance Curves: ROC and Precision-Recall
st.markdown("### 📈 Diagnostic Performance Curves")
pcol1, pcol2 = st.columns(2)

with pcol1:
    roc_path = OUTPUTS_DIR / "roc_curve.png"
    if roc_path.exists():
        st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)

with pcol2:
    pr_path = OUTPUTS_DIR / "precision_recall_curve.png"
    if pr_path.exists():
        st.image(str(pr_path), caption="Precision-Recall (PR) Curve", use_container_width=True)

# Classification Report Table
csv_report_path = OUTPUTS_DIR / "classification_report.csv"
if csv_report_path.exists():
    st.markdown("### 📑 Per-Class Classification Report")
    df_cr = pd.read_csv(csv_report_path, index_col=0)
    st.dataframe(df_cr.style.format(precision=4), use_container_width=True)
