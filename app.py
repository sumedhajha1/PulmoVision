"""PulmoVision AI - Main Application Entrypoint & Home Dashboard

Explainable Deep Learning-Based Pneumonia Detection and Triage Support System Using Chest X-rays.
Designed with deep navy sidebar (#10243A), teal accents (#0D9488), and ice-blue background (#F0F7FA).
"""

from pathlib import Path
import streamlit as st

from config import (
    BEST_MODEL_PATH,
    DATA_DIR,
    EVALUATION_METRICS_PATH,
    MODEL_CONFIG_PATH,
)
from src.dataset_loader import inspect_dataset_directory
from src.ui_components import apply_custom_theme, render_disclaimer, render_header, render_sidebar
from src.utils import load_json

st.set_page_config(
    page_title="PulmoVision AI - Medical Radiography Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styling
apply_custom_theme()
render_sidebar()

# Hero Header
render_header(
    title="PulmoVision AI",
    subtitle="Explainable Deep Learning-Based Pneumonia Detection and Triage Support System Using Chest X-rays",
    icon="🫁",
)

render_disclaimer()

# System Live Status Section
st.markdown("### 📊 System Status & Readiness")
col1, col2, col3 = st.columns(3)

# 1. Dataset Status
ds_report = inspect_dataset_directory(DATA_DIR)
with col1:
    with st.container(border=True):
        if ds_report["is_valid"]:
            st.metric(label="Dataset Cohort", value=f"{ds_report['total_images']:,} CXRs")
            st.caption(f"Train: {sum(ds_report['splits'].get('train', {}).values()):,} · Test: {sum(ds_report['splits'].get('test', {}).values()):,} · Val: {sum(ds_report['splits'].get('val', {}).values()):,}")
        else:
            st.metric(label="Dataset Cohort", value="Incomplete")
            st.caption("Verify data directory organization")

# 2. Model Status
with col2:
    with st.container(border=True):
        model_exists = BEST_MODEL_PATH.exists()
        model_config = load_json(MODEL_CONFIG_PATH)
        if model_exists:
            arch = model_config.get("architecture", "DenseNet121")
            st.metric(label="Active Neural Backbone", value=f"{arch}")
            st.caption(f"Checkpoint: {BEST_MODEL_PATH.name} · Weights: ImageNet")
        else:
            st.metric(label="Active Neural Backbone", value="Pending Training")
            st.caption("Navigate to Model Training page to initiate")

# 3. Test Evaluation Status
eval_metrics = load_json(EVALUATION_METRICS_PATH)
with col3:
    with st.container(border=True):
        if "metrics" in eval_metrics:
            m = eval_metrics["metrics"]
            st.metric(label="Held-Out Test Accuracy", value=f"{m.get('accuracy', 0):.1%}")
            st.caption(f"Sens: {m.get('sensitivity_recall', 0):.1%} · Spec: {m.get('specificity', 0):.1%} · AUC: {m.get('roc_auc', 0):.4f}")
        else:
            st.metric(label="Held-Out Test Accuracy", value="Not Evaluated")
            st.caption("Run evaluation on held-out test split")

# Architecture & How It Works
st.markdown("### 🔬 How PulmoVision AI Works")
col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 22px;">⚡</span>
                <h4 style="margin: 0; color: #10243A;">Diagnostic Workflow Pipeline</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            1. **Radiograph Ingestion:** High-resolution radiographs validated and standardized to $224 \\times 224 \\times 3$.
            2. **DenseNet121 Feature Maps:** Pretrained convolutional layers extract fine-grained parenchymal opacities.
            3. **Softmax Triage Classification:** Outputs calibrated likelihoods for **NORMAL** and **PNEUMONIA**.
            4. **Bayesian Monte Carlo Sampling:** $T=20$ stochastic forward passes measure predictive Shannon entropy.
            5. **Grad-CAM Visual Attribution:** Coarse activation heatmaps pinpoint specific anatomical zones influencing predictions.
            """
        )

with col_arch2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 22px;">🛡️</span>
                <h4 style="margin: 0; color: #10243A;">Clinical AI Architecture Pillars</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - **Transfer Learning:** Pretrained DenseNet121 overcomes medical data scarcity without huge compute requirements.
            - **Inverse-Frequency Weighting:** Compensates for the $2.89 : 1$ class imbalance, ensuring $95.9\\%$ sensitivity.
            - **Explainable AI (XAI):** High-resolution Grad-CAM opens the "black box" for radiologist review.
            - **Epistemic Uncertainty:** Monte Carlo Dropout flags out-of-distribution or noisy radiographs.
            - **100% On-Premise Privacy:** Pure local computation; zero patient data transmitted externally.
            """
        )

# Quick Action Cards
st.markdown("### 🚀 Fast Navigation")
qcol1, qcol2, qcol3 = st.columns(3)
with qcol1:
    with st.container(border=True):
        st.markdown("**🔍 X-ray Analysis Studio**")
        st.caption("Upload a chest radiograph or choose a 1-click test patient to run live Grad-CAM and uncertainty analysis.")
        if st.button("Launch Analysis Studio →", type="primary", use_container_width=True):
            st.switch_page("pages/1_Xray_Analysis.py")
with qcol2:
    with st.container(border=True):
        st.markdown("**📈 Cohort Dataset Explorer**")
        st.caption("Explore class distribution, split statistics, and browse pediatric radiograph sample galleries.")
        if st.button("Explore Dataset →", use_container_width=True):
            st.switch_page("pages/2_Dataset_Explorer.py")
with qcol3:
    with st.container(border=True):
        st.markdown("**🧪 Performance & Metrics**")
        st.caption("Inspect the 624-patient test evaluation, ROC curve, precision-recall curve, and confusion matrix.")
        if st.button("View Evaluation Dashboard →", use_container_width=True):
            st.switch_page("pages/4_Model_Evaluation.py")
