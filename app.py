"""PulmoVision AI - Main Application Entrypoint & Home Dashboard

Explainable Deep Learning-Based Pneumonia Detection and Triage Support System Using Chest X-rays.
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
from src.ui_components import apply_custom_theme, render_disclaimer, render_header
from src.utils import load_json

st.set_page_config(
    page_title="PulmoVision AI - Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()

# Sidebar Navigation Summary
with st.sidebar:
    st.markdown("### 🫁 PulmoVision AI")
    st.caption("v1.0.0 · B.Tech Final Year Engineering Project")
    st.markdown("---")
    st.markdown("**Navigation Guide:**")
    st.markdown(
        """
        - **1. X-ray Analysis:** Upload chest X-rays for inference, Grad-CAM, and Monte Carlo uncertainty.
        - **2. Dataset Explorer:** Inspect class distributions, split sizes, and sample radiographs.
        - **3. Model Training:** Configure and launch Stage 1 & 2 DenseNet121 transfer learning.
        - **4. Model Evaluation:** View held-out test performance, ROC curve, and confusion matrix.
        - **5. About & Methods:** Deep learning theory, mathematical formulations, and viva prep.
        """
    )
    st.markdown("---")
    st.info("System: TensorFlow 2.17 · CPU Mode Active")

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
            st.metric(label="Dataset Status", value=f"Ready ({ds_report['total_images']:,})")
            st.caption(f"Train: {sum(ds_report['splits'].get('train', {}).values()):,} · Test: {sum(ds_report['splits'].get('test', {}).values()):,} · Val: {sum(ds_report['splits'].get('val', {}).values()):,}")
        else:
            st.metric(label="Dataset Status", value="Incomplete")
            st.caption("Verify data directory organization")

# 2. Model Status
with col2:
    with st.container(border=True):
        model_exists = BEST_MODEL_PATH.exists()
        model_config = load_json(MODEL_CONFIG_PATH)
        if model_exists:
            arch = model_config.get("architecture", "DenseNet121")
            st.metric(label="Model Checkpoint", value=f"Active ({arch})")
            st.caption(f"Weights: {BEST_MODEL_PATH.name} · Input: 224x224x3")
        else:
            st.metric(label="Model Checkpoint", value="Pending Training")
            st.caption("Navigate to Model Training page to initiate")

# 3. Test Evaluation Status
eval_metrics = load_json(EVALUATION_METRICS_PATH)
with col3:
    with st.container(border=True):
        if "metrics" in eval_metrics:
            m = eval_metrics["metrics"]
            st.metric(label="Held-Out Test Performance", value=f"{m.get('accuracy', 0):.1%} Acc")
            st.caption(f"Sens: {m.get('sensitivity_recall', 0):.1%} · Spec: {m.get('specificity', 0):.1%} · AUC: {m.get('roc_auc', 0):.3f}")
        else:
            st.metric(label="Held-Out Test Performance", value="Not Evaluated")
            st.caption("Run evaluation on held-out test split")

# Architecture & How It Works
st.markdown("### 🔬 How PulmoVision AI Works")
col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    with st.container(border=True):
        st.markdown("#### Diagnostic Workflow Pipeline")
        st.markdown(
            """
            1. **Image Ingestion & Quality Control:** Radiographs are validated, expanded to 3-channel RGB, and resized to $224 \\times 224$.
            2. **DenseNet121 Feature Extraction:** Pretrained convolutional feature maps extract parenchymal opacities and consolidations.
            3. **Softmax Classification:** Triage classification head predicts probabilities for **NORMAL** and **PNEUMONIA**.
            4. **Monte Carlo Dropout Sampling:** 20 stochastic passes quantify epistemic uncertainty and predictive entropy.
            5. **Grad-CAM Visual Attribution:** Class activation mapping highlights suspicious pulmonary zones influencing triage.
            """
        )

with col_arch2:
    with st.container(border=True):
        st.markdown("#### Core Engineering Pillars")
        st.markdown(
            """
            - **Transfer Learning:** DenseNet121 with ImageNet initialization overcomes medical data scarcity.
            - **Class Imbalance Handling:** Inverse-frequency weighting balances the $2.89:1$ pneumonia-to-normal ratio.
            - **Explainable AI (XAI):** High-resolution Grad-CAM provides visual explanations for medical research.
            - **Bayesian Approximation:** Monte Carlo Dropout flags ambiguous or out-of-distribution radiographs.
            - **100% On-Premise Privacy:** Pure local computation with zero external API calls or data leakage.
            """
        )

# Quick Action Cards
st.markdown("### 🚀 Quick Access")
qcol1, qcol2, qcol3 = st.columns(3)
with qcol1:
    if st.button("🔍 Analyze Chest X-ray", use_container_width=True):
        st.switch_page("pages/1_Xray_Analysis.py")
with qcol2:
    if st.button("📈 Explore Dataset", use_container_width=True):
        st.switch_page("pages/2_Dataset_Explorer.py")
with qcol3:
    if st.button("⚙️ Train or Evaluate Model", use_container_width=True):
        st.switch_page("pages/3_Model_Training.py")
