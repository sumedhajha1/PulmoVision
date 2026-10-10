"""PulmoVision AI - Dataset Explorer Page

Provides interactive exploration of the chest X-ray cohort,
class balance analysis, split distributions, and radiograph sample galleries.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from config import DATA_DIR, SUPPORTED_EXTENSIONS
from src.dataset_loader import inspect_dataset_directory
from src.ui_components import apply_custom_theme, render_disclaimer, render_header

st.set_page_config(
    page_title="Dataset Explorer - PulmoVision AI",
    page_icon="📈",
    layout="wide",
)

apply_custom_theme()

render_header(
    title="Dataset Explorer & Cohort Statistics",
    subtitle="Audit Dataset Organization, Split Distribution, Class Imbalance, and Image Samples",
    icon="📈",
)

render_disclaimer()

# Inspect dataset
report = inspect_dataset_directory(DATA_DIR)

if not report["exists"] or not report["is_valid"]:
    st.error(f"Dataset path `{DATA_DIR}` not found or invalid.")
    st.stop()

# Key Metric Cards
mcol1, mcol2, mcol3, mcol4 = st.columns(4)

total_imgs = report["total_images"]
train_counts = report["splits"].get("train", {})
train_tot = sum(train_counts.values())
test_tot = sum(report["splits"].get("test", {}).values())
val_tot = sum(report["splits"].get("val", {}).values())

with mcol1:
    with st.container(border=True):
        st.metric(label="Total Radiographs", value=f"{total_imgs:,}")
        st.caption("Across train, val, & test splits")

with mcol2:
    with st.container(border=True):
        st.metric(label="Training Cohort", value=f"{train_tot:,}")
        st.caption(f"{(train_tot/total_imgs)*100:.1f}% of total dataset")

with mcol3:
    with st.container(border=True):
        st.metric(label="Held-Out Test Set", value=f"{test_tot:,}")
        st.caption(f"{(test_tot/total_imgs)*100:.1f}% of total dataset")

with mcol4:
    with st.container(border=True):
        st.metric(label="Validation Set", value=f"{val_tot:,}")
        st.caption("Standard benchmark tuning set")

# Visual Distribution Charts
st.markdown("### 📊 Distribution Across Partitions & Classes")

# Build tabular summary
rows = []
for split_name, counts in report["splits"].items():
    for cls_name, cnt in counts.items():
        rows.append({"Split": split_name.capitalize(), "Class": cls_name, "Count": cnt})

df_dist = pd.DataFrame(rows)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_bar = px.bar(
        df_dist,
        x="Split",
        y="Count",
        color="Class",
        barmode="group",
        color_discrete_map={"NORMAL": "#0284c7", "PNEUMONIA": "#e11d48"},
        title="Sample Counts by Split and Clinical Category",
    )
    fig_bar.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    # Class composition in training set
    df_train = df_dist[df_dist["Split"] == "Train"]
    fig_pie = px.pie(
        df_train,
        values="Count",
        names="Class",
        color="Class",
        color_discrete_map={"NORMAL": "#0284c7", "PNEUMONIA": "#e11d48"},
        title="Training Set Class Proportion (Class Imbalance)",
        hole=0.45,
    )
    fig_pie.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_pie, use_container_width=True)

# Dataset Origins & Clinical Metadata
st.markdown("### 🏥 Dataset Clinical Provenance & Acquisition Details")
with st.container(border=True):
    st.markdown("#### Cohort Characteristics")
    st.write(
        "The dataset comprises anterior-posterior (AP) chest radiographs selected from retrospective cohorts of "
        "pediatric patients aged one to five years old from **Guangzhou Women and Children's Medical Center** "
        "(Kermany et al., *Cell*, 2018)."
    )
    st.markdown(
        "- **Class Imbalance:** In the training set, Pneumonia instances (3,875) outnumber Normal instances (1,341) by **2.89 : 1**. PulmoVision AI integrates inverse-frequency loss weighting to prevent bias toward the majority class.\n"
        "- **Clinical Quality Control:** Radiographs were initially screened for clinical readability, removing unreadable or low-resolution scans, followed by expert physician validation.\n"
        "- **Zero Data Leakage:** All test partitions are kept strictly held-out and isolated from training and hyperparameter search."
    )

# Sample Radiograph Galleries
st.markdown("### 🖼️ Representative Radiograph Gallery")

tab_normal, tab_pneu = st.tabs(["Normal Chest X-rays", "Pneumonia Chest X-rays"])

with tab_normal:
    st.caption("Characterized by clear lungs without opacification or consolidation.")
    norm_paths = list((DATA_DIR / "train" / "NORMAL").glob("*.jpeg"))[:4]
    cols = st.columns(4)
    for idx, p in enumerate(norm_paths):
        with cols[idx]:
            img = Image.open(p)
            st.image(img, caption=f"Normal Sample #{idx+1}", use_container_width=True)

with tab_pneu:
    st.caption("Characterized by focal lobar consolidation or diffuse interstitial opacities.")
    pneu_paths = list((DATA_DIR / "train" / "PNEUMONIA").glob("*.jpeg"))[:4]
    cols = st.columns(4)
    for idx, p in enumerate(pneu_paths):
        with cols[idx]:
            img = Image.open(p)
            st.image(img, caption=f"Pneumonia Sample #{idx+1}", use_container_width=True)
