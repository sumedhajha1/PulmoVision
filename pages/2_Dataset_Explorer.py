"""PulmoVision AI - Dataset Explorer Page

Provides interactive exploration of the chest X-ray cohort,
class balance analysis, split distributions, and radiograph sample galleries.
Styled with deep navy sidebar (#10243A), teal accents (#0D9488), and ice-blue theme (#F0F7FA).
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from config import DATA_DIR, SUPPORTED_EXTENSIONS
from src.dataset_loader import inspect_dataset_directory, list_supported_image_files
from src.ui_components import (
    apply_custom_theme,
    render_disclaimer,
    render_header,
    render_image_preview,
    render_sidebar,
)

st.set_page_config(
    page_title="Dataset Explorer - PulmoVision AI",
    page_icon="📈",
    layout="wide",
)

apply_custom_theme()
render_sidebar()

render_header(
    title="Dataset Explorer & Cohort Statistics",
    subtitle="Audit Dataset Organization, Split Distribution, Class Imbalance, and Image Samples",
    icon="📈",
)

render_disclaimer()


def get_sample_images(split: str, class_name: str, limit: int = 4):
    """Return readable sample images across all supported file extensions."""
    dataset_class_dir = DATA_DIR / split / class_name
    gallery_class_dir = DATA_DIR.parent.parent / "assets" / "gallery" / class_name
    search_dirs = [dataset_class_dir, gallery_class_dir]
    image_paths = []
    for class_dir in search_dirs:
        image_paths.extend(list_supported_image_files(class_dir))
        if len(image_paths) >= limit:
            break

    image_paths = sorted(set(image_paths), key=lambda path: path.name.lower())
    readable_paths = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            readable_paths.append(path)
        except (OSError, ValueError):
            continue
        if len(readable_paths) >= limit:
            break
    return readable_paths

# Inspect dataset
report = inspect_dataset_directory(DATA_DIR)

dataset_available = report["exists"] and report["is_valid"]
if not dataset_available:
    st.warning(
        "The full dataset is not available in this environment. Showing verified real-image gallery samples instead."
    )
    report["splits"] = {
        "train": {"NORMAL": 0, "PNEUMONIA": 0},
        "test": {"NORMAL": 0, "PNEUMONIA": 0},
        "val": {"NORMAL": 0, "PNEUMONIA": 0},
    }

# Key Metric Cards
mcol1, mcol2, mcol3, mcol4 = st.columns(4)

total_imgs = report["total_images"]
train_counts = report["splits"].get("train", {})
train_tot = sum(train_counts.values())
test_tot = sum(report["splits"].get("test", {}).values())
val_tot = sum(report["splits"].get("val", {}).values())

with mcol1:
    with st.container(border=True):
        st.metric(label="Total Cohort", value=f"{total_imgs:,}")
        st.caption("Full dataset available" if dataset_available else "Full dataset not mounted")

with mcol2:
    with st.container(border=True):
        st.metric(label="Training Split", value=f"{train_tot:,}")
        st.caption(f"{(train_tot / total_imgs) * 100:.1f}% of entire dataset" if total_imgs else "Not available")

with mcol3:
    with st.container(border=True):
        st.metric(label="Held-Out Test", value=f"{test_tot:,}")
        st.caption(
            f"{(test_tot / total_imgs) * 100:.1f}% isolated clinical benchmark" if total_imgs else "Not available"
        )

with mcol4:
    with st.container(border=True):
        st.metric(label="Validation Split", value=f"{val_tot:,}")
        st.caption("Standard tuning benchmark")

# Visual Distribution Charts
st.markdown("### 📊 Distribution Across Partitions & Classes")

# Build tabular summary
rows = []
for split_name, counts in report["splits"].items():
    for cls_name, cnt in counts.items():
        rows.append({"Split": split_name.capitalize(), "Class": cls_name, "Count": cnt})

df_dist = pd.DataFrame(rows)

chart_col1, chart_col2 = st.columns(2)

# Medical Theme Palette: Navy #10243A (Normal) & Teal #0D9488 (Pneumonia)
color_map = {"NORMAL": "#10243A", "PNEUMONIA": "#0D9488"}

with chart_col1:
    with st.container(border=True):
        fig_bar = px.bar(
            df_dist,
            x="Split",
            y="Count",
            color="Class",
            barmode="group",
            color_discrete_map=color_map,
            title="Radiograph Counts by Split and Clinical Category",
            text_auto=True,
        )
        fig_bar.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#10243A"),
            margin=dict(t=45, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    with st.container(border=True):
        df_train = df_dist[df_dist["Split"] == "Train"]
        fig_pie = px.pie(
            df_train,
            values="Count",
            names="Class",
            color="Class",
            color_discrete_map=color_map,
            title="Training Set Class Proportion (2.89 : 1 Imbalance)",
            hole=0.48,
        )
        fig_pie.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#10243A"),
            margin=dict(t=45, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Dataset Origins & Clinical Metadata
st.markdown("### 🏥 Dataset Clinical Provenance & Acquisition Details")
with st.container(border=True):
    st.markdown("#### Cohort Characteristics & Clinical Origin")
    st.write(
        "The cohort comprises anterior-posterior (AP) chest radiographs selected from retrospective pediatric cohorts "
        "(1 to 5 years old) from **Guangzhou Women and Children's Medical Center** (Kermany et al., *Cell*, 2018)."
    )
    st.markdown(
        "- **Class Asymmetry:** In the training set, Pneumonia scans (3,875) outnumber Normal scans (1,341) by **2.89 : 1**. PulmoVision AI integrates inverse-frequency loss weighting to prevent bias toward the majority class.\n"
        "- **Rigorous Quality Control:** Radiographs were screened for clinical readability, removing low-resolution or motion-blurred scans, followed by multi-expert physician validation.\n"
        "- **Zero Test Leakage:** The 624 held-out test scans are strictly isolated from training, early stopping, and hyperparameter tuning."
    )

# Sample Radiograph Galleries
st.markdown("### 🖼️ Representative Radiograph Gallery")

tab_normal, tab_pneu = st.tabs(["🟢 Normal Chest Radiographs", "🔴 Pneumonia Chest Radiographs"])

with tab_normal:
    st.caption("Normal radiographs exhibit clear lung fields without focal consolidation or abnormal infiltrates.")
    norm_paths = get_sample_images("train", "NORMAL")
    cols = st.columns(4)
    for idx, p in enumerate(norm_paths):
        with cols[idx]:
            with st.container(border=True):
                render_image_preview(p, f"Normal Sample #{idx+1}")
    if not norm_paths:
        st.error("No readable NORMAL images were found. Check the dataset or gallery assets.")

with tab_pneu:
    st.caption("Pneumonia radiographs display diffuse interstitial opacification or dense lobar consolidation.")
    pneu_paths = get_sample_images("train", "PNEUMONIA")
    cols = st.columns(4)
    for idx, p in enumerate(pneu_paths):
        with cols[idx]:
            with st.container(border=True):
                render_image_preview(p, f"Pneumonia Sample #{idx+1}")
    if not pneu_paths:
        st.error("No readable PNEUMONIA images were found. Check the dataset or gallery assets.")
