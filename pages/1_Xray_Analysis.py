"""PulmoVision AI - Chest X-ray Analysis Page

Provides interactive radiograph upload, 1-click test patient demos,
DenseNet121 inference, Monte Carlo Dropout uncertainty, and Grad-CAM visualization.
Styled with deep navy sidebar (#10243A), teal accents (#0D9488), and ice-blue theme (#F0F7FA).
"""

from io import BytesIO
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

from config import (
    BEST_MODEL_PATH,
    DATA_DIR,
    DEFAULT_MC_SAMPLES,
    MEDICAL_DISCLAIMER,
)
from src.dataset_loader import list_supported_image_files
from src.predict import run_inference_pipeline
from src.model import load_trained_model
from src.ui_components import (
    apply_custom_theme,
    render_disclaimer,
    render_header,
    render_image_preview,
    render_sidebar,
)

st.set_page_config(
    page_title="Chest X-ray Analysis - PulmoVision AI",
    page_icon="🫁",
    layout="wide",
)

apply_custom_theme()
render_sidebar()

render_header(
    title="Chest X-ray Analysis & Triage Support",
    subtitle="Interactive Deep Learning Classification with Grad-CAM and Uncertainty Quantification",
    icon="🔍",
)

render_disclaimer()

# Model Cache in Streamlit
@st.cache_resource
def get_cached_model():
    if not BEST_MODEL_PATH.exists():
        return None
    try:
        return load_trained_model(BEST_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None

# Initialize session state for selected image
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None
if "selected_image_name" not in st.session_state:
    st.session_state["selected_image_name"] = ""

# ----------------- INPUT SELECTION SECTION -----------------
st.markdown("### 📥 1. Select or Upload Chest Radiograph")

tab_upload, tab_samples = st.tabs(["📤 Upload Your X-ray File", "⚡ 1-Click Test Demo Patients"])

with tab_upload:
    uploaded_file = st.file_uploader(
        label="Upload Chest X-ray Image",
        type=["jpg", "jpeg", "png"],
        help="Upload an AP (Anterior-Posterior) or PA (Posteroanterior) chest radiograph in JPG, JPEG, or PNG format.",
        key="uploader_widget",
    )
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding: 10px 14px; background: #FFFFFF; border-radius: 10px; border: 1px solid #E2E8F0; color: #475569; font-size: 13px; box-shadow: 0 2px 6px rgba(16, 36, 58, 0.03);">
            <span>📁 <b>Supported Formats:</b> JPG, JPEG, PNG</span>
            <span>⚖️ <b>Max File Size:</b> 200 MB</span>
            <span>🩻 <b>Recommended View:</b> Standard AP / PA Radiograph</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if uploaded_file is not None:
        st.session_state["selected_image"] = uploaded_file
        st.session_state["selected_image_name"] = uploaded_file.name



with tab_samples:
    with st.container(border=True):
        st.markdown("#### Test the system instantly using verified test-set patients:")
        test_dir = DATA_DIR / "test"
        gallery_dir = DATA_DIR.parent.parent / "assets" / "gallery"
        
        sample_col1, sample_col2 = st.columns(2)
        
        norm_samples = list_supported_image_files(test_dir / "NORMAL", limit=10)
        pneu_samples = list_supported_image_files(test_dir / "PNEUMONIA", limit=10)
        if not norm_samples:
            norm_samples = list_supported_image_files(gallery_dir / "NORMAL", limit=10)
        if not pneu_samples:
            pneu_samples = list_supported_image_files(gallery_dir / "PNEUMONIA", limit=10)

        with sample_col1:
            st.markdown(
                """
                <div style="background: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 10px; padding: 14px; margin-bottom: 10px;">
                    <strong style="color: #15803D; font-size: 15px;">🟢 Patient Case A: Healthy / Normal</strong>
                    <p style="color: #166534; font-size: 13px; margin: 4px 0 0 0;">Clear pulmonary parenchyma without focal opacification.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("👉 Load Normal Demo Patient", use_container_width=True):
                if norm_samples:
                    st.session_state["selected_image"] = norm_samples[0]
                    st.session_state["selected_image_name"] = f"Normal Patient ({norm_samples[0].name})"
                    st.rerun()
                else:
                    st.error("No readable NORMAL demo image is available.")

        with sample_col2:
            st.markdown(
                """
                <div style="background: #FEF2F2; border: 1.5px solid #FCA5A5; border-radius: 10px; padding: 14px; margin-bottom: 10px;">
                    <strong style="color: #B91C1C; font-size: 15px;">🔴 Patient Case B: Pneumonia Infiltrate</strong>
                    <p style="color: #991B1B; font-size: 13px; margin: 4px 0 0 0;">Focal consolidation or diffuse parenchymal infiltration.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("👉 Load Pneumonia Demo Patient", use_container_width=True):
                if pneu_samples:
                    st.session_state["selected_image"] = pneu_samples[0]
                    st.session_state["selected_image_name"] = f"Pneumonia Patient ({pneu_samples[0].name})"
                    st.rerun()
                else:
                    st.error("No readable PNEUMONIA demo image is available.")

        st.markdown("---")
        st.caption("Browse across 624 held-out test cohort images:")
        all_samples = norm_samples[:10] + pneu_samples[:10]
        if all_samples:
            chosen = st.selectbox(
                "Select specific cohort scan:",
                all_samples,
                format_func=lambda x: f"[{x.parent.name}] {x.name}",
            )
            if st.button("📥 Load Selected Scan", use_container_width=True):
                st.session_state["selected_image"] = chosen
                st.session_state["selected_image_name"] = f"[{chosen.parent.name}] {chosen.name}"
                st.rerun()

# ----------------- WORKBENCH PREVIEW & CONTROLS -----------------
current_img = st.session_state.get("selected_image")

if current_img is not None:
    st.markdown("---")
    st.markdown("### 🔬 2. Radiograph Workbench")

    wb_col_img, wb_col_ctl = st.columns([1.1, 1.3])

    with wb_col_img:
        with st.container(border=True):
            try:
                image_metadata = render_image_preview(
                    current_img,
                    f"Active Study: {st.session_state['selected_image_name']}",
                )
                if image_metadata is None:
                    st.stop()
                image_size, image_channels = image_metadata
                st.caption(f"Dimensions: {image_size[0]}×{image_size[1]} px · Channels: {image_channels}")
            except Exception as e:
                st.error(f"Cannot decode image: {e}")
                st.stop()

    with wb_col_ctl:
        with st.container(border=True):
            st.markdown("#### Clinical Analysis Parameters")
            st.markdown(
                f"<div style='background: #10243A; border: 1.5px solid #0D9488; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 600; font-size: 13.5px;'>"
                f"<span style='color: #2DD4BF !important; -webkit-text-fill-color: #2DD4BF !important; font-weight: 700;'>Active Study:</span> "
                f"<span style='color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;'>{st.session_state['selected_image_name']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            mc_samples = st.slider(
                "Monte Carlo Uncertainty Passes (T):",
                min_value=5,
                max_value=40,
                value=DEFAULT_MC_SAMPLES,
                step=5,
                help="Runs T stochastic forward passes with active dropout to measure predictive entropy.",
            )

            enable_cam = st.checkbox("Compute High-Resolution Grad-CAM Heatmap", value=True)

            st.write("")
            analyze_btn = st.button("🚀 Analyze Radiograph Now", type="primary", use_container_width=True)

            if st.button("🔄 Clear / Choose Another Image", use_container_width=True):
                st.session_state["selected_image"] = None
                st.session_state["selected_image_name"] = ""
                st.rerun()

    # ----------------- EXECUTE DIAGNOSTIC INFERENCE -----------------
    if analyze_btn:
        model = get_cached_model()
        if model is None:
            st.error("The image preview is ready, but the trained model checkpoint is unavailable.")
            st.info("Add `models/best_model.keras` or train the model from the Model Training page before analyzing.")
            st.stop()
        with st.spinner("Executing DenseNet121 inference, computing Grad-CAM gradients, and sampling Monte Carlo Dropout..."):
            try:
                results = run_inference_pipeline(
                    image_input=current_img,
                    model=model,
                    mc_samples=mc_samples,
                    generate_cam=enable_cam,
                )
            except Exception as e:
                st.error(f"Inference pipeline failed: {str(e)}")
                st.stop()

        st.markdown("---")
        st.markdown("### 📋 3. Diagnostic Results & Clinical Triage")

        # Triage Classification Banner
        res_col1, res_col2 = st.columns([1.3, 1.7])

        with res_col1:
            with st.container(border=True):
                st.markdown('<div class="pv-metric-lbl">Triage Classification</div>', unsafe_allow_html=True)
                cat = results["triage_category"]
                if cat == "PNEUMONIA":
                    badge_html = '<span class="pv-badge-pneumonia">🚨 PNEUMONIA SUSPECTED</span>'
                elif cat == "NORMAL":
                    badge_html = '<span class="pv-badge-normal">✅ NORMAL (NO INFILTRATE)</span>'
                else:
                    badge_html = '<span class="pv-badge-uncertain">⚠️ UNCERTAIN / REVIEW REQUIRED</span>'

                st.markdown(f"<div style='margin-top: 6px; margin-bottom: 14px;'>{badge_html}</div>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 14.5px; margin-bottom: 6px;'><b>Clinical Status:</b> {results['triage_label']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 14.5px;'><b>Model Confidence:</b> <code style='color: #0D9488;'>{results['confidence_estimate']:.1%}</code></p>", unsafe_allow_html=True)

        with res_col2:
            with st.container(border=True):
                st.markdown('<div class="pv-metric-lbl">Probability Distribution</div>', unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 14px; margin-bottom: 2px;'><b>NORMAL:</b> {results['normal_probability']:.1%}</p>", unsafe_allow_html=True)
                st.progress(float(results["normal_probability"]))
                st.markdown(f"<p style='font-size: 14px; margin-top: 10px; margin-bottom: 2px;'><b>PNEUMONIA:</b> {results['pneumonia_probability']:.1%}</p>", unsafe_allow_html=True)
                st.progress(float(results["pneumonia_probability"]))

        # Uncertainty Quantification Cards
        st.markdown("### 🎲 Monte Carlo Dropout Uncertainty Quantification")
        unc = results["uncertainty"]
        ucol1, ucol2, ucol3 = st.columns(3)

        with ucol1:
            with st.container(border=True):
                st.metric(label="Predictive Entropy", value=f"{unc['predictive_entropy']} bits")
                st.caption("Information dispersion across stochastic passes")

        with ucol2:
            with st.container(border=True):
                st.metric(label="Normalized Entropy", value=f"{unc['normalized_entropy']:.3f} / 1.0")
                st.caption("0.0 = High Certainty, 1.0 = Max Ambiguity")

        with ucol3:
            with st.container(border=True):
                pneu_std = unc["std_deviations"][1]
                st.metric(label="Stochastic Std Dev (Pneumonia)", value=f"±{pneu_std:.4f}")
                st.caption(f"Evaluated over {unc['n_samples']} stochastic passes")

        # Grad-CAM Visual Explainability
        if results.get("gradcam_overlay") is not None:
            st.markdown("### 🗺️ Grad-CAM Visual Explainability")
            st.caption(
                "Gradient-weighted Class Activation Mapping calculates gradients of the predicted class score with "
                "respect to the final convolutional feature maps of DenseNet121, localizing pulmonary evidence."
            )

            cam_tab1, cam_tab2 = st.tabs(["🖼️ 3-Panel Panoramic View", "🔬 Detailed Side-by-Side Comparison"])

            with cam_tab1:
                vcol1, vcol2, vcol3 = st.columns(3)
                with vcol1:
                    with st.container(border=True):
                        st.image(results["original_image"], caption="1. Original Radiograph", use_container_width=True)
                with vcol2:
                    with st.container(border=True):
                        st.image(results["gradcam_heatmap"], caption="2. Activation Heatmap (JET)", use_container_width=True)
                with vcol3:
                    with st.container(border=True):
                        st.image(results["gradcam_overlay"], caption="3. Anatomical Overlay", use_container_width=True)

            with cam_tab2:
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        st.image(results["original_image"], caption="Original Radiograph (Full Res)", use_container_width=True)
                with c2:
                    with st.container(border=True):
                        st.image(results["gradcam_overlay"], caption="Anatomical Overlay (Highlighted Lesions)", use_container_width=True)

            # Download Options
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                buf_heat = BytesIO()
                results["gradcam_heatmap"].save(buf_heat, format="PNG")
                st.download_button(
                    "📥 Download Grad-CAM Heatmap (PNG)",
                    data=buf_heat.getvalue(),
                    file_name="pulmovision_heatmap.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with dcol2:
                buf_over = BytesIO()
                results["gradcam_overlay"].save(buf_over, format="PNG")
                st.download_button(
                    "📥 Download Anatomical Overlay (PNG)",
                    data=buf_over.getvalue(),
                    file_name="pulmovision_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )

        # Interpretation and Clinical Limitations Card
        with st.container(border=True):
            st.markdown("#### Clinical Interpretation Notes & Research Scope")
            st.markdown(
                "- **Attribution vs Diagnosis:** Grad-CAM highlights high-frequency activations that contributed to the mathematical loss function; it does not delineate true biological boundaries of consolidation.\n"
                "- **Bayesian Interpretation:** Monte Carlo Dropout simulates posterior weight distributions to capture model epistemic uncertainty. High entropy (>0.80) signals atypical radiographs or out-of-distribution features.\n"
                "- **Screening Context:** In clinical triage, false negatives (missed pneumonias) carry high risk. Sensitivity must always be prioritized alongside clinical laboratory confirmation."
            )
else:
    st.info("👆 Please upload a chest X-ray file above, or click on **'1-Click Test Demo Patients'** to test the system immediately.")
