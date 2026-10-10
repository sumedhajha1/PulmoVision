"""PulmoVision AI - Streamlit UI Styling and Components Module

Provides a premium medical dashboard design:
- Deep navy sidebar (#10243A)
- Teal accents (#0D9488)
- Ice-blue page background (#F0F7FA)
- Crisp white cards with subtle borders & rounded corners
- Standardized medical disclaimer and metric card helpers
"""

import streamlit as st
from config import MEDICAL_DISCLAIMER


def apply_custom_theme():
    """Injects premium medical dashboard CSS styling."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #10243A;
    }

    p, span, label, li {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Ice-blue page background */
    .stApp {
        background-color: #F0F7FA !important;
    }

    /* Main container max width and spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* DEEP NAVY SIDEBAR (#10243A) */
    [data-testid="stSidebar"] {
        background-color: #10243A !important;
        border-right: 1px solid #1E3A5F !important;
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #CBD5E1 !important;
        font-size: 14px !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #1E3A5F !important;
        margin: 16px 0 !important;
    }

    [data-testid="stSidebarNav"] span {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebarNav"] a:hover span {
        color: #2DD4BF !important;
    }

    /* HERO HEADER CARD (Navy gradient with bright white heading and light-blue subtitle) */
    .pv-hero-card {
        background: linear-gradient(135deg, #10243A 0%, #1E3A5F 100%) !important;
        border-radius: 14px !important;
        padding: 22px 26px !important;
        margin-bottom: 18px !important;
        border: 1px solid #2A4D77 !important;
        box-shadow: 0 6px 20px rgba(16, 36, 58, 0.12) !important;
    }

    .pv-hero-card,
    .pv-hero-card *,
    [data-testid="stMarkdownContainer"] .pv-hero-card,
    [data-testid="stMarkdownContainer"] .pv-hero-card * {
        color: #FFFFFF !important;
    }

    .pv-hero-card h1,
    .pv-hero-card .pv-hero-title,
    [data-testid="stMarkdownContainer"] .pv-hero-card h1,
    [data-testid="stMarkdownContainer"] .pv-hero-card .pv-hero-title {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        margin: 0 0 4px 0 !important;
        line-height: 1.25 !important;
        letter-spacing: -0.01em !important;
    }

    .pv-hero-card p,
    .pv-hero-card .pv-hero-subtitle,
    [data-testid="stMarkdownContainer"] .pv-hero-card p,
    [data-testid="stMarkdownContainer"] .pv-hero-card .pv-hero-subtitle {
        color: #D6E4F0 !important;
        -webkit-text-fill-color: #D6E4F0 !important;
        font-size: 14.5px !important;
        margin: 0 !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
    }

    /* Crisp White Cards & Container Borders */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 16px rgba(16, 36, 58, 0.04) !important;
        padding: 4px !important;
        margin-bottom: 12px !important;
    }

    .pv-card {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 16px rgba(16, 36, 58, 0.04) !important;
        margin-bottom: 16px !important;
    }

    .pv-card h1, .pv-card h2, .pv-card h3, .pv-card h4 {
        color: #10243A !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
    }

    .pv-card p, .pv-card li {
        color: #334155 !important;
        font-size: 14.5px !important;
        line-height: 1.65 !important;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 14px 16px !important;
        border-radius: 12px !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #64748B !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    div[data-testid="stMetricValue"] {
        color: #10243A !important;
        font-weight: 800 !important;
        font-size: 28px !important;
    }

    /* Buttons: Primary Teal (#0D9488) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 15.5px !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.28) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(13, 148, 136, 0.42) !important;
    }

    /* Secondary Buttons */
    .stButton > button[kind="secondary"], 
    .stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: #10243A !important;
        border: 1.5px solid #CBD5E1 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        border-color: #0D9488 !important;
        color: #0D9488 !important;
        background-color: #F0FDFA !important;
    }

    /* Medical File Uploader (clean dropzone styling without conflicting wrapper) */
    [data-testid="stFileUploader"] {
        width: 100% !important;
        margin: 0 0 6px 0 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploader"] > label,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
        font-size: 15.5px !important;
        font-weight: 700 !important;
        color: #10243A !important;
        margin-bottom: 8px !important;
        display: block !important;
    }

    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #0D9488 !important;
        border-radius: 12px !important;
        padding: 24px 20px !important;
        text-align: center !important;
        box-shadow: 0 2px 10px rgba(13, 148, 136, 0.05) !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #F0FDFA !important;
        border-color: #0F766E !important;
        box-shadow: 0 4px 16px rgba(13, 148, 136, 0.12) !important;
    }

    [data-testid="stFileUploadDropzone"] button {
        background-color: #FFFFFF !important;
        color: #0D9488 !important;
        border: 1.5px solid #0D9488 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 7px 22px !important;
        font-size: 14px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 6px auto !important;
    }

    [data-testid="stFileUploadDropzone"] button:hover {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border-color: #0D9488 !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.25) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #10243A !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #64748B !important;
        font-size: 12.5px !important;
    }

    /* Tabs Styling with Teal Active Indicators */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #E2E8F0 !important;
        padding: 5px !important;
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        margin-bottom: 12px !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        font-weight: 600 !important;
        padding: 9px 20px !important;
        color: #475569 !important;
        background-color: transparent !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0D9488 !important;
        box-shadow: 0 2px 8px rgba(16, 36, 58, 0.08) !important;
    }

    /* Progress bar with teal fill */
    .stProgress > div > div > div > div {
        background-color: #0D9488 !important;
    }

    /* Clinical Badges */
    .pv-badge-normal {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 8px 18px !important;
        border-radius: 30px !important;
        background: #ECFDF5 !important;
        color: #047857 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: 1.5px solid #A7F3D0 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12) !important;
    }

    .pv-badge-pneumonia {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 8px 18px !important;
        border-radius: 30px !important;
        background: #FEF2F2 !important;
        color: #B91C1C !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: 1.5px solid #FECACA !important;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.12) !important;
    }

    .pv-badge-uncertain {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        padding: 8px 18px !important;
        border-radius: 30px !important;
        background: #FFFBEB !important;
        color: #B45309 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: 1.5px solid #FDE68A !important;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.12) !important;
    }

    /* Medical Disclaimer Banner */
    .pv-disclaimer {
        background: #F0FDFA !important;
        border-left: 5px solid #0D9488 !important;
        border: 1px solid #CCFBF1 !important;
        border-left-width: 5px !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        font-size: 13.5px !important;
        color: #115E59 !important;
        margin-bottom: 18px !important;
        line-height: 1.55 !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.05) !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_sidebar():
    """Renders the standardized deep navy sidebar branding and status badge."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 0 16px 0;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                    <span style="font-size: 28px;">🫁</span>
                    <h2 style="color: #FFFFFF !important; font-weight: 800; font-size: 22px; margin: 0;">
                        PulmoVision AI
                    </h2>
                </div>
                <p style="color: #2DD4BF !important; font-size: 12.5px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.06em;">
                    Clinical Triage Support System
                </p>
                <div style="margin-top: 12px; display: inline-flex; align-items: center; gap: 6px; background: rgba(13, 148, 136, 0.2); border: 1px solid #0D9488; padding: 4px 10px; border-radius: 20px;">
                    <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #2DD4BF;"></span>
                    <span style="color: #F8FAFC !important; font-size: 12px; font-weight: 600;">DenseNet121 Checkpoint Online</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**Navigation Guide:**")
        st.markdown(
            """
            - 🏠 **Home Dashboard:** System status & workflow
            - 🔍 **X-ray Analysis:** Live inference & Grad-CAM
            - 📈 **Dataset Explorer:** Cohort distributions & gallery
            - ⚙️ **Model Training:** DenseNet121 transfer studio
            - 🧪 **Model Evaluation:** Test metrics, ROC & PR curves
            - 📚 **About & Theory:** Deep learning theory & viva prep
            """
        )
        st.markdown("---")
        st.caption("B.Tech Capstone Project · Educational & Research Use Only")


def render_header(title: str, subtitle: str, icon: str = "🫁"):
    """Renders a modern medical hero card with deep navy background, bright white heading, and light blue subtitle."""
    st.markdown(
        f"""
        <div class="pv-hero-card">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span style="font-size: 34px; background: rgba(255, 255, 255, 0.12); padding: 8px 14px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; color: #FFFFFF !important;">
                    {icon}
                </span>
                <div style="flex: 1;">
                    <h1 class="pv-hero-title" style="color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 800; font-size: 26px; margin: 0 0 4px 0; line-height: 1.25;">
                        {title}
                    </h1>
                    <p class="pv-hero-subtitle" style="color: #D6E4F0 !important; -webkit-text-fill-color: #D6E4F0 !important; font-size: 14.5px; margin: 0; font-weight: 500;">
                        {subtitle}
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_disclaimer():
    """Renders the required academic and medical disclaimer banner with crisp visibility."""
    st.markdown(
        f"""
        <div class="pv-disclaimer">
            <strong style="color: #0F766E !important;">⚠️ Academic Research Notice:</strong> 
            <span style="color: #115E59 !important;">{MEDICAL_DISCLAIMER}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
