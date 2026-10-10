"""PulmoVision AI - Streamlit UI Styling and Components Module

Provides modern medical-grade styling, custom file uploader styles,
glowing buttons, high-contrast metric cards, and clinical banners.
"""

import streamlit as st
from config import MEDICAL_DISCLAIMER


def apply_custom_theme():
    """Injects modern, premium medical dashboard CSS styling."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global typography */
    html, body, [class*="css"], .stMarkdown, p, span, div {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        color: #0f172a;
    }

    /* Clean background */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    }

    /* Premium Card containers */
    .pv-card {
        background-color: #ffffff !important;
        border-radius: 14px !important;
        padding: 24px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05) !important;
        margin-bottom: 20px !important;
    }

    .pv-card h1, .pv-card h2, .pv-card h3, .pv-card h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
    }

    .pv-card p, .pv-card li, .pv-card span {
        color: #1e293b !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    /* Metric Label and Value styles */
    .pv-metric-lbl {
        font-size: 12.5px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #64748b !important;
        margin-bottom: 6px !important;
    }

    .pv-metric-val {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        line-height: 1.2 !important;
        margin-bottom: 6px !important;
    }

    .pv-metric-sub {
        font-size: 13.5px !important;
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* High-contrast Clinical Triage Badges */
    .pv-badge-normal {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        padding: 8px 18px !important;
        border-radius: 30px !important;
        background: #ecfdf5 !important;
        color: #065f46 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: 1.5px solid #a7f3d0 !important;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.15) !important;
    }

    .pv-badge-pneumonia {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        padding: 8px 18px !important;
        border-radius: 30px !important;
        background: #fef2f2 !important;
        color: #991b1b !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: 1.5px solid #fecaca !important;
        box-shadow: 0 2px 6px rgba(239, 68, 68, 0.15) !important;
    }

    .pv-badge-uncertain {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        padding: 8px 18px !important;
        border-radius: 30px !important;
        background: #fffbeb !important;
        color: #92400e !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: 1.5px solid #fde68a !important;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.15) !important;
    }

    /* Medical Disclaimer Banner */
    .pv-disclaimer {
        background: #f0fdfa !important;
        border-left: 5px solid #0d9488 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        border: 1px solid #ccfbf1 !important;
        border-left-width: 5px !important;
        font-size: 13.5px !important;
        color: #115e59 !important;
        margin-bottom: 24px !important;
        line-height: 1.6 !important;
        box-shadow: 0 2px 6px rgba(13, 148, 136, 0.06) !important;
    }

    /* Custom Medical Upload Area */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #0284c7 !important;
        border-radius: 14px !important;
        padding: 20px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.05) !important;
        transition: all 0.25s ease-in-out !important;
    }

    [data-testid="stFileUploader"]:hover {
        background-color: #f0f9ff !important;
        border-color: #0369a1 !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.15) !important;
    }

    /* Modern Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 12px 28px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(2, 132, 199, 0.45) !important;
    }

    /* Modern Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #f1f5f9 !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        color: #475569 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0284c7 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "🫁"):
    """Renders a standardized page header with a modern gradient card design."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            border-radius: 14px;
            padding: 24px 28px;
            color: #ffffff;
            margin-bottom: 22px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
        ">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 38px;">{icon}</span>
                <div>
                    <h1 style="color: #ffffff !important; font-weight: 800; font-size: 30px; margin: 0; line-height: 1.2;">
                        {title}
                    </h1>
                    <p style="color: #93c5fd !important; font-size: 15px; margin: 4px 0 0 0; font-weight: 500;">
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
            <strong style="color: #0f766e !important;">⚠️ Academic Research Notice:</strong> 
            <span style="color: #115e59 !important;">{MEDICAL_DISCLAIMER}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
