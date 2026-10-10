"""PulmoVision AI - About, Methodology, and Limitations Page

Academic documentation detailing Convolutional Neural Networks, DenseNet121,
Transfer Learning, Grad-CAM, Monte Carlo Dropout, and Ethical Considerations.
"""

import streamlit as st
from config import MEDICAL_DISCLAIMER
from src.ui_components import apply_custom_theme, render_disclaimer, render_header

st.set_page_config(
    page_title="About & Methodology - PulmoVision AI",
    page_icon="📚",
    layout="wide",
)

apply_custom_theme()

render_header(
    title="Methodology, Deep Learning Theory & Limitations",
    subtitle="Comprehensive Academic Guide to DenseNet121, Grad-CAM, and Monte Carlo Uncertainty Estimation",
    icon="📚",
)

render_disclaimer()

tab_arch, tab_xai, tab_unc, tab_metrics, tab_ethics = st.tabs(
    [
        "1. DenseNet121 & Transfer Learning",
        "2. Explainable AI (Grad-CAM)",
        "3. Monte Carlo Uncertainty",
        "4. Clinical Metrics & Imbalance",
        "5. Limitations & Ethical AI",
    ]
)

with tab_arch:
    st.markdown("### 🧬 Why DenseNet121 for Chest Radiographs?")
    with st.container(border=True):
        st.markdown("#### Dense Convolutional Network (DenseNet) Principles")
        st.write(
            "In traditional convolutional networks (such as VGG or standard CNNs), the $l^{\\text{th}}$ layer receives only the output of the $(l-1)^{\\text{th}}$ layer. "
            "In **DenseNet121** (Huang et al., 2017), each layer connects directly to *all subsequent layers* in a feed-forward fashion:"
        )
        st.code("x_l = H_l([x_0, x_1, x_2, ..., x_{l-1}])", language="python")
        st.markdown("#### Clinical Advantages for Radiography:")
        st.markdown(
            "- **Feature Reuse:** Low-level textural edges (such as rib boundaries and lung contours) remain accessible to the deepest classification layers without degrading.\n"
            "- **Alleviated Vanishing Gradients:** Direct skip connections provide short gradient paths back to early layers during backpropagation.\n"
            "- **Parameter Efficiency:** Narrow feature maps (small growth rate $k=32$) prevent excessive parameter explosion compared to ResNet-152, minimizing overfitting on medical cohorts."
        )

with tab_xai:
    st.markdown("### 🗺️ Gradient-weighted Class Activation Mapping (Grad-CAM)")
    with st.container(border=True):
        st.markdown("#### Mathematical Formulation of Grad-CAM")
        st.write(
            "Grad-CAM (Selvaraju et al., 2017) uses the gradient of the unnormalized class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer:"
        )
        st.code("α_k^c = (1 / Z) * Σ_i Σ_j (∂y^c / ∂A_{i,j}^k)", language="python")
        st.write(
            "where $Z$ is the spatial area ($u \\times v$) of the feature map, capturing the importance of feature channel $k$ for class $c$."
        )
        st.write(
            "The localized coarse heatmap $L_{\\text{Grad-CAM}}^c$ is computed by taking a weighted combination followed by Rectified Linear Unit (ReLU):"
        )
        st.code("L_{Grad-CAM}^c = ReLU( Σ_k α_k^c * A^k )", language="python")
        st.info(
            "💡 **Why ReLU?** Features that negatively correlate with class $c$ (e.g. evidence supporting Normal when testing Pneumonia) are filtered out, highlighting strictly the visual evidence supporting the selected diagnosis."
        )

with tab_unc:
    st.markdown("### 🎲 Uncertainty Quantification via Monte Carlo Dropout")
    with st.container(border=True):
        st.markdown("#### Confidence vs Epistemic Uncertainty")
        st.write(
            "Standard deep networks often produce *overconfident* predictions (e.g. 99% probability on out-of-distribution inputs). "
            "Softmax output represents **relative categorical preference**, not calibrated Bayesian posterior belief."
        )
        st.markdown("#### Monte Carlo Dropout (Gal & Ghahramani, 2016):")
        st.write("By sampling $T$ stochastic dropout masks at test time:")
        st.code("p̄_c = (1 / T) * Σ_{t=1}^T p_c^{(t)}", language="python")
        st.markdown("#### Predictive Shannon Entropy:")
        st.code("H[y | x, D] = - Σ_c p̄_c * log2(p̄_c + ε)", language="python")
        st.write(
            "When all stochastic predictions agree closely, entropy approaches $0.0$ bits. When predictions fluctuate widely across dropout sub-networks, entropy approaches $1.0$ bit, triggering an **'Uncertain result — expert review required'** safety gate."
        )

with tab_metrics:
    st.markdown("### ⚖️ Clinical Performance Metrics & Class Imbalance")
    with st.container(border=True):
        st.markdown("#### Evaluation in Medical Screening Systems")
        st.markdown(
            "- **Sensitivity (Recall) = $\\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$:** Ability to correctly catch sick patients (minimizing missed pneumonias).\n"
            "- **Specificity = $\\frac{\\text{TN}}{\\text{TN} + \\text{FP}}$:** Ability to correctly identify disease-free radiographs.\n"
            "- **Precision (PPV) = $\\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$:** Probability that a flagged radiograph is truly pneumonic.\n"
            "- **F1-Score = $2 \\cdot \\frac{\\text{Prec} \\cdot \\text{Rec}}{\\text{Prec} + \\text{Rec}}$:** Harmonic mean balancing precision and recall under class imbalance."
        )
        st.markdown("#### Inverse-Frequency Class Weighting")
        st.write(
            "The dataset training cohort has 3,875 Pneumonia scans vs 1,341 Normal scans. PulmoVision AI assigns class weights $w_c = \\frac{N}{C \\cdot N_c}$ during backpropagation, penalizing minority class misclassifications by $2.89\\times$ to avoid majority-class complacency."
        )

with tab_ethics:
    st.markdown("### 🛡️ Ethical AI, Medical Limitations, & Clinical Safety")
    with st.container(border=True):
        st.markdown("#### Academic Research & Clinical Scope")
        st.markdown(
            "- **Pediatric Population Bias:** The underlying dataset is derived primarily from pediatric cohorts aged 1-5 years from a single medical center. Anatomic proportions and disease manifestations in adults or elderly patients differ significantly.\n"
            "- **Pathology Confounders:** Opacities caused by pulmonary edema, atelectasis, COVID-19, pleural effusion, or lung neoplasms can mimic pneumonia opacities on chest radiographs.\n"
            "- **DICOM vs Compressed Formats:** The training dataset utilizes 8-bit JPEG scans rather than high-bit-depth raw DICOM files (12-16 bit). Compression artifacts may discard subtle parenchymal textures.\n"
            "- **Triage Support Only:** This tool is engineered strictly as an educational and triage prioritization prototype. In a hospital workflow, AI flags studies for accelerated radiologist review; it never issues autonomous clinical prescriptions."
        )
