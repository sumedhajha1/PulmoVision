"""PulmoVision AI - About, Methodology, and Limitations Page

Academic documentation detailing Convolutional Neural Networks, DenseNet121,
Transfer Learning, Grad-CAM, Monte Carlo Dropout, and Ethical Considerations.
"""

import streamlit as st
from config import MEDICAL_DISCLAIMER
from src.ui_components import apply_custom_theme, render_disclaimer, render_header, render_sidebar

st.set_page_config(
    page_title="About & Methodology - PulmoVision AI",
    page_icon="📚",
    layout="wide",
)

apply_custom_theme()
render_sidebar()

render_header(
    title="Methodology, Deep Learning Theory & Limitations",
    subtitle="Comprehensive Academic Guide to DenseNet121, Grad-CAM, Monte Carlo Uncertainty, and Viva Preparation",
    icon="📚",
)

render_disclaimer()

tab_arch, tab_xai, tab_unc, tab_metrics, tab_ethics, tab_viva = st.tabs(
    [
        "1. DenseNet121 & Transfer Learning",
        "2. Explainable AI (Grad-CAM)",
        "3. Monte Carlo Uncertainty",
        "4. Clinical Metrics & Imbalance",
        "5. Limitations & Ethical AI",
        "6. Final-Year Viva Q&A Guide",
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

with tab_viva:
    st.markdown("### 🎓 Final-Year B.Tech Viva Voce Quick Reference")
    with st.container(border=True):
        st.markdown("#### High-Yield Viva Questions & Clinical Rationale")
        
        with st.expander("Q1: Why did you select DenseNet121 over ResNet50 or standard CNN?", expanded=True):
            st.markdown(
                """
                - **Answer:** DenseNet121 features **direct feed-forward connections** between all layers within dense blocks.
                - Unlike ResNet which uses summation ($x_l = H(x_{l-1}) + x_{l-1}$), DenseNet uses **feature concatenation** ($x_l = H([x_0, x_1, ..., x_{l-1}])$).
                - This promotes maximum feature reuse (preserving subtle low-level parenchymal textures) and mitigates vanishing gradients while requiring significantly fewer parameters (~7M vs ~25M in ResNet50), preventing overfitting on clinical datasets.
                """
            )

        with st.expander("Q2: The training data has 74.3% Pneumonia and 25.7% Normal scans. How did you handle this imbalance?"):
            st.markdown(
                """
                - **Answer:** We implemented **Inverse Class Frequency Weighting**:
                  $$w_c = \\frac{N_{\\text{total}}}{C \\cdot N_c}$$
                  This yielded $w_{\\text{NORMAL}} = 1.9448$ and $w_{\\text{PNEUMONIA}} = 0.6730$, creating an effective **2.89 : 1 loss penalty ratio**.
                - During backpropagation, misclassifying a minority Normal image is penalized nearly $3\\times$ higher than misclassifying a Pneumonia scan.
                - Furthermore, augmentation was applied **strictly to the training partition**, and the held-out test set was evaluated using Sensitivity, Specificity, F1, and PR-AUC.
                """
            )

        with st.expander("Q3: Why is high Sensitivity (95.9%) prioritized over Specificity (72.6%) in clinical triage?"):
            st.markdown(
                """
                - **Answer:** In acute pulmonary screening:
                  - **False Negative (FN, Type II error):** A patient with active pneumonia is sent home undiagnosed, risking severe respiratory deterioration or sepsis.
                  - **False Positive (FP, Type I error):** A healthy individual is flagged for further physician inspection or PCR testing.
                - Because FN carries catastrophic clinical risk, screening models are calibrated to maximize Sensitivity (catching 374 out of 390 pneumonias).
                """
            )

        with st.expander("Q4: How does Grad-CAM compute heatmaps and why is ReLU applied?"):
            st.markdown(
                """
                - **Answer:** Grad-CAM calculates the gradient of the predicted class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer (`conv5_block16_concat` in DenseNet121).
                - Global average pooling calculates importance weights $\\alpha_k^c$.
                - The weighted combination is passed through **ReLU** because we only want to highlight features that have a **positive influence** on the target class score (filtering out features that support other classes).
                """
            )

        with st.expander("Q5: What is the clinical benefit of Monte Carlo Dropout uncertainty estimation?"):
            st.markdown(
                """
                - **Answer:** Standard softmax probabilities suffer from overconfidence even on ambiguous or corrupted radiographs.
                - By keeping dropout active during test-time inference and computing $T=20$ forward passes, we approximate a Bayesian posterior distribution.
                - If the model's stochastic outputs vary widely (high Shannon entropy $> 0.80$ bits), the system triggers an **'Uncertain / Review Required'** flag, preventing automated triage mistakes on ambiguous scans.
                """
            )

