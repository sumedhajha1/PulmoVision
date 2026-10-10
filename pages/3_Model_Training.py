"""PulmoVision AI - Model Training Dashboard

Allows configuring hyperparameters, launching Stage 1 Transfer Learning
and Stage 2 Fine-Tuning, and monitoring learning curves.
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from config import (
    BEST_MODEL_PATH,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    MODEL_CONFIG_PATH,
    OUTPUTS_DIR,
    TRAINING_HISTORY_PATH,
)
from src.dataset_loader import inspect_dataset_directory
from src.train import train_pulmovision_pipeline
from src.ui_components import apply_custom_theme, render_disclaimer, render_header, render_sidebar
from src.utils import load_json

st.set_page_config(
    page_title="Model Training - PulmoVision AI",
    page_icon="⚙️",
    layout="wide",
)

apply_custom_theme()
render_sidebar()

render_header(
    title="Model Training & Fine-Tuning Studio",
    subtitle="Configure DenseNet121 Transfer Learning, Class Weight Balancing, and Monitor Convergence",
    icon="⚙️",
)

render_disclaimer()

# Current Model Status Card
mcol1, mcol2 = st.columns([1, 1])
model_exists = BEST_MODEL_PATH.exists()
config_data = load_json(MODEL_CONFIG_PATH)

with mcol1:
    with st.container(border=True):
        if model_exists:
            st.metric(label="Active Model Checkpoint", value=BEST_MODEL_PATH.name)
            st.caption(f"Completed: {config_data.get('trained_timestamp', 'Available')} · Backbone: {config_data.get('architecture', 'DenseNet121')}")
        else:
            st.metric(label="Active Model Checkpoint", value="None (Untrained)")
            st.caption("No weights found in models/ directory")

with mcol2:
    with st.container(border=True):
        if config_data.get("final_train_acc"):
            st.metric(label="Training Accuracy", value=f"{config_data['final_train_acc']:.1%}")
            st.caption(f"Loss: {config_data.get('final_train_loss', 0):.4f} · Epochs: {config_data.get('epochs_completed', 0)}")
        else:
            st.metric(label="Training Accuracy", value="Awaiting Run")
            st.caption("Initiate training to produce convergence metrics")

# Training Hyperparameter Form
st.markdown("### 🛠️ Training Configuration")

with st.form("training_hyperparams_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        data_path_input = st.text_input("Dataset Directory:", value=str(DATA_DIR))
        batch_size_input = st.select_slider(
            "Batch Size:",
            options=[8, 16, 32, 64],
            value=DEFAULT_BATCH_SIZE,
            help="16 is recommended for CPU stability and memory efficiency.",
        )

    with c2:
        epochs_s1 = st.slider("Stage 1 Epochs (Frozen Backbone):", min_value=1, max_value=20, value=5)
        initial_lr = st.select_slider(
            "Stage 1 Learning Rate:",
            options=[1e-4, 5e-4, 1e-3, 2e-3],
            value=DEFAULT_LEARNING_RATE,
        )

    with c3:
        enable_ft = st.checkbox("Enable Stage 2 Fine-Tuning", value=False)
        epochs_s2 = st.slider(
            "Stage 2 Epochs (Unfrozen Top Blocks):",
            min_value=1,
            max_value=10,
            value=3,
            disabled=not enable_ft,
        )

    st.markdown("**Optimization Safeguards:**")
    st.markdown(
        "- ✅ Inverse Frequency Class Weighting enabled (`~2.89 : 1` ratio compensated)\n"
        "- ✅ Early Stopping (Patience = 4) and Model Checkpointing active\n"
        "- ✅ Learning Rate Reduction on Plateau (Factor = 0.5) active"
    )

    submitted = st.form_submit_button("🚀 Start Training Pipeline", type="primary", use_container_width=True)

if submitted:
    if st.session_state.get("is_training", False):
        st.warning("A training job is already actively executing.")
    else:
        # Validate dataset before starting
        ds_info = inspect_dataset_directory(data_path_input)
        if not ds_info["is_valid"]:
            st.error(f"Cannot start training: Invalid dataset at {data_path_input}")
            st.stop()

        st.session_state["is_training"] = True
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(current_epoch, total_epochs, logs):
            progress_pct = float(current_epoch / total_epochs)
            progress_bar.progress(progress_pct)
            loss_val = logs.get("loss", 0.0)
            acc_val = logs.get("accuracy", 0.0)
            status_text.markdown(
                f"**Epoch {current_epoch}/{total_epochs} completed** · Loss: `{loss_val:.4f}` · Accuracy: `{acc_val:.2%}`"
            )

        with st.spinner("Training in progress... This may take several minutes depending on CPU performance."):
            try:
                train_result = train_pulmovision_pipeline(
                    data_dir=data_path_input,
                    epochs_stage1=epochs_s1,
                    epochs_stage2=epochs_s2 if enable_ft else 0,
                    batch_size=batch_size_input,
                    learning_rate=initial_lr,
                    enable_fine_tuning=enable_ft,
                    progress_hook=update_progress,
                )
                st.success("🎉 Model training completed successfully!")
                st.session_state["is_training"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Training failed with error: {str(e)}")
                st.session_state["is_training"] = False

# Display Training Curves if available
curves_img_path = OUTPUTS_DIR / "training_curves.png"
if curves_img_path.exists():
    st.markdown("### 📈 Learning Curves (Loss & Accuracy)")
    with st.container(border=True):
        st.image(str(curves_img_path), caption="DenseNet121 Training Convergence Curves across Epochs", use_container_width=True)

# CLI Alternative Info
st.markdown("---")
with st.container(border=True):
    st.markdown("#### 💻 Background Headless Training via PowerShell")
    st.markdown(
        """
        To execute long training runs without keeping your web browser open, execute the standalone training script:
        ```powershell
        .\\.venv\\Scripts\\python.exe scripts/train_model.py --epochs 10 --batch_size 16 --learning_rate 0.001
        ```
        The script automatically calculates class weights, logs validation checkpoints, and updates `outputs/training_curves.png`.
        """
    )

