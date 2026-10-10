"""PulmoVision AI - Model Training Pipeline Module

Implements Stage 1 Transfer Learning and Stage 2 Fine-Tuning with
early stopping, model checkpointing, class weight balancing, and history tracking.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    BEST_MODEL_PATH,
    CLASS_INDICES_PATH,
    CLASS_TO_IDX,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FINE_TUNE_EPOCHS,
    DEFAULT_FINE_TUNE_LR,
    DEFAULT_LEARNING_RATE,
    MODEL_CONFIG_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    TRAINING_HISTORY_PATH,
)
from src.dataset_loader import compute_class_weights, load_dataset_splits
from src.model import build_pulmovision_model, prepare_for_finetuning
from src.utils import logger, save_json, set_seed


class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """Callback to report epoch progress to an optional Streamlit or external listener."""

    def __init__(self, total_epochs: int, progress_hook: Optional[Callable[[int, int, dict], None]] = None):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_hook = progress_hook

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_hook:
            self.progress_hook(epoch + 1, self.total_epochs, logs or {})


def plot_and_save_training_curves(history_data: Dict[str, list], output_path: Path) -> None:
    """Generates and saves dual-panel loss and accuracy training curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history_data.get("accuracy", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history_data.get("loss", []), "o-", color="#d32f2f", label="Training Loss")
    if "val_loss" in history_data and any(history_data["val_loss"]):
        ax1.plot(epochs, history_data.get("val_loss", []), "s--", color="#f57c00", label="Validation Loss")
    ax1.set_title("Model Loss Across Epochs", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Categorical Crossentropy", fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history_data.get("accuracy", []), "o-", color="#1976d2", label="Training Accuracy")
    if "val_accuracy" in history_data and any(history_data["val_accuracy"]):
        ax2.plot(epochs, history_data.get("val_accuracy", []), "s--", color="#00796b", label="Validation Accuracy")
    ax2.set_title("Model Accuracy Across Epochs", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_pulmovision_pipeline(
    data_dir: Union[str, Path] = DATA_DIR,
    epochs_stage1: int = DEFAULT_EPOCHS,
    epochs_stage2: int = DEFAULT_FINE_TUNE_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    enable_fine_tuning: bool = False,
    progress_hook: Optional[Callable[[int, int, dict], None]] = None,
) -> Dict[str, Any]:
    """Orchestrates end-to-end model training with DenseNet121 transfer learning.

    Returns:
        Dictionary containing training summary, best metrics, and paths.
    """
    set_seed()
    data_path = Path(data_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save class mapping
    save_json(CLASS_TO_IDX, CLASS_INDICES_PATH)

    # 1. Load Datasets
    train_ds, val_ds, test_ds, train_counts = load_dataset_splits(
        data_dir=data_path,
        batch_size=batch_size,
        use_augmentation=True,
    )

    # Compute balanced class weights
    class_weights = compute_class_weights(train_counts)

    # Evaluation dataset for validation callbacks (fallback to test if val has < 10 samples)
    eval_ds = val_ds if val_ds is not None else test_ds

    # 2. Build Model
    model = build_pulmovision_model(
        weights="imagenet",
        learning_rate=learning_rate,
    )

    # Callbacks
    monitor_metric = "val_loss" if eval_ds is not None else "loss"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor=monitor_metric,
        save_best_only=True,
        mode="min",
        verbose=1,
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )
    lr_reducer_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    total_epochs = epochs_stage1 + (epochs_stage2 if enable_fine_tuning else 0)
    progress_cb = StreamlitProgressCallback(total_epochs=total_epochs, progress_hook=progress_hook)

    callbacks_stage1 = [checkpoint_cb, early_stopping_cb, lr_reducer_cb, progress_cb]

    # 3. Stage 1 Training: Feature Extraction (Backbone Frozen)
    logger.info(f"--- Starting Stage 1 Transfer Learning ({epochs_stage1} epochs) ---")
    history_s1 = model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=epochs_stage1,
        class_weight=class_weights,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    combined_history = {k: [float(v) for v in vals] for k, vals in history_s1.history.items()}

    # 4. Stage 2 Optional Fine-Tuning
    if enable_fine_tuning and epochs_stage2 > 0:
        logger.info(f"--- Starting Stage 2 Fine-Tuning ({epochs_stage2} epochs) ---")
        model = prepare_for_finetuning(model, learning_rate=DEFAULT_FINE_TUNE_LR)

        history_s2 = model.fit(
            train_ds,
            validation_data=eval_ds,
            epochs=epochs_stage1 + epochs_stage2,
            initial_epoch=len(history_s1.epoch),
            class_weight=class_weights,
            callbacks=callbacks_stage1,
            verbose=1,
        )

        for k, vals in history_s2.history.items():
            if k in combined_history:
                combined_history[k].extend([float(v) for v in vals])
            else:
                combined_history[k] = [float(v) for v in vals]

    # 5. Save History & Curves
    save_json(combined_history, TRAINING_HISTORY_PATH)
    curves_path = OUTPUTS_DIR / "training_curves.png"
    plot_and_save_training_curves(combined_history, curves_path)

    # 6. Save Model Config
    config_data = {
        "architecture": "DenseNet121",
        "input_shape": [224, 224, 3],
        "classes": list(CLASS_TO_IDX.keys()),
        "trained_timestamp": datetime.now().isoformat(),
        "epochs_completed": len(combined_history.get("loss", [])),
        "final_train_loss": round(combined_history["loss"][-1], 4) if combined_history.get("loss") else None,
        "final_train_acc": round(combined_history["accuracy"][-1], 4) if combined_history.get("accuracy") else None,
        "final_val_loss": round(combined_history["val_loss"][-1], 4) if combined_history.get("val_loss") else None,
        "final_val_acc": round(combined_history["val_accuracy"][-1], 4) if combined_history.get("val_accuracy") else None,
        "fine_tuning_enabled": enable_fine_tuning,
        "class_weights": class_weights,
    }
    save_json(config_data, MODEL_CONFIG_PATH)

    logger.info("Training pipeline successfully finished!")
    return {
        "status": "SUCCESS",
        "model_path": str(BEST_MODEL_PATH),
        "history": combined_history,
        "config": config_data,
        "curves_plot": str(curves_path),
    }
