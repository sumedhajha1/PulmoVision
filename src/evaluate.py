"""PulmoVision AI - Model Evaluation and Performance Metrics Module

Calculates clinical and statistical performance metrics on the held-out
test dataset, including Sensitivity, Specificity, ROC-AUC, and Confusion Matrix.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf

from config import (
    CLASS_NAMES,
    DATA_DIR,
    EVALUATION_METRICS_PATH,
    OUTPUTS_DIR,
)
from src.dataset_loader import load_dataset_splits
from src.utils import logger, save_json


def evaluate_model_on_test_set(
    model: tf.keras.Model,
    data_dir: Union[str, Path] = DATA_DIR,
    outputs_dir: Union[str, Path] = OUTPUTS_DIR,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Runs rigorous model evaluation on the test set and exports metrics and figures.

    Returns:
        Structured evaluation dictionary containing all calculated metrics.
    """
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    _, _, test_ds, _ = load_dataset_splits(
        data_dir=data_dir,
        batch_size=batch_size,
        use_augmentation=False,
    )

    if test_ds is None:
        raise ValueError(f"No test dataset found under {data_dir}/test.")

    logger.info("Executing evaluation on test dataset...")
    y_true_list = []
    y_prob_list = []

    for images, labels in test_ds:
        preds = model(images, training=False).numpy()
        # Labels are one-hot encoded: [NORMAL, PNEUMONIA]
        true_indices = np.argmax(labels.numpy(), axis=1)
        # Class 1 is PNEUMONIA (positive class)
        pneumonia_probs = preds[:, 1]

        y_true_list.extend(true_indices)
        y_prob_list.extend(pneumonia_probs)

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Confusion Matrix
    # Class 0: NORMAL (Negative), Class 1: PNEUMONIA (Positive)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 2. Clinical and Statistical Metrics
    acc = float(accuracy_score(y_true, y_pred))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Specificity
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    # 3. Classification Report
    cr_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Save Classification Report as CSV
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_csv_path = outputs_path / "classification_report.csv"
    cr_df.to_csv(cr_csv_path)

    # 4. Generate & Save Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (Pred)", "Pneumonia (Pred)"],
        yticklabels=["Normal (True)", "Pneumonia (True)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("PulmoVision AI - Confusion Matrix (Test Set)", fontsize=13, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    cm_path = outputs_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 5. Generate & Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#0066cc", lw=2.5, label=f"DenseNet121 ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Chance Level")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = outputs_path / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # 6. Generate & Save Precision-Recall Curve
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_pts, precision_pts))
    plt.figure(figsize=(7, 6))
    plt.plot(recall_pts, precision_pts, color="#009688", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title("Precision-Recall (PR) Curve", fontsize=13, pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = outputs_path / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    metrics_result = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_test_samples": int(len(y_true)),
        "confusion_matrix": {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "sensitivity_recall": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "negative_predictive_value": round(npv, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
        },
        "per_class": cr_dict,
        "plots": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
            "precision_recall_curve": str(pr_path),
        },
    }

    # Save to JSON
    save_json(metrics_result, EVALUATION_METRICS_PATH)
    logger.info(f"Evaluation complete! Accuracy: {acc:.2%}, Sensitivity: {sens:.2%}, Specificity: {spec:.2%}")
    return metrics_result
