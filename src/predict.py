"""PulmoVision AI - Prediction and Inference Pipeline Module

Integrates image validation, DenseNet121 inference, Monte Carlo Dropout
uncertainty quantification, and Grad-CAM visual explainability.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
import tensorflow as tf

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEFAULT_MC_SAMPLES,
    IDX_TO_CLASS,
)
from src.gradcam import generate_gradcam_heatmap, overlay_gradcam
from src.model import load_trained_model
from src.preprocessing import load_and_preprocess_image, validate_image_file
from src.uncertainty import run_monte_carlo_dropout
from src.utils import logger


def run_inference_pipeline(
    image_input: Union[str, Path, BytesIO, bytes, Image.Image],
    model: Optional[tf.keras.Model] = None,
    model_path: Union[str, Path] = BEST_MODEL_PATH,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    generate_cam: bool = True,
) -> Dict[str, Any]:
    """Runs the complete diagnostic inference pipeline for an input chest X-ray.

    Returns:
        Dictionary containing prediction results, probabilities, confidence,
        uncertainty metrics, Grad-CAM visualizations, and original images.
    """
    # 1. Validation
    if isinstance(image_input, Image.Image):
        buf = BytesIO()
        image_input.save(buf, format="PNG")
        buf.seek(0)
        validation_target = buf
    else:
        validation_target = image_input

    is_valid, error_msg = validate_image_file(validation_target)
    if not is_valid:
        raise ValueError(f"Image validation failed: {error_msg}")

    # 2. Preprocessing
    if isinstance(image_input, Image.Image):
        # Convert PIL directly
        img_rgb = image_input.convert("RGB")
        resized = img_rgb.resize((224, 224), Image.Resampling.LANCZOS)
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

        img_batch = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)
        img_tensor = densenet_preprocess(img_batch.copy())
        original_pil = img_rgb
    else:
        img_tensor, original_pil = load_and_preprocess_image(image_input)

    # 3. Model Loading
    if model is None:
        model = load_trained_model(model_path)

    # 4. Deterministic Model Prediction
    raw_preds = model(img_tensor, training=False).numpy()[0]
    normal_prob = float(raw_preds[0])
    pneumonia_prob = float(raw_preds[1])

    predicted_idx = int(np.argmax(raw_preds))
    predicted_label = IDX_TO_CLASS[predicted_idx]
    confidence_estimate = float(np.max(raw_preds))

    # 5. Monte Carlo Dropout Uncertainty Estimation
    uncertainty_results = run_monte_carlo_dropout(
        model=model,
        img_tensor=img_tensor,
        n_samples=mc_samples,
    )

    # 6. Grad-CAM Explainability
    heatmap_pil = None
    overlay_pil = None
    raw_heatmap = None

    if generate_cam:
        try:
            raw_heatmap = generate_gradcam_heatmap(
                model=model,
                img_tensor=img_tensor,
                class_idx=predicted_idx,
            )
            heatmap_pil, overlay_pil = overlay_gradcam(
                original_img=original_pil,
                heatmap=raw_heatmap,
            )
        except Exception as e:
            logger.error(f"Grad-CAM generation error: {e}")

    # 7. Formulate cautious academic/clinical classification label
    # Distinguish high uncertainty case (e.g. normalized entropy > 0.85)
    normalized_entropy = uncertainty_results["normalized_entropy"]
    if normalized_entropy > 0.85:
        triage_label = "Uncertain result — expert review required"
        triage_category = "UNCERTAIN"
    elif predicted_label == "PNEUMONIA":
        triage_label = "Model classification: Pneumonia suspected"
        triage_category = "PNEUMONIA"
    else:
        triage_label = "Model classification: Normal"
        triage_category = "NORMAL"

    return {
        "predicted_label": predicted_label,
        "predicted_idx": predicted_idx,
        "triage_label": triage_label,
        "triage_category": triage_category,
        "confidence_estimate": confidence_estimate,
        "normal_probability": normal_prob,
        "pneumonia_probability": pneumonia_prob,
        "probabilities": [normal_prob, pneumonia_prob],
        "uncertainty": uncertainty_results,
        "original_image": original_pil,
        "gradcam_heatmap": heatmap_pil,
        "gradcam_overlay": overlay_pil,
        "raw_heatmap": raw_heatmap,
    }
