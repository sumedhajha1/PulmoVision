"""PulmoVision AI - Monte Carlo Dropout Uncertainty Estimation Module

Implements Bayesian approximation via Monte Carlo Dropout (Gal & Ghahramani, 2016).
Computes stochastic forward passes, predictive mean probabilities, predictive entropy,
and predictive variance without altering deterministic layers.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from config import DEFAULT_MC_SAMPLES
from src.utils import logger


def run_monte_carlo_dropout(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    n_samples: int = DEFAULT_MC_SAMPLES,
) -> Dict[str, any]:
    """Performs Monte Carlo Dropout sampling over multiple stochastic forward passes.

    Args:
        model: Trained tf.keras.Model containing PermanentDropout or Dropout layers.
        img_tensor: Preprocessed image tensor with shape (1, 224, 224, 3)
        n_samples: Number of stochastic passes (T), default 20.

    Returns:
        Dictionary containing:
            - stochastic_predictions: Array of shape (T, num_classes)
            - mean_probabilities: Mean probability for each class across T passes
            - std_deviations: Standard deviation for each class across T passes
            - predictive_variance: Variance for each class across T passes
            - predictive_entropy: Raw Shannon predictive entropy in bits (base 2)
            - normalized_entropy: Entropy normalized to [0.0, 1.0] by log2(C)
            - credible_intervals_95: 95% empirical credible interval per class [2.5%, 97.5%]
            - n_samples: Number of passes evaluated
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2 for uncertainty estimation, got {n_samples}")

    # Activate mc_dropout specifically on PermanentDropout layers
    mc_layers = [layer for layer in model.layers if hasattr(layer, "mc_active")]
    for layer in mc_layers:
        layer.mc_active = True

    stochastic_preds = []

    try:
        # If model has no PermanentDropout, fallback to calling model(..., training=True)
        use_training_flag = len(mc_layers) == 0

        tensor_input = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

        for _ in range(n_samples):
            if use_training_flag:
                # Stochastic forward pass with training=True
                preds = model(tensor_input, training=True)
            else:
                # Stochastic forward pass with PermanentDropout active, BatchNorm in inference mode
                preds = model(tensor_input, training=False)
            stochastic_preds.append(preds.numpy()[0])

    finally:
        # Always restore deterministic state
        for layer in mc_layers:
            layer.mc_active = False

    preds_arr = np.array(stochastic_preds, dtype=np.float32)  # Shape: (T, num_classes)

    # 1. Predictive Mean: bar{p}_c = (1/T) sum_{t=1}^T p_c^{(t)}
    mean_probs = np.mean(preds_arr, axis=0)

    # 2. Predictive Variance & Standard Deviation
    var_probs = np.var(preds_arr, axis=0)
    std_probs = np.std(preds_arr, axis=0)

    # 3. Shannon Predictive Entropy: H = - sum_c bar{p}_c * log2(bar{p}_c + eps)
    eps = 1e-12
    entropy_bits = float(-np.sum(mean_probs * np.log2(mean_probs + eps)))

    # Normalized Entropy: H_norm = H / log2(num_classes)
    num_classes = len(mean_probs)
    max_entropy = np.log2(num_classes) if num_classes > 1 else 1.0
    normalized_entropy = float(np.clip(entropy_bits / max_entropy, 0.0, 1.0))

    # 4. 95% Empirical Credible Intervals (2.5th and 97.5th percentiles)
    p2_5 = np.percentile(preds_arr, 2.5, axis=0)
    p97_5 = np.percentile(preds_arr, 97.5, axis=0)
    credible_intervals = {
        cls_idx: (float(p2_5[cls_idx]), float(p97_5[cls_idx])) for cls_idx in range(num_classes)
    }

    return {
        "stochastic_predictions": preds_arr.tolist(),
        "mean_probabilities": mean_probs.tolist(),
        "std_deviations": std_probs.tolist(),
        "predictive_variance": var_probs.tolist(),
        "predictive_entropy": round(entropy_bits, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "credible_intervals_95": credible_intervals,
        "n_samples": n_samples,
    }
