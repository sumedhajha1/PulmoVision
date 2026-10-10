"""PulmoVision AI - Explainable AI (Grad-CAM) Module

Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to compute class-specific visual explanations from the final
convolutional layers of DenseNet121.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from config import GRADCAM_TARGET_LAYER
from src.utils import logger


def find_target_conv_layer(model: tf.keras.Model, target_name: Optional[str] = None) -> tf.keras.layers.Layer:
    """Finds the target convolutional layer in the model.

    If target_name is not found, dynamically finds the deepest Conv2D layer.
    """
    if target_name:
        for layer in model.layers:
            if layer.name == target_name:
                return layer

    # Search for deepest Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            logger.info(f"Using deepest detected Conv2D layer: {layer.name}")
            return layer
        # If it's a nested submodel (e.g. densenet121)
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    logger.info(f"Using deepest detected Conv2D layer in submodel: {sub_layer.name}")
                    return sub_layer

    raise ValueError(f"No suitable convolutional layer found in model.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    class_idx: Optional[int] = None,
    target_layer_name: str = GRADCAM_TARGET_LAYER,
) -> np.ndarray:
    """Computes the 2D Grad-CAM activation heatmap for a given input tensor and class.

    Args:
        model: Trained tf.keras.Model
        img_tensor: Preprocessed image tensor with shape (1, 224, 224, 3)
        class_idx: Numeric index of the target class (0=NORMAL, 1=PNEUMONIA).
                   If None, uses the model's highest predicted class score.
        target_layer_name: Name of the convolutional layer to compute gradients from.

    Returns:
        heatmap: 2D numpy float32 array normalized to [0, 1] with shape (H, W)
    """
    try:
        target_layer = find_target_conv_layer(model, target_layer_name)
    except Exception as e:
        logger.warning(f"Target layer {target_layer_name} lookup failed: {e}. Falling back to default search.")
        target_layer = find_target_conv_layer(model, None)

    # Build intermediate model that maps inputs -> [conv_layer_output, predictions]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    tensor_input = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(tensor_input)
        conv_outputs, predictions = grad_model(tensor_input, training=False)

        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))

        # Target class activation score
        loss = predictions[:, class_idx]

    # Calculate gradients of class score with respect to feature maps
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients could not be computed for the selected layer.")

    # Channel-wise mean of gradients (importance weights alpha_k^c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of forward activation maps
    conv_outputs_val = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs_val), axis=-1)

    # Apply ReLU: only positive contributions toward the class are retained
    cam = tf.maximum(cam, 0.0)

    # Normalize between 0.0 and 1.0
    cam_numpy = cam.numpy()
    max_val = np.max(cam_numpy)
    if max_val > 1e-8:
        heatmap = cam_numpy / max_val
    else:
        heatmap = np.zeros_like(cam_numpy)

    return heatmap.astype(np.float32)


def overlay_gradcam(
    original_img: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[Image.Image, Image.Image]:
    """Overlays the Grad-CAM activation heatmap onto the original chest X-ray.

    Args:
        original_img: PIL Image or RGB numpy array of the original image
        heatmap: 2D float32 array normalized between 0 and 1
        alpha: Blending ratio for heatmap (0 = only original, 1 = only heatmap)
        colormap: OpenCV colormap constant (default: COLORMAP_JET)

    Returns:
        Tuple of (colored_heatmap_pil: Image.Image, overlay_pil: Image.Image)
    """
    if isinstance(original_img, Image.Image):
        orig_np = np.array(original_img.convert("RGB"), dtype=np.uint8)
    else:
        orig_np = np.array(original_img, dtype=np.uint8)

    h, w = orig_np.shape[:2]

    # Resize heatmap to match original image dimensions
    resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    # Convert to 8-bit unsigned integer (0-255)
    heatmap_uint8 = np.uint8(255 * resized_heatmap)

    # Apply colormap (cv2 outputs BGR)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    overlay_rgb = np.uint8(alpha * colored_rgb + (1.0 - alpha) * orig_np)

    return Image.fromarray(colored_rgb), Image.fromarray(overlay_rgb)
