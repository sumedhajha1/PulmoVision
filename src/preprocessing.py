"""PulmoVision AI - Image Preprocessing Module

Provides functions for loading, validating, resizing, and normalizing
chest X-ray images, ensuring identical preprocessing for training and inference.
"""

from io import BytesIO
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
    INPUT_SHAPE,
    SUPPORTED_EXTENSIONS,
)
from src.utils import logger


def validate_image_file(
    file_or_path: Union[str, Path, BytesIO, bytes], max_size_mb: float = 25.0
) -> Tuple[bool, str]:
    """Validates an image file or byte stream before processing.

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    try:
        if isinstance(file_or_path, (str, Path)):
            p = Path(file_or_path)
            if not p.exists():
                return False, f"File does not exist: {p}"
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return (
                    False,
                    f"Unsupported extension '{p.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                )
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"File size ({size_mb:.1f} MB) exceeds maximum limit of {max_size_mb} MB"
            with Image.open(p) as img:
                img.verify()
        else:
            # Bytes or BytesIO
            if isinstance(file_or_path, bytes):
                stream = BytesIO(file_or_path)
            else:
                stream = file_or_path
                stream.seek(0)
            size_mb = len(stream.getvalue() if hasattr(stream, "getvalue") else stream.read()) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"Upload size ({size_mb:.1f} MB) exceeds maximum allowed {max_size_mb} MB"
            stream.seek(0)
            with Image.open(stream) as img:
                img.verify()
        return True, ""
    except Exception as e:
        return False, f"Corrupted or unreadable image: {str(e)}"


def load_and_preprocess_image(
    file_or_path: Union[str, Path, BytesIO, bytes],
    target_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
) -> Tuple[np.ndarray, Image.Image]:
    """Loads an image safely, converts it to 3-channel RGB, resizes it,

    and applies DenseNet121 normalization.

    Returns:
        Tuple of:
            - preprocessed_tensor: np.ndarray with shape (1, 224, 224, 3), float32
            - original_pil_image: PIL Image object (RGB format, original resolution)
    """
    if isinstance(file_or_path, (str, Path)):
        img = Image.open(file_or_path)
    elif isinstance(file_or_path, bytes):
        img = Image.open(BytesIO(file_or_path))
    else:
        file_or_path.seek(0)
        img = Image.open(file_or_path)

    # Convert to RGB (chest X-rays are often single-channel grayscale or RGBA)
    img_rgb = img.convert("RGB")

    # Resize with high-quality Lanczos resampling
    resized_img = img_rgb.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to numpy float32 array
    img_array = np.array(resized_img, dtype=np.float32)

    # Expand batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # Apply DenseNet121 standard preprocessing (ImageNet zero-centering & standard deviation scaling)
    preprocessed_batch = densenet_preprocess(img_batch.copy())

    return preprocessed_batch, img_rgb


def create_augmentation_layer() -> tf.keras.Sequential:
    """Creates a gentle data augmentation pipeline suitable for chest X-rays.

    Only applies small geometric perturbations so clinically significant
    findings (e.g. consolidations, infiltrates) are preserved.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.04, fill_mode="nearest"),
            tf.keras.layers.RandomFlip("horizontal"),  # Standard in CXR modeling
        ],
        name="cxr_augmentation",
    )
