"""Unit tests for PulmoVision AI - Preprocessing Module"""

import io
from PIL import Image
import numpy as np
import pytest

from src.preprocessing import (
    load_and_preprocess_image,
    validate_image_file,
    create_augmentation_layer,
)


def test_validate_image_valid_pil():
    # Create valid synthetic RGB image in memory
    buf = io.BytesIO()
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img.save(buf, format="JPEG")
    buf.seek(0)

    is_valid, err = validate_image_file(buf)
    assert is_valid is True
    assert err == ""


def test_validate_image_corrupted_data():
    buf = io.BytesIO(b"Not an image file byte stream")
    is_valid, err = validate_image_file(buf)
    assert is_valid is False
    assert "Corrupted or unreadable" in err


def test_load_and_preprocess_image_shape_and_dtype():
    # Create synthetic grayscale image (common in X-rays)
    buf = io.BytesIO()
    img = Image.new("L", (300, 400), color=100)
    img.save(buf, format="PNG")
    buf.seek(0)

    tensor, pil_img = load_and_preprocess_image(buf)

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert isinstance(pil_img, Image.Image)
    assert pil_img.mode == "RGB"


def test_augmentation_layer():
    aug = create_augmentation_layer()
    dummy = np.random.uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)
    output = aug(dummy, training=True)
    assert output.shape == (2, 224, 224, 3)
