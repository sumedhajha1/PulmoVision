"""Unit tests for PulmoVision AI - Grad-CAM Explainability"""

import numpy as np
from PIL import Image
import pytest

from src.model import build_pulmovision_model
from src.gradcam import generate_gradcam_heatmap, overlay_gradcam


def test_gradcam_heatmap_generation():
    # Build unweighted model for fast testing
    model = build_pulmovision_model(weights=None)
    dummy_tensor = np.random.randn(1, 224, 224, 3).astype(np.float32)

    heatmap = generate_gradcam_heatmap(model, dummy_tensor, class_idx=1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    # Normalization check: bounds [0, 1]
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0


def test_overlay_gradcam():
    dummy_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    dummy_heatmap = np.random.uniform(0.0, 1.0, (7, 7)).astype(np.float32)

    colored_map, overlay = overlay_gradcam(dummy_img, dummy_heatmap)

    assert isinstance(colored_map, Image.Image)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (224, 224)
