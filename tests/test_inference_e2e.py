"""End-to-End Test for PulmoVision AI Inference, Grad-CAM, and MC Dropout"""

from pathlib import Path
import pytest
from src.predict import run_inference_pipeline
from config import BEST_MODEL_PATH, DATA_DIR


def test_end_to_end_inference():
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Model checkpoint not found")

    test_dir = DATA_DIR / "test" / "PNEUMONIA"
    samples = list(test_dir.glob("*.jpeg"))
    if not samples:
        pytest.skip("No test samples found")

    sample_img = samples[0]
    results = run_inference_pipeline(sample_img, mc_samples=10, generate_cam=True)

    assert results["predicted_label"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= results["confidence_estimate"] <= 1.0
    assert 0.0 <= results["normal_probability"] <= 1.0
    assert 0.0 <= results["pneumonia_probability"] <= 1.0
    assert pytest.approx(results["normal_probability"] + results["pneumonia_probability"], rel=1e-3) == 1.0

    # Uncertainty checks
    unc = results["uncertainty"]
    assert unc["n_samples"] == 10
    assert 0.0 <= unc["normalized_entropy"] <= 1.0

    # Grad-CAM checks
    assert results["gradcam_overlay"] is not None
    assert results["gradcam_heatmap"] is not None
    assert results["raw_heatmap"] is not None
    assert results["raw_heatmap"].shape == (7, 7) or results["raw_heatmap"].ndim == 2
