"""Unit tests for PulmoVision AI - Monte Carlo Dropout Uncertainty Estimation"""

import numpy as np
import pytest

from src.model import build_pulmovision_model
from src.uncertainty import run_monte_carlo_dropout


def test_mc_dropout_uncertainty_metrics():
    model = build_pulmovision_model(weights=None)
    dummy_tensor = np.random.randn(1, 224, 224, 3).astype(np.float32)

    results = run_monte_carlo_dropout(model, dummy_tensor, n_samples=5)

    assert results["n_samples"] == 5
    assert len(results["mean_probabilities"]) == 2
    assert len(results["predictive_variance"]) == 2

    # Check bounds on normalized entropy: [0.0, 1.0]
    assert 0.0 <= results["normalized_entropy"] <= 1.0
    assert results["predictive_entropy"] >= 0.0

    # Probabilities must sum to ~1.0
    assert pytest.approx(sum(results["mean_probabilities"]), rel=1e-3) == 1.0


def test_invalid_n_samples_raises():
    model = build_pulmovision_model(weights=None)
    dummy_tensor = np.random.randn(1, 224, 224, 3).astype(np.float32)

    with pytest.raises(ValueError):
        run_monte_carlo_dropout(model, dummy_tensor, n_samples=1)
