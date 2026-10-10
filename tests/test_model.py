"""Unit tests for PulmoVision AI - Model Architecture"""

import numpy as np
import tensorflow as tf
import pytest

from src.model import build_pulmovision_model, PermanentDropout


def test_build_model_shape_and_softmax():
    # Build model without downloading weights for fast hermetic testing
    model = build_pulmovision_model(weights=None)
    assert model.output_shape == (None, 2)

    # Pass dummy batch
    dummy_input = np.random.randn(2, 224, 224, 3).astype(np.float32)
    preds = model(dummy_input, training=False).numpy()

    assert preds.shape == (2, 2)
    # Check that probabilities sum to 1.0 (Softmax guarantee)
    np.testing.assert_allclose(np.sum(preds, axis=1), [1.0, 1.0], atol=1e-5)
    # Check that probabilities are bounded in [0, 1]
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_permanent_dropout_layer():
    layer = PermanentDropout(rate=0.5)
    inputs = tf.ones((10, 10))

    # Deterministic inference mode (mc_active = False)
    out_eval = layer(inputs, training=False)
    np.testing.assert_array_equal(out_eval.numpy(), inputs.numpy())

    # Monte Carlo active mode (mc_active = True)
    layer.mc_active = True
    out_mc = layer(inputs, training=False)
    # With 50% dropout, some values must be 0 and surviving values scaled by 2
    assert np.any(out_mc.numpy() == 0.0)
