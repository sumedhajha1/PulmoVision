"""PulmoVision AI - Deep Learning Model Module

Defines DenseNet121 architecture with transfer learning, custom
Monte Carlo Dropout support, fine-tuning utilities, and model serialization.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_LEARNING_RATE,
    GRADCAM_TARGET_LAYER,
    INPUT_SHAPE,
)
from src.utils import logger


@tf.keras.utils.register_keras_serializable(package="PulmoVision")
class PermanentDropout(layers.Dropout):
    """Custom Dropout layer for Monte Carlo uncertainty estimation.

    Allows activating dropout independently during inference while
    leaving BatchNormalization layers in deterministic inference mode.
    """

    def __init__(self, rate: float, **kwargs):
        super().__init__(rate, **kwargs)
        self.mc_active = False

    def call(self, inputs, training=None):
        if self.mc_active or (training is True):
            return super().call(inputs, training=True)
        return super().call(inputs, training=False)

    def get_config(self):
        config = super().get_config()
        return config


def build_pulmovision_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = len(CLASS_NAMES),
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    weights: Optional[str] = "imagenet",
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """Constructs the PulmoVision transfer learning model with DenseNet121 backbone.

    Backbone is frozen by default for Stage 1 training.
    """
    logger.info(f"Building DenseNet121 model (weights={weights}, input_shape={input_shape})...")

    # Inputs
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Pretrained DenseNet121 backbone
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
    )
    base_model.trainable = False  # Freeze backbone for Stage 1

    # Feature extraction & classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = PermanentDropout(rate=dropout_rate, name="mc_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PulmoVision_DenseNet121")

    # Compile with Adam and Categorical Crossentropy
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    logger.info(f"Successfully compiled model. Total parameters: {model.count_params():,}")
    return model


def prepare_for_finetuning(
    model: tf.keras.Model,
    learning_rate: float = 1e-5,
    unfreeze_from_layer: str = "conv5_block1_0_bn",
) -> tf.keras.Model:
    """Unfreezes top convolutional layers of DenseNet121 for Stage 2 fine-tuning.

    Uses a significantly lower learning rate to prevent catastrophic forgetting.
    """
    logger.info(f"Configuring model for fine-tuning from layer: {unfreeze_from_layer}...")

    # Find the base model or unfreeze layers directly
    unfreeze = False
    for layer in model.layers:
        if layer.name == unfreeze_from_layer:
            unfreeze = True
        if unfreeze:
            # Keep head BatchNormalization deterministic to maintain stability
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    logger.info(f"Model recompiled for fine-tuning. Trainable parameters: {trainable_params:,}")
    return model


def load_trained_model(
    model_path: Union[str, Path] = BEST_MODEL_PATH,
    compile_model: bool = False,
) -> tf.keras.Model:
    """Safely loads a trained PulmoVision model checkpoint."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at: {path}")

    custom_objects = {"PermanentDropout": PermanentDropout}
    model = tf.keras.models.load_model(
        path,
        custom_objects=custom_objects,
        compile=compile_model,
    )
    logger.info(f"Loaded trained model successfully from: {path}")
    return model
