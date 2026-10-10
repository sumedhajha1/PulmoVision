"""PulmoVision AI - Central Configuration Module

Defines system-wide constants, paths, hyperparameter defaults,
class mapping definitions, and academic/medical disclaimers.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "chest_xray"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
TESTS_DIR = BASE_DIR / "tests"

# Specific File Paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"
EVALUATION_METRICS_PATH = OUTPUTS_DIR / "evaluation_metrics.json"

# Image Specifications for DenseNet121
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Class Label Definitions
# Standard clinical binary mapping: NORMAL=0 (Negative), PNEUMONIA=1 (Positive)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}

# Training Hyperparameter Defaults
DEFAULT_BATCH_SIZE = 16  # Optimized for CPU / typical memory
DEFAULT_EPOCHS = 10
DEFAULT_FINE_TUNE_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_FINE_TUNE_LR = 1e-5
DEFAULT_DROPOUT_RATE = 0.3
RANDOM_SEED = 42

# Explainable AI & Uncertainty Estimation
GRADCAM_TARGET_LAYER = "conv5_block16_2_conv"  # DenseNet121 final convolutional layer
DEFAULT_MC_SAMPLES = 20  # Number of stochastic forward passes for Monte Carlo Dropout

# Mandatory Medical Disclaimer
MEDICAL_DISCLAIMER = (
    "This application is an academic AI research project. Its predictions, confidence estimates, "
    "uncertainty metrics, and visual explanations are experimental and must not be used as a substitute "
    "for professional medical evaluation. This system is not clinically validated. Consult a qualified "
    "healthcare professional for medical assessment."
)
