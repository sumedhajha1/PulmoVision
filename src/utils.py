"""PulmoVision AI - Utility Functions Module

Provides logging helpers, seed initialization, JSON serialization,
and safe path operations.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from config import RANDOM_SEED


def setup_logger(name: str = "PulmoVision") -> logging.Logger:
    """Configures and returns a consistent logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger("PulmoVision")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility across random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    logger.info(f"Initialized random seed to {seed}")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Safely saves a dictionary to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON data to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Safely loads a dictionary from a JSON file. Returns empty dict if missing."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"JSON file not found: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_bytes(size_bytes: int) -> str:
    """Formats raw byte counts into human-readable strings."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"
