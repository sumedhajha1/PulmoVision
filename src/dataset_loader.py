"""PulmoVision AI - Dataset Management and Loader Module

Discovers dataset structure, validates image integrity, calculates
class imbalances, and builds high-performance tf.data pipelines.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from config import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    RANDOM_SEED,
    SUPPORTED_EXTENSIONS,
)
from src.preprocessing import create_augmentation_layer
from src.utils import logger


def inspect_dataset_directory(data_dir: Union[str, Path] = DATA_DIR) -> Dict[str, any]:
    """Inspects the dataset directory, validates splits and class counts,

    and identifies any corrupted or unreadable images.
    """
    data_path = Path(data_dir)
    report = {
        "dataset_path": str(data_path),
        "exists": data_path.exists(),
        "splits": {},
        "total_images": 0,
        "corrupted_images": [],
        "class_names_found": set(),
        "is_valid": False,
        "messages": [],
    }

    if not data_path.exists():
        report["messages"].append(f"Dataset path does not exist: {data_path}")
        return report

    expected_splits = ["train", "val", "test"]
    found_splits = [d.name for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for split in found_splits:
        split_path = data_path / split
        report["splits"][split] = {}
        class_dirs = [d for d in split_path.iterdir() if d.is_dir() and not d.name.startswith(".")]

        for cls_dir in class_dirs:
            cls_name = cls_dir.name.upper()
            report["class_names_found"].add(cls_name)
            images = [
                f
                for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
            ]
            report["splits"][split][cls_name] = len(images)
            report["total_images"] += len(images)

    # Check for canonical labels
    report["class_names_found"] = sorted(list(report["class_names_found"]))
    has_train = "train" in report["splits"]
    has_classes = any("NORMAL" in s and "PNEUMONIA" in s for s in report["splits"].values())

    if has_train and has_classes and report["total_images"] > 0:
        report["is_valid"] = True
        report["messages"].append("Valid dataset structure with canonical NORMAL and PNEUMONIA classes.")
    else:
        report["messages"].append("Dataset missing required 'train' split or expected class directories.")

    return report


def list_supported_image_files(
    directory: Union[str, Path], limit: Optional[int] = None
) -> List[Path]:
    """Return existing image files in a directory, newest UI work excluded."""
    image_dir = Path(directory)
    if not image_dir.exists() or not image_dir.is_dir():
        return []

    paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith(".")
        ),
        key=lambda path: path.name.lower(),
    )
    return paths if limit is None else paths[:limit]


def scan_for_corrupted_images(data_dir: Union[str, Path] = DATA_DIR) -> List[str]:
    """Scans all images in the dataset to verify they can be decoded by PIL."""
    data_path = Path(data_dir)
    corrupted = []
    for ext in SUPPORTED_EXTENSIONS:
        for img_path in data_path.rglob(f"*{ext}"):
            if img_path.name.startswith("."):
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                corrupted.append(str(img_path))
    return corrupted


def compute_class_weights(train_split_counts: Dict[str, int]) -> Dict[int, float]:
    """Computes balanced class weights using inverse class frequency.

    Formula: weight_c = N / (C * N_c)
    """
    total = sum(train_split_counts.values())
    n_classes = len(train_split_counts)
    weights = {}

    for cls_name, count in train_split_counts.items():
        cls_idx = CLASS_TO_IDX.get(cls_name.upper(), 0)
        if count > 0:
            weights[cls_idx] = float(total / (n_classes * count))
        else:
            weights[cls_idx] = 1.0

    logger.info(f"Computed balanced class weights: {weights}")
    return weights


def load_dataset_splits(
    data_dir: Union[str, Path] = DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
    use_augmentation: bool = True,
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], Dict[str, int]]:
    """Loads tf.data datasets for train, val, and test splits with DenseNet preprocessing.

    Returns:
        (train_ds, val_ds, test_ds, train_class_counts)
    """
    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"

    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_path}")

    logger.info(f"Loading training data from {train_path} (batch size: {batch_size})...")

    # Load raw datasets with categorical labels (one-hot [NORMAL, PNEUMONIA])
    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    # Count class samples in train set for class weights
    train_counts = {}
    for cls in CLASS_NAMES:
        cls_dir = train_path / cls
        train_counts[cls] = (
            len([f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])
            if cls_dir.exists()
            else 0
        )

    # Optional validation and test sets
    val_ds = None
    if val_path.exists() and any(val_path.iterdir()):
        val_raw = tf.keras.utils.image_dataset_from_directory(
            val_path,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
        )
        val_ds = val_raw.map(lambda x, y: (densenet_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE
        )

    test_ds = None
    if test_path.exists() and any(test_path.iterdir()):
        test_raw = tf.keras.utils.image_dataset_from_directory(
            test_path,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
        )
        test_ds = test_raw.map(
            lambda x, y: (densenet_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

    # Augmentation and preprocessing for train
    aug_layer = create_augmentation_layer() if use_augmentation else None

    def preprocess_train(x, y):
        if aug_layer:
            x = aug_layer(x, training=True)
        return densenet_preprocess(x), y

    train_ds = train_raw.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, train_counts
