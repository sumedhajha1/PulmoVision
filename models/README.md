# PulmoVision AI - Models Directory

This directory stores trained deep learning model weights and metadata:

- `best_model.keras`: Checkpointed best model weights from Stage 1 / Stage 2 training.
- `class_indices.json`: JSON dictionary mapping class labels to numeric indices (`{"NORMAL": 0, "PNEUMONIA": 1}`).
- `model_config.json`: Model hyperparameters, input dimensions, backbone architecture, and training timestamps.
- `training_history.json`: Epoch-wise loss, accuracy, val_loss, and val_accuracy history.
