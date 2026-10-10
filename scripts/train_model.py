"""PulmoVision AI - CLI Script: Train Model

Usage:
    python scripts/train_model.py --data_dir data/chest_xray --epochs 10 --batch_size 16
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE
from src.train import train_pulmovision_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 Transfer Learning Model for PulmoVision AI")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of Stage 1 epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, help="Number of Stage 2 fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--fine_tune", action="store_true", help="Enable Stage 2 fine-tuning")

    args = parser.parse_args()

    print("=" * 70)
    print(" PulmoVision AI - Starting Model Training Pipeline")
    print("=" * 70)
    print(f"Dataset Path:       {args.data_dir}")
    print(f"Stage 1 Epochs:     {args.epochs}")
    print(f"Stage 2 Epochs:     {args.fine_tune_epochs if args.fine_tune else 'Disabled'}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Initial LR:         {args.lr}")
    print("=" * 70 + "\n")

    result = train_pulmovision_pipeline(
        data_dir=args.data_dir,
        epochs_stage1=args.epochs,
        epochs_stage2=args.fine_tune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        enable_fine_tuning=args.fine_tune,
    )

    print("\n" + "=" * 70)
    print(" Training Completed Successfully!")
    print(f" Model Checkpoint: {result['model_path']}")
    print(f" Training Curves:  {result['curves_plot']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
