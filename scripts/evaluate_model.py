"""PulmoVision AI - CLI Script: Evaluate Model

Usage:
    python scripts/evaluate_model.py --data_dir data/chest_xray
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BEST_MODEL_PATH, DATA_DIR, OUTPUTS_DIR
from src.evaluate import evaluate_model_on_test_set
from src.model import load_trained_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained DenseNet121 Model on Chest X-ray Test Set")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Path to dataset root")
    parser.add_argument("--model_path", type=str, default=str(BEST_MODEL_PATH), help="Path to trained model .keras")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")

    args = parser.parse_args()

    print("=" * 70)
    print(" PulmoVision AI - Held-out Test Set Model Evaluation")
    print("=" * 70)
    print(f"Dataset Path:    {args.data_dir}")
    print(f"Model Path:      {args.model_path}\n")

    if not Path(args.model_path).exists():
        print(f"[ERROR] Trained model file not found at: {args.model_path}")
        print("Please train the model first using: python scripts/train_model.py")
        sys.exit(1)

    print("Loading model checkpoint...")
    model = load_trained_model(args.model_path)

    results = evaluate_model_on_test_set(
        model=model,
        data_dir=args.data_dir,
        outputs_dir=OUTPUTS_DIR,
        batch_size=args.batch_size,
    )

    m = results["metrics"]
    cm = results["confusion_matrix"]

    print("\n" + "=" * 70)
    print(" Performance Metrics Summary:")
    print("=" * 70)
    print(f"  Accuracy:                      {m['accuracy']:.2%}")
    print(f"  Sensitivity / Recall:          {m['sensitivity_recall']:.2%}")
    print(f"  Specificity:                   {m['specificity']:.2%}")
    print(f"  Precision (PPV):               {m['precision']:.2%}")
    print(f"  Negative Predictive Value:     {m['negative_predictive_value']:.2%}")
    print(f"  F1-Score:                      {m['f1_score']:.4f}")
    print(f"  ROC-AUC:                       {m['roc_auc']:.4f}")
    print(f"  Precision-Recall AUC:          {m['pr_auc']:.4f}")
    print("-" * 70)
    print(" Confusion Matrix Counts:")
    print(f"  True Positives (TP):           {cm['true_positive']}  (Pneumonia correctly identified)")
    print(f"  True Negatives (TN):           {cm['true_negative']}  (Normal correctly identified)")
    print(f"  False Positives (FP):          {cm['false_positive']}  (Type I error: Normal misclassified)")
    print(f"  False Negatives (FN):          {cm['false_negative']}  (Type II error: Pneumonia missed)")
    print("-" * 70)
    print(f" Generated Plots saved to:       {OUTPUTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
