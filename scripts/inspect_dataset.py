"""PulmoVision AI - CLI Script: Inspect Dataset

Usage:
    python scripts/inspect_dataset.py --data_dir data/chest_xray
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR
from src.dataset_loader import inspect_dataset_directory, scan_for_corrupted_images


def main():
    parser = argparse.ArgumentParser(description="Inspect Chest X-ray Dataset for PulmoVision AI")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_DIR),
        help="Path to the chest_xray dataset directory",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" PulmoVision AI - Dataset Inspection Report")
    print("=" * 70)
    print(f"Target Directory: {args.data_dir}\n")

    report = inspect_dataset_directory(args.data_dir)

    if not report["exists"]:
        print(f"[ERROR] Directory does not exist: {args.data_dir}")
        sys.exit(1)

    print(f"Total Images Found: {report['total_images']}")
    print(f"Identified Classes: {', '.join(report['class_names_found']) if report['class_names_found'] else 'None'}\n")

    print("-" * 50)
    print(f"{'Split':<12} | {'NORMAL':<12} | {'PNEUMONIA':<12} | {'Total':<10}")
    print("-" * 50)

    for split, counts in report["splits"].items():
        norm = counts.get("NORMAL", 0)
        pneu = counts.get("PNEUMONIA", 0)
        tot = sum(counts.values())
        print(f"{split:<12} | {norm:<12} | {pneu:<12} | {tot:<10}")

    print("-" * 50)

    # Class imbalance in training set
    train_counts = report["splits"].get("train", {})
    if train_counts:
        tot_train = sum(train_counts.values())
        norm_pct = (train_counts.get("NORMAL", 0) / tot_train) * 100 if tot_train else 0
        pneu_pct = (train_counts.get("PNEUMONIA", 0) / tot_train) * 100 if tot_train else 0
        print(f"\nClass Distribution in Training Set:")
        print(f"  - NORMAL:    {norm_pct:.1f}% ({train_counts.get('NORMAL', 0)} images)")
        print(f"  - PNEUMONIA: {pneu_pct:.1f}% ({train_counts.get('PNEUMONIA', 0)} images)")
        print(f"  - Imbalance Ratio: {pneu_pct / norm_pct:.2f} : 1" if norm_pct > 0 else "")

    print("\nScanning for unreadable or corrupted files...")
    corrupted = scan_for_corrupted_images(args.data_dir)
    if corrupted:
        print(f"[WARNING] Found {len(corrupted)} unreadable file(s):")
        for c in corrupted[:5]:
            print(f"  - {c}")
    else:
        print("[SUCCESS] All images decoded successfully without corruption.")

    print("\n" + "=" * 70)
    if report["is_valid"]:
        print("[STATUS] Dataset is VALID and ready for training and evaluation.")
    else:
        print("[STATUS] Dataset structure incomplete. Please verify folder organization.")
    print("=" * 70)


if __name__ == "__main__":
    main()
