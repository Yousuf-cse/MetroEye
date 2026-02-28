"""
Simple Training Script - Uses Existing XGBoostTrainer
======================================================

Wrapper script to train XGBoost model using your labeled data.

Usage:
    python train_model.py --data labeled_training_data.csv
"""

import pandas as pd
import argparse
import sys
import os

# Add brain directory to path
sys.path.append('brain')
from xgboost_trainer import XGBoostTrainer


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost Model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to labeled CSV file')
    parser.add_argument('--output', type=str, default='behavior_classifier.json',
                       help='Output model filename')

    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.data):
        print(f"✗ File not found: {args.data}")
        return

    print(f"\nLoading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"✓ Loaded {len(df)} samples")

    # Train model
    trainer = XGBoostTrainer()
    model, metrics = trainer.train(df, test_size=0.25)

    # Save model
    trainer.save_model(model, args.output)

    print(f"\n✓ Training complete! Model saved to: {args.output}")
    print(f"  Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
