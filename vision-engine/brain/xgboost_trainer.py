"""
XGBoost Classifier Trainer
===========================

PURPOSE:
Train a machine learning classifier to predict suspicious behavior from features.
This REPLACES the rule-based scorer with a learned model.

WHY XGBOOST?
- Fast to train (1-5 minutes on CPU)
- Works well with small datasets (200+ samples)
- Interpretable (SHAP values show feature importance)
- Industry standard for tabular data

LEARNING CONCEPTS:
- Supervised Learning: Model learns from labeled examples
- Train/Test Split: Separate data for training and evaluation
- Evaluation Metrics: Measuring model performance
- Feature Importance: Which features matter most?
- Model Persistence: Saving/loading trained models

WHEN TO USE THIS:
After you collect 200+ labeled behavior samples.
Until then, use rule-based scorer.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import os


class XGBoostTrainer:
    """
    Trains XGBoost classifier on labeled behavior data.

    Example usage:
        # Load your labeled CSV
        df = pd.read_csv('labeled_behaviors.csv')

        # Train model
        trainer = XGBoostTrainer()
        model, metrics = trainer.train(df)

        # Save model
        trainer.save_model(model, 'my_behavior_classifier.json')

        # Use model
        prediction, confidence = trainer.predict(model, new_features)
    """

    def __init__(self):
        """Initialize trainer with default hyperparameters."""
        self.feature_cols = [
            'mean_torso_angle',
            'max_torso_angle',
            'std_torso_angle',
            'mean_speed',
            'max_speed',
            'min_dist_to_edge',
            'dwell_time_near_edge',
            'direction_changes'
        ]

        print("✓ XGBoostTrainer initialized")
        print(f"  Features: {self.feature_cols}")


    def train(self, df: pd.DataFrame, test_size: float = 0.25) -> tuple:
        """
        Train XGBoost classifier on labeled data.

        Args:
            df: DataFrame with features + 'label' column (0=normal, 1=suspicious)
            test_size: Fraction of data to use for testing

        Returns:
            (model, metrics_dict)

        LEARNING: This is where the "magic" happens!
        1. Split data into train/test
        2. Train model on training data
        3. Evaluate on test data (data model hasn't seen)
        4. Report metrics
        """
        print("\n=== Training XGBoost Classifier ===\n")

        # Step 1: Extract features and labels
        print("Step 1: Preparing data...")

        X = df[self.feature_cols]
        y = df['label']  # 0=normal, 1=suspicious

        print(f"  Total samples: {len(df)}")
        print(f"  Normal: {(y==0).sum()}")
        print(f"  Suspicious: {(y==1).sum()}")

        # Check for class imbalance
        suspicious_ratio = y.sum() / len(y)
        if suspicious_ratio < 0.1 or suspicious_ratio > 0.9:
            print(f"  ⚠ WARNING: Imbalanced dataset ({suspicious_ratio:.1%} suspicious)")
            print("    Consider collecting more minority class samples")

        # Step 2: Train/test split
        print(f"\nStep 2: Splitting data ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y  # Maintain class ratio in both splits
        )

        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")

        # Step 3: Train XGBoost model
        print("\nStep 3: Training XGBoost...")

        # LEARNING: These are hyperparameters (settings for the algorithm)
        # You can tune these to improve performance
        model = xgb.XGBClassifier(
            max_depth=5,              # Max tree depth (prevent overfitting)
            n_estimators=100,         # Number of trees
            learning_rate=0.1,        # How fast to learn
            eval_metric='logloss',    # Metric to optimize
            random_state=42,
            use_label_encoder=False
        )

        # Fit model (this is the actual training!)
        model.fit(X_train, y_train)

        print("  ✓ Training complete!")

        # Step 4: Evaluate model
        print("\nStep 4: Evaluating model...")

        metrics = self._evaluate(model, X_train, X_test, y_train, y_test)

        return model, metrics


    def _evaluate(self, model, X_train, X_test, y_train, y_test) -> dict:
        """
        Evaluate trained model and compute metrics.

        LEARNING: Evaluation tells us how good our model is.
        Always evaluate on TEST data (data model hasn't seen during training).
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities (for confidence)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Training metrics
        train_acc = (y_train_pred == y_train).mean()
        print(f"\n  Training Accuracy: {train_acc:.3f}")

        # Test metrics (THIS IS WHAT MATTERS!)
        test_acc = (y_test_pred == y_test).mean()
        print(f"  Test Accuracy: {test_acc:.3f}")

        # Classification report (precision, recall, F1)
        print(f"\n  Classification Report (Test Set):")
        report = classification_report(y_test, y_test_pred, output_dict=True)
        print(classification_report(y_test, y_test_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\n  Confusion Matrix:")
        print(f"                Predicted")
        print(f"                Normal  Suspicious")
        print(f"  Actual Normal    {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"  Actual Suspicious {cm[1,0]:4d}    {cm[1,1]:4d}")

        # ROC-AUC score
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print(f"\n  ROC-AUC Score: {roc_auc:.3f}")
        print(f"    (0.5=random, 1.0=perfect, >0.85=good)")

        # Feature importance
        print(f"\n  Top 5 Most Important Features:")
        importances = model.feature_importances_
        feature_importance = sorted(
            zip(self.feature_cols, importances),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (feat, imp) in enumerate(feature_importance[:5], 1):
            print(f"    {i}. {feat}: {imp:.3f}")

        # Package metrics
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'feature_importance': dict(zip(self.feature_cols, importances))
        }

        return metrics


    def save_model(self, model, filepath: str):
        """
        Save trained model to disk.

        LEARNING: Save your trained model so you can use it later
        without retraining!

        Args:
            model: Trained XGBoost model
            filepath: Where to save (e.g., 'my_model.json')
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        model.save_model(filepath)
        print(f"\n✓ Model saved to: {filepath}")


    def load_model(self, filepath: str):
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded XGBoost model
        """
        model = xgb.XGBClassifier()
        model.load_model(filepath)
        print(f"✓ Model loaded from: {filepath}")
        return model


    def predict(self, model, features: dict) -> tuple:
        """
        Predict using trained model.

        Args:
            model: Trained XGBoost model
            features: Dictionary with feature values

        Returns:
            (prediction, confidence)
            - prediction: 0 (normal) or 1 (suspicious)
            - confidence: 0.0-1.0

        LEARNING: This replaces the rule-based scorer!
        Instead of hand-coded rules, model uses learned patterns.
        """
        # Convert features dict to DataFrame row
        feature_values = [features.get(col, 0) for col in self.feature_cols]
        X = np.array(feature_values).reshape(1, -1)

        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Confidence is the probability of the predicted class
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        return int(prediction), float(confidence)


    def plot_roc_curve(self, model, X_test, y_test, save_path: str = None):
        """
        Plot ROC curve (shows trade-off between true/false positives).

        LEARNING: ROC curve helps you choose operating point.
        - Top-left corner = perfect classifier
        - Diagonal line = random guessing
        """
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Behavior Classifier', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✓ ROC curve saved to: {save_path}")
        else:
            plt.show()


# EDUCATIONAL EXAMPLE
if __name__ == "__main__":
    """
    Run this file to see demo of XGBoost training.

    NOTE: This requires labeled data! Create synthetic data for demo.

    Command: python xgboost_trainer.py
    """
    print("=== XGBoost Trainer Demo ===\n")

    print("Creating synthetic training data for demo...")
    print("(In real use, you'd load your labeled CSV)\n")

    # Generate synthetic data (normally you'd load from CSV)
    np.random.seed(42)

    # Normal behaviors
    normal_samples = pd.DataFrame({
        'mean_torso_angle': np.random.normal(88, 3, 100),
        'max_torso_angle': np.random.normal(92, 4, 100),
        'std_torso_angle': np.random.uniform(1, 4, 100),
        'mean_speed': np.random.normal(80, 20, 100),
        'max_speed': np.random.normal(120, 30, 100),
        'min_dist_to_edge': np.random.normal(180, 40, 100),
        'dwell_time_near_edge': np.random.uniform(0, 2, 100),
        'direction_changes': np.random.poisson(1.5, 100),
        'label': 0
    })

    # Suspicious behaviors
    suspicious_samples = pd.DataFrame({
        'mean_torso_angle': np.random.normal(72, 5, 50),
        'max_torso_angle': np.random.normal(68, 6, 50),
        'std_torso_angle': np.random.uniform(3, 8, 50),
        'mean_speed': np.random.normal(200, 50, 50),
        'max_speed': np.random.normal(350, 60, 50),
        'min_dist_to_edge': np.random.normal(60, 25, 50),
        'dwell_time_near_edge': np.random.uniform(5, 12, 50),
        'direction_changes': np.random.poisson(7, 50),
        'label': 1
    })

    # Combine
    df = pd.concat([normal_samples, suspicious_samples], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    print(f"Generated {len(df)} synthetic samples")
    print(f"  Normal: {(df['label']==0).sum()}")
    print(f"  Suspicious: {(df['label']==1).sum()}")

    # Train model
    trainer = XGBoostTrainer()
    model, metrics = trainer.train(df, test_size=0.3)

    # Save model
    trainer.save_model(model, 'demo_behavior_classifier.json')

    # Test prediction
    print("\n\n=== Testing Predictions ===\n")

    test_features_normal = {
        'mean_torso_angle': 87,
        'max_torso_angle': 91,
        'std_torso_angle': 2.5,
        'mean_speed': 75,
        'max_speed': 110,
        'min_dist_to_edge': 190,
        'dwell_time_near_edge': 1.2,
        'direction_changes': 2
    }

    pred, conf = trainer.predict(model, test_features_normal)
    print(f"Test 1 (Normal behavior):")
    print(f"  Prediction: {'Suspicious' if pred == 1 else 'Normal'}")
    print(f"  Confidence: {conf:.2f} ({conf*100:.0f}%)")

    test_features_suspicious = {
        'mean_torso_angle': 70,
        'max_torso_angle': 65,
        'std_torso_angle': 6.0,
        'mean_speed': 220,
        'max_speed': 380,
        'min_dist_to_edge': 45,
        'dwell_time_near_edge': 9.5,
        'direction_changes': 8
    }

    pred, conf = trainer.predict(model, test_features_suspicious)
    print(f"\nTest 2 (Suspicious behavior):")
    print(f"  Prediction: {'Suspicious' if pred == 1 else 'Normal'}")
    print(f"  Confidence: {conf:.2f} ({conf*100:.0f}%)")

    print("\n\n✓ XGBoost training works!")
    print("\nKEY TAKEAWAYS:")
    print("1. XGBoost learns patterns from YOUR labeled data")
    print("2. You train it from scratch (random initialization → learned patterns)")
    print("3. It outputs predictions + confidence scores (like rule-based, but learned)")
    print("4. You can show judges: 'I trained this classifier on 200+ labeled samples'")
    print("\nFor hackathon:")
    print("  - Collect 200+ labeled samples")
    print("  - Run this trainer")
    print("  - Show judges the training metrics + evaluation plots")
    print("  - Explain: 'My model achieved 85% accuracy with 0.89 ROC-AUC'")
