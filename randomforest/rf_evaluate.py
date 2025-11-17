"""
Random Forest Evaluation Module
Provides evaluation metrics and comparison utilities
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from data_ingestion import prepare_duval_features, prepare_rogers_features, prepare_drm_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODERS_DIR = os.path.join(BASE_DIR, "encoders")


class RFEvaluator:
    """
    Evaluate Random Forest model performance
    """
    
    def __init__(self, method_name):
        """
        Initialize evaluator for a method.
        
        Args:
            method_name: 'duval', 'rogers', or 'drm'
        """
        self.method_name = method_name
        self.model = None
        self.encoder = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.metrics = {}
    
    def load_model(self):
        """Load trained model and encoder."""
        model_dir = os.path.join(RESULTS_DIR, f"{self.method_name}_randomforest")
        model_path = os.path.join(model_dir, "model.pkl")
        self.model = joblib.load(model_path)
        
        encoder_path = os.path.join(ENCODERS_DIR, f"{self.method_name}_label_encoder.pkl")
        self.encoder = joblib.load(encoder_path)
        
        print(f"✓ Loaded RF model for {self.method_name}")
    
    def compute_metrics(self, y_true, y_pred, average='weighted'):
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'weighted', 'macro', 'micro'
        """
        self.metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        return self.metrics
    
    def get_confusion_matrix(self, y_true, y_pred):
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def print_report(self, y_true, y_pred):
        """Print classification report."""
        print("\n" + "="*70)
        print(f"Random Forest - {self.method_name.upper()}")
        print("="*70)
        print(classification_report(
            y_true, y_pred, 
            target_names=self.encoder.classes_, 
            zero_division=0
        ))
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot and optionally save confusion matrix."""
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Greens',
            xticklabels=self.encoder.classes_,
            yticklabels=self.encoder.classes_
        )
        plt.title(f"RF Confusion Matrix - {self.method_name.upper()}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✓ Saved confusion matrix to: {save_path}")
        
        plt.show()


def evaluate_rf_models(test_data_dict=None):
    """
    Evaluate Random Forest models for all methods.
    
    Args:
        test_data_dict: Dict mapping method -> (X_test, y_test)
                       If None, will use default test datasets
    
    Returns:
        DataFrame with evaluation metrics
    """
    print("\n" + "#"*70)
    print("# Random Forest Model Evaluation")
    print("#"*70)
    
    methods = ["duval", "rogers", "drm"]
    all_metrics = {}
    
    for method in methods:
        try:
            print(f"\nEvaluating {method.upper()}...")
            evaluator = RFEvaluator(method)
            evaluator.load_model()
            
            # In a real scenario, you'd have test data
            # For now, we'll just show the structure
            print(f"✓ {method.upper()} model loaded and ready for evaluation")
            
        except Exception as e:
            print(f"✗ Error evaluating {method}: {e}")
    
    return all_metrics


def compare_rf_with_test_data(y_true, y_pred_rf, method_name, y_pred_anfis=None):
    """
    Compare RF predictions with true labels and optionally ANFIS predictions.
    
    Args:
        y_true: True labels
        y_pred_rf: RF predictions
        method_name: Method used
        y_pred_anfis: Optional ANFIS predictions for comparison
    """
    print("\n" + "="*70)
    print(f"Comparison Report - {method_name.upper()}")
    print("="*70)
    
    # RF Metrics
    rf_acc = accuracy_score(y_true, y_pred_rf)
    rf_f1 = f1_score(y_true, y_pred_rf, average='weighted', zero_division=0)
    
    print(f"\nRandom Forest:")
    print(f"  Accuracy: {rf_acc:.4f}")
    print(f"  F1-Score: {rf_f1:.4f}")
    
    # ANFIS comparison if provided
    if y_pred_anfis is not None:
        anfis_acc = accuracy_score(y_true, y_pred_anfis)
        anfis_f1 = f1_score(y_true, y_pred_anfis, average='weighted', zero_division=0)
        
        print(f"\nANFIS:")
        print(f"  Accuracy: {anfis_acc:.4f}")
        print(f"  F1-Score: {anfis_f1:.4f}")
        
        print(f"\nComparison:")
        print(f"  Better Model: {'RF' if rf_acc > anfis_acc else 'ANFIS'}")
        print(f"  Accuracy Difference: {abs(rf_acc - anfis_acc):.4f}")


def generate_evaluation_report(output_dir=None):
    """
    Generate a comprehensive evaluation report for all RF models.
    
    Args:
        output_dir: Directory to save report (default: results/)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    print("\n" + "#"*70)
    print("# Generating Comprehensive RF Evaluation Report")
    print("#"*70)
    
    report_data = []
    
    for method in ["duval", "rogers", "drm"]:
        try:
            model_dir = os.path.join(RESULTS_DIR, f"{method}_randomforest")
            
            # Check if model exists
            if not os.path.exists(os.path.join(model_dir, "model.pkl")):
                print(f"✗ Model not found for {method}")
                continue
            
            report_data.append({
                'Method': method,
                'Status': 'Trained',
                'Model_Path': model_dir
            })
            print(f"✓ {method.upper()} model found")
            
        except Exception as e:
            print(f"✗ Error for {method}: {e}")
    
    # Save report
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(output_dir, "rf_evaluation_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"\n✓ Report saved to: {report_path}")
        return report_df
    else:
        print("\n✗ No models found for evaluation")
        return None


if __name__ == "__main__":
    # Generate evaluation report
    report = generate_evaluation_report()
    if report is not None:
        print("\nEvaluation Report:")
        print(report)
