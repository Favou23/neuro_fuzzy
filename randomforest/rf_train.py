"""
Random Forest Model Training Module
Trains Random Forest models for each method (Duval, Rogers, DRM)
Imports dependencies from parent directory
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from data_ingestion import load_and_prepare_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODERS_DIR = os.path.join(BASE_DIR, "encoders")

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ENCODERS_DIR, exist_ok=True)


def train_random_forest_for_method(method_name, file_path, label_col):
    """
    Train Random Forest model for a specific method.
    
    Args:
        method_name: 'duval', 'rogers', or 'drm'
        file_path: Path to dataset CSV
        label_col: Label column name
        
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*70}")
    print(f"Training Random Forest for {method_name.upper()} Method")
    print(f"{'='*70}")
    
    try:
        # Load and prepare data
        X, y, le = load_and_prepare_data(
            file_path, 
            label_col, 
            method_name,
            augment_iec=True,
            iec_synth_per_class=300,
            iec_ppm_range=(10, 5000),
            iec_jitter=0.08,
            seed=42
        )
        
        print("\nLabel distribution (encoded):")
        counts = pd.Series(y).map(lambda iv: le.inverse_transform([iv])[0]).value_counts()
        print(counts)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Pipeline with SMOTE and RandomForest
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        }
        
        # Grid search
        print("\nPerforming Grid Search...")
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            scoring='accuracy', 
            verbose=1, 
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        print(f"\nBest Parameters: {grid.best_params_}")
        best_pipeline = grid.best_estimator_
        
        # Predictions
        y_pred_train = best_pipeline.predict(X_train)
        y_pred_test = best_pipeline.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:     {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0))
        
        # Save model
        model_dir = os.path.join(RESULTS_DIR, f"{method_name}_randomforest")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(best_pipeline, model_path)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save encoder
        encoder_path = os.path.join(ENCODERS_DIR, f"{method_name}_label_encoder.pkl")
        joblib.dump(le, encoder_path)
        print(f"✓ Encoder saved to: {encoder_path}")
        print(f"  Classes: {list(le.classes_)}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Greens',
            xticklabels=le.classes_, 
            yticklabels=le.classes_
        )
        plt.title(f"Random Forest - {method_name.upper()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        plot_path = os.path.join(model_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to: {plot_path}")
        
        # Feature importance
        clf = best_pipeline.named_steps['clf']
        importances = clf.feature_importances_
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.title(f"Random Forest - {method_name.upper()} Feature Importance")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        importance_path = os.path.join(model_dir, "feature_importance.png")
        plt.savefig(importance_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Feature importance saved to: {importance_path}")
        
        return {
            'method': method_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'best_params': grid.best_params_,
            'encoder': le,
            'model_path': model_path
        }
        
    except Exception as e:
        print(f"\n✗ Error training Random Forest for {method_name}: {e}")
        traceback.print_exc()
        return None


def train_random_forest_models(datasets):
    """
    Train Random Forest models for multiple methods.
    
    Args:
        datasets: Dict mapping method_name -> (file_path, label_col)
                 Example: {
                     'duval': ('path/to/duval_dataset.csv', 'FAULT'),
                     'rogers': ('path/to/rogers_dataset.csv', 'FAULT'),
                     'drm': ('path/to/drm_dataset.csv', 'FAULT')
                 }
    
    Returns:
        List of training results
    """
    print(f"\n{'#'*70}")
    print("# Random Forest Model Training Pipeline")
    print(f"{'#'*70}")
    
    results = []
    for method_name, (file_path, label_col) in datasets.items():
        result = train_random_forest_for_method(method_name, file_path, label_col)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    for result in results:
        print(f"{result['method'].upper()}: Test Accuracy = {result['test_accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    datasets = {
        "duval": (os.path.join(BASE_DIR, "duval_data_generator", "datasets", "duval_polygon_dataset.csv"), "FAULT"),
        "rogers": (os.path.join(BASE_DIR, "rogers_data_generator", "datasets", "rogers_rule_dataset.csv"), "FAULT"),
        "drm": (os.path.join(BASE_DIR, "drm_data_generator", "datasets", "drm_rule_dataset.csv"), "FAULT"),
    }
    
    results = train_random_forest_models(datasets)
