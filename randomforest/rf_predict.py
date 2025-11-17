"""
Random Forest Prediction Module
Provides prediction capabilities for Random Forest models
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from data_ingestion import prepare_duval_features, prepare_rogers_features, prepare_drm_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODERS_DIR = os.path.join(BASE_DIR, "encoders")


def load_rf_model_and_encoder(method_name):
    """
    Load trained Random Forest model and label encoder.
    
    Args:
        method_name: 'duval', 'rogers', or 'drm'
        
    Returns:
        Tuple of (model, encoder)
        
    Raises:
        FileNotFoundError: If model or encoder not found
    """
    model_dir = os.path.join(RESULTS_DIR, f"{method_name}_randomforest")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Random Forest model not found at {model_dir}")
    
    model_path = os.path.normpath(os.path.join(model_dir, "model.pkl"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    
    encoder_path = os.path.join(ENCODERS_DIR, f"{method_name}_label_encoder.pkl")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")
    
    encoder = joblib.load(encoder_path)
    
    return model, encoder


def transform_features(sample_dict, method):
    """
    Transform raw gas measurements to model features.
    
    Args:
        sample_dict: Dictionary with gas measurements
        method: 'duval', 'rogers', or 'drm'
        
    Returns:
        DataFrame with transformed features
    """
    df = pd.DataFrame([sample_dict])
    
    if method.lower() == "duval":
        X = prepare_duval_features(df, assume_ppm=True)
    elif method.lower() == "rogers":
        X = prepare_rogers_features(df)
    elif method.lower() == "drm":
        X = prepare_drm_features(df)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X


def predict_single_rf(sample_dict, method, model, encoder, verbose=True):
    """
    Predict fault for a single sample using Random Forest.
    
    Args:
        sample_dict: Dictionary with gas measurements
        method: 'duval', 'rogers', or 'drm'
        model: Trained Random Forest model
        encoder: Label encoder
        verbose: Print debug info
        
    Returns:
        Tuple of (predicted_label, confidence_score)
    """
    if verbose:
        print(f"\n[RF] Predicting with method: {method.upper()}")
    
    # Transform features
    X = transform_features(sample_dict, method)
    if verbose:
        print(f"[RF] Features: {X.values}")
    
    # Get prediction
    X_array = X.to_numpy(dtype=np.float32)
    y_pred = model.predict(X_array)
    pred_idx = int(y_pred[0])
    
    # Get probability/confidence
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X_array)
            confidence = float(np.max(proba))
        except:
            confidence = None
    else:
        confidence = None
    
    # Decode label
    try:
        pred_label = encoder.inverse_transform([pred_idx])[0]
    except:
        pred_label = str(pred_idx)
    
    if verbose:
        print(f"[RF] Prediction: {pred_label}")
        if confidence:
            print(f"[RF] Confidence: {confidence:.4f}")
    
    return pred_label, confidence


def predict_with_rf(sample_dict, method="all"):
    """
    Predict using Random Forest for one or more methods.
    
    Args:
        sample_dict: Dictionary with gas measurements
        method: 'duval', 'rogers', 'drm', or 'all'
        
    Returns:
        Dict mapping method -> (prediction, confidence)
    """
    methods = [method] if method != "all" else ["duval", "rogers", "drm"]
    results = {}
    
    for m in methods:
        try:
            model, encoder = load_rf_model_and_encoder(m)
            pred_label, confidence = predict_single_rf(sample_dict, m, model, encoder, verbose=True)
            results[m] = {
                'prediction': pred_label,
                'confidence': confidence,
                'status': 'success'
            }
        except Exception as e:
            print(f"[RF] Error predicting for {m}: {e}")
            results[m] = {
                'prediction': None,
                'confidence': None,
                'status': 'error',
                'error': str(e)
            }
    
    return results


def predict_dataset_rf(file_path, method="all", output_file=None):
    """
    Predict faults for all samples in a CSV/Excel file.
    
    Args:
        file_path: Path to CSV or Excel file
        method: 'duval', 'rogers', 'drm', or 'all'
        output_file: Path to save predictions (default: same name + '_rf_predictions.csv')
        
    Returns:
        DataFrame with predictions
    """
    # Load dataset
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"\n[RF] Predicting for {len(df)} samples from {file_path}")
    
    methods = [method] if method != "all" else ["duval", "rogers", "drm"]
    
    for m in methods:
        print(f"\n[RF] Processing method: {m.upper()}")
        try:
            model, encoder = load_rf_model_and_encoder(m)
            
            # Predict for each row
            predictions = []
            confidences = []
            
            for idx, row in df.iterrows():
                try:
                    sample = row.to_dict()
                    pred_label, confidence = predict_single_rf(
                        sample, m, model, encoder, verbose=False
                    )
                    predictions.append(pred_label)
                    confidences.append(confidence if confidence else 0.0)
                except Exception as e:
                    print(f"[RF] Error on row {idx}: {e}")
                    predictions.append(None)
                    confidences.append(0.0)
            
            df[f"{m}_rf_prediction"] = predictions
            df[f"{m}_rf_confidence"] = confidences
            
        except Exception as e:
            print(f"[RF] Error loading model for {m}: {e}")
            df[f"{m}_rf_prediction"] = None
            df[f"{m}_rf_confidence"] = None
    
    # Save results
    if output_file is None:
        base_name = os.path.splitext(file_path)[0]
        output_file = f"{base_name}_rf_predictions.csv"
    
    df.to_csv(output_file, index=False)
    print(f"\n[RF] Predictions saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    # Example usage
    sample = {
        "CH4": 500, 
        "C2H4": 500, 
        "C2H2": 50, 
        "H2": 10, 
        "C2H6": 5, 
        "CO": 0
    }
    
    results = predict_with_rf(sample, method="all")
    print("\n" + "="*50)
    print("Prediction Results:")
    for method, result in results.items():
        print(f"{method.upper()}: {result['prediction']} (confidence: {result['confidence']})")
