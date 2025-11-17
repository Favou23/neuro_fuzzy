#!/usr/bin/env python
"""
Test script for Random Forest predictions
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from randomforest.rf_predict import predict_with_rf, predict_dataset_rf
from randomforest.rf_evaluate import generate_evaluation_report

def test_single_prediction():
    """Test single sample prediction"""
    print("\n" + "="*70)
    print("TEST 1: Single Sample Prediction")
    print("="*70)
    
    sample = {
        "CH4": 500,
        "C2H4": 500,
        "C2H2": 50,
        "H2": 10,
        "C2H6": 5,
        "CO": 0
    }
    
    results = predict_with_rf(sample, method="all")
    
    print("\nRandom Forest Predictions:")
    for method, result in results.items():
        if result['status'] == 'success':
            print(f"  {method.upper():10} → {result['prediction']:10} (confidence: {result['confidence']:.4f})")
        else:
            print(f"  {method.upper():10} → ERROR: {result['error']}")


def test_batch_prediction():
    """Test batch predictions on CSV file"""
    print("\n" + "="*70)
    print("TEST 2: Batch Prediction on test_model.csv")
    print("="*70)
    
    try:
        df_results = predict_dataset_rf(
            file_path="test_model.csv",
            method="all",
            output_file="test_model_rf_predictions.csv"
        )
        
        print("\nFirst 5 predictions:")
        columns_to_show = [col for col in df_results.columns if '_rf_prediction' in col]
        print(df_results[columns_to_show].head())
        
        print(f"\n✓ Results saved to: test_model_rf_predictions.csv")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_evaluation():
    """Test evaluation report generation"""
    print("\n" + "="*70)
    print("TEST 3: Model Evaluation Report")
    print("="*70)
    
    report = generate_evaluation_report()
    if report is not None:
        print("\nModels Found:")
        print(report.to_string())
    else:
        print("No models found for evaluation")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Random Forest Module - Test Suite")
    print("#"*70)
    
    # Test 1: Single prediction
    test_single_prediction()
    
    # Test 2: Batch prediction
    test_batch_prediction()
    
    # Test 3: Evaluation
    test_evaluation()
    
    print("\n" + "#"*70)
    print("# All Tests Completed!")
    print("#"*70 + "\n")
