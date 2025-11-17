# Random Forest Integration Guide

## Overview

A new, independent **Random Forest module** has been created as a standalone directory that maintains clean separation from existing ANFIS code. The module can be used independently or in conjunction with the existing ANFIS models.

## What Was Created

### Directory Structure
```
fuzzy_logic/
├── randomforest/                    # NEW: Independent RF module
│   ├── __init__.py                  # Module interface
│   ├── rf_train.py                  # Training script
│   ├── rf_predict.py                # Prediction module
│   ├── rf_evaluate.py               # Evaluation utilities
│   └── README.md                    # Detailed documentation
├── (existing ANFIS files unchanged)
├── data_ingestion.py                # (used by RF via import)
└── results/                         # Shared results directory
```

## Key Features

### 1. **Completely Independent**
- No modifications to existing ANFIS files
- Self-contained directory with its own modules
- All imports from parent directory are explicit

### 2. **Three Core Modules**

#### `rf_train.py` - Model Training
```python
# Train RF models for all methods
from randomforest.rf_train import train_random_forest_models

datasets = {
    "duval": ("path/to/duval_data.csv", "FAULT"),
    "rogers": ("path/to/rogers_data.csv", "FAULT"),
    "drm": ("path/to/drm_data.csv", "FAULT"),
}

results = train_random_forest_models(datasets)
```

**Outputs:**
- `results/{method}_randomforest/model.pkl` - Trained model
- `results/{method}_randomforest/confusion_matrix.png` - Confusion matrix
- `results/{method}_randomforest/feature_importance.png` - Feature importance
- `encoders/{method}_label_encoder.pkl` - Label encoder (shared with ANFIS)

#### `rf_predict.py` - Predictions
```python
# Single sample prediction
from randomforest.rf_predict import predict_with_rf

sample = {
    "CH4": 500, "C2H4": 500, "C2H2": 50,
    "H2": 10, "C2H6": 5, "CO": 0
}

results = predict_with_rf(sample, method="all")
# Returns: {'duval': {...}, 'rogers': {...}, 'drm': {...}}
```

```python
# Batch predictions
from randomforest.rf_predict import predict_dataset_rf

df_results = predict_dataset_rf(
    "test_data.csv",
    method="all",
    output_file="predictions.csv"
)
```

#### `rf_evaluate.py` - Evaluation
```python
# Generate evaluation report
from randomforest.rf_evaluate import generate_evaluation_report

report = generate_evaluation_report()

# Or custom evaluation
from randomforest.rf_evaluate import RFEvaluator

evaluator = RFEvaluator("duval")
evaluator.load_model()
metrics = evaluator.compute_metrics(y_true, y_pred)
evaluator.print_report(y_true, y_pred)
evaluator.plot_confusion_matrix(y_true, y_pred)
```

## Usage Workflows

### Workflow 1: Train RF Models Only

```bash
cd fuzzy_logic
python randomforest/rf_train.py
```

**What happens:**
1. Loads datasets from data generators
2. Applies SMOTE for class balancing
3. Scales features with StandardScaler
4. Trains RF with hyperparameter tuning (GridSearchCV)
5. Saves models and visualizations
6. Prints accuracy and classification reports

### Workflow 2: Make Predictions with RF

```python
from randomforest.rf_predict import predict_with_rf

# Single sample
sample = {
    "CH4": 500, "C2H4": 500, "C2H2": 50,
    "H2": 10, "C2H6": 5, "CO": 0
}

results = predict_with_rf(sample, method="duval")
print(results)
# Output: {
#   'duval': {
#       'prediction': 'T2',
#       'confidence': 0.95,
#       'status': 'success'
#   }
# }
```

### Workflow 3: Batch Prediction on New Data

```python
from randomforest.rf_predict import predict_dataset_rf

# Predict on all samples in CSV
df = predict_dataset_rf(
    file_path="test_model.csv",
    method="all",
    output_file="test_model_rf_predictions.csv"
)

print(df[['duval_rf_prediction', 'duval_rf_confidence']].head())
```

### Workflow 4: Compare RF with ANFIS

```python
from randomforest.rf_predict import predict_with_rf, predict_dataset_rf
from predict_fault import predict_dataset  # ANFIS predictions

# Get RF predictions
df_rf = predict_dataset_rf("test_data.csv", method="duval")

# Get ANFIS predictions
df_anfis = predict_dataset("test_data.csv", method="duval")

# Compare
comparison = pd.DataFrame({
    'true_label': df['label'],
    'rf_prediction': df_rf['duval_rf_prediction'],
    'anfis_prediction': df_anfis['duval_prediction']
})

# Calculate agreement
agreement = (comparison['rf_prediction'] == comparison['anfis_prediction']).sum()
print(f"Agreement: {agreement}/{len(comparison)} ({100*agreement/len(comparison):.1f}%)")
```

## File Organization

### New Files (No existing files modified)
- `randomforest/__init__.py` - Module interface
- `randomforest/rf_train.py` - Training (standalone)
- `randomforest/rf_predict.py` - Prediction (standalone)
- `randomforest/rf_evaluate.py` - Evaluation (standalone)
- `randomforest/README.md` - Complete RF documentation
- `RF_INTEGRATION_GUIDE.md` - This file

### Shared Resources
- `results/` - Models saved here (shared with ANFIS)
- `encoders/` - Label encoders (shared with ANFIS)
- `data_ingestion.py` - Features import (no modification)
- `fuzzy_logic/` - Imported for parent directory access

## Data Flow

```
Training Data
    ↓
rf_train.py
    ├─→ load_and_prepare_data() [from data_ingestion.py]
    ├─→ Train RF (SMOTE + Scaler + GridSearchCV)
    └─→ Save to results/{method}_randomforest/
    
Test/New Data
    ↓
rf_predict.py
    ├─→ Load model from results/{method}_randomforest/
    ├─→ transform_features() [from data_ingestion.py]
    └─→ predict_with_rf()
    
Evaluation
    ↓
rf_evaluate.py
    ├─→ Load model and encoder
    ├─→ compute_metrics()
    └─→ Generate reports/visualizations
```

## Integration Points

### With Existing ANFIS System
1. **Shared Encoders**: Both systems use same label encoders
2. **Shared Results Directory**: Both save models in `results/`
3. **Shared Data Processing**: Both use `data_ingestion.py` functions

### No Modifications Required To:
- `unified_training.py` - ANFIS training (unchanged)
- `predict_fault.py` - ANFIS prediction (unchanged)
- `data_ingestion.py` - Only read, no modifications
- Any other existing files

## Example: Complete Workflow

```python
#!/usr/bin/env python
"""
Complete workflow: Train RF, predict, and evaluate
"""

import os
import sys

# Add parent to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 1: Train Random Forest models
print("="*70)
print("Step 1: Training Random Forest Models")
print("="*70)

from randomforest.rf_train import train_random_forest_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
datasets = {
    "duval": (os.path.join(BASE_DIR, "duval_data_generator/datasets/duval_polygon_dataset.csv"), "FAULT"),
    "rogers": (os.path.join(BASE_DIR, "rogers_data_generator/datasets/rogers_rule_dataset.csv"), "FAULT"),
    "drm": (os.path.join(BASE_DIR, "drm_data_generator/datasets/drm_rule_dataset.csv"), "FAULT"),
}

train_results = train_random_forest_models(datasets)

# Step 2: Make predictions on test data
print("\n" + "="*70)
print("Step 2: Making Predictions")
print("="*70)

from randomforest.rf_predict import predict_dataset_rf

test_df = predict_dataset_rf(
    file_path="test_model.csv",
    method="all",
    output_file="test_model_rf_predictions.csv"
)

print(f"\nPredictions saved to: test_model_rf_predictions.csv")
print(test_df[['duval_rf_prediction', 'rogers_rf_prediction', 'drm_rf_prediction']].head())

# Step 3: Evaluate models
print("\n" + "="*70)
print("Step 3: Generating Evaluation Report")
print("="*70)

from randomforest.rf_evaluate import generate_evaluation_report

report = generate_evaluation_report()
if report is not None:
    print(report)

print("\n✓ Complete workflow finished!")
```

## Command Line Usage

### Train models
```bash
python randomforest/rf_train.py
```

### Make predictions
```bash
python randomforest/rf_predict.py
```

### Evaluate
```bash
python randomforest/rf_evaluate.py
```

## Performance Considerations

### Training Time
- Depends on dataset size and hyperparameter grid
- GridSearchCV with 3-fold CV performs extensive search
- Typically completes in minutes

### Prediction Time
- Fast (RF is faster than ANFIS typically)
- Batch predictions on CSV: seconds to minutes depending on file size

### Memory Usage
- Models are lightweight (stored as .pkl files)
- SMOTE during training requires additional memory
- Batch predictions use pandas DataFrames

## Troubleshooting

### ImportError when running rf_train.py
**Solution:** Run from project root or ensure parent directory is in Python path
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### FileNotFoundError for datasets
**Solution:** Verify data generator CSV files exist at expected paths
```bash
ls fuzzy_logic/duval_data_generator/datasets/duval_polygon_dataset.csv
ls fuzzy_logic/rogers_data_generator/datasets/rogers_rule_dataset.csv
ls fuzzy_logic/drm_data_generator/datasets/drm_rule_dataset.csv
```

### Feature mismatch error
**Solution:** Ensure input features match training features (CH4, C2H4, C2H2, H2, C2H6, CO)

### Model not found error
**Solution:** Train models first
```bash
python randomforest/rf_train.py
```

## Next Steps

1. **Train models**: `python randomforest/rf_train.py`
2. **Test predictions**: Use `rf_predict.py` with test data
3. **Evaluate**: Check metrics with `rf_evaluate.py`
4. **Compare**: Compare RF results with ANFIS in `predict_fault.py`
5. **Integrate**: Use both models for ensemble predictions

## Architecture Diagram

```
Project Root (fuzzy_logic/)
│
├── Data Layer
│   ├── data_ingestion.py
│   ├── duval_data_generator/
│   ├── rogers_data_generator/
│   └── drm_data_generator/
│
├── ANFIS Layer (Existing)
│   ├── unified_training.py
│   ├── predict_fault.py
│   └── iec_rule_based.py
│
├── Random Forest Layer (NEW)
│   ├── randomforest/
│   │   ├── rf_train.py
│   │   ├── rf_predict.py
│   │   ├── rf_evaluate.py
│   │   └── __init__.py
│
└── Shared Resources
    ├── results/ (Models)
    └── encoders/ (Label encoders)
```

## Support

For detailed usage of individual modules, see:
- Training: `randomforest/README.md` → Training section
- Prediction: `randomforest/README.md` → Making Predictions section
- Evaluation: `randomforest/README.md` → Model Evaluation section

---

**Status:** ✓ Implementation Complete

All files are independent, no existing code was modified, and the RF module is ready for use!
