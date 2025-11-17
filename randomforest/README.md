# Random Forest Module

A modular, independent implementation of Random Forest model training and prediction for transformer fault detection.

## Overview

This module provides standalone implementation for training, prediction, and evaluation of Random Forest models across three diagnostic methods:
- **Duval Triangle Method**
- **Rogers Method**
- **DRM (Domain Reproduction Method)**

## Directory Structure

```
randomforest/
├── __init__.py              # Module initialization
├── rf_train.py              # Training script
├── rf_predict.py            # Prediction module
├── rf_evaluate.py           # Evaluation utilities
└── README.md                # This file
```

## Installation

Ensure parent directory dependencies are installed:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn joblib
```

## Usage

### 1. Training Random Forest Models

**From command line:**
```bash
python randomforest/rf_train.py
```

**Programmatic usage:**
```python
from randomforest.rf_train import train_random_forest_models
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

datasets = {
    "duval": (os.path.join(BASE_DIR, "duval_data_generator/datasets/duval_polygon_dataset.csv"), "FAULT"),
    "rogers": (os.path.join(BASE_DIR, "rogers_data_generator/datasets/rogers_rule_dataset.csv"), "FAULT"),
    "drm": (os.path.join(BASE_DIR, "drm_data_generator/datasets/drm_rule_dataset.csv"), "FAULT"),
}

results = train_random_forest_models(datasets)
```

**Output:**
- Models saved to: `results/{method}_randomforest/model.pkl`
- Encoders saved to: `encoders/{method}_label_encoder.pkl`
- Visualizations: `results/{method}_randomforest/*.png`

### 2. Making Predictions

**Single sample prediction:**
```python
from randomforest.rf_predict import predict_with_rf

sample = {
    "CH4": 500,
    "C2H4": 500,
    "C2H2": 50,
    "H2": 10,
    "C2H6": 5,
    "CO": 0
}

results = predict_with_rf(sample, method="all")
# Or for specific method: method="duval"

print(results)
# Output:
# {
#   'duval': {'prediction': 'T2', 'confidence': 0.95, 'status': 'success'},
#   'rogers': {'prediction': 'T2', 'confidence': 0.92, 'status': 'success'},
#   'drm': {'prediction': 'T2', 'confidence': 0.88, 'status': 'success'}
# }
```

**Batch prediction from CSV/Excel:**
```python
from randomforest.rf_predict import predict_dataset_rf

df_results = predict_dataset_rf(
    file_path="test_data.csv",
    method="all",
    output_file="predictions_with_rf.csv"
)
```

### 3. Model Evaluation

**Generate evaluation report:**
```python
from randomforest.rf_evaluate import generate_evaluation_report

report = generate_evaluation_report()
print(report)
```

**Custom evaluation:**
```python
from randomforest.rf_evaluate import RFEvaluator
from sklearn.metrics import accuracy_score

evaluator = RFEvaluator("duval")
evaluator.load_model()

# Compute metrics on test data
y_true = [...]  # true labels
y_pred = [...]  # predictions

metrics = evaluator.compute_metrics(y_true, y_pred)
print(metrics)

# Print classification report
evaluator.print_report(y_true, y_pred)

# Plot confusion matrix
evaluator.plot_confusion_matrix(y_true, y_pred, save_path="cm.png")
```

## File Descriptions

### `rf_train.py`
Trains Random Forest models for each diagnostic method.

**Key Functions:**
- `train_random_forest_for_method(method_name, file_path, label_col)` - Train single method
- `train_random_forest_models(datasets)` - Train all methods

**Features:**
- SMOTE for class imbalance
- StandardScaler for feature normalization
- GridSearchCV for hyperparameter tuning
- Confusion matrix visualization
- Feature importance plots

### `rf_predict.py`
Provides prediction functionality for new samples.

**Key Functions:**
- `load_rf_model_and_encoder(method_name)` - Load trained model
- `predict_single_rf(sample_dict, method, model, encoder)` - Single sample prediction
- `predict_with_rf(sample_dict, method)` - Predict with one or all methods
- `predict_dataset_rf(file_path, method, output_file)` - Batch predictions

**Features:**
- Automatic feature transformation
- Confidence scoring
- Batch processing
- Support for CSV and Excel files

### `rf_evaluate.py`
Evaluation and comparison utilities.

**Key Classes:**
- `RFEvaluator` - Comprehensive model evaluation

**Key Functions:**
- `evaluate_rf_models(test_data_dict)` - Evaluate all models
- `compare_rf_with_test_data(y_true, y_pred_rf, method_name, y_pred_anfis)` - Compare with ANFIS
- `generate_evaluation_report(output_dir)` - Generate report

**Features:**
- Accuracy, precision, recall, F1-score
- Confusion matrices
- Classification reports
- Comparison with ANFIS

## Output Structure

### Training Outputs
```
results/
├── duval_randomforest/
│   ├── model.pkl                    # Trained model
│   ├── confusion_matrix.png         # Test confusion matrix
│   └── feature_importance.png       # Feature importance plot
├── rogers_randomforest/
│   ├── model.pkl
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── drm_randomforest/
    ├── model.pkl
    ├── confusion_matrix.png
    └── feature_importance.png

encoders/
├── duval_label_encoder.pkl
├── rogers_label_encoder.pkl
└── drm_label_encoder.pkl
```

### Prediction Outputs
```
{original_file}_rf_predictions.csv
Columns:
- Original features
- {method}_rf_prediction: Predicted fault class
- {method}_rf_confidence: Confidence score
```

## Hyperparameters

### Random Forest Grid Search Parameters
- `n_estimators`: [50, 100, 200]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]

### Data Processing
- **SMOTE**: k_neighbors=5
- **Scaler**: StandardScaler
- **Train-Test Split**: 80-20

## Example Workflow

```python
# Step 1: Train models
from randomforest import train_random_forest_models
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
datasets = {
    "duval": (os.path.join(BASE_DIR, "duval_data_generator/datasets/duval_polygon_dataset.csv"), "FAULT"),
    "rogers": (os.path.join(BASE_DIR, "rogers_data_generator/datasets/rogers_rule_dataset.csv"), "FAULT"),
    "drm": (os.path.join(BASE_DIR, "drm_data_generator/datasets/drm_rule_dataset.csv"), "FAULT"),
}

print("Training RF models...")
results = train_random_forest_models(datasets)

# Step 2: Make predictions
from randomforest import predict_with_rf, predict_dataset_rf

print("\nMaking single prediction...")
sample = {"CH4": 500, "C2H4": 500, "C2H2": 50, "H2": 10, "C2H6": 5, "CO": 0}
preds = predict_with_rf(sample, method="all")
print(preds)

print("\nBatch prediction...")
df = predict_dataset_rf("test_data.csv", method="all")
print(df.head())

# Step 3: Evaluate
from randomforest import evaluate_rf_models

print("\nEvaluating models...")
report = evaluate_rf_models()
```

## Dependencies

**Internal:**
- `data_ingestion.py` - Feature preparation functions
- `encoders/` - Label encoders (auto-generated)

**External:**
- scikit-learn >= 1.0
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn
- joblib

## Notes

- All models are trained with 80-20 train-test split
- SMOTE is applied to handle class imbalance
- Features are standardized using StandardScaler
- Models support multiclass classification
- Confidence scores from `predict_proba()` when available
- Full pipeline (SMOTE + Scaler + RF) is saved for consistency

## Troubleshooting

### Model not found error
```
FileNotFoundError: Random Forest model not found
```
**Solution:** Run `python randomforest/rf_train.py` first to train models.

### Import errors
**Solution:** Ensure parent directory is in Python path or run from project root.

### Feature mismatch
**Solution:** Ensure input features match those used during training (CH4, C2H4, C2H2, H2, C2H6, CO).

## License

Same as parent project.
