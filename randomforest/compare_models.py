"""
Compare ANFIS and Random Forest predictions on a dataset or single sample.

Usage examples:

# 1) Batch predictions only (no ground-truth labels):
python randomforest/compare_models.py --file test_model.csv --methods all --out combined_predictions.csv

# 2) Batch predictions + metrics (requires a label column in the CSV):
python randomforest/compare_models.py --file labeled_test.csv --label FAULT --methods duval --out results_with_metrics.csv

# 3) Single sample (no label):
python randomforest/compare_models.py --single "CH4=500,C2H4=500,C2H2=50,H2=10,C2H6=5,CO=0" --methods all

The script will:
 - Load the input CSV (or single sample)
 - For each requested method (duval, rogers, drm):
   - Load ANFIS model and make predictions
   - Load RF model and make predictions and confidence
 - If --label provided and present in CSV, compute accuracy, classification report and confusion matrices for both models
 - Save a combined CSV with predictions
 - Save confusion matrix images to results/

Note: This script expects the trained models and encoders to exist:
 - ANFIS models in results/{method}_anfis/model.pkl and encoders in encoders/{method}_label_encoder.pkl
 - RF models in results/{method}_randomforest/model.pkl and encoders in encoders/{method}_label_encoder.pkl

"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ensure parent fuzzy_logic in path
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# import ANFIS prediction helpers
try:
    from predict_fault import load_model_and_encoder as load_anfis_model_and_encoder
    from predict_fault import transform_features as anfis_transform_features
    from predict_fault import predict_row as anfis_predict_row
except Exception:
    # If importing the whole module fails, we'll import when needed
    load_anfis_model_and_encoder = None
    anfis_transform_features = None
    anfis_predict_row = None

# import RF helpers
from randomforest.rf_predict import load_rf_model_and_encoder, transform_features as rf_transform_features, predict_single_rf

RESULTS_DIR = os.path.join(PARENT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_single_sample(s: str):
    """Parse sample string like 'CH4=500,C2H4=500,...' into dict"""
    d = {}
    for part in s.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            try:
                val = float(v)
            except:
                val = v
            d[k.strip()] = val
    return d


def plot_and_save_cm(y_true, y_pred, labels, out_path, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_on_dataframe(df, methods, label_col=None, out_file=None):
    """Run predictions for ANFIS and RF over dataframe rows"""
    df = df.copy()

    for method in methods:
        method_lower = method.lower()
        # ANFIS
        try:
            # lazy import if earlier failed
            global load_anfis_model_and_encoder, anfis_predict_row
            if load_anfis_model_and_encoder is None:
                from predict_fault import load_model_and_encoder as load_anfis_model_and_encoder
                from predict_fault import predict_row as anfis_predict_row

            anfis_model, anfis_encoder = load_anfis_model_and_encoder(method_lower)
            df[f"{method_lower}_anfis_prediction"] = df.apply(
                lambda r: anfis_predict_row(r.to_dict(), method_lower, anfis_model, anfis_encoder), axis=1
            )
            print(f"[ANFIS] Completed predictions for {method_upper(method_lower)}")
        except Exception as e:
            print(f"[ANFIS] Warning: could not run ANFIS for {method}: {e}")
            df[f"{method_lower}_anfis_prediction"] = None

        # RF
        try:
            rf_model, rf_encoder = load_rf_model_and_encoder(method_lower)
            preds = []
            confs = []
            for idx, row in df.iterrows():
                try:
                    p, c = predict_single_rf(row.to_dict(), method_lower, rf_model, rf_encoder, verbose=False)
                    preds.append(p)
                    confs.append(c if c is not None else 0.0)
                except Exception as e:
                    preds.append(None)
                    confs.append(0.0)
                    print(f"[RF] Error on row {idx}: {e}")
            df[f"{method_lower}_rf_prediction"] = preds
            df[f"{method_lower}_rf_confidence"] = confs
            print(f"[RF] Completed predictions for {method_upper(method_lower)}")
        except Exception as e:
            print(f"[RF] Warning: could not run RF for {method}: {e}")
            df[f"{method_lower}_rf_prediction"] = None
            df[f"{method_lower}_rf_confidence"] = None

        # If label column provided and exists, compute metrics
        if label_col and label_col in df.columns:
            true = df[label_col].astype(str)

            # ANFIS metrics
            if f"{method_lower}_anfis_prediction" in df.columns and df[f"{method_lower}_anfis_prediction"].notnull().any():
                y_pred_anfis = df[f"{method_lower}_anfis_prediction"].astype(str)
                try:
                    acc_a = accuracy_score(true, y_pred_anfis)
                    print(f"\n[METRICS][{method_upper(method_lower)}][ANFIS] Accuracy: {acc_a:.4f}")
                    print(classification_report(true, y_pred_anfis, zero_division=0))
                    cm_path = os.path.join(RESULTS_DIR, f"confusion_{method_lower}_anfis.png")
                    labels = sorted(list(set(true) | set(y_pred_anfis)))
                    plot_and_save_cm(true, y_pred_anfis, labels, cm_path, title=f"ANFIS - {method_upper(method_lower)}")
                    print(f"[METRICS] Saved ANFIS confusion matrix to {cm_path}")
                except Exception as e:
                    print(f"[METRICS] Failed ANFIS metrics for {method}: {e}")

            # RF metrics
            if f"{method_lower}_rf_prediction" in df.columns and df[f"{method_lower}_rf_prediction"].notnull().any():
                y_pred_rf = df[f"{method_lower}_rf_prediction"].astype(str)
                try:
                    acc_r = accuracy_score(true, y_pred_rf)
                    print(f"\n[METRICS][{method_upper(method_lower)}][RF] Accuracy: {acc_r:.4f}")
                    print(classification_report(true, y_pred_rf, zero_division=0))
                    cm_path = os.path.join(RESULTS_DIR, f"confusion_{method_lower}_rf.png")
                    labels = sorted(list(set(true) | set(y_pred_rf)))
                    plot_and_save_cm(true, y_pred_rf, labels, cm_path, title=f"RF - {method_upper(method_lower)}")
                    print(f"[METRICS] Saved RF confusion matrix to {cm_path}")
                except Exception as e:
                    print(f"[METRICS] Failed RF metrics for {method}: {e}")

    # Save combined results
    if out_file is None:
        base = 'combined_predictions.csv'
    else:
        base = out_file
    df.to_csv(base, index=False)
    print(f"\n[OUTPUT] Combined predictions saved to: {base}")
    return df


def method_upper(m):
    return m.upper()


def main():
    parser = argparse.ArgumentParser(description='Run ANFIS and RF predictions and optionally evaluate them')
    parser.add_argument('--file', help='CSV or Excel file to run batch predictions on')
    parser.add_argument('--single', help='Single sample as CSV-like string: CH4=500,C2H4=500,...')
    parser.add_argument('--label', help='Column name in file that contains ground-truth label (optional)')
    parser.add_argument('--methods', nargs='+', default=['all'], help='Methods to run: duval rogers drm or all')
    parser.add_argument('--out', help='Output CSV file path for combined predictions', default=None)
    args = parser.parse_args()

    methods = args.methods
    if methods == ['all']:
        methods = ['duval', 'rogers', 'drm']

    if args.single:
        sample = parse_single_sample(args.single)
        df = pd.DataFrame([sample])
        print(f"Running single sample for methods: {methods}")
        run_on_dataframe(df, methods, label_col=None, out_file=args.out)
        return

    if not args.file:
        print("Error: either --file or --single must be provided")
        return

    # Load dataset
    if args.file.endswith('.xlsx') or args.file.endswith('.xls'):
        df = pd.read_excel(args.file)
    else:
        df = pd.read_csv(args.file)

    print(f"Loaded file {args.file} with shape {df.shape}")

    # If label provided but column not present, warn
    if args.label and args.label not in df.columns:
        print(f"Warning: Label column '{args.label}' not found in file. Metrics will not be computed.")
        args.label = None

    run_on_dataframe(df, methods, label_col=args.label, out_file=args.out)


if __name__ == '__main__':
    main()
