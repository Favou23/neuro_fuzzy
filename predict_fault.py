# predict.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xanfis import AnfisClassifier
from data_ingestion import prepare_duval_features
from iec_rule_based import duval_polygon_classify, duval_polygons

# Make results path relative to the fuzzy_logic package root so this script can
# be executed from the project root (fuzzy_logic) and still find the saved
# models under the `results/` directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PACKAGE_ROOT, "results")

def load_model_and_encoder(method_name):
    # ANFIS model
    model_dir = os.path.join(RESULTS_DIR, f"{method_name}_anfis")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"ANFIS model folder not found at {model_dir}")
    model_path = os.path.normpath(os.path.join(model_dir, "model.pkl"))
    model = AnfisClassifier.load_model(load_path=model_path)

    # Encoder
    encoder_path = os.path.join("encoders", f"{method_name}_label_encoder.pkl")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")
    encoder = joblib.load(encoder_path)

    # Random Forest (optional)
    rf_pipeline = None
    rf_path = os.path.join(RESULTS_DIR, f"{method_name}_rf", "pipeline.joblib")
    if os.path.exists(rf_path):
        try:
            rf_pipeline = joblib.load(rf_path)
        except Exception:
            rf_pipeline = None

    return model, encoder, rf_pipeline

def transform_features(sample: dict, method: str):
    """
    Same wrapper you had. Returns a DataFrame (1-row) with the features the pipeline expects.
    """
    df = pd.DataFrame([sample])
    if method.lower() == "duval":
        X = prepare_duval_features(df, assume_ppm=True)
    elif method.lower() == "rogers":
        # implement prepare_rogers_features if needed
        from fuzzy_logic.data_ingestion import prepare_rogers_features
        X = prepare_rogers_features(df)
    elif method.lower() == "drm":
        from fuzzy_logic.data_ingestion import prepare_drm_features
        X = prepare_drm_features(df)
    else:
        raise ValueError("unknown method")
    return X

def predict_row(sample: dict, method: str, model, encoder, iec_fallback_threshold=0.6):
    X = transform_features(sample, method)
    print("[DEBUG] Features passed to pipeline:\n", X)
    preds = model.predict(X.to_numpy(dtype=np.float32))
    model_prob = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            model_prob = np.max(probs, axis=1)[0]
            pred_idx = int(np.argmax(probs, axis=1)[0])
        except Exception:
            pred_idx = int(np.array(preds).ravel()[0])
    else:
        pred_idx = int(np.array(preds).ravel()[0])

    try:
        model_label = encoder.inverse_transform([pred_idx])[0]
    except Exception:
        model_label = str(pred_idx)
    print(f"[DEBUG] Model predicted: {model_label} (confidence={model_prob})")

    iec_label = None
    if method.lower() == "duval":
        iec_label = duval_polygon_classify(sample["CH4"], sample["C2H4"], sample["C2H2"], duval_polygons)
        print(f"[DEBUG] IEC Duval label: {iec_label}")

    chosen = model_label
    if iec_label and iec_label != "Unclassified":
        if model_prob is None:
            if model_label != iec_label:
                chosen = iec_label
        else:
            if model_prob < iec_fallback_threshold and model_label != iec_label:
                chosen = iec_label

    print(f"[DEBUG] Final chosen label: {chosen}")
    return chosen


def predict_single(sample, method="all" ):
    results = {}
    methods = [method] if method != "all" else ["duval", "rogers", "drm"]
    for m in methods:
        model, encoder, rf_pipeline = load_model_and_encoder(m)
        anfis_choice = predict_row(sample, m, model, encoder)
        rf_choice = None
        if rf_pipeline is not None:
            try:
                # rf_pipeline expects the same feature set the RF was trained on
                X = transform_features(sample, m)
                rf_pred_idx = rf_pipeline.predict(X)
                try:
                    rf_choice = encoder.inverse_transform([int(rf_pred_idx.ravel()[0])])[0]
                except Exception:
                    rf_choice = str(int(rf_pred_idx.ravel()[0]))
            except Exception as e:
                print(f"RF prediction error for {m}: {e}")

        results[m] = {"anfis": anfis_choice, "rf": rf_choice}

        # 🔍 Debug mapping (print once per method)
        print(f"[{m.upper()}] Mapping:", dict(enumerate(encoder.classes_)))

    return results

# if __name__ == "__main__":
#     model, encoder = load_model_and_encoder("duval")
#     # example sample (raw ppm)
#     example = {"CH4": 100.0, "C2H4": 50.0, "C2H2": 50.0, "H2": 10.0, "C2H6": 5.0}
#     pred = predict_row(example, "duval", model, encoder)
#     print("Prediction:", pred)

def predict_dataset(file_path, method="all", output="predicted_results.csv"):
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
    methods = [method] if method != "all" else ["duval", "rogers", "drm"]

    # detect label column if present (default to FAULT)
    label_col_candidates = ["FAULT", "fault", "label", "Label"]
    label_col = None
    for c in label_col_candidates:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        print("No label column detected in dataset. Predictions will be generated but accuracies cannot be computed unless you supply a label column.")
    else:
        print(f"Found label column: {label_col}. Will compute model accuracies where possible.")

    for m in methods:
        print(f"\n--- Predicting with method: {m} ---")
        model, encoder, rf_pipeline = load_model_and_encoder(m)

        # compute features for all rows depending on method
        if m.lower() == 'duval':
            X = prepare_duval_features(df, assume_ppm=True)
        elif m.lower() == 'rogers':
            from data_ingestion import prepare_rogers_features
            X = prepare_rogers_features(df)
        elif m.lower() == 'drm':
            from data_ingestion import prepare_drm_features
            X = prepare_drm_features(df)
        else:
            raise ValueError(f"Unknown method: {m}")

        # ANFIS predictions (model predicts integer class indices)
        try:
            anfis_preds_idx = model.predict(X.to_numpy(dtype=np.float32))
            anfis_preds_idx = np.asarray(anfis_preds_idx).ravel().astype(int)
            try:
                anfis_preds = encoder.inverse_transform(anfis_preds_idx)
            except Exception:
                anfis_preds = [str(i) for i in anfis_preds_idx]
        except Exception as e:
            print(f"ANFIS prediction error for {m}: {e}")
            anfis_preds = [None] * len(X)
            anfis_preds_idx = [None] * len(X)

        df[f"{m}_anfis_prediction"] = anfis_preds

        # RF predictions (if available)
        rf_preds = [None] * len(X)
        rf_preds_idx = [None] * len(X)
        if rf_pipeline is not None:
            try:
                rf_pred_idx = rf_pipeline.predict(X)
                rf_pred_idx = np.asarray(rf_pred_idx).ravel().astype(int)
                try:
                    rf_preds = encoder.inverse_transform(rf_pred_idx)
                except Exception:
                    rf_preds = [str(i) for i in rf_pred_idx]
                rf_preds_idx = rf_pred_idx
            except Exception as e:
                print(f"RF prediction error for {m}: {e}")

        df[f"{m}_rf_prediction"] = rf_preds

        # If label column exists, compute and print accuracies
        if label_col is not None:
            true_labels = df[label_col].astype(str)
            # ensure encoder can transform true labels; if not, try matching case
            try:
                true_idx = encoder.transform(true_labels)
                # ANFIS accuracy
                if any(v is None for v in anfis_preds_idx):
                    print(f"ANFIS predictions unavailable for {m}; cannot compute accuracy.")
                else:
                    acc_anfis = accuracy_score(true_idx, anfis_preds_idx)
                    print(f"ANFIS accuracy for {m}: {acc_anfis:.4f}")
                # RF accuracy
                if rf_pipeline is None or any(v is None for v in rf_preds_idx):
                    print(f"RF predictions unavailable for {m}; cannot compute RF accuracy.")
                else:
                    acc_rf = accuracy_score(true_idx, rf_preds_idx)
                    print(f"RF accuracy for {m}: {acc_rf:.4f}")
            except Exception as e:
                print(f"Could not transform true labels using encoder for {m}: {e}")

        # Debug mapping
        print(f"[{m.upper()}] Mapping:", dict(enumerate(encoder.classes_)))

    df.to_csv(output, index=False)
    print(f"Predictions saved to {output}")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict transformer fault using ANFIS models.")
    parser.add_argument("--method", choices=["duval", "rogers", "drm", "all"], default="all",
                        help="Which method to use (default: all)")
    parser.add_argument("--file", help="CSV/Excel file for batch prediction")
    args = parser.parse_args()

    if args.file:
        predict_dataset(args.file, method=args.method)
    else:
        # Example single test case
        gas_sample = {"CH4": 500, "C2H4": 500, "C2H2": 50, "H2": 10, "C2H6": 5, "CO": 0}
        results = predict_single(gas_sample, method=args.method)
        print("\nPrediction for single sample:")
        for k, v in results.items():
            print(f"{k.upper()}: {v}")