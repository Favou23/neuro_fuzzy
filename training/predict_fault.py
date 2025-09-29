# predict.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from xanfis import AnfisClassifier
from data_ingestion import prepare_duval_features
from iec_rule_based import duval_polygon_classify, duval_polygons

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def load_model_and_encoder(method_name):
    model_dir = os.path.join(RESULTS_DIR, f"{method_name}_anfis")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Pipeline not found at {model_dir}")

    model_path = os.path.normpath(os.path.join(model_dir, "model.pkl"))
    model = AnfisClassifier.load_model(load_path=model_path)
    encoder_path = os.path.join("encoders", f"{method_name}_label_encoder.pkl")
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")
    encoder = joblib.load(encoder_path)
    return model, encoder

def transform_features(sample: dict, method: str):
    """
    Same wrapper you had. Returns a DataFrame (1-row) with the features the pipeline expects.
    """
    df = pd.DataFrame([sample])
    if method.lower() == "duval":
        X = prepare_duval_features(df, assume_ppm=True)
    elif method.lower() == "rogers":
        # implement prepare_rogers_features if needed
        from data_ingestion import prepare_rogers_features
        X = prepare_rogers_features(df)
    elif method.lower() == "drm":
        from data_ingestion import prepare_drm_features
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
        model, encoder = load_model_and_encoder(m)
        results[m] = predict_row(sample, m, model, encoder)

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

    for m in methods:
        model, encoder = load_model_and_encoder(m)
        df[f"{m}_prediction"] = df.apply(
            lambda r: predict_row(r.to_dict(), m, model, encoder), axis=1
        )

        #  Debug mapping
        print(f"[{m.upper()}] Mapping:", dict(enumerate(encoder.classes_)))

    df.to_csv(output, index=False)
    print(f" Predictions saved to {output}")


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