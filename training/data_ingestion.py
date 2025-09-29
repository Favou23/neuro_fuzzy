# data_ingestion.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from duval_polygon_generator import generate_duval_polygon_dataset

def _normalize_columns(df):
    df.columns = df.columns.str.strip().str.upper()
    return df

def _find_col(df, col_name):
    cols = {c.strip().upper(): c for c in df.columns}
    key = col_name.strip().upper()
    if key in cols:
        return cols[key]
    raise KeyError(f"Column '{col_name}' not found. Available: {list(df.columns)}")

def prepare_duval_features(df, assume_ppm=True):
    """
    Prepare Duval features: returns DataFrame with columns [CH4_PCT, C2H4_PCT, C2H2_PCT].
    If assume_ppm=True, converts raw ppm to percentages.
    If assume_ppm=False, expects columns CH4,C2H4,C2H2 to already be percentages.
    """
    df = df.copy()
    required = ["CH4", "C2H4", "C2H2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Duval: {missing}")

    if assume_ppm:
        total = df["CH4"].astype(float) + df["C2H4"].astype(float) + df["C2H2"].astype(float)
        df["CH4_PCT"] = df["CH4"].astype(float).div(total.replace(0, np.nan)).mul(100.0).fillna(0.0)
        df["C2H4_PCT"] = df["C2H4"].astype(float).div(total.replace(0, np.nan)).mul(100.0).fillna(0.0)
        df["C2H2_PCT"] = df["C2H2"].astype(float).div(total.replace(0, np.nan)).mul(100.0).fillna(0.0)
    else:
        df = df.rename(columns={"CH4": "CH4_PCT", "C2H4": "C2H4_PCT", "C2H2": "C2H2_PCT"})
        df["CH4_PCT"] = df["CH4_PCT"].astype(float).fillna(0.0)
        df["C2H4_PCT"] = df["C2H4_PCT"].astype(float).fillna(0.0)
        df["C2H2_PCT"] = df["C2H2_PCT"].astype(float).fillna(0.0)

    return df[["CH4_PCT", "C2H4_PCT", "C2H2_PCT"]]

def prepare_rogers_features(df):
    required = ["H2", "CH4", "C2H2", "C2H4", "C2H6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Rogers: {missing}")

    df = df.copy()
    df["R1"] = df["CH4"].astype(float) / df["H2"].replace(0, np.nan)
    df["R2"] = df["C2H2"].astype(float) / df["C2H4"].replace(0, np.nan)
    df["R5"] = df["C2H4"].astype(float) / df["C2H6"].replace(0, np.nan)
    df[["R1", "R2", "R5"]] = df[["R1","R2","R5"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[["R1", "R2", "R5"]]

def prepare_drm_features(df, L1=None):
    required = ["H2", "CH4", "C2H2", "C2H4", "C2H6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for DRM: {missing}")

    df = df.copy()
    if L1 is None:
        L1 = {"H2":100.0, "CH4":120.0, "C2H2":35.0, "C2H4":50.0, "C2H6":65.0}
    df["R1"] = df["CH4"].astype(float) / df["H2"].replace(0, np.nan)
    df["R2"] = df["C2H2"].astype(float) / df["C2H4"].replace(0, np.nan)
    df["R3"] = df["C2H4"].astype(float) / df["C2H6"].replace(0, np.nan)
    df["R4"] = df["C2H2"].astype(float) / df["CH4"].replace(0, np.nan)
    df[["R1","R2","R3","R4"]] = df[["R1","R2","R3","R4"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_mask = (
        (df["H2"].astype(float) >= 2 * L1["H2"]) |
        (df["CH4"].astype(float) >= 2 * L1["CH4"]) |
        (df["C2H2"].astype(float) >= 2 * L1["C2H2"]) |
        (df["C2H4"].astype(float) >= 2 * L1["C2H4"]) |
        (df["C2H6"].astype(float) >= 2 * L1["C2H6"])
    )
    df.loc[~valid_mask, ["R1","R2","R3","R4"]] = 0.0
    return df[["R1","R2","R3","R4"]]

def load_and_prepare_data(filepath, label_col, method_name,
                          augment_iec=False, iec_synth_per_class=300,
                          iec_ppm_range=(10,5000), iec_jitter=0.08, seed=42):
    """
    Load CSV and compute method-specific features.
    If augment_iec=True and method_name=='duval', we generate IEC-synthetic samples and append them.
    Returns: X (DataFrame of features), y_encoded (np.array), label_encoder
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(base_dir, filepath))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Data file not found: {abs_path}")

    df = pd.read_csv(abs_path)
    print(f"\nLoaded {method_name.upper()} dataset with shape: {df.shape}")

    df = _normalize_columns(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    method = method_name.strip().lower()

    # compute features (these are what the model will be trained on)
    if method == "duval":
        X = prepare_duval_features(df, assume_ppm=True)
    elif method == "rogers":
        X = prepare_rogers_features(df)
    elif method == "drm":
        X = prepare_drm_features(df)
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    # labels
    if label_col.upper() not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")
    y_series = df[_find_col(df, label_col)]

    # encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_series)

    # # IEC augmentation if requested (only for Duval)
    # if augment_iec and method == "duval":
        
    #     synth_df = generate_duval_polygon_dataset(samples_per_class=iec_synth_per_class,
    #                                               seed=seed, ppm_range=iec_ppm_range, jitter=iec_jitter)
    #     # ensure columns uppercase to match pipeline
    #     synth_df.columns = synth_df.columns.str.strip().upper()
    #     # compute Duval features for synthetic (ppm -> %)
    #     X_synth = prepare_duval_features(synth_df, assume_ppm=True)
    #     # encode synth labels using same encoder; ensure all synth classes are in encoder.classes_
    #     synth_labels = synth_df["FAULT"].astype(str)
    #     # If label encoder hasn't seen a synth class (shouldn't happen because synth uses IEC classes),
    #     # we map synth labels to encoder classes; if mismatch raise.
    #     unknowns = set(synth_labels.unique()) - set(le.classes_)
    #     if unknowns:
    #         raise ValueError(f"Synthetic labels contain unknown classes: {unknowns}")
    #     y_synth = le.transform(synth_labels)
    #     # append
    #     X = pd.concat([X, X_synth], ignore_index=True)
    #     y_encoded = np.concatenate([y_encoded, y_synth])
    #     print(f"Augmented data with IEC-synthetic samples: {len(X_synth)} per class (total added {len(X_synth)})")

    return X, y_encoded, le
