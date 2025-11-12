# rogers_rule_based_generator.py
import numpy as np
import pandas as pd
import os
import random
from fuzzy_logic.iec_rule_based import rogers_classify_row  # <-- use the IEC rules

def gen_positive(mean, std):
    """Gaussian noise, positive only"""
    return max(0.01, np.random.normal(loc=mean, scale=std))

def generate_rogers_dataset(samples_per_class=500, seed=42, save_path="duval_data_generator/datasets/rogers_rule_dataset.csv"):
    random.seed(seed)
    np.random.seed(seed)
    
    rows = []
    counts = {}  # track how many per class
    
    # target classes we want
    target_classes = ["Normal", "D1", "D2", "T1", "T2", "T3"]
    max_needed = {cls: samples_per_class for cls in target_classes}
    
    while any(counts.get(cls, 0) < max_needed[cls] for cls in target_classes):
        gases = {
            "H2": gen_positive(200, 80),
            "CH4": gen_positive(200, 80),
            "C2H2": gen_positive(50, 20),
            "C2H4": gen_positive(100, 40),
            "C2H6": gen_positive(80, 30),
        }
        label = rogers_classify_row(gases)
        if label in target_classes and counts.get(label, 0) < max_needed[label]:
            rows.append([gases["CH4"], gases["H2"], gases["C2H2"], gases["C2H4"], gases["C2H6"], label])
            counts[label] = counts.get(label, 0) + 1
    
    df = pd.DataFrame(rows, columns=["CH4", "H2", "C2H2", "C2H4", "C2H6", "FAULT"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f" Rogers dataset saved to {save_path} with shape {df.shape}")
    print(df['FAULT'].value_counts())
    return df

if __name__ == "__main__":
    generate_rogers_dataset(samples_per_class=500)

