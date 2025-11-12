# duval_polygon_generator.py
import numpy as np
import pandas as pd
import random
from matplotlib.path import Path
import os
from fuzzy_logic.iec_rule_based import duval_polygons, percentages_to_xy  # note import

# convert triplets -> XY once
duval_polygons_xy = {z: [percentages_to_xy(*t) for t in triplets]
                      for z, triplets in duval_polygons.items()}
duval_paths = {z: Path(xy) for z, xy in duval_polygons_xy.items()}

def sample_point_in_path(path, max_trials=2000):
    xmin = min(p[0] for p in path.vertices)
    xmax = max(p[0] for p in path.vertices)
    ymin = min(p[1] for p in path.vertices)
    ymax = max(p[1] for p in path.vertices)
    for _ in range(max_trials):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        if path.contains_point((x, y)):
            return (x, y)
    raise RuntimeError("Failed to sample in polygon")

def xy_to_percentages(x, y):
    # compute barycentric weights (solves for wA,wB,wC s.t. (x,y)=wA*A+wB*B+wC*C, wA+wB+wC=1)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    denom = ((B[1]-C[1])*(A[0]-C[0]) + (C[0]-B[0])*(A[1]-C[1]))
    wA = ((B[1]-C[1])*(x - C[0]) + (C[0]-B[0])*(y - C[1])) / denom
    wB = ((C[1]-A[1])*(x - C[0]) + (A[0]-C[0])*(y - C[1])) / denom
    wC = 1.0 - wA - wB
    weights = np.array([wA, wB, wC])
    weights = np.clip(weights, 0.0, 1.0)
    if weights.sum() == 0:
        return 0.0, 0.0, 0.0
    weights = weights / weights.sum()
    return float(weights[0]*100.0), float(weights[1]*100.0), float(weights[2]*100.0)

def generate_duval_polygon_dataset(samples_per_class=300, seed=42, ppm_range=(10, 5000), jitter=0.08):
    """
    Generate raw-ppm dataset by sampling inside IEC Duval polygons.
    Returns DataFrame with columns: CH4, C2H4, C2H2, FAULT
    - samples_per_class: number of points per fault polygon
    - ppm_range: (min_ppm, max_ppm) used to scale percentages -> ppm (random per sample)
    - jitter: multiplicative noise fraction (e.g. 0.08 means +/-8%)
    """
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for zone, path in duval_paths.items():
        for _ in range(samples_per_class):
            x, y = sample_point_in_path(path)
            ch4_pct, c2h4_pct, c2h2_pct = xy_to_percentages(x, y)

            base_ppm = random.uniform(*ppm_range)
            ch4_ppm = ch4_pct/100.0 * base_ppm
            c2h4_ppm = c2h4_pct/100.0 * base_ppm
            c2h2_ppm = c2h2_pct/100.0 * base_ppm

            ch4_ppm *= random.uniform(1.0-jitter, 1.0+jitter)
            c2h4_ppm *= random.uniform(1.0-jitter, 1.0+jitter)
            c2h2_ppm *= random.uniform(1.0-jitter, 1.0+jitter)

            rows.append([round(ch4_ppm,4), round(c2h4_ppm,4), round(c2h2_ppm,4), zone])

    df = pd.DataFrame(rows, columns=["CH4", "C2H4", "C2H2", "FAULT"])
    return df

if __name__ == "__main__":
    out = generate_duval_polygon_dataset(samples_per_class=500)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ds_dir = os.path.join(base_dir, "duval_data_generator", "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    out.to_csv(os.path.join(ds_dir, "duval_polygon_dataset.csv"), index=False)
    print("Saved sample dataset to:", os.path.join(ds_dir, "duval_polygon_dataset.csv"))
