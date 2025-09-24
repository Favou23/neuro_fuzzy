# iec_rule_based.py
import numpy as np
from matplotlib.path import Path

# -----------------------
# Helpers
# -----------------------
def ppm_to_percentages(ch4, c2h4, c2h2):
    ch4 = float(ch4)
    c2h4 = float(c2h4)
    c2h2 = float(c2h2)
    total = ch4 + c2h4 + c2h2
    if total <= 0:
        return 0.0, 0.0, 0.0
    return (ch4 / total) * 100.0, (c2h4 / total) * 100.0, (c2h2 / total) * 100.0

# triangle vertices for barycentric mapping
A = np.array([0.0, 0.0])             # 100% CH4
B = np.array([1.0, 0.0])             # 100% C2H4
C = np.array([0.5, np.sqrt(3) / 2])  # 100% C2H2

def percentages_to_xy(ch4_pct, c2h4_pct, c2h2_pct):
    x = (ch4_pct/100.0)*A[0] + (c2h4_pct/100.0)*B[0] + (c2h2_pct/100.0)*C[0]
    y = (ch4_pct/100.0)*A[1] + (c2h4_pct/100.0)*B[1] + (c2h2_pct/100.0)*C[1]
    return (x, y)

def _ensure_paths(polygons):
    """
    polygons may be:
      - dict zone -> list of triplets (ch4%,c2h4%,c2h2%)
      - dict zone -> list of (x,y)
    Returns dict zone -> matplotlib.path.Path
    """
    first_zone = next(iter(polygons))
    first_vertex = polygons[first_zone][0]
    if isinstance(first_vertex, (list, tuple)) and len(first_vertex) == 3:
        polygons_xy = {}
        for zone, triplets in polygons.items():
            polygons_xy[zone] = [percentages_to_xy(*t) for t in triplets]
        return {z: Path(np.array(pts)) for z, pts in polygons_xy.items()}
    else:
        return {z: Path(np.array(pts)) for z, pts in polygons.items()}

# Duval polygon triplets (Table 2 you pasted)
duval_polygons = {
    "PD": [(98,2,0), (100,0,0), (98,0,2)],
    "D1": [(0,0,100),(0,23,77),(64,23,13),(87,0,13)],
    "D2": [(0,23,77),(0,71,29),(31,40,29),(47,40,13),(64,23,13)],
    "DT": [(0,71,29),(0,85,15),(35,50,15),(46,50,4),(96,0,4),(87,0,13),(47,40,13),(31,40,29)],
    "T1": [(76,20,4),(80,20,0),(98,2,0),(98,0,2),(96,0,4)],
    "T2": [(46,50,4),(50,50,0),(80,20,0),(76,20,4)],
    "T3": [(0,85,15),(0,100,0),(50,50,0),(35,50,15)],
}

# Path objects we use by default
DUVAL_PATHS = _ensure_paths(duval_polygons)

def duval_polygon_classify(ch4, c2h4, c2h2, polygons=None):
    """
    Classify sample using Duval polygons.
    polygons: optional same structure as duval_polygons_percent_triplets (triplets) or XY polygons.
    If polygons is None we use the built-in DUVAL_PATHS defined above.
    """
    paths = DUVAL_PATHS if polygons is None else _ensure_paths(polygons)
    ch4_pct, c2h4_pct, c2h2_pct = ppm_to_percentages(ch4, c2h4, c2h2)
    pt = percentages_to_xy(ch4_pct, c2h4_pct, c2h2_pct)
    for zone, path in paths.items():
        if path.contains_point(pt):
            return zone
    return "Unclassified"

# quick threshold fallback (keeps your previous quick rules)
def duval_threshold_classify(ch4, c2h4, c2h2):
    ch4_pct, c2h4_pct, c2h2_pct = ppm_to_percentages(ch4, c2h4, c2h2)
    if c2h2_pct < 4 and c2h4_pct < 20:
        return "PD"
    if c2h2_pct >= 29 and c2h4_pct < 23:
        return "D2"
    if 13 <= c2h2_pct <= 29 and c2h4_pct < 23:
        return "D1"
    if c2h2_pct < 10 and c2h4_pct > 50:
        return "T3"
    if c2h2_pct < 13 and 23 <= c2h4_pct <= 50:
        return "T2"
    if c2h2_pct < 15 and c2h4_pct < 23 and ch4_pct > 50:
        return "T1"
    if 10 <= c2h2_pct <= 15 and 23 <= c2h4_pct <= 50:
        return "DT"
    return "Unclassified"
