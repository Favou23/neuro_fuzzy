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


def rogers_classify_row(row):
    """
    Rogers classification expects row with H2, CH4, C2H2, C2H4, C2H6.
    Returns one of: Normal, D1, D2, T1, T2, T3, Unknown
    """
    try:
        h2 = float(row["H2"])
        ch4 = float(row["CH4"])
        c2h2 = float(row["C2H2"])
        c2h4 = float(row["C2H4"])
        c2h6 = float(row["C2H6"])
    except Exception:
        return "Unknown"

    r1 = ch4 / h2 if h2 != 0 else 0.0
    r2 = c2h2 / c2h4 if c2h4 != 0 else 0.0
    r5 = c2h4 / c2h6 if c2h6 != 0 else 0.0

    # IEC-style rules (from your provided rules)
    if r2 < 0.01 and 0.1 < r1 < 1.0 and r5 < 1.0:
        return "Normal"
    if r2 < 0.1 and r1 < 0.1 and r5 < 1.0:
        return "D1"
    if 0.1 <= r2 <= 3.0 and 0.1 <= r1 <= 1.0 and r5 >= 3.0:
        return "D2"
    if r2 < 0.1 and 0.1 < r1 < 1.0 and 1.0 <= r5 <= 3.0:
        return "T1"
    if r2 < 0.1 and r1 >= 1.0 and 1.0 <= r5 <= 3.0:
        return "T2"
    if r2 < 0.1 and r1 >= 1.0 and r5 >= 3.0:
        return "T3"
    return "Unknown"

def drm_classify_row(row, L1_thresholds=None):
    """
    DRM classification using row dict (H2, CH4, C2H2, C2H4, C2H6).
    Returns: 'Corona', 'Arcing', 'Thermal', 'Not applicable', or 'Unknown'
    """
    if L1_thresholds is None:
        L1_thresholds = {"H2":50.0, "CH4":10.0, "C2H2":0.5, "C2H4":5.0, "C2H6":5.0}

    try:
        H2 = float(row["H2"])
        CH4 = float(row["CH4"])
        C2H2 = float(row["C2H2"])
        C2H4 = float(row["C2H4"])
        C2H6 = float(row["C2H6"])
    except Exception:
        return "Unknown"

    # precondition: at least one gas >= 2 * L1
    applicable = (
        H2 >= 2 * L1_thresholds["H2"] or
        CH4 >= 2 * L1_thresholds["CH4"] or
        C2H2 >= 2 * L1_thresholds["C2H2"] or
        C2H4 >= 2 * L1_thresholds["C2H4"] or
        C2H6 >= 2 * L1_thresholds["C2H6"]
    )
    if not applicable:
        return "Not applicable"

    R1 = CH4 / H2 if H2 != 0 else 0.0
    R2 = C2H2 / C2H4 if C2H4 != 0 else 0.0
    R3 = C2H2 / CH4 if CH4 != 0 else 0.0
    R4 = C2H4 / C2H6 if C2H6 != 0 else 0.0

    if R2 < 0.1 and R1 < 0.1:
        return "Corona"
    if R3 > 1.5 and R4 < 0.5:
        return "Arcing"
    if R1 > 1.0 and R4 > 1.0:
        return "Thermal"
    return "Unknown"