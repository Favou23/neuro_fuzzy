# QUICKSTART GUIDE — How to Run the Neuro-Fuzzy Project

This guide shows a person how to run all commands without confusion. Just copy-paste the commands into your terminal.

---

## 📁 Project Structure

After running predictions, all outputs are saved to:
```
fuzzy_logic/
├── outputs/                    # All prediction & comparison results
│   ├── anfis/                  # ANFIS model outputs
│   ├── random_forest/          # Random Forest model outputs
│   └── comparison/             # Combined model comparisons & metrics
├── randomforest/               # Random Forest module (do not edit)
├── results/                    # Trained models & encoders (do not edit)
└── [data files]
```

---

## 🚀 Quick Commands

### 1️⃣ Run ANFIS Model Only

#### All three methods (Duval, Rogers, DRM):
```bash
cd c:\Users\USER\my_desktop\neuro_fuzzy\fuzzy_logic
python predict_fault.py --file test_model.csv --method all
```

#### Single method (e.g., Duval):
```bash
python predict_fault.py --file test_model.csv --method duval
```

#### Single sample (gas readings):
```bash
python predict_fault.py --single "{\"H2\":100, \"CH4\":50, \"C2H6\":30, \"C2H4\":20, \"C2H2\":5, \"CO\":15}"
```

**Output location:** `outputs/anfis/`

---

### 2️⃣ Run Random Forest Model Only

#### All three methods:
```bash
python randomforest/rf_predict.py --file test_model.csv --method all
```

#### Single method (e.g., Rogers):
```bash
python randomforest/rf_predict.py --file test_model.csv --method rogers
```

#### Single sample:
```bash
python randomforest/rf_predict.py --single "{\"H2\":100, \"CH4\":50, \"C2H6\":30, \"C2H4\":20, \"C2H2\":5, \"CO\":15}"
```

**Output location:** `outputs/random_forest/`

---

### 3️⃣ Compare Both Models (ANFIS + Random Forest)

#### Compare all methods on a CSV file:
```bash
python compare_models.py --file test_model.csv --method all --output compare_results.csv
```

#### Compare with ground-truth labels (if you have them):
```bash
python compare_models.py --file labeled_test.csv --method all --label-column FAULT --output results.csv
```

#### Compare single samples:
```bash
python compare_models.py --single "{\"H2\":100, \"CH4\":50, \"C2H6\":30, \"C2H4\":20, \"C2H2\":5, \"CO\":15}" --method duval
```

**Output location:** `outputs/comparison/`
- `results.csv` — merged predictions from both models
- `results_metrics.txt` — accuracy, precision, recall, F1-scores
- `cm_*.png` — confusion matrix plots
- `agreement_report.txt` — how often ANFIS & RF agree

---

## 📊 What Each Command Does

| Command | Input | Output | When to Use |
|---------|-------|--------|------------|
| `predict_fault.py --file X.csv` | CSV with gas readings | ANFIS predictions | Quick ANFIS-only predictions |
| `randomforest/rf_predict.py --file X.csv` | CSV with gas readings | RF predictions | Quick RF-only predictions |
| `compare_models.py --file X.csv` | CSV with gas readings | Both models' predictions + metrics | Compare ANFIS vs RF |
| `compare_models.py --file X.csv --label-column FAULT` | CSV with gas readings + ground truth | Full evaluation report + confusion matrices | Full evaluation & analysis |

---

## 🎯 Example Workflows

### Workflow 1: Test ANFIS on your data
```bash
cd c:\Users\USER\my_desktop\neuro_fuzzy\fuzzy_logic
python predict_fault.py --file your_data.csv --method duval
# Check outputs/anfis/ for results
```

### Workflow 2: Test Random Forest on your data
```bash
cd c:\Users\USER\my_desktop\neuro_fuzzy\fuzzy_logic
python randomforest/rf_predict.py --file your_data.csv --method duval
# Check outputs/random_forest/ for results
```

### Workflow 3: Compare both models and get full metrics (with labeled data)
```bash
cd c:\Users\USER\my_desktop\neuro_fuzzy\fuzzy_logic
python compare_models.py --file your_labeled_data.csv --label-column TrueLabel --method all
# Check outputs/comparison/ for:
#   - results.csv (merged predictions)
#   - results_metrics.txt (accuracy, precision, recall, F1)
#   - cm_*.png (confusion matrix plots)
```

---

## 📝 Input File Format

Your CSV file should contain these columns (in any order):
- `H2` — Hydrogen concentration (ppm)
- `CH4` — Methane concentration (ppm)
- `C2H6` — Ethane concentration (ppm)
- `C2H4` — Ethylene concentration (ppm)
- `C2H2` — Acetylene concentration (ppm)
- `CO` — Carbon Monoxide concentration (ppm)

**Optional:** If comparing with ground truth, add a label column:
- `FAULT` or `TrueLabel` or `ground_truth` (name doesn't matter, just specify it with `--label-column`)

Example labeled file:
```csv
H2,CH4,C2H6,C2H4,C2H2,CO,FAULT
100,50,30,20,5,15,Normal
500,200,100,80,10,50,T2
1000,800,300,200,50,100,T3
```

---

## 🔧 Advanced Usage

### Single Method Only
```bash
# Run only Duval method (ANFIS)
python predict_fault.py --file test.csv --method duval

# Run only Rogers method (Random Forest)
python randomforest/rf_predict.py --file test.csv --method rogers
```

### Custom JSON Sample Input
```bash
# For Duval method (4 features: H2, CH4, C2H6, C2H4)
python predict_fault.py --single "{\"H2\":100, \"CH4\":50, \"C2H6\":30, \"C2H4\":20}"

# For Rogers method (6 features: all)
python compare_models.py --single "{\"H2\":100, \"CH4\":50, \"C2H6\":30, \"C2H4\":20, \"C2H2\":5, \"CO\":15}" --method rogers
```

---

## 📂 Output Directory Structure

After running predictions, you'll see:

```
outputs/
├── anfis/
│   ├── duval_predictions.csv
│   ├── rogers_predictions.csv
│   ├── drm_predictions.csv
│   └── anfis_summary.txt
├── random_forest/
│   ├── duval_rf_predictions.csv
│   ├── rogers_rf_predictions.csv
│   ├── drm_rf_predictions.csv
│   └── rf_summary.txt
└── comparison/
    ├── results.csv
    ├── results_metrics.txt
    ├── cm_duval_anfis.png
    ├── cm_duval_rf.png
    ├── cm_rogers_anfis.png
    ├── cm_rogers_rf.png
    ├── cm_drm_anfis.png
    ├── cm_drm_rf.png
    └── agreement_report.txt
```

---

## ✅ Troubleshooting

### "File not found" error
- Make sure your CSV file is in the `fuzzy_logic/` directory, or provide the full path:
  ```bash
  python predict_fault.py --file "C:\path\to\your\file.csv"
  ```

### "Column not found" error
- Your CSV must have the required gas concentration columns: `H2`, `CH4`, `C2H6`, `C2H4`, `C2H2`, `CO`
- Check that column names match exactly (case-sensitive)

### No metrics printed
- Make sure you provided `--label-column` with the correct column name
- Example: `--label-column FAULT` (if your CSV has a column named "FAULT")

### Many warnings about "StandardScaler feature names"
- These are **non-fatal warnings** — predictions are still correct
- Warnings go away in updated version (auto-fixed by refactoring)

---

## 📞 Summary

| Goal | Command |
|------|---------|
| **Quick test of ANFIS** | `python predict_fault.py --file test_model.csv --method all` |
| **Quick test of RF** | `python randomforest/rf_predict.py --file test_model.csv --method all` |
| **Compare both models** | `python compare_models.py --file test_model.csv --method all` |
| **Full evaluation (with labels)** | `python compare_models.py --file labeled.csv --label-column FAULT --method all` |
| **Test single sample** | `python compare_models.py --single "{...}" --method duval` |

All outputs go to `outputs/` directory. **No messy files scattered around!**

---

**For questions or issues, check the command syntax above or see individual module READMEs.**
