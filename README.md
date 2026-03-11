# ⚡ Electro-Thermal Power Management for Electric Vehicles
### Machine Learning-Based Heating Power Prediction — BMW i3 Real-World Dataset

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2Stage-orange?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🔍 Problem Statement

Battery Electric Vehicles (EVs) lose **30–50% driving range in winter** due to cabin heating.
A BMW i3 heater draws up to **7000W continuously** from the same battery that powers the motor.

This project builds a **2-stage machine learning model** that predicts instantaneous heating power
demand using only vehicle sensor data — no direct heater signal needed.
The prediction output can feed directly into an **Energy Management System (EMS)** to:

- Pre-schedule battery power for heating vs traction
- Route regenerative braking energy to the heater
- Reduce peak battery current and extend battery life
- Improve winter range estimation accuracy

---

## 📊 Dataset

| Property | Value |
|---|---|
| Vehicle | BMW i3 (60 Ah battery) |
| Routes | Munich area — highway, urban, mixed |
| Total trips | 70 (TripA = 32 summer, TripB = 38 winter) |
| Used for model | TripB (38 winter trips) only |
| Total rows | 627,092 rows after cleaning |
| Time resolution | 0.1 seconds (10 readings/second) |
| Target variable | Requested Heating Power [W] — range 0 to 7000W |

### Signal Categories

| Category | Signals |
|---|---|
| Environmental | Ambient temperature, Elevation |
| Vehicle Dynamics | Velocity, Motor torque, Throttle, Acceleration, Regen signal |
| Battery | Voltage, Current, Temperature, SoC |
| Thermal Circuit | Cabin temp, Coolant temps, Heat exchanger temp, Requested coolant temp |

---

## 🏗️ Model Architecture

A **2-Stage XGBoost Pipeline** was used to handle the zero-inflated heating target:

```
INPUT: 31 sensor features (per timestep)
            │
   ┌─────────▼──────────┐
   │   STAGE 1          │
   │  XGBClassifier     │   → Is the heater ON or OFF?
   │  (Idle vs Active)  │
   └──────┬───────┬─────┘
     OFF  │       │  ON
   ┌──────▼─┐   ┌─▼────────────────────┐
   │  0W    │   │  STAGE 2             │
   └────────┘   │  XGBRegressor        │   → How many Watts?
                │  (Active rows only)  │
                └─────────┬────────────┘
                    Predicted Power (W)
                    rounded to 40W steps
```

**Why 2-stage?**
A single regressor struggles when most rows have `target = 0W` (idle) and active rows range
from 1000W–7000W. Separating idle detection from power regression solves this cleanly.

---

## ⚙️ Feature Engineering

From 48 raw CSV columns, **31 final features** were used after:
- Removing 1 target column, 1 time index, 5 leakage columns, 1 duplicate
- Engineering 10 new physical features
- Selecting top 40 by correlation, removing near-duplicates (>99.3% correlated)

### Key Engineered Features

| Feature | Formula | Physical Meaning |
|---|---|---|
| `f_cabin_ambient_dt` | Cabin temp − Ambient temp | How warm cabin is vs outside |
| `f_coolant_req_dt` | Requested coolant − Heatercore actual | Heating deficit in coolant circuit |
| `f_warmup_needed` | max(0, 20 − Ambient temp) | Degrees below comfort threshold |
| `f_battery_cold` | max(0, 15 − Battery temp) | Battery cold stress |
| `f_battery_power_w` | Voltage × Current | Total battery power draw |
| `f_cabin_coolant_dt` | Requested coolant − Cabin temp | Temperature gap to reach comfort |

### Removed Leakage Columns

| Column | Reason |
|---|---|
| Heating Power CAN / LIN | Alternate versions of the target |
| Heater Voltage / Current | Only present when heater is ON — reveals the answer |
| Heater Signal | Binary flag directly encoding heater state |

---

## 📈 Results

### Train / Validation / Test Split
Data was split at the **trip level** (not row level) to ensure the model is tested on
entire trips it never saw during training.

| Split | Trips | Rows |
|---|---|---|
| Train | 26 trips | 402,241 |
| Validation | 6 trips | 115,487 |
| Test | 6 trips | 109,364 |

### Final Performance Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| **R² Score (all rows)** | 97.35% | 83.30% | 73.62% |
| **R² Score (active only)** | 97.26% | 82.99% | **80.30%** |
| **MAE — active rows** | 143W | 315W | **303W** |
| **RMSE** | 218W | 445W | 604W |
| **MAE as % of full scale** | 2.04% | 4.48% | **4.32%** |
| **Idle Detection Accuracy** | — | **99.93%** | 87.27% |
| **Idle Detection F1** | — | **99.93%** | 83.56% |

> Full scale = 0–7000W. Active rows = rows where heating demand > 120W.

### Key Plots

| Plot | Description |
|---|---|
| `plots/00_summary_dashboard.png` | All metrics in one presentation-ready image |
| `plots/04_actual_vs_pred_test.png` | Actual vs predicted scatter — test trips |
| `plots/06_metrics_comparison.png` | R², RMSE, MAE bar charts across splits |
| `plots/09_feature_importance.png` | Top 20 features for classifier and regressor |
| `plots/11_sample_trip_timeseries.png` | Actual vs predicted over time for 3 trips |

---

## 🗂️ Repository Structure

```
Electro-thermal-modelling/
├── scripts/
│   └── tripB_only_final_plots.py     # Complete pipeline — loads, trains, evaluates, plots
├── outputs/
│   ├── final_metrics.csv             # All numeric results
│   ├── selected_features.csv         # 31 features used with importance scores
│   ├── classifier_stats.csv          # Idle/active F1 and accuracy
│   ├── split_summary.csv             # Trip count per split
│   └── run_summary.json              # Full run config and results
├── plots/                            # 14 publication-ready graphs
│   ├── 00_summary_dashboard.png
│   ├── 01_target_distribution.png
│   └── ...
├── models/
│   └── tripB_2stage_model.joblib     # Saved model — ready to deploy
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### Install dependencies
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib
```

### Run the full pipeline
```bash
python scripts/tripB_only_final_plots.py
```

This will:
1. Load all 38 TripB CSV files from `E:\IML\measurement\`
2. Clean data, remove leakage columns, engineer features
3. Split trips into train/validation/test
4. Train Stage 1 classifier + Stage 2 regressor with early stopping
5. Export all metrics, predictions, graphs, and saved model

> **Note:** Raw CSV data files are not included in this repo due to size.
> Update `DATA_DIR` and `OUTPUT_DIR` in the script to match your local paths.

### Load saved model for inference
```python
import joblib
import pandas as pd

bundle = joblib.load("models/tripB_2stage_model.joblib")
clf, reg, imp = bundle["clf"], bundle["reg"], bundle["imputer"]
features = bundle["features"]

# X_new = your new sensor data as a DataFrame
X_new_imputed = pd.DataFrame(imp.transform(X_new[features]), columns=features)

# Predict
prob = clf.predict_proba(X_new_imputed)[:, 1]
import numpy as np
pred = np.zeros(len(X_new_imputed))
active = prob >= 0.5
pred[active] = np.expm1(reg.predict(X_new_imputed.iloc[np.where(active)[0]]))
pred = np.clip(np.round(pred / 40) * 40, 0, 7000)
print("Predicted heating power (W):", pred)
```

---

## 👥 Team

| Name | Reg No |
|---|---|
| Tanush Pavan V | CB.EN.U4EEE24147 |
| Tejas | CB.EN.U4EEE24148 |
| Thirupugazhl | CB.EN.U4EEE24149 |
| Harish V | CB.EN.U4EEE24115 |

**Department:** Electrical and Electronics Engineering
**Institution:** Amrita School of Engineering
**Date:** March 2026

---
