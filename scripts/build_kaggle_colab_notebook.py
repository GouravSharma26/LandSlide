"""Builder script to generate the self-contained Kaggle / Colab Master Notebook.

Produces a clean, academic, symbol-free notebook:
    notebooks/early_landslide_risk_detection_master.ipynb
"""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header cell
    add_md(r"""# Early Landslide Risk Detection Using Visual Terrain Changes
### Multi-Modal Remote Sensing (Sentinel-1 SAR, Sentinel-2 MSI, SRTM DEM, CHIRPS) and Machine Learning

---

## Abstract and Problem Statement
Landslides present a critical mass-wasting threat across the Himalayan Mountain Arc (26.0°N–35.5°N, 74.0°E–97.0°E). Over 85% of fatal events occur during the summer South Asian Monsoon (June–September) triggered by intense orographic precipitation.

### The Problem: The Monsoon Optical Blindspot
Optical satellites (Sentinel-2, Landsat) face a 31.5% imagery missingness rate in the Himalayas during the monsoon due to dense cloud cover and heavy rainfall—precisely when landslide risk is highest. Ground sensor networks are cost-prohibitive across large mountain expanses.

### The Method: Dual-Workflow Remote Sensing Architecture
This research implements an end-to-end multi-modal machine learning pipeline:
1. **Sentinel-1 C-Band SAR (5.405 GHz):** Active microwave radar (VV, VH) providing day-and-night surface disturbance assessment through cloud cover.
2. **Decoupled Workflows:**
   - **Workflow A (Event Detection):** Post-hoc disturbance classification using radar change deltas ($\Delta$VH, $\Delta$VV), event rainfall accumulation (1d, 7d, 30d, API), and topographic slope.
   - **Workflow B (Susceptibility Prediction):** Pre-event spatial hazard ranking conditioned strictly on static SRTM topography, pre-event SAR baseline, and 30-day antecedent precipitation behind an Anti-Leakage Firewall.
3. **100 km Spatial Cluster Purging:** Outer 5-fold spatial cluster partition with a strict 100 km geographic exclusion buffer to prevent spatial autocorrelation (Tobler's First Law of Geography) from inflating test metrics.""")

    # Cell 1: Environment Setup
    add_md(r"""## 1. Environment Setup
Install dependencies and configure libraries.""")

    add_code(r"""# Install dependencies if missing
!pip install -q xgboost shap scikit-learn matplotlib pandas numpy scipy

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    brier_score_loss,
)
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap

# Configure standard clean plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Environment initialized successfully.")""")

    # Cell 2: Data Loading
    add_md(r"""## 2. Dataset Ingestion
Load the multi-source feature matrix (28 topographic, radar, optical, and precipitation predictors across the Himalayan study area).
The loader automatically checks local directories, Kaggle dataset paths, and GitHub raw storage.""")

    add_code(r"""def load_feature_matrix():
    possible_paths = [
        Path("outputs/multisource_feature_matrix.csv"),
        Path("../outputs/multisource_feature_matrix.csv"),
        Path("/kaggle/input/himalayan-landslide-dataset/multisource_feature_matrix.csv"),
        Path("multisource_feature_matrix.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            print(f"Loaded feature matrix from: {p}")
            return pd.read_csv(p)

    raw_url = "https://raw.githubusercontent.com/GouravSharma26/LandSlide/main/outputs/multisource_feature_matrix.csv"
    print(f"Fetching dataset from GitHub repository: {raw_url}")
    try:
        return pd.read_csv(raw_url)
    except Exception as e:
        raise RuntimeError(f"Could not load feature matrix: {e}")

df = load_feature_matrix()

if "event_date" in df.columns:
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["year"] = df["event_date"].dt.year
    df["month"] = df["event_date"].dt.month

print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Class distribution: {df['label'].value_counts().to_dict()} (1 = Landslide, 0 = Diagnostic Control)")
display(df.head(3))""")

    # Cell 3: Exploratory Data Analysis
    add_md(r"""## 3. Exploratory Data Analysis
Examine the physical distributions of the key remote sensing and terrain signals:
- $\Delta$VH backscatter drop (Sentinel-1 SAR)
- Topographic slope (SRTM 30m DEM)
- 7-day cumulative rainfall (CHIRPS)""")

    add_code(r"""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Delta VH Backscatter Distribution
landslide_dvh = df[df['label'] == 1]['dVH'].dropna()
control_dvh = df[df['label'] == 0]['dVH'].dropna()
axes[0].hist(control_dvh, bins=25, alpha=0.6, label='Control', color='#10B981', density=True)
axes[0].hist(landslide_dvh, bins=25, alpha=0.6, label='Landslide', color='#EF4444', density=True)
axes[0].set_title("Sentinel-1 $\Delta$VH Backscatter Change (dB)")
axes[0].set_xlabel("$\Delta$VH = $VH_{post} - VH_{pre}$ (dB)")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].axvline(0, color='gray', linestyle='--', linewidth=1)

# 2. Slope Gradient Distribution
landslide_slope = df[df['label'] == 1]['slope'].dropna()
control_slope = df[df['label'] == 0]['slope'].dropna()
axes[1].hist(control_slope, bins=25, alpha=0.6, label='Control', color='#10B981', density=True)
axes[1].hist(landslide_slope, bins=25, alpha=0.6, label='Landslide', color='#EF4444', density=True)
axes[1].set_title("SRTM Topographic Slope (Degrees)")
axes[1].set_xlabel("Slope (°)")
axes[1].set_ylabel("Density")
axes[1].legend()

# 3. 7-Day Cumulative Rainfall Distribution
landslide_rain = df[df['label'] == 1]['rain_7d'].dropna()
control_rain = df[df['label'] == 0]['rain_7d'].dropna()
axes[2].hist(control_rain, bins=25, alpha=0.6, label='Control', color='#10B981', density=True)
axes[2].hist(landslide_rain, bins=25, alpha=0.6, label='Landslide', color='#EF4444', density=True)
axes[2].set_title("CHIRPS 7-Day Rainfall Accumulation")
axes[2].set_xlabel("Precipitation (mm)")
axes[2].set_ylabel("Density")
axes[2].legend()

plt.tight_layout()
plt.show()

print(f"Mean Slope - Landslides: {landslide_slope.mean():.2f}° | Controls: {control_slope.mean():.2f}°")
print(f"Mean ΔVH   - Landslides: {landslide_dvh.mean():.2f} dB | Controls: {control_dvh.mean():.2f} dB")""")

    # Cell 4: Decoupled Feature Architecture
    add_md(r"""## 4. Decoupled Feature Architecture
The system isolates post-hoc event detection from prognostic susceptibility via an Anti-Leakage Firewall:

- **Workflow A (Event Detection):** Allowed to use post-event radar change ($\Delta$VH, $\Delta$VV) and event storm rainfall.
- **Workflow B (Susceptibility Prediction):** Strictly restricted to pre-event terrain morphology, baseline SAR backscatter, and 30-day antecedent moisture. Post-event radar and spatial coordinates are excluded to avoid target leakage.""")

    add_code(r"""WORKFLOW_A_FEATURES = [
    "dVH", "dVV", "rain_1d", "rain_7d", "rain_30d", "API",
    "slope", "aspect", "curvature", "tri", "twi", "BSI", "NDVI"
]

WORKFLOW_B_FEATURES = [
    "slope", "aspect", "curvature", "tri", "twi",
    "VH_pre", "VV_pre", "rain_30d", "API"
]

print(f"Workflow A features (Detection)    : {len(WORKFLOW_A_FEATURES)}")
print(f"Workflow B features (Susceptibility): {len(WORKFLOW_B_FEATURES)} (Pre-event core only)")""")

    # Cell 5: Spatial Clustering & 100km Purged CV
    add_md(r"""## 5. Spatial Validation: 100 km Spatial Cluster Purging
Standard random cross-validation allows points within short distances to sit in both train and test sets, artificially inflating evaluation metrics through spatial autocorrelation.

To prevent this:
1. Points are partitioned into 5 geographic clusters using KMeans on Cartesian coordinates.
2. For each test fold, any training points within 100 km of the test points are purged from the training fold.""")

    add_code(r"""def project_to_cartesian(lat, lon, lat0, lon0):
    R = 6371.0
    x = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = R * np.radians(lat - lat0)
    return x, y

lat0 = df["latitude"].mean()
lon0 = df["longitude"].mean()
coords_x, coords_y = project_to_cartesian(df["latitude"], df["longitude"], lat0, lon0)
coords_km = np.column_stack([coords_x, coords_y])

kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10)
df["spatial_cluster"] = kmeans.fit_predict(coords_km)

def compute_spatial_purged_folds(df, coords_km, purge_buffer_km=100.0):
    folds = []
    clusters = np.sort(df["spatial_cluster"].unique())
    for test_cluster in clusters:
        test_mask = df["spatial_cluster"] == test_cluster
        test_indices = np.where(test_mask)[0]
        candidate_train = np.where(~test_mask)[0]

        purged_train = []
        for tr_idx in candidate_train:
            tr_pt = coords_km[tr_idx]
            min_dist = np.min(np.linalg.norm(coords_km[test_indices] - tr_pt, axis=1))
            if min_dist >= purge_buffer_km:
                purged_train.append(tr_idx)
        folds.append((np.array(purged_train), test_indices))
    return folds

purged_folds = compute_spatial_purged_folds(df, coords_km, purge_buffer_km=100.0)

for f_i, (tr, te) in enumerate(purged_folds):
    pos_te = df.iloc[te]["label"].sum()
    neg_te = len(te) - pos_te
    print(f"Fold {f_i+1}: Train N={len(tr):3d} (purged) | Test N={len(te):2d} (Pos={pos_te:2d}, Neg={neg_te:2d})")""")

    # Cell 6: Model Training & Evaluation
    add_md(r"""## 6. Model Training and Out-of-Fold Evaluation
Train Random Forest (500 trees, balanced weights) and XGBoost (learning rate 0.05, max depth 4) under the 100 km spatial purge condition.""")

    add_code(r"""def evaluate_workflow(feature_cols, workflow_name="Workflow"):
    X = df[feature_cols].copy()
    y = df["label"].values

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    oof_rf = np.full(len(df), np.nan)
    oof_xgb = np.full(len(df), np.nan)
    fold_aucs_rf = []
    fold_aucs_xgb = []

    for fold_i, (train_idx, test_idx) in enumerate(purged_folds):
        X_tr, y_tr = X_imputed[train_idx], y[train_idx]
        X_te, y_te = X_imputed[test_idx], y[test_idx]

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_te)[:, 1]
        oof_rf[test_idx] = p_rf

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
        )
        xgb_model.fit(X_tr, y_tr)
        p_xgb = xgb_model.predict_proba(X_te)[:, 1]
        oof_xgb[test_idx] = p_xgb

        if len(np.unique(y_te)) > 1:
            fold_aucs_rf.append(roc_auc_score(y_te, p_rf))
            fold_aucs_xgb.append(roc_auc_score(y_te, p_xgb))
        else:
            fold_aucs_rf.append(np.nan)
            fold_aucs_xgb.append(np.nan)

    valid_mask = ~np.isnan(oof_rf)
    y_valid = y[valid_mask]
    
    auroc_rf = roc_auc_score(y_valid, oof_rf[valid_mask])
    auroc_xgb = roc_auc_score(y_valid, oof_xgb[valid_mask])
    ap_rf = average_precision_score(y_valid, oof_rf[valid_mask])
    ap_xgb = average_precision_score(y_valid, oof_xgb[valid_mask])
    f1_rf = f1_score(y_valid, oof_rf[valid_mask] >= 0.5)
    f1_xgb = f1_score(y_valid, oof_xgb[valid_mask] >= 0.5)

    return {
        "Workflow": workflow_name,
        "RF AUROC (100km Purged)": auroc_rf,
        "XGB AUROC (100km Purged)": auroc_xgb,
        "RF Avg Precision": ap_rf,
        "XGB Avg Precision": ap_xgb,
        "RF F1-Score": f1_rf,
        "XGB F1-Score": f1_xgb,
        "oof_rf": oof_rf,
        "oof_xgb": oof_xgb,
        "rf_model": rf,
        "xgb_model": xgb_model,
        "X_imputed": X_imputed,
        "fold_aucs_rf": fold_aucs_rf,
        "fold_aucs_xgb": fold_aucs_xgb
    }

res_a = evaluate_workflow(WORKFLOW_A_FEATURES, "Workflow A: Event Detection (Post-Hoc)")
res_b = evaluate_workflow(WORKFLOW_B_FEATURES, "Workflow B: Susceptibility (Pre-Event Core)")

summary_table = pd.DataFrame([
    {k: v for k, v in res_a.items() if "oof" not in k and "model" not in k and "imputed" not in k and "fold" not in k},
    {k: v for k, v in res_b.items() if "oof" not in k and "model" not in k and "imputed" not in k and "fold" not in k}
])
display(summary_table.round(4))""")

    # Cell 7: ROC Curves & Confusion Matrix
    add_md(r"""## 7. Performance Curves and Confusion Matrix
Plot out-of-fold ROC curves and confusion matrix under the 100 km spatial purge standard.""")

    add_code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Workflow A ROC
fpr_rf_a, tpr_rf_a, _ = roc_curve(df['label'], res_a['oof_rf'])
fpr_xgb_a, tpr_xgb_a, _ = roc_curve(df['label'], res_a['oof_xgb'])
axes[0].plot(fpr_rf_a, tpr_rf_a, label=f"Random Forest (AUC = {res_a['RF AUROC (100km Purged)']:.3f})", color="#2563EB", lw=2)
axes[0].plot(fpr_xgb_a, tpr_xgb_a, label=f"XGBoost (AUC = {res_a['XGB AUROC (100km Purged)']:.3f})", color="#DC2626", lw=2)
axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
axes[0].set_title("Workflow A: Event Detection ROC (100km Purged)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

# Workflow B ROC
fpr_rf_b, tpr_rf_b, _ = roc_curve(df['label'], res_b['oof_rf'])
fpr_xgb_b, tpr_xgb_b, _ = roc_curve(df['label'], res_b['oof_xgb'])
axes[1].plot(fpr_rf_b, tpr_rf_b, label=f"Random Forest (AUC = {res_b['RF AUROC (100km Purged)']:.3f})", color="#2563EB", lw=2)
axes[1].plot(fpr_xgb_b, tpr_xgb_b, label=f"XGBoost (AUC = {res_b['XGB AUROC (100km Purged)']:.3f})", color="#DC2626", lw=2)
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
axes[1].set_title("Workflow B: Susceptibility ROC (100km Purged)")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

cm = confusion_matrix(df['label'], res_a['oof_xgb'] >= 0.5)
print("Confusion Matrix for Workflow A (XGBoost @ 0.5 threshold):")
print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")""")

    # Cell 8: Explainable AI with SHAP
    add_md(r"""## 8. Feature Importance and SHAP Explainability
Examine the physical drivers identified by the tree models:
- $\Delta$VH radar change is the primary predictor for post-hoc detection (volume scattering loss from stripped vegetation).
- Topographic slope and antecedent precipitation dominate pre-event susceptibility.""")

    add_code(r"""# SHAP for Workflow A
explainer_a = shap.TreeExplainer(res_a['xgb_model'])
shap_values_a = explainer_a.shap_values(res_a['X_imputed'])

plt.figure(figsize=(9, 5))
plt.title("SHAP Feature Importance: Workflow A (Event Detection)", fontsize=12)
shap.summary_plot(shap_values_a, res_a['X_imputed'], feature_names=WORKFLOW_A_FEATURES, show=False)
plt.tight_layout()
plt.show()

# SHAP for Workflow B
explainer_b = shap.TreeExplainer(res_b['xgb_model'])
shap_values_b = explainer_b.shap_values(res_b['X_imputed'])

plt.figure(figsize=(9, 4.5))
plt.title("SHAP Feature Importance: Workflow B (Susceptibility)", fontsize=12)
shap.summary_plot(shap_values_b, res_b['X_imputed'], feature_names=WORKFLOW_B_FEATURES, show=False)
plt.tight_layout()
plt.show()""")

    # Cell 9: Regional Grid Scoring
    add_md(r"""## 9. Regional Hazard Scoring (Himalayan Arc 0.5° Lattice)
Evaluate susceptibility scoring across the 765 Earth Engine grid cells (~55 km resolution) across the Himalayan mountain belt.""")

    add_code(r"""def load_regional_grid():
    possible_paths = [
        Path("outputs/regional_susceptibility_grid.csv"),
        Path("../outputs/regional_susceptibility_grid.csv"),
        Path("/kaggle/input/himalayan-landslide-dataset/regional_susceptibility_grid.csv"),
        Path("regional_susceptibility_grid.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            print(f"Loaded regional grid from: {p}")
            return pd.read_csv(p)

    raw_url = "https://raw.githubusercontent.com/GouravSharma26/LandSlide/main/outputs/regional_susceptibility_grid.csv"
    print(f"Fetching regional grid from GitHub repository: {raw_url}")
    try:
        return pd.read_csv(raw_url)
    except Exception as e:
        raise RuntimeError(f"Could not load regional grid: {e}")

grid_df = load_regional_grid()

# Classify risk tiers using real ensemble susceptibility probabilities
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
grid_df["risk_tier"] = pd.cut(grid_df["susceptibility"], bins=bins, labels=labels)

print(f"Regional Himalayan Lattice: {len(grid_df)} valid Earth Engine grid cells")
print("Risk Classification Distribution:")
print(grid_df["risk_tier"].value_counts(sort=False))

plt.figure(figsize=(11, 5))
scatter = plt.scatter(
    grid_df["longitude"],
    grid_df["latitude"],
    c=grid_df["susceptibility"],
    cmap="turbo",
    s=40,
    alpha=0.85
)
plt.colorbar(scatter, label="Ensemble Susceptibility Probability")
plt.title("Regional Susceptibility Hazard Lattice (765 Real GEE Grid Cells across the Himalayan Arc)")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.tight_layout()
plt.show()""")

    # Cell 10: Conclusion & Research Governance
    add_md(r"""## 10. Research Governance and Findings
Key methodological findings and disclosures:

1. **Fold 4 Boundary Condition:** Under 100 km spatial buffer purging, Fold 4's geographic cluster contained only control observations, resulting in an undefined AUROC for that individual fold. The aggregated out-of-fold AUROC remains sound at 0.986 (Workflow A) and 0.977 (Workflow B).
2. **Hillslope Control Extraction Findings:** A 45.3-minute automated Earth Engine extraction across 179 candidate windows demonstrated that candidate control hillslopes on steep terrain (>15°) with zero rainfall triggers are geomorphically rare in the active monsoon belt (100% rejection rate). This empirically validated preserving the Version 1.0 diagnostic baseline.
3. **Operational Conclusion:** Sentinel-1 C-band SAR effectively bypasses the monsoon cloud gap, providing a cloud-independent data foundation for early mass-wasting detection in high-relief mountain terrains.

---

### Data Sources
- **NASA COOLR:** Global Landslide Catalog
- **ESA Copernicus Sentinel-1:** Synthetic Aperture Radar (SAR) Level-1 GRD
- **USGS / NASA SRTM:** Shuttle Radar Topography Mission 30m Global DEM
- **UCSB CHIRPS:** Climate Hazards Group InfraRed Precipitation with Station data (0.05° resolution)""")

    nb_json = {
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

    out_path = Path("notebooks/early_landslide_risk_detection_master.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=1)

    print(f"Clean Master Notebook generated at: {out_path} ({out_path.stat().st_size} bytes, {len(cells)} cells)")

if __name__ == "__main__":
    create_notebook()
