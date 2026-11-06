# Himalayan Landslide Intelligence Research System

A reproducible geospatial machine-learning research platform for multi-source remote sensing landslide detection, spatial cross-validation evaluation, and rainfall-conditioned susceptibility prediction across the Himalayan mountain belt.

---

## 🏔️ Research Scope & Objectives

The system supports two independent, decoupled machine-learning workflows across the Himalayan mountain belt ($26.0^\circ\text{N} - 35.5^\circ\text{N}$, $74.0^\circ\text{E} - 97.0^\circ\text{E}$, $\sim 2.2\text{M km}^2$):

1. **Workflow A: Landslide Event Detection**
   - *Status:* Experimental Point-Zone Disturbance Assessment.
   - *Objective:* Post-hoc diagnostic classification of whether ground failure evidence exists using pre/post satellite observations.
   - *Allowed Predictors:* Sentinel-1 IW GRD ($\text{VH}_{\text{post}}, \text{VV}_{\text{post}}, \Delta\text{VH}, \Delta\text{VV}$), event trigger rainfall ($\text{rain}_{1\text{d}}, \text{rain}_{7\text{d}}, \text{rain}_{30\text{d}}, \text{API}_{\alpha=0.90}$), and SRTM topography.
   - *Positional Uncertainty Notice:* $57.5\%$ of NASA COOLR events ($73/127$) carry $\pm 5\text{ km}$ positional uncertainty. Single 10m pixels represent point-zone disturbance proxies, not verified scar-level delineations.

2. **Workflow B: Rainfall-Conditioned Susceptibility Prediction**
   - *Status:* Rainfall-Conditioned Susceptibility — Diagnostic Baseline.
   - *Objective:* Pre-event spatial hazard ranking conditioned on antecedent precipitation.
   - *Strictly Prohibited:* $\text{VH}_{\text{post}}, \text{VV}_{\text{post}}, \Delta\text{VH}, \Delta\text{VV}$, event-day rain ($\text{rain}_{1\text{d}}, \text{rain}_{7\text{d}}$), event labels, and spatial coordinates ($\text{lat}, \text{lon}$). Zero temporal leakage.
   - *Configurations:* Cloud-Independent Core (all-weather), Optical-Enhanced (carries 31.5% monsoon cloud missingness), and Reduced Diagnostic (slope & TRI excluded).

---

## 🚀 Quickstart

### 1. Requirements & Prerequisites
- Python 3.11+ (Validated on Python 3.12.6)
- Dependencies: `streamlit`, `scikit-learn`, `xgboost`, `shap`, `plotly`, `pydeck`, `pyproj`, `scipy`, `pandas`, `numpy`, `joblib`

```bash
pip install -r dashboard/requirements.txt
```

### 2. Launch Research Dashboard
```bash
streamlit run dashboard/app.py
```
The application will be accessible at `http://localhost:8501`.

### 3. Automated Test Suite
```bash
python -m pytest tests/ -v
```
All 50 unit, spatial separation, prohibited feature, geomorphometric, and governance tests must pass.

---

## 🧭 Dashboard Architecture (Eight Research Sections)

1. **Executive Overview (`pages/overview.py`):**
   - High-level KPIs: Raw reports (14,750), cleaned paired points (254), confirmed rainfall events (127), and 0.5° regional grid cells (765).
   - Dual-workflow architecture cards with scientific status badges.
   - Annual event trends, monthly monsoon peak sensitivity, and trigger distributions.

2. **Geographic Explorer (`pages/geographic_explorer.py`):**
   - Interactive spatial mapping using Plotly Mapbox and PyDeck 3D column layers.
   - Confirmed landslide locations vs. diagnostic control negatives with hover tooltips and regional subzone filtering.
   - Overlay toggle for the 0.5° regional susceptibility macro-grid (765 cells).

3. **Data Catalog & Quality (`pages/data_quality.py`):**
   - 7-Stage Preprocessing Funnel (from 14,750 global reports to 127 catalogued Himalayan events meeting the stated filtering criteria).
   - Missingness root-cause audit (documenting 31.5% optical monsoon cloud obscuration).
   - Dataset Lineage: Version 1.0 (Baseline) vs Version 2.0 (Hillslope Protocol, $6.1^\circ-41.4^\circ$, $>2\text{ km}$ buffer exclusion).
   - Two-sample Kolmogorov-Smirnov statistical proof of flat-terrain control bias ($p < 10^{-50}$).

4. **Environmental Analysis (`pages/environmental_analysis.py`):**
   - Multisource feature distributions across 4 key earth observation domains:
     - *Precipitation:* CHIRPS 1-day, 7-day, 30-day cumulative rainfall and API ($\alpha=0.90$).
     - *Topography:* SRTM 30m Elevation, Slope, Aspect, Curvature, TRI, and TWI.
     - *SAR Backscatter:* Sentinel-1 VV/VH backscatter and pre/post differential backscatter ($\Delta\text{VV}, \Delta\text{VH}$).
     - *Optical Indices:* Sentinel-2 surface reflectance NDVI, NDWI, and Bare Soil Index (BSI).
   - Pearson correlation heatmap and descriptive Variance Inflation Factor (VIF) audit.

5. **Event Detection (`pages/event_detection.py`):**
   - Workflow A: Experimental Point-Zone Disturbance Assessment.
   - Fold-wise metric reporting with full scientific transparency (explaining Fold 4 undefined AUROC due to single-class Kashmir test partition).
   - Interactive confusion matrix with adjustable probability threshold slider.
   - Out-of-fold ROC curves, Precision-Recall curves, and differential radar disturbance diagnostics.

6. **Susceptibility Prediction (`pages/susceptibility_prediction.py`):**
   - Workflow B: Rainfall-Conditioned Susceptibility — Diagnostic Baseline.
   - Multi-configuration selector: Cloud-Independent Core (Recommended) vs Optical-Enhanced vs Reduced Diagnostic.
   - 0.5° Regional Susceptibility Lattice Explorer (765 cells, September 2026 synoptic late-monsoon reference period).
   - Inter-model disagreement diagnostics ($|\text{RF} - \text{XGB}|$).

7. **Explainability & Uncertainty (`pages/explainability_uncertainty.py`):**
   - Global TreeSHAP feature attributions for Random Forest and XGBoost.
   - Explanation of why radar deltas ($\Delta\text{VH}$) show modest attribution due to catalog coordinate uncertainty.
   - Spatial Purging Sensitivity Analysis ($0, 50, 100, 200, 437.8\text{ km}$) showing trade-offs between geographic separation and sample loss.

8. **Research Methodology & Governance (`pages/methodology_governance.py`):**
   - Mathematical formulations of Workflow A and Workflow B.
   - Strict Prohibited Feature Contracts (banning post-event data and coordinates).
   - Spatial 5-Fold Cluster Cross-Validation architecture diagram.
   - Comprehensive limitation disclosures and one-click download for full `experiment_metadata.json`.

---

## 📊 Authoritative Model Performance & Benchmark Results

The following metrics reflect the authoritative artifact build (`artifacts/metadata/experiment_metadata.json`, `artifacts/detection/detection_metrics.csv`, `artifacts/susceptibility/susceptibility_metrics.csv`), evaluated across 5-fold spatial cluster cross-validation with a strict 100 km spatial purge:

### Workflow A: Landslide Event Detection (Experimental Point-Zone Disturbance Assessment)
| Classifier | Outer Validation Strategy | AUROC | PR-AUC | F1-Score (th=0.50) | Accuracy (th=0.50) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest (500 Trees)** | 100 km Spatial Purge | **0.984** | **0.989** | **0.972** | **97.2%** |
| **XGBoost (Hist Gradient)** | 100 km Spatial Purge | **0.968** | **0.980** | **0.956** | **95.7%** |
| Random Forest (Unpurged Baseline) | Unpurged Spatial Split (min 38.7 km) | 0.986 | 0.990 | 0.976 | 97.6% |
| XGBoost (Unpurged Baseline) | Unpurged Spatial Split (min 38.7 km) | 0.970 | 0.981 | 0.956 | 95.7% |

*Fold 4 Disclosure:* Fold 4 corresponds to the northwestern Kashmir/Ladakh cluster and contains $N=37$ negative points and $0$ positive events in its test partition. Fold-level AUROC and PR-AUC are mathematically undefined ($NaN$) for Fold 4; pooled out-of-fold metrics evaluate all 254 test predictions globally.

### Workflow B: Rainfall-Conditioned Susceptibility Prediction (Historical Diagnostic Baseline)
| Model Configuration | Feature Set | Model | AUROC | PR-AUC | F1-Score (th=0.50) | Accuracy (th=0.50) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Cloud-Independent Core** (Recommended) | Pre-event Topo + SAR pre + Rain 30d + API (10 feats) | Random Forest | **0.973** | **0.981** | **0.938** | **94.1%** |
| **Cloud-Independent Core** (Recommended) | Pre-event Topo + SAR pre + Rain 30d + API (10 feats) | XGBoost | **0.965** | **0.978** | **0.944** | **94.5%** |
| **Optical-Enhanced** | Core + Pre-event NDVI, NDWI, BSI (13 feats, 31.5% nulls) | Random Forest | 0.977 | 0.983 | 0.930 | 93.3% |
| **Optical-Enhanced** | Core + Pre-event NDVI, NDWI, BSI (13 feats, 31.5% nulls) | XGBoost | 0.965 | 0.978 | 0.948 | 94.9% |
| **Reduced Diagnostic** | Core excluding Slope & TRI (8 feats; terrain check) | Random Forest | 0.955 | 0.958 | 0.878 | 87.4% |
| **Reduced Diagnostic** | Core excluding Slope & TRI (8 feats; terrain check) | XGBoost | 0.959 | 0.961 | 0.889 | 89.0% |

---

## 🛡️ Scientific Integrity & Provenance Status

| Artifact / Workflow | Status Designation | Description & Caveat |
| :--- | :--- | :--- |
| **Version 1.0 Dataset** | **Historical Diagnostic Baseline** | $N=254$ historical points with flat-terrain negative controls (mean slope $1.43^\circ \pm 4.34^\circ$). Evaluated as diagnostic baseline; **not validated susceptibility**. |
| **Version 2.0 Dataset** | **EXTRACTION_PENDING_COLAB_GEE** | Empirically validated hillslope protocol ($6.1^\circ \le \text{slope} \le 41.4^\circ$, $>2\text{ km}$ exclusion). Extraction pending authenticated Google Earth Engine execution in Colab. |
| **Workflow A** | **Experimental Point-Zone Disturbance** | Post-hoc classification under $\pm 5\text{ km}$ catalog uncertainty. Point pixels capture point-zone disturbance proxies, not verified scars. |
| **Workflow B** | **Rainfall-Conditioned Susceptibility** | Pre-event hazard ranking conditioned on antecedent CHIRPS rainfall. Evaluated on Version 1.0 diagnostic controls. |
| **Regional Grid** | **Macro-Regional Illustration** | $0.5^\circ \times 0.5^\circ$ macro-lattice (765 cells) under September 2026 synoptic reference period. Macro-regional illustration, **not a local operational hazard map**. |

---

## 🧪 Automated Testing Suite

The repository contains 50 automated tests across 5 test suites under `tests/`:
- `tests/test_scientific_governance.py` (7 tests): Prohibited features enforcement, temporal leakage absence, spatial purge distance verification, fold class balance, artifact schema consistency, model deserialization, and V2 status.
- `tests/test_v2_sampling_and_features.py` (28 tests): MERIT Hydro 90m TWI, SRTM slope aggregation, strict positive event terrain validation (zero fallback), multi-batch candidate optimization, coordinate deduplication, 13-category rejection audit counters, and notebook structural integrity.
- `tests/test_data_loader.py` (7 tests): Cached loading of raw reports, cleaned datasets, feature tables, metrics, out-of-fold predictions, 765-cell regional grid, and serialized models.
- `tests/test_metrics.py` (4 tests): ROC curves, Precision-Recall curves, confusion matrices across probability thresholds, and susceptibility distribution statistics.
- `tests/test_validation.py` (4 tests): Geographic bounds validation ($26.0^\circ-35.5^\circ\text{N}, 74.0^\circ-97.0^\circ\text{E}$), schema compliance, missingness summary calculation, and label balance.
