# Himalayan Landslide Research: Multi-Source Remote Sensing for Event Detection and Susceptibility Analytics

An applied geospatial machine-learning research system for rainfall-triggered landslide analysis across the Himalayan mountain belt ($26.0^\circ\text{N} - 35.5^\circ\text{N}$, $74.0^\circ\text{E} - 97.0^\circ\text{E}$, $\sim 2.2\text{M km}^2$).

The project implements two decoupled machine-learning workflows:
1. **Workflow A (Event Detection):** Post-hoc disturbance classification using pre/post Sentinel-1 SAR change, event precipitation, and terrain derivatives.
2. **Workflow B (Susceptibility Prediction):** Pre-event spatial hazard modeling using static terrain and antecedent precipitation, evaluated against a diagnostic baseline.

---

## Research Scope and Problem Formulation

Rainfall-triggered mass wasting in the Himalayas is driven by steep, seismically active topography interacting with heavy South Asian monsoon precipitation. To avoid methodological conflation and temporal leakage, the two problems are separated:

### Workflow A: Landslide Event Detection
- **Classification Objective:** Determine whether satellite observations at a reported location indicate ground failure following a triggering event window.
- **Scientific Classification:** Experimental Point-Zone Disturbance Assessment.
- **Permitted Predictors (13 features):** Sentinel-1 IW GRD backscatter ($\text{VH}_{\text{post}}, \text{VV}_{\text{post}}, \Delta\text{VH}, \Delta\text{VV}$), event-window rainfall ($\text{rain}_{1\text{d}}, \text{rain}_{7\text{d}}, \text{rain}_{30\text{d}}, \text{API}_{\alpha=0.90}$), and SRTM 30m topography (elevation, slope, aspect, curvature, TWI).
- **Positional Uncertainty Limitation:** 57.5% of NASA COOLR catalog reports ($73/127$) carry $\pm 5\text{ km}$ location error. Point extractions represent point-zone disturbance proxies, not polygon-verified scar delineations.

### Workflow B: Rainfall-Conditioned Susceptibility Prediction
- **Classification Objective:** Rank pre-event spatial proneness using static topography and antecedent precipitation.
- **Scientific Classification:** Rainfall-Conditioned Susceptibility Diagnostic Baseline.
- **Permitted Predictors (10 features in Core):** SRTM topography (elevation, slope, aspect, curvature, TRI, TWI), pre-event SAR backscatter ($\text{VH}_{\text{pre}}, \text{VV}_{\text{pre}}$), 30-day antecedent rainfall ($\text{rain}_{30\text{d}}$), and Antecedent Precipitation Index ($\text{API}_{\alpha=0.90}$).
- **Strictly Prohibited:** Post-event backscatter ($\text{VH}_{\text{post}}, \text{VV}_{\text{post}}$), differential change ($\Delta\text{VH}, \Delta\text{VV}$), trigger rainfall ($\text{rain}_{1\text{d}}, \text{rain}_{7\text{d}}$), event labels, and spatial coordinates ($\text{lat}, \text{lon}$).
- **Configurations Evaluated:**
  1. *Cloud-Independent Core (10 features):* All-weather SAR, topography, and antecedent rainfall.
  2. *Optical-Enhanced (13 features):* Adds pre-event Sentinel-2 NDVI, NDWI, and BSI (carries 31.5% monsoon cloud missingness).
  3. *Reduced Diagnostic (8 features):* Excludes slope and TRI to evaluate non-slope topographic signals.

---

## Data Sources and Repository Datasets

The research framework synthesizes satellite remote sensing, digital elevation topography, and high-resolution precipitation observations across five authoritative sources:

### 1. NASA Global Landslide Catalog (GLC / COOLR)
- **Provider:** NASA Goddard Space Flight Center (Cooperative Open Online Landslide Repository)
- **Coverage & Timeframe:** Global records filtered to the Himalayan study envelope ($26.0^\circ\text{N} - 35.5^\circ\text{N}$, $74.0^\circ\text{E} - 97.0^\circ\text{E}$) spanning **2017 to 2020**.
- **Extracted Attributes:** Verified historical landslide events, geographic coordinates, occurrence timestamps, positional confidence tiers ($\le 5\text{ km}$ accuracy retained), and reported triggering mechanisms (`monsoon`, `downpour`, `continuous_rain`).

### 2. Copernicus Sentinel-1 Synthetic Aperture Radar (SAR)
- **Provider:** European Space Agency (ESA) / Copernicus Programme
- **Sensor & Mode:** C-Band SAR, Level-1 Ground Range Detected (GRD), Interferometric Wide (IW) swath mode, 10-meter spatial resolution, 12-day repeat orbit.
- **Scientific Role:** Provides all-weather, day-and-night surface sensing independent of cloud cover. Captures dual-polarization backscatter intensity ($\text{VV}, \text{VH}$) in ascending and descending passes before and after triggering events. Computes differential backscatter change ($\Delta\text{VH}, \Delta\text{VV}$) to capture physical surface roughness disruption, vegetation removal, and soil moisture shifts associated with mass movement.

### 3. Copernicus Sentinel-2 Multi-Spectral Instrument (MSI)
- **Provider:** European Space Agency (ESA) / Copernicus Programme
- **Sensor Specifications:** Multi-spectral surface reflectance across 13 spectral bands at 10 m and 20 m spatial resolution.
- **Scientific Role:** Measures vegetative vigor, surface wetness, and soil exposure using cloud-masked optical scenes. Extracted indices include Normalized Difference Vegetation Index ($\text{NDVI} = \frac{B8 - B4}{B8 + B4}$), Normalized Difference Water Index ($\text{NDWI} = \frac{B3 - B8}{B3 + B8}$), and Bare Soil Index ($\text{BSI} = \frac{(B11 + B4) - (B8 + B2)}{(B11 + B4) + (B8 + B2)}$).

### 4. SRTM GL1 & MERIT Hydro Digital Elevation Models
- **Provider:** NASA / USGS / JAXA / University of Tokyo
- **Resolution:** 30-meter global 1-arc-second SRTM elevation paired with 90-meter MERIT Hydro flow routing.
- **Scientific Role:** Provides topographic morphometry and terrain hydrology. Extracted parameters include elevation (meters), slope angle (degrees), aspect direction (compass degrees), profile curvature ($\text{m}^{-1}$ via $3\times 3$ Laplacian kernel), Terrain Ruggedness Index ($\text{TRI}$), and Topographic Wetness Index ($\text{TWI} = \ln(\text{upa} / \tan(\text{slope}))$).

### 5. CHIRPS Daily Gridded Precipitation
- **Provider:** Climate Hazards Center (CHC), University of California, Santa Barbara (UCSB)
- **Resolution:** 0.05° (~5.5 km) daily gridded rainfall estimates.
- **Scientific Role:** Merges thermal infrared satellite observations with ground-station gauge data to calculate meteorological triggers. Computes short-term triggering totals ($1\text{-day}$ peak rain, $7\text{-day}$ cumulative) and antecedent soil saturation metrics ($30\text{-day}$ cumulative rain, and the Antecedent Precipitation Index $\text{API}_t = \sum_{i=1}^{30} P_{t-i} \cdot 0.90^i$).

### Repository Dataset Structure

| Dataset Name | File Path | Records | Description |
| :--- | :--- | :---: | :--- |
| **Raw Landslide Inventory** | `Landslide_Reports.csv` | 14,750 | Global landslide database from NASA COOLR |
| **Observation Records (Events & Controls)** | `outputs/phase2_cleaned_dataset.csv` | 254 | 127 verified events and 127 paired diagnostic controls |
| **Multi-Source Feature Matrix** | `outputs/phase3_feature_table.csv` | 254 | Unified table with 28 topographic, radar, optical, and rainfall features |
| **Model Performance Metrics** | `outputs/phase4_metrics_comparison.csv` | — | Cross-validation comparison between core and reduced models |
| **Spatial Out-of-Fold Predictions** | `outputs/phase4_oof_predictions.csv` | 254 | Spatial cross-validation probabilities and fold assignments |
| **Regional Susceptibility Grid (0.5°)** | `outputs/phase5_susceptibility_grid.csv` | 765 | 0.5° regional hazard scoring across the entire Himalayan arc |

---

## Dataset Lineage and Methodological Limitations

| Component | Status Designation | Description and Methodological Caveat |
| :--- | :--- | :--- |
| **Version 1.0 Dataset** | **Historical Diagnostic Baseline** | 254 points ($127$ positive events from NASA COOLR paired with $127$ unstratified negative controls). Controls were sampled from flat terrain (mean slope $1.43^\circ \pm 4.34^\circ$ vs $21.83^\circ \pm 11.99^\circ$ for positive events). This serves as a diagnostic baseline; **it is not a validated operational susceptibility model**. |
| **Version 2.0 Dataset** | **EXTRACTION_PENDING_COLAB_GEE** | Geomorphometrically constrained hillslope sampling protocol ($6.1^\circ \le \text{slope} \le 41.4^\circ$, $>2\text{ km}$ buffer exclusion, permanent water and snow masks, SAR temporal consistency screening). Specification complete and Colab notebook prepared; **extraction has not yet been executed in authenticated Google Earth Engine**. |
| **Workflow A Models** | **Point-Zone Disturbance Assessment** | Post-hoc classification under catalog coordinate uncertainty. Single-pixel $\Delta\text{VH}$ features capture local zone disturbance rather than verified failure scars. |
| **Workflow B Models** | **Diagnostic Susceptibility** | Conditioned on antecedent CHIRPS precipitation. Evaluated against V1 diagnostic controls; high metrics partly reflect mountain-versus-plains separation. |
| **Regional Grid (0.5°)** | **Macro-Regional Illustration** | $0.5^\circ \times 0.5^\circ$ grid (765 valid cells, ~55 km width) conditioned on a September 2026 synoptic late-monsoon window. Designed for regional basin comparison; **not a local operational hazard map**. |

---

## Validation Strategy: Spatially Purged Cross-Validation

To prevent spatial auto-correlation from inflating model evaluation, the dataset is evaluated using 5-fold spatial cluster cross-validation:
1. **Spatial Partitioning:** Points are projected onto Azimuthal Equidistant metric coordinates and partitioned into 5 geographic clusters using KMeans.
2. **100 km Spatial Purge:** For each fold, any training sample located within 100 km of any test sample in that fold is excluded from training. Retained training set size ranges from 189 to 217 samples (actual minimum test-to-train distance $\ge 101.8\text{ km}$).
3. **Fold 4 Disclosure:** Fold 4 corresponds to the northwestern Kashmir cluster and contains 37 negative controls and 0 positive events in its test set. Fold-level AUROC and PR-AUC are mathematically undefined ($NaN$) for Fold 4. Pooled out-of-fold metrics evaluate all 254 test predictions globally.

---

## Model Evaluation Results

The following metrics reflect the authoritative artifact build (`artifacts/metadata/experiment_metadata.json`, `artifacts/detection/detection_metrics.csv`, `artifacts/susceptibility/susceptibility_metrics.csv`):

### Workflow A: Landslide Event Detection (100 km Purged Outer CV)
| Classifier | Outer Validation Strategy | AUROC | PR-AUC | F1-Score ($\tau=0.50$) | Accuracy ($\tau=0.50$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest (500 Trees)** | 100 km Spatial Purge | **0.984** | **0.989** | **0.972** | **97.2%** |
| **XGBoost (Hist Gradient)** | 100 km Spatial Purge | **0.968** | **0.980** | **0.956** | **95.7%** |
| Random Forest (Unpurged Baseline) | Spatial Split (min separation 38.7 km) | 0.986 | 0.990 | 0.976 | 97.6% |
| XGBoost (Unpurged Baseline) | Spatial Split (min separation 38.7 km) | 0.970 | 0.981 | 0.956 | 95.7% |

### Workflow B: Susceptibility Prediction (100 km Purged Outer CV)
| Configuration | Features | Model | AUROC | PR-AUC | F1-Score ($\tau=0.50$) | Accuracy ($\tau=0.50$) | Missingness Handling |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Cloud-Independent Core** (Recommended) | Topo + SAR pre + Rain 30d + API (10 feats) | Random Forest | **0.973** | **0.981** | **0.938** | **94.1%** | Fold-isolated median imputation |
| **Cloud-Independent Core** (Recommended) | Topo + SAR pre + Rain 30d + API (10 feats) | XGBoost | **0.965** | **0.978** | **0.944** | **94.5%** | Native directional routing |
| **Optical-Enhanced** | Core + NDVI, NDWI, BSI (13 feats) | Random Forest | 0.977 | 0.983 | 0.930 | 93.3% | Fold-isolated median imputation |
| **Optical-Enhanced** | Core + NDVI, NDWI, BSI (13 feats) | XGBoost | 0.965 | 0.978 | 0.948 | 94.9% | Native directional routing (31.5% nulls) |
| **Reduced Diagnostic** | Core excluding Slope & TRI (8 feats) | Random Forest | 0.955 | 0.958 | 0.878 | 87.4% | Fold-isolated median imputation |
| **Reduced Diagnostic** | Core excluding Slope & TRI (8 feats) | XGBoost | 0.959 | 0.961 | 0.889 | 89.0% | Native directional routing |

*Key Finding on Optical Features:* Adding optical indices provides minimal gain (+0.004 AUROC in RF; -0.0003 in XGBoost) while introducing 31.5% missingness due to monsoon cloud cover. The Cloud-Independent Core model is the more dependable operational setup.

---

## Interactive Dashboard Structure

The research dashboard is organized into seven analytical sections:

1. **Overview (`dashboard/pages/overview.py`):** Study area context, research workflows, temporal event distribution (2017–2020), monthly monsoon seasonality, and primary reported triggers.
2. **Study Area (`dashboard/pages/geographic_explorer.py`):** Interactive spatial map of positive events, controls, and 0.5° macro-grid.
3. **Data Quality (`dashboard/pages/data_quality.py`):** 7-stage preprocessing funnel ($14,750 \rightarrow 127$), missing values by feature with optical co-dependence explanation, and Version 1.0 vs Version 2.0 control sample comparison.
4. **Environmental Analysis (`dashboard/pages/environmental_analysis.py`):** Feature distributions across precipitation, topography, SAR backscatter, and optical domains, with correlation heatmaps and Variance Inflation Factor (VIF) tables.
5. **Event Detection (`dashboard/pages/event_detection.py`):** Fold-wise AUROC/PR-AUC reporting with Fold 4 disclosure, decision threshold slider with confusion matrix, out-of-fold curves, and differential radar diagnostic plots.
6. **Susceptibility Prediction (`dashboard/pages/susceptibility_prediction.py`):** Comparison of Core, Optical-Enhanced, and Reduced models, alongside the 0.5° regional lattice explorer and model disagreement analysis.
7. **Feature Importance (`dashboard/pages/explainability_uncertainty.py`):** TreeSHAP feature attributions and spatial purging sensitivity curves ($0, 50, 100, 200, 437.8\text{ km}$).

---

## Quickstart

### Prerequisites
- Python 3.11 or 3.12 (validated on Python 3.12.6)
- Dependencies: `streamlit`, `scikit-learn`, `xgboost`, `shap`, `plotly`, `pydeck`, `pyproj`, `scipy`, `pandas`, `numpy`, `joblib`

### Installation
```bash
pip install -r requirements.txt
```

### Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
The application will open at `http://localhost:8501`.

### Run the Test Suite
```bash
python -m pytest tests/ -v
```
All 59 unit, geomorphometric, mapping, governance, and schema tests must pass.

---

## Streamlit Cloud Deployment

This dashboard is pre-configured for one-click deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **New app** and select:
   - **Repository:** `GouravSharma26/LandSlide`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
3. Click **Deploy**.

All required datasets and models are pre-packaged in `outputs/` and `artifacts/`. No external database, Google Earth Engine credentials, or private API keys are required for the web application.

---

## Automated Testing Suite

The repository contains 59 automated tests in `tests/`:
- `tests/test_scientific_governance.py` (7 tests): Prohibited-feature enforcement, temporal leakage checks, spatial purge distances, fold balance, artifact schemas, and model deserialization.
- `tests/test_v2_sampling_and_features.py` (28 tests): MERIT Hydro 90m TWI, slope aggregation, positive-event validation, multi-batch candidate optimization, coordinate deduplication, 13 rejection categories, and Colab notebook structure.
- `tests/test_maps.py` (8 tests): Geospatial map rendering, observation filter routing, attribute selection, and Plotly 6 MapLibre compatibility simulation.
- `tests/test_data_loader.py` (7 tests): Data loaders for raw reports, cleaned subsets, feature tables, metrics, predictions, regional grids, and models.
- `tests/test_metrics.py` (5 tests): ROC curves, PR curves, confusion matrix calculations across decision thresholds, susceptibility distribution statistics, and model comparison bar chart schema regression test.
- `tests/test_validation.py` (4 tests): Geographic coordinate bounds ($26.0^\circ-35.5^\circ\text{N}, 74.0^\circ-97.0^\circ\text{E}$), schema compliance, missingness summaries, and label balance.
