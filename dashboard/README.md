# Himalayan Landslide Intelligence Dashboard

A modern, high-fidelity geospatial intelligence platform for rainfall-triggered landslide detection, spatial cross-validation evaluation, and regional susceptibility analytics across the Himalayan mountain belt.

---

## 🏔️ Overview

The **Himalayan Landslide Intelligence Dashboard** is a full-featured interactive analytics platform built with Streamlit, Plotly, PyDeck, Scikit-learn, XGBoost, and SHAP. It visualizes data from NASA's Global Landslide Catalog (COOLR), ESA Sentinel-1 SAR (IW GRD), ESA Sentinel-2 MSI surface reflectance, CHIRPS daily precipitation, and USGS SRTM 30m digital elevation derivatives across the Himalayan belt ($26.0^\circ\text{N} - 35.0^\circ\text{N}$, $74.0^\circ\text{E} - 97.0^\circ\text{E}$).

---

## 🚀 Quickstart

### 1. Requirements & Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 2. Installation
Install project dependencies from `dashboard/requirements.txt`:

```bash
pip install -r dashboard/requirements.txt
```

### 3. Launch Application
Run the dashboard locally from the workspace root:

```bash
streamlit run dashboard/app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 🧭 Dashboard Architecture & Pages

The application is structured into seven modular analytical views accessible via the dark navy sidebar:

1. **Executive Overview (`pages/overview.py`):**
   - High-level KPIs: Raw reports (14,750), cleaned observations (254), model AUROC (0.976), 0.5° grid cells (765), and date ranges.
   - Annual event trends, monthly monsoon sensitivity charts, trigger distribution, and benchmark model comparison scorecards.

2. **Geographic Explorer (`pages/geographic_explorer.py`):**
   - Interactive spatial mapping using Plotly Mapbox and PyDeck 3D column layers.
   - Landslide positive events vs. hard control negatives with hover tooltips and regional subzone filtering.

3. **Landslide Data Explorer (`pages/data_explorer.py`):**
   - Interactive search and query engine over NASA COOLR reports and cleaned observations.
   - Multi-attribute filtering (trigger, accuracy, source catalog) and CSV download capabilities.

4. **Environmental Analysis (`pages/environmental_analysis.py`):**
   - Multisource feature distributions across 4 key earth observation domains:
     - *Precipitation:* CHIRPS 1-day, 7-day, 30-day cumulative rainfall and Antecedent Precipitation Index (API).
     - *Topography:* SRTM 30m Elevation, Slope, Aspect, Curvature, TRI, and TWI.
     - *SAR Radar:* Sentinel-1 VV/VH backscatter and pre/post differential backscatter ($\Delta\text{VV}$, $\Delta\text{VH}$).
     - *Optical:* Sentinel-2 surface reflectance NDVI, NDWI, and Bare Soil Index (BSI).
   - Interactive Pearson correlation heatmap and transparent missingness visualization.

5. **Machine Learning Performance (`pages/model_performance.py`):**
   - Spatial 5-fold cross-validation benchmarking (Azimuthal Equidistant KMeans clustered folds).
   - Out-of-fold ROC curves, Precision-Recall curves, confusion matrices with adjustable decision thresholds.
   - Global feature importance (Random Forest MDI vs. XGBoost Gain) and TreeExplainer SHAP attribution.

6. **Susceptibility Analysis (`pages/susceptibility_analysis.py`):**
   - Macro-scale $0.5^\circ \times 0.5^\circ$ (~55 km) regional susceptibility grid.
   - Dual model outputs (RF and XGBoost) and ensemble susceptibility score.
   - Spatial uncertainty diagnostics: RF tree standard deviation and inter-model disagreement ($|\text{RF} - \text{XGB}|$).
   - Susceptibility zonation classification (Very Low to Very High).

7. **Data Quality & Methodology (`pages/data_quality.py`):**
   - Transparent missingness audit (documenting 31.5% optical monsoon cloud obscuration).
   - Step-by-step cleaning funnel from 14,750 raw records down to 127 positive events.
   - Honest documentation of the negative-sampling terrain confound and sample size limitations.
   - Operational safety disclaimer (not a real-time warning system).

---

## 🧪 Automated Testing

Execute the test suite to verify data loading, bounding box constraints, schema validation, and metric consistency:

```bash
pytest tests/ -v
```

---

## 🛡️ Scientific Integrity & Provenance

- **Terrain Confound:** In Phase 2, negative samples were constrained to slope $< 5^\circ$. Consequently, models easily differentiate steep terrain from flat terrain. The *Reduced Feature Set* (excluding slope and TRI) is presented as the primary benchmark.
- **Resolution Honesty:** The regional susceptibility grid is evaluated at $0.5^\circ$ (~55 km resolution) across 765 valid cells, suitable for macro-basin prioritization, not 30m slope-scale site hazard mapping.
- **Zero Fabrication:** All metrics, coordinates, predictions, and feature distributions are computed directly from the project's source datasets. Missing data is never silently filled with synthetic values.

