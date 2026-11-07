# Himalayan Landslide Intelligence: Multi-Source Remote Sensing for Landslide Event Detection and Rainfall-Conditioned Susceptibility Analytics

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

1. **Executive Overview (`dashboard/pages/overview.py`):** Key metrics, annual event distribution (2017–2020), monthly monsoon seasonality, and workflow status cards.
2. **Geographic Explorer (`dashboard/pages/geographic_explorer.py`):** Interactive Mapbox and PyDeck views of positive events, controls, and regional macro-grid overlays.
3. **Data Catalog & Quality (`dashboard/pages/data_quality.py`):** 7-stage inventory filtering funnel ($14,750 \rightarrow 127$), optical missingness audit, V1 flat-terrain Kolmogorov-Smirnov test ($p < 10^{-50}$), and V2 sampling specification.
4. **Environmental Analysis (`dashboard/pages/environmental_analysis.py`):** Feature distributions across precipitation, topography, SAR backscatter, and optical domains, with correlation heatmaps and Variance Inflation Factor (VIF) tables.
5. **Event Detection (`dashboard/pages/event_detection.py`):** Fold-wise AUROC/PR-AUC reporting, threshold slider with confusion matrix, out-of-fold curves, and differential radar diagnostic plots.
6. **Susceptibility Prediction (`dashboard/pages/susceptibility_prediction.py`):** Comparison of Core, Optical-Enhanced, and Reduced models, alongside the 0.5° regional lattice explorer and model disagreement analysis.
7. **Explainability & Uncertainty (`dashboard/pages/explainability_uncertainty.py`):** TreeSHAP feature attributions and spatial purging sensitivity curves ($0, 50, 100, 200, 437.8\text{ km}$).

---

## Quickstart

### Prerequisites
- Python 3.11 or 3.12 (validated on Python 3.12.6)
- Dependencies: `streamlit`, `scikit-learn`, `xgboost`, `shap`, `plotly`, `pydeck`, `pyproj`, `scipy`, `pandas`, `numpy`, `joblib`

### Installation
```bash
pip install -r dashboard/requirements.txt
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
All 50 unit, geomorphometric, governance, and schema tests must pass.

### Rebuild Research Artifacts (Optional)
To retrain models, rerun spatial cross-validation, and regenerate all serialized artifacts:
```bash
python scripts/build_research_artifacts.py
```

---

## Automated Testing Suite

The repository contains 50 automated tests in `tests/`:
- `tests/test_scientific_governance.py` (7 tests): Prohibited-feature enforcement, temporal leakage checks, spatial purge distances, fold balance, artifact schemas, and model deserialization.
- `tests/test_v2_sampling_and_features.py` (28 tests): MERIT Hydro 90m TWI, slope aggregation, positive-event validation, multi-batch candidate optimization, coordinate deduplication, 13 rejection categories, and Colab notebook structure.
- `tests/test_data_loader.py` (7 tests): Data loaders for raw reports, cleaned subsets, feature tables, metrics, predictions, regional grids, and models.
- `tests/test_metrics.py` (4 tests): ROC curves, PR curves, confusion matrix calculations across decision thresholds, and susceptibility distribution statistics.
- `tests/test_validation.py` (4 tests): Geographic coordinate bounds ($26.0^\circ-35.5^\circ\text{N}, 74.0^\circ-97.0^\circ\text{E}$), schema compliance, missingness summaries, and label balance.
