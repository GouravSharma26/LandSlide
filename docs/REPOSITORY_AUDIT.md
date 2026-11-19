# Repository Audit — LandSlide × GeoTrustNet

**Audit Date:** 2026-11-19  
**Auditor:** Antigravity AI Assistant  
**Status:** Complete — Phase 0  

---

## 1. Git and Environment Audit

- **Repository Path:** `C:\Gourav Sharma\work\LANDSLIDE`
- **Active Git Branch:** `main`
- **Head Commit:** `323bb80` (*"Replace simulated regional grid in master notebook with real Earth Engine regional susceptibility grid"*)
- **Remote Origin:** `https://github.com/GouravSharma26/LandSlide.git` (up to date with `origin/main`)
- **Working Tree Status:** Clean, apart from untracked `geotrustnet/` pack and minor line-ending normalization in `scripts/build_kaggle_colab_notebook.py`.
- **Python Version:** Python 3.12.6 (tags/v3.12.6, 64-bit AMD64)
- **Host OS / Platform:** Windows 11 Pro (`Windows-11-10.0.26300-SP0`)
- **Compute Environment:** CPU only (`torch.cuda.is_available() == False`). Installed PyTorch is `2.14.0+cpu`.
  - *Constraint:* All local training routines must support CPU smoke-tests. Full GPU training and large-scale raster experiments must be executed on Google Colab, Kaggle, or GPU clusters with explicit non-GPU failure reporting.
- **Dependency Management:** `pip` via `requirements.txt` and `dashboard/requirements.txt`.

---

## 2. Project Structure and Components

```text
LANDSLIDE/
├── .gitignore                     # Ignores __pycache__, venvs, /docs/, /doc/, Screenshorts/
├── .streamlit/config.toml         # Dark theme configuration
├── Landslide_Reports.csv          # Raw NASA COOLR catalog export (14,750 rows, 10.5 MB)
├── README.md                      # Primary repository README with lineage disclosures
├── requirements.txt               # Application and testing dependencies
├── artifacts/                     # 21 standardized research artifacts
│   ├── common/                    # dataset_summary.json, preprocessing_funnel.csv, feature_missingness.csv, vif_report.csv
│   ├── detection/                 # detection_feature_table.csv, metrics, fold_metrics, predictions, confusion_matrix, shap, models.joblib
│   ├── metadata/                  # experiment_metadata.json, feature_schema.json, model_metadata.json, spatial_fold_distance_report.json
│   └── susceptibility/            # susceptibility_feature_table.csv, metrics, fold_metrics, grid_0.5deg.csv, shap, predictions
├── dashboard/                     # Multi-page Streamlit application
│   ├── app.py                     # Main router and entry point
│   ├── config.py                  # Paths, theme colors, and regional bounds
│   ├── assets/styles.css          # Dark navy minimalist CSS stylesheet
│   ├── components/                # charts.py, kpi_cards.py, maps.py, sidebar.py
│   ├── pages/                     # 7 active pages (overview, geographic_explorer, data_quality, environmental_analysis, event_detection, susceptibility_prediction, explainability_uncertainty)
│   └── utils/                     # data_loader.py, metrics.py, preprocessing.py, validation.py, visualization.py
├── docs/                          # Internal research documentation & guides
│   ├── ARCHITECTURE_DESIGN.md
│   ├── DASHBOARD_IMPLEMENTATION_PLAN.md
│   ├── DATA_DICTIONARY.md
│   ├── hillslope_control_extraction_executed.ipynb
│   ├── PRESENTATION_AND_VIVA_GUIDE.md
│   ├── PROJECT_AUDIT_REPORT.md
│   ├── RESEARCH_PIPELINE_DESIGN.md
│   └── V2_EXTRACTION_PROTOCOL_AND_EXECUTION_GUIDE.md
├── geotrustnet/                   # Starter pack: AGENTS.md, START_HERE.md, docs/geotrustnet/
├── notebooks/                     # Exploratory & execution notebooks
│   ├── data_ingestion.ipynb
│   ├── data_preprocessing.ipynb
│   ├── early_landslide_risk_detection_master.ipynb
│   ├── feature_engineering.ipynb
│   ├── hillslope_control_extraction.ipynb
│   ├── machine_learning_modeling.ipynb
│   └── susceptibility_evaluation.ipynb
├── outputs/                       # 6 cleaned production CSV tables
│   ├── cleaned_observation_records.csv
│   ├── model_evaluation_metrics.csv
│   ├── multisource_feature_matrix.csv
│   ├── regional_grid_features_0.5deg.csv
│   ├── regional_susceptibility_grid.csv
│   └── spatial_oof_predictions.csv
├── scripts/                       # build_research_artifacts.py, v2_sampling_protocol.py, build_kaggle_colab_notebook.py
└── tests/                         # 6 pytest test suites (59 unit and regression tests)
```

---

## 3. Existing Capabilities and Verified Paths

| Component | Verified Path | Current Behavior | Evidence / Metrics | Preservation Decision |
|---|---|---|---|---|
| **Entry Point** | `dashboard/app.py` | 7-page Streamlit dashboard router | Imports and launches without error | **PRESERVE 100%** |
| **Page 1: Overview** | `dashboard/pages/overview.py` | Executive KPIs, annual/monthly event frequency, trigger breakdowns | Real NASA COOLR ($N=14,750$) and cleaned Himalayan points ($N=254$) | **PRESERVE** |
| **Page 2: Study Area** | `dashboard/pages/geographic_explorer.py` | Folium point map + PyDeck 3D grid viewer | Renders 254 observation coordinates and 765 valid grid cells | **PRESERVE** |
| **Page 3: Data Quality** | `dashboard/pages/data_quality.py` | Preprocessing funnel & feature missingness table | 7-stage reduction and 31.5% optical gap audit | **PRESERVE** |
| **Page 4: Environmental Analysis** | `dashboard/pages/environmental_analysis.py` | Distribution of rainfall, slope, SAR backscatter, optical indices | Real CHIRPS, SRTM, Sentinel-1/2 multi-sensor plots | **PRESERVE** |
| **Page 5: Event Detection** | `dashboard/pages/event_detection.py` | Workflow A: Post-hoc disturbance classification | AUROC: 0.984 (RF) / 0.968 (XGB) under 100 km spatial purge | **PRESERVE & EXTEND** |
| **Page 6: Susceptibility Prediction** | `dashboard/pages/susceptibility_prediction.py` | Workflow B: Prognostic terrain hazard ranking | AUROC: 0.973 (Core) / 0.977 (Core + Optical) | **PRESERVE** |
| **Page 7: Feature Importance** | `dashboard/pages/explainability_uncertainty.py` | TreeSHAP feature importance & spatial sensitivity curves | Evaluates 0 km to 437.8 km buffer purges | **PRESERVE** |
| **Artifact Generation** | `scripts/build_research_artifacts.py` | Generates all 21 standardized CSV/JSON artifacts | 1,027 lines of validated artifact builders | **PRESERVE** |
| **Test Suite** | `tests/` (59 tests) | 6 test suites covering loaders, maps, governance, metrics | 59/59 passed in 13.99s (100%) | **PRESERVE (Regressions)** |
| **Live Deployment** | Streamlit Community Cloud | Deployed at `landslide-ncg2cbc9pfgrt4kpujo7xq.streamlit.app` | Operational production app | **PRESERVE** |

---

## 4. Data Inventory & Provenance

| Dataset / File | Source | License | Modality / Format | Resolution / Bounds | Labels | Current Role |
|---|---|---|---|---|---|---|
| `Landslide_Reports.csv` | NASA Global Landslide Catalog (COOLR) | Open Access / Public Domain | Tabular CSV (14,750 rows, 10.5 MB) | Global ($1988–2021$); $\pm 5\text{ km}$ coordinate accuracy | Reported historical landslide events | Raw catalog source |
| `cleaned_observation_records.csv` | NASA COOLR filtered | Project Generated | Tabular CSV (254 rows $\times$ 8 cols) | WGS84 (EPSG:4326), $26.0^\circ–35.5^\circ\text{N}$, $74.0^\circ–97.0^\circ\text{E}$ | 127 positive events, 127 diagnostic controls | Spatial training coordinates |
| `multisource_feature_matrix.csv` | SRTM + Sentinel-1 + Sentinel-2 + CHIRPS via GEE | Derived from Copernicus, USGS, UCSB | Tabular CSV (254 rows $\times$ 28 cols) | 30m SRTM, 10m Sentinel-1/2, 0.05° CHIRPS | Binary: 1 (Landslide), 0 (Diagnostic Control) | Feature matrix for tabular ML |
| `regional_susceptibility_grid.csv` | Google Earth Engine lattice | Derived from Copernicus, USGS, UCSB | Tabular CSV (765 rows $\times$ 7 cols) | $0.5^\circ \times 0.5^\circ$ macro-lattice (~55 km width) | Continuous susceptibility probabilities $[0, 1]$ | Macro-regional hazard layer |
| **Sen12Landslides** *(Target for GeoTrustNet)* | Höhn et al. (Nature Scientific Data 2025) | Open Research License (HuggingFace) | Spatio-temporal raster patches (GeoTIFF) | 10m Sentinel-1, 10m Sentinel-2, DEM time series | Pixel-level binary segmentation masks | Target dataset for GeoTrustNet segmentation |

---

## 5. Baseline Verification and Test Outcomes

- **Test Command:** `python -m pytest tests/ -v`
- **Test Result:** **59 passed in 13.99s (100% pass rate)**.
  - `tests/test_data_loader.py`: 7 passed
  - `tests/test_maps.py`: 8 passed
  - `tests/test_metrics.py`: 5 passed
  - `tests/test_scientific_governance.py`: 7 passed
  - `tests/test_v2_sampling_and_features.py`: 28 passed
  - `tests/test_validation.py`: 4 passed
- **Dashboard Import Smoke Test:** `python -c "import dashboard.app; ..."`
  - Result: All 7 dashboard pages import and initialize with zero runtime errors.
- **Known Limitations & Boundary Disclosures:**
  1. *Tabular Scale vs Raster Segmentation:* The existing LandSlide project operates on tabular point samples ($N=254$) extracted at nominal landslide coordinates. It does not perform pixel-level patch segmentation.
  2. *Fold 4 NaN Disclosure:* Under 100 km spatial cluster buffer purging, Fold 4's geographic partition contains only control observations, resulting in an undefined individual fold AUROC. This is honestly disclosed and documented in `artifacts/metadata/experiment_metadata.json`.
  3. *Version 2.0 Hillslope Audit:* A 45.3-minute automated GEE audit proved that finding steep slopes ($>15^\circ$) in the active monsoon belt without rainfall triggers is geomorphologically rare (100% rejection rate across 179 candidate windows). Version 1.0 remains the authoritative diagnostic baseline.

---

## 6. Risks and Constraints

1. **Task Formulation Divergence (Tabular Susceptibility vs Spatio-Temporal Segmentation):**
   - *Risk:* Overwriting the existing tabular machine learning models with a patch-segmentation model would break the existing 7-page dashboard and invalidating existing tests.
   - *Mitigation:* Explicitly isolate GeoTrustNet under `src/geotrustnet/` as a modular deep-learning engine for Task A (Segmentation), while keeping the tabular models intact for Task B (Regional Susceptibility).
2. **Compute Constraint (CPU vs GPU):**
   - *Risk:* The local machine has no CUDA-capable GPU (`torch.cuda.is_available() == False`).
   - *Mitigation:* Ensure all GeoTrustNet modules support deterministic CPU smoke-tests, shape tests, and synthetic fixture tests locally, while providing Google Colab / Kaggle execution scripts for GPU training.
3. **Data Leakage & Spatial Autocorrelation:**
   - *Risk:* Splitting adjacent raster patches randomly causes severe spatial data leakage.
   - *Mitigation:* Enforce strict event-level and geographic region-held-out splits with saved manifest hashes.
4. **Dashboard Stability & Streamlit Cloud:**
   - *Risk:* Installing heavy deep learning dependencies (e.g., full PyTorch CUDA or complex GIS stacks) can break the Streamlit Community Cloud build.
   - *Mitigation:* Keep `requirements.txt` lightweight for deployment; isolate GeoTrustNet training requirements in `requirements-geotrustnet.txt` or optional dependencies with graceful fallback.

---

## 7. Recommendations for Next Phases

1. **Keep Untouched:**
   - `dashboard/` (all 7 existing pages, components, and utils).
   - `outputs/` (all 6 production CSV tables).
   - `artifacts/` (all 21 standardized CSV, JSON, joblib artifacts).
   - `tests/` (all 59 existing unit tests).
   - `Landslide_Reports.csv` (10.5 MB raw source catalog).
2. **New Modules to Create (under `src/geotrustnet/`):**
   - `src/geotrustnet/data/`: Dataset loaders, raster preprocessing, geospatial transforms, leakage-safe split manifests.
   - `src/geotrustnet/models/`: SAR encoder, optical encoder, temporal module, terrain branch, reliability-aware fusion, U-Net decoder, uncertainty head.
   - `src/geotrustnet/training/`: Training engine, losses (Dice/Focal/BCE), checkpointing, CPU smoke-tests.
   - `src/geotrustnet/evaluation/`: Segmentation metrics (IoU, F1, PR-AUC), calibration metrics (ECE, Brier, NLL), geographic holdout evaluation.
   - `src/geotrustnet/inference/`: Georeferenced raster prediction export.
3. **Phase 1 Priorities:**
   - Define data contracts and schema for raster patches (Sen12Landslides).
   - Implement synthetic raster test fixtures for CPU testing.
   - Build leakage-safe region/event split generator with checksums.
