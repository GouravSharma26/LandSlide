# Dashboard Implementation Plan: Himalayan Landslide Intelligence

**Project:** Himalayan Landslide Intelligence – Rainfall-Triggered Landslide Detection & Susceptibility Analytics  
**Architecture:** Modular Python / Streamlit Application  
**Execution Command:** `streamlit run dashboard/app.py`  
**Target Completion:** November 2026

---

## 1. System Architecture & Directory Layout

The application is structured into modular layers adhering to separation of concerns: presentation components, analytical engines, data loading, validation, and styling.

```text
LANDSLIDE/
├── dashboard/
│   ├── app.py                      # Application entry point, multi-page router & theme setup
│   ├── config.py                   # Centralized paths, color palettes, bounding boxes, feature lists
│   ├── requirements.txt            # Explicit dependency specifications
│   ├── README.md                   # Installation, execution, and architectural guide
│   ├── assets/
│   │   ├── styles.css              # Custom CSS: dark navy theme, glassmorphism cards, badges
│   │   └── logo.png                # Generated modern geospatial dashboard insignia
│   ├── components/
│   │   ├── sidebar.py              # Navigation menu, global geographic/date filters, reset button
│   │   ├── kpi_cards.py            # Reusable metric indicator cards with deltas and status pills
│   │   ├── charts.py               # Reusable Plotly charts (histograms, bars, scatter, ROC, PR, heatmap)
│   │   └── maps.py                 # PyDeck and Plotly geospatial renderers (points, density, 0.5° grid)
│   ├── pages/
│   │   ├── overview.py             # Page 1: Executive Overview & High-level KPIs
│   │   ├── geographic_explorer.py  # Page 2: Interactive Spatial Explorer
│   │   ├── data_explorer.py        # Page 3: Landslide Records & Source Catalog Explorer
│   │   ├── environmental_analysis.py # Page 4: Multisource Environmental & Trigger Analysis
│   │   ├── model_performance.py    # Page 5: ML Benchmarks, ROC/PR Curves & SHAP Explainability
│   │   ├── susceptibility_analysis.py # Page 6: Regional 0.5° Susceptibility Grid & Uncertainty
│   │   └── data_quality.py         # Page 7: Data Quality, Missingness & Methodology Caveats
│   └── utils/
│       ├── data_loader.py          # Cached data ingest with schema enforcement & error fallbacks
│       ├── preprocessing.py        # Coordinate transformations, binning, and cleaning helpers
│       ├── validation.py           # Integrity checks, bounding-box assertion, null-value audits
│       ├── metrics.py              # Statistical summaries, ROC/PR calculations, confusion matrices
│       └── visualization.py        # Plotly layout themes, color scales, and formatting utilities
├── tests/
│   ├── test_data_loader.py         # Automated tests for data loading and caching
│   ├── test_validation.py          # Automated tests for coordinate ranges and schema integrity
│   └── test_metrics.py             # Automated tests for model metric calculations and consistency
├── outputs/                        # Research artifacts from Phases 2-5
├── Landslide_Reports.csv           # Raw NASA catalog
├── PROJECT_AUDIT.md                # Forensic repository audit
├── DATA_DICTIONARY.md              # Exhaustive variable metadata
└── DASHBOARD_IMPLEMENTATION_PLAN.md # This document
```

---

## 2. UI/UX Design System & Aesthetics

### Color Palette
- **Deep Navy Background:** `#0B132B` (primary dark canvas)
- **Secondary Card Background:** `#1C2541` (elevated surface container)
- **Tertiary Surface / Borders:** `#2A3B5C` and `#3A506B`
- **Primary Accent (Cyan/Teal):** `#48E5C2` / `#5BC0BE` (metrics, positive highlights, high susceptibility)
- **Secondary Accent (Purple/Indigo):** `#6C63FF` / `#7B2CBF` (machine learning, predictions)
- **Warning Accent (Coral/Orange):** `#FF6B6B` / `#FFAA00` (uncertainty, high risk alerts)
- **Text Palette:** Pure White (`#FFFFFF`) for headers, Off-White (`#E0E6ED`) for body, Muted Slate (`#8E9AAF`) for secondary labels.

### Typography & Structure
- Clean sans-serif font stack (`system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`).
- Glassmorphism effect on KPI cards (`backdrop-filter: blur(8px); background: rgba(28, 37, 65, 0.85); border: 1px solid rgba(72, 229, 194, 0.2)`).
- Collapsible responsive sidebar with clear active page indicators.

---

## 3. Detailed Page Implementation Blueprint

### Page 1: Executive Overview (`pages/overview.py`)
- **Key Metrics / KPIs:**
  - Raw Landslide Reports (14,750 global).
  - Cleaned Himalayan Observations (254 points).
  - Positive Events (127) vs. Synthetic Hard Negatives (127).
  - Best Model AUROC (0.976 XGBoost Reduced / 0.982 RF Full).
  - 0.5-Degree Susceptibility Grid Cells (765 cells covering ~2.2M $\text{km}^2$).
  - Observation Temporal Span (2017 to 2020).
- **Interactive Visualizations:**
  - Landslide occurrences by year (temporal trend).
  - Monthly distribution (monsoon seasonality analysis highlighting July–September peaks).
  - Landslide trigger distribution (rainfall, monsoon, downpour).
  - High-level model comparison scorecard (AUROC, PR-AUC, F1).

### Page 2: Geographic Explorer (`pages/geographic_explorer.py`)
- **Interactive Spatial Map:**
  - Multi-layer geospatial viewer using Plotly Mapbox and PyDeck.
  - Positive landslide events (glowing cyan/red markers with hover details: date, trigger, accuracy).
  - Negative control points (purple/gray markers).
  - Susceptibility grid heatmap overlay (0.5-degree cells color-coded by predicted probability).
- **Controls & Filters:**
  - Filter by Country / Administrative Region (India, Nepal, Bhutan, China, Pakistan).
  - Latitude and Longitude bounding box slider.
  - Layer toggles: Positives, Negatives, Susceptibility Grid, Uncertainty Overlay.

### Page 3: Landslide Data Explorer (`pages/data_explorer.py`)
- **Detailed Inventory Analysis:**
  - NASA COOLR / GLC event distribution analysis.
  - Location accuracy breakdown (known $\le 1\text{ km}$, $5\text{ km}$, $25\text{ km}$, etc.).
  - Fatalities and injuries distribution analysis.
  - Filterable data table with search, sorting, and multi-attribute filtering (country, year, trigger, setting).
  - Direct CSV download of filtered subsets.

### Page 4: Environmental Analysis (`pages/environmental_analysis.py`)
- **Multisource Earth Observation Visualizer:**
  - **Precipitation:** 1-day, 7-day, 30-day cumulative rainfall and Antecedent Precipitation Index (API).
  - **Topography:** Elevation, Slope, Aspect, Curvature, Terrain Ruggedness Index (TRI), Topographic Wetness Index (TWI).
  - **Optical:** NDVI (vegetation), NDWI (water/moisture), BSI (bare soil).
  - **SAR Radar:** Sentinel-1 VV/VH backscatter and temporal change ($\Delta\text{VV}$, $\Delta\text{VH}$).
- **Analytical Modules:**
  - Feature distributions comparing Positive vs. Negative samples (KDE / box plots).
  - Interactive Pearson correlation heatmap across all 18 features.
  - Transparent missingness viewer highlighting the 31.5% optical gaps caused by monsoon clouds.

### Page 5: Machine Learning Performance (`pages/model_performance.py`)
- **Spatial Out-of-Fold Evaluation:**
  - Benchmark comparison table: Full Set (18 features) vs. Reduced Set (16 features without slope/TRI).
  - Out-of-fold ROC curves (RF vs. XGBoost) with AUROC annotations.
  - Out-of-fold Precision-Recall curves with PR-AUC annotations.
  - Confusion matrices at customizable decision thresholds (default: $0.5$).
- **Explainable AI (XAI):**
  - Gini / Gain feature importance bar charts.
  - Interactive SHAP summary plots and mean absolute SHAP value rankings for both RF and XGBoost.
  - Documentation of the negative-sampling terrain confound and spatial cross-validation rationale.

### Page 6: Susceptibility Analysis (`pages/susceptibility_analysis.py`)
- **Macro-Grid Geovisualization:**
  - Display the 765 valid cells from `phase5_susceptibility_grid.csv`.
  - Toggle between Ensemble Susceptibility, Random Forest Susceptibility, and XGBoost Susceptibility.
  - Dual Uncertainty Views:
    1. RF Tree Uncertainty (standard deviation across 500 decision trees).
    2. Inter-Model Disagreement ($|\text{RF} - \text{XGB}|$).
- **Filtering & Zonation:**
  - Susceptibility tier classification: Very Low ($<0.2$), Low ($0.2-0.4$), Moderate ($0.4-0.6$), High ($0.6-0.8$), Very High ($>0.8$).
  - Spatial summary by latitude/longitude zones.
  - Clear disclaimer stating the grid is $0.5^\circ \times 0.5^\circ$ (~55 km) macro-resolution.

### Page 7: Data Quality and Methodology (`pages/data_quality.py`)
- **Transparency & Scientific Integrity:**
  - Missing value completeness audit across all datasets.
  - Step-by-step cleaning funnel visualization (from 14,750 raw records down to 127 positive events).
  - Negative sampling methodology review (slope $< 5^\circ$ adaptive buffer).
  - Honest disclosure of sample size limitations (254 points) and terrain confound.
  - Explicit warning: Not a real-time operational warning system.

---

## 4. Engineering & Robustness Strategy

1. **Deterministic Model & SHAP Loading:**
   - To prevent crashes from pickle version mismatch across environments, the app will inspect `phase4_final_models.joblib`. If unpickling encounters an XGBoost serialization error, it will immediately execute the verified, deterministic refit from Phase 5 Cell 2 on `outputs/phase3_feature_table.csv` using identical seeds (`random_state=42`). Execution takes ~1.5 seconds and is cached via `@st.cache_resource`.
2. **Data Caching:**
   - All dataset loads wrapped with `@st.cache_data` for instantaneous page-to-page navigation.
3. **Automated Testing Suite:**
   - Pytest suite covering data loading, column schemas, coordinate bounds, metric calculations, and validation edge cases.
4. **Verification & Screenshot Capture:**
   - Run the application on a local port, verify all pages render error-free, and capture full screenshots of all seven pages for submission.

