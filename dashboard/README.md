# Himalayan Landslide Intelligence Research Dashboard

An interactive visualization and analytics platform built with Streamlit, Plotly, PyDeck, Scikit-learn, XGBoost, and SHAP to support geospatial research on rainfall-triggered landslides across the Himalayan mountain belt ($26.0^\circ\text{N} - 35.5^\circ\text{N}$, $74.0^\circ\text{E} - 97.0^\circ\text{E}$).

---

## Quickstart

### Installation
From the project root:
```bash
pip install -r dashboard/requirements.txt
```

### Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Access the application locally at `http://localhost:8501`.

---

## Dashboard Architecture

The dashboard is organized into seven analytical sections accessible through the sidebar navigation:

1. **Executive Overview (`dashboard/pages/overview.py`):**
   - High-level KPIs: Raw NASA catalog reports (14,750), cleaned paired points (254), catalogued rainfall events meeting criteria (127), and 0.5° regional grid cells (765).
   - Temporal event trends (2017–2020), monsoon seasonality distribution, reported trigger breakdown, and dual-workflow summary.

2. **Geographic Explorer (`dashboard/pages/geographic_explorer.py`):**
   - Interactive spatial visualization with Plotly Mapbox and PyDeck 3D column layers.
   - Spatial distribution of catalogued events versus diagnostic negative controls, with physiographic subregion filtering.
   - Macro-regional 0.5° susceptibility grid overlay toggle.

3. **Data Catalog & Quality (`dashboard/pages/data_quality.py`):**
   - 7-stage preprocessing funnel from 14,750 raw global reports to 127 catalogued Himalayan events.
   - Feature missingness audit documenting the 31.5% optical null rate during the monsoon season.
   - Version 1.0 versus Version 2.0 lineage comparison and two-sample Kolmogorov-Smirnov test confirming V1 flat-terrain bias ($p < 10^{-50}$).

4. **Environmental Analysis (`dashboard/pages/environmental_analysis.py`):**
   - Multisource feature distributions across precipitation (CHIRPS), topography (SRTM), SAR backscatter (Sentinel-1), and optical indices (Sentinel-2).
   - Correlation heatmaps and descriptive Variance Inflation Factor (VIF) tables.

5. **Event Detection (`dashboard/pages/event_detection.py`):**
   - Workflow A: Experimental Point-Zone Disturbance Assessment.
   - Fold-wise metric reporting with explicit disclosure of Fold 4 (Kashmir cluster containing 0 positive test instances, resulting in undefined fold AUROC/PR-AUC).
   - Interactive decision threshold slider, confusion matrix, and out-of-fold ROC/PR curves.
   - Differential radar change diagnostics ($\Delta\text{VH}$ vs $\Delta\text{VV}$).

6. **Susceptibility Prediction (`dashboard/pages/susceptibility_prediction.py`):**
   - Workflow B: Rainfall-Conditioned Susceptibility Diagnostic Baseline.
   - Comparison across Cloud-Independent Core (10 features), Optical-Enhanced (13 features), and Reduced (8 features, slope and TRI excluded) configurations.
   - Macro-regional 0.5° lattice explorer (765 cells, September 2026 synoptic late-monsoon window) and inter-model disagreement analysis ($|\text{RF} - \text{XGB}|$).

7. **Explainability & Uncertainty (`dashboard/pages/explainability_uncertainty.py`):**
   - Global TreeSHAP feature attributions for Random Forest and XGBoost.
   - Explanation of radar change attribution in the presence of $\pm 5\text{ km}$ catalog location uncertainty.
   - Spatial purging sensitivity curves across buffer distances ($0, 50, 100, 200, 437.8\text{ km}$).

---

## Scientific Governance and Limitations

- **Workflow A:** Evaluates point-zone disturbance proxies, not verified landslide scars, due to catalog coordinate error (57.5% within $\pm 5\text{ km}$).
- **Workflow B:** Evaluated against Version 1.0 diagnostic controls (mean slope $1.43^\circ$); metrics reflect mountain-versus-plains contrast. It is not an operational slope-stability model.
- **Version 2.0 Status:** Candidate hillslope-control protocol is designed and tested via script and Colab notebook; genuine extraction remains `EXTRACTION_PENDING_COLAB_GEE`.
- **0.5-Degree Grid:** Represents a synoptic macro-regional climate-terrain illustration for September 2026; it is not suitable for localized engineering hazard assessments.
