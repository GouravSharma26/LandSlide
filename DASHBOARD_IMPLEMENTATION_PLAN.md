# Himalayan Landslide Intelligence — Dashboard Implementation Plan

**Project:** Himalayan Landslide Intelligence: Multi-Source Remote Sensing for Landslide Event Detection and Susceptibility Prediction  
**Status:** Step 4 Completed & Verified (All 8 Sections Fully Operational)

---

## 1. System Architecture Overview

The dashboard is organized into eight modular analytical views styled with a dark navy (`#0E1117`), cyan (`#00F0FF`), and purple (`#A855F7`) aesthetic, communicating scientific rigor, transparent limitations, and decoupled ML workflows.

```
dashboard/
├── app.py                             # Main Streamlit router with custom CSS injection
├── config.py                          # Artifact paths, color schemes, feature schemas, status badges
├── assets/
│   ├── logo.png                       # Visual branding
│   ├── styles.css                     # Custom glassmorphism dark navy stylesheet
│   └── screenshots/                   # Verification screenshots of all 8 sections
│       ├── page1_executive_overview.png
│       ├── page2_geographic_explorer.png
│       ├── page3_data_catalog_quality.png
│       ├── page4_environmental_analysis.png
│       ├── page5_event_detection.png
│       ├── page6_susceptibility_prediction.png
│       ├── page7_explainability_uncertainty.png
│       └── page8_methodology_governance.png
├── components/
│   ├── sidebar.py                     # 8-section research navigation, filters, provenance status
│   ├── kpi_cards.py                   # High-contrast metric cards with glassmorphism borders
│   ├── charts.py                      # Reusable Plotly charts (ROC, PR, SHAP, sensitivity, distributions)
│   └── maps.py                        # Plotly Mapbox and PyDeck 3D geospatial visualizers
├── pages/
│   ├── overview.py                    # Section 1: Executive Overview
│   ├── geographic_explorer.py         # Section 2: Geographic Explorer
│   ├── data_quality.py                # Section 3: Data Catalog & Quality
│   ├── environmental_analysis.py      # Section 4: Environmental Analysis
│   ├── event_detection.py             # Section 5: Event Detection (Workflow A)
│   ├── susceptibility_prediction.py   # Section 6: Susceptibility Prediction (Workflow B)
│   ├── explainability_uncertainty.py  # Section 7: Explainability & Uncertainty
│   └── methodology_governance.py      # Section 8: Research Methodology & Governance
└── utils/
    ├── data_loader.py                 # Cached artifact loaders with fallback logic
    ├── metrics.py                     # Evaluation metrics and curve calculators
    ├── preprocessing.py               # Geographic bounds and subzone filtering
    ├── validation.py                  # Schema and missingness validators
    └── visualization.py               # Custom dark Plotly theme generator
```

---

## 2. Eight Section Implementation Specifications

### Section 1: Executive Overview
- **Objective:** High-level executive synthesis of dataset, geographic domain, and dual ML capabilities.
- **Components:**
  - 8 KPI cards: Raw reports (14,750), cleaned paired points (254), positive rainfall events (127), baseline controls (127), peak detection AUROC (0.984), peak susceptibility AUROC (0.973), regional grid cells (765), and spatial purge distance (100 km).
  - Decoupled workflow cards summarizing Workflow A (Detection) and Workflow B (Susceptibility).
  - Temporal distribution charts (yearly counts and monthly monsoon sensitivity).
  - Primary trigger breakdown donut chart and model benchmark scorecard.

### Section 2: Geographic Explorer
- **Objective:** Interactive spatial exploration across the Himalayan mountain arc ($26.0^\circ\text{N} - 35.5^\circ\text{N}$, $74.0^\circ\text{E} - 97.0^\circ\text{E}$).
- **Components:**
  - Regional subzone filtering: India (J&K / Ladakh / HP), India (Uttarakhand), India (Northeast / Sikkim), Nepal.
  - Layer toggles: Observation sites (positive vs control markers) with hover metadata (coordinates, trigger, accuracy).
  - 0.5° regional susceptibility lattice overlay (765 cells) with continuous probability heatmaps.
  - 3D column landscape layer in PyDeck.
  - Disclosure alert on the 0.5° macro resolution and catalog coordinate uncertainty.

### Section 3: Data Catalog & Quality
- **Objective:** Rigorous audit of data cleaning, missing data root causes, and Version 2.0 hillslope protocol.
- **Components:**
  - 7-Stage Preprocessing Funnel table (tracking sample losses from 14,750 down to 127 events).
  - Missingness percentage chart and physical root-cause attribution table (documenting 31.5% monsoon cloud obscuration).
  - Version 1.0 vs Version 2.0 comparison table.
  - Two-sample Kolmogorov-Smirnov test statistics proving flat-terrain bias in Version 1.0 ($p < 10^{-50}$).
  - CSV download buttons for the funnel and missingness audit tables.

### Section 4: Environmental Analysis
- **Objective:** Physical feature distributions across 4 Earth observation domains.
- **Components:**
  - 5 interactive tabs: Rainfall & Hydrology, Topography, Sentinel-1 SAR Backscatter, Sentinel-2 Optical Indices, and Correlation/VIF.
  - Box and violin plots comparing positive events vs. diagnostic controls.
  - Descriptive Variance Inflation Factor (VIF) audit table and Pearson correlation heatmap.
  - Verified Antecedent Precipitation Index ($\alpha = 0.90$).

### Section 5: Event Detection (Workflow A)
- **Objective:** Post-hoc satellite disturbance classification.
- **Classification:** *Experimental Point-Zone Disturbance Assessment*.
- **Components:**
  - Positional uncertainty callout box: 57.5% of events carry $\pm 5\text{ km}$ catalog error; single 10m pixels capture zone disturbance, not verified scars.
  - Outer fold transparency table: Reports Fold 0 through Fold 4, with Fold 4 AUROC transparently reported as undefined ($NaN$) due to 0 positive events in the northwestern Kashmir test partition.
  - Calibrated probability threshold slider with dynamic confusion matrix, specificity, and sensitivity readouts.
  - Out-of-fold ROC curves and Precision-Recall curves.
  - Differential SAR disturbance diagnostic scatter plot ($\Delta\text{VH}$ vs $\Delta\text{VV}$).

### Section 6: Susceptibility Prediction (Workflow B)
- **Objective:** Pre-event spatial landslide susceptibility modeling.
- **Classification:** *Rainfall-Conditioned Susceptibility — Diagnostic Baseline*.
- **Components:**
  - Multi-configuration selector: Cloud-Independent Core Model (10 features, 0% optical nulls) vs Optical-Enhanced Model (13 features) vs Reduced Diagnostic Model (8 features, slope & TRI excluded).
  - Zero temporal leakage enforcement notice (banning post-event radar, deltas, event rain, and coordinates).
  - 0.5° Regional Susceptibility Lattice Explorer (765 cells) conditioned on the September 2026 synoptic late-monsoon window.
  - Susceptibility score histogram and inter-model disagreement distribution ($|\text{RF} - \text{XGB}|$).

### Section 7: Explainability & Uncertainty
- **Objective:** Global TreeSHAP interpretability and spatial autocorrelation sensitivity analysis.
- **Components:**
  - TreeSHAP mean $|SHAP|$ bar charts for Random Forest and XGBoost across both workflows.
  - Physical explanation for why slope dominates and why differential radar deltas show modest point attribution under coordinate uncertainty.
  - Spatial Purging Sensitivity Analysis ($0, 50, 100, 200, 437.8\text{ km}$): Dual-axis chart showing retained training samples vs out-of-fold AUROC as buffer distance increases.

### Section 8: Research Methodology & Governance
- **Objective:** Formal mathematical formulations, prohibited variable contracts, and reproducibility audit.
- **Components:**
  - Formal mathematical equations for Workflow A and Workflow B.
  - Prohibited feature matrix detailing allowed vs banned variables for each pipeline.
  - Spatial 5-fold cluster cross-validation and inner spatial grouping architecture diagram.
  - Mandatory scientific limitation disclosures (not a real-time warning system, positional uncertainty, flat-terrain bias).
  - One-click download button for the complete `experiment_metadata.json` audit log.

---

## 3. Verification & Compliance Record

- **Test Suite Status:** 22/22 tests passing (`python -m pytest tests/ -v`).
- **Live Application Run:** Verified on Streamlit daemon (`http://localhost:8501`).
- **Visual Capture:** Full-page screenshots captured for all 8 sections under `dashboard/assets/screenshots/`.
- **Primary Data Source:** Sourced exclusively from standardized artifacts under `artifacts/`.
