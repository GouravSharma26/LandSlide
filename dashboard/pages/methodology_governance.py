"""Page 8: Research Methodology, Scientific Governance & Architecture."""

import json
import streamlit as st
import pandas as pd
from typing import Dict, Any

from dashboard.config import THEME_COLORS
from dashboard.utils.data_loader import (
    load_feature_schema,
    load_experiment_metadata,
    load_v2_distribution_validation,
    load_dataset_summary,
)
from dashboard.components.kpi_cards import render_kpi_row


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 8: Research Methodology & Governance."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid #38BDF8;">
                Scientific Integrity & Governance
            </div>
            <h1 class="brand-title">RESEARCH METHODOLOGY & GOVERNANCE</h1>
            <div class="brand-subtitle">Formal Mathematical Formulations, Prohibited Variable Contracts & Reproducibility Audit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load artifacts
    schema_doc = load_feature_schema()
    exp_meta = load_experiment_metadata()
    v2_val = load_v2_distribution_validation()
    summary = load_dataset_summary()

    # Governance KPIs
    render_kpi_row([
        {"label": "Independent Workflows", "value": "2 Separate", "subtext": "Detection vs Susceptibility", "highlight_color": "cyan", "icon": "🔀"},
        {"label": "Spatial Purging Standard", "value": "100 km", "subtext": "Zero train within 100km of test", "highlight_color": "purple", "icon": "📏"},
        {"label": "Inner CV Strategy", "value": "Spatial Grouped", "subtext": "StratifiedGroupKFold on clusters", "highlight_color": "cyan", "icon": "🧩"},
        {"label": "Temporal Leakage", "value": "0.0% (Zero)", "subtext": "Post-event data strictly banned in B", "highlight_color": "blue", "icon": "🛡️"},
    ])

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 1. Dual Workflow Formalism
    st.markdown("### 1. Dual Independent Research Formulations")
    st.markdown(
        """
        To prevent the conflation of post-hoc damage detection with pre-event hazard susceptibility, the system implements two independent modeling pipelines:
        """
    )

    w1, w2 = st.columns(2)
    with w1:
        st.markdown(
            """
            <div style="background: #1E293B; border-top: 3px solid #A855F7; border-radius: 4px; padding: 14px; min-height: 250px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #A855F7; text-transform: uppercase;">WORKFLOW A</div>
                <div style="font-size: 1.0rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">Landslide Event Detection</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 8px;">
                    <strong>Objective:</strong> Post-hoc diagnostic disturbance classification.
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.5;">
                    • <strong>Status:</strong> Experimental Point-Zone Disturbance Assessment.<br>
                    • <strong>Research Question:</strong> <em>"Can multisource satellite observations identify evidence of a recently occurred landslide?"</em><br>
                    • <strong>Mathematical Model:</strong> $P(\\text{Event} = 1 \\mid \\mathbf{X}_{\\text{post}}, \\Delta\\mathbf{X}_{\\text{SAR}}, \\mathbf{R}_{\\text{trigger}}, \\mathbf{T})$<br>
                    • <strong>Positional Scale Limitation:</strong> NASA catalog ±5 km error means single 10m pixels evaluate point-zone proxy, not verified scar delineations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with w2:
        st.markdown(
            """
            <div style="background: #1E293B; border-top: 3px solid #00F0FF; border-radius: 4px; padding: 14px; min-height: 250px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #00F0FF; text-transform: uppercase;">WORKFLOW B</div>
                <div style="font-size: 1.0rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">Landslide Susceptibility Prediction</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-bottom: 8px;">
                    <strong>Objective:</strong> Pre-event spatial hazard ranking conditioned on antecedent rainfall.
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.5;">
                    • <strong>Status:</strong> Rainfall-Conditioned Susceptibility (Diagnostic Baseline).<br>
                    • <strong>Research Question:</strong> <em>"Can pre-event environmental and terrain conditions identify hazard-prone hillslopes?"</em><br>
                    • <strong>Mathematical Model:</strong> $P(\\text{Susceptible} = 1 \\mid \\mathbf{T}, \\mathbf{X}_{\\text{pre}}, \\mathbf{R}_{30\\text{d}}, \\text{API}_{\\alpha=0.90})$<br>
                    • <strong>Zero Temporal Leakage:</strong> Post-event radar, deltas, event-day rain, and coordinates are banned.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # 2. Prohibited Features Contract
    st.markdown("### 2. Prohibited Feature & Leakage Prevention Matrix")
    matrix_data = [
        {"Feature / Variable": "SRTM Topography (elevation, slope, aspect, curvature, tri, twi)", "Workflow A (Detection)": "✅ Permitted", "Workflow B (Susceptibility)": "✅ Permitted", "Scientific Rule": "Static terrain baseline invariant to event date"},
        {"Feature / Variable": "Sentinel-1 Post-Event Backscatter (VH_post, VV_post)", "Workflow A (Detection)": "✅ Permitted", "Workflow B (Susceptibility)": "❌ STRICTLY PROHIBITED", "Scientific Rule": "Post-event satellite observation leaks future ground disturbance into susceptibility"},
        {"Feature / Variable": "Differential SAR Change (dVH, dVV)", "Workflow A (Detection)": "✅ Permitted", "Workflow B (Susceptibility)": "❌ STRICTLY PROHIBITED", "Scientific Rule": "Differential backscatter calculated using post-event imagery; strictly banned before event date"},
        {"Feature / Variable": "Event-Day Trigger Rainfall (rain_1d, rain_7d)", "Workflow A (Detection)": "✅ Permitted", "Workflow B (Susceptibility)": "❌ STRICTLY PROHIBITED", "Scientific Rule": "Event-day precipitation is a dynamic trigger, not a pre-event hazard condition"},
        {"Feature / Variable": "Antecedent Soil Moisture Proxy (rain_30d, API alpha=0.90)", "Workflow A (Detection)": "✅ Permitted", "Workflow B (Susceptibility)": "✅ Permitted", "Scientific Rule": "Permitted in B as antecedent rainfall conditioning; evaluated backward from date"},
        {"Feature / Variable": "Pre-Event Sentinel-1 Baseline (VH_pre, VV_pre)", "Workflow A (Detection)": "❌ Omitted (focuses on delta/post)", "Workflow B (Susceptibility)": "✅ Permitted", "Scientific Rule": "Pre-existing structural roughness texture baseline"},
        {"Feature / Variable": "Spatial Coordinates (latitude, longitude)", "Workflow A (Detection)": "❌ STRICTLY PROHIBITED", "Workflow B (Susceptibility)": "❌ STRICTLY PROHIBITED", "Scientific Rule": "Coordinates allow models to memorize geographic hot spots without learning physical mechanics"}
    ]
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # 3. Spatial Purging & Inner CV Architecture
    st.markdown("### 3. Spatial Cluster Cross-Validation & Purging Architecture")
    st.markdown(
        """
        ```
        5-FOLD SPATIAL CLUSTER CV (Azimuthal Equidistant Metric Coordinates)
        ├── Outer Split: KMeans (k=5) separates Himalayan belt into 5 geographic clusters.
        ├── Spatial Purging (100km Standard):
        │   └── In Fold k, any training point within 100km of ANY test point in Fold k is purged.
        │       Result: Actual minimum distance between train and test is strictly >= 101.8 km.
        └── Inner Cross-Validation (Strictly on purged training partition):
            ├── Spatially-Grouped Stratified CV (StratifiedGroupKFold on KMeans clusters).
            ├── Fallback: StratifiedKFold only if severe sample depletion produces mono-class inner folds.
            └── Preprocessing Isolation: SimpleImputer(median) fit strictly inside training folds.
        ```
        """
    )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # 4. Comprehensive Limitations & Disclosures
    st.markdown("### 4. Mandatory Scientific Limitations & Disclosures")
    st.markdown(
        """
        1. **Positional Uncertainty Limitation:** 57.5% of NASA COOLR landslide reports carry ±5 km positional uncertainty. Single-pixel radar backscatter extractions represent point-zone disturbance proxies, not verified ground failure scars.
        2. **Version 1.0 Flat-Terrain Control Bias:** Historical Version 1.0 controls have mean slope 1.43° ± 4.34°, creating an artificial mountain-vs-plains separation that inflates accuracy metrics. Version 2.0 hillslope controls must be extracted in Colab to validate true hillslope discrimination.
        3. **Macro-Regional Grid Resolution:** The regional 0.5° grid (~55 km cell width, 765 valid cells) reflects synoptic climatic-topographic hazard exposure for September 2026. It cannot be used for engineering slope stability.
        4. **Monsoon Cloud Obscuration:** Sentinel-2 optical data suffers from 31.5% missingness during the Himalayan monsoon, making the Cloud-Independent Core model the only operationally viable real-time configuration.
        """
    )

    # Download Experiment Metadata
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if exp_meta:
        st.download_button(
            "📥 Download Complete Experiment Metadata JSON (Provenance Audit Log)",
            json.dumps(exp_meta, indent=2),
            "himalayan_experiment_metadata.json",
            "application/json",
            use_container_width=True
        )

