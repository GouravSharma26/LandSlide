"""Page 7: Data Quality, Missingness & Methodology Caveats."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any

from dashboard.utils.data_loader import (
    load_phase2_cleaned,
    load_phase3_features,
    load_phase5_grid,
    load_raw_landslides,
)
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.charts import render_missingness_chart
from dashboard.utils.validation import calculate_missingness_summary
from dashboard.config import THEME_COLORS
from dashboard.utils.visualization import apply_custom_theme


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 7: Data Quality and Methodology."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge">Quality & Provenance</div>
            <h1 class="brand-title">DATA QUALITY & SCIENTIFIC METHODOLOGY</h1>
            <div class="brand-subtitle">Transparent Assessment of Missingness, Preprocessing Funnels & Model Limitations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_p3 = load_phase3_features()
    df_raw = load_raw_landslides()
    df_p2 = load_phase2_cleaned()

    # Mandatory Scientific Disclaimer Alert
    st.markdown(
        """
        <div class="alert-box alert-danger">
            <span style="font-size: 1.3rem;">🛑</span>
            <div>
                <b>Operational Warning & Scientific Disclaimer:</b> This research system is an analytical prototype designed to explore multi-sensor feature fusion. It is <b>NOT a certified, real-time operational landslide early-warning system</b>. Do not use this tool for evacuation planning, civil emergency response, or life-safety decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Missingness summary
    miss_summary = calculate_missingness_summary(df_p3) if not df_p3.empty else pd.DataFrame()

    # Data Quality KPIs
    total_cells = 254
    optical_missing_pct = 31.5
    sar_complete_pct = 97.2
    rainfall_complete_pct = 100.0

    render_kpi_row([
        {"label": "Rainfall Completeness", "value": f"{rainfall_complete_pct:.0f}%", "subtext": "CHIRPS daily (0 nulls)", "highlight_color": "cyan", "icon": "🌧️"},
        {"label": "SAR Completeness", "value": f"{sar_complete_pct:.1f}%", "subtext": "Sentinel-1 GRD IW (VV/VH)", "highlight_color": "cyan", "icon": "🛰️"},
        {"label": "Optical Missingness", "value": f"{optical_missing_pct:.1f}%", "subtext": "Monsoon cloud obscuration (80 pts)", "highlight_color": "danger", "icon": "☁️"},
        {"label": "Observation Sample", "value": f"{total_cells} pts", "subtext": "127 Positives : 127 Negatives", "highlight_color": "purple", "icon": "⚖️"},
    ])

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Missingness Breakdown Chart & Table
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">📊 Variable Completeness & Null Value Attribution</div>
            <div class="card-description">Auditing missing data rates across all 28 columns of the Phase 3 feature engineering table.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not miss_summary.empty:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.plotly_chart(render_missingness_chart(miss_summary), use_container_width=True)
        with c2:
            st.dataframe(
                miss_summary[["column", "non_null_count", "missing_count", "missing_pct"]],
                use_container_width=True,
                height=450,
            )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Preprocessing Funnel & Methodology Stages
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">🔄 Inventory Preprocessing Funnel (From 14,750 down to 127 Events)</div>
            <div class="card-description">Sequential screening stages applied in Phase 2 to isolate high-fidelity rainfall-triggered events.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    funnel_stages = [
        {"Stage": "1. Raw NASA Global Landslide Catalog", "Count": 14750, "Loss": "Initial uncurated global repository"},
        {"Stage": "2. Valid Parseable Date", "Count": 13113, "Loss": "-1,637 missing date records"},
        {"Stage": "3. Himalayan Bounding Box (26–35°N, 74–97°E)", "Count": 2154, "Loss": "-10,959 outside regional ROI"},
        {"Stage": "4. Sentinel-1/2 Active Baseline (2017–2020)", "Count": 386, "Loss": "-1,768 outside satellite coverage period"},
        {"Stage": "5. Rainfall-Triggered Only (rain, monsoon, downpour)", "Count": 278, "Loss": "-108 non-rainfall triggers (earthquake, mining)"},
        {"Stage": "6. Positional Accuracy ≤ 5 km", "Count": 148, "Loss": "-130 coarse or unverified coordinates"},
        {"Stage": "7. Spatio-Temporal Deduplication (1 km / 1 day)", "Count": 127, "Loss": "-21 duplicate or clustered reports"},
    ]
    df_funnel = pd.DataFrame(funnel_stages)

    fig_funnel = px.funnel(
        df_funnel,
        y="Stage",
        x="Count",
        color_discrete_sequence=[THEME_COLORS["primary"]],
    )
    fig_funnel = apply_custom_theme(
        fig_funnel,
        title="NASA Inventory Cleaning Funnel",
        height=380,
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Detailed Scientific Limitations Documentation
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">🔬 Key Scientific Limitations & Methodology Audit</div>
            <div style="color: #CBD5E1; line-height: 1.6; font-size: 0.92rem;">
                <h4 style="color: #48E5C2; margin-top: 1rem;">1. The Negative-Sampling Terrain Confound</h4>
                <p>
                    In Phase 2, control negative samples were constrained to areas with <code>slope &lt; 5°</code> and distance <code>&gt; 250m</code> from any known landslide. Because steep topography is a prerequisite for slope failure, the model achieves near-perfect discrimination (&gt;0.98 AUROC) simply by checking whether terrain is steep or flat. Removing slope and TRI (the Reduced Feature Set) mitigates this, but residual elevation and curvature correlation persist. Future iterations must sample negatives from <i>unfailed steep slopes</i> under heavy rainfall to test true triggering dynamics.
                </p>

                <h4 style="color: #6C63FF; margin-top: 1rem;">2. Small Observation Sample Size (N = 254)</h4>
                <p>
                    Only 127 confirmed rainfall-triggered landslide events met all strict quality and satellite-synchronization criteria across the entire Himalayan belt. While 5-fold Spatial Cluster Cross-Validation protects against spatial overfitting, the small sample size limits the statistical power of subgroup analyses (e.g. geological lithology or microclimate interactions).
                </p>

                <h4 style="color: #3A86FF; margin-top: 1rem;">3. Spatial Scale Asymmetry</h4>
                <p>
                    Features enter the pipeline at heterogeneous native resolutions:
                    SRTM Topography (30m), Sentinel-2 Optical (10–20m), Sentinel-1 SAR (10m), and CHIRPS Precipitation (~5.5 km). Furthermore, the regional susceptibility grid is evaluated at <b>0.5° (~55 km)</b>. Consequently, fine-scale slope processes (such as road-cut destabilization or debris-flow channels) are smoothed at regional scales.
                </p>

                <h4 style="color: #FFB703; margin-top: 1rem;">4. Monsoon Optical Cloud Obscuration</h4>
                <p>
                    Landslides in the Himalayas correlate strongly with the South Asian Monsoon. During peak monsoon months (July–September), persistent cloud cover degrades optical surface reflectance, resulting in a 31.5% missing rate for NDVI, NDWI, and BSI. All-weather C-band Sentinel-1 SAR backscatter serves as the vital fallback.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

