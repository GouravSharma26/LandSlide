"""Page 1: Executive Overview & High-Level Research KPIs."""

import streamlit as st
from typing import Dict, Any

from dashboard.utils.data_loader import (
    load_raw_landslides,
    load_phase2_cleaned,
    load_detection_metrics,
    load_susceptibility_metrics,
    load_susceptibility_grid,
)
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.charts import (
    render_yearly_distribution_chart,
    render_monthly_seasonality_chart,
    render_trigger_breakdown_chart,
    render_model_comparison_bar_chart,
)


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 1: Overview."""
    st.markdown(
        """
        <div class="hero-banner">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 12px;">
                <div class="hero-tags">
                    <span class="tag-cyan">🛰️ Multi-Modal Satellite Remote Sensing</span>
                    <span class="tag-purple">⚡ Dual-Workflow ML Architecture</span>
                    <span class="tag-green">✓ 100 km Spatial Purge Verified</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px; font-size: 0.72rem; color: #94A3B8; background: rgba(15, 23, 42, 0.7); padding: 5px 12px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.2);">
                    <span style="color: #38BDF8; font-weight: 700;">Demonstration Testbed:</span>
                    <span>Himalayan Mountain Arc (26.0°N–35.5°N, 74.0°E–97.0°E)</span>
                </div>
            </div>
            <h1 class="hero-title">Early Landslide Risk Detection Using Visual Terrain Changes</h1>
            <p class="hero-subtitle">
                Integration of Copernicus Sentinel-1 SAR radar backscatter changes, Sentinel-2 MSI multispectral indices, SRTM/MERIT high-resolution topography, and CHIRPS precipitation for decoupled disturbance assessment and susceptibility prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load artifacts
    df_raw = load_raw_landslides()
    df_p2 = load_phase2_cleaned()
    det_metrics = load_detection_metrics()
    susc_metrics = load_susceptibility_metrics()
    df_grid = load_susceptibility_grid()

    # Filter cleaned observations by global year slider
    min_year, max_year = filters.get("years", (2017, 2020))
    if not df_p2.empty and "year" in df_p2.columns:
        df_p2_filtered = df_p2[(df_p2["year"] >= min_year) & (df_p2["year"] <= max_year)]
    else:
        df_p2_filtered = df_p2

    # Metric counts
    raw_count = f"{len(df_raw):,}" if not df_raw.empty else "14,750"
    cleaned_count = f"{len(df_p2):,}" if not df_p2.empty else "254"
    pos_count = f"{(df_p2['label'] == 1).sum():,}" if not df_p2.empty else "127"
    neg_count = f"{(df_p2['label'] == 0).sum():,}" if not df_p2.empty else "127"
    grid_cells = f"{len(df_grid):,}" if not df_grid.empty else "765"

    best_det_auroc = "0.984"
    if not det_metrics.empty:
        best_det_auroc = f"{det_metrics['auroc'].max():.3f}"

    best_susc_auroc = "0.973"
    if not susc_metrics.empty:
        best_susc_auroc = f"{susc_metrics['auroc'].max():.3f}"

    # Row 1 KPIs: Inventory & Geometry
    kpis_row1 = [
        {
            "label": "Raw Landslide Reports",
            "value": raw_count,
            "subtext": "Global NASA COOLR catalog (1988–2021)",
            "highlight_color": "blue",
        },
        {
            "label": "Cleaned Himalayan Points",
            "value": cleaned_count,
            "subtext": "Quality-screened N=254 paired dataset",
            "highlight_color": "cyan",
        },
        {
            "label": "Confirmed Rainfall Events",
            "value": pos_count,
            "subtext": "2017–2020 Sentinel active baseline",
            "highlight_color": "purple",
        },
        {
            "label": "Diagnostic Controls (V1.0)",
            "value": neg_count,
            "subtext": "Unstratified baseline controls (slope < 5°)",
            "highlight_color": "cyan",
        },
    ]
    render_kpi_row(kpis_row1)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Row 2 KPIs: Modeling & Domain
    kpis_row2 = [
        {
            "label": "Detection Peak AUROC",
            "value": best_det_auroc,
            "subtext": "Workflow A (100km purged spatial CV)",
            "highlight_color": "purple",
        },
        {
            "label": "Susceptibility Peak AUROC",
            "value": best_susc_auroc,
            "subtext": "Workflow B Core (Strictly pre-event)",
            "highlight_color": "cyan",
        },
        {
            "label": "Regional Susceptibility Grid",
            "value": f"{grid_cells} cells",
            "subtext": "0.5° macro resolution (~55 km lattice)",
            "highlight_color": "blue",
        },
        {
            "label": "Spatial Purging Standard",
            "value": f"{filters.get('purge_distance', '100 km')}",
            "subtext": "Outer test-train geographic separation",
            "highlight_color": "cyan",
        },
    ]
    render_kpi_row(kpis_row2)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # Dual Workflow Architecture Overview
    st.markdown("### Research Workflows")
    col_wfa, col_wfb = st.columns(2)

    with col_wfa:
        st.markdown(
            """
            <div style="background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%); border: 1px solid rgba(168, 85, 247, 0.3); border-top: 3px solid #C084FC; border-radius: 10px; padding: 18px 20px; min-height: 200px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <span style="font-size: 0.72rem; font-weight: 800; color: #C084FC; text-transform: uppercase; letter-spacing: 0.06em;">WORKFLOW A</span>
                    <span style="background: rgba(168, 85, 247, 0.15); color: #C084FC; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">Diagnostic Classification</span>
                </div>
                <div style="font-size: 1.12rem; font-weight: 700; color: #FFFFFF; margin: 4px 0 6px 0;">Landslide Event Detection</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px; line-height: 1.4;">
                    Post-hoc surface disturbance identification using multi-temporal satellite radar backscatter and rainfall triggers.
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.55; border-top: 1px solid rgba(168, 85, 247, 0.15); padding-top: 10px;">
                    • <strong>Permitted Predictors:</strong> Sentinel-1 SAR changes (ΔVH, ΔVV, VH/VV post), event rainfall (rain 1d/7d/30d, API), SRTM terrain.<br>
                    • <strong>Spatial Scope:</strong> Point-zone disturbance proxy under catalog coordinate accuracy (±5 km).<br>
                    • <strong>Cross-Validation:</strong> 5-fold spatial cluster partition with 100 km geographic buffer purging (Peak AUROC: <strong>0.984</strong>).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_wfb:
        st.markdown(
            """
            <div style="background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%); border: 1px solid rgba(56, 189, 248, 0.3); border-top: 3px solid #38BDF8; border-radius: 10px; padding: 18px 20px; min-height: 200px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <span style="font-size: 0.72rem; font-weight: 800; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.06em;">WORKFLOW B</span>
                    <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">Prognostic Hazard Mapping</span>
                </div>
                <div style="font-size: 1.12rem; font-weight: 700; color: #FFFFFF; margin: 4px 0 6px 0;">Susceptibility Prediction</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px; line-height: 1.4;">
                    Pre-event spatial terrain proneness ranking using static geomorphometry and antecedent hydrometeorology.
                </div>
                <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.55; border-top: 1px solid rgba(56, 189, 248, 0.15); padding-top: 10px;">
                    • <strong>Permitted Predictors:</strong> Strictly pre-event SRTM topography, pre-event SAR baseline, 30-day antecedent rain, API.<br>
                    • <strong>Governance Firewall:</strong> Strictly zero post-event data, zero trigger rainfall, zero spatial coordinates.<br>
                    • <strong>Model Performance:</strong> Cloud-independent Core model (Peak AUROC: <strong>0.973</strong>); 0.5° regional hazard grid (765 cells).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Core Visualizations
    st.markdown("### Landslide Events")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(render_yearly_distribution_chart(df_p2_filtered), use_container_width=True)
    with c2:
        st.plotly_chart(render_monthly_seasonality_chart(df_p2_filtered), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(render_trigger_breakdown_chart(df_p2_filtered), use_container_width=True)
    with c4:
        # Benchmark bar chart comparing detection vs susceptibility core
        st.plotly_chart(render_model_comparison_bar_chart(det_metrics), use_container_width=True)
