"""Page 2: Interactive Geographic Explorer covering the Himalayan region."""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from dashboard.utils.data_loader import (
    load_phase2_cleaned,
    load_susceptibility_grid,
    load_raw_landslides,
)
from dashboard.utils.preprocessing import filter_by_bounds, assign_regional_subzones
from dashboard.components.maps import (
    render_observation_map,
    render_susceptibility_grid_map,
    render_pydeck_susceptibility_map,
)
from dashboard.components.kpi_cards import render_kpi_row


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 2: Geographic Explorer."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge">Geospatial Explorer</div>
            <h1 class="brand-title">HIMALAYAN SPATIAL INTELLIGENCE</h1>
            <div class="brand-subtitle">Interactive Geospatial Mapping of Landslide Observations & 0.5° Susceptibility Lattice</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Resolution Notice
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 3px solid #64748B; padding: 10px 14px; margin-bottom: 18px; border-radius: 4px;">
            <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.45;">
                <strong style="color: #F8FAFC;">Resolution & Ground-Truth Disclosure:</strong> The regional susceptibility grid operates at <strong>0.5° macro resolution (~55 km cell width, 765 valid cells)</strong> conditioned on late-monsoon CHIRPS antecedent rainfall. It represents macro-regional climate-terrain hazard exposure and is classified as <strong>Illustrative</strong> (not an engineering slope stability map). Individual point markers carry the NASA catalog's ±5 km positional uncertainty.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load datasets
    df_p2 = load_phase2_cleaned()
    df_grid = load_susceptibility_grid()

    if df_p2.empty or df_grid.empty:
        st.warning("Geospatial datasets could not be loaded. Please verify output files.")
        return

    # Add regional subzone tags
    df_p2_tagged = assign_regional_subzones(df_p2)
    df_grid_tagged = assign_regional_subzones(df_grid)

    # Filter by region if requested
    selected_region = filters.get("region", "All Himalayan Regions")
    if selected_region != "All Himalayan Regions":
        df_p2_tagged = df_p2_tagged[df_p2_tagged["region_subzone"] == selected_region]
        df_grid_tagged = df_grid_tagged[df_grid_tagged["region_subzone"] == selected_region]

    # Map Controls
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">🗺️ Geospatial Layer & Visualization Controls</div>
            <div class="card-description">Toggle layers, choose render mode, and filter observation points across the Himalayan mountain arc.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        map_mode = st.selectbox(
            "Visualization Layer",
            [
                "Susceptibility Grid Heatmap (0.5°)",
                "Observation Sites (Positives vs Negatives)",
                "3D Susceptibility Landscape (PyDeck)",
            ],
            index=0,
        )

    with ctrl_col2:
        if "Grid" in map_mode or "3D" in map_mode:
            grid_metric = st.selectbox(
                "Grid Attribute",
                [
                    ("susceptibility", "Ensemble Susceptibility"),
                    ("susceptibility_rf", "Random Forest Susceptibility"),
                    ("susceptibility_xgb", "XGBoost Susceptibility"),
                    ("rf_tree_uncertainty", "Tree Uncertainty (RF std dev)"),
                    ("model_disagreement", "Model Disagreement (|RF - XGB|)"),
                ],
                format_func=lambda x: x[1],
            )[0]
        else:
            grid_metric = "susceptibility"

    with ctrl_col3:
        show_positives = st.checkbox("Show Positive Events (127)", value=True)

    with ctrl_col4:
        show_negatives = st.checkbox("Show Diagnostic Controls (127)", value=True)

    # Quick summary KPIs
    pos_visible = (df_p2_tagged["label"] == 1).sum() if not df_p2_tagged.empty else 0
    neg_visible = (df_p2_tagged["label"] == 0).sum() if not df_p2_tagged.empty else 0
    grid_cells_visible = len(df_grid_tagged)

    geo_kpis = [
        {"label": "Active Region", "value": selected_region.split("(")[0].strip(), "highlight_color": "blue", "icon": "📍"},
        {"label": "Visible Positives", "value": f"{pos_visible}", "subtext": "Historical ground failure points", "highlight_color": "purple", "icon": "🔴"},
        {"label": "Visible Controls", "value": f"{neg_visible}", "subtext": "Unstratified flat-terrain controls", "highlight_color": "cyan", "icon": "🟢"},
        {"label": "Grid Cells (0.5°)", "value": f"{grid_cells_visible}", "subtext": "Synoptic climate-terrain cells", "highlight_color": "blue", "icon": "🗺️"},
    ]
    render_kpi_row(geo_kpis)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Render Chosen Map
    if map_mode == "Observation Sites (Positives vs Negatives)":
        fig = render_observation_map(
            df_p2_tagged,
            show_positives=show_positives,
            show_negatives=show_negatives,
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif map_mode == "Susceptibility Grid Heatmap (0.5°)":
        fig = render_susceptibility_grid_map(
            df_grid_tagged,
            value_column=grid_metric,
            title=f"Himalayan Regional Susceptibility Grid — {grid_metric.replace('_', ' ').title()}",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif map_mode == "3D Susceptibility Landscape (PyDeck)":
        deck = render_pydeck_susceptibility_map(df_grid_tagged, elevation_scale=40)
        st.pydeck_chart(deck)

    # Data Table Preview
    with st.expander("🔍 Inspect Underlying Spatial Coordinates Table", expanded=False):
        st.dataframe(
            df_p2_tagged[["latitude", "longitude", "event_date", "Location Accuracy", "Landslide Trigger", "label", "region_subzone"]].head(50),
            use_container_width=True,
        )
