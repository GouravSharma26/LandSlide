"""Page 2: Interactive Geographic Explorer covering the Himalayan region."""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from dashboard.utils.data_loader import (
    load_phase2_cleaned,
    load_phase5_grid,
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
            <div class="brand-subtitle">Interactive High-Resolution Mapping of Landslide Events & Susceptibility Grids</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load datasets
    df_p2 = load_phase2_cleaned()
    df_grid = load_phase5_grid()

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
        show_negatives = st.checkbox("Show Control Negatives (127)", value=True)

    # Quick summary KPIs
    pos_visible = (df_p2_tagged["label"] == 1).sum() if not df_p2_tagged.empty else 0
    neg_visible = (df_p2_tagged["label"] == 0).sum() if not df_p2_tagged.empty else 0
    grid_visible = len(df_grid_tagged)

    render_kpi_row([
        {"label": "Visible Positives", "value": pos_visible, "subtext": "Landslide locations", "highlight_color": "danger", "icon": "📍"},
        {"label": "Visible Negatives", "value": neg_visible, "subtext": "Hard control sites", "highlight_color": "cyan", "icon": "🛡️"},
        {"label": "Grid Cells", "value": grid_visible, "subtext": "0.5° macro cells", "highlight_color": "purple", "icon": "🌐"},
        {"label": "Selected Subzone", "value": selected_region.split(" ")[0], "subtext": selected_region, "highlight_color": "blue", "icon": "🗺️"},
    ])

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Map Rendering
    if map_mode == "Susceptibility Grid Heatmap (0.5°)":
        colorscale = "Purples" if "uncertainty" in grid_metric else "Blues" if "disagreement" in grid_metric else "YlOrRd"
        st.plotly_chart(
            render_susceptibility_grid_map(
                df_grid_tagged,
                value_column=grid_metric,
                title=f"Himalayan Susceptibility Grid — {grid_metric.replace('_', ' ').title()}",
                color_scale=colorscale,
                height=600,
            ),
            use_container_width=True,
        )
    elif map_mode == "Observation Sites (Positives vs Negatives)":
        st.plotly_chart(
            render_observation_map(
                df_p2_tagged,
                show_positives=show_positives,
                show_negatives=show_negatives,
                height=600,
            ),
            use_container_width=True,
        )
    else:
        st.pydeck_chart(
            render_pydeck_susceptibility_map(
                df_grid_tagged,
                value_column=grid_metric,
            )
        )

    # Coordinate inspection table
    with st.expander("🔍 Inspect Underlying Coordinates & Attribution Data", expanded=False):
        tab_obs, tab_grd = st.tabs(["Observation Sites (254 Points)", "Susceptibility Grid Cells (765 Cells)"])
        with tab_obs:
            st.dataframe(df_p2_tagged, use_container_width=True, height=300)
        with tab_grd:
            st.dataframe(df_grid_tagged, use_container_width=True, height=300)

