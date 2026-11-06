"""Page 4: Environmental & Remote Sensing Feature Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from dashboard.utils.data_loader import load_phase3_features
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.charts import render_correlation_heatmap
from dashboard.config import (
    THEME_COLORS,
    TOPOGRAPHIC_FEATURES,
    OPTICAL_FEATURES,
    SAR_FEATURES,
    RAINFALL_FEATURES,
    FULL_FEATURE_COLS,
)
from dashboard.utils.visualization import apply_custom_theme


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 4: Environmental Analysis."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge">Environmental Analytics</div>
            <h1 class="brand-title">MULTISOURCE SATELLITE & TERRAIN ANALYSIS</h1>
            <div class="brand-subtitle">High-Fidelity Feature Distributions Across Topographic, SAR, Optical & Rainfall Domains</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_p3 = load_phase3_features()
    if df_p3.empty:
        st.warning("Phase 3 feature table not found. Verify outputs/phase3_feature_table.csv.")
        return

    # Data Quality Callout regarding optical missingness
    st.markdown(
        """
        <div class="alert-box alert-warning">
            <span style="font-size: 1.25rem;">⚠️</span>
            <div>
                <b>Missing Data Transparency:</b> Heavy monsoon cloud cover prevents Sentinel-2 optical extraction for 80 of the 254 points (31.5% null rate in BSI, NDVI, NDWI). In contrast, all-weather Sentinel-1 SAR GRD and CHIRPS precipitation have &gt;97% data completeness. Missing values are displayed as-is and never silently imputed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Domain Selector
    domain_tabs = st.tabs([
        "🌧️ Precipitation & Hydrology",
        "⛰️ Topographic Derivatives",
        "🛰️ SAR Radar (Sentinel-1)",
        "🌿 Optical Indices (Sentinel-2)",
        "🔥 Correlation Heatmap",
    ])

    with domain_tabs[0]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Precipitation Metrics: CHIRPS Daily & Cumulative Rainfall</div>
                <div class="card-description">Comparing 1-day, 7-day, and 30-day rainfall plus Antecedent Precipitation Index (API) between landslide events and stable controls.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rf_col1, rf_col2 = st.columns(2)
        with rf_col1:
            fig_api = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="API",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_api = apply_custom_theme(
                fig_api,
                title="Antecedent Precipitation Index (API, k=0.85)",
                x_title="Observation Class",
                y_title="API Value (mm)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_api, use_container_width=True)

        with rf_col2:
            fig_r30 = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="rain_30d",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_r30 = apply_custom_theme(
                fig_r30,
                title="30-Day Cumulative Precipitation (mm)",
                x_title="Observation Class",
                y_title="Rainfall (mm)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_r30, use_container_width=True)

        rf_col3, rf_col4 = st.columns(2)
        with rf_col3:
            fig_r1 = px.histogram(
                df_p3,
                x="rain_1d",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                barmode="overlay",
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                opacity=0.7,
            )
            fig_r1 = apply_custom_theme(
                fig_r1,
                title="1-Day Rainfall Event Distribution",
                x_title="1-Day Precipitation (mm)",
                y_title="Frequency",
                height=320,
            )
            st.plotly_chart(fig_r1, use_container_width=True)

        with rf_col4:
            fig_r7 = px.histogram(
                df_p3,
                x="rain_7d",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                barmode="overlay",
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                opacity=0.7,
            )
            fig_r7 = apply_custom_theme(
                fig_r7,
                title="7-Day Cumulative Precipitation Distribution",
                x_title="7-Day Precipitation (mm)",
                y_title="Frequency",
                height=320,
            )
            st.plotly_chart(fig_r7, use_container_width=True)

    with domain_tabs[1]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Topographic Derivatives: SRTM 30m & MERIT Hydro</div>
                <div class="card-description">Elevation, slope, aspect, curvature, terrain ruggedness (TRI), and topographic wetness (TWI). Note the strong slope separation arising from the &lt;5° negative sampling constraint.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tp1, tp2 = st.columns(2)
        with tp1:
            fig_slope = px.box(
                df_p3.dropna(subset=["slope"]),
                x=df_p3.dropna(subset=["slope"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="slope",
                color=df_p3.dropna(subset=["slope"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_slope = apply_custom_theme(
                fig_slope,
                title="Slope Angle (Degrees) — Demonstrating Terrain Confound",
                x_title="Observation Class",
                y_title="Slope (°)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_slope, use_container_width=True)

        with tp2:
            fig_elev = px.box(
                df_p3.dropna(subset=["elevation"]),
                x=df_p3.dropna(subset=["elevation"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="elevation",
                color=df_p3.dropna(subset=["elevation"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_elev = apply_custom_theme(
                fig_elev,
                title="Elevation (SRTM 30m, meters)",
                x_title="Observation Class",
                y_title="Elevation (m)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_elev, use_container_width=True)

        tp3, tp4 = st.columns(2)
        with tp3:
            fig_tri = px.box(
                df_p3.dropna(subset=["tri"]),
                x=df_p3.dropna(subset=["tri"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="tri",
                color=df_p3.dropna(subset=["tri"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_tri = apply_custom_theme(
                fig_tri,
                title="Terrain Ruggedness Index (TRI)",
                x_title="Observation Class",
                y_title="TRI (m)",
                height=340,
                show_legend=False,
            )
            st.plotly_chart(fig_tri, use_container_width=True)

        with tp4:
            fig_twi = px.box(
                df_p3.dropna(subset=["twi"]),
                x=df_p3.dropna(subset=["twi"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="twi",
                color=df_p3.dropna(subset=["twi"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_twi = apply_custom_theme(
                fig_twi,
                title="Topographic Wetness Index (TWI)",
                x_title="Observation Class",
                y_title="TWI",
                height=340,
                show_legend=False,
            )
            st.plotly_chart(fig_twi, use_container_width=True)

    with domain_tabs[2]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Sentinel-1 Synthetic Aperture Radar (SAR) Backscatter & Change</div>
                <div class="card-description">All-weather C-band radar backscatter (VV and VH post-event) and pre/post differential backscatter (ΔVV, ΔVH) capturing surface structural disturbance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sar1, sar2 = st.columns(2)
        with sar1:
            fig_dvh = px.box(
                df_p3.dropna(subset=["dVH"]),
                x=df_p3.dropna(subset=["dVH"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="dVH",
                color=df_p3.dropna(subset=["dVH"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_dvh = apply_custom_theme(
                fig_dvh,
                title="Cross-Polarization Change ΔVH (dB)",
                x_title="Observation Class",
                y_title="ΔVH = VH_post - VH_pre (dB)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_dvh, use_container_width=True)

        with sar2:
            fig_dvv = px.box(
                df_p3.dropna(subset=["dVV"]),
                x=df_p3.dropna(subset=["dVV"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="dVV",
                color=df_p3.dropna(subset=["dVV"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_dvv = apply_custom_theme(
                fig_dvv,
                title="Co-Polarization Change ΔVV (dB)",
                x_title="Observation Class",
                y_title="ΔVV = VV_post - VV_pre (dB)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_dvv, use_container_width=True)

    with domain_tabs[3]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Sentinel-2 Surface Reflectance Multispectral Indices</div>
                <div class="card-description">NDVI (vegetation vigour), NDWI (water/moisture), and BSI (bare soil) computed from cloud-masked surface reflectance. Note the transparent display of 80 missing observations.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        opt1, opt2 = st.columns(2)
        with opt1:
            fig_ndvi = px.box(
                df_p3.dropna(subset=["NDVI"]),
                x=df_p3.dropna(subset=["NDVI"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="NDVI",
                color=df_p3.dropna(subset=["NDVI"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_ndvi = apply_custom_theme(
                fig_ndvi,
                title="Normalized Difference Vegetation Index (NDVI)",
                x_title="Observation Class",
                y_title="NDVI [-1, 1]",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_ndvi, use_container_width=True)

        with opt2:
            fig_bsi = px.box(
                df_p3.dropna(subset=["BSI"]),
                x=df_p3.dropna(subset=["BSI"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                y="BSI",
                color=df_p3.dropna(subset=["BSI"])["label"].map({1: "Landslide Event", 0: "Stable Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Stable Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_bsi = apply_custom_theme(
                fig_bsi,
                title="Bare Soil Index (BSI)",
                x_title="Observation Class",
                y_title="BSI [-1, 1]",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_bsi, use_container_width=True)

    with domain_tabs[4]:
        st.plotly_chart(
            render_correlation_heatmap(df_p3, FULL_FEATURE_COLS),
            use_container_width=True,
        )

