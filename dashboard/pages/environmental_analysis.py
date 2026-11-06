"""Page 4: Environmental & Remote Sensing Feature Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

from dashboard.utils.data_loader import load_phase3_features, load_vif_report, load_feature_correlation
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.charts import render_correlation_heatmap
from dashboard.config import (
    THEME_COLORS,
    TOPOGRAPHIC_FEATURES,
    OPTICAL_FEATURES,
    SAR_FEATURES,
    RAINFALL_FEATURES,
)
from dashboard.utils.visualization import apply_custom_theme


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 4: Environmental Analysis."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge">Environmental Analytics</div>
            <h1 class="brand-title">MULTISOURCE SATELLITE & TERRAIN ANALYSIS</h1>
            <div class="brand-subtitle">Physical Feature Distributions Across Topographic, Radar SAR, Spectral & Rainfall Domains</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_p3 = load_phase3_features()
    vif_df = load_vif_report()
    corr_df = load_feature_correlation()

    if df_p3.empty:
        st.warning("Phase 3 feature table not found. Verify outputs/phase3_feature_table.csv.")
        return

    # Data Quality Callout regarding optical missingness
    st.markdown(
        """
        <div class="alert-box alert-warning">
            <span style="font-size: 1.25rem;">⚠️</span>
            <div>
                <b>Missingness & Multicollinearity Notice:</b> Sentinel-2 optical indices suffer from <b>31.5% null values (80/254 points)</b> due to heavy monsoon cloud cover. In contrast, all-weather Sentinel-1 C-band SAR and CHIRPS precipitation have &gt;97% completeness. Variance Inflation Factors (VIF) reflect severe collinearity among SAR polarizations, which is safely handled by tree ensembles through fold-isolated feature routing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Domain Tabs
    domain_tabs = st.tabs([
        "🌧️ Rainfall & Hydrology",
        "⛰️ Topographic Derivatives",
        "🛰️ SAR Backscatter (Sentinel-1)",
        "🌿 Optical Indices (Sentinel-2)",
        "🔥 Correlation & VIF Analysis",
    ])

    with domain_tabs[0]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Precipitation Metrics: CHIRPS Daily & Cumulative Rainfall</div>
                <div class="card-description">Comparing 1-day, 7-day, 30-day rainfall and Antecedent Precipitation Index (API, verified alpha=0.90) between landslide events and controls.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rf_col1, rf_col2 = st.columns(2)
        with rf_col1:
            fig_api = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="API",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_api = apply_custom_theme(
                fig_api,
                title="Antecedent Precipitation Index (API, verified alpha=0.90)",
                x_title="Observation Class",
                y_title="API Value (mm)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_api, use_container_width=True)

        with rf_col2:
            fig_r30 = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="rain_30d",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_r30 = apply_custom_theme(
                fig_r30,
                title="30-Day Cumulative Antecedent Rainfall (CHIRPS)",
                x_title="Observation Class",
                y_title="Cumulative Rainfall (mm)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_r30, use_container_width=True)

        rf_col3, rf_col4 = st.columns(2)
        with rf_col3:
            fig_r1 = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="rain_1d",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_r1 = apply_custom_theme(
                fig_r1,
                title="Event-Day Trigger Rainfall (rain_1d) — Detection Only",
                x_title="Observation Class",
                y_title="1-Day Rainfall (mm)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_r1, use_container_width=True)

        with rf_col4:
            fig_r7 = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="rain_7d",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_r7 = apply_custom_theme(
                fig_r7,
                title="7-Day Cumulative Trigger Rainfall (rain_7d) — Detection Only",
                x_title="Observation Class",
                y_title="7-Day Rainfall (mm)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_r7, use_container_width=True)

    with domain_tabs[1]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Topographic Features: SRTM 30m Digital Elevation Derivatives</div>
                <div class="card-description">Visualizing slope, elevation, terrain ruggedness (TRI), and topographic wetness (TWI).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tp_col1, tp_col2 = st.columns(2)
        with tp_col1:
            fig_slope = px.violin(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="slope",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                box=True,
                points="all",
            )
            fig_slope = apply_custom_theme(
                fig_slope,
                title="Hillslope Gradient Distribution (Slope in Degrees)",
                x_title="Observation Class",
                y_title="Slope (°)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_slope, use_container_width=True)

        with tp_col2:
            fig_elev = px.violin(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="elevation",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                box=True,
                points="all",
            )
            fig_elev = apply_custom_theme(
                fig_elev,
                title="Surface Elevation Distribution (SRTM 30m)",
                x_title="Observation Class",
                y_title="Elevation (meters)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_elev, use_container_width=True)

    with domain_tabs[2]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Sentinel-1 C-Band SAR: Pre/Post Backscatter & Change Deltas</div>
                <div class="card-description">Interrogating pre-event, post-event backscatter and differential change (dVH, dVV in dB).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sar_col1, sar_col2 = st.columns(2)
        with sar_col1:
            fig_dvh = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="dVH",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_dvh = apply_custom_theme(
                fig_dvh,
                title="Differential Cross-Polarization (dVH = VH_post - VH_pre)",
                x_title="Observation Class",
                y_title="Delta VH (dB)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_dvh, use_container_width=True)

        with sar_col2:
            fig_dvv = px.box(
                df_p3,
                x=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="dVV",
                color=df_p3["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_dvv = apply_custom_theme(
                fig_dvv,
                title="Differential Co-Polarization (dVV = VV_post - VV_pre)",
                x_title="Observation Class",
                y_title="Delta VV (dB)",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_dvv, use_container_width=True)

    with domain_tabs[3]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Sentinel-2 Optical Multispectral Indices (NDVI, NDWI, BSI)</div>
                <div class="card-description">Pre-event vegetation and moisture indices (exhibiting 31.5% monsoon cloud obscuration).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            fig_ndvi = px.box(
                df_p3.dropna(subset=["NDVI"]),
                x=df_p3.dropna(subset=["NDVI"])["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="NDVI",
                color=df_p3.dropna(subset=["NDVI"])["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_ndvi = apply_custom_theme(
                fig_ndvi,
                title="Normalized Difference Vegetation Index (NDVI)",
                x_title="Observation Class",
                y_title="NDVI Score [-1.0 to 1.0]",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_ndvi, use_container_width=True)

        with opt_col2:
            fig_ndwi = px.box(
                df_p3.dropna(subset=["NDWI"]),
                x=df_p3.dropna(subset=["NDWI"])["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                y="NDWI",
                color=df_p3.dropna(subset=["NDWI"])["label"].map({1: "Landslide Event", 0: "Diagnostic Control"}),
                color_discrete_map={"Landslide Event": THEME_COLORS["danger"], "Diagnostic Control": THEME_COLORS["primary"]},
                points="all",
            )
            fig_ndwi = apply_custom_theme(
                fig_ndwi,
                title="Normalized Difference Water Index (NDWI)",
                x_title="Observation Class",
                y_title="NDWI Score [-1.0 to 1.0]",
                height=380,
                show_legend=False,
            )
            st.plotly_chart(fig_ndwi, use_container_width=True)

    with domain_tabs[4]:
        st.markdown(
            """
            <div class="content-card">
                <div class="card-title">Collinearity & Multicollinearity Diagnostics (Pearson Correlation & VIF)</div>
                <div class="card-description">Descriptive analysis separating collinear candidate features from fold-isolated tree preprocessing.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_vif, c_corr = st.columns([1, 1.4])
        with c_vif:
            st.markdown("#### Variance Inflation Factor (VIF) Audit")
            st.caption("Descriptive exploratory audit; tree models do not drop features on VIF.")
            if not vif_df.empty:
                st.dataframe(vif_df, use_container_width=True, hide_index=True, height=450)
        with c_corr:
            st.markdown("#### Pearson Correlation Heatmap")
            corr_features = ["slope", "elevation", "tri", "twi", "VH_pre", "VV_pre", "VH_post", "VV_post", "dVH", "dVV", "rain_30d", "API"]
            fig_corr = render_correlation_heatmap(df_p3, corr_features)
            st.plotly_chart(fig_corr, use_container_width=True)
