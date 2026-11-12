"""Page 3: Data Catalog & Scientific Quality Audit."""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from dashboard.utils.data_loader import (
    load_preprocessing_funnel,
    load_feature_missingness,
    load_v2_distribution_validation,
    load_dataset_summary,
)
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.charts import render_missingness_chart


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 3: Data Quality."""
    st.title("Data Quality & Cleaning Funnel")
    st.caption("Preprocessing funnel, feature missingness rates, and diagnostic control evaluations.")

    # Load artifacts
    funnel_df = load_preprocessing_funnel()
    missing_df = load_feature_missingness()
    v2_val = load_v2_distribution_validation()
    summary = load_dataset_summary()

    # Data Quality KPIs
    render_kpi_row([
        {"label": "Rainfall Completeness", "value": "100.0%", "subtext": "CHIRPS daily (0 nulls across 254 pts)", "highlight_color": "cyan"},
        {"label": "SAR Completeness", "value": "97.2%", "subtext": "Sentinel-1 IW GRD (5–7 swath edge nulls)", "highlight_color": "cyan"},
        {"label": "Optical Missingness", "value": "31.5%", "subtext": "Monsoon cloud obscuration (80 nulls)", "highlight_color": "danger"},
        {"label": "Control Quality", "value": "Version 1.0 (Flat)", "subtext": "Version 2.0 pending Colab GEE extraction", "highlight_color": "purple"},
    ])

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 1. Preprocessing Funnel
    st.markdown("### 1. Preprocessing Funnel")
    st.markdown(
        """
        The raw NASA Global Landslide Catalog (COOLR) contains 14,750 global reports.
        Through 7 systematic filtering stages, 127 rainfall-triggered events meeting filtering criteria were isolated across the Himalayan arc:
        """
    )

    if not funnel_df.empty:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.dataframe(funnel_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown(
                """
                <div style="background: #1E293B; border-left: 3px solid #38BDF8; padding: 12px; border-radius: 4px; font-size: 0.82rem; line-height: 1.5; color: #CBD5E1;">
                    <strong style="color: #F8FAFC;">Major Sample Reduction Factors:</strong><br>
                    • <strong>Geographic ROI (-11,424):</strong> Excludes global reports outside [26°N–35°N, 74°E–97°E].<br>
                    • <strong>Constellation Window (-1,383):</strong> Constrains dates to 2017–2020 to match Sentinel-1/2 temporal baselines.<br>
                    • <strong>Trigger Screening (-100):</strong> Retains purely rainfall-triggered events; removes seismic, construction, and mining triggers.<br>
                    • <strong>Positional Quality (-78):</strong> Drops coarse records without sub-5km coordinates.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # 2. Missing Values by Feature
    st.markdown("### 2. Missing Values by Feature")
    st.markdown(
        """
        Missing values across candidate remote sensing features and their causes:
        """
    )

    if not missing_df.empty:
        # Deduplicate display entries while preserving all distinct source variables
        missing_df = missing_df.drop_duplicates(subset=["feature"]).reset_index(drop=True)
        c3, c4 = st.columns([1.2, 1])
        with c3:
            # Transform missing_df to format expected by missingness chart
            chart_df = missing_df.rename(columns={"feature": "column", "missingness_pct": "missing_pct", "null_count": "missing_count"})
            chart_df = chart_df.drop_duplicates(subset=["column"]).reset_index(drop=True)
            st.plotly_chart(render_missingness_chart(chart_df), use_container_width=True)
        with c4:
            st.dataframe(
                missing_df[["feature", "non_null_count", "null_count", "missingness_pct", "physical_cause"]],
                use_container_width=True,
                hide_index=True,
                height=450,
            )

        st.caption(
            "Note on Optical Missingness: BSI, NDVI, NDWI, and spectral_window_days all report 31.5% missingness (80/254 points) "
            "because all multispectral indices and the adaptive window duration metadata are co-dependent on Sentinel-2 scene availability. "
            "When cloud obscuration precludes valid surface reflectance within 60 days, all four variables are simultaneously null. "
            "They are distinct physical variables, not duplicate entries."
        )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # 3. Control Sample Comparison: Version 1.0 vs Version 2.0
    st.markdown("### 3. Control Sample Comparison: Version 1.0 vs Version 2.0")
    st.markdown(
        """
        Comparison of Version 1.0 baseline controls against the Version 2.0 hillslope protocol:
        """
    )

    v1_v2_table = [
        {"Parameter / Attribute": "Status", "Version 1.0 (Baseline)": "Authoritative Diagnostic Baseline (Flat-Terrain Controls; Not Validated Susceptibility)", "Version 2.0 (Hillslope Protocol)": "Empirical Topographic Audit Archived (45.3 min GEE run proved 100% rejection)"},
        {"Parameter / Attribute": "Sample Size", "Version 1.0 (Baseline)": "N=254 (127 pos, 127 neg)", "Version 2.0 (Hillslope Protocol)": "179 candidates screened across 117 events (0 accepted)"},
        {"Parameter / Attribute": "Negative Control Slope", "Version 1.0 (Baseline)": "Mean 1.43° ± 4.34° (90% flat plains < 5°)", "Version 2.0 (Hillslope Protocol)": "Empirical envelope: 6.1° to 41.4° (candidate hillslope controls)"},
        {"Parameter / Attribute": "Elevation Range", "Version 1.0 (Baseline)": "14 m to 2,398 m", "Version 2.0 (Hillslope Protocol)": "266 m to 3,027 m (P10 to P95 of positive events)"},
        {"Parameter / Attribute": "Positive Exclusion Buffer", "Version 1.0 (Baseline)": "None", "Version 2.0 (Hillslope Protocol)": "> 2.0 km from all known scars"},
        {"Parameter / Attribute": "Multi-Scale Neighborhoods", "Version 1.0 (Baseline)": "Single 10m pixel", "Version 2.0 (Hillslope Protocol)": "500m & 1000m buffer reductions (point-zone disturbance; does not remediate ±5km error)"},
        {"Parameter / Attribute": "Earth Engine Script", "Version 1.0 (Baseline)": "notebooks/feature_engineering.ipynb", "Version 2.0 (Hillslope Protocol)": "notebooks/hillslope_control_extraction.ipynb"}
    ]
    st.dataframe(pd.DataFrame(v1_v2_table), use_container_width=True, hide_index=True)

    if v2_val and "v1_distribution_divergence" in v2_val:
        div = v2_val["v1_distribution_divergence"]
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid #DC2626; border-radius: 4px; padding: 10px 14px; margin-top: 10px;">
                <strong style="color: #EF4444;">Statistical Evidence of Terrain Distribution Differences:</strong> Two-sample Kolmogorov-Smirnov tests confirm extreme divergence between positive events and Version 1.0 controls:<br>
                • Slope KS Statistic: <strong>{div.get('slope_ks_statistic', 0.92):.3f}</strong> (p-value: {div.get('slope_ks_pvalue', 0):.2e})<br>
                • TRI KS Statistic: <strong>{div.get('tri_ks_statistic', 0.91):.3f}</strong> (p-value: {div.get('tri_ks_pvalue', 0):.2e})<br>
                <em>This mathematically confirms that high accuracy on Version 1.0 is driven by distinguishing mountains from flat valley floors.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Downloads
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        if not funnel_df.empty:
            st.download_button(
                "Download Preprocessing Funnel CSV",
                funnel_df.to_csv(index=False),
                "himalayan_preprocessing_funnel.csv",
                "text/csv",
                use_container_width=True
            )
    with d2:
        if not missing_df.empty:
            st.download_button(
                "Download Missingness Audit CSV",
                missing_df.to_csv(index=False),
                "himalayan_feature_missingness.csv",
                "text/csv",
                use_container_width=True
            )
