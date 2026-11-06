"""Page 3: Landslide Data Explorer & Inventory Catalog Inspector."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any

from dashboard.utils.data_loader import load_raw_landslides, load_phase2_cleaned
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.config import THEME_COLORS
from dashboard.utils.visualization import apply_custom_theme


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 3: Landslide Data Explorer."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge">Data Explorer</div>
            <h1 class="brand-title">LANDSLIDE DATA & CATALOG EXPLORER</h1>
            <div class="brand-subtitle">Interactive Querying, Inventory Profiling, and Attribution Inspection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_raw = load_raw_landslides()
    df_p2 = load_phase2_cleaned()

    # Scope selector: Raw Global/Himalayan vs Cleaned Phase 2 subset
    scope = st.radio(
        "Select Dataset Scope",
        ["Cleaned Himalayan Observation Set (254 rows)", "Raw Landslide Reports (14,750 rows)"],
        horizontal=True,
    )

    df_active = df_p2.copy() if "Cleaned" in scope else df_raw.copy()

    # Filtering Panel
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">🔍 Multi-Attribute Query Filters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        # Search keyword
        search_query = st.text_input("Search Text (Location, Trigger, Catalog)", placeholder="e.g. Nepal, rain, GLC...")

    with f_col2:
        # Trigger filter
        trigger_col = "Landslide Trigger" if "Landslide Trigger" in df_active.columns else None
        if trigger_col and not df_active.empty:
            triggers = sorted(df_active[trigger_col].dropna().unique().tolist())
            selected_triggers = st.multiselect("Trigger Category", triggers, default=[])
        else:
            selected_triggers = []

    with f_col3:
        # Location Accuracy filter
        acc_col = "Location Accuracy" if "Location Accuracy" in df_active.columns else None
        if acc_col and not df_active.empty:
            accuracies = sorted(df_active[acc_col].dropna().unique().tolist())
            selected_accs = st.multiselect("Location Accuracy", accuracies, default=[])
        else:
            selected_accs = []

    # Apply search and filters
    filtered_df = df_active.copy()

    if search_query:
        mask = False
        for c in filtered_df.select_dtypes(include=["object"]).columns:
            mask = mask | filtered_df[c].astype(str).str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]

    if selected_triggers and trigger_col:
        filtered_df = filtered_df[filtered_df[trigger_col].isin(selected_triggers)]

    if selected_accs and acc_col:
        filtered_df = filtered_df[filtered_df[acc_col].isin(selected_accs)]

    # Metrics on filtered view
    render_kpi_row([
        {"label": "Filtered Records", "value": f"{len(filtered_df):,}", "subtext": f"of {len(df_active):,} total records", "highlight_color": "cyan", "icon": "📊"},
        {"label": "Distinct Triggers", "value": filtered_df[trigger_col].nunique() if trigger_col else "N/A", "subtext": "Reported causal mechanisms", "highlight_color": "purple", "icon": "⚡"},
        {"label": "Catalogs Represented", "value": filtered_df["Imported Event Source Catalog"].nunique() if "Imported Event Source Catalog" in filtered_df.columns else "N/A", "subtext": "Institutional sources", "highlight_color": "blue", "icon": "📁"},
        {"label": "High Accuracy Ratio", "value": f"{(filtered_df[acc_col].isin(['known', '1km', '5km'])).mean():.1%}" if acc_col and len(filtered_df) > 0 else "N/A", "subtext": "Positional precision ≤ 5km", "highlight_color": "cyan", "icon": "🎯"},
    ])

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Detailed Charts
    c1, c2 = st.columns(2)

    with c1:
        # Location Accuracy breakdown
        if acc_col and not filtered_df.empty:
            acc_counts = filtered_df[acc_col].value_counts().reset_index()
            acc_counts.columns = ["Accuracy", "Count"]
            fig_acc = px.bar(
                acc_counts,
                x="Accuracy",
                y="Count",
                text="Count",
                color="Count",
                color_continuous_scale=[[0, "#3A86FF"], [1, "#48E5C2"]],
            )
            fig_acc = apply_custom_theme(
                fig_acc,
                title="Location Accuracy Tier Distribution",
                x_title="Positional Accuracy Window",
                y_title="Record Count",
                height=340,
                show_legend=False,
            )
            st.plotly_chart(fig_acc, use_container_width=True)

    with c2:
        # Imported Event Source Catalog breakdown
        cat_col = "Imported Event Source Catalog" if "Imported Event Source Catalog" in filtered_df.columns else None
        if cat_col and not filtered_df.empty:
            cat_counts = filtered_df[cat_col].value_counts().head(8).reset_index()
            cat_counts.columns = ["Catalog", "Count"]
            fig_cat = px.pie(
                cat_counts,
                names="Catalog",
                values="Count",
                hole=0.45,
                color_discrete_sequence=[THEME_COLORS["primary"], THEME_COLORS["secondary"], THEME_COLORS["accent_blue"], THEME_COLORS["warning"]],
            )
            fig_cat = apply_custom_theme(
                fig_cat,
                title="Imported Event Source Catalog Breakdown",
                height=340,
                show_legend=True,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    # Interactive Data Table
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">📋 Tabular Data Catalog</div>
            <div class="card-description">Inspect full individual rows, sort columns, and export custom filtered subsets.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(filtered_df, use_container_width=True, height=350)

    # Download CSV button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="himalayan_landslides_filtered.csv",
        mime="text/csv",
    )

