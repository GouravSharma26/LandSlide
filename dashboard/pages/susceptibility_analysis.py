"""Page 6: Himalayan Regional Susceptibility & Uncertainty Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any

from dashboard.utils.data_loader import load_phase5_grid
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.maps import render_susceptibility_grid_map
from dashboard.config import THEME_COLORS, SUSCEPTIBILITY_TIERS
from dashboard.utils.metrics import compute_susceptibility_statistics
from dashboard.utils.visualization import apply_custom_theme


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 6: Susceptibility Analysis."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge">Regional Susceptibility</div>
            <h1 class="brand-title">HIMALAYAN SUSCEPTIBILITY & UNCERTAINTY</h1>
            <div class="brand-subtitle">Macro-Scale Spatial Susceptibility Inference at 0.5° Resolution with Model Uncertainty</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_grid = load_phase5_grid()
    if df_grid.empty:
        st.warning("Susceptibility grid not found. Verify outputs/phase5_susceptibility_grid.csv.")
        return

    # Mandatory Resolution Honesty Callout
    st.markdown(
        """
        <div class="alert-box alert-warning">
            <span style="font-size: 1.25rem;">⚠️</span>
            <div>
                <b>Resolution & Scope Disclosure:</b> The susceptibility grid displayed here is computed at <b>0.5-degree spatial resolution (~55 km cell width)</b>, spanning 765 valid cells across the Himalayan arc. This represents macro-regional susceptibility suitable for synoptic prioritization, <b>NOT a high-resolution 30-metre slope stability raster</b>. Local slope-scale landslides cannot be pin-pointed from this macro grid.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Susceptibility Stats
    stats = compute_susceptibility_statistics(df_grid)

    render_kpi_row([
        {"label": "Total Grid Cells", "value": f"{stats['total_cells']:,}", "subtext": "0.5° spacing (~55 km)", "highlight_color": "blue", "icon": "🌐"},
        {"label": "Mean Susceptibility", "value": f"{stats['mean_susceptibility']:.3f}", "subtext": f"Median: {stats['median_susceptibility']:.3f}", "highlight_color": "cyan", "icon": "📊"},
        {"label": "High-Risk Cells (≥0.6)", "value": f"{stats['high_risk_cells_pct']:.1f}%", "subtext": f"{int(stats['total_cells'] * stats['high_risk_cells_pct'] / 100)} cells", "highlight_color": "danger", "icon": "🔥"},
        {"label": "Mean Tree Uncertainty", "value": f"{stats['mean_tree_uncertainty']:.3f}", "subtext": f"Mean Disagreement: {stats['mean_model_disagreement']:.3f}", "highlight_color": "purple", "icon": "🎯"},
    ])

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Layer & Category Filtering Controls
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">🎛️ Susceptibility Layer Selection & Risk Zonation Filter</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns([1.5, 1.5, 1])

    with f1:
        layer_option = st.selectbox(
            "Select Map Layer",
            [
                ("susceptibility", "Ensemble Susceptibility (RF + XGB) / 2"),
                ("susceptibility_rf", "Random Forest Susceptibility"),
                ("susceptibility_xgb", "XGBoost Susceptibility"),
                ("rf_tree_uncertainty", "Tree Uncertainty (RF std dev across 500 trees)"),
                ("model_disagreement", "Model Disagreement (|RF - XGB|)"),
            ],
            format_func=lambda x: x[1],
        )[0]

    with f2:
        tier_options = ["All Categories"] + [t["label"] for t in SUSCEPTIBILITY_TIERS]
        selected_tier = st.selectbox("Filter by Susceptibility Category", tier_options)

    with f3:
        min_cutoff = st.slider("Minimum Score Cutoff", 0.0, 1.0, 0.0, 0.05)

    # Apply filters to grid
    plot_grid = df_grid.copy()
    if selected_tier != "All Categories":
        plot_grid = plot_grid[plot_grid["susceptibility_tier"] == selected_tier]
    if min_cutoff > 0:
        plot_grid = plot_grid[plot_grid[layer_option] >= min_cutoff]

    # Map Heatmap
    color_scale = "Purples" if "uncertainty" in layer_option else "Blues" if "disagreement" in layer_option else "YlOrRd"
    st.plotly_chart(
        render_susceptibility_grid_map(
            plot_grid,
            value_column=layer_option,
            title=f"Himalayan Regional Susceptibility — {layer_option.replace('_', ' ').title()}",
            color_scale=color_scale,
            height=620,
        ),
        use_container_width=True,
    )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Category Distribution Breakdown & Disagreement Scatter
    c_hist, c_scatter = st.columns(2)

    with c_hist:
        tier_counts = df_grid["susceptibility_tier"].value_counts()[
            [t["label"] for t in SUSCEPTIBILITY_TIERS]
        ].reset_index()
        tier_counts.columns = ["Tier", "Count"]

        fig_tiers = px.bar(
            tier_counts,
            x="Tier",
            y="Count",
            text="Count",
            color="Tier",
            color_discrete_map={t["label"]: t["color"] for t in SUSCEPTIBILITY_TIERS},
        )
        fig_tiers = apply_custom_theme(
            fig_tiers,
            title="Susceptibility Zonation Distribution (765 Cells)",
            x_title="Susceptibility Tier",
            y_title="Grid Cell Count",
            height=360,
            show_legend=False,
        )
        st.plotly_chart(fig_tiers, use_container_width=True)

    with c_scatter:
        fig_scatter = px.scatter(
            df_grid,
            x="susceptibility_rf",
            y="susceptibility_xgb",
            color="model_disagreement",
            color_continuous_scale="Viridis",
            hover_data=["latitude", "longitude", "susceptibility"],
        )
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="rgba(255, 255, 255, 0.4)", dash="dash"),
        )
        fig_scatter = apply_custom_theme(
            fig_scatter,
            title="Model Concordance: Random Forest vs. XGBoost",
            x_title="Random Forest Susceptibility",
            y_title="XGBoost Susceptibility",
            height=360,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Susceptibility Legend and Tiers Reference
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">📖 Susceptibility Zonation Legend & Interpretation Guidelines</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; margin-top: 1rem;">
                <div style="background: rgba(46, 196, 182, 0.15); border-left: 4px solid #2EC4B6; padding: 0.6rem; border-radius: 4px;">
                    <b style="color: #2EC4B6;">Very Low [0.0 – 0.2]</b><br>
                    <span style="font-size: 0.78rem; color: #94A3B8;">Stable plateau or low plain; minimal rainfall triggering hazard.</span>
                </div>
                <div style="background: rgba(112, 214, 255, 0.15); border-left: 4px solid #70D6FF; padding: 0.6rem; border-radius: 4px;">
                    <b style="color: #70D6FF;">Low [0.2 – 0.4]</b><br>
                    <span style="font-size: 0.78rem; color: #94A3B8;">Moderate slope with low rainfall accumulation history.</span>
                </div>
                <div style="background: rgba(255, 209, 102, 0.15); border-left: 4px solid #FFD166; padding: 0.6rem; border-radius: 4px;">
                    <b style="color: #FFD166;">Moderate [0.4 – 0.6]</b><br>
                    <span style="font-size: 0.78rem; color: #94A3B8;">Steep terrain subjected to intermittent monsoon downpours.</span>
                </div>
                <div style="background: rgba(255, 159, 28, 0.15); border-left: 4px solid #FF9F1C; padding: 0.6rem; border-radius: 4px;">
                    <b style="color: #FF9F1C;">High [0.6 – 0.8]</b><br>
                    <span style="font-size: 0.78rem; color: #94A3B8;">Active Himalayan thrust zones with heavy antecedent moisture.</span>
                </div>
                <div style="background: rgba(231, 29, 54, 0.15); border-left: 4px solid #E71D36; padding: 0.6rem; border-radius: 4px;">
                    <b style="color: #E71D36;">Very High [0.8 – 1.0]</b><br>
                    <span style="font-size: 0.78rem; color: #94A3B8;">Acute terrain instability combined with extreme precipitation.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

