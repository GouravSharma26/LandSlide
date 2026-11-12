"""Page 7: Explainability, TreeSHAP Attributions & Spatial Uncertainty Analysis."""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from dashboard.config import THEME_COLORS
from dashboard.utils.data_loader import (
    load_detection_shap_summary,
    load_susceptibility_shap_summary,
    load_spatial_sensitivity,
    load_spatial_distance_report,
)
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.charts import render_spatial_purging_sensitivity_chart
from dashboard.utils.visualization import apply_custom_theme


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 7: Feature Importance & Uncertainty."""
    st.title("Feature Importance & Uncertainty")
    st.caption("TreeSHAP attributions, spatial purging decay curves, and predictive uncertainty metrics.")

    # Load artifacts
    det_shap = load_detection_shap_summary()
    susc_shap = load_susceptibility_shap_summary()
    sens_df = load_spatial_sensitivity()
    dist_report = load_spatial_distance_report()

    # Section KPIs
    render_kpi_row([
        {"label": "Top Detection Driver", "value": "Slope (°)", "subtext": "XGB Mean |SHAP| = 3.827", "highlight_color": "purple"},
        {"label": "Top Susceptibility Driver", "value": "Slope (°)", "subtext": "XGB Mean |SHAP| = 2.931", "highlight_color": "cyan"},
        {"label": "Secondary Susc. Driver", "value": "TRI (m)", "subtext": "Terrain Ruggedness Index", "highlight_color": "cyan"},
        {"label": "Purging Range Evaluated", "value": "437.8 km", "subtext": "Empirical semivariogram range", "highlight_color": "blue"},
    ])

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 1. Feature Importance (TreeSHAP)
    st.markdown("### 1. Feature Importance (TreeSHAP)")
    st.markdown(
        """
        TreeSHAP decomposes the model output into additive feature contributions.
        Below we compare mean absolute SHAP values across Random Forest and XGBoost for both workflows:
        """
    )

    tab_shap_det, tab_shap_susc = st.tabs(["Workflow A: Detection Features", "Workflow B: Susceptibility Core Features"])

    with tab_shap_det:
        if not det_shap.empty:
            c1, c2 = st.columns([1.4, 1])
            with c1:
                det_shap_sorted = det_shap.sort_values(by="xgb_mean_abs_shap", ascending=True)
                fig_shap_det = go.Figure()
                fig_shap_det.add_trace(go.Bar(
                    y=det_shap_sorted["feature"],
                    x=det_shap_sorted["xgb_mean_abs_shap"],
                    orientation="h",
                    name="XGBoost Mean |SHAP|",
                    marker_color=THEME_COLORS["accent_purple"]
                ))
                fig_shap_det.add_trace(go.Bar(
                    y=det_shap_sorted["feature"],
                    x=det_shap_sorted["rf_mean_abs_shap"],
                    orientation="h",
                    name="Random Forest Mean |SHAP|",
                    marker_color=THEME_COLORS["primary"]
                ))
                fig_shap_det.update_layout(barmode="group")
                fig_shap_det = apply_custom_theme(
                    fig_shap_det,
                    title="Workflow A: TreeSHAP Feature Attributions (Detection)",
                    x_title="Mean |SHAP Value| (Impact on Model Decision)",
                    y_title="Predictor Feature",
                    height=450
                )
                st.plotly_chart(fig_shap_det, use_container_width=True)
            with c2:
                st.markdown(
                    """
                    <div style="background: #1E293B; border-left: 3px solid #A855F7; padding: 12px; border-radius: 4px; font-size: 0.80rem; line-height: 1.5; color: #CBD5E1;">
                        <strong style="color: #F8FAFC;">Detection Attribution Analysis:</strong><br>
                        • <strong>Slope & Elevation:</strong> Dominate classification due to the flat-terrain control bias in Version 1.0.<br>
                        • <strong>Post-Event Radar (VH_post):</strong> Third strongest predictor, reflecting microwave backscatter signature.<br>
                        • <strong>Radar Deltas (dVH, dVV):</strong> Exhibit lower attribution (0.002 to 0.087). This occurs because single 10m pixels sampled under ±5 km coordinate uncertainty miss the true scar, diminishing the observable delta signal.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(det_shap[["feature", "xgb_mean_abs_shap", "rf_mean_abs_shap"]], use_container_width=True, hide_index=True)

    with tab_shap_susc:
        if not susc_shap.empty:
            c3, c4 = st.columns([1.4, 1])
            with c3:
                susc_shap_sorted = susc_shap.sort_values(by="xgb_mean_abs_shap", ascending=True)
                fig_shap_susc = go.Figure()
                fig_shap_susc.add_trace(go.Bar(
                    y=susc_shap_sorted["feature"],
                    x=susc_shap_sorted["xgb_mean_abs_shap"],
                    orientation="h",
                    name="XGBoost Mean |SHAP|",
                    marker_color=THEME_COLORS["primary"]
                ))
                fig_shap_susc.add_trace(go.Bar(
                    y=susc_shap_sorted["feature"],
                    x=susc_shap_sorted["rf_mean_abs_shap"],
                    orientation="h",
                    name="Random Forest Mean |SHAP|",
                    marker_color=THEME_COLORS["secondary"]
                ))
                fig_shap_susc.update_layout(barmode="group")
                fig_shap_susc = apply_custom_theme(
                    fig_shap_susc,
                    title="Workflow B: TreeSHAP Feature Attributions (Susceptibility Core)",
                    x_title="Mean |SHAP Value| (Impact on Model Decision)",
                    y_title="Predictor Feature",
                    height=400
                )
                st.plotly_chart(fig_shap_susc, use_container_width=True)
            with c4:
                st.markdown(
                    """
                    <div style="background: #1E293B; border-left: 3px solid #00F0FF; padding: 12px; border-radius: 4px; font-size: 0.80rem; line-height: 1.5; color: #CBD5E1;">
                        <strong style="color: #F8FAFC;">Susceptibility Attribution Analysis:</strong><br>
                        • <strong>Terrain Ruggedness (TRI) & Slope:</strong> Absorb &gt;70% of total model gain.<br>
                        • <strong>Pre-Event Radar (VH_pre):</strong> Provides baseline roughness and structural texture.<br>
                        • <strong>Antecedent Rainfall (rain_30d & API):</strong> Provide crucial seasonal conditioning, elevating hazard scores in heavily saturated catchments.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(susc_shap[["feature", "xgb_mean_abs_shap", "rf_mean_abs_shap"]], use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # 2. Spatial Purging Sensitivity Analysis
    st.markdown("### 2. Spatial Purging Sensitivity")
    st.markdown(
        """
        To prevent spatial autocorrelation leakage, we evaluated model performance when strictly purging training samples located within varying buffer distances from test points:
        """
    )

    if not sens_df.empty:
        c_sens_plot, c_sens_txt = st.columns([1.5, 1])
        with c_sens_plot:
            st.plotly_chart(render_spatial_purging_sensitivity_chart(sens_df), use_container_width=True)
        with c_sens_txt:
            st.markdown(
                """
                <div style="background: #1E293B; border-radius: 4px; padding: 12px; font-size: 0.80rem; line-height: 1.5; color: #CBD5E1;">
                    <strong style="color: #F8FAFC;">Sample Depletion vs Geographic Separation Trade-off:</strong><br>
                    • <strong>0 km (Unpurged Baseline):</strong> Minimum distance is 38.7 km. Retains 203.2 training samples/fold.<br>
                    • <strong>50 km Purge:</strong> Purges only 0.4 samples/fold (0.2%). Retains 202.8 samples.<br>
                    • <strong>100 km Purge (Selected Standard):</strong> Guarantees &gt;100 km separation across all folds while purging only 1.6 samples/fold (1.5% loss).<br>
                    • <strong>200 km Purge:</strong> Purges 17.0 samples/fold (8.4% loss).<br>
                    • <strong>437.8 km Purge (Semivariogram Range):</strong> Drops 62.0 training points per fold (30.5% sample loss), depleting small N=254 data and causing fold variance.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show sensitivity summary table
        sens_summary = sens_df.groupby("purge_buffer_km").agg(
            retained_train_avg=("n_train_retained", "mean"),
            purged_train_avg=("n_train_purged", "mean"),
            rf_auroc_mean=("rf_auroc", "mean"),
            xgb_auroc_mean=("xgb_auroc", "mean"),
            rf_f1_mean=("rf_f1", "mean"),
            xgb_f1_mean=("xgb_f1", "mean")
        ).reset_index()
        st.dataframe(sens_summary, use_container_width=True, hide_index=True)

