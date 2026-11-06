"""Page 5: Machine Learning Model Benchmarks, Spatial CV & SHAP Explainability."""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any

from dashboard.utils.data_loader import (
    load_phase4_metrics,
    load_phase4_oof,
    load_or_train_models,
    compute_shap_explanations,
)
from dashboard.components.kpi_cards import render_kpi_row
from dashboard.components.charts import (
    render_roc_curves,
    render_pr_curves,
    render_confusion_matrix_chart,
    render_feature_importance_chart,
    render_shap_summary_chart,
)
from dashboard.utils.metrics import compute_confusion_matrix_at_threshold


def render(filters: Dict[str, Any]) -> None:
    """Renders Page 5: Machine Learning Performance."""
    st.markdown(
        """
        <div class="header-container">
            <div class="brand-badge">Machine Learning Benchmarks</div>
            <h1 class="brand-title">SPATIAL CROSS-VALIDATION & MODEL EXPLAINABILITY</h1>
            <div class="brand-subtitle">Rigorous Spatial Evaluation (Random Forest vs. XGBoost) with SHAP Attribution</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_metrics = load_phase4_metrics()
    df_oof = load_phase4_oof()

    if df_metrics.empty or df_oof.empty:
        st.warning("Model evaluation data not found. Please verify Phase 4 output files.")
        return

    # Critical Methodology Callout
    st.markdown(
        """
        <div class="alert-box alert-warning">
            <span style="font-size: 1.25rem;">⚠️</span>
            <div>
                <b>Scientific Caveat — Negative-Sampling Confound:</b> All metrics displayed below are <b>Spatial Out-of-Fold (OOF)</b> evaluations from 5-fold KMeans clustered cross-validation on projected coordinates (preventing spatial autocorrelation leakage). However, because negative controls were constrained to slope &lt; 5° during Phase 2 sampling, models achieved high discriminability between steep and flat terrain. The <b>Reduced Feature Set</b> (excluding slope and TRI) is presented as the primary benchmark to minimize this terrain confound.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature Set Selector
    feature_set_choice = st.radio(
        "Select Feature Configuration Benchmark",
        ["Reduced Feature Set (16 features — slope & TRI removed)", "Full Feature Set (18 features — includes slope & TRI)"],
        horizontal=True,
    )
    feature_set_key = "reduced" if "Reduced" in feature_set_choice else "full"

    # Filter metrics row
    row_match = df_metrics[df_metrics["feature_set"].str.contains(feature_set_key, case=False, na=False)]
    if row_match.empty:
        row_match = df_metrics.iloc[0:1]

    rf_auc = float(row_match["rf_auroc"].values[0])
    xgb_auc = float(row_match["xgb_auroc"].values[0])
    rf_prauc = float(row_match["rf_prauc"].values[0])
    xgb_prauc = float(row_match["xgb_prauc"].values[0])
    rf_f1 = float(row_match["rf_f1"].values[0])
    xgb_f1 = float(row_match["xgb_f1"].values[0])

    # Top KPI Metrics (Out-of-Fold)
    render_kpi_row([
        {"label": "XGBoost AUROC (OOF)", "value": f"{xgb_auc:.4f}", "subtext": "Spatial 5-Fold CV", "highlight_color": "cyan", "icon": "⚡"},
        {"label": "Random Forest AUROC (OOF)", "value": f"{rf_auc:.4f}", "subtext": "Spatial 5-Fold CV", "highlight_color": "purple", "icon": "🌲"},
        {"label": "XGBoost PR-AUC", "value": f"{xgb_prauc:.4f}", "subtext": "Precision-Recall Area", "highlight_color": "cyan", "icon": "📈"},
        {"label": "XGBoost F1 @ 0.5", "value": f"{xgb_f1:.4f}", "subtext": f"RF F1: {rf_f1:.4f}", "highlight_color": "blue", "icon": "🎯"},
    ])

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # ROC and PR Curves Side-by-Side
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(render_roc_curves(df_oof, feature_set_key), use_container_width=True)
    with c2:
        st.plotly_chart(render_pr_curves(df_oof, feature_set_key), use_container_width=True)

    # Interactive Confusion Matrix & Decision Threshold Slider
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">🎚️ Decision Threshold & Confusion Matrix Analysis</div>
            <div class="card-description">Adjust classification decision threshold to inspect trade-offs between false alarms (FP) and missed landslides (FN) on spatial out-of-fold predictions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    thresh_col, model_col = st.columns([2, 1])
    with thresh_col:
        threshold = st.slider("Classification Threshold (Probability Cutoff)", 0.10, 0.90, 0.50, 0.05)
    with model_col:
        selected_cm_model = st.selectbox("Model for Confusion Matrix", ["XGBoost", "Random Forest"])

    prob_col = f"oof_xgb_{feature_set_key}" if selected_cm_model == "XGBoost" else f"oof_rf_{feature_set_key}"
    cm_dict = compute_confusion_matrix_at_threshold(
        df_oof["label"].values,
        df_oof[prob_col].values,
        threshold=threshold,
    )

    cm_c1, cm_c2 = st.columns([1.3, 1])
    with cm_c1:
        st.plotly_chart(
            render_confusion_matrix_chart(cm_dict, model_name=f"{selected_cm_model} ({feature_set_key.upper()})"),
            use_container_width=True,
        )
    with cm_c2:
        st.markdown(
            f"""
            <div style="background: #1C2541; border: 1px solid #2A3B5C; border-radius: 10px; padding: 1.25rem;">
                <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 0.75rem; font-size: 1.05rem;">
                    Operating Point Diagnostics (Threshold = {threshold:.2f})
                </div>
                <div style="margin-bottom: 0.5rem; color: #E2E8F0;">
                    • <b>Accuracy:</b> <span style="color: #48E5C2;">{cm_dict['accuracy']:.3f}</span>
                </div>
                <div style="margin-bottom: 0.5rem; color: #E2E8F0;">
                    • <b>Precision:</b> <span style="color: #48E5C2;">{cm_dict['precision']:.3f}</span>
                </div>
                <div style="margin-bottom: 0.5rem; color: #E2E8F0;">
                    • <b>Recall (Sensitivity):</b> <span style="color: #48E5C2;">{cm_dict['recall']:.3f}</span>
                </div>
                <div style="margin-bottom: 0.5rem; color: #E2E8F0;">
                    • <b>F1-Score:</b> <span style="color: #6C63FF;">{cm_dict['f1']:.3f}</span>
                </div>
                <div style="margin-top: 1rem; font-size: 0.8rem; color: #94A3B8; border-top: 1px solid #2A3B5C; padding-top: 0.5rem;">
                    Missed Landslides (FN): <b style="color: #FF4D6D;">{cm_dict['fn']}</b> | False Alarms (FP): <b style="color: #FFB703;">{cm_dict['fp']}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Feature Importance & SHAP Interpretability Section
    st.markdown(
        """
        <div class="content-card">
            <div class="card-title">🧠 Explainable AI: Feature Importance & SHAP Attribution</div>
            <div class="card-description">Examining global feature contributions using TreeExplainer SHAP values computed on the 254-point observation set.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models_dict = load_or_train_models()
    if models_dict and feature_set_key in models_dict:
        m_rf = models_dict[feature_set_key]["rf"]
        m_xgb = models_dict[feature_set_key]["xgb"]
        cols = models_dict[feature_set_key]["feature_cols"]

        st.plotly_chart(
            render_feature_importance_chart(m_rf, m_xgb, cols),
            use_container_width=True,
        )

        with st.spinner("Computing SHAP values..."):
            shap_rf, shap_xgb, X_shap, _, _ = compute_shap_explanations(feature_set_key)

        if shap_xgb is not None:
            shap_c1, shap_c2 = st.columns(2)
            with shap_c1:
                st.plotly_chart(
                    render_shap_summary_chart(shap_xgb, X_shap, f"XGBoost ({feature_set_key.upper()})"),
                    use_container_width=True,
                )
            with shap_c2:
                st.plotly_chart(
                    render_shap_summary_chart(shap_rf, X_shap, f"Random Forest ({feature_set_key.upper()})"),
                    use_container_width=True,
                )

    # Benchmarks Comparison Table
    with st.expander("📊 Complete Out-of-Fold Evaluation Matrix (Phase 4 vs Phase 5)", expanded=False):
        st.dataframe(df_metrics, use_container_width=True)

