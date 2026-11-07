"""Automated tests for evaluation metrics, curves, and confusion matrix computations."""

import pytest
import numpy as np
import pandas as pd
from dashboard.utils.data_loader import load_phase4_oof, load_phase5_grid
from dashboard.utils.metrics import (
    compute_roc_curve,
    compute_pr_curve,
    compute_confusion_matrix_at_threshold,
    compute_susceptibility_statistics,
)
from dashboard.components.charts import render_model_comparison_bar_chart
from dashboard.utils.data_loader import load_detection_metrics, load_susceptibility_metrics
import plotly.graph_objects as go


def test_compute_roc_curve():
    """Verifies ROC curve calculation on out-of-fold predictions."""
    oof = load_phase4_oof()
    y_true = oof["label"].values
    p_xgb = oof["oof_xgb_reduced"].values

    roc_res = compute_roc_curve(y_true, p_xgb)
    assert "fpr" in roc_res
    assert "tpr" in roc_res
    assert "auroc" in roc_res
    assert 0.95 <= roc_res["auroc"] <= 1.0, f"Unexpected AUROC: {roc_res['auroc']}"


def test_compute_pr_curve():
    """Verifies PR curve calculation on out-of-fold predictions."""
    oof = load_phase4_oof()
    y_true = oof["label"].values
    p_xgb = oof["oof_xgb_reduced"].values

    pr_res = compute_pr_curve(y_true, p_xgb)
    assert "precision" in pr_res
    assert "recall" in pr_res
    assert "prauc" in pr_res
    assert 0.95 <= pr_res["prauc"] <= 1.0, f"Unexpected PR-AUC: {pr_res['prauc']}"


def test_compute_confusion_matrix_at_threshold():
    """Verifies confusion matrix properties at default 0.5 threshold."""
    oof = load_phase4_oof()
    y_true = oof["label"].values
    p_xgb = oof["oof_xgb_reduced"].values

    cm_res = compute_confusion_matrix_at_threshold(y_true, p_xgb, threshold=0.5)
    assert cm_res["tn"] + cm_res["fp"] + cm_res["fn"] + cm_res["tp"] == 254
    assert 0.85 <= cm_res["f1"] <= 1.0
    assert 0.85 <= cm_res["accuracy"] <= 1.0


def test_compute_susceptibility_statistics():
    """Verifies descriptive statistics on susceptibility grid."""
    grid = load_phase5_grid()
    stats = compute_susceptibility_statistics(grid)
    assert stats["total_cells"] == 765
    assert 0.0 < stats["mean_susceptibility"] < 1.0
    assert 0.0 <= stats["mean_tree_uncertainty"] <= 0.5
    assert 0.0 <= stats["mean_model_disagreement"] <= 1.0


def test_render_model_comparison_bar_chart_schemas():
    """Regression test ensuring render_model_comparison_bar_chart handles both
    standard tidy metrics schema (detection_metrics / susceptibility_metrics)
    and legacy wide schema without raising KeyError: 'rf_auroc'.
    """
    # 1. Primary failure case: det_metrics from detection_metrics.csv
    det = load_detection_metrics()
    assert not det.empty, "Detection metrics table should be populated"
    fig_det = render_model_comparison_bar_chart(det)
    assert isinstance(fig_det, go.Figure)
    assert len(fig_det.data) == 2  # RF and XGBoost traces

    # 2. Susceptibility metrics schema
    susc = load_susceptibility_metrics()
    assert not susc.empty, "Susceptibility metrics table should be populated"
    fig_susc = render_model_comparison_bar_chart(susc)
    assert isinstance(fig_susc, go.Figure)
    assert len(fig_susc.data) == 2

    # 3. Legacy wide schema with rf_auroc / xgb_auroc
    legacy_df = pd.DataFrame([{
        "feature_set": "REDUCED",
        "rf_auroc": 0.955, "rf_prauc": 0.961, "rf_f1": 0.885,
        "xgb_auroc": 0.958, "xgb_prauc": 0.963, "xgb_f1": 0.890
    }])
    fig_legacy = render_model_comparison_bar_chart(legacy_df)
    assert isinstance(fig_legacy, go.Figure)
    assert len(fig_legacy.data) == 2

    # 4. Empty dataframe graceful handling
    fig_empty = render_model_comparison_bar_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


