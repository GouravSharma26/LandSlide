"""Metrics, evaluation curves, and uncertainty calculations for machine learning models."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes False Positive Rate, True Positive Rate, and AUROC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auroc": float(auroc),
    }


def compute_pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes Precision, Recall, and PR-AUC."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "prauc": float(prauc),
    }


def compute_confusion_matrix_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculates confusion matrix and summary scores at a specific probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }


def compute_susceptibility_statistics(grid_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes descriptive and uncertainty statistics for the susceptibility grid."""
    if grid_df.empty or "susceptibility" not in grid_df.columns:
        return {}

    return {
        "mean_susceptibility": float(grid_df["susceptibility"].mean()),
        "median_susceptibility": float(grid_df["susceptibility"].median()),
        "std_susceptibility": float(grid_df["susceptibility"].std()),
        "mean_tree_uncertainty": float(grid_df["rf_tree_uncertainty"].mean()) if "rf_tree_uncertainty" in grid_df.columns else 0.0,
        "mean_model_disagreement": float(grid_df["model_disagreement"].mean()) if "model_disagreement" in grid_df.columns else 0.0,
        "max_model_disagreement": float(grid_df["model_disagreement"].max()) if "model_disagreement" in grid_df.columns else 0.0,
        "high_risk_cells_pct": float((grid_df["susceptibility"] >= 0.6).mean() * 100.0),
        "total_cells": len(grid_df),
    }

