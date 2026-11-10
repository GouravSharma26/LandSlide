"""Data loading module for Himalayan Landslide Intelligence Research Dashboard.

Provides cached readers for standardized research artifacts, models, SHAP explainers,
and legacy diagnostic datasets with full error handling.
"""

import json
import logging
from typing import Dict, Any
import pandas as pd
import streamlit as st
import joblib

from dashboard.config import (
    DATASET_SUMMARY_PATH,
    PREPROCESSING_FUNNEL_PATH,
    FEATURE_MISSINGNESS_PATH,
    FEATURE_CORRELATION_PATH,
    VIF_REPORT_PATH,
    DETECTION_FEATURE_TABLE_PATH,
    DETECTION_METRICS_PATH,
    DETECTION_FOLD_METRICS_PATH,
    DETECTION_PREDICTIONS_PATH,
    DETECTION_CONFUSION_MATRIX_PATH,
    DETECTION_SHAP_SUMMARY_PATH,
    DETECTION_MODELS_PATH,
    SUSCEPTIBILITY_FEATURE_TABLE_PATH,
    SUSCEPTIBILITY_METRICS_PATH,
    SUSCEPTIBILITY_FOLD_METRICS_PATH,
    SUSCEPTIBILITY_PREDICTIONS_PATH,
    SUSCEPTIBILITY_SHAP_SUMMARY_PATH,
    SUSCEPTIBILITY_GRID_PATH,
    SUSCEPTIBILITY_MODELS_PATH,
    FEATURE_SCHEMA_PATH,
    EXPERIMENT_METADATA_PATH,
    SPATIAL_DISTANCE_REPORT_PATH,
    SPATIAL_SENSITIVITY_PATH,
    V2_DISTRIBUTION_VAL_PATH,
    RAW_LANDSLIDES_PATH,
    PHASE2_CLEANED_PATH,
    PHASE3_FEATURES_PATH,
    PHASE4_METRICS_PATH,
    PHASE4_OOF_PATH,
    PHASE4_MODELS_PATH,
    PHASE5_GRID_PATH,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Common Artifact Loaders
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset_summary() -> Dict[str, Any]:
    """Loads dataset summary metadata JSON."""
    if not DATASET_SUMMARY_PATH.exists():
        return {}
    with open(DATASET_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_preprocessing_funnel() -> pd.DataFrame:
    """Loads the 7-stage preprocessing funnel CSV."""
    if not PREPROCESSING_FUNNEL_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREPROCESSING_FUNNEL_PATH)


@st.cache_data(show_spinner=False)
def load_feature_missingness() -> pd.DataFrame:
    """Loads the feature missingness summary CSV."""
    if not FEATURE_MISSINGNESS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_MISSINGNESS_PATH)


@st.cache_data(show_spinner=False)
def load_feature_correlation() -> pd.DataFrame:
    """Loads the numerical feature correlation matrix CSV."""
    if not FEATURE_CORRELATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_CORRELATION_PATH, index_col=0)


@st.cache_data(show_spinner=False)
def load_vif_report() -> pd.DataFrame:
    """Loads descriptive Variance Inflation Factor report CSV."""
    if not VIF_REPORT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(VIF_REPORT_PATH)


# -----------------------------------------------------------------------------
# Workflow A: Event Detection Loaders
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_detection_feature_table() -> pd.DataFrame:
    """Loads the clean detection feature table (13 features + label + coords)."""
    if not DETECTION_FEATURE_TABLE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DETECTION_FEATURE_TABLE_PATH)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_detection_metrics() -> pd.DataFrame:
    """Loads pooled out-of-fold metrics for Workflow A."""
    if not DETECTION_METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DETECTION_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_detection_fold_metrics() -> pd.DataFrame:
    """Loads fold-wise evaluation metrics for Workflow A."""
    if not DETECTION_FOLD_METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DETECTION_FOLD_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_detection_predictions() -> pd.DataFrame:
    """Loads out-of-fold prediction probabilities and class decisions for Detection."""
    if not DETECTION_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DETECTION_PREDICTIONS_PATH)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_detection_confusion_matrix() -> pd.DataFrame:
    """Loads detection confusion matrix counts."""
    if not DETECTION_CONFUSION_MATRIX_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DETECTION_CONFUSION_MATRIX_PATH)


@st.cache_data(show_spinner=False)
def load_detection_shap_summary() -> pd.DataFrame:
    """Loads mean absolute TreeSHAP values for Detection."""
    if not DETECTION_SHAP_SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DETECTION_SHAP_SUMMARY_PATH)


@st.cache_resource(show_spinner=False)
def load_detection_models() -> Dict[str, Any]:
    """Loads serialized Random Forest and XGBoost detection models."""
    if not DETECTION_MODELS_PATH.exists():
        return {}
    return joblib.load(DETECTION_MODELS_PATH)


# -----------------------------------------------------------------------------
# Workflow B: Susceptibility Loaders
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_susceptibility_feature_table() -> pd.DataFrame:
    """Loads the strictly pre-event susceptibility feature table."""
    if not SUSCEPTIBILITY_FEATURE_TABLE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SUSCEPTIBILITY_FEATURE_TABLE_PATH)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_susceptibility_metrics() -> pd.DataFrame:
    """Loads pooled out-of-fold metrics across Core, Optical, and Reduced susceptibility configurations."""
    if not SUSCEPTIBILITY_METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SUSCEPTIBILITY_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_susceptibility_fold_metrics() -> pd.DataFrame:
    """Loads fold-wise evaluation metrics for Susceptibility."""
    if not SUSCEPTIBILITY_FOLD_METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SUSCEPTIBILITY_FOLD_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_susceptibility_predictions() -> pd.DataFrame:
    """Loads out-of-fold prediction probabilities for Susceptibility configurations."""
    if not SUSCEPTIBILITY_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SUSCEPTIBILITY_PREDICTIONS_PATH)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_susceptibility_shap_summary() -> pd.DataFrame:
    """Loads mean absolute TreeSHAP values for Susceptibility."""
    if not SUSCEPTIBILITY_SHAP_SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SUSCEPTIBILITY_SHAP_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_susceptibility_grid() -> pd.DataFrame:
    """Loads the 0.5 deg macro-regional susceptibility grid (765 valid cells)."""
    if not SUSCEPTIBILITY_GRID_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SUSCEPTIBILITY_GRID_PATH)


@st.cache_resource(show_spinner=False)
def load_susceptibility_models() -> Dict[str, Any]:
    """Loads serialized Core, Optical, and Reduced susceptibility models."""
    if not SUSCEPTIBILITY_MODELS_PATH.exists():
        return {}
    return joblib.load(SUSCEPTIBILITY_MODELS_PATH)


# -----------------------------------------------------------------------------
# Metadata Loaders
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_feature_schema() -> Dict[str, Any]:
    """Loads declarative feature schema and prohibited variable lists."""
    if not FEATURE_SCHEMA_PATH.exists():
        return {}
    with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_experiment_metadata() -> Dict[str, Any]:
    """Loads comprehensive experiment metadata JSON."""
    if not EXPERIMENT_METADATA_PATH.exists():
        return {}
    with open(EXPERIMENT_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_spatial_sensitivity() -> pd.DataFrame:
    """Loads spatial purging sensitivity sweep CSV (0 to 437.8 km)."""
    if not SPATIAL_SENSITIVITY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SPATIAL_SENSITIVITY_PATH)


@st.cache_data(show_spinner=False)
def load_spatial_distance_report() -> Dict[str, Any]:
    """Loads pairwise geographic distance and fold purging report."""
    if not SPATIAL_DISTANCE_REPORT_PATH.exists():
        return {}
    with open(SPATIAL_DISTANCE_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_v2_distribution_validation() -> Dict[str, Any]:
    """Loads Version 2.0 candidate distribution validation JSON."""
    if not V2_DISTRIBUTION_VAL_PATH.exists():
        return {}
    with open(V2_DISTRIBUTION_VAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Legacy / Diagnostic Loaders (Preserved for backwards compatibility & tests)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_raw_landslides() -> pd.DataFrame:
    """Loads and standardizes raw NASA COOLR Landslide Reports CSV."""
    if not RAW_LANDSLIDES_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(RAW_LANDSLIDES_PATH, low_memory=False)
        if "Event Date" in df.columns:
            df["event_date_parsed"] = pd.to_datetime(
                df["Event Date"], format="%d/%m/%Y, %H:%M", errors="coerce"
            )
            df["year"] = df["event_date_parsed"].dt.year
            df["month"] = df["event_date_parsed"].dt.month
            df["month_name"] = df["event_date_parsed"].dt.strftime("%b")
        return df
    except Exception as e:
        logger.error(f"Error loading raw landslides: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_phase2_cleaned() -> pd.DataFrame:
    """Loads Phase 2 cleaned dataset."""
    if not PHASE2_CLEANED_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PHASE2_CLEANED_PATH)
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["year"] = df["event_date"].dt.year
        df["month_name"] = df["event_date"].dt.strftime("%b")
        return df
    except Exception as e:
        logger.error(f"Error loading Phase 2 cleaned data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_phase3_features() -> pd.DataFrame:
    """Loads Phase 3 feature table."""
    if not PHASE3_FEATURES_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PHASE3_FEATURES_PATH)
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"Error loading Phase 3 feature table: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_phase4_metrics() -> pd.DataFrame:
    """Loads Phase 4 legacy metrics table."""
    if not PHASE4_METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PHASE4_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_phase4_oof() -> pd.DataFrame:
    """Loads Phase 4 legacy out-of-fold predictions."""
    if not PHASE4_OOF_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PHASE4_OOF_PATH)


@st.cache_data(show_spinner=False)
def load_phase5_grid() -> pd.DataFrame:
    """Loads Phase 5 legacy regional grid."""
    if not PHASE5_GRID_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PHASE5_GRID_PATH)


@st.cache_resource(show_spinner=False)
def load_or_train_models() -> Dict[str, Any]:
    """Loads or provides trained models for legacy compatibility."""
    if SUSCEPTIBILITY_MODELS_PATH.exists() and DETECTION_MODELS_PATH.exists():
        s_m = joblib.load(SUSCEPTIBILITY_MODELS_PATH)
        d_m = joblib.load(DETECTION_MODELS_PATH)
        return {
            "reduced": {"rf": s_m["reduced"]["rf"], "xgb": s_m["reduced"]["xgb"]},
            "core": {"rf": s_m["core"]["rf"], "xgb": s_m["core"]["xgb"]},
            "full": {"rf": d_m["rf"], "xgb": d_m["xgb"]},
            "detection": {"rf": d_m["rf"], "xgb": d_m["xgb"]}
        }
    elif PHASE4_MODELS_PATH.exists():
        try:
            return joblib.load(PHASE4_MODELS_PATH)
        except Exception:
            pass
    return {}


# -----------------------------------------------------------------------------
# Descriptive Alias Loaders
# -----------------------------------------------------------------------------
load_cleaned_observations = load_phase2_cleaned
load_multisource_features = load_phase3_features
load_model_metrics = load_phase4_metrics
load_spatial_oof = load_phase4_oof
load_regional_grid = load_phase5_grid

