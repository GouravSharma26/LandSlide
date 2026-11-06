"""Data loading module for Himalayan Landslide Intelligence Dashboard.

Provides cached readers for project CSV datasets, models, and SHAP explainers
with fallback handlers for version mismatches and missing files.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

from dashboard.config import (
    RAW_LANDSLIDES_PATH,
    PHASE2_CLEANED_PATH,
    PHASE3_FEATURES_PATH,
    PHASE4_METRICS_PATH,
    PHASE4_OOF_PATH,
    PHASE4_MODELS_PATH,
    PHASE5_CHECKPOINT_PATH,
    PHASE5_GRID_PATH,
    FULL_FEATURE_COLS,
    REDUCED_FEATURE_COLS,
)

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def load_raw_landslides() -> pd.DataFrame:
    """Loads and standardizes the raw NASA COOLR Landslide Reports CSV."""
    if not RAW_LANDSLIDES_PATH.exists():
        logger.warning(f"File not found: {RAW_LANDSLIDES_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(RAW_LANDSLIDES_PATH, low_memory=False)
        # Parse dates safely
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
    """Loads the Phase 2 cleaned dataset (positives and hard negatives)."""
    if not PHASE2_CLEANED_PATH.exists():
        logger.warning(f"File not found: {PHASE2_CLEANED_PATH}")
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
    """Loads the Phase 3 feature table (static and dynamic features)."""
    if not PHASE3_FEATURES_PATH.exists():
        logger.warning(f"File not found: {PHASE3_FEATURES_PATH}")
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
    """Loads the Phase 4 model evaluation comparison table."""
    if not PHASE4_METRICS_PATH.exists():
        logger.warning(f"File not found: {PHASE4_METRICS_PATH}")
        return pd.DataFrame()

    try:
        return pd.read_csv(PHASE4_METRICS_PATH)
    except Exception as e:
        logger.error(f"Error loading Phase 4 metrics comparison: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_phase4_oof() -> pd.DataFrame:
    """Loads the Phase 4 spatial out-of-fold predictions."""
    if not PHASE4_OOF_PATH.exists():
        logger.warning(f"File not found: {PHASE4_OOF_PATH}")
        return pd.DataFrame()

    try:
        return pd.read_csv(PHASE4_OOF_PATH)
    except Exception as e:
        logger.error(f"Error loading Phase 4 OOF predictions: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_phase5_checkpoint() -> pd.DataFrame:
    """Loads the Phase 5 intermediate grid checkpoint."""
    if not PHASE5_CHECKPOINT_PATH.exists():
        logger.warning(f"File not found: {PHASE5_CHECKPOINT_PATH}")
        return pd.DataFrame()

    try:
        return pd.read_csv(PHASE5_CHECKPOINT_PATH)
    except Exception as e:
        logger.error(f"Error loading Phase 5 checkpoint: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_phase5_grid() -> pd.DataFrame:
    """Loads the final Phase 5 0.5-degree susceptibility grid."""
    if not PHASE5_GRID_PATH.exists():
        logger.warning(f"File not found: {PHASE5_GRID_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(PHASE5_GRID_PATH)
        # Classify into susceptibility tiers
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
        df["susceptibility_tier"] = pd.cut(
            df["susceptibility"], bins=bins, labels=labels, include_lowest=True
        )
        return df
    except Exception as e:
        logger.error(f"Error loading Phase 5 susceptibility grid: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_or_train_models() -> Dict[str, Any]:
    """Loads the serialized model pipelines or deterministically fits them.

    XGBoost format changes between major versions (e.g. 2.x to 3.x) can cause
    joblib unpickling failures. If unpickling encounters an issue, this function
    re-fits the exact Phase 5 cell 2 models on phase3_feature_table.csv with
    random_state=42 (<1.5s runtime).
    """
    df = load_phase3_features()
    if df.empty:
        return {}

    # Attempt loading from joblib first
    if PHASE4_MODELS_PATH.exists():
        try:
            models_dict = joblib.load(PHASE4_MODELS_PATH)
            # Verify models can predict
            X_test = df[REDUCED_FEATURE_COLS].iloc[:2]
            _ = models_dict["reduced"]["rf"].predict_proba(X_test)
            _ = models_dict["reduced"]["xgb"].predict_proba(X_test)
            return models_dict
        except Exception as e:
            logger.info(f"Joblib load failed ({e}); falling back to deterministic fit.")

    # Re-fit identical estimators as specified in Phase 4 / Phase 5
    y = df["label"].copy()
    models = {}

    for label, cols in [("full", FULL_FEATURE_COLS), ("reduced", REDUCED_FEATURE_COLS)]:
        X = df[cols].copy()

        # Random Forest Pipeline
        rf_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500, class_weight="balanced", random_state=42
                ),
            ),
        ])
        rf_pipe.fit(X, y)

        # XGBoost Classifier
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        xgb_clf = xgb.XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            tree_method="hist",
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
        )
        xgb_clf.fit(X, y)

        models[label] = {
            "rf": rf_pipe,
            "xgb": xgb_clf,
            "feature_cols": cols,
        }

    return models


@st.cache_resource(show_spinner=False)
def compute_shap_explanations(feature_set: str = "reduced") -> Tuple[Any, Any, pd.DataFrame, Any, Any]:
    """Computes TreeExplainer SHAP values for both RF and XGBoost.

    Returns:
        (shap_values_rf, shap_values_xgb, X, rf_model, xgb_model)
    """
    df = load_phase3_features()
    if df.empty:
        return None, None, pd.DataFrame(), None, None

    models = load_or_train_models()
    if not models or feature_set not in models:
        return None, None, pd.DataFrame(), None, None

    cols = models[feature_set]["feature_cols"]
    X = df[cols].copy()

    rf_pipe = models[feature_set]["rf"]
    xgb_clf = models[feature_set]["xgb"]

    # Impute X for RF SHAP (sklearn RF requires imputed array)
    imputer = rf_pipe.named_steps["impute"]
    X_imputed = pd.DataFrame(imputer.transform(X), columns=cols)

    rf_clf = rf_pipe.named_steps["clf"]
    explainer_rf = shap.TreeExplainer(rf_clf)
    shap_vals_rf = explainer_rf.shap_values(X_imputed)

    # If RF returns [n_samples, n_features, 2], select positive class (index 1)
    if isinstance(shap_vals_rf, list):
        shap_vals_rf_pos = shap_vals_rf[1]
    elif hasattr(shap_vals_rf, "ndim") and shap_vals_rf.ndim == 3:
        shap_vals_rf_pos = shap_vals_rf[:, :, 1]
    else:
        shap_vals_rf_pos = shap_vals_rf

    # XGBoost SHAP
    explainer_xgb = shap.TreeExplainer(xgb_clf)
    shap_vals_xgb = explainer_xgb.shap_values(X)

    return shap_vals_rf_pos, shap_vals_xgb, X, rf_clf, xgb_clf

