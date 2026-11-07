"""Configuration module for Himalayan Landslide Intelligence Research Dashboard.

Defines global file paths, bounding boxes, feature lists, color schemes,
and standardized artifact references.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Artifact Subdirectories
COMMON_DIR = ARTIFACTS_DIR / "common"
DETECTION_DIR = ARTIFACTS_DIR / "detection"
SUSCEPTIBILITY_DIR = ARTIFACTS_DIR / "susceptibility"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

# Standardized Artifact Paths
DATASET_SUMMARY_PATH = COMMON_DIR / "dataset_summary.json"
PREPROCESSING_FUNNEL_PATH = COMMON_DIR / "preprocessing_funnel.csv"
FEATURE_MISSINGNESS_PATH = COMMON_DIR / "feature_missingness.csv"
FEATURE_CORRELATION_PATH = COMMON_DIR / "feature_correlation.csv"
VIF_REPORT_PATH = COMMON_DIR / "vif_report.csv"

# Workflow A Artifacts
DETECTION_FEATURE_TABLE_PATH = DETECTION_DIR / "detection_feature_table.csv"
DETECTION_METRICS_PATH = DETECTION_DIR / "detection_metrics.csv"
DETECTION_FOLD_METRICS_PATH = DETECTION_DIR / "detection_fold_metrics.csv"
DETECTION_PREDICTIONS_PATH = DETECTION_DIR / "detection_predictions.csv"
DETECTION_CONFUSION_MATRIX_PATH = DETECTION_DIR / "detection_confusion_matrix.csv"
DETECTION_FEATURE_IMPORTANCE_PATH = DETECTION_DIR / "detection_feature_importance.csv"
DETECTION_SHAP_SUMMARY_PATH = DETECTION_DIR / "detection_shap_summary.csv"
DETECTION_MODELS_PATH = DETECTION_DIR / "detection_models.joblib"

# Workflow B Artifacts
SUSCEPTIBILITY_FEATURE_TABLE_PATH = SUSCEPTIBILITY_DIR / "susceptibility_feature_table.csv"
SUSCEPTIBILITY_METRICS_PATH = SUSCEPTIBILITY_DIR / "susceptibility_metrics.csv"
SUSCEPTIBILITY_FOLD_METRICS_PATH = SUSCEPTIBILITY_DIR / "susceptibility_fold_metrics.csv"
SUSCEPTIBILITY_PREDICTIONS_PATH = SUSCEPTIBILITY_DIR / "susceptibility_predictions.csv"
SUSCEPTIBILITY_FEATURE_IMPORTANCE_PATH = SUSCEPTIBILITY_DIR / "susceptibility_feature_importance.csv"
SUSCEPTIBILITY_SHAP_SUMMARY_PATH = SUSCEPTIBILITY_DIR / "susceptibility_shap_summary.csv"
SUSCEPTIBILITY_GRID_PATH = SUSCEPTIBILITY_DIR / "susceptibility_grid_0.5deg.csv"
SUSCEPTIBILITY_MODELS_PATH = SUSCEPTIBILITY_DIR / "susceptibility_models.joblib"

# Metadata Artifacts
FEATURE_SCHEMA_PATH = METADATA_DIR / "feature_schema.json"
EXPERIMENT_METADATA_PATH = METADATA_DIR / "experiment_metadata.json"
SPATIAL_DISTANCE_REPORT_PATH = METADATA_DIR / "spatial_fold_distance_report.json"
SPATIAL_SENSITIVITY_PATH = METADATA_DIR / "spatial_purging_sensitivity.csv"
V2_DISTRIBUTION_VAL_PATH = METADATA_DIR / "v2_candidate_distribution_validation.json"

# Historical / Raw Paths
RAW_LANDSLIDES_PATH = BASE_DIR / "Landslide_Reports.csv"
PHASE2_CLEANED_PATH = OUTPUTS_DIR / "phase2_cleaned_dataset.csv"
PHASE3_FEATURES_PATH = OUTPUTS_DIR / "phase3_feature_table.csv"
PHASE4_METRICS_PATH = OUTPUTS_DIR / "phase4_metrics_comparison.csv"
PHASE4_OOF_PATH = OUTPUTS_DIR / "phase4_oof_predictions.csv"
PHASE4_MODELS_PATH = OUTPUTS_DIR / "phase4_final_models.joblib"
PHASE5_GRID_PATH = OUTPUTS_DIR / "phase5_susceptibility_grid.csv"

# Geographic Domain: Himalayan Belt
ROI_BOUNDS = {
    "lat_min": 26.0,
    "lat_max": 35.5,
    "lon_min": 74.0,
    "lon_max": 97.0,
}

HIMALAYAN_CENTER = {"lat": 30.5, "lon": 85.5}

# Feature Schemas
DETECTION_FEATURES = [
    "elevation", "slope", "aspect", "curvature", "twi",
    "VH_post", "VV_post", "dVH", "dVV",
    "rain_1d", "rain_7d", "rain_30d", "API"
]

SUSCEPTIBILITY_CORE_FEATURES = [
    "elevation", "slope", "aspect", "curvature", "tri", "twi",
    "VH_pre", "VV_pre",
    "rain_30d", "API"
]

SUSCEPTIBILITY_OPTICAL_FEATURES = SUSCEPTIBILITY_CORE_FEATURES + [
    "NDVI", "NDWI", "BSI"
]

SUSCEPTIBILITY_REDUCED_FEATURES = [
    "elevation", "aspect", "curvature", "twi",
    "VH_pre", "VV_pre",
    "rain_30d", "API"
]

# Legacy Feature Aliases (for backward compatibility with test suite)
TOPOGRAPHIC_FEATURES = [
    "elevation", "slope", "aspect", "curvature", "tri", "twi"
]
OPTICAL_FEATURES = [
    "BSI", "NDVI", "NDWI"
]
SAR_FEATURES = [
    "VH_post", "VH_pre", "VV_post", "VV_pre", "dVH", "dVV"
]
RAINFALL_FEATURES = [
    "API", "rain_1d", "rain_7d", "rain_30d"
]
FULL_FEATURE_COLS = [
    "aspect", "curvature", "elevation", "slope", "tri", "twi",
    "BSI", "NDVI", "NDWI", "spectral_window_days",
    "VH_post", "VV_post", "dVH", "dVV",
    "API", "rain_1d", "rain_7d", "rain_30d"
]

REDUCED_FEATURE_COLS = [
    "aspect", "curvature", "elevation", "twi",
    "BSI", "NDVI", "NDWI", "spectral_window_days",
    "VH_post", "VV_post", "dVH", "dVV",
    "API", "rain_1d", "rain_7d", "rain_30d"
]


# Professional Visual Identity Palette
THEME_COLORS = {
    "background": "#0E1117",
    "card_bg": "#1E293B",
    "card_border": "#334155",
    "primary": "#00F0FF",         # Electric Cyan
    "secondary": "#38BDF8",       # Sky Blue
    "accent_blue": "#38BDF8",     # Sky Blue alias
    "accent_purple": "#A855F7",   # Deep Purple
    "positive": "#EF4444",        # Crimson Red (Landslide Hazard)
    "danger": "#EF4444",          # Crimson Red alias for hazard/positive
    "negative": "#10B981",        # Emerald Green (Stable Control)
    "success": "#10B981",         # Emerald Green alias for control
    "warning": "#F59E0B",         # Amber Warning
    "text_main": "#F8FAFC",       # Primary text
    "text_primary": "#F8FAFC",    # Primary text alias
    "text_muted": "#94A3B8",      # Secondary text
    "text_secondary": "#94A3B8",  # Secondary text alias
    "grid_color": "rgba(255, 255, 255, 0.07)", # Chart grid lines
}

# Scientific Status Badges
STATUS_BADGES = {
    "version_1_0": {
        "text": "Version 1.0: Historical Diagnostic Baseline",
        "color": "#38BDF8",
        "desc": "254 historical points with unstratified flat-terrain controls (slope < 5°). Used to evaluate terrain confound."
    },
    "version_2_0": {
        "text": "Version 2.0: EXTRACTION_PENDING_COLAB_GEE",
        "color": "#F59E0B",
        "desc": "Empirically validated hillslope protocol (slope 6.1°-41.4°). Feature extraction prepared for Colab execution."
    },
    "workflow_a": {
        "text": "Workflow A: Experimental Point-Zone Disturbance Assessment",
        "color": "#A855F7",
        "desc": "Post-hoc satellite disturbance classification under ±5km positional uncertainty limitation."
    },
    "workflow_b": {
        "text": "Workflow B: Rainfall-Conditioned Susceptibility (Diagnostic Baseline)",
        "color": "#00F0FF",
        "desc": "Pre-event spatial hazard modeling conditioned on 30-day antecedent CHIRPS precipitation (alpha=0.90)."
    },
    "regional_grid": {
        "text": "Regional Grid: Illustrative Macro-Regional Layer (0.5° Lattice)",
        "color": "#64748B",
        "desc": "Synoptic September 2026 late-monsoon window over 765 valid cells (~55 km). Not a site-scale engineering hazard map."
    }
}
