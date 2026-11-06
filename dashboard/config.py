"""Configuration module for Himalayan Landslide Intelligence Dashboard.

Defines global file paths, bounding boxes, feature lists, color schemes,
and default settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Data File Paths
RAW_LANDSLIDES_PATH = BASE_DIR / "Landslide_Reports.csv"
PHASE2_CLEANED_PATH = OUTPUTS_DIR / "phase2_cleaned_dataset.csv"
PHASE3_FEATURES_PATH = OUTPUTS_DIR / "phase3_feature_table.csv"
PHASE4_METRICS_PATH = OUTPUTS_DIR / "phase4_metrics_comparison.csv"
PHASE4_OOF_PATH = OUTPUTS_DIR / "phase4_oof_predictions.csv"
PHASE4_MODELS_PATH = OUTPUTS_DIR / "phase4_final_models.joblib"
PHASE5_CHECKPOINT_PATH = OUTPUTS_DIR / "phase5_dynamic_grid_checkpoint_0.5deg.csv"
PHASE5_GRID_PATH = OUTPUTS_DIR / "phase5_susceptibility_grid.csv"
PHASE5_MAPS_PNG_PATH = OUTPUTS_DIR / "phase5_susceptibility_maps.png"

# Geographic Domain: Himalayan Belt
ROI_BOUNDS = {
    "lat_min": 26.0,
    "lat_max": 35.5,
    "lon_min": 74.0,
    "lon_max": 97.0,
}

HIMALAYAN_CENTER = {"lat": 30.5, "lon": 85.5}

# Temporal Domain
S2_START = "2017-01-01"
S2_END = "2023-12-31"

# Feature Sets
TOPOGRAPHIC_FEATURES = [
    "elevation",
    "slope",
    "aspect",
    "curvature",
    "tri",
    "twi",
]

OPTICAL_FEATURES = [
    "BSI",
    "NDVI",
    "NDWI",
]

SAR_FEATURES = [
    "VH_post",
    "VH_pre",
    "VV_post",
    "VV_pre",
    "dVH",
    "dVV",
]

RAINFALL_FEATURES = [
    "API",
    "rain_1d",
    "rain_7d",
    "rain_30d",
]

FULL_FEATURE_COLS = [
    "aspect",
    "curvature",
    "elevation",
    "slope",
    "tri",
    "twi",
    "BSI",
    "NDVI",
    "NDWI",
    "spectral_window_days",
    "VH_post",
    "VV_post",
    "dVH",
    "dVV",
    "API",
    "rain_1d",
    "rain_7d",
    "rain_30d",
]

REDUCED_FEATURE_COLS = [
    "aspect",
    "curvature",
    "elevation",
    "twi",
    "BSI",
    "NDVI",
    "NDWI",
    "spectral_window_days",
    "VH_post",
    "VV_post",
    "dVH",
    "dVV",
    "API",
    "rain_1d",
    "rain_7d",
    "rain_30d",
]

# Color Palette (Dark Navy, Blue/Purple Accents, Cyan Highlights)
THEME_COLORS = {
    "background": "#0B132B",
    "card_bg": "#1C2541",
    "card_border": "#2A3B5C",
    "sidebar_bg": "#0D1B2A",
    "primary": "#48E5C2",       # Neon Cyan / Aqua
    "secondary": "#6C63FF",     # Royal Purple / Indigo
    "accent_blue": "#3A86FF",   # Vivid Blue
    "accent_purple": "#7B2CBF", # Deep Purple
    "warning": "#FFB703",       # Amber Warning
    "danger": "#FF4D6D",        # Coral Red
    "success": "#38B000",       # Vibrant Green
    "text_primary": "#FFFFFF",  # Pure White
    "text_secondary": "#A0AEC0",# Muted Silver
    "grid_color": "rgba(255, 255, 255, 0.08)",
}

# Susceptibility Tiers
SUSCEPTIBILITY_TIERS = [
    {"label": "Very Low", "min": 0.0, "max": 0.2, "color": "#2EC4B6"},
    {"label": "Low", "min": 0.2, "max": 0.4, "color": "#70D6FF"},
    {"label": "Moderate", "min": 0.4, "max": 0.6, "color": "#FFD166"},
    {"label": "High", "min": 0.6, "max": 0.8, "color": "#FF9F1C"},
    {"label": "Very High", "min": 0.8, "max": 1.0, "color": "#E71D36"},
]
