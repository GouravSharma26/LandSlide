"""Automated tests for data loading modules."""

import pytest
import pandas as pd
from dashboard.utils.data_loader import (
    load_raw_landslides,
    load_phase2_cleaned,
    load_phase3_features,
    load_phase4_metrics,
    load_phase4_oof,
    load_phase5_grid,
    load_or_train_models,
)
from dashboard.config import (
    FULL_FEATURE_COLS,
    REDUCED_FEATURE_COLS,
)


def test_load_raw_landslides():
    """Verifies that raw landslide reports load and have required structure."""
    df = load_raw_landslides()
    assert not df.empty, "Raw landslides dataset should not be empty"
    assert len(df) >= 10000, f"Expected >10,000 rows in raw catalog, got {len(df)}"
    assert "Latitude" in df.columns or "latitude" in df.columns
    assert "Longitude" in df.columns or "longitude" in df.columns


def test_load_phase2_cleaned():
    """Verifies that Phase 2 cleaned dataset has 254 balanced observations."""
    df = load_phase2_cleaned()
    assert not df.empty, "Phase 2 dataset should not be empty"
    assert len(df) == 254, f"Expected 254 observations, got {len(df)}"
    assert "label" in df.columns
    assert (df["label"] == 1).sum() == 127, "Expected exactly 127 positive samples"
    assert (df["label"] == 0).sum() == 127, "Expected exactly 127 negative samples"


def test_load_phase3_features():
    """Verifies that Phase 3 feature table contains all expected features."""
    df = load_phase3_features()
    assert not df.empty, "Phase 3 feature table should not be empty"
    assert len(df) == 254, f"Expected 254 rows, got {len(df)}"
    for col in REDUCED_FEATURE_COLS:
        assert col in df.columns, f"Feature '{col}' missing from Phase 3 table"


def test_load_phase4_metrics():
    """Verifies that Phase 4 metrics comparison table loads with correct benchmarks."""
    df = load_phase4_metrics()
    assert not df.empty, "Phase 4 metrics table should not be empty"
    assert "rf_auroc" in df.columns
    assert "xgb_auroc" in df.columns
    assert df["xgb_auroc"].max() > 0.95, "Expected high AUROC benchmark for XGBoost"


def test_load_phase4_oof():
    """Verifies that Phase 4 out-of-fold predictions match 254 observations."""
    df = load_phase4_oof()
    assert not df.empty, "Phase 4 OOF table should not be empty"
    assert len(df) == 254
    assert "oof_rf_reduced" in df.columns
    assert "oof_xgb_reduced" in df.columns


def test_load_phase5_grid():
    """Verifies that Phase 5 regional susceptibility grid loads 765 valid cells."""
    df = load_phase5_grid()
    assert not df.empty, "Phase 5 grid should not be empty"
    assert len(df) == 765, f"Expected 765 cells, got {len(df)}"
    assert "susceptibility" in df.columns
    assert "rf_tree_uncertainty" in df.columns
    assert "model_disagreement" in df.columns
    assert df["susceptibility"].between(0.0, 1.0).all(), "Susceptibility scores must be within [0, 1]"


def test_load_or_train_models():
    """Verifies model loading and inference capability."""
    models = load_or_train_models()
    assert "reduced" in models, "Reduced model set should be available"
    assert "full" in models, "Full model set should be available"
    assert "rf" in models["reduced"]
    assert "xgb" in models["reduced"]

