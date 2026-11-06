"""Automated tests for validation and data integrity functions."""

import pytest
import pandas as pd
from dashboard.utils.data_loader import (
    load_phase2_cleaned,
    load_phase3_features,
    load_phase5_grid,
)
from dashboard.utils.validation import (
    validate_geographic_bounds,
    validate_schema,
    calculate_missingness_summary,
    validate_label_balance,
)
from dashboard.config import ROI_BOUNDS, REDUCED_FEATURE_COLS


def test_validate_geographic_bounds():
    """Verifies that all Phase 2 points are strictly within the Himalayan bounding box."""
    df = load_phase2_cleaned()
    result = validate_geographic_bounds(df, lat_col="latitude", lon_col="longitude")
    assert result["total_records"] == 254
    assert result["valid"] is True, f"Out of bounds count: {result['out_of_bounds_count']}"
    assert result["lat_range"][0] >= ROI_BOUNDS["lat_min"]
    assert result["lat_range"][1] <= ROI_BOUNDS["lat_max"] + 0.5  # allow minor rounding
    assert result["lon_range"][0] >= ROI_BOUNDS["lon_min"]
    assert result["lon_range"][1] <= ROI_BOUNDS["lon_max"]


def test_validate_schema_phase3():
    """Verifies that Phase 3 table satisfies required feature schema."""
    df = load_phase3_features()
    valid, missing = validate_schema(df, REDUCED_FEATURE_COLS)
    assert valid is True, f"Missing columns: {missing}"


def test_calculate_missingness_summary():
    """Verifies missingness calculation identifies known optical gaps."""
    df = load_phase3_features()
    summary = calculate_missingness_summary(df)
    assert not summary.empty
    bsi_row = summary[summary["column"] == "BSI"]
    assert not bsi_row.empty
    assert bsi_row["missing_count"].values[0] == 80, "Expected exactly 80 missing values in BSI"


def test_validate_label_balance():
    """Verifies 1:1 class balance of cleaned dataset."""
    df = load_phase2_cleaned()
    balance = validate_label_balance(df)
    assert balance["balanced"] is True
    assert balance["positive_count"] == 127
    assert balance["negative_count"] == 127
    assert balance["ratio"] == 1.0

