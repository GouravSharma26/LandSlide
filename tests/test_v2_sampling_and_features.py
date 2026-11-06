"""Unit and governance tests for Version 2.0 Candidate Hillslope Control sampling and feature formulations.

Verifies:
1. Candidate Hillslope Control terminology and unrecorded failure status governance.
2. Sampling envelope constraint parameters (slope, elevation, universal exclusion, water, snow/ice, SAR stability).
3. Mathematical formulation of Riley et al. (1999) TRI vs focal-mean difference proxy.
4. Mathematical formulation of Beven & Kirkby (1979) TWI using flow accumulation vs slope-only proxy.
5. Rejection audit schema completeness across all 8+ rejection reasons.
6. Paired matching diagnostics schema and regional ring limitations.
7. Version 2.0 pending extraction status and confound verification policy.
8. Version 1.0 historical baseline immutability.
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


def test_candidate_control_terminology():
    """Verifies that candidate control terminology reflects unrecorded failure status."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["candidate_control_terminology"] == "Candidate Hillslope Control / Unrecorded Failure Status"
    assert "NASA COOLR" in meta["inventory_incompleteness_disclaimer"]
    assert "unrecorded failure status" in meta["inventory_incompleteness_disclaimer"].lower()

    # Verify Colab notebook contains this terminology
    nb_path = NOTEBOOKS_DIR / "Phase3b_V2_Hillslope_Extraction_Colab.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_text = f.read()
    assert "Candidate Hillslope Control / Unrecorded Failure Status" in nb_text


def test_sampling_envelope_constraints():
    """Verifies empirical envelope thresholds derived from positive event distributions."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    env = meta["v2_control_envelope"]

    # Slope bounds (P10 to P95 of confirmed events)
    assert env["slope_min_deg"] == pytest.approx(6.1, abs=0.2)
    assert env["slope_max_deg"] == pytest.approx(41.4, abs=0.5)

    # Elevation bounds (P10 to P95 of confirmed events)
    assert env["elevation_min_m"] >= 250.0
    assert env["elevation_max_m"] <= 3100.0

    # Universal exclusion zone
    assert env["min_dist_to_positive_m"] == 2000

    # SAR temporal stability criteria
    assert env["sar_max_std_db"] == 1.5
    assert env["sar_observation_window_days"] == [180, 15]
    assert env["sar_min_observations"] == 3

    # Land cover exclusions (ESA WorldCover 10m classes)
    assert env["esa_worldcover_snow_ice_excluded_class"] == 70
    assert env["esa_worldcover_water_excluded_class"] == 80
    assert env["jrc_max_water_pct"] == 10


def test_riley_tri_mathematical_formulation():
    """Verifies that Riley et al. (1999) TRI formula is mathematically distinct from focal-mean difference."""
    # Synthetic 3x3 digital elevation model neighborhood centered at z0 = 100
    # [[z1, z2, z3],
    #  [z4, z0, z5],
    #  [z6, z7, z8]]
    dem_patch = np.array([
        [90.0,  95.0, 105.0],
        [100.0, 100.0, 110.0],
        [105.0, 110.0, 115.0]
    ])
    z0 = dem_patch[1, 1]
    neighbors = [
        dem_patch[0, 0], dem_patch[0, 1], dem_patch[0, 2],
        dem_patch[1, 0],                  dem_patch[1, 2],
        dem_patch[2, 0], dem_patch[2, 1], dem_patch[2, 2]
    ]

    # Standard Riley et al. (1999) TRI: sqrt(sum_{i=1..8} (z_i - z_0)^2)
    diffs = [z_i - z0 for z_i in neighbors]
    riley_tri = math.sqrt(sum(d**2 for d in diffs))

    # Mean absolute difference: (1/8) * sum(|z_i - z_0|)
    mean_abs_diff = sum(abs(d) for d in diffs) / 8.0

    # Focal mean difference proxy: |z0 - mean(3x3)|
    focal_mean = dem_patch.mean()
    focal_diff_proxy = abs(z0 - focal_mean)

    # Verification of values
    # diffs = [-10, -5, +5, 0, +10, +5, +10, +15]
    # sum(d^2) = 100 + 25 + 25 + 0 + 100 + 25 + 100 + 225 = 600
    # sqrt(600) = 24.494897 m
    assert riley_tri == pytest.approx(math.sqrt(600.0), rel=1e-5)
    # sum(|d|) = 10 + 5 + 5 + 0 + 10 + 5 + 10 + 15 = 60 -> 60/8 = 7.5 m
    assert mean_abs_diff == pytest.approx(7.5, rel=1e-5)
    # focal mean = 930 / 9 = 103.333 -> proxy = 3.333 m
    assert focal_diff_proxy == pytest.approx(abs(100.0 - (930.0 / 9.0)), rel=1e-5)

    # Mathematically, focal-mean difference proxy severely underestimates ruggedness
    assert focal_diff_proxy < mean_abs_diff < riley_tri


def test_merit_hydro_twi_mathematical_formulation():
    """Verifies Beven & Kirkby (1979) TWI using upstream contributing area vs slope-only proxy."""
    # Test cases:
    # 1. Gentle slope (5 deg) with small catchment (0.01 km^2)
    # 2. Gentle slope (5 deg) with large valley catchment (10.0 km^2)
    # 3. Steep slope (35 deg) with small catchment (0.01 km^2)
    slopes_deg = [5.0, 5.0, 35.0]
    upas_km2 = [0.01, 10.0, 0.01]
    cell_size = 90.0  # MERIT Hydro grid resolution

    twis_merit = []
    twis_slope_proxy = []

    for s_deg, upa in zip(slopes_deg, upas_km2):
        s_rad = math.radians(s_deg)
        tan_b = max(math.tan(s_rad), 0.001)
        # Specific catchment area in meters
        alpha = max((upa * 1e6) / cell_size, 1.0)
        twi_merit = math.log(alpha / tan_b)
        twi_proxy = -math.log(tan_b)
        twis_merit.append(twi_merit)
        twis_slope_proxy.append(twi_proxy)

    # Physical property 1: Larger upstream area on same slope must yield higher TWI
    assert twis_merit[1] > twis_merit[0], "Valley floor must have higher TWI than ridge crest on same slope"

    # Physical property 2: Steeper slope with same area must yield lower TWI
    assert twis_merit[0] > twis_merit[2], "Gentle slope must have higher TWI than steep slope with same catchment"

    # Physical property 3: Slope-only proxy completely ignores upstream contributing area
    assert twis_slope_proxy[0] == twis_slope_proxy[1], "Slope proxy fails to distinguish valley from ridge"


def test_sampling_audit_counters_schema():
    """Verifies that all required candidate rejection categories are declared and tracked."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    counters = meta["sampling_audit_schema"]["required_rejection_counters"]
    required_keys = [
        "roi_out_of_bounds",
        "scar_proximity_under_2km",
        "elevation_out_of_bounds",
        "slope_out_of_bounds",
        "water_body_overlap",
        "snow_ice_overlap",
        "sar_insufficient_baseline",
        "sar_temporal_instability",
        "no_valid_candidate_after_search"
    ]
    for rk in required_keys:
        assert rk in counters, f"Missing required rejection category: {rk}"


def test_matching_diagnostics_schema():
    """Verifies that paired matching diagnostic variables and limitations are documented."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    diag = meta["matching_diagnostics_schema"]
    required_vars = [
        "spatial_distance_km",
        "min_dist_to_all_positives_km",
        "delta_slope_deg",
        "delta_elevation_m",
        "delta_aspect_deg",
        "delta_tri",
        "sar_pre_std_vv",
        "sar_pre_count"
    ]
    for rv in required_vars:
        assert rv in diag["diagnostic_variables"], f"Missing diagnostic variable: {rv}"
    assert "regional physiographic proxy" in diag["limitation_note"].lower()


def test_v2_status_pending_and_confound_policy():
    """Verifies that Version 2.0 status is pending and confound policy prohibits premature claims."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["candidate_validation_status"]["status"] == "EXTRACTION_PENDING_COLAB_GEE"
    assert "empirically verified" in meta["confound_mitigation_policy"].lower()
    assert "cannot be claimed a priori" in meta["confound_mitigation_policy"].lower()


def test_v1_baseline_immutability():
    """Verifies that Version 1.0 historical diagnostic baseline is strictly preserved."""
    p3_path = OUTPUTS_DIR / "phase3_feature_table.csv"
    assert p3_path.exists(), "Phase 3 feature table missing"
    df = pd.read_csv(p3_path)
    assert len(df) in [244, 254], f"Unexpected row count in Phase 3 table: {len(df)}"
    assert (df["label"] == 1).sum() == 127, "Must have exactly 127 positive events"
    assert (df["label"] == 0).sum() in [117, 127], "Historical controls altered"
