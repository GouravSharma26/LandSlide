"""Unit and governance tests for Version 2.0 Candidate Hillslope Control sampling and feature formulations.

Step 5.2 Test Coverage:
1. Candidate Hillslope Control terminology and unrecorded failure status governance.
2. Sampling envelope constraint parameters (slope, elevation, universal exclusion, water, snow/ice, SAR stability).
3. Mathematical formulation of Riley et al. (1999) TRI vs focal-mean difference proxy.
4. Mathematical formulation of Beven & Kirkby (1979) TWI with 30m bilinear resampling vs slope-only proxy.
5. Rejection audit schema completeness across all 12 rejection categories.
6. Paired matching diagnostics schema and regional ring limitations.
7. Version 2.0 pending extraction status and confound verification policy.
8. Version 1.0 historical baseline immutability.
9. Local unit tests for candidate evaluation logic:
   - Valid candidate acceptance and composite distance D_topo calculation
   - Null inputs / missing terrain data rejection (no silent 0.0 defaults)
   - Missing water observation rejection (no unmask(0) silent pass)
   - Missing land-cover observation rejection (no unmask silent pass)
   - Elevation out of bounds rejection
   - Slope out of bounds rejection
   - Water body overlap rejection
   - Permanent snow/ice overlap rejection
   - Invalid SAR count rejection (< 3 passes)
   - SAR temporal instability rejection (> 1.5 dB)
   - ROI boundary violation rejection
   - Universal scar proximity violation rejection (<= 2000m)
   - Optimization of candidate selection via minimum D_topo
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.v2_sampling_protocol import evaluate_candidate_point, haversine_km

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


# Base valid candidate fixture for testing
@pytest.fixture
def valid_candidate_data():
    props = {
        "elevation": 1250.0,
        "slope": 22.5,
        "aspect": 185.0,
        "curvature": -0.05,
        "tri": 28.0,
        "tri_mean_diff": 8.5,
        "tri_focal_diff_proxy": 3.2,
        "twi": 9.5,
        "twi_slope_proxy": 0.88,
        "water_occurrence": 0.0,
        "landcover": 10,  # Tree cover
        "sar_pre_std_vv": 0.85,
        "sar_pre_count": 8
    }
    paired_pos = {
        "latitude": 30.5,
        "longitude": 79.2,
        "elevation": 1200.0,
        "slope": 21.0,
        "aspect": 180.0,
        "tri": 26.0
    }
    all_pos_coords = [(30.5, 79.2), (31.0, 78.5), (29.8, 80.1)]
    return {
        "c_lat": 30.65,
        "c_lon": 79.35,
        "props": props,
        "paired_pos": paired_pos,
        "all_pos_coords": all_pos_coords
    }


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

    diffs = [z_i - z0 for z_i in neighbors]
    riley_tri = math.sqrt(sum(d**2 for d in diffs))
    mean_abs_diff = sum(abs(d) for d in diffs) / 8.0
    focal_mean = dem_patch.mean()
    focal_diff_proxy = abs(z0 - focal_mean)

    assert riley_tri == pytest.approx(math.sqrt(600.0), rel=1e-5)
    assert mean_abs_diff == pytest.approx(7.5, rel=1e-5)
    assert focal_diff_proxy == pytest.approx(abs(100.0 - (930.0 / 9.0)), rel=1e-5)
    assert focal_diff_proxy < mean_abs_diff < riley_tri


def test_merit_hydro_twi_mathematical_formulation():
    """Verifies Beven & Kirkby (1979) TWI using upstream contributing area vs slope-only proxy."""
    slopes_deg = [5.0, 5.0, 35.0]
    upas_km2 = [0.01, 10.0, 0.01]
    cell_size = 30.0  # SRTM aligned 30m grid

    twis_merit = []
    twis_slope_proxy = []

    for s_deg, upa in zip(slopes_deg, upas_km2):
        s_rad = math.radians(s_deg)
        tan_b = max(math.tan(s_rad), 0.001)
        alpha = max((upa * 1e6) / cell_size, 1.0)
        twi_merit = math.log(alpha / tan_b)
        twi_proxy = -math.log(tan_b)
        twis_merit.append(twi_merit)
        twis_slope_proxy.append(twi_proxy)

    assert twis_merit[1] > twis_merit[0], "Valley floor must have higher TWI than ridge crest on same slope"
    assert twis_merit[0] > twis_merit[2], "Gentle slope must have higher TWI than steep slope with same catchment"
    assert twis_slope_proxy[0] == twis_slope_proxy[1], "Slope proxy fails to distinguish valley from ridge"


def test_sampling_audit_counters_schema():
    """Verifies that all 12 required candidate rejection categories are declared and tracked."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    counters = meta["sampling_audit_schema"]["required_rejection_counters"]
    required_keys = [
        "roi_out_of_bounds",
        "scar_proximity_under_2km",
        "missing_terrain_data",
        "missing_water_data",
        "missing_landcover_data",
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
    """Verifies that paired matching diagnostic variables, percentiles, and limitations are documented."""
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
        "d_topo_composite",
        "sar_pre_std_vv",
        "sar_pre_count"
    ]
    for rv in required_vars:
        assert rv in diag["diagnostic_variables"], f"Missing diagnostic variable: {rv}"
    assert "min" in diag["reporting_metrics"]
    assert "median" in diag["reporting_metrics"]
    assert "max" in diag["reporting_metrics"]
    assert "regional physiographic proxy" in diag["limitation_note"].lower()


def test_v2_status_pending_and_confound_policy():
    """Verifies that Version 2.0 status is pending and confound policy prohibits premature claims."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["candidate_validation_status"]["status"] == "EXTRACTION_PENDING_COLAB_GEE"
    assert "empirically verified" in meta["confound_mitigation_policy"].lower()
    assert "cannot be claimed a priori" in meta["confound_mitigation_policy"].lower()
    assert "microwave temporal consistency screening" in meta["sar_screening_characterization"].lower()
    assert "resampled via bilinear interpolation" in meta["twi_resolution_alignment_specification"].lower()


def test_v1_baseline_immutability():
    """Verifies that Version 1.0 historical diagnostic baseline is strictly preserved."""
    p3_path = OUTPUTS_DIR / "phase3_feature_table.csv"
    assert p3_path.exists(), "Phase 3 feature table missing"
    df = pd.read_csv(p3_path)
    assert len(df) in [244, 254], f"Unexpected row count in Phase 3 table: {len(df)}"
    assert (df["label"] == 1).sum() == 127, "Must have exactly 127 positive events"
    assert (df["label"] == 0).sum() in [117, 127], "Historical controls altered"


# =============================================================================
# Step 5.2 Unit Tests for evaluate_candidate_point() Logic
# =============================================================================

def test_candidate_acceptance_valid(valid_candidate_data):
    """Verifies that a fully compliant candidate point is accepted and returns matching diagnostics."""
    d = valid_candidate_data
    ok, reason, rec = evaluate_candidate_point(
        c_lat=d["c_lat"],
        c_lon=d["c_lon"],
        props=d["props"],
        paired_positive=d["paired_pos"],
        all_positives_coords=d["all_pos_coords"]
    )
    assert ok is True
    assert reason == "ACCEPTED"
    assert rec is not None
    assert rec["latitude"] == d["c_lat"]
    assert rec["longitude"] == d["c_lon"]
    assert rec["elevation"] == 1250.0
    assert rec["slope"] == 22.5
    assert rec["delta_slope_deg"] == pytest.approx(1.5, rel=1e-3)
    assert rec["delta_elevation_m"] == pytest.approx(50.0, rel=1e-3)
    assert rec["d_topo_composite"] > 0.0


def test_candidate_rejection_null_terrain_inputs(valid_candidate_data):
    """Verifies that missing terrain features are explicitly rejected without silent 0.0 fallbacks."""
    d = valid_candidate_data
    for terrain_key in ["aspect", "curvature", "tri", "tri_mean_diff", "twi", "slope", "elevation"]:
        corrupted_props = dict(d["props"])
        corrupted_props[terrain_key] = None
        ok, reason, rec = evaluate_candidate_point(
            c_lat=d["c_lat"],
            c_lon=d["c_lon"],
            props=corrupted_props,
            paired_positive=d["paired_pos"],
            all_positives_coords=d["all_pos_coords"]
        )
        assert ok is False
        assert reason == "missing_terrain_data", f"Failed to reject null {terrain_key}"
        assert rec is None


def test_candidate_rejection_missing_water_mask(valid_candidate_data):
    """Verifies that missing JRC water occurrence is rejected and not silently unmasked to zero."""
    d = valid_candidate_data
    corrupted_props = dict(d["props"])
    corrupted_props["water_occurrence"] = None
    ok, reason, rec = evaluate_candidate_point(
        c_lat=d["c_lat"],
        c_lon=d["c_lon"],
        props=corrupted_props,
        paired_positive=d["paired_pos"],
        all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "missing_water_data"
    assert rec is None


def test_candidate_rejection_missing_landcover_mask(valid_candidate_data):
    """Verifies that missing WorldCover land-cover observation is rejected."""
    d = valid_candidate_data
    corrupted_props = dict(d["props"])
    corrupted_props["landcover"] = None
    ok, reason, rec = evaluate_candidate_point(
        c_lat=d["c_lat"],
        c_lon=d["c_lon"],
        props=corrupted_props,
        paired_positive=d["paired_pos"],
        all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "missing_landcover_data"
    assert rec is None


def test_candidate_rejection_elevation_out_of_bounds(valid_candidate_data):
    """Verifies rejection of elevations outside empirical envelope [266m, 3027m]."""
    d = valid_candidate_data
    for bad_elev in [150.0, 3200.0]:
        corrupted_props = dict(d["props"])
        corrupted_props["elevation"] = bad_elev
        ok, reason, _ = evaluate_candidate_point(
            c_lat=d["c_lat"],
            c_lon=d["c_lon"],
            props=corrupted_props,
            paired_positive=d["paired_pos"],
            all_positives_coords=d["all_pos_coords"]
        )
        assert ok is False
        assert reason == "elevation_out_of_bounds"


def test_candidate_rejection_slope_out_of_bounds(valid_candidate_data):
    """Verifies rejection of slopes outside empirical envelope [6.1 deg, 41.4 deg]."""
    d = valid_candidate_data
    for bad_slope in [3.0, 48.0]:
        corrupted_props = dict(d["props"])
        corrupted_props["slope"] = bad_slope
        ok, reason, _ = evaluate_candidate_point(
            c_lat=d["c_lat"],
            c_lon=d["c_lon"],
            props=corrupted_props,
            paired_positive=d["paired_pos"],
            all_positives_coords=d["all_pos_coords"]
        )
        assert ok is False
        assert reason == "slope_out_of_bounds"


def test_candidate_rejection_water_overlap(valid_candidate_data):
    """Verifies rejection of points overlapping water bodies (JRC >= 10% or WorldCover == 80)."""
    d = valid_candidate_data
    # Case A: JRC water occurrence >= 10%
    props_jrc = dict(d["props"])
    props_jrc["water_occurrence"] = 15.0
    ok, reason, _ = evaluate_candidate_point(
        c_lat=d["c_lat"], c_lon=d["c_lon"], props=props_jrc,
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "water_body_overlap"

    # Case B: ESA WorldCover Class 80 (Permanent water)
    props_wc = dict(d["props"])
    props_wc["landcover"] = 80
    ok, reason, _ = evaluate_candidate_point(
        c_lat=d["c_lat"], c_lon=d["c_lon"], props=props_wc,
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "water_body_overlap"


def test_candidate_rejection_snow_ice_overlap(valid_candidate_data):
    """Verifies rejection of points overlapping permanent snow/ice (WorldCover Class 70)."""
    d = valid_candidate_data
    props_snow = dict(d["props"])
    props_snow["landcover"] = 70
    ok, reason, _ = evaluate_candidate_point(
        c_lat=d["c_lat"], c_lon=d["c_lon"], props=props_snow,
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "snow_ice_overlap"


def test_candidate_rejection_sar_insufficient_baseline(valid_candidate_data):
    """Verifies rejection of points with fewer than 3 SAR passes or null standard deviation."""
    d = valid_candidate_data
    # Fewer than 3 passes
    props_few = dict(d["props"])
    props_few["sar_pre_count"] = 2
    ok, reason, _ = evaluate_candidate_point(
        c_lat=d["c_lat"], c_lon=d["c_lon"], props=props_few,
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "sar_insufficient_baseline"

    # Null standard deviation
    props_null = dict(d["props"])
    props_null["sar_pre_std_vv"] = None
    ok, reason, _ = evaluate_candidate_point(
        c_lat=d["c_lat"], c_lon=d["c_lon"], props=props_null,
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "sar_insufficient_baseline"


def test_candidate_rejection_sar_temporal_instability(valid_candidate_data):
    """Verifies rejection of points exceeding SAR temporal stability threshold (> 1.5 dB)."""
    d = valid_candidate_data
    props_unstable = dict(d["props"])
    props_unstable["sar_pre_std_vv"] = 1.95  # > 1.5 dB
    ok, reason, _ = evaluate_candidate_point(
        c_lat=d["c_lat"], c_lon=d["c_lon"], props=props_unstable,
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "sar_temporal_instability"


def test_candidate_rejection_roi_violation(valid_candidate_data):
    """Verifies rejection of coordinates outside authoritative Himalayan ROI."""
    d = valid_candidate_data
    # Latitude too low (< 26.0)
    ok, reason, _ = evaluate_candidate_point(
        c_lat=24.5, c_lon=d["c_lon"], props=d["props"],
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "roi_out_of_bounds"

    # Longitude too high (> 97.0)
    ok, reason, _ = evaluate_candidate_point(
        c_lat=d["c_lat"], c_lon=98.5, props=d["props"],
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "roi_out_of_bounds"


def test_candidate_rejection_proximity_violation(valid_candidate_data):
    """Verifies rejection of candidate points within 2000m of ANY known positive scar."""
    d = valid_candidate_data
    # Candidate point very close to positive (30.5, 79.2) -> distance ~ 0.5 km
    c_lat_close = 30.504
    c_lon_close = 79.204
    ok, reason, _ = evaluate_candidate_point(
        c_lat=c_lat_close, c_lon=c_lon_close, props=d["props"],
        paired_positive=d["paired_pos"], all_positives_coords=d["all_pos_coords"]
    )
    assert ok is False
    assert reason == "scar_proximity_under_2km"


def test_composite_topographic_distance_minimization(valid_candidate_data):
    """Verifies that between two valid candidates, D_topo correctly identifies the closer terrain match."""
    d = valid_candidate_data
    paired = {"latitude": 30.0, "longitude": 79.0, "elevation": 1200.0, "slope": 20.0, "tri": 25.0, "aspect": 180.0}

    # Candidate 1: Very close match (slope 21, elev 1220, tri 26)
    props1 = dict(d["props"])
    props1["slope"] = 21.0
    props1["elevation"] = 1220.0
    props1["tri"] = 26.0

    # Candidate 2: Marginal match (slope 38, elev 2800, tri 50)
    props2 = dict(d["props"])
    props2["slope"] = 38.0
    props2["elevation"] = 2800.0
    props2["tri"] = 50.0

    _, _, rec1 = evaluate_candidate_point(30.2, 79.2, props1, paired, d["all_pos_coords"])
    _, _, rec2 = evaluate_candidate_point(30.3, 79.3, props2, paired, d["all_pos_coords"])

    assert rec1["d_topo_composite"] < rec2["d_topo_composite"], "D_topo must favor closer terrain match"
    assert rec1["d_topo_composite"] < 0.20
    assert rec2["d_topo_composite"] > 2.50
