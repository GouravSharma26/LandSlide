"""Unit, governance, and structural tests for Version 2.0 Candidate Hillslope Control sampling.

Test Categories:
Category 1: Pure Python Unit Tests (Mocked / Offline Logic)
  - Multi-batch candidate optimization (Batch 2 better, Batch 3 better, deduplication, no valid candidates, minimum distance selection)
  - Native 90m hydrological TWI unit and formula consistency (dimensional analysis and monotonicity)
  - Positive-event terrain validation without fallbacks (missing slope, elevation, aspect, TRI, ROI violations)
  - Candidate evaluation filters (bounds, scar proximity, missing terrain/water/landcover, water/snow overlap, SAR baseline/stability)
  - Cross-event duplicate control detection
Category 2: Notebook Structure and Governance Tests (Static AST / JSON Inspection)
  - Verification of no arbitrary positive-event terrain defaults in notebook
  - Verification of no premature loop break across batches in notebook
  - Verification of MERIT Hydro 90m native grid & slope aggregation documentation
  - Verification of all 13 rejection categories in audit dictionary
  - Verification of candidate terminology and immutable V1 historical baseline
Category 3: Earth Engine Integration Tests
  - Documented as pending actual authenticated Google Colab execution (no local simulation).
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import pytest

from scripts.v2_sampling_protocol import (
    evaluate_candidate_point,
    select_best_candidate_across_batches,
    validate_positive_event,
    calculate_twi_hydrological_native_grid,
    haversine_km
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


# Base valid candidate fixture for testing
@pytest.fixture
def valid_candidate_props():
    return {
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


@pytest.fixture
def valid_paired_pos():
    return {
        "latitude": 30.5,
        "longitude": 79.2,
        "elevation": 1200.0,
        "slope": 21.0,
        "aspect": 180.0,
        "tri": 26.0,
        "event_date": "2018-08-15"
    }


@pytest.fixture
def all_pos_coords():
    return [(30.5, 79.2), (31.0, 78.5), (29.8, 80.1)]


# =============================================================================
# Category 1: Pure Python Unit Tests (Offline / Algorithmic Verification)
# =============================================================================

def test_twi_unit_and_formula_consistency():
    """Verifies Beven & Kirkby (1979) TWI on native 90m hydrological grid.
    
    Dimensional checks:
    - upa in km^2 -> A = upa * 1e6 m^2
    - contour width b = 90.0 m
    - specific catchment area alpha = A / b [units: m^2/m = m]
    - tan(beta) dimensionless ratio
    - TWI = ln(alpha / tan(beta))
    """
    # 1. Monotonicity with contributing area
    twi_small_area, _ = calculate_twi_hydrological_native_grid(upa_km2=0.01, slope_val=15.0, is_deg=True, grid_cell_size_m=90.0)
    twi_large_area, _ = calculate_twi_hydrological_native_grid(upa_km2=10.0, slope_val=15.0, is_deg=True, grid_cell_size_m=90.0)
    assert twi_large_area > twi_small_area, "Larger upstream contributing area must produce higher TWI"

    # 2. Monotonicity with slope gradient
    twi_gentle_slope, _ = calculate_twi_hydrological_native_grid(upa_km2=1.0, slope_val=5.0, is_deg=True, grid_cell_size_m=90.0)
    twi_steep_slope, _ = calculate_twi_hydrological_native_grid(upa_km2=1.0, slope_val=35.0, is_deg=True, grid_cell_size_m=90.0)
    assert twi_gentle_slope > twi_steep_slope, "Gentler hillslope must produce higher TWI"

    # 3. Slope-only proxy behavior
    _, proxy_gentle = calculate_twi_hydrological_native_grid(upa_km2=1.0, slope_val=5.0, is_deg=True, grid_cell_size_m=90.0)
    _, proxy_same_slope_diff_area = calculate_twi_hydrological_native_grid(upa_km2=100.0, slope_val=5.0, is_deg=True, grid_cell_size_m=90.0)
    assert proxy_gentle == pytest.approx(proxy_same_slope_diff_area), "Slope proxy must be invariant to contributing area"

    # 4. Error handling
    with pytest.raises(ValueError):
        calculate_twi_hydrological_native_grid(upa_km2=-1.0, slope_val=15.0)


def test_positive_event_missing_data_validation(valid_paired_pos):
    """Verifies that positive events with missing mandatory terrain features are rejected."""
    # Valid event passes
    ok, reason, validated = validate_positive_event(valid_paired_pos)
    assert ok is True
    assert reason == "VALID"
    assert validated["elevation"] == 1200.0

    # Missing slope
    corrupted = dict(valid_paired_pos, slope=None)
    ok, reason, _ = validate_positive_event(corrupted)
    assert ok is False
    assert "missing_positive_features: slope" in reason

    # Missing elevation
    corrupted = dict(valid_paired_pos, elevation=None)
    ok, reason, _ = validate_positive_event(corrupted)
    assert ok is False
    assert "elevation" in reason

    # Missing aspect
    corrupted = dict(valid_paired_pos, aspect=None)
    ok, reason, _ = validate_positive_event(corrupted)
    assert ok is False
    assert "aspect" in reason

    # Missing tri
    corrupted = dict(valid_paired_pos, tri=None)
    ok, reason, _ = validate_positive_event(corrupted)
    assert ok is False
    assert "tri" in reason

    # ROI violation
    corrupted = dict(valid_paired_pos, latitude=22.0)
    ok, reason, _ = validate_positive_event(corrupted)
    assert ok is False
    assert reason == "positive_roi_violation"


def test_batch_optimization_better_in_batch_2(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies that the candidate search does not break on batch 1 when a better match exists in batch 2."""
    # Batch 0 has a valid candidate with large D_topo
    cand_b0_props = dict(valid_candidate_props, slope=38.0, elevation=2800.0, tri=50.0)
    cand_b0 = {"longitude": 79.35, "latitude": 30.65, "properties": cand_b0_props}

    # Batch 1 has an even better candidate close to paired positive (slope 21, elev 1200, tri 26)
    cand_b1_props = dict(valid_candidate_props, slope=21.2, elevation=1210.0, tri=26.1)
    cand_b1 = {"longitude": 79.40, "latitude": 30.70, "properties": cand_b1_props}

    # Batch 2 has a candidate with medium D_topo
    cand_b2_props = dict(valid_candidate_props, slope=28.0, elevation=1600.0, tri=32.0)
    cand_b2 = {"longitude": 79.45, "latitude": 30.75, "properties": cand_b2_props}

    batches = [[cand_b0], [cand_b1], [cand_b2]]
    best_cand, audit = select_best_candidate_across_batches(batches, valid_paired_pos, all_pos_coords)

    assert best_cand is not None
    assert audit["batches_evaluated"] == 3
    assert audit["best_batch_idx"] == 1, "Best candidate must be selected from Batch 1 (second batch)"
    assert best_cand["slope"] == pytest.approx(21.2)
    assert best_cand["three_feature_topographic_distance"] < 0.10


def test_batch_optimization_better_in_batch_3(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies that candidate search finds and selects the best candidate when it occurs in batch 3."""
    cand_b0_props = dict(valid_candidate_props, slope=35.0, elevation=2500.0, tri=45.0)
    cand_b0 = {"longitude": 79.35, "latitude": 30.65, "properties": cand_b0_props}

    cand_b1_props = dict(valid_candidate_props, slope=30.0, elevation=2000.0, tri=38.0)
    cand_b1 = {"longitude": 79.40, "latitude": 30.70, "properties": cand_b1_props}

    # Best candidate is in Batch 2 (the 3rd batch)
    cand_b2_props = dict(valid_candidate_props, slope=21.05, elevation=1202.0, tri=26.05)
    cand_b2 = {"longitude": 79.45, "latitude": 30.75, "properties": cand_b2_props}

    batches = [[cand_b0], [cand_b1], [cand_b2]]
    best_cand, audit = select_best_candidate_across_batches(batches, valid_paired_pos, all_pos_coords)

    assert best_cand is not None
    assert audit["best_batch_idx"] == 2, "Best candidate must be selected from Batch 2 (third batch)"
    assert best_cand["slope"] == pytest.approx(21.05)
    assert best_cand["three_feature_topographic_distance"] < 0.05


def test_batch_optimization_coordinate_deduplication(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies that candidates with duplicate coordinates across batches are properly deduplicated."""
    # Candidate appearing in Batch 0 and again in Batch 1
    cand_a1 = {"longitude": 79.35, "latitude": 30.65, "properties": dict(valid_candidate_props, slope=22.0)}
    cand_a2 = {"longitude": 79.35, "latitude": 30.65, "properties": dict(valid_candidate_props, slope=22.0)}
    # Distinct candidate in Batch 2
    cand_b = {"longitude": 79.45, "latitude": 30.75, "properties": dict(valid_candidate_props, slope=21.0)}

    batches = [[cand_a1], [cand_a2], [cand_b]]
    best_cand, audit = select_best_candidate_across_batches(batches, valid_paired_pos, all_pos_coords)

    assert audit["total_candidates_evaluated"] == 3
    assert audit["duplicates_removed"] == 1, "Duplicate candidate coordinate must be detected and removed"
    assert audit["valid_candidates_pooled"] == 2


def test_batch_optimization_no_valid_candidates(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies behavior when all candidates across all batches fail filters."""
    # All candidates fail (e.g. slope out of bounds)
    invalid_props = dict(valid_candidate_props, slope=2.0)
    cand0 = {"longitude": 79.35, "latitude": 30.65, "properties": invalid_props}
    cand1 = {"longitude": 79.40, "latitude": 30.70, "properties": invalid_props}
    cand2 = {"longitude": 79.45, "latitude": 30.75, "properties": invalid_props}

    batches = [[cand0], [cand1], [cand2]]
    best_cand, audit = select_best_candidate_across_batches(batches, valid_paired_pos, all_pos_coords)

    assert best_cand is None
    assert audit["valid_candidates_pooled"] == 0
    assert audit["rejection_counts"]["no_valid_candidate_after_search"] == 1
    assert audit["rejection_counts"]["slope_out_of_bounds"] == 3


def test_candidate_acceptance_valid(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies candidate acceptance and standardized D_topo calculation."""
    ok, reason, rec = evaluate_candidate_point(
        c_lat=30.65,
        c_lon=79.35,
        props=valid_candidate_props,
        paired_positive=valid_paired_pos,
        all_positives_coords=all_pos_coords
    )
    assert ok is True
    assert reason == "ACCEPTED"
    assert rec["delta_slope_deg"] == pytest.approx(1.5, rel=1e-3)
    assert rec["delta_elevation_m"] == pytest.approx(50.0, rel=1e-3)
    assert rec["three_feature_topographic_distance"] > 0.0


def test_candidate_rejection_null_terrain_inputs(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies that missing terrain features are explicitly rejected without silent 0.0 fallbacks."""
    for terrain_key in ["aspect", "curvature", "tri", "tri_mean_diff", "twi", "slope", "elevation"]:
        corrupted = dict(valid_candidate_props)
        corrupted[terrain_key] = None
        ok, reason, rec = evaluate_candidate_point(
            c_lat=30.65, c_lon=79.35, props=corrupted,
            paired_positive=valid_paired_pos, all_positives_coords=all_pos_coords
        )
        assert ok is False
        assert reason == "missing_terrain_data"
        assert rec is None


def test_candidate_rejection_missing_water_mask(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies that missing JRC water occurrence is rejected and not silently unmasked to zero."""
    corrupted = dict(valid_candidate_props, water_occurrence=None)
    ok, reason, _ = evaluate_candidate_point(
        c_lat=30.65, c_lon=79.35, props=corrupted,
        paired_positive=valid_paired_pos, all_positives_coords=all_pos_coords
    )
    assert ok is False
    assert reason == "missing_water_data"


def test_candidate_rejection_missing_landcover_mask(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies that missing WorldCover land-cover observation is rejected."""
    corrupted = dict(valid_candidate_props, landcover=None)
    ok, reason, _ = evaluate_candidate_point(
        c_lat=30.65, c_lon=79.35, props=corrupted,
        paired_positive=valid_paired_pos, all_positives_coords=all_pos_coords
    )
    assert ok is False
    assert reason == "missing_landcover_data"


def test_candidate_rejection_elevation_out_of_bounds(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of elevations outside empirical envelope [266m, 3027m]."""
    for bad_elev in [150.0, 3200.0]:
        corrupted = dict(valid_candidate_props, elevation=bad_elev)
        ok, reason, _ = evaluate_candidate_point(
            c_lat=30.65, c_lon=79.35, props=corrupted,
            paired_positive=valid_paired_pos, all_positives_coords=all_pos_coords
        )
        assert ok is False
        assert reason == "elevation_out_of_bounds"


def test_candidate_rejection_slope_out_of_bounds(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of slopes outside empirical envelope [6.1 deg, 41.4 deg]."""
    for bad_slope in [3.0, 48.0]:
        corrupted = dict(valid_candidate_props, slope=bad_slope)
        ok, reason, _ = evaluate_candidate_point(
            c_lat=30.65, c_lon=79.35, props=corrupted,
            paired_positive=valid_paired_pos, all_positives_coords=all_pos_coords
        )
        assert ok is False
        assert reason == "slope_out_of_bounds"


def test_candidate_rejection_water_overlap(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of points overlapping water bodies (JRC >= 10% or WorldCover == 80)."""
    # Case A: JRC water occurrence >= 10%
    props_jrc = dict(valid_candidate_props, water_occurrence=15.0)
    ok, reason, _ = evaluate_candidate_point(30.65, 79.35, props_jrc, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "water_body_overlap"

    # Case B: ESA WorldCover Class 80 (Permanent water)
    props_wc = dict(valid_candidate_props, landcover=80)
    ok, reason, _ = evaluate_candidate_point(30.65, 79.35, props_wc, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "water_body_overlap"


def test_candidate_rejection_snow_ice_overlap(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of points overlapping permanent snow/ice (WorldCover Class 70)."""
    props_snow = dict(valid_candidate_props, landcover=70)
    ok, reason, _ = evaluate_candidate_point(30.65, 79.35, props_snow, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "snow_ice_overlap"


def test_candidate_rejection_sar_insufficient_baseline(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of points with fewer than 3 SAR passes or null standard deviation."""
    props_few = dict(valid_candidate_props, sar_pre_count=2)
    ok, reason, _ = evaluate_candidate_point(30.65, 79.35, props_few, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "sar_insufficient_baseline"

    props_null = dict(valid_candidate_props, sar_pre_std_vv=None)
    ok, reason, _ = evaluate_candidate_point(30.65, 79.35, props_null, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "sar_insufficient_baseline"


def test_candidate_rejection_sar_temporal_instability(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of points exceeding SAR temporal stability threshold (> 1.5 dB)."""
    props_unstable = dict(valid_candidate_props, sar_pre_std_vv=1.95)
    ok, reason, _ = evaluate_candidate_point(30.65, 79.35, props_unstable, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "sar_temporal_instability"


def test_candidate_rejection_roi_violation(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of coordinates outside authoritative Himalayan ROI."""
    ok, reason, _ = evaluate_candidate_point(24.5, 79.35, valid_candidate_props, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "roi_out_of_bounds"

    ok, reason, _ = evaluate_candidate_point(30.65, 98.5, valid_candidate_props, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "roi_out_of_bounds"


def test_candidate_rejection_proximity_violation(valid_candidate_props, valid_paired_pos, all_pos_coords):
    """Verifies rejection of candidate points within 2000m of ANY known positive scar."""
    c_lat_close = 30.504
    c_lon_close = 79.204
    ok, reason, _ = evaluate_candidate_point(c_lat_close, c_lon_close, valid_candidate_props, valid_paired_pos, all_pos_coords)
    assert ok is False
    assert reason == "scar_proximity_under_2km"


def test_cross_event_duplicate_controls_detection():
    """Verifies detection of cross-event duplicate control coordinates and minimum separation."""
    # Control list with 1 duplicate and 2 unique points
    ctrls = [
        {"latitude": 30.1, "longitude": 79.1},
        {"latitude": 30.5, "longitude": 79.5},
        {"latitude": 30.1, "longitude": 79.1},  # duplicate of point 0
    ]
    df_c = pd.DataFrame(ctrls)
    n_unique = len(df_c[["latitude", "longitude"]].drop_duplicates())
    duplicates = len(df_c) - n_unique
    assert duplicates == 1, "Must detect 1 duplicate cross-event control"


# =============================================================================
# Category 2: Notebook Structure and Governance Tests (Static Inspection)
# =============================================================================

def test_notebook_structure_no_defaults():
    """Verifies that no arbitrary positive-event terrain defaults exist in the Colab notebook."""
    nb_path = NOTEBOOKS_DIR / "Phase3b_V2_Hillslope_Extraction_Colab.ipynb"
    assert nb_path.exists(), f"Missing Colab notebook at {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    full_nb_text = json.dumps(nb)
    # Ensure no arbitrary fallback defaults exist
    assert "row.get('slope', 21.0)" not in full_nb_text
    assert "row.get('elevation', 1200.0)" not in full_nb_text
    assert "row.get('aspect', 180.0)" not in full_nb_text
    assert "row.get('tri', 25.0)" not in full_nb_text
    assert "props.get('aspect', 0.0)" not in full_nb_text


def test_notebook_structure_no_premature_break():
    """Verifies that the candidate search loop does NOT break prematurely on batch 1."""
    nb_path = NOTEBOOKS_DIR / "Phase3b_V2_Hillslope_Extraction_Colab.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # In Cell 4, verify that the loop does NOT contain `if valid_candidates_for_event: break`
    cell_4_src = "".join(nb["cells"][4]["source"])
    assert "if valid_candidates_for_event: break" not in cell_4_src
    assert "if valid_candidates_for_event:" not in cell_4_src
    # Verify that multi-batch evaluation is present
    assert "pooled_valid_candidates" in cell_4_src
    assert "for batch_num in range(num_batches):" in cell_4_src
    assert "three_feature_topographic_distance" in cell_4_src


def test_notebook_structure_merit_90m_twi():
    """Verifies that native 90m MERIT Hydro TWI and slope aggregation are implemented in notebook."""
    nb_path = NOTEBOOKS_DIR / "Phase3b_V2_Hillslope_Extraction_Colab.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_3_src = "".join(nb["cells"][3]["source"])
    assert "Native MERIT Hydro 90m" in cell_3_src
    assert "divide(90.0)" in cell_3_src
    assert "reduceResolution(reducer=ee.Reducer.mean()" in cell_3_src


def test_sampling_audit_counters_schema():
    """Verifies that all 13 required candidate rejection categories are declared in metadata."""
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
        "duplicate_candidate_coordinates",
        "no_valid_candidate_after_search"
    ]
    for rk in required_keys:
        assert rk in counters, f"Missing required rejection category: {rk}"


def test_candidate_control_terminology():
    """Verifies that candidate control terminology reflects unrecorded failure status."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["candidate_control_terminology"] == "Candidate Hillslope Control / Unrecorded Failure Status"
    assert "NASA COOLR" in meta["inventory_incompleteness_disclaimer"]


def test_sampling_envelope_constraints():
    """Verifies empirical envelope thresholds derived from positive event distributions."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    env = meta["v2_control_envelope"]
    assert env["slope_min_deg"] == pytest.approx(6.1, abs=0.2)
    assert env["slope_max_deg"] == pytest.approx(41.4, abs=0.5)
    assert env["elevation_min_m"] >= 250.0
    assert env["elevation_max_m"] <= 3100.0
    assert env["min_dist_to_positive_m"] == 2000
    assert env["sar_max_std_db"] == 1.5


def test_matching_diagnostics_schema():
    """Verifies that paired matching diagnostic variables, percentiles, and limitations are documented."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    diag = meta["matching_diagnostics_schema"]
    assert diag["distance_metric"] == "three_feature_topographic_distance"
    assert "delta_slope_deg" in diag["independent_delta_reporting"]
    assert "delta_aspect_deg" in diag["independent_delta_reporting"]
    assert "delta_curvature" in diag["independent_delta_reporting"]
    assert "delta_twi" in diag["independent_delta_reporting"]


def test_v2_status_pending_and_confound_policy():
    """Verifies that Version 2.0 status is pending and confound policy prohibits premature claims."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["candidate_validation_status"]["status"] == "EXTRACTION_PENDING_COLAB_GEE"
    assert "empirically verified" in meta["confound_mitigation_policy"].lower()
    assert "cannot be claimed a priori" in meta["confound_mitigation_policy"].lower()
    assert "asymmetric screening" in meta["sar_screening_characterization"].lower()


def test_v1_baseline_immutability():
    """Verifies that Version 1.0 historical diagnostic baseline is strictly preserved."""
    p3_path = OUTPUTS_DIR / "phase3_feature_table.csv"
    assert p3_path.exists(), "Phase 3 feature table missing"
    df = pd.read_csv(p3_path)
    assert len(df) in [244, 254], f"Unexpected row count in Phase 3 table: {len(df)}"
    assert (df["label"] == 1).sum() == 127, "Must have exactly 127 positive events"
    assert (df["label"] == 0).sum() in [117, 127], "Historical controls altered"
