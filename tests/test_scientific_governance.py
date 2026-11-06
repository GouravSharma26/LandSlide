"""Comprehensive Scientific Governance and Integrity Test Suite.

Verifies:
1. Prohibited features enforcement (zero banned variables in feature tables).
2. Temporal leakage absence in susceptibility models (strictly pre-event predictors).
3. Actual spatial purge enforcement (min train-test distance >= purge distance).
4. Spatial fold separation and retained training sample counts.
5. Artifact consistency and metadata completeness.
6. Model serialization, deserialization, and inference sanity.
7. Version 2.0 status and distribution divergence confirmation.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
COMMON_DIR = ARTIFACTS_DIR / "common"
DETECTION_DIR = ARTIFACTS_DIR / "detection"
SUSCEPTIBILITY_DIR = ARTIFACTS_DIR / "susceptibility"
METADATA_DIR = ARTIFACTS_DIR / "metadata"


def test_prohibited_features_enforcement():
    """Verifies that no prohibited variables exist in feature tables."""
    # 1. Detection
    det_df = pd.read_csv(DETECTION_DIR / "detection_feature_table.csv")
    prohibited_det = ["VH_pre", "VV_pre"]
    for p in prohibited_det:
        assert p not in det_df.columns, f"Prohibited feature {p} found in detection feature table!"

    # 2. Susceptibility
    susc_df = pd.read_csv(SUSCEPTIBILITY_DIR / "susceptibility_feature_table.csv")
    prohibited_susc = ["VH_post", "VV_post", "dVH", "dVV", "rain_1d", "rain_7d"]
    for p in prohibited_susc:
        assert p not in susc_df.columns, f"Prohibited post-event feature {p} found in susceptibility table!"


def test_temporal_leakage_absence():
    """Verifies that susceptibility models do not ingest post-event or delta variables."""
    models_dict = joblib.load(SUSCEPTIBILITY_DIR / "susceptibility_models.joblib")
    prohibited = ["VH_post", "VV_post", "dVH", "dVV", "rain_1d", "rain_7d", "latitude", "longitude"]
    
    for config_name in ["core", "optical", "reduced"]:
        features = models_dict[config_name]["feature_cols"]
        for p in prohibited:
            assert p not in features, f"Temporal leakage: {p} found in {config_name} susceptibility model!"
        if config_name == "reduced":
            assert "slope" not in features and "tri" not in features, "Reduced model must exclude slope and tri"


def test_actual_purge_enforcement():
    """Verifies that in purged folds, actual min train-test distance >= purge buffer."""
    sens_df = pd.read_csv(METADATA_DIR / "spatial_purging_sensitivity.csv")
    for _, row in sens_df.iterrows():
        buf = row["purge_buffer_km"]
        if buf > 0:
            actual_min = row["actual_min_distance_to_test_km"]
            # Allow a tiny 0.1km numerical tolerance
            assert actual_min >= (buf - 0.1), (
                f"Purge violation: buffer={buf}km but actual min distance={actual_min}km in fold {row['fold']}"
            )


def test_spatial_fold_separation_and_retained_counts():
    """Verifies that every fold retains sufficient positive and negative training samples."""
    fold_df = pd.read_csv(DETECTION_DIR / "detection_fold_metrics.csv")
    assert len(fold_df) == 5, "Must have exactly 5 outer spatial folds"
    assert fold_df["n_test"].sum() == 254, "Total test samples across folds must equal 254"

    for _, row in fold_df.iterrows():
        assert row["retained_positives"] >= 30, f"Fold {row['fold']} has too few positives: {row['retained_positives']}"
        assert row["retained_negatives"] >= 30, f"Fold {row['fold']} has too few negatives: {row['retained_negatives']}"
        assert row["n_train_retained"] >= 100, f"Fold {row['fold']} retained too few samples: {row['n_train_retained']}"


def test_artifact_consistency_and_metadata():
    """Verifies all required artifact files exist and match schema contracts."""
    required_files = [
        COMMON_DIR / "dataset_summary.json",
        COMMON_DIR / "preprocessing_funnel.csv",
        COMMON_DIR / "feature_missingness.csv",
        COMMON_DIR / "feature_correlation.csv",
        COMMON_DIR / "vif_report.csv",
        DETECTION_DIR / "detection_feature_table.csv",
        DETECTION_DIR / "detection_metrics.csv",
        DETECTION_DIR / "detection_fold_metrics.csv",
        DETECTION_DIR / "detection_predictions.csv",
        DETECTION_DIR / "detection_confusion_matrix.csv",
        DETECTION_DIR / "detection_feature_importance.csv",
        DETECTION_DIR / "detection_shap_summary.csv",
        DETECTION_DIR / "detection_models.joblib",
        SUSCEPTIBILITY_DIR / "susceptibility_feature_table.csv",
        SUSCEPTIBILITY_DIR / "susceptibility_metrics.csv",
        SUSCEPTIBILITY_DIR / "susceptibility_fold_metrics.csv",
        SUSCEPTIBILITY_DIR / "susceptibility_predictions.csv",
        SUSCEPTIBILITY_DIR / "susceptibility_feature_importance.csv",
        SUSCEPTIBILITY_DIR / "susceptibility_shap_summary.csv",
        SUSCEPTIBILITY_DIR / "susceptibility_grid_0.5deg.csv",
        SUSCEPTIBILITY_DIR / "susceptibility_models.joblib",
        METADATA_DIR / "feature_schema.json",
        METADATA_DIR / "experiment_metadata.json",
        METADATA_DIR / "spatial_fold_distance_report.json",
        METADATA_DIR / "spatial_purging_sensitivity.csv",
        METADATA_DIR / "v2_candidate_distribution_validation.json",
    ]
    for rf in required_files:
        assert rf.exists(), f"Missing required artifact: {rf}"
        assert rf.stat().st_size > 0, f"Empty artifact file: {rf}"

    with open(METADATA_DIR / "experiment_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["random_seed"] == 42
    assert "version_1_0" in meta["dataset_lineage"]
    assert "EXTRACTION_PENDING_COLAB_GEE" in meta["dataset_lineage"]["version_2_0"]


def test_model_serialization_and_inference():
    """Verifies that serialized joblib models can be loaded and generate valid probabilities."""
    # Detection models
    det_dict = joblib.load(DETECTION_DIR / "detection_models.joblib")
    rf_det = det_dict["rf"]
    xgb_det = det_dict["xgb"]
    det_cols = det_dict["feature_cols"]

    # Susceptibility models
    susc_dict = joblib.load(SUSCEPTIBILITY_DIR / "susceptibility_models.joblib")
    rf_susc = susc_dict["core"]["rf"]
    xgb_susc = susc_dict["core"]["xgb"]
    susc_cols = susc_dict["core"]["feature_cols"]

    # Create dummy test row
    X_det_dummy = pd.DataFrame([{c: 10.0 for c in det_cols}])
    X_susc_dummy = pd.DataFrame([{c: 10.0 for c in susc_cols}])

    prob_det_rf = rf_det.predict_proba(X_det_dummy)[0, 1]
    prob_det_xgb = xgb_det.predict_proba(X_det_dummy)[0, 1]
    prob_susc_rf = rf_susc.predict_proba(X_susc_dummy)[0, 1]
    prob_susc_xgb = xgb_susc.predict_proba(X_susc_dummy)[0, 1]

    for p in [prob_det_rf, prob_det_xgb, prob_susc_rf, prob_susc_xgb]:
        assert 0.0 <= p <= 1.0, f"Invalid predicted probability: {p}"


def test_v2_status_and_distribution_validation():
    """Verifies that Version 2.0 status is pending and distribution divergence is confirmed."""
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "r", encoding="utf-8") as f:
        v2_val = json.load(f)
    assert v2_val["candidate_validation_status"]["status"] == "EXTRACTION_PENDING_COLAB_GEE"
    # Slope KS test confirms severe flat-terrain divergence in Version 1.0
    assert v2_val["v1_distribution_divergence"]["slope_ks_pvalue"] < 1e-10
    assert v2_val["v1_distribution_divergence"]["slope_ks_statistic"] > 0.80

