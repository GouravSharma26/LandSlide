"""Unit tests for GeoTrustNet Phase 1 data contracts, validation, splits, and datasets."""

import os
import sys
from pathlib import Path

# On Windows, resolve torch DLL directory search path before importing torch
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.geotrustnet.data.contracts import PatchMetadata, PatchSample
from src.geotrustnet.data.validation import (
    validate_patch_spatial,
    validate_temporal_order,
    validate_future_data_leakage,
    validate_label_alignment,
    validate_split_disjointness,
)
from src.geotrustnet.data.splits import (
    generate_region_event_splits,
    save_split_manifest,
)
from src.geotrustnet.data.synthetic import (
    create_synthetic_patch,
    generate_synthetic_dataset,
)
from src.geotrustnet.data.datasets import (
    GeoTrustNetPatchDataset,
    collate_geotrustnet_patches,
)
from src.geotrustnet.data.inspector import inspect_manifest


@pytest.fixture
def spec_path():
    p = Path("configs/data/sen12landslides_spec.json")
    assert p.exists(), "sen12landslides_spec.json must exist"
    return p


@pytest.fixture
def synthetic_sample():
    return create_synthetic_patch(
        sample_id="TEST_001",
        region_id="REG_01",
        event_id="EVT_01",
        height=128,
        width=128,
        time_steps=5,
        seed=100,
    )


# -----------------------------------------------------------------------------
# 1. Schema & Specification Tests
# -----------------------------------------------------------------------------

def test_sen12landslides_spec_schema(spec_path):
    """Verifies that the Sen12Landslides specification conforms to all required schema contracts."""
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    assert spec["dataset_id"] == "Sen12Landslides"
    assert "spatial_specification" in spec
    assert spec["spatial_specification"]["patch_height"] == 128
    assert spec["spatial_specification"]["patch_width"] == 128
    assert spec["spatial_specification"]["pixel_resolution_meters"] == 10.0

    # Verify modalities
    modalities = spec["modalities"]
    assert "optical" in modalities
    assert len(modalities["optical"]["bands"]) == 10
    assert "sar" in modalities
    assert modalities["sar"]["polarizations"] == ["VV", "VH"]
    assert "dem" in modalities
    assert "mask" in modalities
    assert modalities["mask"]["binary"] is True


# -----------------------------------------------------------------------------
# 2. Patch Contract & Shape Tests
# -----------------------------------------------------------------------------

def test_patch_contract_valid(synthetic_sample):
    """Verifies that a well-formed PatchSample passes internal contract validation."""
    is_valid, errors = synthetic_sample.validate()
    assert is_valid, f"Validation failed with errors: {errors}"
    assert synthetic_sample.optical.shape == (5, 10, 128, 128)
    assert synthetic_sample.sar.shape == (5, 2, 128, 128)
    assert synthetic_sample.dem.shape == (2, 128, 128)
    assert synthetic_sample.mask.shape == (1, 128, 128)


def test_patch_spatial_validation_invalid_shapes():
    """Verifies that non-matching spatial shapes are rejected."""
    bad_arr = np.zeros((10, 64, 64), dtype=np.float32)
    is_valid, errors = validate_patch_spatial(bad_arr, expected_shape=(128, 128))
    assert not is_valid
    assert any("Spatial dimensions" in e for e in errors)


def test_patch_spatial_validation_nan_and_inf():
    """Verifies that unmasked NaN and Inf values are caught."""
    nan_arr = np.zeros((128, 128), dtype=np.float32)
    nan_arr[10, 10] = np.nan
    is_valid, errors = validate_patch_spatial(nan_arr, allow_nan=False)
    assert not is_valid
    assert any("NaN" in e for e in errors)

    inf_arr = np.zeros((128, 128), dtype=np.float32)
    inf_arr[5, 5] = np.inf
    is_valid_inf, errors_inf = validate_patch_spatial(inf_arr)
    assert not is_valid_inf
    assert any("Infinite" in e for e in errors_inf)


# -----------------------------------------------------------------------------
# 3. Temporal Validation & Leakage Prevention Tests
# -----------------------------------------------------------------------------

def test_temporal_order_validation_success():
    """Verifies that chronological timestamps pass validation."""
    ts = ["2020-07-01T00:00:00Z", "2020-07-07T00:00:00Z", "2020-07-13T00:00:00Z"]
    is_valid, errors = validate_temporal_order(ts)
    assert is_valid
    assert len(errors) == 0


def test_temporal_order_validation_inversion():
    """Verifies that temporal inversions are caught."""
    ts_reversed = ["2020-07-13T00:00:00Z", "2020-07-07T00:00:00Z"]
    is_valid, errors = validate_temporal_order(ts_reversed)
    assert not is_valid
    assert any("Temporal inversion" in e for e in errors)


def test_future_data_leakage_detection():
    """Verifies that observation timestamps after the prediction cutoff are caught as leakage."""
    ts = ["2020-07-01T00:00:00Z", "2020-07-15T00:00:00Z", "2020-07-25T00:00:00Z"]
    cutoff = "2020-07-20T00:00:00Z"
    is_valid, errors = validate_future_data_leakage(ts, cutoff)
    assert not is_valid
    assert any("Future data leakage detected" in e for e in errors)


# -----------------------------------------------------------------------------
# 4. Label Alignment & Binary Mask Tests
# -----------------------------------------------------------------------------

def test_label_alignment_valid():
    """Verifies that binary masks match the spatial grid."""
    mask = np.zeros((1, 128, 128), dtype=np.float32)
    mask[0, 50:60, 50:60] = 1.0
    is_valid, errors = validate_label_alignment(mask, expected_spatial_shape=(128, 128))
    assert is_valid
    assert len(errors) == 0


def test_label_alignment_non_binary_values():
    """Verifies that non-binary ground truth values (e.g. 2 or -1) are rejected."""
    bad_mask = np.zeros((1, 128, 128), dtype=np.float32)
    bad_mask[0, 10, 10] = 2.0
    is_valid, errors = validate_label_alignment(bad_mask)
    assert not is_valid
    assert any("Non-binary ground-truth value" in e for e in errors)


# -----------------------------------------------------------------------------
# 5. Split Disjointness & Reproducibility Tests
# -----------------------------------------------------------------------------

def test_split_generation_disjointness():
    """Verifies that region-held-out splits produce strictly zero leakage across train, val, and test."""
    # Create test DataFrame with 5 regions, 3 events each, 2 samples per event = 30 samples
    records = []
    for r in range(5):
        for e in range(3):
            for s in range(2):
                records.append({
                    "sample_id": f"SMP_{r}_{e}_{s}",
                    "region_id": f"REG_{r}",
                    "event_id": f"EVT_{r}_{e}",
                })
    df = pd.DataFrame(records)

    split_df, split_hash = generate_region_event_splits(
        df, group_col="region_id", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )

    is_disjoint, report = validate_split_disjointness(split_df, group_cols=["region_id", "event_id"])
    assert is_disjoint, f"Split disjointness violated: {report['violations']}"
    assert len(report["violations"]) == 0
    assert len(split_hash) == 64  # valid SHA-256


def test_split_reproducibility():
    """Verifies that identical seeds produce identical hashes, and different seeds produce different splits."""
    records = [{"sample_id": f"S_{i}", "region_id": f"R_{i%4}", "event_id": f"E_{i%8}"} for i in range(20)]
    df = pd.DataFrame(records)

    df1, hash1 = generate_region_event_splits(df, group_col="region_id", seed=42)
    df2, hash2 = generate_region_event_splits(df, group_col="region_id", seed=42)
    df3, hash3 = generate_region_event_splits(df, group_col="region_id", seed=99)

    assert hash1 == hash2, "Split hash must be identical for identical seeds"
    assert hash1 != hash3, "Split hash should vary across different random seeds"


# -----------------------------------------------------------------------------
# 6. Dataset & Missing Modality Tests
# -----------------------------------------------------------------------------

def test_missing_modality_handling():
    """Verifies that missing modalities produce valid zero-tensors and explicit boolean false flags."""
    sample_no_opt = create_synthetic_patch(
        sample_id="NO_OPT",
        region_id="REG_01",
        event_id="EVT_01",
        has_optical=False,
        has_sar=True,
        has_dem=True,
    )
    assert sample_no_opt.optical_available is False
    assert sample_no_opt.optical is None
    is_valid, _ = sample_no_opt.validate()
    assert is_valid


def test_geotrustnet_dataset_loading_and_collate():
    """Verifies that GeoTrustNetPatchDataset loads from manifests and collates into PyTorch batches."""
    manifest_path = Path("data/manifests/synthetic_split_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("synthetic_split_manifest.csv not yet generated")

    manifest_df = pd.read_csv(manifest_path)
    ds = GeoTrustNetPatchDataset(manifest_df=manifest_df, split="train", fixed_time_steps=5)
    assert len(ds) > 0

    item = ds[0]
    assert item["optical"].shape == (5, 10, 128, 128)
    assert item["sar"].shape == (5, 2, 128, 128)
    assert item["dem"].shape == (2, 128, 128)
    assert item["mask"].shape == (1, 128, 128)

    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)
    batch = next(iter(loader))
    assert batch["optical"].shape == (2, 5, 10, 128, 128)
    assert batch["sar"].shape == (2, 5, 2, 128, 128)
    assert batch["dem"].shape == (2, 2, 128, 128)
    assert batch["mask"].shape == (2, 1, 128, 128)
    assert batch["has_optical"].shape == (2,)


def test_manifest_inspector():
    """Verifies that the dataset inspector utility computes correct summary statistics."""
    manifest_path = Path("data/manifests/synthetic_split_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("synthetic_split_manifest.csv not yet generated")

    stats = inspect_manifest(manifest_path)
    assert stats["total_samples"] == 24
    assert stats["n_regions"] == 4
    assert stats["is_leakage_free_disjoint"] is True
