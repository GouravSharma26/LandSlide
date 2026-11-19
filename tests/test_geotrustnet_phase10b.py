"""Phase 10B Verification Test Suite: NetCDF Pipeline, Nodata Masking, Dual Disjointness, and Temporal Mode."""

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

import pytest
import numpy as np
import pandas as pd
import torch

from src.geotrustnet.data.netcdf_loader import load_sen12landslides_netcdf
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches
from src.geotrustnet.data.splits import generate_region_event_splits
from src.geotrustnet.data.validation import validate_split_disjointness
from src.geotrustnet.training.losses import DiceLoss, CombinedBCEDiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.models.geotrustnet import GeoTrustNet
from scripts.create_sample_netcdf_fixtures import generate_sample_netcdf


@pytest.fixture(scope="session")
def netcdf_fixtures(tmp_path_factory):
    """Generates temporary NetCDF fixtures for testing."""
    fixture_dir = tmp_path_factory.mktemp("netcdf_test")
    generate_sample_netcdf(fixture_dir, height=64, width=64, time_steps=4, seed=123)
    return fixture_dir


class TestNetCDFLoader:
    """Verifies that NetCDF files from Sen12Landslides are parsed and loaded correctly."""

    def test_load_sen12landslides_netcdf_sar_and_optical(self, netcdf_fixtures):
        sar_path = netcdf_fixtures / "sample_sar.nc"
        opt_path = netcdf_fixtures / "sample_optical.nc"

        sample = load_sen12landslides_netcdf(
            sar_path=sar_path,
            optical_path=opt_path,
            sample_id="NC_001",
            region_id="REG_TEST",
            event_id="EVT_TEST",
            fixed_time_steps=4,
            height=64,
            width=64,
            normalize=True,
        )

        assert sample.metadata.patch_id == "NC_001"
        assert sample.sar is not None
        assert sample.sar.shape == (4, 2, 64, 64)
        assert sample.optical is not None
        assert sample.optical.shape == (4, 10, 64, 64)
        assert sample.dem is not None
        assert sample.dem.shape == (2, 64, 64)
        assert sample.mask is not None
        assert sample.mask.shape == (1, 64, 64)

        # Check validity masks
        assert sample.sar_valid_mask is not None
        assert sample.optical_valid_mask is not None
        assert sample.sar_available is True
        assert sample.optical_available is True
        assert sample.dem_available is True

    def test_netcdf_dataset_integration(self, netcdf_fixtures, tmp_path):
        sar_path = netcdf_fixtures / "sample_sar.nc"

        manifest_df = pd.DataFrame([{
            "sample_id": "NC_SMP_1",
            "region_id": "REG_NC",
            "event_id": "EVT_NC",
            "event_date": "2020-07-27T10:00:00Z",
            "has_optical": False,
            "has_sar": True,
            "has_dem": True,
            "has_mask": True,
            "n_time_steps": 4,
            "landslide_pixel_ratio": 0.05,
            "file_path": str(sar_path),
            "split": "train",
        }])

        dataset = GeoTrustNetPatchDataset(
            manifest_df=manifest_df,
            split="train",
            fixed_time_steps=4,
            height=64,
            width=64,
        )
        assert len(dataset) == 1
        item = dataset[0]

        assert "sar" in item
        assert item["sar"].shape == (4, 2, 64, 64)
        assert "mask" in item
        assert "mask_valid" in item
        assert item["mask_valid"].shape == (1, 64, 64)


class TestNodataMaskingInLossAndMetrics:
    """Verifies that nodata pixels (255) are strictly excluded from loss and evaluation."""

    def test_loss_excludes_nodata_pixels(self):
        loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

        # 1 sample, 1 channel, 4x4
        # Target: 8 zeros, 4 ones, 4 nodata (255)
        targets = torch.tensor([[[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [255.0, 255.0, 255.0, 255.0],
        ]]])

        # Model logits: correct predictions on valid pixels, but wild predictions on nodata
        logits = torch.tensor([[[
            [-5.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, -5.0],
            [5.0, 5.0, 5.0, 5.0],
            [-100.0, 100.0, -50.0, 50.0],  # wild predictions on nodata
        ]]])

        # Loss should be calculated ignoring the 255 pixels
        loss_with_nodata = loss_fn(logits, targets)
        assert not torch.isnan(loss_with_nodata)
        assert loss_with_nodata.item() < 0.1  # Highly accurate on valid pixels!

    def test_metrics_exclude_nodata_pixels(self):
        targets = np.array([
            [0, 0, 1, 1],
            [255, 255, 255, 255],  # nodata row
        ])
        # Probabilities: perfect on first row, arbitrary on nodata row
        probs = np.array([
            [0.1, 0.1, 0.9, 0.9],
            [0.0, 1.0, 0.0, 1.0],
        ])

        metrics = compute_segmentation_metrics(probs, targets, threshold=0.5)
        # Nodata pixels (4 pixels) should be excluded
        assert metrics["nodata_pixels_excluded"] == 4
        assert metrics["total_pixels"] == 4
        assert metrics["tp"] == 2
        assert metrics["tn"] == 2
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["dice"] == 1.0
        assert metrics["iou"] == 1.0


class TestDualSplitDisjointness:
    """Verifies that generate_region_event_splits enforces zero leakage across both region and event."""

    def test_dual_disjointness_connected_components(self):
        # Create dataset where some events belong to same region
        records = [
            {"sample_id": "S1", "region_id": "R1", "event_id": "E1"},
            {"sample_id": "S2", "region_id": "R1", "event_id": "E2"},
            {"sample_id": "S3", "region_id": "R2", "event_id": "E3"},
            {"sample_id": "S4", "region_id": "R2", "event_id": "E4"},
            {"sample_id": "S5", "region_id": "R3", "event_id": "E5"},
            {"sample_id": "S6", "region_id": "R3", "event_id": "E6"},
            {"sample_id": "S7", "region_id": "R4", "event_id": "E7"},
            {"sample_id": "S8", "region_id": "R5", "event_id": "E8"},
        ]
        df = pd.DataFrame(records)

        split_df, split_hash = generate_region_event_splits(
            df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
        )

        is_disjoint, report = validate_split_disjointness(
            split_df, split_col="split", group_cols=["region_id", "event_id"]
        )
        assert is_disjoint is True
        assert len(report["violations"]) == 0
        assert len(split_hash) == 64


class TestGeoTrustNetTemporalMode:
    """Verifies GeoTrustNet use_temporal flag."""

    def test_geotrustnet_temporal_mode_true(self):
        # 5D SAR input [B=2, T=3, C=2, H=32, W=32]
        sar_5d = torch.randn(2, 3, 2, 32, 32)
        model = GeoTrustNet(features=[16, 32, 64], use_temporal=True)
        assert model.use_temporal is True
        assert hasattr(model, "sar_convlstm")

        logits = model(sar=sar_5d)
        assert logits.shape == (2, 1, 32, 32)

    def test_geotrustnet_temporal_mode_false(self):
        sar_5d = torch.randn(2, 3, 2, 32, 32)
        model = GeoTrustNet(features=[16, 32, 64], use_temporal=False)
        assert model.use_temporal is False

        # When use_temporal=False, it uses the latest post-event slice
        logits = model(sar=sar_5d)
        assert logits.shape == (2, 1, 32, 32)
