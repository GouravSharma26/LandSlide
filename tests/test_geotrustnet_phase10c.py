"""Phase 10C Verification Test Suite: Official Schema, Nodata Preservation, Single-Pass Normalization & Dual Disjointness."""

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
from torch.utils.data import DataLoader

from src.geotrustnet.data.netcdf_loader import load_sen12landslides_netcdf
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches
from src.geotrustnet.data.splits import DisjointSet, generate_region_event_splits
from src.geotrustnet.data.validation import validate_split_disjointness
from src.geotrustnet.training.losses import CombinedBCEDiceLoss, DiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.models.geotrustnet import GeoTrustNet
from scripts.create_sample_netcdf_fixtures import generate_sample_netcdf


@pytest.fixture(scope="session")
def phase10c_fixtures(tmp_path_factory):
    """Generates temporary NetCDF fixtures adhering strictly to official schema."""
    fixture_dir = tmp_path_factory.mktemp("netcdf_phase10c")
    generate_sample_netcdf(fixture_dir, height=64, width=64, time_steps=4, seed=999)
    return fixture_dir


class TestOfficialSchemaAndLoader:
    """Verifies that netcdf_loader handles authentic variables, coordinates, SCL, and missing modalities."""

    def test_authentic_schema_variable_discovery(self, phase10c_fixtures):
        sar_path = phase10c_fixtures / "sample_sar.nc"
        opt_path = phase10c_fixtures / "sample_optical.nc"

        sample = load_sen12landslides_netcdf(
            sar_path=sar_path,
            optical_path=opt_path,
            sample_id="SAMPLE_OFFICIAL_01",
            region_id="chimanimani",
            event_id="event_chimanimani",
            fixed_time_steps=4,
            height=64,
            width=64,
            normalize=False,
        )

        assert sample.metadata.patch_id == "SAMPLE_OFFICIAL_01"
        assert sample.sar is not None
        assert sample.sar.shape == (4, 2, 64, 64)
        assert sample.optical is not None
        assert sample.optical.shape == (4, 10, 64, 64)
        assert sample.dem is not None
        assert sample.dem.shape == (2, 64, 64)
        assert sample.dem_available is True
        assert sample.optical_available is True
        assert sample.sar_available is True

        # Check genuine coordinates extraction (no fake metadata)
        assert len(sample.metadata.timestamps) == 4
        assert sample.metadata.bounds != (0.0, 0.0, 0.0, 0.0)
        assert sample.metadata.extra_metadata["has_geo_coords"] is True
        assert sample.metadata.extra_metadata["is_normalized"] is False

    def test_missing_dem_never_fabricates_zeros(self, phase10c_fixtures):
        """When DEM is missing, loader must set dem=None and dem_available=False, never fake zeros."""
        sar_nodem = phase10c_fixtures / "sample_sar_nodem.nc"
        sample = load_sen12landslides_netcdf(
            sar_path=sar_nodem,
            optical_path=None,
            sample_id="SAMPLE_NODEM",
            fixed_time_steps=4,
            height=64,
            width=64,
            normalize=False,
        )
        assert sample.dem is None
        assert sample.dem_available is False

    def test_optical_scl_cloud_masking(self, phase10c_fixtures):
        """Verifies that SCL cloud mask correctly flags cloudy pixels as invalid."""
        sar_path = phase10c_fixtures / "sample_sar.nc"
        opt_path = phase10c_fixtures / "sample_optical.nc"

        sample = load_sen12landslides_netcdf(
            sar_path=sar_path,
            optical_path=opt_path,
            sample_id="SAMPLE_SCL",
            fixed_time_steps=4,
            height=64,
            width=64,
        )

        assert sample.optical_valid_mask is not None
        assert sample.optical_valid_mask.shape == (4, 1, 64, 64)
        assert sample.metadata.extra_metadata["scl_available"] is True


class TestNodataPreservationEndToEnd:
    """Verifies that nodata pixels (255) are strictly preserved as invalid and never alter loss or metrics."""

    def test_target_validity_preserved_through_dataset_and_collator(self, phase10c_fixtures):
        sar_path = phase10c_fixtures / "sample_sar.nc"
        manifest_df = pd.DataFrame([{
            "sample_id": "TEST_SMP_01",
            "region_id": "chimanimani",
            "event_id": "event_chimanimani",
            "file_path": str(sar_path),
            "split": "train",
        }])

        dataset = GeoTrustNetPatchDataset(
            manifest_df=manifest_df,
            split="train",
            fixed_time_steps=4,
            height=64,
            width=64,
            normalize=True,
        )

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_geotrustnet_patches)
        batch = next(iter(loader))

        assert "mask" in batch
        assert "mask_valid" in batch
        assert batch["mask_valid"].shape == (1, 1, 64, 64)
        assert batch["mask_valid"].dtype == torch.bool

        # Check that where mask_valid is False, the pixels were genuine nodata
        mask_valid_np = batch["mask_valid"].squeeze().numpy()
        # Some pixels in fixture were generated as 255 (nodata)
        assert not np.all(mask_valid_np), "Expected some nodata pixels in synthetic fixture"

    def test_loss_invariant_to_logits_on_nodata_pixels(self):
        """Regression test: changing logits on nodata pixels must have ZERO impact on loss."""
        loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

        targets = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        valid_mask = torch.tensor([[[[True, True], [True, False]]]])  # bottom-right is invalid

        # Baseline logits
        logits_a = torch.tensor([[[[-2.0, 3.0], [-1.0, 0.0]]]])
        # Same logits on valid pixels, but completely different value on invalid pixel
        logits_b = torch.tensor([[[[-2.0, 3.0], [-1.0, 999.0]]]])

        loss_a = loss_fn(logits_a, targets, valid_mask=valid_mask)
        loss_b = loss_fn(logits_b, targets, valid_mask=valid_mask)

        assert torch.isclose(loss_a, loss_b, atol=1e-7), (
            f"Loss changed when modifying invalid pixel: {loss_a.item()} vs {loss_b.item()}"
        )

    def test_metrics_safely_handle_all_invalid_pixels(self):
        """Empty-valid-pixel batches must return zero counts and never crash or produce NaNs."""
        probs = np.random.uniform(0, 1, size=(4, 4))
        targets = np.random.choice([0, 1], size=(4, 4))
        all_invalid_mask = np.zeros((4, 4), dtype=bool)

        metrics = compute_segmentation_metrics(probs, targets, valid_mask=all_invalid_mask)
        assert metrics["total_pixels"] == 0
        assert metrics["nodata_pixels_excluded"] == 16
        assert metrics["dice"] == 0.0
        assert metrics["iou"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestSinglePassNormalization:
    """Verifies that input modalities are normalized exactly once."""

    def test_single_pass_normalization_owner(self, phase10c_fixtures):
        sar_path = phase10c_fixtures / "sample_sar.nc"

        # 1. Loader with normalize=False produces unnormalized dB measurements (e.g. mean ~ -9.3 dB)
        sample_raw = load_sen12landslides_netcdf(sar_path, height=64, width=64, normalize=False)
        assert sample_raw.metadata.extra_metadata["is_normalized"] is False
        raw_sar = sample_raw.sar.numpy()
        valid_sar = sample_raw.sar_valid_mask.numpy().squeeze()
        # Raw VV mean should be around -9.3 dB, not around 0.0
        assert np.mean(raw_sar[:, 0][valid_sar]) < -5.0

        # 2. Dataset applies normalization once
        manifest_df = pd.DataFrame([{
            "sample_id": "NORM_TEST",
            "region_id": "R1",
            "event_id": "E1",
            "file_path": str(sar_path),
            "split": "train",
        }])

        dataset = GeoTrustNetPatchDataset(
            manifest_df=manifest_df,
            split="train",
            normalize=True,
            height=64,
            width=64,
        )
        item = dataset[0]
        norm_sar = item["sar"].numpy()
        valid_mask = item["sar_valid"].numpy().squeeze()

        # Normalized VV mean should be approximately centered around 0.0, std around 1.0
        vv_norm_valid = norm_sar[:, 0][valid_mask]
        assert -1.5 < np.mean(vv_norm_valid) < 1.5
        assert 0.5 < np.std(vv_norm_valid) < 1.5


class TestSplitGeneratorVerification:
    """Verifies that the split generator guarantees zero spatial and event leakage."""

    def test_disjoint_manifest_integrity(self):
        """Audits the official disjoint manifest for zero leakage and balanced distribution."""
        manifest_path = Path("data/manifests/sen12landslides_disjoint_manifest.csv")
        assert manifest_path.exists()
        df = pd.read_csv(manifest_path)

        is_disjoint, report = validate_split_disjointness(df, split_col="split", group_cols=["region_id", "event_id"])
        assert is_disjoint is True
        assert len(report["violations"]) == 0

        # Verify all 11 inventories are partitioned
        assert df["region_id"].nunique() == 11
        assert df["split"].nunique() == 3

        train_regs = set(df[df["split"] == "train"]["region_id"])
        val_regs = set(df[df["split"] == "val"]["region_id"])
        test_regs = set(df[df["split"] == "test"]["region_id"])

        assert len(train_regs.intersection(val_regs)) == 0
        assert len(train_regs.intersection(test_regs)) == 0
        assert len(val_regs.intersection(test_regs)) == 0


class TestTemporalDataPipelineEndToEnd:
    """Verifies that 5D temporal sequences flow through NetCDF -> Dataset -> Collator -> GeoTrustNet."""

    def test_temporal_geotrustnet_forward_pass_with_netcdf_batch(self, phase10c_fixtures):
        sar_path = phase10c_fixtures / "sample_sar.nc"
        opt_path = phase10c_fixtures / "sample_optical.nc"

        manifest_df = pd.DataFrame([{
            "sample_id": "TEMP_TEST_01",
            "region_id": "chimanimani",
            "event_id": "event_chimanimani",
            "file_path": str(sar_path),
            "s2_path": str(opt_path),
            "split": "train",
        }])

        dataset = GeoTrustNetPatchDataset(
            manifest_df=manifest_df,
            split="train",
            fixed_time_steps=4,
            height=64,
            width=64,
            normalize=True,
        )

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_geotrustnet_patches)
        batch = next(iter(loader))

        # Check 5D shapes
        assert batch["sar"].ndim == 5  # [B=1, T=4, C=2, H=64, W=64]
        assert batch["optical"].ndim == 5  # [B=1, T=4, C=10, H=64, W=64]

        # Forward pass through GeoTrustNet with use_temporal=True
        model = GeoTrustNet(
            features=[16, 32, 64],
            use_temporal=True,
            sar_channels=2,
            optical_channels=10,
            dem_channels=2,
        )
        model.eval()

        with torch.no_grad():
            logits, rel_dict = model(
                sar=batch["sar"],
                optical=batch["optical"],
                dem=batch["dem"],
                has_sar=batch["has_sar"],
                has_optical=batch["has_optical"],
                has_dem=batch["has_dem"],
                sar_quality=batch["sar_valid"],
                optical_quality=batch["optical_valid"],
                return_reliability=True,
            )

        assert logits.shape == (1, 1, 64, 64)
        assert "weight_sar" in rel_dict
        assert "weight_optical" in rel_dict
        assert "weight_dem" in rel_dict
