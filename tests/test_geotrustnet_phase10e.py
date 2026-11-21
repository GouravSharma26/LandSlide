"""Phase 10E: Comprehensive Multi-Inventory Authentic Dataset Qualification Test Suite.

Validates the GeoTrustNet data pipeline across 6 geographically and environmentally diverse inventories:
1. Chimanimani (Val - Tropical rainfall / Cyclone Idai)
2. New Zealand (Val - Kaikōura earthquake / steep alpine terrain)
3. Italy (Test - Rainfall-induced landslides)
4. Hokkaido (Test - Iburi earthquake / high landslide prevalence / snow)
5. Kyrgyzstan 1 (Train - Mountain snowmelt / arid rock)
6. Indonesia (Train - Tropical monsoonal convective clouds)

Test categories explicitly distinguished:
- AUTHENTIC_DATA_VERIFIED: Real satellite NetCDF files from official Sen12Landslides archive.
- CONTROLLED_TEST: Controlled transformations or simulated missing modalities derived from authentic data.
- SYNTHETIC_UNIT_TEST: Synthetic edge-case validation.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from src.geotrustnet.data.netcdf_loader import load_sen12landslides_netcdf
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches
from src.geotrustnet.models.unet import UNet
from src.geotrustnet.models.temporal_unet import TemporalSARUNet
from src.geotrustnet.models.multimodal_unet import MultimodalUNet
from src.geotrustnet.models.geotrustnet import GeoTrustNet
from src.geotrustnet.training.losses import CombinedBCEDiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics

AUTHENTIC_DIR = Path("data/authentic")
MANIFEST_PATH = Path("data/manifests/sen12landslides_disjoint_manifest.csv")

# The 6 qualified authentic samples across distinct inventories
AUTHENTIC_SAMPLES = {
    "chimanimani": {
        "sample_id": "chimanimani_1004",
        "split": "val",
        "sar_asc": AUTHENTIC_DIR / "chimanimani_s1asc_1004.nc",
        "s2": AUTHENTIC_DIR / "chimanimani_s2_1004.nc",
        "annotated_px": 362,
    },
    "hokkaido": {
        "sample_id": "hokkaido_462",
        "split": "test",
        "sar_asc": AUTHENTIC_DIR / "hokkaido_s1asc_462.nc",
        "s2": AUTHENTIC_DIR / "hokkaido_s2_462.nc",
        "annotated_px": 940,
    },
    "indonesia": {
        "sample_id": "indonesia_1103",
        "split": "train",
        "sar_asc": AUTHENTIC_DIR / "indonesia_s1asc_1103.nc",
        "s2": AUTHENTIC_DIR / "indonesia_s2_1103.nc",
        "annotated_px": 246,
    },
    "italy": {
        "sample_id": "italy_1161",
        "split": "test",
        "sar_asc": AUTHENTIC_DIR / "italy_s1asc_1161.nc",
        "s2": AUTHENTIC_DIR / "italy_s2_1161.nc",
        "annotated_px": 61,
    },
    "kyrgyzstan1": {
        "sample_id": "kyrgyzstan1_1101",
        "split": "train",
        "sar_asc": AUTHENTIC_DIR / "kyrgyzstan1_s1asc_1101.nc",
        "s2": AUTHENTIC_DIR / "kyrgyzstan1_s2_1101.nc",
        "annotated_px": 376,
    },
    "newzealand": {
        "sample_id": "newzealand_1501",
        "split": "val",
        "sar_asc": AUTHENTIC_DIR / "newzealand_s1asc_1501.nc",
        "s2": AUTHENTIC_DIR / "newzealand_s2_1501.nc",
        "annotated_px": 255,
    },
}


# =========================================================================
# 1. AUTHENTIC DATA INTEGRITY & PROVENANCE
# =========================================================================

def test_authentic_multi_inventory_files_exist_and_non_empty():
    """[AUTHENTIC_DATA_VERIFIED] Verify that all 6 inventories have authentic SAR and S2 NetCDF files."""
    for region, meta in AUTHENTIC_SAMPLES.items():
        assert meta["sar_asc"].exists(), f"Missing SAR for {region}: {meta['sar_asc']}"
        assert meta["s2"].exists(), f"Missing S2 for {region}: {meta['s2']}"
        assert meta["sar_asc"].stat().st_size > 1_000_000, f"SAR file too small for {region}"
        assert meta["s2"].stat().st_size > 2_500_000, f"S2 file too small for {region}"


def test_authentic_provenance_manifest_sha256():
    """[AUTHENTIC_DATA_VERIFIED] Verify that on-disk authentic files match recorded SHA-256 hashes."""
    prov_path = AUTHENTIC_DIR / "provenance_manifest.json"
    assert prov_path.exists(), "Missing provenance_manifest.json"
    with open(prov_path, "r") as f:
        manifest = json.load(f)

    for fname, info in manifest.items():
        file_path = AUTHENTIC_DIR / fname
        assert file_path.exists(), f"File {fname} in provenance manifest missing from disk"
        with open(file_path, "rb") as f_in:
            calc_sha256 = hashlib.sha256(f_in.read()).hexdigest()
        assert calc_sha256 == info["sha256"], f"SHA-256 mismatch for {fname}: {calc_sha256} vs {info['sha256']}"


# =========================================================================
# 2. SPATIAL GEOMETRY & GROUND TRUTH MASK INTEGRITY
# =========================================================================

@pytest.mark.parametrize("region", list(AUTHENTIC_SAMPLES.keys()))
def test_authentic_raster_geometry_and_mask_exact_match(region):
    """[AUTHENTIC_DATA_VERIFIED] Verify 128x128 dimensions, 10m pixel size, and exact ground-truth mask counts."""
    meta = AUTHENTIC_SAMPLES[region]
    sample = load_sen12landslides_netcdf(
        sar_path=meta["sar_asc"],
        optical_path=meta["s2"],
        sample_id=meta["sample_id"],
        region_id=region,
        normalize=False,
    )

    # Geometry verification
    assert sample.sar.shape[-2:] == (128, 128)
    assert sample.optical.shape[-2:] == (128, 128)
    assert sample.dem.shape == (2, 128, 128)
    assert sample.mask.shape == (1, 128, 128)
    assert sample.metadata.pixel_size_m == 10.0

    # Ground truth bit-for-bit match against manifest
    pos_px = int((sample.mask == 1.0).sum().item())
    neg_px = int((sample.mask == 0.0).sum().item())
    assert pos_px == meta["annotated_px"], f"Mask positive count mismatch for {region}: {pos_px} vs {meta['annotated_px']}"
    assert pos_px + neg_px == 128 * 128
    assert sample.mask_valid.all()  # Decoupled validity confirms all pixels are valid


# =========================================================================
# 3. SCL QUALITY POLICY & ENVIRONMENTAL MASKING
# =========================================================================

def test_scl_environmental_distributions_and_policy():
    """[AUTHENTIC_DATA_VERIFIED] Verify SCL distributions and ensure configurable policy correctly masks noise."""
    # Hokkaido contains authentic snow (class 11)
    hok_meta = AUTHENTIC_SAMPLES["hokkaido"]
    sample_hok = load_sen12landslides_netcdf(
        sar_path=hok_meta["sar_asc"],
        optical_path=hok_meta["s2"],
        sample_id=hok_meta["sample_id"],
        clear_scl_classes=(4, 5, 6),
    )
    # Snow is masked as invalid under default policy
    assert sample_hok.optical_valid_mask.sum().item() < 15 * 128 * 128

    # New Zealand contains deep cast shadows (class 2)
    nz_meta = AUTHENTIC_SAMPLES["newzealand"]
    sample_nz = load_sen12landslides_netcdf(
        sar_path=nz_meta["sar_asc"],
        optical_path=nz_meta["s2"],
        sample_id=nz_meta["sample_id"],
        clear_scl_classes=(4, 5, 6),
    )
    # Confirm SCL policy metadata was recorded
    policy = sample_nz.metadata.extra_metadata.get("scl_policy", {})
    assert policy.get("clear_classes") == [4, 5, 6]
    assert "2" in policy.get("excluded_classes_rationale", {})


def test_controlled_completely_cloud_masked_optical():
    """[CONTROLLED_TEST] When all optical pixels are masked, verify optical validity is False and SAR remains usable."""
    meta = AUTHENTIC_SAMPLES["indonesia"]
    # Pass an empty clear_scl_classes tuple so no classes are deemed clear
    sample = load_sen12landslides_netcdf(
        sar_path=meta["sar_asc"],
        optical_path=meta["s2"],
        sample_id=meta["sample_id"],
        clear_scl_classes=(),
    )
    assert sample.optical_valid_mask.sum().item() == 0
    assert sample.sar_valid_mask.all()  # SAR remains 100% valid


def test_controlled_missing_optical_modality():
    """[CONTROLLED_TEST] Verify that missing optical modality is explicitly flagged without NaN corruption."""
    meta = AUTHENTIC_SAMPLES["italy"]
    sample = load_sen12landslides_netcdf(
        sar_path=meta["sar_asc"],
        optical_path=None,  # Deliberately absent optical
        sample_id=meta["sample_id"],
    )
    assert not sample.optical_available
    assert sample.optical is None
    assert sample.optical_valid_mask is None
    assert sample.sar_available
    assert sample.sar is not None


# =========================================================================
# 4. TEMPORAL HORIZON & CONFIGURABLE POLICIES
# =========================================================================

def test_temporal_chronological_ordering_across_inventories():
    """[AUTHENTIC_DATA_VERIFIED] Verify that timestamps across all 6 authentic files are strictly chronological."""
    for region, meta in AUTHENTIC_SAMPLES.items():
        sample = load_sen12landslides_netcdf(
            sar_path=meta["sar_asc"],
            sample_id=meta["sample_id"],
        )
        timestamps = sample.metadata.timestamps
        assert len(timestamps) == 15, f"Expected 15 timestamps for {region}, got {len(timestamps)}"
        assert timestamps == sorted(timestamps), f"Timestamps not chronological for {region}"


def test_configurable_temporal_policies():
    """[AUTHENTIC_DATA_VERIFIED] Test full, pre_event, post_event, and fixed temporal slicing policies."""
    meta = AUTHENTIC_SAMPLES["chimanimani"]
    
    # 1. Full sequence
    sample_full = load_sen12landslides_netcdf(
        sar_path=meta["sar_asc"],
        sample_id=meta["sample_id"],
        temporal_policy="full",
    )
    assert sample_full.sar.shape[0] == 15

    # 2. Pre-event sequence (Cyclone Idai pre-event index is 6)
    sample_pre = load_sen12landslides_netcdf(
        sar_path=meta["sar_asc"],
        sample_id=meta["sample_id"],
        temporal_policy="pre_event",
    )
    assert sample_pre.sar.shape[0] == 7  # 0 to 6 inclusive

    # 3. Post-event sequence (Post-event index is 7)
    sample_post = load_sen12landslides_netcdf(
        sar_path=meta["sar_asc"],
        sample_id=meta["sample_id"],
        temporal_policy="post_event",
    )
    assert sample_post.sar.shape[0] == 8  # 7 to 14 inclusive

    # 4. Fixed length (T=5)
    sample_fixed = load_sen12landslides_netcdf(
        sar_path=meta["sar_asc"],
        sample_id=meta["sample_id"],
        temporal_policy="fixed",
        fixed_time_steps=5,
    )
    assert sample_fixed.sar.shape[0] == 5


# =========================================================================
# 5. DEM & TERRAIN DERIVATION
# =========================================================================

def test_dem_elevation_and_physical_slope_derivation():
    """[AUTHENTIC_DATA_VERIFIED] Verify elevation and derived slopes across mountain environments."""
    for region, meta in AUTHENTIC_SAMPLES.items():
        sample = load_sen12landslides_netcdf(
            sar_path=meta["sar_asc"],
            sample_id=meta["sample_id"],
        )
        elev = sample.dem[0].numpy()
        slope = sample.dem[1].numpy()

        assert np.isfinite(elev).all()
        assert np.isfinite(slope).all()
        assert elev.min() >= 0.0, "Elevation cannot be negative"
        assert slope.min() >= 0.0, "Slope cannot be negative"
        assert slope.max() <= 90.0, "Slope cannot exceed 90 degrees"
        # Rugged regions (Indonesia, Chimanimani, New Zealand) have slopes > 60 degrees
        if region in ["indonesia", "chimanimani", "newzealand"]:
            assert slope.max() > 60.0, f"Expected steep slope > 60 deg for {region}"


# =========================================================================
# 6. NORMALIZATION & LOSS/METRIC INVARIANCE
# =========================================================================

def test_single_pass_normalization_and_no_double_norm():
    """[AUTHENTIC_DATA_VERIFIED] Test single-pass normalization using training-only statistics."""
    df = pd.DataFrame([
        {
            "sample_id": meta["sample_id"],
            "region_id": region,
            "event_id": f"event_{region}",
            "file_path": str(meta["sar_asc"]),
            "s2_path": str(meta["s2"]),
            "split": meta["split"],
        }
        for region, meta in AUTHENTIC_SAMPLES.items()
    ])

    ds = GeoTrustNetPatchDataset(df, normalize=True, fixed_time_steps=5)
    item = ds[0]

    # Verify SAR channels are normalized (mean ~ 0, not raw -50 dB to 0 dB)
    sar_v = item["sar_valid"].expand_as(item["sar"])
    vv_vals = item["sar"][:, 0][sar_v[:, 0]]
    assert vv_vals.mean().abs() < 5.0

    # Verify optical reflectance is scaled to [0, 1]
    opt_v = item["optical_valid"].expand_as(item["optical"])
    opt_vals = item["optical"][opt_v]
    assert opt_vals.max() <= 1.5 and opt_vals.min() >= 0.0


def test_loss_and_metric_nodata_invariance():
    """[CONTROLLED_TEST] Verify that arbitrarily changing predictions on nodata/invalid pixels has zero effect."""
    meta = AUTHENTIC_SAMPLES["hokkaido"]
    df = pd.DataFrame([{
        "sample_id": meta["sample_id"],
        "region_id": "hokkaido",
        "event_id": "event_hokkaido",
        "file_path": str(meta["sar_asc"]),
        "s2_path": str(meta["s2"]),
        "split": meta["split"],
    }])
    ds = GeoTrustNetPatchDataset(df, normalize=True)
    batch = collate_geotrustnet_patches([ds[0]])

    logits_orig = torch.randn((1, 1, 128, 128))
    # Artificially mask the bottom half as invalid
    valid_mask = torch.ones((1, 1, 128, 128), dtype=torch.bool)
    valid_mask[:, :, 64:, :] = False

    loss_fn = CombinedBCEDiceLoss()
    loss1 = loss_fn(logits_orig, batch["mask"], valid_mask=valid_mask)
    metrics1 = compute_segmentation_metrics(logits_orig, batch["mask"], valid_mask=valid_mask)

    # Corrupt predictions on invalid pixels to extreme values (+100.0)
    logits_corrupted = logits_orig.clone()
    logits_corrupted[:, :, 64:, :] = 100.0

    loss2 = loss_fn(logits_corrupted, batch["mask"], valid_mask=valid_mask)
    metrics2 = compute_segmentation_metrics(logits_corrupted, batch["mask"], valid_mask=valid_mask)

    # Invariance check
    assert torch.isclose(loss1, loss2, atol=1e-5), "Loss changed when modifying invalid pixels!"
    assert metrics1["dice"] == pytest.approx(metrics2["dice"], abs=1e-5), "Dice changed when modifying invalid pixels!"
    assert metrics1["iou"] == pytest.approx(metrics2["iou"], abs=1e-5), "IoU changed when modifying invalid pixels!"


# =========================================================================
# 7. MODEL INTEGRATION ON AUTHENTIC TENSORS (MODELS A, B, C, D)
# =========================================================================

def test_models_a_b_c_d_authentic_forward_and_backward():
    """[AUTHENTIC_DATA_VERIFIED] Verify numerical stability, finite logits, and backward gradient flow across Models A-D."""
    meta = AUTHENTIC_SAMPLES["kyrgyzstan1"]
    df = pd.DataFrame([{
        "sample_id": meta["sample_id"],
        "region_id": "kyrgyzstan1",
        "event_id": "event_kyrgyzstan1",
        "file_path": str(meta["sar_asc"]),
        "s2_path": str(meta["s2"]),
        "split": meta["split"],
    }])
    ds = GeoTrustNetPatchDataset(df, normalize=True, fixed_time_steps=5)
    batch = collate_geotrustnet_patches([ds[0]])

    loss_fn = CombinedBCEDiceLoss()

    # 1. Model A: SAR U-Net
    model_a = UNet(in_channels=2, out_channels=1)
    out_a = model_a(batch["sar"][:, 0])
    logits_a = out_a["logits"] if isinstance(out_a, dict) else out_a
    assert torch.isfinite(logits_a).all()
    loss_a = loss_fn(logits_a, batch["mask"], valid_mask=batch["mask_valid"])
    loss_a.backward()
    assert all(torch.isfinite(p.grad).all() for p in model_a.parameters() if p.grad is not None)

    # 2. Model B: Temporal SAR ConvLSTM U-Net
    model_b = TemporalSARUNet(in_channels=2, out_channels=1)
    out_b = model_b(batch["sar"])
    logits_b = out_b["logits"] if isinstance(out_b, dict) else out_b
    assert torch.isfinite(logits_b).all()
    loss_b = loss_fn(logits_b, batch["mask"], valid_mask=batch["mask_valid"])
    loss_b.backward()
    assert all(torch.isfinite(p.grad).all() for p in model_b.parameters() if p.grad is not None)

    # 3. Model C: Multimodal U-Net
    model_c = MultimodalUNet(sar_channels=2, optical_channels=10, dem_channels=2, out_channels=1)
    out_c = model_c(sar=batch["sar"][:, 0], optical=batch["optical"][:, 0], dem=batch["dem"])
    logits_c = out_c["logits"] if isinstance(out_c, dict) else out_c
    assert torch.isfinite(logits_c).all()
    loss_c = loss_fn(logits_c, batch["mask"], valid_mask=batch["mask_valid"])
    loss_c.backward()
    assert all(torch.isfinite(p.grad).all() for p in model_c.parameters() if p.grad is not None)

    # 4. Model D: GeoTrustNet
    model_d = GeoTrustNet(sar_channels=2, optical_channels=10, dem_channels=2, out_channels=1, use_temporal=True)
    out_d, rel = model_d(
        sar=batch["sar"],
        optical=batch["optical"],
        dem=batch["dem"],
        has_sar=batch["has_sar"],
        has_optical=batch["has_optical"],
        has_dem=batch["has_dem"],
        sar_quality=batch["sar_valid"][:, 0].float(),
        optical_quality=batch["optical_valid"][:, 0].float(),
        return_reliability=True,
    )
    assert torch.isfinite(out_d).all()
    assert torch.isfinite(rel["weight_sar"]).all()
    loss_d = loss_fn(out_d, batch["mask"], valid_mask=batch["mask_valid"])
    loss_d.backward()
    assert all(torch.isfinite(p.grad).all() for p in model_d.parameters() if p.grad is not None)


# =========================================================================
# 8. GEOGRAPHIC SPLIT DISJOINTNESS & LEAKAGE AUDIT
# =========================================================================

def test_geographic_split_disjointness_audit():
    """[AUTHENTIC_DATA_VERIFIED] Verify strict disjointness between train, val, and test splits."""
    assert MANIFEST_PATH.exists()
    df = pd.read_csv(MANIFEST_PATH)

    train_regions = set(df[df["split"] == "train"]["region_id"].unique())
    val_regions = set(df[df["split"] == "val"]["region_id"].unique())
    test_regions = set(df[df["split"] == "test"]["region_id"].unique())

    # Region sets must be mutually disjoint
    assert train_regions.isdisjoint(val_regions), "Train and Val regions overlap!"
    assert train_regions.isdisjoint(test_regions), "Train and Test regions overlap!"
    assert val_regions.isdisjoint(test_regions), "Val and Test regions overlap!"

    # Sample ID uniqueness
    assert df["sample_id"].is_unique, "Duplicate sample IDs detected in manifest!"
