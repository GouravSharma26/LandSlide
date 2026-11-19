"""Unit tests for GeoTrustNet Phase 4 Multimodal Fusion (SAR, Optical, DEM) and missing-modality handling."""

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

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.geotrustnet.models.encoders import SAREncoder, OpticalEncoder, TerrainEncoder
from src.geotrustnet.models.fusion import FeatureLevelFusion
from src.geotrustnet.models.multimodal_unet import MultimodalUNet
from src.geotrustnet.training.engine import (
    GeoTrustNetTrainer,
    extract_batch_tensors,
    seed_everything,
)
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches


class TestModalityEncoders:
    """Test individual modality encoders and their availability masking behavior."""

    def test_sar_encoder_output_shapes(self):
        """SAREncoder processes [B, 2, H, W] and outputs bottleneck and multi-scale skips."""
        encoder = SAREncoder(in_channels=2, features=[16, 32, 64])
        x = torch.randn(2, 2, 64, 64)
        bot, skips = encoder(x)

        assert bot.shape == (2, 64, 16, 16)
        assert len(skips) == 2
        assert skips[0].shape == (2, 16, 64, 64)
        assert skips[1].shape == (2, 32, 32, 32)

    def test_optical_encoder_output_shapes(self):
        """OpticalEncoder processes 10 spectral bands [B, 10, H, W]."""
        encoder = OpticalEncoder(in_channels=10, features=[16, 32, 64])
        x = torch.randn(2, 10, 64, 64)
        bot, skips = encoder(x)

        assert bot.shape == (2, 64, 16, 16)
        assert len(skips) == 2

    def test_dem_encoder_output_shapes(self):
        """TerrainEncoder processes elevation and slope [B, 2, H, W]."""
        encoder = TerrainEncoder(in_channels=2, features=[16, 32, 64])
        x = torch.randn(2, 2, 64, 64)
        bot, skips = encoder(x)

        assert bot.shape == (2, 64, 16, 16)

    def test_encoder_availability_masking(self):
        """When a modality is marked unavailable (False/0), its outputs are strictly zero."""
        encoder = SAREncoder(in_channels=2, features=[16, 32])
        x = torch.randn(2, 2, 32, 32)
        avail = torch.tensor([1, 0], dtype=torch.bool)  # Item 0 is available, item 1 is unavailable

        bot, skips = encoder(x, is_available=avail)

        # Item 0 must have non-zero activations
        assert not torch.allclose(bot[0], torch.zeros_like(bot[0]))
        # Item 1 must be strictly zero
        assert torch.allclose(bot[1], torch.zeros_like(bot[1]))
        assert torch.allclose(skips[0][1], torch.zeros_like(skips[0][1]))


class TestFeatureLevelFusion:
    """Test feature fusion conditioning, gates, and missing-modality indicator handling."""

    def test_fusion_concat_shapes(self):
        """Concatenation fusion correctly projects merged features back to feature_dim."""
        fusion = FeatureLevelFusion(feature_dim=32, num_modalities=3, out_dim=32, fusion_type="concat")
        z1 = torch.randn(2, 32, 16, 16)
        z2 = torch.randn(2, 32, 16, 16)
        z3 = torch.randn(2, 32, 16, 16)
        avails = [
            torch.tensor([1, 1], dtype=torch.bool),
            torch.tensor([1, 0], dtype=torch.bool),  # optical missing for item 1
            torch.tensor([1, 1], dtype=torch.bool),
        ]

        fused = fusion([z1, z2, z3], avails)
        assert fused.shape == (2, 32, 16, 16)
        assert not torch.isnan(fused).any()

    def test_fusion_gated_and_residual(self):
        """Verify gated and residual_sar fusion types execute without numerical instabilities."""
        for ftype in ["gated", "residual_sar"]:
            fusion = FeatureLevelFusion(feature_dim=16, num_modalities=3, out_dim=16, fusion_type=ftype)
            feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
            avails = [torch.tensor([1, 1], dtype=torch.bool) for _ in range(3)]
            fused = fusion(feats, avails)
            assert fused.shape == (2, 16, 8, 8), f"Failed for fusion_type '{ftype}'"


class TestMultimodalUNet:
    """Test end-to-end multimodal segmentation network and missing modality robustness."""

    def test_all_modalities_forward(self):
        """Model produces [B, 1, H, W] when SAR, Optical, and DEM are all present."""
        model = MultimodalUNet(features=[16, 32, 64])
        sar = torch.randn(2, 2, 64, 64)
        opt = torch.randn(2, 10, 64, 64)
        dem = torch.randn(2, 2, 64, 64)

        logits = model(sar=sar, optical=opt, dem=dem)
        assert logits.shape == (2, 1, 64, 64)
        assert not torch.isnan(logits).any()

    def test_missing_optical_fallback(self):
        """When optical imagery is missing (cloud/night), model operates on SAR + DEM."""
        model = MultimodalUNet(features=[16, 32, 64])
        sar = torch.randn(2, 2, 64, 64)
        dem = torch.randn(2, 2, 64, 64)

        logits = model(sar=sar, optical=None, dem=dem)
        assert logits.shape == (2, 1, 64, 64)
        assert not torch.isnan(logits).any()

    def test_sar_only_fallback(self):
        """Model operates cleanly when only SAR is available."""
        model = MultimodalUNet(features=[16, 32, 64])
        sar = torch.randn(2, 2, 64, 64)

        logits = model(sar=sar, optical=None, dem=None)
        assert logits.shape == (2, 1, 64, 64)
        assert not torch.isnan(logits).any()

    def test_optical_only_fallback(self):
        """Model operates cleanly when only optical is available."""
        model = MultimodalUNet(features=[16, 32, 64])
        opt = torch.randn(2, 10, 64, 64)

        logits = model(sar=None, optical=opt, dem=None)
        assert logits.shape == (2, 1, 64, 64)
        assert not torch.isnan(logits).any()

    def test_dem_only_fallback(self):
        """Model operates cleanly when only DEM is available."""
        model = MultimodalUNet(features=[16, 32, 64])
        dem = torch.randn(2, 2, 64, 64)

        logits = model(sar=None, optical=None, dem=dem)
        assert logits.shape == (2, 1, 64, 64)
        assert not torch.isnan(logits).any()

    def test_gradient_propagation_active_modalities(self):
        """Gradients propagate through all active encoders and fusion layers."""
        model = MultimodalUNet(features=[16, 32, 64])
        sar = torch.randn(2, 2, 64, 64)
        opt = torch.randn(2, 10, 64, 64)
        dem = torch.randn(2, 2, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()

        logits = model(sar=sar, optical=opt, dem=dem)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        assert model.sar_encoder.inc.double_conv[0].weight.grad is not None
        assert model.optical_encoder.inc.double_conv[0].weight.grad is not None
        assert model.dem_encoder.inc.double_conv[0].weight.grad is not None
        assert model.bottleneck_fusion.fusion_block.double_conv[0].weight.grad is not None

    def test_predict_proba_range(self):
        """predict_proba outputs values strictly within [0, 1]."""
        model = MultimodalUNet(features=[16, 32])
        sar = torch.randn(1, 2, 32, 32)
        dem = torch.randn(1, 2, 32, 32)

        probs = model.predict_proba(sar=sar, optical=None, dem=dem)
        assert probs.shape == (1, 1, 32, 32)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)


class TestMultimodalTrainingAndPersistence:
    """Test training loop integration, checkpoint persistence, and reproducibility."""

    def test_multimodal_reproducibility(self):
        """Identical random seeds yield identical model weights."""
        seed_everything(777)
        m1 = MultimodalUNet(features=[16, 32])
        w1 = m1.sar_encoder.inc.double_conv[0].weight.clone()

        seed_everything(777)
        m2 = MultimodalUNet(features=[16, 32])
        w2 = m2.sar_encoder.inc.double_conv[0].weight.clone()

        assert torch.allclose(w1, w2)

    def test_multimodal_checkpoint_save_and_load(self, tmp_path):
        """Saved multimodal checkpoint accurately restores model weights."""
        model = MultimodalUNet(features=[16, 32])
        trainer = GeoTrustNetTrainer(
            model=model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
            is_multimodal=True,
        )

        dummy_metrics = {"iou": 0.58, "dice": 0.72}
        ckpt_path = trainer.save_checkpoint("multi_ckpt.pt", epoch=1, val_loss=0.22, metrics=dummy_metrics)
        assert ckpt_path.exists()

        restored_model = MultimodalUNet(features=[16, 32])
        restored_trainer = GeoTrustNetTrainer(
            model=restored_model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
            is_multimodal=True,
        )
        loaded = restored_trainer.load_checkpoint(ckpt_path)

        assert loaded["epoch"] == 1
        assert loaded["metrics"]["iou"] == 0.58
        w_orig = model.bottleneck_fusion.fusion_block.double_conv[0].weight
        w_restored = restored_model.bottleneck_fusion.fusion_block.double_conv[0].weight
        assert torch.allclose(w_orig, w_restored)

    def test_multimodal_fit_smoke_test(self, tmp_path):
        """Trainer executes 1 epoch on synthetic multi-modal dataset with is_multimodal=True."""
        manifest_path = Path("data/manifests/synthetic_split_manifest.csv")
        if not manifest_path.exists():
            pytest.skip("Synthetic manifest not found")

        df = pd.read_csv(manifest_path)
        root = Path(".")

        train_ds = GeoTrustNetPatchDataset(df, split="train", data_root=root)
        val_ds = GeoTrustNetPatchDataset(df, split="val", data_root=root)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)

        model = MultimodalUNet(features=[8, 16], bilinear=True)
        trainer = GeoTrustNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path,
            device="cpu",
            modality="multimodal",
            is_multimodal=True,
        )

        summary = trainer.fit(epochs=1)
        assert summary["epochs_trained"] == 1
        assert "best_metric_value" in summary
        assert (tmp_path / "best_model.pt").exists()
        assert (tmp_path / "training_history.json").exists()
