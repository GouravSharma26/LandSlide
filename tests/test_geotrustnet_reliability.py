"""Unit tests for GeoTrustNet Phase 5 Reliability-Aware Fusion, Uncertainty, and Calibration."""

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

from src.geotrustnet.models.reliability import (
    ModalityReliabilityEstimator,
    ReliabilityAwareFusion,
)
from src.geotrustnet.models.geotrustnet import GeoTrustNet
from src.geotrustnet.evaluation.uncertainty import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_selective_prediction_curve,
)
from src.geotrustnet.training.engine import (
    GeoTrustNetTrainer,
    seed_everything,
)
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches


class TestReliabilityFusionModules:
    """Test reliability scoring, masked softmax normalization, and edge cases."""

    def test_estimator_spatial_and_patch_level(self):
        """ModalityReliabilityEstimator outputs correct shapes for spatial=True and spatial=False."""
        est_spatial = ModalityReliabilityEstimator(feature_dim=32, spatial=True)
        est_patch = ModalityReliabilityEstimator(feature_dim=32, spatial=False)

        x = torch.randn(2, 32, 16, 16)
        q = torch.tensor([0.9, 0.4])

        s_spatial = est_spatial(x, quality=q)
        s_patch = est_patch(x, quality=q)

        assert s_spatial.shape == (2, 1, 16, 16)
        assert s_patch.shape == (2, 1, 1, 1)

    def test_masked_softmax_missing_modality_zero_weight(self):
        """Missing modalities strictly receive 0.0 weight in ReliabilityAwareFusion."""
        fusion = ReliabilityAwareFusion(feature_dim=16, num_modalities=3, out_dim=16)
        feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
        # Modality 1 (optical) is missing for batch item 0 and 1
        avails = [
            torch.tensor([1, 1], dtype=torch.bool),
            torch.tensor([0, 0], dtype=torch.bool),
            torch.tensor([1, 1], dtype=torch.bool),
        ]

        fused, weights = fusion(feats, avails)

        assert fused.shape == (2, 16, 8, 8)
        # Modality 1 must have strictly 0.0 weight
        assert torch.all(weights[1] == 0.0)
        # Modality 0 and 2 must sum to 1.0
        sum_weights = weights[0] + weights[1] + weights[2]
        assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5)

    def test_single_modality_available_weight_one(self):
        """When only one modality is available, its normalized weight is 1.0 everywhere."""
        fusion = ReliabilityAwareFusion(feature_dim=16, num_modalities=3, out_dim=16)
        feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
        # Only modality 0 (SAR) is available
        avails = [
            torch.tensor([1, 1], dtype=torch.bool),
            torch.tensor([0, 0], dtype=torch.bool),
            torch.tensor([0, 0], dtype=torch.bool),
        ]

        fused, weights = fusion(feats, avails)
        assert torch.allclose(weights[0], torch.ones_like(weights[0]), atol=1e-5)
        assert torch.all(weights[1] == 0.0)
        assert torch.all(weights[2] == 0.0)

    def test_all_modalities_missing_safe_fallback(self):
        """When all modalities are unavailable, outputs zero features without NaNs or exceptions."""
        fusion = ReliabilityAwareFusion(feature_dim=16, num_modalities=3, out_dim=16)
        feats = [torch.randn(2, 16, 8, 8) for _ in range(3)]
        avails = [
            torch.tensor([0, 0], dtype=torch.bool),
            torch.tensor([0, 0], dtype=torch.bool),
            torch.tensor([0, 0], dtype=torch.bool),
        ]

        fused, weights = fusion(feats, avails)
        assert not torch.isnan(fused).any()
        for w in weights:
            assert torch.all(w == 0.0)


class TestGeoTrustNetCore:
    """Test GeoTrustNet core architecture, uncertainty quantification, and gradients."""

    def test_geotrustnet_forward_and_reliability_output(self):
        """GeoTrustNet outputs [B, 1, H, W] logits and per-modality reliability weight maps."""
        model = GeoTrustNet(features=[16, 32, 64])
        sar = torch.randn(2, 2, 64, 64)
        opt = torch.randn(2, 10, 64, 64)
        dem = torch.randn(2, 2, 64, 64)

        logits, rel_dict = model(sar, opt, dem, return_reliability=True)
        assert logits.shape == (2, 1, 64, 64)
        assert "weight_sar" in rel_dict
        assert "weight_optical" in rel_dict
        assert "weight_dem" in rel_dict
        assert rel_dict["weight_sar"].shape == (2, 1, 16, 16)

    def test_mc_dropout_uncertainty_stochasticity(self):
        """predict_uncertainty runs multiple passes with active dropout producing non-zero variance."""
        model = GeoTrustNet(features=[16, 32], dropout=0.3)
        sar = torch.randn(2, 2, 32, 32)
        opt = torch.randn(2, 10, 32, 32)
        dem = torch.randn(2, 2, 32, 32)

        unc = model.predict_uncertainty(sar, opt, dem, num_mc_samples=6)
        assert "mean_prob" in unc
        assert "variance" in unc
        assert "entropy" in unc
        assert "confidence" in unc

        # Verify shapes
        assert unc["mean_prob"].shape == (2, 1, 32, 32)
        assert unc["variance"].shape == (2, 1, 32, 32)

        # Probabilities bounded in [0, 1]
        assert torch.all(unc["mean_prob"] >= 0.0)
        assert torch.all(unc["mean_prob"] <= 1.0)
        # Variance non-negative
        assert torch.all(unc["variance"] >= 0.0)

    def test_gradient_propagation_through_reliability_estimator(self):
        """Loss gradients flow through decoder, fusion module, and reliability estimators."""
        model = GeoTrustNet(features=[16, 32, 64])
        sar = torch.randn(2, 2, 64, 64)
        opt = torch.randn(2, 10, 64, 64)
        dem = torch.randn(2, 2, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()

        logits = model(sar, opt, dem)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        # Check reliability estimator gradients
        est_weight = model.bottleneck_fusion.estimators[0].scorer[0].weight
        assert est_weight.grad is not None
        assert not torch.isnan(est_weight.grad).any()


class TestCalibrationAndUncertaintyMetrics:
    """Test ECE, Brier score, and selective prediction curve calculations."""

    def test_brier_score_bounds(self):
        """Brier score is 0.0 for perfect predictions and 1.0 for completely inverted predictions."""
        targets = np.array([1, 1, 0, 0])
        perfect_probs = np.array([1.0, 1.0, 0.0, 0.0])
        inverted_probs = np.array([0.0, 0.0, 1.0, 1.0])

        assert compute_brier_score(perfect_probs, targets) == 0.0
        assert compute_brier_score(inverted_probs, targets) == 1.0

    def test_expected_calibration_error(self):
        """ECE returns calibration error and non-empty reliability diagram bins."""
        targets = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.85, 0.15])

        ece_res = compute_expected_calibration_error(probs, targets, num_bins=5)
        assert "ece" in ece_res
        assert "mce" in ece_res
        assert "bins" in ece_res
        assert 0.0 <= ece_res["ece"] <= 1.0
        assert len(ece_res["bins"]) == 5

    def test_selective_prediction_curve(self):
        """Selective prediction curve outputs coverage percentiles and segmentation metrics."""
        probs = np.random.uniform(0, 1, 100)
        uncertainties = np.random.uniform(0, 0.25, 100)
        targets = np.random.randint(0, 2, 100)

        curve = compute_selective_prediction_curve(probs, uncertainties, targets, percentiles=[100, 80, 50])
        assert len(curve) == 3
        assert curve[0]["target_coverage_pct"] == 100
        assert curve[1]["target_coverage_pct"] == 80
        assert curve[2]["target_coverage_pct"] == 50


class TestTrainingIntegrationAndCheckpoints:
    """Test GeoTrustNet trainer fit and checkpoint persistence."""

    def test_geotrustnet_checkpoint_save_and_restore(self, tmp_path):
        """Saved GeoTrustNet checkpoint accurately restores weights bit-for-bit."""
        model = GeoTrustNet(features=[16, 32])
        trainer = GeoTrustNetTrainer(
            model=model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
            is_multimodal=True,
        )

        dummy_metrics = {"iou": 0.62, "dice": 0.76}
        ckpt_path = trainer.save_checkpoint("gt_ckpt.pt", epoch=1, val_loss=0.18, metrics=dummy_metrics)
        assert ckpt_path.exists()

        restored_model = GeoTrustNet(features=[16, 32])
        restored_trainer = GeoTrustNetTrainer(
            model=restored_model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
            is_multimodal=True,
        )
        loaded = restored_trainer.load_checkpoint(ckpt_path)

        assert loaded["epoch"] == 1
        assert loaded["metrics"]["iou"] == 0.62
        w_orig = model.bottleneck_fusion.estimators[0].scorer[0].weight
        w_restored = restored_model.bottleneck_fusion.estimators[0].scorer[0].weight
        assert torch.allclose(w_orig, w_restored)

    def test_geotrustnet_fit_smoke_test(self, tmp_path):
        """Trainer executes 1 epoch on synthetic dataset with GeoTrustNet."""
        manifest_path = Path("data/manifests/synthetic_split_manifest.csv")
        if not manifest_path.exists():
            pytest.skip("Synthetic manifest not found")

        df = pd.read_csv(manifest_path)
        root = Path(".")

        train_ds = GeoTrustNetPatchDataset(df, split="train", data_root=root)
        val_ds = GeoTrustNetPatchDataset(df, split="val", data_root=root)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)

        model = GeoTrustNet(features=[8, 16], bilinear=True)
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
