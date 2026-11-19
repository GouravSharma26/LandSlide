"""Unit tests for GeoTrustNet Phase 2 U-Net architecture, losses, metrics, and training engine."""

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

from src.geotrustnet.models.unet import UNet, DoubleConv, Down, Up
from src.geotrustnet.training.losses import DiceLoss, CombinedBCEDiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.training.engine import (
    GeoTrustNetTrainer,
    extract_batch_tensors,
    seed_everything,
)
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches


class TestUNetArchitecture:
    """Test U-Net tensor shapes, gradients, and configuration options."""

    def test_unet_output_shape_bilinear(self):
        """Verify U-Net forward pass produces correct [B, 1, H, W] logits with bilinear upsampling."""
        model = UNet(in_channels=2, out_channels=1, features=[16, 32, 64], bilinear=True)
        x = torch.randn(2, 2, 128, 128)
        logits = model(x)
        assert logits.shape == (2, 1, 128, 128)

    def test_unet_output_shape_transposed_conv(self):
        """Verify U-Net forward pass produces correct [B, 1, H, W] logits with transposed convolutions."""
        model = UNet(in_channels=2, out_channels=1, features=[16, 32, 64], bilinear=False)
        x = torch.randn(3, 2, 64, 64)
        logits = model(x)
        assert logits.shape == (3, 1, 64, 64)

    def test_predict_proba_range(self):
        """Verify predict_proba outputs values strictly in range [0, 1]."""
        model = UNet(in_channels=2, out_channels=1, features=[16, 32], bilinear=True)
        x = torch.randn(1, 2, 64, 64)
        probs = model.predict_proba(x)
        assert probs.shape == (1, 1, 64, 64)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

    def test_gradient_propagation(self):
        """Verify all trainable parameters receive gradients during backward pass."""
        model = UNet(in_channels=2, out_channels=1, features=[16, 32, 64], bilinear=True)
        x = torch.randn(2, 2, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()
        criterion = nn.BCEWithLogitsLoss()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} did not receive a gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"

    def test_parameter_count(self):
        """Verify parameter counting utility returns positive integer."""
        model = UNet(in_channels=2, out_channels=1, features=[32, 64, 128, 256], bilinear=True)
        n_params = model.count_parameters()
        assert isinstance(n_params, int)
        assert n_params > 1_000_000


class TestLossFunctions:
    """Test DiceLoss and CombinedBCEDiceLoss numerical behavior."""

    def test_dice_loss_perfect_prediction(self):
        """When logits strongly predict target, Dice loss should approach 0."""
        loss_fn = DiceLoss()
        target = torch.ones((1, 1, 32, 32))
        logits = torch.ones((1, 1, 32, 32)) * 10.0  # sigmoid(10) ≈ 0.99995
        loss = loss_fn(logits, target)
        assert loss.item() < 0.01

    def test_dice_loss_inverse_prediction(self):
        """When logits predict opposite of target, Dice loss should approach 1."""
        loss_fn = DiceLoss()
        target = torch.ones((1, 1, 32, 32))
        logits = torch.ones((1, 1, 32, 32)) * -10.0  # sigmoid(-10) ≈ 0.00005
        loss = loss_fn(logits, target)
        assert loss.item() > 0.95

    def test_combined_bce_dice_loss_return_dict(self):
        """Verify CombinedBCEDiceLoss returns dictionary of loss components."""
        loss_fn = CombinedBCEDiceLoss(bce_weight=0.6, dice_weight=0.4)
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

        total_loss, metrics_dict = loss_fn(logits, targets, return_dict=True)
        assert isinstance(total_loss, torch.Tensor)
        assert "loss_total" in metrics_dict
        assert "loss_bce" in metrics_dict
        assert "loss_dice" in metrics_dict
        assert metrics_dict["loss_total"] > 0


class TestSegmentationMetrics:
    """Test precision, recall, IoU, Dice, and edge case handling."""

    def test_perfect_metrics(self):
        """When prediction exactly matches target, IoU, Dice, Precision, and Recall are 1.0."""
        target = np.array([[1, 1], [0, 0]])
        probs = np.array([[0.9, 0.8], [0.1, 0.2]])

        m = compute_segmentation_metrics(probs, target, threshold=0.5)
        assert m["iou"] == 1.0
        assert m["dice"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["tn"] == 2
        assert m["fn"] == 0

    def test_empty_mask_handling(self):
        """When ground truth and prediction are both completely empty, IoU and Dice are 1.0."""
        target = np.zeros((10, 10))
        probs = np.zeros((10, 10))

        m = compute_segmentation_metrics(probs, target, threshold=0.5)
        assert m["iou"] == 1.0
        assert m["dice"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["tp"] == 0
        assert m["fp"] == 0

    def test_empty_mask_false_positives(self):
        """When ground truth is empty but model predicts false positives, precision and IoU are 0.0."""
        target = np.zeros((10, 10))
        probs = np.ones((10, 10)) * 0.9

        m = compute_segmentation_metrics(probs, target, threshold=0.5)
        assert m["iou"] == 0.0
        assert m["dice"] == 0.0
        assert m["precision"] == 0.0
        assert m["fp"] == 100

    def test_raw_logits_handling(self):
        """Passing raw logits with negative values automatically invokes sigmoid."""
        target = np.array([[1, 0], [0, 1]])
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])  # Raw logits

        m = compute_segmentation_metrics(logits, target, threshold=0.5)
        assert m["iou"] == 1.0
        assert m["dice"] == 1.0


class TestTrainingEngineAndCheckpoints:
    """Test trainer execution, checkpoint persistence, and reproducibility."""

    def test_seed_everything_reproducibility(self):
        """Models initialized under identical seed produce identical initial weights."""
        seed_everything(123)
        m1 = UNet(in_channels=2, out_channels=1, features=[16, 32])
        w1 = m1.inc.double_conv[0].weight.clone()

        seed_everything(123)
        m2 = UNet(in_channels=2, out_channels=1, features=[16, 32])
        w2 = m2.inc.double_conv[0].weight.clone()

        assert torch.allclose(w1, w2)

    def test_checkpoint_save_and_load(self, tmp_path):
        """Saved checkpoint accurately restores model weights."""
        model = UNet(in_channels=2, out_channels=1, features=[16, 32])
        trainer = GeoTrustNetTrainer(
            model=model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
        )

        dummy_metrics = {"iou": 0.45, "dice": 0.60}
        ckpt_path = trainer.save_checkpoint("test_ckpt.pt", epoch=3, val_loss=0.25, metrics=dummy_metrics)
        assert ckpt_path.exists()

        new_model = UNet(in_channels=2, out_channels=1, features=[16, 32])
        new_trainer = GeoTrustNetTrainer(
            model=new_model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
        )
        loaded = new_trainer.load_checkpoint(ckpt_path)
        assert loaded["epoch"] == 3
        assert loaded["metrics"]["iou"] == 0.45

        # Compare weights
        w_orig = model.inc.double_conv[0].weight
        w_restored = new_model.inc.double_conv[0].weight
        assert torch.allclose(w_orig, w_restored)

    def test_trainer_fit_on_synthetic_fixtures(self, tmp_path):
        """Verify GeoTrustNetTrainer executes 1 epoch on synthetic fixtures without error."""
        manifest_path = Path("data/manifests/synthetic_split_manifest.csv")
        if not manifest_path.exists():
            pytest.skip("Synthetic manifest not generated")

        df = pd.read_csv(manifest_path)
        root = Path(".")

        train_ds = GeoTrustNetPatchDataset(df, split="train", data_root=root)
        val_ds = GeoTrustNetPatchDataset(df, split="val", data_root=root)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)

        model = UNet(in_channels=2, out_channels=1, features=[8, 16], bilinear=True)
        trainer = GeoTrustNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path,
            device="cpu",
            modality="sar",
            temporal_idx=-1,
        )

        summary = trainer.fit(epochs=1)
        assert summary["epochs_trained"] == 1
        assert "best_metric_value" in summary
        assert (tmp_path / "best_model.pt").exists()
        assert (tmp_path / "training_history.json").exists()
