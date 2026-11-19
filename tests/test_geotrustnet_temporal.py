"""Unit tests for GeoTrustNet Phase 3 ConvLSTM, Spatio-Temporal SAR U-Net, and masking."""

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

from src.geotrustnet.models.convlstm import ConvLSTM, ConvLSTMCell
from src.geotrustnet.models.temporal_unet import TemporalSARUNet
from src.geotrustnet.training.engine import (
    GeoTrustNetTrainer,
    extract_batch_tensors,
    seed_everything,
)
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches


class TestConvLSTMCells:
    """Test ConvLSTMCell and multi-layer ConvLSTM dynamics and state handling."""

    def test_convlstm_cell_initialization(self):
        """Cell forward pass with state=None automatically initializes zero tensors."""
        cell = ConvLSTMCell(in_channels=8, hidden_channels=16, kernel_size=3)
        x = torch.randn(2, 8, 32, 32)
        h, c = cell(x, state=None)

        assert h.shape == (2, 16, 32, 32)
        assert c.shape == (2, 16, 32, 32)
        assert not torch.isnan(h).any()

    def test_convlstm_cell_masking_freeze(self):
        """When valid_mask is 0 (missing acquisition), hidden and cell states are frozen unchanged."""
        cell = ConvLSTMCell(in_channels=4, hidden_channels=8, kernel_size=3)
        x = torch.randn(2, 4, 16, 16)
        h_init = torch.randn(2, 8, 16, 16)
        c_init = torch.randn(2, 8, 16, 16)

        # Batch item 0 is valid (1), batch item 1 is invalid/missing (0)
        mask = torch.tensor([1, 0], dtype=torch.bool)
        h_next, c_next = cell(x, state=(h_init, c_init), valid_mask=mask)

        # Item 0 must be updated
        assert not torch.allclose(h_next[0], h_init[0])
        assert not torch.allclose(c_next[0], c_init[0])

        # Item 1 must be perfectly preserved
        assert torch.allclose(h_next[1], h_init[1])
        assert torch.allclose(c_next[1], c_init[1])

    def test_multilayer_convlstm_forward(self):
        """Multi-layer ConvLSTM processes 5D sequence [B, T, C, H, W] and returns layer states."""
        model = ConvLSTM(in_channels=4, hidden_channels=[8, 16], num_layers=2)
        x = torch.randn(2, 5, 4, 16, 16)
        out, states = model(x)

        assert out.shape == (2, 5, 16, 16, 16)
        assert len(states) == 2
        assert states[0][0].shape == (2, 8, 16, 16)
        assert states[1][0].shape == (2, 16, 16, 16)


class TestTemporalSARUNet:
    """Test Spatio-Temporal SAR U-Net forward shapes, backprop, and temporal aggregations."""

    def test_temporal_sar_unet_shape_various_timesteps(self):
        """Model outputs [B, 1, H, W] for variable sequence lengths T=1, T=3, T=5."""
        model = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32, 64])

        for t in [1, 3, 5]:
            x = torch.randn(2, t, 2, 64, 64)
            logits = model(x)
            assert logits.shape == (2, 1, 64, 64), f"Failed for T={t}"

    def test_temporal_sar_unet_4d_fallback(self):
        """When 4D tensor [B, C, H, W] is passed, model unsqueezes to T=1 and executes."""
        model = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32, 64])
        x = torch.randn(2, 2, 64, 64)
        logits = model(x)
        assert logits.shape == (2, 1, 64, 64)

    def test_temporal_sar_unet_gradient_bptt(self):
        """Gradients propagate through time through both CNN spatial stem and ConvLSTM temporal gates."""
        model = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32, 64])
        x = torch.randn(2, 4, 2, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()

        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        # Check CNN encoder gradient
        assert model.inc.double_conv[0].weight.grad is not None
        assert not torch.isnan(model.inc.double_conv[0].weight.grad).any()

        # Check ConvLSTM gate gradient
        convlstm_weight = model.convlstm.cell_list[0].conv.weight
        assert convlstm_weight.grad is not None
        assert not torch.isnan(convlstm_weight.grad).any()

    def test_temporal_aggregation_strategies(self):
        """Verify 'last', 'mean', and 'attention' aggregation modes execute without error."""
        for agg in ["last", "mean", "attention"]:
            model = TemporalSARUNet(
                in_channels=2,
                out_channels=1,
                features=[16, 32],
                aggregation=agg,
            )
            x = torch.randn(2, 3, 2, 32, 32)
            mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
            logits = model(x, valid_masks=mask)
            assert logits.shape == (2, 1, 32, 32), f"Failed for aggregation '{agg}'"

    def test_spatial_5d_mask_handling(self):
        """Model accepts 5D spatial masks [B, T, 1, H, W] and handles downsampling to bottleneck."""
        model = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32, 64])
        x = torch.randn(2, 4, 2, 64, 64)
        mask_5d = torch.randint(0, 2, (2, 4, 1, 64, 64)).bool()

        logits = model(x, valid_masks=mask_5d)
        assert logits.shape == (2, 1, 64, 64)


class TestTemporalTrainingAndPersistence:
    """Test training loop execution, checkpointing, and reproducibility for Phase 3."""

    def test_temporal_reproducibility(self):
        """Identical random seeds yield identical weights and identical predictions."""
        seed_everything(999)
        m1 = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32])
        w1 = m1.convlstm.cell_list[0].conv.weight.clone()

        seed_everything(999)
        m2 = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32])
        w2 = m2.convlstm.cell_list[0].conv.weight.clone()

        assert torch.allclose(w1, w2)

    def test_temporal_checkpoint_save_and_load(self, tmp_path):
        """Saved temporal model checkpoint restores weights accurately."""
        model = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32])
        trainer = GeoTrustNetTrainer(
            model=model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
            is_temporal=True,
        )

        dummy_metrics = {"iou": 0.52, "dice": 0.68}
        ckpt_path = trainer.save_checkpoint("temp_ckpt.pt", epoch=2, val_loss=0.31, metrics=dummy_metrics)
        assert ckpt_path.exists()

        restored_model = TemporalSARUNet(in_channels=2, out_channels=1, features=[16, 32])
        restored_trainer = GeoTrustNetTrainer(
            model=restored_model,
            train_loader=None,
            val_loader=None,
            output_dir=tmp_path,
            is_temporal=True,
        )
        loaded = restored_trainer.load_checkpoint(ckpt_path)

        assert loaded["epoch"] == 2
        assert loaded["metrics"]["iou"] == 0.52
        w_orig = model.convlstm.cell_list[0].conv.weight
        w_restored = restored_model.convlstm.cell_list[0].conv.weight
        assert torch.allclose(w_orig, w_restored)

    def test_temporal_fit_smoke_test(self, tmp_path):
        """Trainer executes 1 epoch on synthetic multi-temporal dataset with is_temporal=True."""
        manifest_path = Path("data/manifests/synthetic_split_manifest.csv")
        if not manifest_path.exists():
            pytest.skip("Synthetic manifest not found")

        df = pd.read_csv(manifest_path)
        root = Path(".")

        train_ds = GeoTrustNetPatchDataset(df, split="train", data_root=root, fixed_time_steps=3)
        val_ds = GeoTrustNetPatchDataset(df, split="val", data_root=root, fixed_time_steps=3)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_geotrustnet_patches)

        model = TemporalSARUNet(in_channels=2, out_channels=1, features=[8, 16], bilinear=True)
        trainer = GeoTrustNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=tmp_path,
            device="cpu",
            modality="sar",
            temporal_idx=None,
            is_temporal=True,
        )

        summary = trainer.fit(epochs=1)
        assert summary["epochs_trained"] == 1
        assert "best_metric_value" in summary
        assert (tmp_path / "best_model.pt").exists()
        assert (tmp_path / "training_history.json").exists()
