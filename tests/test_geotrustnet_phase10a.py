"""Unit and integration test suite for GeoTrustNet Phase 10A (Model A Baseline Benchmark)."""

import json
import os
import sys
from pathlib import Path
import pytest

# Windows DLL initialization for torch
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.models.unet import UNet
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics


def test_model_a_parameter_count():
    """Verify Model A architecture and exact 1,948,001 parameter count."""
    model = UNet(in_channels=2, out_channels=1, features=[32, 64, 128, 256], bilinear=True, dropout=0.1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 1948001, f"Expected 1,948,001 parameters, got {total_params}"


def test_data_audit_report_schema_and_discrepancy_resolution():
    """Verify data audit report resolves the validation sample discrepancy."""
    audit_file = ROOT_DIR / "artifacts" / "geotrustnet" / "phase10a" / "data_audit_report.json"
    assert audit_file.exists(), f"Audit report missing: {audit_file}"

    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["status"] == "PASSED — DISCREPANCY RESOLVED"
    assert "discrepancy_resolution" in audit
    disc = audit["discrepancy_resolution"]
    assert disc["synthetic_total_pixels"] == 98304
    assert disc["synthetic_pilot_loaded_count"] == 6
    assert disc["real_expected_validation_patches"] == 853
    assert disc["real_val_inventories"]["chimanimani"] == 345
    assert disc["real_val_inventories"]["newzealand"] == 508


def test_sar_normalization_pipeline():
    """Verify SAR z-score normalization in dataset loader."""
    manifest_path = ROOT_DIR / "data" / "manifests" / "synthetic_split_manifest.csv"
    df = pd.read_csv(manifest_path)

    ds_raw = GeoTrustNetPatchDataset(df, split="val", data_root=ROOT_DIR, normalize=False)
    ds_norm = GeoTrustNetPatchDataset(df, split="val", data_root=ROOT_DIR, normalize=True)

    raw_batch = ds_raw[0]
    norm_batch = ds_norm[0]

    sar_raw = raw_batch["sar"]
    sar_norm = norm_batch["sar"]

    assert sar_raw.shape == sar_norm.shape
    # Check that normalization actually modified values
    assert not torch.allclose(sar_raw, sar_norm)
    # Check that normalized VV is centered
    vv_norm = sar_norm[:, 0]
    assert vv_norm.min() < 0 < vv_norm.max()


def test_checkpoints_reload_and_inference():
    """Verify that all Phase 10A checkpoints reload cleanly and produce valid probability masks."""
    ckpt_dir = ROOT_DIR / "artifacts" / "geotrustnet" / "phase10a" / "checkpoints"
    checkpoints = ["best_val_loss.pt", "best_val_dice_at_050.pt", "latest_checkpoint.pt"]

    model = UNet(in_channels=2, out_channels=1, features=[32, 64, 128, 256], bilinear=True, dropout=0.1)

    dummy_input = torch.randn(2, 2, 128, 128)

    for ckpt_name in checkpoints:
        path = ckpt_dir / ckpt_name
        assert path.exists(), f"Checkpoint {ckpt_name} missing"
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "val_loss" in ckpt

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
            probs = torch.sigmoid(logits)
        assert probs.shape == (2, 1, 128, 128)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_training_history_convergence():
    """Verify that training loss decreases across the 20 epochs."""
    history_file = ROOT_DIR / "artifacts" / "geotrustnet" / "phase10a" / "training_history.csv"
    assert history_file.exists()

    df = pd.read_csv(history_file)
    assert len(df) == 20
    first_loss = df["train_loss"].iloc[0]
    final_loss = df["train_loss"].iloc[-1]
    assert final_loss < first_loss, f"Train loss should decrease: {first_loss} -> {final_loss}"
    # Verify val dice reaches high value
    max_val_dice = df["dice"].max()
    assert max_val_dice > 0.85, f"Validation Dice should reach > 0.85, got {max_val_dice}"


def test_threshold_analysis_and_calibration():
    """Verify threshold analysis table and optimal tau* selection."""
    th_file = ROOT_DIR / "artifacts" / "geotrustnet" / "phase10a" / "threshold_analysis.csv"
    assert th_file.exists()

    df = pd.read_csv(th_file)
    assert len(df) >= 12
    assert "threshold" in df.columns
    assert "dice" in df.columns
    assert "precision" in df.columns
    assert "recall" in df.columns

    # Verify best threshold has highest Dice
    best_row = df.sort_values("dice", ascending=False).iloc[0]
    assert best_row["dice"] > 0.90
    assert 0.50 <= best_row["threshold"] <= 0.99


def test_visual_prediction_audits_exist():
    """Verify that all 6 diagnostic 8-panel visual audit panels are generated."""
    vis_dir = ROOT_DIR / "artifacts" / "geotrustnet" / "phase10a" / "validation_predictions"
    expected_cases = [
        "case_01_high_performing.png",
        "case_02_low_performing.png",
        "case_03_small_scar.png",
        "case_04_large_scar.png",
        "case_05_false_positive.png",
        "case_06_false_negative.png",
    ]
    for case_file in expected_cases:
        p = vis_dir / case_file
        assert p.exists() and p.stat().st_size > 10000, f"Visual panel {case_file} missing or incomplete"


def test_phase8_pilot_preservation():
    """Verify that Phase 8 pilot artifacts were not overwritten by Phase 10A."""
    p8_dir = ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "pilot_model_a_sar_unet"
    assert (p8_dir / "best_checkpoint.pt").exists()
    assert (p8_dir / "pilot_summary.json").exists()
    assert (p8_dir / "training_history.csv").exists()


def test_provenance_audit_report_and_classification():
    """Verify Phase 10A.1 provenance audit report and classification."""
    prov_file = ROOT_DIR / "artifacts" / "geotrustnet" / "phase10a" / "provenance_audit_report.json"
    assert prov_file.exists(), f"Provenance audit file missing: {prov_file}"

    with open(prov_file, "r", encoding="utf-8") as f:
        prov = json.load(f)

    assert prov["phase10a_classification"] == "SYNTHETIC PIPELINE BASELINE BENCHMARK (MODEL A)"
    assert prov["actual_loaded_data_provenance"]["train_split"]["sample_count"] == 12
    assert prov["actual_loaded_data_provenance"]["validation_split"]["sample_count"] == 6
    assert prov["real_sen12landslides_status"]["demonstrated_real_evaluation"] is False
    assert len(prov["actual_loaded_data_provenance"]["train_split"]["sample_ids"]) == 12
    assert len(prov["actual_loaded_data_provenance"]["validation_split"]["sample_ids"]) == 6
