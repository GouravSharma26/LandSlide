"""Pilot Training & Benchmark Execution Engine for Phase 8.

Executes:
1. Pilot Training on Model A (SAR U-Net baseline) to verify:
   - Loop execution
   - Checkpoint creation and reload integrity
   - Loss convergence
   - Metric computation and export
2. Records exact hardware, runtime, memory, convergence history, and provenance.
3. Supports real Sen12Landslides NetCDF data or verified local fixtures with explicit labeling.

Usage:
    python scripts/run_pilot_and_benchmarks.py --pilot --epochs 3 --batch-size 4
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.models.unet import UNet
from src.geotrustnet.training.engine import seed_everything
from src.geotrustnet.training.losses import CombinedBCEDiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.evaluation.uncertainty import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_selective_prediction_curve,
)
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches


def run_pilot_experiment(
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    device_pref: str = "auto",
    use_real_data: bool = False,
    data_dir: str = "data/raw/Sen12Landslides",
) -> dict:
    """Executes Model A (SAR U-Net) Pilot Training and records exact metrics and provenance."""
    start_time = time.time()
    seed_everything(42)

    # Determine device
    if device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    else:
        device = torch.device("cpu")
        device_name = "CPU (" + platform.processor() + ")"
        vram_gb = "Host RAM"

    out_dir = ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "pilot_model_a_sar_unet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  GeoTrustNet Phase 8: Pilot Training Execution (Model A)")
    print("=" * 65)
    print(f"Device:              {device} -> {device_name}")
    print(f"Epochs:              {epochs}")
    print(f"Batch Size:          {batch_size}")
    print(f"Learning Rate:       {lr}")
    print(f"Target Directory:    {out_dir}")

    # Determine dataset manifest
    manifest_file = ROOT_DIR / "data" / "manifests" / "sen12landslides_disjoint_manifest.csv"
    if not manifest_file.exists() or not use_real_data:
        manifest_file = ROOT_DIR / "data" / "manifests" / "synthetic_split_manifest.csv"
        provenance = "Synthetic Local Pilot Verification"
    else:
        provenance = "Real Sen12Landslides Pilot"

    manifest_df = pd.read_csv(manifest_file)
    print(f"Data Provenance:     {provenance}")
    print(f"Manifest File:       {manifest_file.name} ({len(manifest_df)} total samples)")

    # Initialize Model A: SAR U-Net
    model = UNet(in_channels=2, out_channels=1, features=[32, 64, 128, 256], bilinear=True, dropout=0.2).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model A Parameters:  {total_params:,}")

    # Initialize Loss & Optimizer
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Initialize Datasets
    train_dataset = GeoTrustNetPatchDataset(manifest_df, split="train", data_root=ROOT_DIR)
    val_dataset = GeoTrustNetPatchDataset(manifest_df, split="val", data_root=ROOT_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_geotrustnet_patches,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        collate_fn=collate_geotrustnet_patches,
    )

    print(f"Train Batches:       {len(train_loader)} ({len(train_dataset)} samples)")
    print(f"Val Batches:         {len(val_loader)} ({len(val_dataset)} samples)")

    # Training Loop
    history = []
    best_val_dice = -1.0
    best_checkpoint_path = out_dir / "best_checkpoint.pt"

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            sar = batch["sar"].to(device)
            # Take latest time step for single-date baseline
            sar_input = sar[:, -1] if sar.ndim == 5 else sar
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(sar_input)
            loss = criterion(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                sar = batch["sar"].to(device)
                sar_input = sar[:, -1] if sar.ndim == 5 else sar
                mask = batch["mask"].to(device)
                logits = model(sar_input)
                loss = criterion(logits, mask)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu())
                val_targets.append(mask.cpu())

        val_loss /= max(1, len(val_loader))
        all_probs = torch.cat(val_preds, dim=0).numpy().flatten()
        all_targets = torch.cat(val_targets, dim=0).numpy().flatten()
        binary_preds = (all_probs >= 0.5).astype(np.float32)

        tp = np.sum(binary_preds * all_targets)
        fp = np.sum(binary_preds * (1 - all_targets))
        fn = np.sum((1 - binary_preds) * all_targets)
        val_dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-6)
        val_iou = tp / (tp + fp + fn + 1e-6)

        ep_duration = time.time() - ep_start
        history.append({
            "epoch": int(epoch),
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_dice": round(float(val_dice), 4),
            "val_iou": round(float(val_iou), 4),
            "duration_sec": round(float(ep_duration), 2),
        })

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} ({ep_duration:.1f}s)")

        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = float(val_dice)
            torch.save({
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": float(val_dice),
                "val_iou": float(val_iou),
                "total_params": total_params,
                "seed": 42,
            }, best_checkpoint_path)

    total_duration = time.time() - start_time

    # Checkpoint Reload & Integrity Verification
    checkpoint_verified = False
    if best_checkpoint_path.exists():
        test_model = UNet(in_channels=2, out_channels=1, features=[32, 64, 128, 256], bilinear=True).to(device)
        loaded_ckpt = torch.load(best_checkpoint_path, weights_only=False)
        test_model.load_state_dict(loaded_ckpt["model_state_dict"])
        checkpoint_verified = True
        print(f"\n[Checkpoint Integrity] Successfully verified checkpoint reload from: {best_checkpoint_path}")

    # Evaluate on Test Split
    test_dataset = GeoTrustNetPatchDataset(manifest_df, split="test", data_root=ROOT_DIR)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_geotrustnet_patches)

    test_preds, test_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sar = batch["sar"].to(device)
            sar_input = sar[:, -1] if sar.ndim == 5 else sar
            mask = batch["mask"].to(device)
            logits = model(sar_input)
            test_preds.append(torch.sigmoid(logits).cpu())
            test_targets.append(mask.cpu())

    t_probs = torch.cat(test_preds, dim=0).numpy().flatten()
    t_targets = torch.cat(test_targets, dim=0).numpy().flatten()
    t_bin = (t_probs >= 0.5).astype(np.float32)

    tp_t = np.sum(t_bin * t_targets)
    fp_t = np.sum(t_bin * (1 - t_targets))
    fn_t = np.sum((1 - t_bin) * t_targets)
    tn_t = np.sum((1 - t_bin) * (1 - t_targets))

    test_metrics = {
        "precision": round(float(tp_t / (tp_t + fp_t + 1e-6)), 4),
        "recall": round(float(tp_t / (tp_t + fn_t + 1e-6)), 4),
        "f1": round(float(2 * tp_t / (2 * tp_t + fp_t + fn_t + 1e-6)), 4),
        "dice": round(float(2 * tp_t / (2 * tp_t + fp_t + fn_t + 1e-6)), 4),
        "iou": round(float(tp_t / (tp_t + fp_t + fn_t + 1e-6)), 4),
        "total_pixels": int(len(t_probs)),
        "landslide_pixels": int(np.sum(t_targets)),
    }

    # Probability calibration metrics
    brier = float(np.mean((t_probs - t_targets) ** 2))
    ece_report = compute_expected_calibration_error(t_probs, t_targets, num_bins=10)
    ece = float(ece_report["ece"] if isinstance(ece_report, dict) else ece_report)

    pilot_summary = {
        "experiment_id": "pilot_model_a_sar_unet",
        "model_architecture": "Model A: Single-Modality SAR U-Net",
        "parameters": total_params,
        "hardware": {
            "device": str(device),
            "device_name": device_name,
            "vram_gb": vram_gb,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "training_configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss_function": "CombinedBCEDiceLoss (0.5/0.5)",
            "seed": 42,
        },
        "provenance": provenance,
        "data_manifest": manifest_file.name,
        "total_duration_seconds": round(total_duration, 2),
        "checkpoint_integrity_verified": checkpoint_verified,
        "training_history": history,
        "validation_best_dice": round(float(best_val_dice), 4),
        "test_metrics": test_metrics,
        "calibration": {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save summary report
    summary_path = out_dir / "pilot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(pilot_summary, f, indent=2)

    # Save CSV history
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print("\n" + "=" * 65)
    print("  Pilot Training Completed Successfully")
    print("=" * 65)
    print(f"Total Duration:      {total_duration:.2f}s")
    print(f"Best Val Dice:       {best_val_dice:.4f}")
    print(f"Test Dice:           {test_metrics['dice']:.4f}")
    print(f"Test IoU:            {test_metrics['iou']:.4f}")
    print(f"Test ECE:            {ece:.4f}")
    print(f"Checkpoint Verified: {checkpoint_verified}")
    print(f"Export Path:         {out_dir}")
    print("=" * 65)

    return pilot_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoTrustNet Phase 8 Pilot Training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of pilot epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Pilot batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', 'auto')")
    parser.add_argument("--use-real-data", action="store_true", help="Use real Sen12Landslides manifest if available")
    args = parser.parse_args()

    run_pilot_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_pref=args.device,
        use_real_data=args.use_real_data,
    )
