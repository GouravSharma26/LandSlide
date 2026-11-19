"""GeoTrustNet Phase 10A: Full-Scale Model A (SAR U-Net) Baseline Benchmark.

Executes:
1. End-to-end controlled 20-epoch training of Model A (SAR U-Net) on Sentinel-1 SAR (VV + VH).
2. Uses official standardized z-score input normalization.
3. Strict evaluation on full Validation split after every epoch at threshold 0.50.
4. Full checkpoint management:
   - best_val_loss.pt
   - best_val_dice_at_050.pt
   - latest_checkpoint.pt
5. Full post-training Validation Threshold Calibration on the validation split.
6. Inventory-level generalization breakdown (macro-average, median, worst, best).
7. Visual prediction audits (8-panel panels covering high/low performance, FP/FN, small/large scars).
8. Exports complete metrics, curves, and artifacts to artifacts/geotrustnet/phase10a/.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

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
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.models.unet import UNet
from src.geotrustnet.training.engine import seed_everything
from src.geotrustnet.training.losses import CombinedBCEDiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches


def get_file_sha256(filepath: Path) -> str:
    """Calculate sha256 checksum of file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_phase10a_benchmark(
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_pref: str = "auto",
    use_real_manifest: bool = False,
    data_manifest_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete Phase 10A benchmark for Model A (SAR U-Net)."""
    start_time = time.time()
    seed_everything(42)

    # Output directory
    out_dir = Path(output_dir) if output_dir else ROOT_DIR / "artifacts" / "geotrustnet" / "phase10a"
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "validation_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration
    if device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        vram_info = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM"
    else:
        device = torch.device("cpu")
        device_name = f"CPU ({platform.processor()})"
        vram_info = "Host System RAM"

    print("=" * 72)
    print("  GeoTrustNet Phase 10A: Model A (SAR U-Net) Baseline Benchmark")
    print("=" * 72)
    print(f"Device:             {device} ({device_name}, {vram_info})")
    print(f"Epochs:             {epochs}")
    print(f"Batch Size:         {batch_size}")
    print(f"Learning Rate:      {lr}")
    print(f"Weight Decay:       {weight_decay}")
    print(f"Output Directory:   {out_dir}")

    # Dataset manifest resolution
    if data_manifest_path:
        manifest_file = ROOT_DIR / data_manifest_path
    elif use_real_manifest:
        manifest_file = ROOT_DIR / "data" / "manifests" / "sen12landslides_disjoint_manifest.csv"
    else:
        # Default verified local manifest
        manifest_file = ROOT_DIR / "data" / "manifests" / "synthetic_split_manifest.csv"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    manifest_hash = get_file_sha256(manifest_file)
    manifest_df = pd.read_csv(manifest_file)
    provenance = "Sen12Landslides Disjoint Manifest" if "chimanimani" in str(manifest_df["region_id"].values) else "Verified Diagnostic Fixtures"

    print(f"Manifest File:      {manifest_file.name} ({len(manifest_df)} total samples)")
    print(f"Manifest Hash:      {manifest_hash[:16]}...")
    print(f"Data Provenance:    {provenance}")

    # Datasets and Loaders (with explicit Z-Score normalization)
    train_dataset = GeoTrustNetPatchDataset(manifest_df, split="train", data_root=ROOT_DIR, normalize=True)
    val_dataset = GeoTrustNetPatchDataset(manifest_df, split="val", data_root=ROOT_DIR, normalize=True)

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

    print(f"Train Samples:      {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Val Samples:        {len(val_dataset)} ({len(val_loader)} batches)")

    # Model A: SAR U-Net
    model = UNet(
        in_channels=2,  # SAR VV, VH
        out_channels=1,
        features=[32, 64, 128, 256],
        bilinear=True,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model A Parameters: {total_params:,} (1.95M parameter architecture)")

    # Loss, Optimizer, Scheduler
    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-6)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Save Experiment Config
    exp_config = {
        "experiment_id": "phase10a_model_a_sar_unet",
        "phase": "10A",
        "model_name": "Model A (SAR U-Net)",
        "in_channels": 2,
        "out_channels": 1,
        "features": [32, 64, 128, 256],
        "bilinear": True,
        "dropout": 0.1,
        "trainable_parameters": total_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "loss": "CombinedBCEDiceLoss(bce=0.5, dice=0.5)",
        "scheduler": "CosineAnnealingLR(T_max=20, eta_min=1e-6)",
        "seed": 42,
        "device": str(device),
        "device_name": device_name,
        "manifest_path": str(manifest_file).replace("\\", "/"),
        "manifest_sha256": manifest_hash,
        "normalization": {
            "type": "z-score standardized",
            "vv_mean": -9.3099,
            "vv_std": 3.4868,
            "vh_mean": -15.8861,
            "vh_std": 4.1814,
        },
        "disjoint_split_policy": "Strict geographic isolation; train and val share 0 inventories; test held out",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # Training Loop
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_dice_at_050 = -1.0

    print("-" * 72)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Dice@0.5':<10} | {'IoU@0.5':<10} | {'Prec':<8} | {'Rec':<8} | {'Time':<6}")
    print("-" * 72)

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
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Validation Pass (Fixed threshold 0.50)
        model.eval()
        val_loss = 0.0
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                sar = batch["sar"].to(device)
                sar_input = sar[:, -1] if sar.ndim == 5 else sar
                mask = batch["mask"].to(device)

                logits = model(sar_input)
                loss = criterion(logits, mask)
                val_loss += loss.item()

                val_logits_list.append(logits.detach().cpu())
                val_targets_list.append(mask.detach().cpu())

        val_loss /= max(1, len(val_loader))
        concat_logits = torch.cat(val_logits_list, dim=0)
        concat_targets = torch.cat(val_targets_list, dim=0)

        metrics_at_050 = compute_segmentation_metrics(
            logits_or_probs=concat_logits,
            targets=concat_targets,
            threshold=0.50,
            compute_curve_auc=True,
        )

        ep_duration = time.time() - ep_start
        gpu_mem = f"{torch.cuda.memory_allocated(0)/(1024**2):.1f}MB" if torch.cuda.is_available() else "HostRAM"

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "dice": metrics_at_050["dice"],
            "iou": metrics_at_050["iou"],
            "precision": metrics_at_050["precision"],
            "recall": metrics_at_050["recall"],
            "specificity": metrics_at_050["specificity"],
            "tp": metrics_at_050["tp"],
            "fp": metrics_at_050["fp"],
            "tn": metrics_at_050["tn"],
            "fn": metrics_at_050["fn"],
            "ground_truth_prevalence": metrics_at_050["positive_prevalence"],
            "predicted_prevalence": metrics_at_050["predicted_prevalence"],
            "pr_auc": metrics_at_050["pr_auc"],
            "roc_auc": metrics_at_050["roc_auc"],
            "learning_rate": round(current_lr, 7),
            "epoch_duration_sec": round(ep_duration, 2),
            "gpu_memory": gpu_mem,
        }
        history.append(row)

        print(
            f"{epoch:02d}/{epochs:02d}  | {train_loss:<10.4f} | {val_loss:<10.4f} | "
            f"{metrics_at_050['dice']:<10.4f} | {metrics_at_050['iou']:<10.4f} | "
            f"{metrics_at_050['precision']:<8.4f} | {metrics_at_050['recall']:<8.4f} | {ep_duration:<5.1f}s"
        )

        # Checkpoint Policy: Save latest
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "val_loss": val_loss,
            "metrics": metrics_at_050,
            "config": exp_config,
        }
        torch.save(latest_ckpt, ckpt_dir / "latest_checkpoint.pt")

        # Save Best Validation Loss Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(latest_ckpt, ckpt_dir / "best_val_loss.pt")

        # Save Best Validation Dice Checkpoint (at threshold 0.50)
        if metrics_at_050["dice"] > best_val_dice_at_050:
            best_val_dice_at_050 = metrics_at_050["dice"]
            torch.save(latest_ckpt, ckpt_dir / "best_val_dice_at_050.pt")

    # Export training history CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)
    print(f"\nTraining complete. History exported to {out_dir / 'training_history.csv'}")

    # Plot Training & Validation Loss/Dice Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161a23")
        ax.grid(True, linestyle="--", alpha=0.3, color="#888888")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    epochs_arr = history_df["epoch"].values
    ax1.plot(epochs_arr, history_df["train_loss"].values, label="Train Loss", color="#00bcd4", lw=2)
    ax1.plot(epochs_arr, history_df["val_loss"].values, label="Val Loss", color="#ff7043", lw=2)
    ax1.set_title("Model A: Combined BCE+Dice Loss Convergence", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss", color="white")
    ax1.legend(facecolor="#161a23", edgecolor="#444444", labelcolor="white")

    ax2.plot(epochs_arr, history_df["dice"].values, label="Val Dice @ 0.50", color="#4caf50", lw=2)
    ax2.plot(epochs_arr, history_df["iou"].values, label="Val IoU @ 0.50", color="#ab47bc", lw=2)
    ax2.plot(epochs_arr, history_df["precision"].values, label="Val Precision", color="#29b6f6", ls="--")
    ax2.plot(epochs_arr, history_df["recall"].values, label="Val Recall", color="#ffa726", ls="--")
    ax2.set_title("Model A: Validation Segmentation Metrics (@ 0.50)", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Metric Score", color="white")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(facecolor="#161a23", edgecolor="#444444", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # =========================================================================
    # STAGE 5: Validation Threshold Calibration (FULL Validation Split Only)
    # =========================================================================
    print("\n" + "=" * 72)
    print("  Stage 5: Validation Threshold Calibration (Full Validation Split)")
    print("=" * 72)

    # Load best validation model
    best_ckpt = torch.load(ckpt_dir / "best_val_loss.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_probs_all = []
    val_targets_all = []
    val_regions_all = []
    val_sample_ids_all = []
    val_raw_inputs_all = []

    with torch.no_grad():
        for batch in val_loader:
            sar = batch["sar"].to(device)
            sar_input = sar[:, -1] if sar.ndim == 5 else sar
            logits = model(sar_input)
            probs = torch.sigmoid(logits)

            val_probs_all.append(probs.cpu())
            val_targets_all.append(batch["mask"].cpu())
            val_raw_inputs_all.append(sar_input.cpu())
            val_regions_all.extend(batch.get("region_ids", batch.get("region_id", [])))
            val_sample_ids_all.extend(batch.get("sample_ids", batch.get("sample_id", [])))

    val_probs_tensor = torch.cat(val_probs_all, dim=0)    # [N, 1, H, W]
    val_targets_tensor = torch.cat(val_targets_all, dim=0)  # [N, 1, H, W]
    val_raw_tensor = torch.cat(val_raw_inputs_all, dim=0)  # [N, 2, H, W]

    flat_probs = val_probs_tensor.numpy().flatten()
    flat_targets = val_targets_tensor.numpy().flatten().astype(np.int64)

    # Documented threshold grid
    th_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.99]

    th_records = []
    best_th_dice = -1.0
    optimal_th = 0.50

    for th in th_grid:
        m = compute_segmentation_metrics(
            logits_or_probs=flat_probs,
            targets=flat_targets,
            threshold=th,
            compute_curve_auc=False,
        )
        rec = {
            "threshold": th,
            "dice": m["dice"],
            "iou": m["iou"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "predicted_prevalence": m["predicted_prevalence"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
        }
        th_records.append(rec)
        if m["dice"] > best_th_dice:
            best_th_dice = m["dice"]
            optimal_th = th

    th_df = pd.DataFrame(th_records)
    th_df.to_csv(out_dir / "threshold_analysis.csv", index=False)
    print(f"Optimal Validation Operating Threshold (tau*): {optimal_th:.3f} (Dice: {best_th_dice:.4f})")
    print(f"Threshold analysis saved to: {out_dir / 'threshold_analysis.csv'}")

    # Plot Precision-Recall & Threshold Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161a23")
        ax.grid(True, linestyle="--", alpha=0.3, color="#888888")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    # PR Curve
    prec_pts, rec_pts, _ = precision_recall_curve(flat_targets, flat_probs)
    pr_auc_val = auc(rec_pts, prec_pts)
    ax1.plot(rec_pts, prec_pts, color="#ff4081", lw=2.5, label=f"PR Curve (AUC = {pr_auc_val:.4f})")
    ax1.set_title("Model A: Validation Precision-Recall Curve", color="white", fontsize=12)
    ax1.set_xlabel("Recall", color="white")
    ax1.set_ylabel("Precision", color="white")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(facecolor="#161a23", edgecolor="#444444", labelcolor="white")

    # Dice & IoU vs Threshold
    ax2.plot(th_df["threshold"], th_df["dice"], color="#4caf50", lw=2, label="Dice Score")
    ax2.plot(th_df["threshold"], th_df["iou"], color="#29b6f6", lw=2, label="IoU Score")
    ax2.plot(th_df["threshold"], th_df["precision"], color="#ab47bc", ls="--", label="Precision")
    ax2.plot(th_df["threshold"], th_df["recall"], color="#ffa726", ls="--", label="Recall")
    ax2.axvline(optimal_th, color="#e91e63", ls=":", lw=2, label=f"Optimal tau* = {optimal_th}")
    ax2.set_title("Model A: Operating Metric vs Decision Threshold", color="white", fontsize=12)
    ax2.set_xlabel("Classification Threshold (tau)", color="white")
    ax2.set_ylabel("Score", color="white")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(facecolor="#161a23", edgecolor="#444444", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_dir / "threshold_plots.png", dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Probability Histogram Plot
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0e1117")
    ax.set_facecolor("#161a23")
    ax.grid(True, linestyle="--", alpha=0.3, color="#888888")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    pos_probs = flat_probs[flat_targets == 1]
    neg_probs = flat_probs[flat_targets == 0]

    ax.hist(neg_probs, bins=50, alpha=0.6, color="#42a5f5", label=f"Background (N={len(neg_probs):,})", density=True)
    ax.hist(pos_probs, bins=50, alpha=0.7, color="#ef5350", label=f"Landslides (N={len(pos_probs):,})", density=True)
    ax.axvline(optimal_th, color="#ffeb3b", ls="--", lw=2, label=f"tau* = {optimal_th}")
    ax.set_title("Model A: Predicted Probability Density by Class", color="white", fontsize=12)
    ax.set_xlabel("Predicted Probability P(Landslide)", color="white")
    ax.set_ylabel("Normalized Density", color="white")
    ax.legend(facecolor="#161a23", edgecolor="#444444", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_dir / "probability_histograms.png", dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # =========================================================================
    # STAGE 6: Generalization Analysis by Region / Inventory
    # =========================================================================
    print("\n" + "=" * 72)
    print("  Stage 6: Regional Inventory Generalization Analysis")
    print("=" * 72)

    unique_regions = sorted(list(set(val_regions_all)))
    inv_rows = []

    for reg in unique_regions:
        reg_indices = [i for i, r in enumerate(val_regions_all) if r == reg]
        reg_probs = val_probs_tensor[reg_indices].numpy().flatten()
        reg_targets = val_targets_tensor[reg_indices].numpy().flatten().astype(np.int64)

        # Evaluate at fixed 0.50 and optimal tau*
        m_050 = compute_segmentation_metrics(reg_probs, reg_targets, threshold=0.50, compute_curve_auc=False)
        m_opt = compute_segmentation_metrics(reg_probs, reg_targets, threshold=optimal_th, compute_curve_auc=False)

        inv_row = {
            "region_id": reg,
            "sample_count": len(reg_indices),
            "total_pixels": int(reg_targets.size),
            "positive_pixels": int((reg_targets == 1).sum()),
            "ground_truth_prevalence_pct": round(float((reg_targets == 1).mean() * 100.0), 3),
            "dice_at_050": m_050["dice"],
            "iou_at_050": m_050["iou"],
            "precision_at_050": m_050["precision"],
            "recall_at_050": m_050["recall"],
            "specificity_at_050": m_050["specificity"],
            "predicted_prevalence_at_050_pct": round(m_050["predicted_prevalence"] * 100.0, 3),
            "dice_at_optimal": m_opt["dice"],
            "iou_at_optimal": m_opt["iou"],
            "precision_at_optimal": m_opt["precision"],
            "recall_at_optimal": m_opt["recall"],
            "specificity_at_optimal": m_opt["specificity"],
            "predicted_prevalence_at_optimal_pct": round(m_opt["predicted_prevalence"] * 100.0, 3),
        }
        inv_rows.append(inv_row)

    inv_df = pd.DataFrame(inv_rows)
    inv_df.to_csv(out_dir / "inventory_metrics.csv", index=False)

    macro_dice_050 = float(inv_df["dice_at_050"].mean())
    macro_dice_opt = float(inv_df["dice_at_optimal"].mean())
    median_dice_opt = float(inv_df["dice_at_optimal"].median())
    worst_inv = inv_df.sort_values("dice_at_optimal").iloc[0]["region_id"]
    best_inv = inv_df.sort_values("dice_at_optimal", ascending=False).iloc[0]["region_id"]

    print(f"Inventories Evaluated:    {len(unique_regions)}")
    print(f"Macro-Average Dice @0.5:  {macro_dice_050:.4f}")
    print(f"Macro-Average Dice @tau*: {macro_dice_opt:.4f}")
    print(f"Median Dice @tau*:        {median_dice_opt:.4f}")
    print(f"Worst Region:             {worst_inv} (Dice: {inv_df[inv_df['region_id']==worst_inv]['dice_at_optimal'].values[0]:.4f})")
    print(f"Best Region:              {best_inv} (Dice: {inv_df[inv_df['region_id']==best_inv]['dice_at_optimal'].values[0]:.4f})")
    print(f"Inventory breakdown exported to: {out_dir / 'inventory_metrics.csv'}")

    # =========================================================================
    # STAGE 7: Visual Prediction Audit (8-Panel Diagnostic Panels)
    # =========================================================================
    print("\n" + "=" * 72)
    print("  Stage 7: Visual Prediction Audits (8-Panel Diagnostic Figures)")
    print("=" * 72)

    # Identify diverse representative cases across validation set
    num_val = len(val_sample_ids_all)
    sample_scores = []
    for i in range(num_val):
        p_i = val_probs_tensor[i, 0].numpy().flatten()
        y_i = val_targets_tensor[i, 0].numpy().flatten().astype(np.int64)
        pos_count = int((y_i == 1).sum())
        dice_i = compute_segmentation_metrics(p_i, y_i, threshold=optimal_th, compute_curve_auc=False)["dice"]
        fp_count = int(((p_i >= optimal_th) & (y_i == 0)).sum())
        fn_count = int(((p_i < optimal_th) & (y_i == 1)).sum())
        sample_scores.append({
            "index": i,
            "sample_id": val_sample_ids_all[i],
            "region_id": val_regions_all[i],
            "dice": dice_i,
            "positive_pixels": pos_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
        })

    sample_scores_df = pd.DataFrame(sample_scores)

    # Select representative cases
    # 1. High performing
    high_perf_idx = int(sample_scores_df.sort_values("dice", ascending=False).iloc[0]["index"])
    # 2. Low performing
    low_perf_idx = int(sample_scores_df.sort_values("dice", ascending=True).iloc[0]["index"])
    # 3. Small scar
    small_scar_idx = int(sample_scores_df[sample_scores_df["positive_pixels"] > 0].sort_values("positive_pixels").iloc[0]["index"])
    # 4. Large scar
    large_scar_idx = int(sample_scores_df.sort_values("positive_pixels", ascending=False).iloc[0]["index"])
    # 5. False positive case
    fp_case_idx = int(sample_scores_df.sort_values("fp_count", ascending=False).iloc[0]["index"])
    # 6. False negative case
    fn_case_idx = int(sample_scores_df.sort_values("fn_count", ascending=False).iloc[0]["index"])

    cases = [
        ("case_01_high_performing", high_perf_idx, "High-Performing Detection"),
        ("case_02_low_performing", low_perf_idx, "Low-Performing Detection"),
        ("case_03_small_scar", small_scar_idx, "Small Landslide Scar"),
        ("case_04_large_scar", large_scar_idx, "Large Landslide Scar"),
        ("case_05_false_positive", fp_case_idx, "High False-Positive Challenge"),
        ("case_06_false_negative", fn_case_idx, "False-Negative Challenge"),
    ]

    for case_id, idx, label in cases:
        raw_vv = val_raw_tensor[idx, 0].numpy()  # VV
        raw_vh = val_raw_tensor[idx, 1].numpy()  # VH
        gt_mask = val_targets_tensor[idx, 0].numpy()
        pred_prob = val_probs_tensor[idx, 0].numpy()
        pred_050 = (pred_prob >= 0.50).astype(np.float32)
        pred_opt = (pred_prob >= optimal_th).astype(np.float32)

        # Composite overlay
        overlay = np.zeros((128, 128, 3), dtype=np.float32)
        # Background SAR normalized intensity in gray
        norm_vv = (raw_vv - raw_vv.min()) / (raw_vv.max() - raw_vv.min() + 1e-6)
        overlay[..., 0] = norm_vv * 0.7
        overlay[..., 1] = norm_vv * 0.7
        overlay[..., 2] = norm_vv * 0.7

        # False positives in Blue, False negatives in Yellow, True positives in Green
        tp_mask = (pred_opt == 1) & (gt_mask == 1)
        fp_mask = (pred_opt == 1) & (gt_mask == 0)
        fn_mask = (pred_opt == 0) & (gt_mask == 1)

        overlay[tp_mask] = [0.0, 1.0, 0.0]  # Green TP
        overlay[fp_mask] = [0.0, 0.4, 1.0]  # Blue FP
        overlay[fn_mask] = [1.0, 0.8, 0.0]  # Yellow FN

        fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor="#0e1117")
        fig.suptitle(f"Model A Audit Panel: {label} ({val_sample_ids_all[idx]} - {val_regions_all[idx]})", color="white", fontsize=14, y=0.98)

        for ax in axes.flat:
            ax.set_facecolor("#161a23")
            ax.axis("off")

        # 1. Raw VV
        im0 = axes[0, 0].imshow(raw_vv, cmap="gray")
        axes[0, 0].set_title(f"1. Raw SAR VV (dB)\nRange: [{raw_vv.min():.1f}, {raw_vv.max():.1f}]", color="white", fontsize=10)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Raw VH
        im1 = axes[0, 1].imshow(raw_vh, cmap="gray")
        axes[0, 1].set_title(f"2. Raw SAR VH (dB)\nRange: [{raw_vh.min():.1f}, {raw_vh.max():.1f}]", color="white", fontsize=10)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 3. Normalized Dual-Polarization Composite
        dp_comp = np.stack([norm_vv, (raw_vh - raw_vh.min()) / (raw_vh.max() - raw_vh.min() + 1e-6), norm_vv], axis=-1)
        axes[0, 2].imshow(dp_comp)
        axes[0, 2].set_title("3. Dual-Pol SAR Composite\n(VV, VH, VV)", color="white", fontsize=10)

        # 4. Ground Truth Mask
        im3 = axes[0, 3].imshow(gt_mask, cmap="Reds", vmin=0, vmax=1)
        prev_pct = (gt_mask == 1).sum() / gt_mask.size * 100.0
        axes[0, 3].set_title(f"4. Ground Truth Mask\nLandslide Pixels: {prev_pct:.2f}%", color="white", fontsize=10)
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

        # 5. Predicted Probability Map
        im4 = axes[1, 0].imshow(pred_prob, cmap="viridis", vmin=0, vmax=1)
        axes[1, 0].set_title(f"5. Model A Probability Map\nMean P: {pred_prob.mean():.3f}", color="white", fontsize=10)
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 6. Binary at 0.50
        im5 = axes[1, 1].imshow(pred_050, cmap="inferno", vmin=0, vmax=1)
        dice_050_val = compute_segmentation_metrics(pred_prob.flatten(), gt_mask.flatten(), threshold=0.50, compute_curve_auc=False)["dice"]
        axes[1, 1].set_title(f"6. Binarized at tau=0.50\nDice: {dice_050_val:.4f}", color="white", fontsize=10)
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 7. Binary at optimal threshold tau*
        im6 = axes[1, 2].imshow(pred_opt, cmap="inferno", vmin=0, vmax=1)
        dice_opt_val = compute_segmentation_metrics(pred_prob.flatten(), gt_mask.flatten(), threshold=optimal_th, compute_curve_auc=False)["dice"]
        axes[1, 2].set_title(f"7. Binarized at tau*={optimal_th}\nDice: {dice_opt_val:.4f}", color="white", fontsize=10)
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # 8. Detection Overlay
        axes[1, 3].imshow(overlay)
        axes[1, 3].set_title("8. Diagnostic Classification Overlay\nGreen: TP | Blue: FP | Yellow: FN", color="white", fontsize=10)

        plt.tight_layout()
        plt.savefig(vis_dir / f"{case_id}.png", dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    print(f"Generated {len(cases)} comprehensive 8-panel visual audit figures in: {vis_dir}")

    # =========================================================================
    # Export Overall Metrics JSON
    # =========================================================================
    pooled_050 = compute_segmentation_metrics(flat_probs, flat_targets, threshold=0.50, compute_curve_auc=True)
    pooled_opt = compute_segmentation_metrics(flat_probs, flat_targets, threshold=optimal_th, compute_curve_auc=True)

    final_metrics = {
        "phase": "10A",
        "model": "Model A: SAR U-Net",
        "parameters": total_params,
        "epochs_completed": epochs,
        "runtime_seconds": round(time.time() - start_time, 2),
        "device": str(device),
        "device_name": device_name,
        "best_epoch": int(history_df.sort_values("dice", ascending=False).iloc[0]["epoch"]),
        "best_val_loss": round(float(history_df["val_loss"].min()), 5),
        "validation_pooled_metrics_at_050": pooled_050,
        "validation_pooled_metrics_at_optimal_threshold": pooled_opt,
        "optimal_threshold": optimal_th,
        "pr_auc": round(float(pr_auc_val), 5),
        "macro_average_inventory_dice_at_optimal": round(macro_dice_opt, 5),
        "median_inventory_dice_at_optimal": round(median_dice_opt, 5),
        "worst_inventory": worst_inv,
        "best_inventory": best_inv,
        "artifacts": {
            "experiment_config": str(out_dir / "experiment_config.json"),
            "training_history": str(out_dir / "training_history.csv"),
            "threshold_analysis": str(out_dir / "threshold_analysis.csv"),
            "inventory_metrics": str(out_dir / "inventory_metrics.csv"),
            "checkpoints": {
                "best_val_loss": str(ckpt_dir / "best_val_loss.pt"),
                "best_val_dice_at_050": str(ckpt_dir / "best_val_dice_at_050.pt"),
                "latest_checkpoint": str(ckpt_dir / "latest_checkpoint.pt"),
            },
            "plots": {
                "training_curves": str(out_dir / "training_curves.png"),
                "threshold_plots": str(out_dir / "threshold_plots.png"),
                "probability_histograms": str(out_dir / "probability_histograms.png"),
            },
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 72)
    print("  Phase 10A Benchmark Execution Complete")
    print("=" * 72)
    print(f"Total Runtime:           {final_metrics['runtime_seconds']}s")
    print(f"Val Dice @ 0.50:         {pooled_050['dice']:.4f}")
    print(f"Val Dice @ tau* ({optimal_th}):  {pooled_opt['dice']:.4f} (IoU: {pooled_opt['iou']:.4f})")
    print(f"Val Precision @ tau*:    {pooled_opt['precision']:.4f}")
    print(f"Val Recall @ tau*:       {pooled_opt['recall']:.4f}")
    print(f"PR-AUC:                  {final_metrics['pr_auc']:.4f}")
    print(f"Metrics JSON:            {out_dir / 'metrics.json'}")
    print("=" * 72)

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 10A Model A SAR U-Net Benchmark")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--real", action="store_true", help="Use real Sen12Landslides manifest")
    args = parser.parse_args()

    run_phase10a_benchmark(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_pref=args.device,
        use_real_manifest=args.real,
    )
