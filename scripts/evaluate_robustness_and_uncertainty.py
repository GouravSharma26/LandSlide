"""Robustness evaluation and controlled ablation benchmark for GeoTrustNet Phase 5."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure workspace root is in path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

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
import pandas as pd
from torch.utils.data import DataLoader

from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches
from src.geotrustnet.models.geotrustnet import GeoTrustNet
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.evaluation.uncertainty import (
    compute_brier_score,
    compute_expected_calibration_error,
)


def evaluate_under_conditions(
    model: GeoTrustNet,
    dataloader: DataLoader,
    device: torch.device,
    optical_dropout_ratio: float = 0.0,
    force_missing_modalities: Optional[list] = None,
    mc_samples: int = 5,
) -> dict:
    """Evaluate GeoTrustNet under controlled sensor degradation and missingness."""
    model.eval()
    force_missing = force_missing_modalities or []

    all_means = []
    all_targets = []

    for batch in dataloader:
        mask = batch["mask"].to(device).float()
        sar = batch["sar"].to(device).float() if "sar" not in force_missing else None
        opt = batch["optical"].to(device).float() if "optical" not in force_missing else None
        dem = batch["dem"].to(device).float() if "dem" not in force_missing else None

        has_sar = batch["has_sar"].to(device).bool() if "sar" not in force_missing else torch.zeros(len(mask), device=device, dtype=torch.bool)
        has_opt = batch["has_optical"].to(device).bool() if "optical" not in force_missing else torch.zeros(len(mask), device=device, dtype=torch.bool)
        has_dem = batch["has_dem"].to(device).bool() if "dem" not in force_missing else torch.zeros(len(mask), device=device, dtype=torch.bool)

        sar_q = batch["sar_valid"].to(device).float() if sar is not None else None
        opt_q = batch["optical_valid"].to(device).float() if opt is not None else None

        # Simulate synthetic cloud contamination on optical data
        if opt is not None and optical_dropout_ratio > 0.0:
            B, C, H, W = opt.shape if opt.ndim == 4 else (opt.shape[0], opt.shape[2], opt.shape[3], opt.shape[4])
            cloud_mask = torch.rand((B, 1, H, W), device=device) < optical_dropout_ratio
            if opt.ndim == 5:
                cloud_mask = cloud_mask.unsqueeze(1)
            opt = opt.masked_fill(cloud_mask, 0.0)
            if opt_q is not None:
                opt_q = opt_q.masked_fill(cloud_mask, 0.0)

        unc_dict = model.predict_uncertainty(
            sar=sar,
            optical=opt,
            dem=dem,
            has_sar=has_sar,
            has_optical=has_opt,
            has_dem=has_dem,
            sar_quality=sar_q,
            optical_quality=opt_q,
            num_mc_samples=mc_samples,
        )

        all_means.append(unc_dict["mean_prob"].detach().cpu())
        all_targets.append(mask.detach().cpu())

    concat_means = torch.cat(all_means, dim=0)
    concat_targets = torch.cat(all_targets, dim=0)

    metrics = compute_segmentation_metrics(concat_means, concat_targets)
    brier = compute_brier_score(concat_means, concat_targets)
    ece_data = compute_expected_calibration_error(concat_means, concat_targets, num_bins=10)

    return {
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "pr_auc": metrics["pr_auc"],
        "brier_score": round(brier, 5),
        "ece": ece_data["ece"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoTrustNet Robustness under Degradation")
    parser.add_argument("--checkpoint", type=str, default="artifacts/geotrustnet/experiments/geotrustnet_phase5/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mc-samples", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = root / args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}. Run train_geotrustnet.py first.")
        return

    manifest_path = root / "data/manifests/synthetic_split_manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    test_ds = GeoTrustNetPatchDataset(manifest_df, split="test", data_root=root)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_geotrustnet_patches)

    model = GeoTrustNet(features=[32, 64, 128, 256])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print("==================================================")
    print("GeoTrustNet Controlled Robustness & Degradation Benchmark")
    print("==================================================")

    # 1. Optical Cloud Degradation Sweep
    print("\n[Experiment 1] Simulating Optical Cloud Contamination:")
    cloud_results = {}
    for cloud_pct in [0.0, 0.25, 0.50, 0.75, 1.0]:
        res = evaluate_under_conditions(
            model=model,
            dataloader=test_loader,
            device=device,
            optical_dropout_ratio=cloud_pct,
            mc_samples=args.mc_samples,
        )
        cloud_results[f"cloud_{int(cloud_pct * 100)}pct"] = res
        print(f"Cloud {int(cloud_pct * 100):3d}% -> IoU: {res['iou']:.4f}, Dice: {res['dice']:.4f}, ECE: {res['ece']:.4f}")

    # 2. Modality Ablation Sweep
    print("\n[Experiment 2] Modality Missingness & Fallback Sweep:")
    modality_results = {}
    scenarios = [
        ("Full Multimodal (SAR+Opt+DEM)", []),
        ("Missing Optical (SAR+DEM)", ["optical"]),
        ("SAR Only", ["optical", "dem"]),
        ("Optical Only", ["sar", "dem"]),
        ("DEM Only", ["sar", "optical"]),
    ]
    for name, missing in scenarios:
        res = evaluate_under_conditions(
            model=model,
            dataloader=test_loader,
            device=device,
            force_missing_modalities=missing,
            mc_samples=args.mc_samples,
        )
        modality_results[name] = res
        print(f"{name:<30} -> IoU: {res['iou']:.4f}, Dice: {res['dice']:.4f}, ECE: {res['ece']:.4f}")

    # Save complete robustness benchmark report
    benchmark_report = {
        "cloud_degradation_benchmark": cloud_results,
        "modality_missingness_benchmark": modality_results,
    }
    out_path = root / "artifacts/geotrustnet/experiments/geotrustnet_phase5/robustness_benchmark_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nRobustness benchmark report successfully saved to: {out_path}")


if __name__ == "__main__":
    main()
