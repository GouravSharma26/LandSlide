"""CLI entry point to train and evaluate GeoTrustNet with reliability-aware fusion and uncertainty estimation."""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure workspace root is in path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Windows DLL initialization for torch (must import torch before pandas/numpy on Windows)
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches
from src.geotrustnet.models.geotrustnet import GeoTrustNet
from src.geotrustnet.training.engine import GeoTrustNetTrainer, seed_everything
from src.geotrustnet.training.losses import CombinedBCEDiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.evaluation.uncertainty import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_selective_prediction_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoTrustNet with Reliability-Aware Fusion & Uncertainty")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/geotrustnet_phase5_experiment.json",
        help="Path to experiment configuration JSON",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--mc-samples", type=int, default=10, help="Number of Monte Carlo Dropout passes")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda')")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test on CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    epochs = 2 if args.smoke_test else (args.epochs or config["training"]["epochs"])
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["learning_rate"]
    mc_samples = 4 if args.smoke_test else args.mc_samples
    seed = config.get("seed", 42)

    seed_everything(seed)

    if args.device:
        device = torch.device(args.device)
    elif args.smoke_test:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================")
    print(f"GeoTrustNet Phase 5: Reliability-Aware Fusion & Uncertainty")
    print(f"==================================================")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr} | MC Samples: {mc_samples}")

    manifest_path = root / config["split_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run generate_fixtures_and_manifests.py first.")

    manifest_df = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(manifest_df)} total samples.")

    train_ds = GeoTrustNetPatchDataset(manifest_df, split="train", data_root=root)
    val_ds = GeoTrustNetPatchDataset(manifest_df, split="val", data_root=root)
    test_ds = GeoTrustNetPatchDataset(manifest_df, split="test", data_root=root)

    print(f"Splits -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_geotrustnet_patches)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_geotrustnet_patches)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_geotrustnet_patches)

    # Initialize GeoTrustNet Core Architecture
    model_cfg_path = root / config["model_config"]
    if model_cfg_path.exists():
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        features = model_cfg.get("features", [32, 64, 128, 256])
        dropout = model_cfg.get("dropout", 0.2)
        spatial_rel = model_cfg.get("spatial_reliability", True)
    else:
        features = [32, 64, 128, 256]
        dropout = 0.2
        spatial_rel = True

    model = GeoTrustNet(
        features=features,
        sar_channels=2,
        optical_channels=10,
        dem_channels=2,
        out_channels=1,
        bilinear=True,
        dropout=dropout,
        spatial_reliability=spatial_rel,
    )
    print(f"Model initialized: GeoTrustNet Core ({model.count_parameters():,} trainable parameters)")

    criterion = CombinedBCEDiceLoss(
        bce_weight=config["loss"]["bce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        smooth=config["loss"]["smooth"],
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = root / config.get("output_directory", "artifacts/geotrustnet/experiments/geotrustnet_phase5")

    trainer = GeoTrustNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        experiment_config=config,
        monitor_metric=config["training"].get("monitor_metric", "val_iou"),
        monitor_mode=config["training"].get("monitor_mode", "max"),
        patience=config["training"].get("patience", 5),
        modality="multimodal",
        is_multimodal=True,
    )

    print("\nStarting GeoTrustNet training loop...")
    summary = trainer.fit(epochs=epochs)

    print("\n--- Training Complete ---")
    print(f"Best Epoch: {summary['best_epoch']} | Best {summary['monitored_metric']}: {summary['best_metric_value']}")
    print(f"Outputs saved to: {output_dir}")

    # Evaluate on held-out test split with Monte Carlo Dropout Uncertainty
    best_ckpt_path = output_dir / "best_model.pt"
    if best_ckpt_path.exists():
        print(f"\nEvaluating best GeoTrustNet checkpoint with {mc_samples} MC Dropout passes...")
        trainer.load_checkpoint(best_ckpt_path)
        model.eval()

        all_means = []
        all_variances = []
        all_targets = []
        all_rel_sar = []
        all_rel_opt = []
        all_rel_dem = []

        for batch in test_loader:
            mask = batch["mask"].to(device).float()
            unc_dict = model.predict_uncertainty(
                sar=batch["sar"].to(device).float(),
                optical=batch["optical"].to(device).float(),
                dem=batch["dem"].to(device).float(),
                has_sar=batch["has_sar"].to(device).bool(),
                has_optical=batch["has_optical"].to(device).bool(),
                has_dem=batch["has_dem"].to(device).bool(),
                sar_quality=batch["sar_valid"].to(device).float(),
                optical_quality=batch["optical_valid"].to(device).float(),
                num_mc_samples=mc_samples,
            )
            all_means.append(unc_dict["mean_prob"].detach().cpu())
            all_variances.append(unc_dict["variance"].detach().cpu())
            all_targets.append(mask.detach().cpu())
            all_rel_sar.append(unc_dict["reliability_sar"].mean().item())
            all_rel_opt.append(unc_dict["reliability_optical"].mean().item())
            all_rel_dem.append(unc_dict["reliability_dem"].mean().item())

        concat_means = torch.cat(all_means, dim=0)
        concat_vars = torch.cat(all_variances, dim=0)
        concat_targets = torch.cat(all_targets, dim=0)

        # 1. Standard segmentation metrics
        test_metrics = compute_segmentation_metrics(concat_means, concat_targets)

        # 2. Calibration metrics
        brier = compute_brier_score(concat_means, concat_targets)
        ece_res = compute_expected_calibration_error(concat_means, concat_targets, num_bins=10)

        # 3. Selective prediction curve
        selective_curve = compute_selective_prediction_curve(concat_means, concat_vars, concat_targets)

        print(f"\n--- Held-Out Test Evaluation Summary ---")
        print(f"IoU: {test_metrics['iou']:.4f} | Dice: {test_metrics['dice']:.4f} | PR-AUC: {test_metrics['pr_auc']}")
        print(f"Brier Score: {brier:.4f} | Expected Calibration Error (ECE): {ece_res['ece']:.4f}")
        print(f"Mean Reliability Weights -> SAR: {float(pd.Series(all_rel_sar).mean()):.3f}, "
              f"Optical: {float(pd.Series(all_rel_opt).mean()):.3f}, DEM: {float(pd.Series(all_rel_dem).mean()):.3f}")

        # Save uncertainty report
        report_data = {
            "model": "GeoTrustNet-Core",
            "mc_samples": mc_samples,
            "metrics": test_metrics,
            "calibration": {
                "brier_score": round(brier, 5),
                "ece": ece_res["ece"],
                "mce": ece_res["mce"],
                "reliability_diagram_bins": ece_res["bins"],
            },
            "mean_modality_reliability": {
                "sar": round(float(pd.Series(all_rel_sar).mean()), 4),
                "optical": round(float(pd.Series(all_rel_opt).mean()), 4),
                "dem": round(float(pd.Series(all_rel_dem).mean()), 4),
            },
            "selective_prediction_curve": selective_curve,
        }

        unc_report_path = output_dir / "test_uncertainty_report.json"
        with open(unc_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Detailed uncertainty and calibration report saved to: {unc_report_path}")


if __name__ == "__main__":
    main()
