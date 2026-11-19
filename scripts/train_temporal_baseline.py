"""CLI entry point to train and evaluate the GeoTrustNet Spatio-Temporal ConvLSTM on Sentinel-1 SAR."""

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
from src.geotrustnet.models.temporal_unet import TemporalSARUNet
from src.geotrustnet.training.engine import GeoTrustNetTrainer, seed_everything
from src.geotrustnet.training.losses import CombinedBCEDiceLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoTrustNet Sentinel-1 ConvLSTM Temporal Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/temporal_s1_convlstm_experiment.json",
        help="Path to experiment configuration JSON",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cpu' or 'cuda')")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test on CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = root / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Allow CLI overrides
    epochs = 2 if args.smoke_test else (args.epochs or config["training"]["epochs"])
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["learning_rate"]
    seed = config.get("seed", 42)

    seed_everything(seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif args.smoke_test:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================")
    print(f"GeoTrustNet Phase 3: Spatio-Temporal ConvLSTM Baseline Training")
    print(f"==================================================")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")

    # Load dataset manifests
    manifest_path = root / config["split_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run generate_fixtures_and_manifests.py first.")

    manifest_df = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(manifest_df)} total samples.")

    t_steps = config.get("temporal_time_steps", 5)
    train_ds = GeoTrustNetPatchDataset(manifest_df, split="train", data_root=root, fixed_time_steps=t_steps)
    val_ds = GeoTrustNetPatchDataset(manifest_df, split="val", data_root=root, fixed_time_steps=t_steps)
    test_ds = GeoTrustNetPatchDataset(manifest_df, split="test", data_root=root, fixed_time_steps=t_steps)

    print(f"Splits -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_geotrustnet_patches,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_geotrustnet_patches,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_geotrustnet_patches,
    )

    # Load model architecture config
    model_cfg_path = root / config["model_config"]
    if model_cfg_path.exists():
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        features = model_cfg.get("features", [32, 64, 128, 256])
        convlstm_layers = model_cfg.get("convlstm_layers", 1)
        dropout = model_cfg.get("dropout", 0.1)
        aggregation = model_cfg.get("aggregation", "last")
    else:
        features = [32, 64, 128, 256]
        convlstm_layers = 1
        dropout = 0.1
        aggregation = "last"

    model = TemporalSARUNet(
        in_channels=2,  # Sentinel-1 SAR VV, VH
        out_channels=1,
        features=features,
        convlstm_layers=convlstm_layers,
        bilinear=True,
        dropout=dropout,
        aggregation=aggregation,
    )
    print(f"Model initialized: TemporalSARUNet ({model.count_parameters():,} trainable parameters)")

    criterion = CombinedBCEDiceLoss(
        bce_weight=config["loss"]["bce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        smooth=config["loss"]["smooth"],
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = root / config.get("output_directory", "artifacts/geotrustnet/experiments/temporal_s1")

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
        modality=config.get("modality", "sar"),
        temporal_idx=None,  # Pass full temporal 5D tensor
        is_temporal=True,
    )

    print("\nStarting temporal training loop...")
    summary = trainer.fit(epochs=epochs)

    print("\n--- Training Complete ---")
    print(f"Best Epoch: {summary['best_epoch']} | Best {summary['monitored_metric']}: {summary['best_metric_value']}")
    print(f"Outputs saved to: {output_dir}")

    # Evaluate on held-out test split
    best_ckpt_path = output_dir / "best_model.pt"
    if best_ckpt_path.exists():
        print("\nEvaluating best temporal checkpoint on held-out test split...")
        trainer.load_checkpoint(best_ckpt_path)
        test_loss, test_metrics = trainer.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Metrics: IoU={test_metrics['iou']:.4f}, Dice={test_metrics['dice']:.4f}, "
              f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}")

        # Save test evaluation report
        test_eval_path = output_dir / "test_evaluation_report.json"
        with open(test_eval_path, "w", encoding="utf-8") as f:
            json.dump({"test_loss": round(test_loss, 5), "metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
