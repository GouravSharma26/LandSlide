"""Unified Benchmark & Ablation Runner for GeoTrustNet.

Coordinates genuine model inference and evaluation across:
    - Model A: SAR U-Net Baseline
    - Model B: Temporal SAR ConvLSTM
    - Model C: Conventional Multimodal Fusion
    - Model D: GeoTrustNet (Reliability-Aware Fusion & Uncertainty)

All metrics, calibration values, and confusion matrices are computed directly
from model predictions on DataLoader batches. No hardcoded or mock values are permitted.

Safeguards:
  - Real-data benchmarks CANNOT silently fall back to synthetic fixtures.
  - Evaluation threshold is fixed to 0.5 (no post-hoc test-set threshold search).
  - Checkpoint provenance, parameter counts, configuration hashes, and dataset provenance
    are recorded in every generated artifact.

Usage:
    # Run genuine smoke-test evaluation on synthetic fixtures:
    python scripts/run_benchmarks_and_ablations.py --model all --smoke-test

    # Run genuine benchmark on real Sen12Landslides data (fails if files missing):
    python scripts/run_benchmarks_and_ablations.py --model all --real-data --device cpu
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.training.engine import seed_everything
from src.geotrustnet.models.unet import UNet
from src.geotrustnet.models.temporal_unet import TemporalSARUNet
from src.geotrustnet.models.multimodal_unet import MultimodalUNet
from src.geotrustnet.models.geotrustnet import GeoTrustNet
from src.geotrustnet.data.datasets import GeoTrustNetPatchDataset, collate_geotrustnet_patches
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.evaluation.uncertainty import (
    compute_brier_score,
    compute_expected_calibration_error,
)


def load_benchmark_config(config_path: Path, smoke_test: bool = False) -> Dict[str, Any]:
    """Loads and resolves benchmark configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if smoke_test:
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 2
        cfg["device_preference"] = "cpu"

    return cfg


def resolve_split_manifest(cfg: Dict[str, Any], allow_synthetic_fallback: bool = False) -> Tuple[Path, str, bool]:
    """Resolves whether real or synthetic split manifest is used, verifying file existence.

    Args:
        cfg: Experiment configuration dictionary.
        allow_synthetic_fallback: If True (smoke test mode), allows fallback to synthetic fixtures.
                                  If False (real data mode), raises FileNotFoundError if real files are missing.

    Returns:
        (resolved_manifest_path, provenance_label, is_real_data)
    """
    manifest_rel = cfg.get("split_manifest", "")
    real_path = ROOT_DIR / manifest_rel if manifest_rel else None

    # Check if real manifest exists and whether files referenced inside actually exist
    if real_path and real_path.exists():
        try:
            df = pd.read_csv(real_path)
            if "file_path" in df.columns:
                sample_files = df["file_path"].dropna().head(5)
                files_exist = any((ROOT_DIR / p).exists() or Path(p).exists() for p in sample_files)
                if files_exist:
                    return real_path, "sen12landslides_real", True
        except Exception:
            pass

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"Real Sen12Landslides dataset files referenced in '{manifest_rel}' were not found on disk. "
            "Silent fallback to synthetic data is prohibited for real-data benchmark execution. "
            "To run a rapid synthetic smoke test, pass --smoke-test explicitly."
        )

    # Fallback to synthetic manifest only when explicitly allowed
    fallback_rel = cfg.get("split_manifest_fallback", "data/manifests/synthetic_split_manifest.csv")
    fallback_path = ROOT_DIR / fallback_rel
    if fallback_path.exists():
        return fallback_path, "synthetic_validation_fixtures", False

    raise FileNotFoundError(f"Neither real manifest ({real_path}) nor fallback ({fallback_path}) could be resolved.")


def instantiate_model(model_key: str, cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, Optional[Path]]:
    """Instantiates model architecture and loads checkpoint weights if available."""
    features = [32, 64, 128, 256]

    checkpoint_candidates = {
        "model_a": [
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "benchmark_model_a" / "best_model.pt",
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "baseline_s1" / "best_model.pt",
        ],
        "model_b": [
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "benchmark_model_b" / "best_model.pt",
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "temporal_s1" / "best_model.pt",
        ],
        "model_c": [
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "benchmark_model_c" / "best_model.pt",
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "multimodal_fusion" / "best_model.pt",
        ],
        "model_d": [
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "benchmark_model_d" / "best_model.pt",
            ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "geotrustnet_phase5" / "best_model.pt",
        ],
    }

    if model_key == "model_a":
        model = UNet(in_channels=2, out_channels=1, features=features)
    elif model_key == "model_b":
        model = TemporalSARUNet(in_channels=2, out_channels=1, features=features)
    elif model_key == "model_c":
        model = MultimodalUNet(sar_channels=2, optical_channels=10, dem_channels=2, features=features)
    elif model_key == "model_d":
        model = GeoTrustNet(sar_channels=2, optical_channels=10, dem_channels=2, features=features)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    # Search and load trained weights
    loaded_ckpt = None
    for ckpt_path in checkpoint_candidates.get(model_key, []):
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                loaded_ckpt = ckpt_path
                break
            except Exception as e:
                print(f"    Warning: Could not load checkpoint from {ckpt_path.name}: {e}")

    model.to(device)
    model.eval()

    if loaded_ckpt:
        print(f"  - Loaded checkpoint: {loaded_ckpt.relative_to(ROOT_DIR)}")
    else:
        print(f"  - Initialized with fresh random weights (no checkpoint found)")

    return model, loaded_ckpt


def run_benchmark_experiment(
    model_key: str,
    smoke_test: bool = True,
    real_data: bool = False,
    device_pref: str = "cpu",
) -> Dict[str, Any]:
    """Runs genuine model evaluation across test batches and saves reproducible reports."""
    config_map = {
        "model_a": ROOT_DIR / "configs" / "experiments" / "benchmark_model_a_sar_unet.json",
        "model_b": ROOT_DIR / "configs" / "experiments" / "benchmark_model_b_temporal_sar.json",
        "model_c": ROOT_DIR / "configs" / "experiments" / "benchmark_model_c_multimodal_fusion.json",
        "model_d": ROOT_DIR / "configs" / "experiments" / "benchmark_model_d_geotrustnet.json",
    }

    if model_key not in config_map:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(config_map.keys())}")

    cfg = load_benchmark_config(config_map[model_key], smoke_test=smoke_test)
    if device_pref:
        cfg["device_preference"] = device_pref

    seed_everything(cfg.get("seed", 42))
    device = torch.device(cfg["device_preference"] if torch.cuda.is_available() and cfg["device_preference"] == "cuda" else "cpu")

    out_dir = ROOT_DIR / cfg["output_directory"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve manifest: forbid synthetic fallback if real_data=True
    allow_fallback = smoke_test and not real_data
    manifest_path, provenance_label, is_real_data = resolve_split_manifest(cfg, allow_synthetic_fallback=allow_fallback)

    print(f"\n[{cfg['model_designation']}] Genuine Evaluation Runner")
    print(f"  - Config: {config_map[model_key].name}")
    print(f"  - Manifest: {manifest_path.name} ({provenance_label})")
    print(f"  - Device: {device}")

    # Build Test Dataset and DataLoader
    target_split = "test"
    try:
        dataset = GeoTrustNetPatchDataset(
            manifest_df=manifest_path,
            split=target_split,
            data_root=ROOT_DIR,
            normalize=True,
        )
    except ValueError:
        # Fallback to val if test has 0 samples in manifest
        print("    Test split had 0 samples, evaluating on val split...")
        target_split = "val"
        dataset = GeoTrustNetPatchDataset(
            manifest_df=manifest_path,
            split=target_split,
            data_root=ROOT_DIR,
            normalize=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"].get("batch_size", 2),
        shuffle=False,
        collate_fn=collate_geotrustnet_patches,
    )

    # Instantiate model
    model, loaded_ckpt = instantiate_model(model_key, cfg, device)

    # Run genuine inference loop
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_valid_masks: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            y = batch["mask"].to(device).float()
            mask_valid = batch.get("mask_valid", None)
            if mask_valid is not None:
                mask_valid = mask_valid.to(device).bool()

            # Execute forward pass based on model architecture
            if model_key == "model_a":
                # Single-modality SAR baseline: take latest time step [B, 2, H, W]
                sar = batch["sar"].to(device).float()
                if sar.ndim == 5:
                    sar = sar[:, -1, :, :, :]
                logits = model(sar)

            elif model_key == "model_b":
                # Spatio-temporal SAR ConvLSTM: requires 5D sequence [B, T, 2, H, W]
                sar = batch["sar"].to(device).float()
                if sar.ndim == 4:
                    sar = sar.unsqueeze(1)
                logits = model(sar)

            elif model_key == "model_c":
                # Multimodal conventional fusion
                sar = batch["sar"].to(device).float() if "sar" in batch else None
                optical = batch["optical"].to(device).float() if "optical" in batch and batch["has_optical"].any() else None
                dem = batch["dem"].to(device).float() if "dem" in batch else None
                has_sar = batch["has_sar"].to(device).bool() if "has_sar" in batch else None
                has_optical = batch["has_optical"].to(device).bool() if "has_optical" in batch else None
                has_dem = batch["has_dem"].to(device).bool() if "has_dem" in batch else None

                logits = model(
                    sar=sar,
                    optical=optical,
                    dem=dem,
                    has_sar=has_sar,
                    has_optical=has_optical,
                    has_dem=has_dem,
                )

            elif model_key == "model_d":
                # GeoTrustNet reliability-aware fusion
                sar = batch["sar"].to(device).float() if "sar" in batch else None
                optical = batch["optical"].to(device).float() if "optical" in batch and batch["has_optical"].any() else None
                dem = batch["dem"].to(device).float() if "dem" in batch else None
                has_sar = batch["has_sar"].to(device).bool() if "has_sar" in batch else None
                has_optical = batch["has_optical"].to(device).bool() if "has_optical" in batch else None
                has_dem = batch["has_dem"].to(device).bool() if "has_dem" in batch else None
                sar_q = batch["sar_valid"].to(device).float() if "sar_valid" in batch and batch["sar_valid"] is not None else None
                optical_q = batch["optical_valid"].to(device).float() if "optical_valid" in batch and batch["optical_valid"] is not None else None

                logits = model(
                    sar=sar,
                    optical=optical,
                    dem=dem,
                    has_sar=has_sar,
                    has_optical=has_optical,
                    has_dem=has_dem,
                    sar_quality=sar_q,
                    optical_quality=optical_q,
                )

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            if mask_valid is not None:
                all_valid_masks.append(mask_valid.cpu().numpy())

    # Concatenate predictions across all batches
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0,))
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,))
    valid_np = np.concatenate(all_valid_masks, axis=0) if all_valid_masks else None

    # Compute genuine segmentation metrics with fixed threshold=0.5 (no test threshold tuning)
    EVALUATION_THRESHOLD = 0.5
    metrics = compute_segmentation_metrics(
        logits_or_probs=probs_np,
        targets=targets_np,
        threshold=EVALUATION_THRESHOLD,
        compute_curve_auc=True,
        valid_mask=valid_np,
    )

    # Compute genuine calibration metrics strictly on valid pixels
    probs_flat = probs_np.ravel()
    targets_flat = targets_np.ravel().astype(np.int64)

    if valid_np is not None:
        valid_flat = valid_np.ravel().astype(bool) & ((targets_flat == 0) | (targets_flat == 1))
    else:
        valid_flat = (targets_flat == 0) | (targets_flat == 1)

    p_eval = probs_flat[valid_flat]
    y_eval = targets_flat[valid_flat]

    if len(p_eval) > 0:
        brier = compute_brier_score(p_eval, y_eval)
        ece_res = compute_expected_calibration_error(p_eval, y_eval, num_bins=10)
        ece = ece_res.get("ece", 0.0)
        mce = ece_res.get("mce", 0.0)
    else:
        brier = 0.0
        ece = 0.0
        mce = 0.0

    calibration = {
        "brier_score": round(brier, 5),
        "ece": round(ece, 5),
        "mce": round(mce, 5),
    }

    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()

    result = {
        "model_key": model_key,
        "model_designation": cfg["model_designation"],
        "experiment_id": cfg["experiment_id"],
        "dataset_manifest": str(manifest_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "data_provenance": provenance_label,
        "is_smoke_test": smoke_test,
        "is_mock_metrics": False,
        "real_data_evaluated": is_real_data,
        "evaluation_type": "inference_evaluation_only",
        "split_evaluated": target_split,
        "sample_count_evaluated": len(dataset),
        "checkpoint_path": str(loaded_ckpt.relative_to(ROOT_DIR)).replace("\\", "/") if loaded_ckpt else None,
        "checkpoint_loaded": bool(loaded_ckpt),
        "config_sha256": config_hash,
        "fixed_evaluation_threshold": EVALUATION_THRESHOLD,
        "provenance_statement": (
            "Empirical benchmark evaluation on real Sen12Landslides satellite imagery."
            if is_real_data
            else "Controlled smoke-test verification on synthetic validation fixtures. Real Sen12Landslides imagery was not present on local path."
        ),
        "metrics": metrics,
        "calibration": calibration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_path = out_dir / "experiment_summary.json"
    metrics_path = out_dir / "test_evaluation_report.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  - Genuine Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f} | ECE: {calibration['ece']:.4f}")
    print(f"  - Evaluation Saved: {metrics_path.relative_to(ROOT_DIR)}")
    return result


def main():
    parser = argparse.ArgumentParser(description="GeoTrustNet Controlled Benchmark Runner")
    parser.add_argument("--model", type=str, default="all", choices=["model_a", "model_b", "model_c", "model_d", "all"], help="Model to evaluate")
    parser.add_argument("--smoke-test", action="store_true", help="Execute rapid smoke test verification on synthetic fixtures")
    parser.add_argument("--real-data", action="store_true", help="Execute evaluation on real Sen12Landslides data (fails if files missing)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device preference")
    args = parser.parse_args()

    targets = ["model_a", "model_b", "model_c", "model_d"] if args.model == "all" else [args.model]

    print("=" * 65)
    print("  GeoTrustNet Benchmark & Controlled Evaluation Runner")
    print("=" * 65)
    print(f"Target Models: {targets}")
    print(f"Smoke Test:    {args.smoke_test}")
    print(f"Real Data:     {args.real_data}")
    print(f"Device:        {args.device}")

    results = []
    for m in targets:
        res = run_benchmark_experiment(
            model_key=m,
            smoke_test=args.smoke_test,
            real_data=args.real_data,
            device_pref=args.device,
        )
        results.append(res)

    print("\n" + "=" * 65)
    print("  Benchmark Summary Table (Genuine Empirical Outputs)")
    print("=" * 65)
    print(f"{'Model':<35} | {'Dice':<8} | {'IoU':<8} | {'ECE':<8}")
    print("-" * 65)
    for r in results:
        m_name = r["model_designation"].split("—")[0].strip()
        d = r["metrics"]["dice"]
        iou = r["metrics"]["iou"]
        ece = r["calibration"]["ece"]
        print(f"{m_name:<35} | {d:<8.4f} | {iou:<8.4f} | {ece:<8.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
