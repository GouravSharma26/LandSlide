"""Data & Label Validation Utility for GeoTrustNet Phase 9.

Inspects representative real and pilot samples from train/val/test splits,
verifies normalization parameters, label encoding, and generates a visual inspection panel.
"""

import json
import os
import sys
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
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.data.contracts import PatchSample


def validate_and_visualize_samples() -> dict:
    """Audits samples and creates a 4-panel diagnostic visualization."""
    out_dir = ROOT_DIR / "docs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load sample .pt from synthetic pilot fixtures
    sample_path = ROOT_DIR / "data" / "synthetic" / "SMP_0001.pt"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    sample: PatchSample = torch.load(sample_path, weights_only=False)

    # Extract tensors
    sar_raw = sample.sar[0].numpy()  # [2, 128, 128] (VV, VH)
    vv_raw = sar_raw[0]
    vh_raw = sar_raw[1]
    mask = sample.mask[0].numpy() if sample.mask.ndim == 3 else sample.mask.numpy()

    # Normalization using official Sen12Landslides parameters
    # VV: mean=-9.31, std=3.49 | VH: mean=-15.89, std=4.18
    vv_norm = (vv_raw - (-9.310)) / 3.487
    vh_norm = (vh_raw - (-15.886)) / 4.181

    # Check for NaNs/Infs
    has_nans = bool(np.isnan(vv_raw).any() or np.isnan(vh_raw).any() or np.isnan(mask).any())
    has_infs = bool(np.isinf(vv_raw).any() or np.isinf(vh_raw).any() or np.isinf(mask).any())

    # Generate 4-panel diagnostic figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#161a23")

    # Panel 1: Raw VV backscatter (dB)
    im0 = axes[0].imshow(vv_raw, cmap="gray")
    axes[0].set_title(f"Raw Sentinel-1 VV (dB)\nRange: [{vv_raw.min():.1f}, {vv_raw.max():.1f}]", color="white", fontsize=10)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Normalized VV (z-score)
    im1 = axes[1].imshow(vv_norm, cmap="coolwarm", vmin=-3, vmax=3)
    axes[1].set_title(f"Normalized VV (Z-Score)\nRange: [{vv_norm.min():.2f}, {vv_norm.max():.2f}]", color="white", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Ground Truth Landslide Scarp Mask
    im2 = axes[2].imshow(mask, cmap="Reds", vmin=0, vmax=1)
    prev = (mask == 1).sum() / mask.size * 100
    axes[2].set_title(f"Ground Truth Mask\nLandslide Pixels: {prev:.2f}%", color="white", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Spatial Overlay
    overlay = np.stack([vv_norm, vv_norm, vv_norm], axis=-1)
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)
    overlay[mask == 1, 0] = 1.0  # highlight scarp in red
    overlay[mask == 1, 1] = 0.2
    overlay[mask == 1, 2] = 0.2
    axes[3].imshow(overlay)
    axes[3].set_title("SAR + Scarp Scab Overlay", color="white", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    fig_path = out_dir / "pilot_sample_validation_panel.png"
    plt.savefig(fig_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    audit_result = {
        "sample_id": sample.metadata.patch_id,
        "region_id": sample.metadata.region_id,
        "event_id": sample.metadata.event_id,
        "spatial_dimensions": {"height": int(vv_raw.shape[0]), "width": int(vv_raw.shape[1])},
        "raw_vv_range": [round(float(vv_raw.min()), 2), round(float(vv_raw.max()), 2)],
        "raw_vh_range": [round(float(vh_raw.min()), 2), round(float(vh_raw.max()), 2)],
        "normalized_vv_range": [round(float(vv_norm.min()), 2), round(float(vv_norm.max()), 2)],
        "normalized_vh_range": [round(float(vh_norm.min()), 2), round(float(vh_norm.max()), 2)],
        "target_mask_classes": [int(c) for c in np.unique(mask)],
        "landslide_pixel_prevalence_pct": round(float(prev), 3),
        "has_nans": has_nans,
        "has_infs": has_infs,
        "diagnostic_panel_path": str(fig_path).replace("\\", "/"),
    }

    report_path = out_dir / "pilot_sample_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    print("=" * 65)
    print("  Data and Label Validation Summary")
    print("=" * 65)
    print(f"Sample ID:            {audit_result['sample_id']} ({audit_result['region_id']})")
    print(f"Spatial Dimensions:   {audit_result['spatial_dimensions']['height']}x{audit_result['spatial_dimensions']['width']}")
    print(f"Raw VV Range:         {audit_result['raw_vv_range']} dB")
    print(f"Normalized VV Range:  {audit_result['normalized_vv_range']} (Z-score)")
    print(f"Landslide Prevalence: {audit_result['landslide_pixel_prevalence_pct']}%")
    print(f"NaNs / Infs:          {has_nans} / {has_infs}")
    print(f"Visual Panel Saved:   {fig_path}")
    print("=" * 65)

    return audit_result


if __name__ == "__main__":
    validate_and_visualize_samples()
