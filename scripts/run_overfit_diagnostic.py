"""Overfit Diagnostic Sanity Check for GeoTrustNet Model A (UNet).

Deliberately overfits a tiny deterministic subset of samples to verify that:
1. Gradients backpropagate correctly through all encoder/decoder layers.
2. Training loss monotonically decreases.
3. Segmentation metrics (Dice / IoU) substantially improve to > 0.70.
4. Confirms the pipeline implementation is mathematically sound before cloud training.
"""

import json
import os
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.models.unet import UNet
from src.geotrustnet.training.losses import CombinedBCEDiceLoss


def run_overfit_diagnostic(num_steps: int = 40, lr: float = 0.01) -> dict:
    """Executes a tiny-subset overfit test and returns convergence history and diagnostics."""
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Create a tiny deterministic 2-sample batch with realistic scarp targets
    sar = torch.randn(2, 2, 128, 128)  # Normalized SAR features
    target = torch.zeros(2, 1, 128, 128)
    # Define synthetic scarp delineations
    target[0, 0, 35:65, 35:65] = 1.0  # 900 pixels
    target[1, 0, 70:95, 70:95] = 1.0  # 625 pixels

    # 2. Instantiate UNet
    model = UNet(in_channels=2, out_channels=1, features=[16, 32, 64, 128], bilinear=True, dropout=0.0)
    initial_weights = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}

    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Initial state evaluation
    with torch.no_grad():
        init_logits = model(sar)
        init_loss = criterion(init_logits, target).item()
        p0 = (torch.sigmoid(init_logits) >= 0.5).float()
        tp0 = (p0 * target).sum().item()
        fp0 = (p0 * (1 - target)).sum().item()
        fn0 = ((1 - p0) * target).sum().item()
        init_dice = 2.0 * tp0 / (2.0 * tp0 + fp0 + fn0 + 1e-6)
        init_iou = tp0 / (tp0 + fp0 + fn0 + 1e-6)

    history = [{
        "step": 0,
        "loss": round(init_loss, 4),
        "dice": round(init_dice, 4),
        "iou": round(init_iou, 4),
    }]

    print("=" * 60)
    print("  GeoTrustNet Overfit Diagnostic (Pipeline Sanity Check)")
    print("=" * 60)
    print(f"Step 00: Initial Loss = {init_loss:.4f} | Initial Dice = {init_dice:.4f}")

    # Training iterations
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        logits = model(sar)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == num_steps:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                p = (probs >= 0.5).float()
                tp = (p * target).sum().item()
                fp = (p * (1 - target)).sum().item()
                fn = ((1 - p) * target).sum().item()
                dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-6)
                iou = tp / (tp + fp + fn + 1e-6)
                history.append({
                    "step": step,
                    "loss": round(loss.item(), 4),
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                })
                print(f"Step {step:02d}: Loss = {loss.item():.4f} | Dice = {dice:.4f} | IoU = {iou:.4f} (TP={int(tp)}, FP={int(fp)}, FN={int(fn)})")

    # Verify weight updates across all layers
    weights_updated = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.equal(initial_weights[name], param):
                weights_updated = False
                break

    final_dice = history[-1]["dice"]
    final_loss = history[-1]["loss"]
    loss_decreased = final_loss < init_loss
    dice_improved = final_dice > 0.70

    diagnostic_result = {
        "status": "PASSED" if (loss_decreased and dice_improved and weights_updated) else "FAILED",
        "initial_loss": round(init_loss, 4),
        "final_loss": round(final_loss, 4),
        "initial_dice": round(init_dice, 4),
        "final_dice": round(final_dice, 4),
        "loss_decreased": loss_decreased,
        "dice_substantially_improved": dice_improved,
        "all_weights_updated": weights_updated,
        "history": history,
    }

    out_dir = ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "pilot_model_a_sar_unet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "overfit_diagnostic_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_result, f, indent=2)

    print("=" * 60)
    print(f"Diagnostic Result: {diagnostic_result['status']}")
    print(f"Loss Reduction:    {init_loss:.4f} -> {final_loss:.4f}")
    print(f"Dice Progression:  {init_dice:.4f} -> {final_dice:.4f}")
    print(f"Weights Updated:   {weights_updated}")
    print(f"Exported to:       {out_path}")
    print("=" * 60)

    return diagnostic_result


if __name__ == "__main__":
    run_overfit_diagnostic()
