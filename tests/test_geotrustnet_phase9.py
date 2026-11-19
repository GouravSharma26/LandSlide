"""Unit and integration tests for GeoTrustNet Phase 9 Diagnostics & Benchmark Readiness.

Verifies:
1. Deterministic hand-calculated metric identities (Dice, IoU, Precision, Recall).
2. CombinedBCEDiceLoss numerical stability under extreme logits and empty/full masks.
3. Gradient backpropagation and parameter weight updates.
4. Tiny-subset overfit diagnostic pipeline sanity check.
5. Validation threshold optimization and probability distribution calibration.
6. Geographic inventory isolation across all 11 Sen12Landslides regions.
7. Phase 9 benchmark execution plan configuration schema.
"""

import json
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.models.unet import UNet
from src.geotrustnet.training.losses import CombinedBCEDiceLoss, DiceLoss
from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics


def test_hand_calculated_metric_identities():
    """Verify pixel-level segmentation metric calculations against exact hand-computed values."""
    # Construct exact confusion matrix:
    # TP = 40, FP = 10, FN = 20, TN = 930 (total = 1000)
    targets = np.zeros(1000, dtype=np.float32)
    targets[:60] = 1.0  # 60 positive ground truth

    preds = np.zeros(1000, dtype=np.float32)
    preds[:40] = 1.0    # 40 true positives
    preds[60:70] = 1.0  # 10 false positives

    tp = float(np.sum(preds * targets))
    fp = float(np.sum(preds * (1 - targets)))
    fn = float(np.sum((1 - preds) * targets))
    tn = float(np.sum((1 - preds) * (1 - targets)))

    assert tp == 40.0
    assert fp == 10.0
    assert fn == 20.0
    assert tn == 930.0

    expected_dice = (2.0 * tp) / (2.0 * tp + fp + fn)  # 80 / 110 = 0.72727...
    expected_iou = tp / (tp + fp + fn)                  # 40 / 70  = 0.57142...
    expected_prec = tp / (tp + fp)                      # 40 / 50  = 0.8000
    expected_rec = tp / (tp + fn)                       # 40 / 60  = 0.6666...

    metrics = compute_segmentation_metrics(
        logits_or_probs=preds,
        targets=targets,
        threshold=0.5,
    )

    assert np.isclose(metrics["dice"], expected_dice, atol=1e-3)
    assert np.isclose(metrics["iou"], expected_iou, atol=1e-3)
    assert np.isclose(metrics["precision"], expected_prec, atol=1e-3)
    assert np.isclose(metrics["recall"], expected_rec, atol=1e-3)

    # Verify algebraic identity: Dice = 2*IoU / (1 + IoU)
    derived_dice = (2.0 * expected_iou) / (1.0 + expected_iou)
    assert np.isclose(derived_dice, expected_dice, atol=1e-6)


def test_loss_function_numerical_stability():
    """Verify loss function behavior with extreme logits, empty masks, and full masks."""
    loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-6)

    # 1. Normal case
    logits = torch.randn(2, 1, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > 0.9).float()
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss), "Loss must be finite"

    # 2. Extreme positive logits (+100) and negative logits (-100)
    extreme_logits = torch.tensor([[-100.0, 100.0], [100.0, -100.0]]).view(1, 1, 2, 2)
    loss_extreme = loss_fn(extreme_logits, targets[:1, :, :2, :2])
    assert torch.isfinite(loss_extreme), "Loss with extreme logits must be finite without overflow"

    # 3. Empty ground truth (all zeros)
    empty_targets = torch.zeros(1, 1, 32, 32)
    loss_empty = loss_fn(logits[:1], empty_targets)
    assert torch.isfinite(loss_empty), "Loss with empty target must be finite"

    # 4. Full ground truth (all ones)
    full_targets = torch.ones(1, 1, 32, 32)
    loss_full = loss_fn(logits[:1], full_targets)
    assert torch.isfinite(loss_full), "Loss with full target must be finite"


def test_weight_update_gradient_flow():
    """Verify that forward-backward pass updates UNet parameter weights."""
    model = UNet(in_channels=2, out_channels=1, features=[16, 32, 64], bilinear=True)
    initial_param = model.inc.double_conv[0].weight.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = CombinedBCEDiceLoss()

    x = torch.randn(2, 2, 32, 32)
    y = (torch.rand(2, 1, 32, 32) > 0.8).float()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    updated_param = model.inc.double_conv[0].weight
    assert not torch.equal(initial_param, updated_param), "Parameter weights must update after optimizer step"


def test_overfit_diagnostic_sanity():
    """Verify that the overfit diagnostic script executes and demonstrates loss decrease."""
    from scripts.run_overfit_diagnostic import run_overfit_diagnostic

    res = run_overfit_diagnostic(num_steps=15, lr=0.01)
    assert res["status"] == "PASSED"
    assert res["loss_decreased"] is True
    assert res["all_weights_updated"] is True
    assert res["final_dice"] > res["initial_dice"]


def test_validation_threshold_optimizer():
    """Verify that threshold analysis sweeps thresholds and identifies calibrated operating point."""
    from scripts.analyze_validation_thresholds import run_threshold_analysis

    res = run_threshold_analysis()
    assert "default_threshold_0_50" in res
    assert "optimal_threshold" in res
    # At default threshold, predicted prevalence was artificially 100%
    assert res["default_threshold_0_50"]["predicted_prevalence_pct"] > 95.0
    # Optimal threshold improves Dice substantially
    assert res["optimal_threshold"]["dice"] > res["default_threshold_0_50"]["dice"]
    assert res["optimal_threshold"]["predicted_prevalence_pct"] < 10.0


def test_disjoint_inventory_split_isolation():
    """Verify that all 11 Sen12Landslides inventories are partitioned strictly disjointly."""
    manifest_path = ROOT_DIR / "data" / "manifests" / "sen12landslides_disjoint_manifest.csv"
    assert manifest_path.exists(), "sen12landslides_disjoint_manifest.csv must exist"

    df = pd.read_csv(manifest_path)
    inventories = sorted(df["region_id"].unique().tolist())
    assert len(inventories) == 11, f"Must have exactly 11 inventories, found {len(inventories)}"

    train_set = set(df[df["split"] == "train"]["region_id"])
    val_set = set(df[df["split"] == "val"]["region_id"])
    test_set = set(df[df["split"] == "test"]["region_id"])

    assert len(train_set.intersection(val_set)) == 0, "Train and Val must share 0 inventories"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test must share 0 inventories"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test must share 0 inventories"
    assert train_set.union(val_set).union(test_set) == set(inventories), "All inventories must be allocated"


def test_benchmark_execution_plan_schema():
    """Verify phase9_benchmark_execution_plan.json contains required compute budgets and experiment pipeline."""
    plan_path = ROOT_DIR / "configs" / "training" / "phase9_benchmark_execution_plan.json"
    assert plan_path.exists(), "phase9_benchmark_execution_plan.json must exist"

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    assert "execution_platforms" in plan
    assert "hyperparameter_standards" in plan
    assert "compute_budgets" in plan
    assert "experiment_pipeline" in plan
    assert len(plan["experiment_pipeline"]) >= 4

    exp_ids = [e["experiment_id"] for e in plan["experiment_pipeline"]]
    assert "benchmark_model_a_sar_unet_v2" in exp_ids
    assert "benchmark_model_c_multimodal_fusion_v2" in exp_ids
    assert "benchmark_model_d_geotrustnet_v2" in exp_ids
