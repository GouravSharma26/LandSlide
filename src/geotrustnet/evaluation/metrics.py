"""Evaluation metrics for landslide binary segmentation.

Computes precision, recall, F1 / Dice, IoU (Jaccard), PR-AUC, ROC-AUC,
confusion matrix counts, and class prevalence with rigorous edge-case
and empty-mask handling.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


def compute_segmentation_metrics(
    logits_or_probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    compute_curve_auc: bool = True,
) -> Dict[str, Any]:
    """Calculate comprehensive binary segmentation metrics.

    Args:
        logits_or_probs: Model predictions. If values outside [0, 1] are detected,
                         sigmoid is applied automatically.
        targets: Binary ground truth {0, 1} matching spatial shape.
        threshold: Classification threshold for binarizing probabilities (default 0.5).
        compute_curve_auc: Whether to compute pixel-level PR-AUC and ROC-AUC.

    Returns:
        Dictionary of metrics:
          - precision
          - recall
          - f1 (Dice)
          - dice
          - iou (Jaccard)
          - tp, fp, tn, fn
          - total_pixels
          - positive_prevalence
          - pr_auc (optional)
          - roc_auc (optional)
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(logits_or_probs, torch.Tensor):
        probs_np = logits_or_probs.detach().cpu().float().numpy()
    else:
        probs_np = np.asarray(logits_or_probs, dtype=np.float32)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.asarray(targets)

    # Apply sigmoid if raw logits were passed (any value < 0 or > 1)
    if np.any(probs_np < 0.0) or np.any(probs_np > 1.0):
        probs_np = 1.0 / (1.0 + np.exp(-np.clip(probs_np, -50.0, 50.0)))

    # Flatten arrays
    probs_flat = probs_np.ravel()
    targets_flat = targets_np.ravel().astype(np.int64)

    if probs_flat.shape[0] != targets_flat.shape[0]:
        raise ValueError(
            f"Shape mismatch: predictions have {probs_flat.shape[0]} elements, "
            f"targets have {targets_flat.shape[0]} elements"
        )

    # Thresholded predictions
    preds_binary = (probs_flat >= threshold).astype(np.int64)

    # Confusion matrix elements
    tp = int(np.sum((preds_binary == 1) & (targets_flat == 1)))
    fp = int(np.sum((preds_binary == 1) & (targets_flat == 0)))
    tn = int(np.sum((preds_binary == 0) & (targets_flat == 0)))
    fn = int(np.sum((preds_binary == 0) & (targets_flat == 1)))

    total_pixels = int(tp + fp + tn + fn)
    total_positives = int(tp + fn)
    total_predicted = int(tp + fp)
    prevalence = float(total_positives / total_pixels) if total_pixels > 0 else 0.0

    # Handle edge cases (e.g. empty masks)
    if total_positives == 0:
        # Ground truth has zero landslide pixels
        if total_predicted == 0:
            # Model correctly predicted empty mask
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            iou = 1.0
        else:
            # Model predicted landslide pixels where none exist (false alarms)
            precision = 0.0
            recall = 1.0  # Or undefined; convention 0.0 or 1.0; 0.0 for precision
            f1 = 0.0
            iou = 0.0
    else:
        # Ground truth has landslide pixels
        precision = float(tp / total_predicted) if total_predicted > 0 else 0.0
        recall = float(tp / total_positives) if total_positives > 0 else 0.0
        if precision + recall > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0

        denom_iou = tp + fp + fn
        iou = float(tp / denom_iou) if denom_iou > 0 else 0.0

    dice = f1  # Dice is mathematically identical to F1 for binary segmentation

    # Curve AUC calculations
    pr_auc = None
    roc_auc = None
    if compute_curve_auc and total_pixels > 0:
        # Check if both classes are present in ground truth
        unique_targets = np.unique(targets_flat)
        if len(unique_targets) > 1:
            try:
                prec_curve, rec_curve, _ = precision_recall_curve(targets_flat, probs_flat)
                pr_auc = float(auc(rec_curve, prec_curve))
            except Exception:
                pr_auc = None

            try:
                roc_auc = float(roc_auc_score(targets_flat, probs_flat))
            except Exception:
                roc_auc = None
        else:
            # Only single class in targets
            pr_auc = 1.0 if total_positives == 0 and total_predicted == 0 else 0.0
            roc_auc = None

    total_negatives = int(tn + fp)
    specificity = float(tn / total_negatives) if total_negatives > 0 else 1.0
    predicted_prevalence = float(total_predicted / total_pixels) if total_pixels > 0 else 0.0

    return {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "specificity": round(specificity, 5),
        "f1": round(f1, 5),
        "dice": round(dice, 5),
        "iou": round(iou, 5),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_pixels": total_pixels,
        "positive_prevalence": round(prevalence, 6),
        "predicted_prevalence": round(predicted_prevalence, 6),
        "pr_auc": round(pr_auc, 5) if pr_auc is not None else None,
        "roc_auc": round(roc_auc, 5) if roc_auc is not None else None,
    }
