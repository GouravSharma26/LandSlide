"""Calibration, uncertainty metrics, and selective prediction analysis for GeoTrustNet.

Implements Brier Score, Expected Calibration Error (ECE), reliability diagram generation,
and selective prediction curves (error rejection analysis).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def compute_brier_score(
    probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute Brier Score (Mean Squared Error of probability forecasts).

    Args:
        probs: Predicted probabilities in [0, 1]
        targets: Binary ground truth {0, 1}

    Returns:
        Scalar Brier score in [0, 1] (lower is better calibrated)
    """
    if isinstance(probs, torch.Tensor):
        p = probs.detach().cpu().float().numpy().ravel()
    else:
        p = np.asarray(probs, dtype=np.float32).ravel()

    if isinstance(targets, torch.Tensor):
        y = targets.detach().cpu().float().numpy().ravel()
    else:
        y = np.asarray(targets, dtype=np.float32).ravel()

    return float(np.mean((p - y) ** 2))


def compute_expected_calibration_error(
    probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """Calculate Expected Calibration Error (ECE) and reliability diagram bin statistics.

    Args:
        probs: Predicted probabilities in [0, 1]
        targets: Binary ground truth {0, 1}
        num_bins: Number of confidence bins (default 10)

    Returns:
        Dictionary containing:
          - 'ece': Expected Calibration Error
          - 'mce': Maximum Calibration Error
          - 'bins': List of dicts per bin (confidence, accuracy, count, lower, upper)
    """
    if isinstance(probs, torch.Tensor):
        p = probs.detach().cpu().float().numpy().ravel()
    else:
        p = np.asarray(probs, dtype=np.float32).ravel()

    if isinstance(targets, torch.Tensor):
        y = targets.detach().cpu().float().numpy().ravel()
    else:
        y = np.asarray(targets, dtype=np.float32).ravel()

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    total_samples = len(p)

    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (p >= bin_lower) & (p <= bin_upper)
        else:
            in_bin = (p >= bin_lower) & (p < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_conf = float(np.mean(p[in_bin]))
            bin_acc = float(np.mean(y[in_bin]))
            cal_diff = abs(bin_acc - bin_conf)

            ece += (bin_count / total_samples) * cal_diff
            mce = max(mce, cal_diff)

            bin_details.append({
                "bin_index": i,
                "bin_lower": round(float(bin_lower), 3),
                "bin_upper": round(float(bin_upper), 3),
                "bin_center": round(float((bin_lower + bin_upper) / 2.0), 3),
                "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
                "count": bin_count,
                "diff": round(cal_diff, 4),
            })
        else:
            bin_details.append({
                "bin_index": i,
                "bin_lower": round(float(bin_lower), 3),
                "bin_upper": round(float(bin_upper), 3),
                "bin_center": round(float((bin_lower + bin_upper) / 2.0), 3),
                "confidence": 0.0,
                "accuracy": 0.0,
                "count": 0,
                "diff": 0.0,
            })

    return {
        "ece": round(float(ece), 5),
        "mce": round(float(mce), 5),
        "num_bins": num_bins,
        "total_samples": total_samples,
        "bins": bin_details,
    }


def compute_selective_prediction_curve(
    probs: Union[torch.Tensor, np.ndarray],
    uncertainties: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    percentiles: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Analyze selective prediction: evaluate metrics as uncertain pixels are filtered out.

    Args:
        probs: Predicted probabilities [B, 1, H, W] or flattened
        uncertainties: Epistemic variance or entropy matching probs
        targets: Binary ground truth {0, 1}
        percentiles: Coverage percentages to evaluate (default [100, 90, 80, 70, 60, 50])
        threshold: Decision threshold for binarizing probabilities (default 0.5)

    Returns:
        List of dicts containing retained coverage, IoU, Dice, Precision, and Recall
    """
    if percentiles is None:
        percentiles = [100, 90, 80, 70, 60, 50, 40, 30]

    if isinstance(probs, torch.Tensor):
        p = probs.detach().cpu().float().numpy().ravel()
    else:
        p = np.asarray(probs, dtype=np.float32).ravel()

    if isinstance(uncertainties, torch.Tensor):
        u = uncertainties.detach().cpu().float().numpy().ravel()
    else:
        u = np.asarray(uncertainties, dtype=np.float32).ravel()

    if isinstance(targets, torch.Tensor):
        y = targets.detach().cpu().float().numpy().ravel()
    else:
        y = np.asarray(targets, dtype=np.float32).ravel()

    results = []

    for pct in percentiles:
        if pct == 100:
            retained_mask = np.ones(len(p), dtype=bool)
        else:
            cutoff_uncertainty = np.percentile(u, pct)
            retained_mask = u <= cutoff_uncertainty

        retained_count = int(np.sum(retained_mask))
        coverage = float(retained_count / len(p)) if len(p) > 0 else 0.0

        if retained_count > 0:
            p_ret = p[retained_mask]
            y_ret = y[retained_mask]

            preds = (p_ret >= threshold).astype(int)
            tp = int(np.sum((preds == 1) & (y_ret == 1)))
            fp = int(np.sum((preds == 1) & (y_ret == 0)))
            fn = int(np.sum((preds == 0) & (y_ret == 1)))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        else:
            prec, rec, f1, iou = 0.0, 0.0, 0.0, 0.0

        results.append({
            "target_coverage_pct": pct,
            "actual_coverage": round(coverage, 4),
            "retained_pixels": retained_count,
            "iou": round(iou, 5),
            "dice": round(f1, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
        })

    return results
