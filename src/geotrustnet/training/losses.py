"""Loss functions for binary landslide segmentation.

Implements DiceLoss and CombinedBCEDiceLoss with class-imbalance weighting
and numerical stability guarantees.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.

    Operates directly on unnormalized model logits by internally applying sigmoid.

    Args:
        smooth: Smoothing coefficient to avoid division by zero (default 1e-6).
        reduction: 'mean' or 'none'.
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Dice loss with optional nodata masking.

        Args:
            logits: Predicted raw logits of shape [B, 1, H, W] or [B, H, W]
            targets: Binary ground truth {0, 1} matching logits shape
            valid_mask: Optional boolean or {0, 1} mask where True indicates valid pixels.
                        If None, pixels where targets is 0 or 1 are considered valid.

        Returns:
            Scalar Dice loss (or tensor if reduction='none')
        """
        if logits.shape != targets.shape:
            targets = targets.reshape_as(logits)

        # Build or align valid_mask
        if valid_mask is not None:
            if valid_mask.shape != targets.shape:
                valid_mask = valid_mask.reshape_as(targets)
            valid_flat = valid_mask.reshape(valid_mask.size(0), -1).float()
        else:
            # Exclude sentinel nodata (e.g. 255 or -9999)
            valid_flat = ((targets == 0) | (targets == 1)).reshape(targets.size(0), -1).float()

        probs = torch.sigmoid(logits)

        # Flatten per sample and apply validity weighting [B, -1]
        probs_flat = probs.reshape(probs.size(0), -1) * valid_flat
        targets_flat = targets.reshape(targets.size(0), -1).float() * valid_flat

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        dice_coeff = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice_coeff

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedBCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy with Logits and Soft Dice Loss.

    loss = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss

    Args:
        bce_weight: Multiplier for BCE loss (default 0.5).
        dice_weight: Multiplier for Dice loss (default 0.5).
        pos_weight: Optional positive-class weight for BCE to handle landslide pixel sparsity.
        smooth: Smoothing term for Dice calculation.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1e-6,
    ):
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        pos_weight_tensor = (
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None
        )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute combined loss with nodata exclusion.

        Args:
            logits: Predicted raw logits [B, 1, H, W]
            targets: Binary ground truth [B, 1, H, W]
            valid_mask: Optional boolean or {0, 1} mask where True indicates valid pixels.
                        Pixels with nodata (e.g. 255) are excluded from loss computation.
            return_dict: If True, returns (total_loss, dict of component losses)

        Returns:
            Scalar total loss (or tuple with loss details)
        """
        if logits.shape != targets.shape:
            targets = targets.reshape_as(logits)

        # Build validity boolean mask: only include valid pixels where targets in {0, 1}
        if valid_mask is not None:
            if valid_mask.shape != targets.shape:
                valid_mask = valid_mask.reshape_as(targets)
            valid_b = valid_mask.bool() & ((targets == 0) | (targets == 1)) & (~torch.isnan(targets))
        else:
            valid_b = ((targets == 0) | (targets == 1)) & (~torch.isnan(targets))

        # Move pos_weight to same device as logits if needed
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        if valid_b.any():
            valid_logits = logits[valid_b]
            valid_targets = targets[valid_b].float()
            bce_val = F.binary_cross_entropy_with_logits(
                valid_logits,
                valid_targets,
                pos_weight=self.bce.pos_weight,
            )
        else:
            bce_val = torch.tensor(0.0, device=logits.device, requires_grad=True)

        dice_val = self.dice(logits, targets, valid_mask=valid_b)

        total_loss = self.bce_weight * bce_val + self.dice_weight * dice_val

        if return_dict:
            metrics = {
                "loss_total": float(total_loss.detach().item()),
                "loss_bce": float(bce_val.detach().item()),
                "loss_dice": float(dice_val.detach().item()),
            }
            return total_loss, metrics

        return total_loss
