"""Training engine, checkpoint manager, and experiment runner for GeoTrustNet baselines."""

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.geotrustnet.evaluation.metrics import compute_segmentation_metrics
from src.geotrustnet.training.losses import CombinedBCEDiceLoss


def seed_everything(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    """Safely obtain current git commit hash."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "unknown_or_no_git"


def extract_batch_tensors(
    batch: Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]],
    device: torch.device,
    modality: str = "sar",
    temporal_idx: Optional[int] = -1,
    is_temporal: bool = False,
    is_multimodal: bool = False,
    return_mask: bool = False,
) -> Union[Tuple[Any, torch.Tensor], Tuple[Any, torch.Tensor, Optional[torch.Tensor]]]:
    """Extract input features and target binary mask from batch.

    Args:
        batch: DataLoader output dictionary or tuple
        device: Target execution device
        modality: Modality key to extract if dictionary ('sar', 'optical', 'dem', 'multimodal')
        temporal_idx: Time slice index if modality tensor is 5D and is_temporal is False.
        is_temporal: If True, preserves the 5D temporal dimension [B, T, C, H, W].
        is_multimodal: If True, extracts all modalities (SAR, Optical, DEM) and availability flags.
        return_mask: If True, also returns temporal validity mask [B, T] or [B, T, 1, H, W].

    Returns:
        If return_mask is False: (x, y)
        If return_mask is True: (x, y, mask)
    """
    if is_multimodal or modality == "multimodal":
        if isinstance(batch, dict):
            raw_y = batch["mask"]
            y = raw_y.to(device).float()
            if y.ndim == 3:
                y = y.unsqueeze(1)

            multimodal_inputs = {
                "sar": batch["sar"].to(device).float() if "sar" in batch and batch["sar"] is not None else None,
                "optical": batch["optical"].to(device).float() if "optical" in batch and batch["optical"] is not None else None,
                "dem": batch["dem"].to(device).float() if "dem" in batch and batch["dem"] is not None else None,
                "has_sar": batch["has_sar"].to(device).bool() if "has_sar" in batch else None,
                "has_optical": batch["has_optical"].to(device).bool() if "has_optical" in batch else None,
                "has_dem": batch["has_dem"].to(device).bool() if "has_dem" in batch else None,
                "sar_quality": batch["sar_valid"].to(device).float() if "sar_valid" in batch and batch["sar_valid"] is not None else None,
                "optical_quality": batch["optical_valid"].to(device).float() if "optical_valid" in batch and batch["optical_valid"] is not None else None,
            }
            if return_mask:
                return multimodal_inputs, y, None
            return multimodal_inputs, y

    valid_mask = None
    if isinstance(batch, dict):
        raw_x = batch[modality]
        raw_y = batch["mask"]
        mask_key = f"{modality}_valid"
        if mask_key in batch:
            valid_mask = batch[mask_key].to(device)
    elif isinstance(batch, (tuple, list)):
        raw_x = batch[0]
        raw_y = batch[1]
        if len(batch) > 2:
            valid_mask = batch[2].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Move to target device
    x = raw_x.to(device).float()
    y = raw_y.to(device).float()

    # If x is 5D [B, T, C, H, W] and not in temporal mode, extract specified time slice
    if x.ndim == 5 and not is_temporal and temporal_idx is not None:
        t_len = x.shape[1]
        t_idx = temporal_idx if temporal_idx >= 0 else t_len + temporal_idx
        t_idx = max(0, min(t_len - 1, t_idx))
        x = x[:, t_idx, :, :, :]

    # Ensure y is 4D [B, 1, H, W]
    if y.ndim == 3:
        y = y.unsqueeze(1)

    if return_mask:
        return x, y, valid_mask
    return x, y


class GeoTrustNetTrainer:
    """Trainer for GeoTrustNet baseline segmentation models.

    Supports training/validation loops, early stopping, CosineAnnealing / ReduceLROnPlateau
    schedulers, checkpoint persistence, and comprehensive metric evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
        monitor_metric: str = "val_iou",
        monitor_mode: str = "max",
        patience: int = 10,
        modality: str = "sar",
        temporal_idx: Optional[int] = -1,
        is_temporal: bool = False,
        is_multimodal: bool = False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = (
            criterion
            if criterion is not None
            else CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        )
        self.optimizer = (
            optimizer
            if optimizer is not None
            else AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        )
        self.scheduler = scheduler

        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.modality = modality
        self.temporal_idx = temporal_idx
        self.is_temporal = is_temporal or (temporal_idx is None and not is_multimodal and modality != "multimodal")
        self.is_multimodal = is_multimodal or (modality == "multimodal")

        self.output_dir = Path(output_dir) if output_dir is not None else Path("artifacts/geotrustnet/checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = experiment_config or {}
        self.history: List[Dict[str, Any]] = []
        self.best_metric_val = -float("inf") if monitor_mode == "max" else float("inf")
        self.best_epoch = 0

    def _forward_model(self, x: Union[torch.Tensor, Dict[str, Any]], mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Call model handling dict inputs (multimodal), temporal inputs with masks, or standard tensors."""
        if isinstance(x, dict):
            import inspect
            sig = inspect.signature(self.model.forward)
            filtered_kwargs = {k: v for k, v in x.items() if k in sig.parameters}
            return self.model(**filtered_kwargs)
        if self.is_temporal and mask is not None:
            try:
                return self.model(x, valid_masks=mask)
            except TypeError:
                return self.model(x)
        return self.model(x)

    def train_epoch(self) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            x, y, mask = extract_batch_tensors(
                batch=batch,
                device=self.device,
                modality=self.modality,
                temporal_idx=self.temporal_idx,
                is_temporal=self.is_temporal,
                is_multimodal=self.is_multimodal,
                return_mask=True,
            )

            target_valid = None
            if isinstance(batch, dict) and "mask_valid" in batch and batch["mask_valid"] is not None:
                target_valid = batch["mask_valid"].to(self.device).bool()

            self.optimizer.zero_grad()
            logits = self._forward_model(x, mask)
            loss = self.criterion(logits, y, valid_mask=target_valid)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Tuple[float, Dict[str, Any]]:
        """Run validation or evaluation pass on given dataloader.

        Returns:
            Tuple of (average_loss, segmentation_metrics_dict)
        """
        loader = dataloader or self.val_loader
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_logits = []
        all_targets = []
        all_valid_masks = []

        for batch in loader:
            x, y, mask = extract_batch_tensors(
                batch=batch,
                device=self.device,
                modality=self.modality,
                temporal_idx=self.temporal_idx,
                is_temporal=self.is_temporal,
                is_multimodal=self.is_multimodal,
                return_mask=True,
            )

            target_valid = None
            if isinstance(batch, dict) and "mask_valid" in batch and batch["mask_valid"] is not None:
                target_valid = batch["mask_valid"].to(self.device).bool()

            logits = self._forward_model(x, mask)
            loss = self.criterion(logits, y, valid_mask=target_valid)

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())
            if target_valid is not None:
                all_valid_masks.append(target_valid.detach().cpu())
            else:
                all_valid_masks.append(torch.ones_like(y.detach().cpu(), dtype=torch.bool))

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if all_logits and all_targets:
            concat_logits = torch.cat(all_logits, dim=0)
            concat_targets = torch.cat(all_targets, dim=0)
            concat_valid = torch.cat(all_valid_masks, dim=0) if all_valid_masks else None
            metrics = compute_segmentation_metrics(concat_logits, concat_targets, valid_mask=concat_valid)
        else:
            metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

        return avg_loss, metrics

    def is_better(self, current: float) -> bool:
        """Check if current metric value improves on best observed so far."""
        if self.monitor_mode == "max":
            return current > self.best_metric_val
        return current < self.best_metric_val

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, Any],
    ) -> Path:
        """Save training checkpoint to disk."""
        checkpoint_path = self.output_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
            "config": self.config,
            "saved_at": time.time(),
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoint weights into model and return checkpoint metadata."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt

    def fit(self, epochs: int) -> Dict[str, Any]:
        """Execute full training loop with validation, early stopping, and checkpointing.

        Args:
            epochs: Maximum number of epochs to train

        Returns:
            Dictionary containing best metrics, history, and training summary
        """
        start_time = time.time()
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)

            # Retrieve current lr
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric_to_step = val_loss if "loss" in self.monitor_metric else val_metrics.get("iou", val_loss)
                    self.scheduler.step(metric_to_step)
                else:
                    self.scheduler.step()

            # Identify monitored value
            if self.monitor_metric == "val_loss":
                current_monitored = val_loss
            else:
                metric_key = self.monitor_metric.replace("val_", "")
                current_monitored = val_metrics.get(metric_key, 0.0)

            # Check improvement
            improved = self.is_better(current_monitored)
            if improved:
                self.best_metric_val = current_monitored
                self.best_epoch = epoch
                epochs_no_improve = 0
                self.save_checkpoint("best_model.pt", epoch, val_loss, val_metrics)
            else:
                epochs_no_improve += 1

            # Save latest checkpoint
            self.save_checkpoint("latest_checkpoint.pt", epoch, val_loss, val_metrics)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_iou": val_metrics.get("iou", 0.0),
                "val_dice": val_metrics.get("dice", 0.0),
                "val_precision": val_metrics.get("precision", 0.0),
                "val_recall": val_metrics.get("recall", 0.0),
                "val_pr_auc": val_metrics.get("pr_auc"),
                "learning_rate": current_lr,
                "duration_seconds": round(time.time() - epoch_start, 2),
                "improved": improved,
            }
            self.history.append(epoch_record)

            if epochs_no_improve >= self.patience:
                break

        total_duration = time.time() - start_time

        # Save training history JSON
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        summary = {
            "experiment_id": self.config.get("experiment_id", f"exp_{int(start_time)}"),
            "git_commit": get_git_commit(),
            "device": str(self.device),
            "epochs_trained": len(self.history),
            "best_epoch": self.best_epoch,
            "monitored_metric": self.monitor_metric,
            "best_metric_value": round(self.best_metric_val, 5),
            "total_duration_seconds": round(total_duration, 2),
            "config": self.config,
        }

        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
