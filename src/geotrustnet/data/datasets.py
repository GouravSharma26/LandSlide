"""PyTorch Dataset implementations for GeoTrustNet multimodal spatio-temporal patches."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.geotrustnet.data.contracts import PatchSample


class GeoTrustNetPatchDataset(Dataset):
    """Multimodal spatio-temporal landslide patch dataset.

    Handles Sentinel-1 SAR, Sentinel-2 optical, DEM topography, and binary landslide masks.
    Supports missing modality masking and variable time steps.
    """

    def __init__(
        self,
        manifest_df: Union[pd.DataFrame, str, Path],
        split: Optional[str] = None,
        data_root: Optional[Path] = None,
        fixed_time_steps: int = 5,
        height: int = 128,
        width: int = 128,
        normalize: bool = False,
        clear_scl_classes: Tuple[int, ...] = (4, 5, 6),
        temporal_policy: str = "fixed",
    ):
        self.data_root = Path(data_root) if data_root is not None else Path(".")
        self.split = split
        self.fixed_time_steps = fixed_time_steps
        self.height = height
        self.width = width
        self.normalize = normalize
        self.clear_scl_classes = clear_scl_classes
        self.temporal_policy = temporal_policy

        if isinstance(manifest_df, (str, Path)):
            manifest_df = pd.read_csv(manifest_df)

        # Standard Sen12Landslides normalization parameters
        self.vv_mean = -9.3099
        self.vv_std = 3.4868
        self.vh_mean = -15.8861
        self.vh_std = 4.1814

        if split is not None and "split" in manifest_df.columns:
            self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        else:
            self.df = manifest_df.reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}' in manifest")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        file_path = self.data_root / row["file_path"]

        if file_path.suffix == ".pt":
            sample: PatchSample = torch.load(file_path, weights_only=False)
        elif file_path.suffix in [".nc", ".netcdf", ".h5"]:
            from src.geotrustnet.data.netcdf_loader import load_sen12landslides_netcdf
            optical_path = self.data_root / row["s2_path"] if ("s2_path" in row and pd.notna(row["s2_path"])) else None
            # load_sen12landslides_netcdf called with normalize=False; dataset is the sole normalization owner
            sample: PatchSample = load_sen12landslides_netcdf(
                sar_path=file_path,
                optical_path=optical_path,
                sample_id=str(row.get("sample_id", f"sample_{idx}")),
                region_id=str(row.get("region_id", "unknown_region")),
                event_id=str(row.get("event_id", "unknown_event")),
                fixed_time_steps=self.fixed_time_steps,
                height=self.height,
                width=self.width,
                normalize=False,
                clear_scl_classes=self.clear_scl_classes,
                temporal_policy=self.temporal_policy,
            )
        else:
            raise NotImplementedError(f"Unsupported raw file extension: {file_path.suffix}")

        if self.temporal_policy in ["full", "pre_event", "post_event"] and sample.sar is not None:
            T = sample.sar.shape[0]
        else:
            T = self.fixed_time_steps
        H, W = self.height, self.width
        is_already_normalized = bool(
            getattr(sample, "metadata", None)
            and sample.metadata.extra_metadata.get("is_normalized", False)
        )

        # 1. Optical (10 channels)
        if sample.optical_available and sample.optical is not None:
            optical = sample.optical[:T]
            optical_valid = (
                sample.optical_valid_mask[:T]
                if sample.optical_valid_mask is not None
                else torch.ones((optical.shape[0], 1, H, W), dtype=torch.bool)
            )
            # Pad if shorter than fixed_time_steps
            if optical.shape[0] < T:
                pad_t = T - optical.shape[0]
                optical = torch.cat([optical, torch.zeros((pad_t, 10, H, W))], dim=0)
                optical_valid = torch.cat([optical_valid, torch.zeros((pad_t, 1, H, W), dtype=torch.bool)], dim=0)

            # Optical surface reflectance scale to [0, 1] if normalizing
            if self.normalize and not is_already_normalized:
                optical = torch.where(optical_valid.expand_as(optical), optical / 10000.0, torch.zeros_like(optical))

            has_opt = True
        else:
            optical = torch.zeros((T, 10, H, W), dtype=torch.float32)
            optical_valid = torch.zeros((T, 1, H, W), dtype=torch.bool)
            has_opt = False

        # 2. SAR (2 channels: VV, VH)
        if sample.sar_available and sample.sar is not None:
            sar = sample.sar[:T]
            sar_valid = (
                sample.sar_valid_mask[:T]
                if sample.sar_valid_mask is not None
                else torch.ones((sar.shape[0], 1, H, W), dtype=torch.bool)
            )
            if sar.shape[0] < T:
                pad_t = T - sar.shape[0]
                sar = torch.cat([sar, torch.zeros((pad_t, 2, H, W))], dim=0)
                sar_valid = torch.cat([sar_valid, torch.zeros((pad_t, 1, H, W), dtype=torch.bool)], dim=0)

            if self.normalize and not is_already_normalized:
                # Z-score normalize VV (dim 1, index 0) and VH (dim 1, index 1) strictly on valid pixels
                sar_norm = sar.clone()
                v_mask = sar_valid.squeeze(2) if sar_valid.ndim == 5 else sar_valid
                # [T, 1, H, W] expands to [T, 2, H, W]
                v_exp = sar_valid.expand_as(sar_norm)
                vv_norm = torch.where(v_exp[:, 0], (sar_norm[:, 0] - self.vv_mean) / self.vv_std, torch.zeros_like(sar_norm[:, 0]))
                vh_norm = torch.where(v_exp[:, 1], (sar_norm[:, 1] - self.vh_mean) / self.vh_std, torch.zeros_like(sar_norm[:, 1]))
                sar = torch.stack([vv_norm, vh_norm], dim=1)

            has_sar = True
        else:
            sar = torch.zeros((T, 2, H, W), dtype=torch.float32)
            sar_valid = torch.zeros((T, 1, H, W), dtype=torch.bool)
            has_sar = False

        # 3. DEM (2 channels: elevation, slope)
        if sample.dem_available and sample.dem is not None:
            dem = sample.dem
            has_dem = True
        else:
            dem = torch.zeros((2, H, W), dtype=torch.float32)
            has_dem = False

        # 4. Target Mask (1 channel) with decoupled validity
        raw_mask = sample.mask if sample.mask is not None else torch.zeros((1, H, W), dtype=torch.float32)
        if raw_mask.ndim == 2:
            raw_mask = raw_mask.unsqueeze(0)

        if hasattr(sample, "mask_valid") and sample.mask_valid is not None:
            mask_valid = sample.mask_valid
            if mask_valid.ndim == 2:
                mask_valid = mask_valid.unsqueeze(0)
        else:
            # Fallback for legacy fixtures: identify sentinel 255
            mask_valid = (raw_mask != 255) & (~torch.isnan(raw_mask))

        mask_clean = torch.where(mask_valid, (raw_mask > 0).float(), torch.zeros_like(raw_mask))

        return {
            "sample_id": row["sample_id"],
            "region_id": row["region_id"],
            "event_id": row["event_id"],
            "optical": optical.float(),
            "sar": sar.float(),
            "dem": dem.float(),
            "mask": mask_clean.float(),
            "mask_valid": mask_valid.bool(),
            "optical_valid": optical_valid,
            "sar_valid": sar_valid,
            "has_optical": torch.tensor(has_opt, dtype=torch.bool),
            "has_sar": torch.tensor(has_sar, dtype=torch.bool),
            "has_dem": torch.tensor(has_dem, dtype=torch.bool),
        }


def collate_geotrustnet_patches(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader batching."""
    return {
        "sample_ids": [b["sample_id"] for b in batch],
        "region_ids": [b["region_id"] for b in batch],
        "event_ids": [b["event_id"] for b in batch],
        "optical": torch.stack([b["optical"] for b in batch], dim=0),           # [B, T, 10, H, W]
        "sar": torch.stack([b["sar"] for b in batch], dim=0),                   # [B, T, 2, H, W]
        "dem": torch.stack([b["dem"] for b in batch], dim=0),                   # [B, 2, H, W]
        "mask": torch.stack([b["mask"] for b in batch], dim=0),                 # [B, 1, H, W]
        "mask_valid": torch.stack([b["mask_valid"] for b in batch], dim=0) if "mask_valid" in batch[0] else torch.ones_like(torch.stack([b["mask"] for b in batch], dim=0), dtype=torch.bool),
        "optical_valid": torch.stack([b["optical_valid"] for b in batch], dim=0),# [B, T, 1, H, W]
        "sar_valid": torch.stack([b["sar_valid"] for b in batch], dim=0),       # [B, T, 1, H, W]
        "has_optical": torch.stack([b["has_optical"] for b in batch], dim=0),   # [B]
        "has_sar": torch.stack([b["has_sar"] for b in batch], dim=0),           # [B]
        "has_dem": torch.stack([b["has_dem"] for b in batch], dim=0),           # [B]
    }
