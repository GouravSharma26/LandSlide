"""Data contracts and schemas for GeoTrustNet.

Defines typed dataclasses and validation interfaces for multimodal spatio-temporal
landslide observation patches.
"""

import os
import sys
from pathlib import Path

# On Windows, resolve torch DLL directory search path before importing torch
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


@dataclass
class PatchMetadata:
    """Metadata describing a single spatio-temporal observation patch."""
    patch_id: str
    region_id: str
    event_id: str
    event_date: str
    timestamps: List[str]
    crs: str = "EPSG:4326"
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # min_lon, min_lat, max_lon, max_lat
    pixel_size_m: float = 10.0
    height: int = 128
    width: int = 128
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchSample:
    """A complete multimodal patch sample passed to GeoTrustNet."""
    metadata: PatchMetadata
    # Tensors:
    # optical: shape [T, C_opt, H, W] or None if unavailable
    optical: Optional[torch.Tensor] = None
    # sar: shape [T, C_sar, H, W] or None if unavailable
    sar: Optional[torch.Tensor] = None
    # dem: shape [C_dem, H, W] or None if unavailable
    dem: Optional[torch.Tensor] = None
    # mask: shape [1, H, W] binary {0, 1}
    mask: Optional[torch.Tensor] = None
    # Validity / availability flags
    optical_available: bool = True
    sar_available: bool = True
    dem_available: bool = True
    # Pixel-level quality / cloud masks:
    # optical_valid_mask: shape [T, 1, H, W] boolean (True where not cloudy/nodata)
    optical_valid_mask: Optional[torch.Tensor] = None
    # sar_valid_mask: shape [T, 1, H, W] boolean
    sar_valid_mask: Optional[torch.Tensor] = None
    # Target mask validity: shape [1, H, W] boolean (True where target in {0, 1}, False for nodata=255)
    mask_valid: Optional[torch.Tensor] = None

    def validate(self) -> Tuple[bool, List[str]]:
        """Validates internal shape and contract consistency."""
        errors = []
        H, W = self.metadata.height, self.metadata.width

        # Check mask
        if self.mask is not None:
            if self.mask.ndim != 3 or self.mask.shape[1:] != (H, W):
                errors.append(f"Mask shape {self.mask.shape} does not match (1, {H}, {W})")

        # Check mask_valid
        if self.mask_valid is not None:
            if self.mask_valid.ndim != 3 or self.mask_valid.shape[1:] != (H, W):
                errors.append(f"Mask validity shape {self.mask_valid.shape} does not match (1, {H}, {W})")
            unique_vals = torch.unique(self.mask).cpu().numpy()
            if not np.all(np.isin(unique_vals, [0, 1])):
                errors.append(f"Mask values must be binary {{0, 1}}, got: {unique_vals}")

        # Check optical
        if self.optical_available and self.optical is not None:
            if self.optical.ndim != 4 or self.optical.shape[2:] != (H, W):
                errors.append(f"Optical shape {self.optical.shape} does not match (T, C, {H}, {W})")
            if self.optical_valid_mask is not None:
                if self.optical_valid_mask.shape[0] != self.optical.shape[0] or self.optical_valid_mask.shape[2:] != (H, W):
                    errors.append(f"Optical valid mask shape {self.optical_valid_mask.shape} mismatch with optical")

        # Check SAR
        if self.sar_available and self.sar is not None:
            if self.sar.ndim != 4 or self.sar.shape[2:] != (H, W):
                errors.append(f"SAR shape {self.sar.shape} does not match (T, C, {H}, {W})")

        # Check DEM
        if self.dem_available and self.dem is not None:
            if self.dem.ndim != 3 or self.dem.shape[1:] != (H, W):
                errors.append(f"DEM shape {self.dem.shape} does not match (C, {H}, {W})")

        # Ensure at least one modality is present
        if not (self.optical_available or self.sar_available or self.dem_available):
            errors.append("At least one modality must be marked available")

        return len(errors) == 0, errors


@dataclass
class ManifestEntry:
    """A single record in a dataset manifest."""
    sample_id: str
    region_id: str
    event_id: str
    event_date: str
    split: str  # 'train', 'val', 'test'
    has_optical: bool
    has_sar: bool
    has_dem: bool
    has_mask: bool
    n_time_steps: int
    landslide_pixel_ratio: float
    file_path: str
    checksum_sha256: str
