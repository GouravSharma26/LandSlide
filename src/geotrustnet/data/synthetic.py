"""Synthetic fixture generator for fast, reproducible CPU testing without external data downloads."""

import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.geotrustnet.data.contracts import PatchMetadata, PatchSample


def create_synthetic_patch(
    sample_id: str,
    region_id: str,
    event_id: str,
    height: int = 128,
    width: int = 128,
    time_steps: int = 5,
    has_optical: bool = True,
    has_sar: bool = True,
    has_dem: bool = True,
    seed: int = 42,
    has_landslide: bool = True,
) -> PatchSample:
    """Creates a deterministic synthetic multimodal patch sample for CPU unit testing."""
    rng = np.random.RandomState(seed + hash(sample_id) % 10000)

    # Base timestamps (e.g. 5 dates separated by 6 days)
    base_date = datetime.datetime(2020, 7, 15, 10, 0, 0)
    timestamps = [
        (base_date + datetime.timedelta(days=i * 6)).isoformat() + "Z"
        for i in range(time_steps)
    ]
    event_date = timestamps[time_steps // 2]

    metadata = PatchMetadata(
        patch_id=sample_id,
        region_id=region_id,
        event_id=event_id,
        event_date=event_date,
        timestamps=timestamps,
        crs="EPSG:4326",
        bounds=(78.0, 30.0, 78.1, 30.1),
        pixel_size_m=10.0,
        height=height,
        width=width,
    )

    # 1. Optical: shape [T, 10, H, W], surface reflectance [0, 6000]
    optical_tensor = None
    optical_valid = None
    if has_optical:
        # Base reflectance + spatial gradients
        opt_data = rng.uniform(500, 3500, size=(time_steps, 10, height, width)).astype(np.float32)
        # Cloud simulation: occasionally mask random pixels
        cloud_mask = (rng.uniform(0, 1, size=(time_steps, 1, height, width)) > 0.85).astype(bool)
        optical_tensor = torch.from_numpy(opt_data)
        optical_valid = torch.from_numpy(~cloud_mask)

    # 2. SAR: shape [T, 2, H, W] in decibels [-30, 2]
    sar_tensor = None
    sar_valid = None
    if has_sar:
        sar_vv = rng.normal(-10.0, 2.5, size=(time_steps, 1, height, width)).astype(np.float32)
        sar_vh = rng.normal(-16.0, 3.0, size=(time_steps, 1, height, width)).astype(np.float32)
        sar_data = np.concatenate([sar_vv, sar_vh], axis=1)
        sar_tensor = torch.from_numpy(sar_data)
        sar_valid = torch.ones((time_steps, 1, height, width), dtype=torch.bool)

    # 3. DEM: shape [2, H, W] (elevation in meters, slope in degrees)
    dem_tensor = None
    if has_dem:
        elev = rng.uniform(800.0, 2800.0, size=(1, height, width)).astype(np.float32)
        slope = rng.uniform(5.0, 42.0, size=(1, height, width)).astype(np.float32)
        dem_data = np.concatenate([elev, slope], axis=0)
        dem_tensor = torch.from_numpy(dem_data)

    # 4. Binary Landslide Mask: shape [1, H, W]
    mask_data = np.zeros((1, height, width), dtype=np.float32)
    if has_landslide:
        # Synthetic landslide scar: small spatial cluster of 1s in center
        cy, cx = height // 2, width // 2
        radius = rng.randint(6, 18)
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((y - cy)**2 + (x - cx)**2)
        mask_data[0, dist_from_center <= radius] = 1.0

        # If landslide occurred, simulate backscatter drop after event date in SAR VH
        if has_sar:
            post_idx = time_steps // 2 + 1
            sar_tensor[post_idx:, 1:2, dist_from_center <= radius] -= 3.5

    mask_tensor = torch.from_numpy(mask_data)
    mask_valid_tensor = torch.ones_like(mask_tensor, dtype=torch.bool)

    return PatchSample(
        metadata=metadata,
        optical=optical_tensor,
        sar=sar_tensor,
        dem=dem_tensor,
        mask=mask_tensor,
        optical_available=has_optical,
        sar_available=has_sar,
        dem_available=has_dem,
        optical_valid_mask=optical_valid,
        sar_valid_mask=sar_valid,
        mask_valid=mask_valid_tensor,
    )


def generate_synthetic_dataset(
    output_dir: Path,
    n_regions: int = 4,
    events_per_region: int = 3,
    patches_per_event: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a small synthetic multi-patch dataset on disk and returns its manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    patch_idx = 0
    for r in range(n_regions):
        region_id = f"REG_{r+1:02d}"
        for e in range(events_per_region):
            event_id = f"EVT_{r+1:02d}_{e+1:02d}"
            for p in range(patches_per_event):
                patch_idx += 1
                sample_id = f"SMP_{patch_idx:04d}"

                # Simulate modality availability variations
                has_optical = (patch_idx % 7 != 0)  # occasional optical dropout
                has_sar = True
                has_dem = True

                sample = create_synthetic_patch(
                    sample_id=sample_id,
                    region_id=region_id,
                    event_id=event_id,
                    has_optical=has_optical,
                    has_sar=has_sar,
                    has_dem=has_dem,
                    seed=seed + patch_idx,
                )

                # Save sample as .pt file for fast deterministic loading
                file_path = output_dir / f"{sample_id}.pt"
                torch.save(sample, file_path)

                # Compute landslide pixel ratio
                ls_ratio = float(sample.mask.sum() / sample.mask.numel()) if sample.mask is not None else 0.0

                manifest_rows.append({
                    "sample_id": sample_id,
                    "region_id": region_id,
                    "event_id": event_id,
                    "event_date": sample.metadata.event_date,
                    "has_optical": has_optical,
                    "has_sar": has_sar,
                    "has_dem": has_dem,
                    "has_mask": True,
                    "n_time_steps": len(sample.metadata.timestamps),
                    "landslide_pixel_ratio": ls_ratio,
                    "file_path": str(file_path.relative_to(output_dir.parent.parent)),
                })

    df = pd.DataFrame(manifest_rows)
    return df
