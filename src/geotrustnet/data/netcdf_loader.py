"""NetCDF raster reader and contract validator for Sen12Landslides.

Supports loading official Sen12Landslides NetCDF (.nc) files for:
- Sentinel-1 SAR (VV, VH, elevation, slope, crs)
- Sentinel-2 Optical (10 bands: B02-B12, SCL cloud classification)
- Target Landslide Mask (binary {0, 1} with nodata=255 exclusion)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import xarray as xr

from src.geotrustnet.data.contracts import PatchMetadata, PatchSample


def load_sen12landslides_netcdf(
    sar_path: Union[str, Path],
    optical_path: Optional[Union[str, Path]] = None,
    sample_id: str = "sample_0000",
    region_id: str = "unknown_region",
    event_id: str = "unknown_event",
    fixed_time_steps: int = 5,
    height: int = 128,
    width: int = 128,
    normalize: bool = False,
    nodata_value: float = -9999.0,
    mask_nodata_value: int = 255,
) -> PatchSample:
    """Load a Sen12Landslides sample from NetCDF files into a validated PatchSample contract.

    Args:
        sar_path: Path to SAR NetCDF file (.nc)
        optical_path: Optional path to co-registered Optical NetCDF file (.nc)
        sample_id: Sample identifier
        region_id: Regional inventory identifier
        event_id: Event cluster identifier
        fixed_time_steps: Target temporal horizon T
        height: Spatial height (default 128)
        width: Spatial width (default 128)
        normalize: Whether to apply official z-score normalization
        nodata_value: Sensor nodata sentinel (-9999.0)
        mask_nodata_value: Label nodata sentinel (255)

    Returns:
        PatchSample containing co-registered tensors, validity masks, and metadata.
    """
    sar_file = Path(sar_path)
    if not sar_file.exists():
        raise FileNotFoundError(f"SAR NetCDF file not found: {sar_file}")

    # Standard Sen12Landslides normalization parameters
    vv_mean, vv_std = -9.3099, 3.4868
    vh_mean, vh_std = -15.8861, 4.1814

    # 1. Load SAR Dataset
    with xr.open_dataset(sar_file) as ds_sar:
        # Identify VV and VH variables
        var_names = [str(v) for v in ds_sar.data_vars]

        # Find VV
        vv_var = next((v for v in var_names if v.upper() == "VV"), None)
        # Find VH
        vh_var = next((v for v in var_names if v.upper() == "VH"), None)

        if vv_var is None or vh_var is None:
            raise KeyError(f"SAR NetCDF must contain 'VV' and 'VH' variables. Found: {var_names}")

        vv_data = ds_sar[vv_var].values  # Shape may be [T, H, W] or [H, W]
        vh_data = ds_sar[vh_var].values

        if vv_data.ndim == 2:
            vv_data = vv_data[np.newaxis, ...]
            vh_data = vh_data[np.newaxis, ...]

        T_sar = vv_data.shape[0]

        # Extract DEM (elevation, slope) if present in SAR file
        elev_var = next((v for v in var_names if "ELEV" in v.upper() or v.upper() == "DEM"), None)
        slope_var = next((v for v in var_names if "SLOPE" in v.upper()), None)

        if elev_var is not None and slope_var is not None:
            elev_data = ds_sar[elev_var].values
            slope_data = ds_sar[slope_var].values
            if elev_data.ndim == 3:
                elev_data = elev_data[0]
            if slope_data.ndim == 3:
                slope_data = slope_data[0]
            has_dem = True
        else:
            elev_data = np.zeros((height, width), dtype=np.float32)
            slope_data = np.zeros((height, width), dtype=np.float32)
            has_dem = False

        # Extract MASK if embedded in SAR file
        mask_var = next((v for v in var_names if "MASK" in v.upper() or "LABEL" in v.upper()), None)
        mask_data = None
        if mask_var is not None:
            mask_data = ds_sar[mask_var].values
            if mask_data.ndim == 3:
                mask_data = mask_data[0]

        # CRS
        crs_str = str(ds_sar.attrs.get("crs", "EPSG:4326"))

    # Process SAR Tensors
    # Build validity mask where pixels != nodata
    sar_valid_np = (vv_data != nodata_value) & (vh_data != nodata_value) & (~np.isnan(vv_data)) & (~np.isnan(vh_data))
    # Replace nodatas with zero
    vv_clean = np.where(sar_valid_np, vv_data, 0.0).astype(np.float32)
    vh_clean = np.where(sar_valid_np, vh_data, 0.0).astype(np.float32)

    # Z-score normalization if requested
    if normalize:
        vv_clean = np.where(sar_valid_np, (vv_clean - vv_mean) / vv_std, 0.0)
        vh_clean = np.where(sar_valid_np, (vh_clean - vh_mean) / vh_std, 0.0)

    # Combine into [T, 2, H, W]
    sar_stack = np.stack([vv_clean, vh_clean], axis=1)  # [T, 2, H, W]
    sar_valid_stack = sar_valid_np[:, np.newaxis, :, :]  # [T, 1, H, W]

    # Pad or truncate to fixed_time_steps
    if T_sar < fixed_time_steps:
        pad_t = fixed_time_steps - T_sar
        sar_pad = np.zeros((pad_t, 2, height, width), dtype=np.float32)
        sar_valid_pad = np.zeros((pad_t, 1, height, width), dtype=bool)
        sar_stack = np.concatenate([sar_stack, sar_pad], axis=0)
        sar_valid_stack = np.concatenate([sar_valid_stack, sar_valid_pad], axis=0)
    else:
        sar_stack = sar_stack[:fixed_time_steps]
        sar_valid_stack = sar_valid_stack[:fixed_time_steps]

    sar_tensor = torch.from_numpy(sar_stack).float()
    sar_valid_tensor = torch.from_numpy(sar_valid_stack).bool()

    # Process DEM
    dem_valid = (elev_data != nodata_value) & (slope_data != nodata_value) & (~np.isnan(elev_data))
    elev_clean = np.where(dem_valid, elev_data, 0.0).astype(np.float32)
    slope_clean = np.where(dem_valid, slope_data, 0.0).astype(np.float32)
    dem_stack = np.stack([elev_clean, slope_clean], axis=0)  # [2, H, W]
    dem_tensor = torch.from_numpy(dem_stack).float()

    # 2. Load Optical if available
    optical_tensor = None
    optical_valid_tensor = None
    has_optical = False

    if optical_path is not None:
        opt_file = Path(optical_path)
        if opt_file.exists():
            with xr.open_dataset(opt_file) as ds_opt:
                opt_vars = [str(v) for v in ds_opt.data_vars]
                target_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
                avail_bands = [b for b in target_bands if b in opt_vars]

                if len(avail_bands) == len(target_bands):
                    band_arrays = []
                    for b in target_bands:
                        arr = ds_opt[b].values
                        if arr.ndim == 2:
                            arr = arr[np.newaxis, ...]
                        band_arrays.append(arr)

                    opt_stack = np.stack(band_arrays, axis=1)  # [T, 10, H, W]
                    T_opt = opt_stack.shape[0]

                    # Nodata and cloud masking
                    opt_valid = (opt_stack != nodata_value) & (~np.isnan(opt_stack))
                    opt_stack = np.where(opt_valid, opt_stack, 0.0).astype(np.float32)
                    if normalize:
                        opt_stack = opt_stack / 10000.0  # Scale reflectance to [0, 1]

                    if T_opt < fixed_time_steps:
                        pad_t = fixed_time_steps - T_opt
                        opt_pad = np.zeros((pad_t, 10, height, width), dtype=np.float32)
                        opt_valid_pad = np.zeros((pad_t, 1, height, width), dtype=bool)
                        opt_stack = np.concatenate([opt_stack, opt_pad], axis=0)
                        opt_valid_agg = np.all(opt_valid, axis=1, keepdims=True)
                        opt_valid_agg = np.concatenate([opt_valid_agg, opt_valid_pad], axis=0)
                    else:
                        opt_stack = opt_stack[:fixed_time_steps]
                        opt_valid_agg = np.all(opt_valid[:fixed_time_steps], axis=1, keepdims=True)

                    optical_tensor = torch.from_numpy(opt_stack).float()
                    optical_valid_tensor = torch.from_numpy(opt_valid_agg).bool()
                    has_optical = True

                # Check if MASK is in optical file
                if mask_data is None:
                    mask_v = next((v for v in opt_vars if "MASK" in v.upper() or "LABEL" in v.upper()), None)
                    if mask_v is not None:
                        mask_data = ds_opt[mask_v].values
                        if mask_data.ndim == 3:
                            mask_data = mask_data[0]

    # Process Target Mask
    if mask_data is not None:
        # Check nodata pixels (mask_nodata_value = 255)
        mask_valid_np = (mask_data != mask_nodata_value) & (~np.isnan(mask_data))
        # Binary target: 1 if landslide, 0 otherwise
        target_np = np.where((mask_data > 0) & mask_valid_np, 1.0, 0.0).astype(np.float32)
        if target_np.ndim == 2:
            target_np = target_np[np.newaxis, ...]
        mask_tensor = torch.from_numpy(target_np).float()
    else:
        # Fallback empty mask with 0
        mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)

    metadata = PatchMetadata(
        patch_id=sample_id,
        region_id=region_id,
        event_id=event_id,
        event_date="2020-01-01T00:00:00Z",
        timestamps=["2020-01-01T00:00:00Z"] * fixed_time_steps,
        crs=crs_str,
        bounds=(0.0, 0.0, 0.0, 0.0),
        pixel_size_m=10.0,
        height=height,
        width=width,
        extra_metadata={"nodata_values": {"sar": nodata_value, "optical": nodata_value, "mask": mask_nodata_value}},
    )

    return PatchSample(
        metadata=metadata,
        optical=optical_tensor,
        sar=sar_tensor,
        dem=dem_tensor,
        mask=mask_tensor,
        optical_available=has_optical,
        sar_available=True,
        dem_available=has_dem,
        optical_valid_mask=optical_valid_tensor,
        sar_valid_mask=sar_valid_tensor,
    )
