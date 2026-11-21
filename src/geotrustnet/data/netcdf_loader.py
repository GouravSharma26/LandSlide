"""NetCDF raster reader and contract validator for Sen12Landslides.

Supports loading official Sen12Landslides NetCDF (.nc) files for:
- Sentinel-1 SAR (VV, VH backscatter, DEM elevation/slope)
- Sentinel-2 Optical (10 bands: B02-B12, SCL cloud classification)
- Target Landslide Mask (binary {0, 1} with decoupled nodata=255 validity mask)
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
    clear_scl_classes: Tuple[int, ...] = (4, 5, 6),
    temporal_policy: str = "fixed",
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
        normalize: Whether to apply official z-score normalization (default False; dataset owns normalization)
        nodata_value: Sensor nodata sentinel (-9999.0)
        mask_nodata_value: Label nodata sentinel (255)
        clear_scl_classes: Tuple of ESA SCL classes considered clear/valid optical surface observations.
            Default is (4, 5, 6) for Vegetation, Not-vegetated (bare soil/rock), and Water.
            Class 7 ('unclassified') is excluded by default due to classification ambiguity.
            Class 2 ('dark area pixels') is excluded by default due to topographic shadow SNR loss.
        temporal_policy: Policy for temporal sequence selection. Supported:
            'fixed' (default): slice to fixed_time_steps (or pad with false validity if shorter).
            'full': retain complete available multi-temporal sequence (no padding/truncation).
            'pre_event': slice strictly to pre-event acquisitions (using pre_post_dates metadata).
            'post_event': slice strictly to post-event acquisitions.
            'chronological': chronological sequence up to fixed_time_steps.

    Returns:
        PatchSample containing co-registered tensors, validity masks, and metadata.
    """
    sar_file = Path(sar_path)
    if not sar_file.exists():
        raise FileNotFoundError(f"SAR NetCDF file not found: {sar_file}")

    # Standard Sen12Landslides normalization parameters (Höhn et al., 2025)
    vv_mean, vv_std = -9.3099, 3.4868
    vh_mean, vh_std = -15.8861, 4.1814

    # 1. Load SAR Dataset
    with xr.open_dataset(sar_file) as ds_sar:
        var_names = [str(v) for v in ds_sar.data_vars]

        # Case-insensitive discovery for VV and VH
        vv_var = next((v for v in var_names if v.upper() == "VV"), None)
        vh_var = next((v for v in var_names if v.upper() == "VH"), None)

        if vv_var is None or vh_var is None:
            raise KeyError(
                f"SAR NetCDF must contain 'VV' and 'VH' variables. Found variables: {var_names}"
            )

        vv_data = ds_sar[vv_var].values  # Shape may be [T, H, W] or [H, W]
        vh_data = ds_sar[vh_var].values

        if vv_data.ndim == 2:
            vv_data = vv_data[np.newaxis, ...]
            vh_data = vh_data[np.newaxis, ...]

        T_sar = vv_data.shape[0]

        # Check dimensions
        if vv_data.shape[-2:] != (height, width):
            # Allow flexible spatial resolution but validate
            h_actual, w_actual = vv_data.shape[-2:]
            height, width = h_actual, w_actual

        # Extract DEM (elevation, slope) if present in SAR file
        elev_var = next((v for v in var_names if v.upper() in ["DEM", "ELEV", "ELEVATION"]), None)
        slope_var = next((v for v in var_names if "SLOPE" in v.upper()), None)

        if elev_var is not None:
            elev_data = ds_sar[elev_var].values
            if elev_data.ndim == 3:
                elev_data = elev_data[0]
            if slope_var is not None:
                slope_data = ds_sar[slope_var].values
                if slope_data.ndim == 3:
                    slope_data = slope_data[0]
            else:
                # Physically derive topographic slope in degrees from 10m DEM elevation
                gy, gx = np.gradient(elev_data.astype(np.float32), 10.0, 10.0)
                slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
                slope_data = np.degrees(slope_rad).astype(np.float32)
            has_dem = True
        else:
            elev_data = None
            slope_data = None
            has_dem = False

        # Extract MASK if embedded in SAR file
        mask_var = next((v for v in var_names if v.upper() in ["MASK", "LABEL", "LABELS", "LANDSLIDE"]), None)
        mask_data = None
        if mask_var is not None:
            mask_data = ds_sar[mask_var].values
            if mask_data.ndim == 3:
                mask_data = mask_data[0]

        # Extract actual coordinates without fabricating defaults
        timestamps = []
        if "time" in ds_sar.coords:
            try:
                timestamps = [str(t) for t in ds_sar["time"].values]
            except Exception:
                timestamps = []

        bounds = (0.0, 0.0, 0.0, 0.0)
        has_geo = False
        if "x" in ds_sar.coords and "y" in ds_sar.coords:
            try:
                x_vals = ds_sar["x"].values
                y_vals = ds_sar["y"].values
                bounds = (float(np.min(x_vals)), float(np.min(y_vals)), float(np.max(x_vals)), float(np.max(y_vals)))
                has_geo = True
            except Exception:
                pass

        crs_str = str(ds_sar.attrs.get("crs", ds_sar.attrs.get("spatial_ref", "EPSG:4326")))

        # Extract pre-event and post-event split metadata if present
        pre_post_raw = ds_sar.attrs.get("pre_post_dates", None)
        pre_idx, post_idx = None, None
        if pre_post_raw is not None:
            if isinstance(pre_post_raw, str):
                import ast
                try:
                    parsed = ast.literal_eval(pre_post_raw)
                except Exception:
                    parsed = None
            else:
                parsed = pre_post_raw
            if isinstance(parsed, tuple) and len(parsed) > 0:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                pre_idx = parsed.get("pre")
                post_idx = parsed.get("post")

    # Process SAR Tensors
    sar_valid_np = (vv_data != nodata_value) & (vh_data != nodata_value) & (~np.isnan(vv_data)) & (~np.isnan(vh_data))
    vv_clean = np.where(sar_valid_np, vv_data, 0.0).astype(np.float32)
    vh_clean = np.where(sar_valid_np, vh_data, 0.0).astype(np.float32)

    # Optional normalization (dataset is default owner)
    is_normalized = False
    if normalize:
        vv_clean = np.where(sar_valid_np, (vv_clean - vv_mean) / vv_std, 0.0)
        vh_clean = np.where(sar_valid_np, (vh_clean - vh_mean) / vh_std, 0.0)
        is_normalized = True

    sar_stack = np.stack([vv_clean, vh_clean], axis=1)  # [T, 2, H, W]
    sar_valid_stack = sar_valid_np[:, np.newaxis, :, :]  # [T, 1, H, W]

    # Slicing according to temporal_policy
    if temporal_policy == "full":
        pass
    elif temporal_policy == "pre_event":
        if pre_idx is not None and pre_idx < T_sar:
            sar_stack = sar_stack[: pre_idx + 1]
            sar_valid_stack = sar_valid_stack[: pre_idx + 1]
    elif temporal_policy == "post_event":
        if post_idx is not None and post_idx < T_sar:
            sar_stack = sar_stack[post_idx :]
            sar_valid_stack = sar_valid_stack[post_idx :]
    elif temporal_policy in ["fixed", "chronological"]:
        if T_sar < fixed_time_steps:
            pad_t = fixed_time_steps - T_sar
            sar_pad = np.zeros((pad_t, 2, height, width), dtype=np.float32)
            sar_valid_pad = np.zeros((pad_t, 1, height, width), dtype=bool)
            sar_stack = np.concatenate([sar_stack, sar_pad], axis=0)
            sar_valid_stack = np.concatenate([sar_valid_stack, sar_valid_pad], axis=0)
        else:
            sar_stack = sar_stack[:fixed_time_steps]
            sar_valid_stack = sar_valid_stack[:fixed_time_steps]
    else:
        raise ValueError(f"Unsupported temporal_policy: {temporal_policy}")

    sar_tensor = torch.from_numpy(sar_stack).float()
    sar_valid_tensor = torch.from_numpy(sar_valid_stack).bool()

    # Process DEM
    if has_dem and elev_data is not None and slope_data is not None:
        dem_valid = (elev_data != nodata_value) & (slope_data != nodata_value) & (~np.isnan(elev_data))
        elev_clean = np.where(dem_valid, elev_data, 0.0).astype(np.float32)
        slope_clean = np.where(dem_valid, slope_data, 0.0).astype(np.float32)
        dem_stack = np.stack([elev_clean, slope_clean], axis=0)  # [2, H, W]
        dem_tensor = torch.from_numpy(dem_stack).float()
    else:
        # Never silently replace missing DEM with a plausible-looking zero tensor
        dem_tensor = None
        has_dem = False

    # 2. Load Optical if available
    optical_tensor = None
    optical_valid_tensor = None
    has_optical = False
    scl_available = False

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

                    # Check for SCL cloud classification
                    scl_var = next((v for v in opt_vars if v.upper() in ["SCL", "SCENE_CLASSIFICATION"]), None)
                    if scl_var is not None:
                        scl_data = ds_opt[scl_var].values
                        if scl_data.ndim == 2:
                            scl_data = scl_data[np.newaxis, ...]
                        # Clean classes parameterized by clear_scl_classes (default 4=vegetation, 5=bare soil/rock, 6=water)
                        # Excludes class 7 (unclassified) and class 2 (dark area) by default
                        scl_clean = np.isin(scl_data, list(clear_scl_classes))
                        if scl_clean.ndim == 3:
                            scl_clean = scl_clean[:, np.newaxis, :, :]
                        # Broadcast across channels
                        opt_valid = scl_clean & (opt_stack != nodata_value) & (~np.isnan(opt_stack))
                        scl_available = True
                    else:
                        opt_valid = (opt_stack != nodata_value) & (~np.isnan(opt_stack))

                    opt_stack = np.where(opt_valid, opt_stack, 0.0).astype(np.float32)
                    if normalize:
                        opt_stack = opt_stack / 10000.0  # Scale reflectance to [0, 1]
                        is_normalized = True

                    opt_valid_agg = np.all(opt_valid, axis=1, keepdims=True)  # [T, 1, H, W]

                    if temporal_policy == "full":
                        pass
                    elif temporal_policy == "pre_event":
                        if pre_idx is not None and pre_idx < T_opt:
                            opt_stack = opt_stack[: pre_idx + 1]
                            opt_valid_agg = opt_valid_agg[: pre_idx + 1]
                    elif temporal_policy == "post_event":
                        if post_idx is not None and post_idx < T_opt:
                            opt_stack = opt_stack[post_idx :]
                            opt_valid_agg = opt_valid_agg[post_idx :]
                    elif temporal_policy in ["fixed", "chronological"]:
                        if T_opt < fixed_time_steps:
                            pad_t = fixed_time_steps - T_opt
                            opt_pad = np.zeros((pad_t, 10, height, width), dtype=np.float32)
                            opt_valid_pad = np.zeros((pad_t, 1, height, width), dtype=bool)
                            opt_stack = np.concatenate([opt_stack, opt_pad], axis=0)
                            opt_valid_agg = np.concatenate([opt_valid_agg, opt_valid_pad], axis=0)
                        else:
                            opt_stack = opt_stack[:fixed_time_steps]
                            opt_valid_agg = opt_valid_agg[:fixed_time_steps]
                    else:
                        raise ValueError(f"Unsupported temporal_policy: {temporal_policy}")

                    optical_tensor = torch.from_numpy(opt_stack).float()
                    optical_valid_tensor = torch.from_numpy(opt_valid_agg).bool()
                    has_optical = True

                # Check if DEM is in optical file if not already loaded
                if not has_dem:
                    elev_var_opt = next((v for v in opt_vars if v.upper() in ["DEM", "ELEV", "ELEVATION"]), None)
                    slope_var_opt = next((v for v in opt_vars if "SLOPE" in v.upper()), None)
                    if elev_var_opt is not None:
                        elev_data = ds_opt[elev_var_opt].values
                        if elev_data.ndim == 3:
                            elev_data = elev_data[0]
                        if slope_var_opt is not None:
                            slope_data = ds_opt[slope_var_opt].values[0] if ds_opt[slope_var_opt].values.ndim == 3 else ds_opt[slope_var_opt].values
                        else:
                            gy, gx = np.gradient(elev_data.astype(np.float32), 10.0, 10.0)
                            slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
                            slope_data = np.degrees(slope_rad).astype(np.float32)
                        dem_valid = (elev_data != nodata_value) & (~np.isnan(elev_data))
                        elev_clean = np.where(dem_valid, elev_data, 0.0).astype(np.float32)
                        slope_clean = np.where(dem_valid, slope_data, 0.0).astype(np.float32)
                        dem_stack = np.stack([elev_clean, slope_clean], axis=0)
                        dem_tensor = torch.from_numpy(dem_stack).float()
                        has_dem = True

                # Check if MASK is in optical file
                if mask_data is None:
                    mask_v = next((v for v in opt_vars if v.upper() in ["MASK", "LABEL", "LABELS", "LANDSLIDE"]), None)
                    if mask_v is not None:
                        mask_data = ds_opt[mask_v].values
                        if mask_data.ndim == 3:
                            mask_data = mask_data[0]

    # Process Target Mask and Decoupled Target Validity
    if mask_data is not None:
        # Check nodata pixels (mask_nodata_value = 255)
        mask_valid_np = (mask_data != mask_nodata_value) & (~np.isnan(mask_data))
        if mask_data.ndim == 2:
            mask_data = mask_data[np.newaxis, ...]
            mask_valid_np = mask_valid_np[np.newaxis, ...]
        # Binary target: 1 if landslide, 0 otherwise (only meaningful on valid pixels)
        target_np = np.where(mask_valid_np, (mask_data > 0).astype(np.float32), 0.0)
        mask_tensor = torch.from_numpy(target_np).float()
        mask_valid_tensor = torch.from_numpy(mask_valid_np).bool()
    else:
        # If mask is not found, return empty mask with all-False validity
        mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)
        mask_valid_tensor = torch.zeros((1, height, width), dtype=torch.bool)

    metadata = PatchMetadata(
        patch_id=sample_id,
        region_id=region_id,
        event_id=event_id,
        event_date="2020-01-01T00:00:00Z",
        timestamps=timestamps,
        crs=crs_str,
        bounds=bounds,
        pixel_size_m=10.0,
        height=height,
        width=width,
        extra_metadata={
            "nodata_values": {"sar": nodata_value, "optical": nodata_value, "mask": mask_nodata_value},
            "is_normalized": is_normalized,
            "has_geo_coords": has_geo,
            "scl_available": scl_available,
            "temporal_policy": temporal_policy,
            "effective_timesteps": sar_stack.shape[0],
            "pre_post_indices": {"pre": pre_idx, "post": post_idx},
            "scl_policy": {
                "clear_classes": list(clear_scl_classes),
                "excluded_classes_rationale": {
                    "7": "Unclassified: ESA Sen2Cor low-confidence classification excluded to prevent cloud/border noise.",
                    "2": "Dark area pixels: Topographic cast shadows with low optical SNR excluded so model relies on SAR.",
                },
            },
        },
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
        mask_valid=mask_valid_tensor,
    )
