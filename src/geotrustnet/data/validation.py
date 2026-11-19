"""Validation utilities for geospatial raster patches, temporal sequences, and labels."""

import datetime
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import torch
import pandas as pd


def validate_patch_spatial(
    array: np.ndarray,
    expected_shape: Tuple[int, int] = (128, 128),
    allow_nan: bool = False,
) -> Tuple[bool, List[str]]:
    """Validates spatial dimensions and numerical integrity of a 2D/3D/4D raster array."""
    errors = []
    spatial_dims = array.shape[-2:]
    if spatial_dims != expected_shape:
        errors.append(f"Spatial dimensions {spatial_dims} do not match expected {expected_shape}")

    if not allow_nan and np.isnan(array).any():
        errors.append("Array contains unmasked NaN values")

    if np.isinf(array).any():
        errors.append("Array contains Infinite (Inf) values")

    return len(errors) == 0, errors


def validate_temporal_order(
    timestamps: List[str],
    date_format: str = "%Y-%m-%dT%H:%M:%SZ",
    strict_ascending: bool = True,
) -> Tuple[bool, List[str]]:
    """Validates that a sequence of ISO8601 timestamps is strictly monotonically increasing."""
    errors = []
    if not timestamps:
        return False, ["Timestamp list is empty"]

    parsed_dates = []
    for i, ts in enumerate(timestamps):
        try:
            # Handle standard ISO formats with or without Z
            clean_ts = ts.replace("Z", "+00:00") if "Z" in ts else ts
            dt = datetime.datetime.fromisoformat(clean_ts)
            parsed_dates.append(dt)
        except Exception as e:
            errors.append(f"Invalid timestamp at index {i} '{ts}': {e}")
            return False, errors

    if strict_ascending:
        for i in range(len(parsed_dates) - 1):
            if parsed_dates[i] >= parsed_dates[i + 1]:
                errors.append(
                    f"Temporal inversion at index {i}->{i+1}: "
                    f"'{timestamps[i]}' is not strictly before '{timestamps[i+1]}'"
                )

    return len(errors) == 0, errors


def validate_future_data_leakage(
    input_timestamps: List[str],
    cutoff_datetime: str,
) -> Tuple[bool, List[str]]:
    """Verifies that none of the input observation timestamps exceed the prediction cutoff."""
    errors = []
    try:
        clean_cutoff = cutoff_datetime.replace("Z", "+00:00") if "Z" in cutoff_datetime else cutoff_datetime
        cutoff = datetime.datetime.fromisoformat(clean_cutoff)
    except Exception as e:
        return False, [f"Invalid cutoff datetime '{cutoff_datetime}': {e}"]

    for i, ts in enumerate(input_timestamps):
        clean_ts = ts.replace("Z", "+00:00") if "Z" in ts else ts
        dt = datetime.datetime.fromisoformat(clean_ts)
        if dt > cutoff:
            errors.append(
                f"Future data leakage detected at index {i}: "
                f"timestamp '{ts}' is after prediction cutoff '{cutoff_datetime}'"
            )

    return len(errors) == 0, errors


def validate_label_alignment(
    mask: np.ndarray,
    expected_spatial_shape: Tuple[int, int] = (128, 128),
) -> Tuple[bool, List[str]]:
    """Verifies that the ground-truth mask is binary {0, 1} and matches spatial dimensions."""
    errors = []
    if mask.shape[-2:] != expected_spatial_shape:
        errors.append(f"Mask spatial shape {mask.shape[-2:]} does not match {expected_spatial_shape}")

    unique_vals = np.unique(mask)
    for v in unique_vals:
        if v not in (0, 1):
            errors.append(f"Non-binary ground-truth value found in mask: {v}. Must be 0 (stable) or 1 (landslide)")

    return len(errors) == 0, errors


def validate_split_disjointness(
    df: pd.DataFrame,
    split_col: str = "split",
    group_cols: List[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Mathematically verifies that Train, Val, and Test splits share zero overlapping groups/samples."""
    if group_cols is None:
        group_cols = ["region_id", "event_id"]

    report: Dict[str, Any] = {
        "is_disjoint": True,
        "sample_counts": {},
        "group_overlaps": {},
        "violations": [],
    }

    splits = ["train", "val", "test"]
    for s in splits:
        report["sample_counts"][s] = int((df[split_col] == s).sum())

    # Check overlaps across pairs
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in pairs:
        s1_df = df[df[split_col] == s1]
        s2_df = df[df[split_col] == s2]

        for g_col in group_cols:
            if g_col in df.columns:
                set1 = set(s1_df[g_col].dropna().unique())
                set2 = set(s2_df[g_col].dropna().unique())
                intersection = set1.intersection(set2)
                pair_key = f"{s1}_vs_{s2}_{g_col}"
                report["group_overlaps"][pair_key] = len(intersection)
                if len(intersection) > 0:
                    report["is_disjoint"] = False
                    report["violations"].append(
                        f"Data leakage: {len(intersection)} overlapping {g_col} values between {s1} and {s2}: {list(intersection)[:5]}"
                    )

    # Check sample_id uniqueness
    if "sample_id" in df.columns:
        dups = df["sample_id"].duplicated().sum()
        if dups > 0:
            report["is_disjoint"] = False
            report["violations"].append(f"Duplicate sample_id count in manifest: {dups}")

    return report["is_disjoint"], report
