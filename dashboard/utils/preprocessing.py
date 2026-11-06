"""Preprocessing and filtering helpers for Himalayan Landslide Intelligence."""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np

from dashboard.config import ROI_BOUNDS


def filter_by_bounds(
    df: pd.DataFrame,
    lat_min: float = ROI_BOUNDS["lat_min"],
    lat_max: float = ROI_BOUNDS["lat_max"],
    lon_min: float = ROI_BOUNDS["lon_min"],
    lon_max: float = ROI_BOUNDS["lon_max"],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Filters dataframe to records within specified bounding coordinates."""
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return df

    mask = (
        (df[lat_col] >= lat_min)
        & (df[lat_col] <= lat_max)
        & (df[lon_col] >= lon_min)
        & (df[lon_col] <= lon_max)
    )
    return df[mask].copy()


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    date_col: str = "event_date",
) -> pd.DataFrame:
    """Filters dataframe by date range."""
    if df.empty or date_col not in df.columns:
        return df

    filtered = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(filtered[date_col]):
        filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")

    if start_date is not None:
        filtered = filtered[filtered[date_col] >= start_date]
    if end_date is not None:
        filtered = filtered[filtered[date_col] <= end_date]

    return filtered


def filter_by_categories(
    df: pd.DataFrame,
    column: str,
    selected_values: List[str],
) -> pd.DataFrame:
    """Filters dataframe by membership in a list of categorical values."""
    if df.empty or column not in df.columns or not selected_values:
        return df

    return df[df[column].isin(selected_values)].copy()


def get_country_from_coords(lat: float, lon: float) -> str:
    """Simple bounding box heuristic for regional Himalayan jurisdiction labelling."""
    if 26.0 <= lat <= 31.0 and 80.0 <= lon <= 88.2:
        return "Nepal"
    elif 26.8 <= lat <= 28.3 and 88.8 <= lon <= 92.2:
        return "Bhutan"
    elif 31.0 <= lat <= 35.0 and 74.0 <= lon <= 78.5:
        return "India (J&K / Ladakh / HP)"
    elif 29.0 <= lat <= 31.5 and 77.5 <= lon <= 81.0:
        return "India (Uttarakhand)"
    elif 26.5 <= lat <= 28.5 and 88.0 <= lon <= 97.0:
        return "India (Northeast / Sikkim)"
    elif 33.0 <= lat <= 35.0 and 74.0 <= lon <= 77.0:
        return "Pakistan / PoK"
    elif lat >= 31.0 and lon >= 79.0:
        return "China (Tibet Autonomous Region)"
    return "Himalayan Belt"


def assign_regional_subzones(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a readable regional subzone tag based on coordinates."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    df_copy = df.copy()
    df_copy["region_subzone"] = [
        get_country_from_coords(lat, lon)
        for lat, lon in zip(df_copy["latitude"], df_copy["longitude"])
    ]
    return df_copy

