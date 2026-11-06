"""Data validation and integrity checking utilities for Himalayan Landslide Intelligence."""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from dashboard.config import ROI_BOUNDS


def validate_geographic_bounds(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    bounds: Dict[str, float] = ROI_BOUNDS,
) -> Dict[str, Any]:
    """Verifies that coordinates fall strictly within the specified ROI bounds."""
    if df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return {
            "valid": False,
            "total_records": 0,
            "in_bounds_count": 0,
            "out_of_bounds_count": 0,
            "lat_range": (None, None),
            "lon_range": (None, None),
        }

    total = len(df)
    in_bounds = df[
        (df[lat_col] >= bounds["lat_min"])
        & (df[lat_col] <= bounds["lat_max"])
        & (df[lon_col] >= bounds["lon_min"])
        & (df[lon_col] <= bounds["lon_max"])
    ]
    in_count = len(in_bounds)
    out_count = total - in_count

    return {
        "valid": out_count == 0,
        "total_records": total,
        "in_bounds_count": in_count,
        "out_of_bounds_count": out_count,
        "lat_range": (float(df[lat_col].min()), float(df[lat_col].max())),
        "lon_range": (float(df[lon_col].min()), float(df[lon_col].max())),
    }


def validate_schema(
    df: pd.DataFrame, required_columns: List[str]
) -> Tuple[bool, List[str]]:
    """Checks whether a DataFrame contains all required columns."""
    if df.empty:
        return False, required_columns

    missing = [c for c in required_columns if c not in df.columns]
    return len(missing) == 0, missing


def calculate_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generates an honest, transparent missingness breakdown by column."""
    if df.empty:
        return pd.DataFrame()

    total = len(df)
    null_counts = df.isnull().sum()
    pcts = (null_counts / total) * 100.0

    summary = pd.DataFrame({
        "column": df.columns,
        "data_type": [str(t) for t in df.dtypes],
        "non_null_count": total - null_counts.values,
        "missing_count": null_counts.values,
        "missing_pct": pcts.values,
        "completeness_pct": (100.0 - pcts.values),
    })

    return summary.sort_values(by="missing_count", ascending=False).reset_index(drop=True)


def validate_label_balance(df: pd.DataFrame, label_col: str = "label") -> Dict[str, Any]:
    """Audits the balance of positive and negative classes."""
    if df.empty or label_col not in df.columns:
        return {"balanced": False, "counts": {}}

    counts = df[label_col].value_counts().to_dict()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    ratio = (pos / neg) if neg > 0 else 0.0

    return {
        "balanced": pos == neg,
        "positive_count": pos,
        "negative_count": neg,
        "ratio": ratio,
        "counts": counts,
    }

