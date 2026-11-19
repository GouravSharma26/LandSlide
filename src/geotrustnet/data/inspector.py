"""Dataset inspection and metadata profiling utility for GeoTrustNet."""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.geotrustnet.data.validation import validate_split_disjointness


def inspect_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Inspects a dataset manifest CSV/JSON and returns comprehensive provenance statistics."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix == ".csv":
        df = pd.read_csv(manifest_path)
    elif manifest_path.suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("records", data))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")

    total_samples = len(df)
    n_regions = df["region_id"].nunique() if "region_id" in df.columns else 0
    n_events = df["event_id"].nunique() if "event_id" in df.columns else 0

    opt_pct = float(df["has_optical"].mean() * 100.0) if "has_optical" in df.columns else 0.0
    sar_pct = float(df["has_sar"].mean() * 100.0) if "has_sar" in df.columns else 0.0
    dem_pct = float(df["has_dem"].mean() * 100.0) if "has_dem" in df.columns else 0.0

    mean_ls_ratio = float(df["landslide_pixel_ratio"].mean()) if "landslide_pixel_ratio" in df.columns else 0.0

    split_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}

    is_disjoint, disjoint_report = (
        validate_split_disjointness(df, split_col="split", group_cols=["region_id", "event_id"])
        if "split" in df.columns
        else (False, {"violations": ["No split column present"]})
    )

    stats = {
        "manifest_path": str(manifest_path),
        "total_samples": total_samples,
        "n_regions": n_regions,
        "n_events": n_events,
        "modality_coverage_percent": {
            "optical": round(opt_pct, 1),
            "sar": round(sar_pct, 1),
            "dem": round(dem_pct, 1),
        },
        "mean_landslide_pixel_ratio": round(mean_ls_ratio, 4),
        "split_counts": split_counts,
        "is_leakage_free_disjoint": is_disjoint,
        "disjoint_report": disjoint_report,
    }
    return stats


def print_manifest_summary(manifest_path: Path) -> None:
    """Pretty prints the manifest inspection report."""
    stats = inspect_manifest(manifest_path)
    print("=" * 60)
    print("GEOTRUSTNET DATASET MANIFEST INSPECTION")
    print("=" * 60)
    print(f"Manifest File  : {stats['manifest_path']}")
    print(f"Total Patches  : {stats['total_samples']}")
    print(f"Unique Regions : {stats['n_regions']}")
    print(f"Unique Events  : {stats['n_events']}")
    print(f"Mean LS Pixels : {stats['mean_landslide_pixel_ratio']:.4f}")
    print("\nModality Availability:")
    for mod, pct in stats["modality_coverage_percent"].items():
        print(f"  - {mod.upper():8s}: {pct}%")
    print("\nSplit Distribution:")
    for sp, cnt in stats["split_counts"].items():
        print(f"  - {sp:8s}: {cnt:4d} patches ({cnt/stats['total_samples']*100:.1f}%)")
    print(f"\nLeakage-Free Disjointness: {'PASSED' if stats['is_leakage_free_disjoint'] else 'FAILED'}")
    if not stats["is_leakage_free_disjoint"]:
        print(f"Violations: {stats['disjoint_report']['violations']}")
    print("=" * 60)
