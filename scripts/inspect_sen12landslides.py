"""Sen12Landslides Dataset Inspection and Scientific Provenance Auditor.

Audits dataset manifests, checks modality availability, measures class imbalance,
verifies geographic region distributions, and validates leakage-free splits.

Usage:
    python scripts/inspect_sen12landslides.py --manifest data/manifests/synthetic_split_manifest.csv
    python scripts/inspect_sen12landslides.py --manifest data/manifests/sen12landslides_split_manifest.csv --output-report reports/dataset_audit.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Windows DLL initialization for torch (must import torch before pandas/numpy on Windows)
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.geotrustnet.data.validation import validate_split_disjointness


def audit_dataset_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Audits manifest and returns detailed provenance and distribution metrics."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix == ".csv":
        df = pd.read_csv(manifest_path)
    elif manifest_path.suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw.get("records", raw))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")

    total_samples = len(df)
    n_regions = df["region_id"].nunique() if "region_id" in df.columns else 0
    n_events = df["event_id"].nunique() if "event_id" in df.columns else 0

    # Modality coverage
    opt_cov = float(df["has_optical"].mean() * 100.0) if "has_optical" in df.columns else 100.0
    sar_cov = float(df["has_sar"].mean() * 100.0) if "has_sar" in df.columns else 100.0
    dem_cov = float(df["has_dem"].mean() * 100.0) if "has_dem" in df.columns else 100.0

    # Class balance
    mean_ls_ratio = float(df["landslide_pixel_ratio"].mean()) if "landslide_pixel_ratio" in df.columns else 0.0
    max_ls_ratio = float(df["landslide_pixel_ratio"].max()) if "landslide_pixel_ratio" in df.columns else 0.0
    min_ls_ratio = float(df["landslide_pixel_ratio"].min()) if "landslide_pixel_ratio" in df.columns else 0.0

    # Split breakdown
    split_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    split_pcts = {k: round(v / total_samples * 100.0, 1) for k, v in split_counts.items()} if total_samples > 0 else {}

    # Leakage check
    is_disjoint = False
    disjoint_report = {}
    if "split" in df.columns and "region_id" in df.columns:
        is_disjoint, disjoint_report = validate_split_disjointness(
            df, split_col="split", group_cols=["region_id"]
        )

    # Regional distribution
    region_summary = {}
    if "region_id" in df.columns:
        reg_counts = df["region_id"].value_counts().to_dict()
        region_summary = {
            "region_count": len(reg_counts),
            "samples_per_region": reg_counts,
        }

    # Duplicate file checks
    duplicate_count = 0
    if "file_sha256" in df.columns:
        duplicate_count = int(df["file_sha256"].duplicated().sum())

    audit_result = {
        "manifest_path": str(manifest_path).replace("\\", "/"),
        "total_samples": total_samples,
        "n_regions": n_regions,
        "n_events": n_events,
        "modality_coverage_percent": {
            "optical": round(opt_cov, 1),
            "sar": round(sar_cov, 1),
            "dem": round(dem_cov, 1),
        },
        "landslide_pixel_prevalence": {
            "mean_ratio": round(mean_ls_ratio, 6),
            "min_ratio": round(min_ls_ratio, 6),
            "max_ratio": round(max_ls_ratio, 6),
            "imbalance_severity": "Severe (< 5% positive)" if mean_ls_ratio < 0.05 else "Moderate",
        },
        "splits": {
            "counts": split_counts,
            "percentages": split_pcts,
            "is_leakage_free_disjoint": is_disjoint,
            "leakage_violations": disjoint_report.get("violations", []),
        },
        "region_summary": region_summary,
        "duplicate_files_detected": duplicate_count,
        "provenance_status": "Verified Clean" if is_disjoint and duplicate_count == 0 else "Warnings Present",
    }
    return audit_result


def main():
    parser = argparse.ArgumentParser(description="Sen12Landslides Manifest & Provenance Inspector")
    parser.add_argument("--manifest", type=str, default="data/manifests/synthetic_split_manifest.csv", help="Path to manifest CSV/JSON")
    parser.add_argument("--output-report", type=str, default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    manifest_file = Path(args.manifest)
    if not manifest_file.exists():
        # Fallback to relative from root
        manifest_file = ROOT_DIR / args.manifest
        if not manifest_file.exists():
            print(f"[Error] Manifest file not found: {args.manifest}")
            sys.exit(1)

    report = audit_dataset_manifest(manifest_file)

    print("=" * 65)
    print("  Sen12Landslides Dataset Manifest & Provenance Audit")
    print("=" * 65)
    print(f"Manifest Path:      {report['manifest_path']}")
    print(f"Total Samples:      {report['total_samples']}")
    print(f"Unique Regions:     {report['n_regions']}")
    print(f"Unique Events:      {report['n_events']}")
    print(f"Duplicates:         {report['duplicate_files_detected']}")
    print(f"Leakage-Free:       {report['splits']['is_leakage_free_disjoint']}")
    print("\nModality Coverage:")
    for mod, cov in report["modality_coverage_percent"].items():
        print(f"  - {mod.upper():<8}: {cov}%")

    print("\nSplit Allocation:")
    for sp, cnt in report["splits"]["counts"].items():
        pct = report["splits"]["percentages"].get(sp, 0.0)
        print(f"  - {sp.upper():<6}: {cnt:>5} samples ({pct}%)")

    print("\nLandslide Prevalence (Class Balance):")
    print(f"  - Mean Landslide Ratio: {report['landslide_pixel_prevalence']['mean_ratio'] * 100:.2f}%")
    print(f"  - Severity:             {report['landslide_pixel_prevalence']['imbalance_severity']}")
    print(f"Provenance Status:  {report['provenance_status']}")
    print("=" * 65)

    if args.output_report:
        out_p = Path(args.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nAudit report exported to: {out_p}")


if __name__ == "__main__":
    main()
