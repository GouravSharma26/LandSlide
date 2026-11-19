"""Compiler for official Sen12Landslides manifests and leakage-safe disjoint splits.

Processes the official metadata from Höhn et al. (2025):
- data/manifests/sen12landslides_splits_aligned.json
Produces:
1. data/manifests/sen12landslides_iid_manifest.csv (official benchmark baseline split)
2. data/manifests/sen12landslides_disjoint_manifest.csv (GeoTrustNet strict disjoint inventory holdout)
3. docs/reports/sen12landslides_real_data_audit.json (detailed real-world statistics)
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


def compile_official_manifests():
    json_path = ROOT_DIR / "data" / "manifests" / "sen12landslides_splits_aligned.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Official splits JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_splits = json.load(f)

    records = []
    for split_name in ["train", "val", "test"]:
        for item in raw_splits[split_name]:
            rec = {
                "sample_id": item["id"],
                "region_id": item["inventory"],
                "event_id": f"event_{item['inventory']}",
                "s1asc_path": item["s1asc"],
                "s1dsc_path": item["s1dsc"],
                "s2_path": item["s2"],
                "pixel_annotated": item["pixel_annotated"],
                "landslide_pixel_ratio": round(item["pixel_annotated"] / (128 * 128), 6),
                "official_split": split_name,
                "has_optical": True,
                "has_sar": True,
                "has_dem": True,
                "height": 128,
                "width": 128,
                "crs": "EPSG:4326"
            }
            records.append(rec)

    df = pd.DataFrame(records)
    print(f"[Compiler] Loaded {len(df)} official aligned samples.")

    # 1. Save official IID split manifest
    df_iid = df.copy()
    df_iid["split"] = df_iid["official_split"]
    df_iid["file_path"] = df_iid["s1asc_path"]
    iid_csv = ROOT_DIR / "data" / "manifests" / "sen12landslides_iid_manifest.csv"
    df_iid.to_csv(iid_csv, index=False)
    print(f"[Compiler] Saved official IID manifest: {iid_csv}")

    # 2. Generate strict disjoint inventory holdout
    # We partition the 11 inventories into strictly disjoint sets:
    # Train (7 inventories): china, indonesia, itogon, kyrgyzstan1, kyrgyzstan2, hiroshima, usa (1,894 samples)
    # Val (2 inventories): newzealand, chimanimani (853 samples)
    # Test (2 inventories): italy, hokkaido (1,943 samples - major diverse out-of-region holdouts)
    train_invs = {"china", "indonesia", "itogon", "kyrgyzstan1", "kyrgyzstan2", "hiroshima", "usa"}
    val_invs = {"newzealand", "chimanimani"}
    test_invs = {"italy", "hokkaido"}

    def assign_disjoint_split(inv):
        if inv in train_invs:
            return "train"
        elif inv in val_invs:
            return "val"
        elif inv in test_invs:
            return "test"
        else:
            return "train"

    df_disjoint = df.copy()
    df_disjoint["split"] = df_disjoint["region_id"].apply(assign_disjoint_split)
    df_disjoint["file_path"] = df_disjoint["s1asc_path"]
    disjoint_csv = ROOT_DIR / "data" / "manifests" / "sen12landslides_disjoint_manifest.csv"
    df_disjoint.to_csv(disjoint_csv, index=False)
    print(f"[Compiler] Saved disjoint inventory holdout manifest: {disjoint_csv}")

    # 3. Compute detailed real-world audit metrics
    inv_stats = df.groupby("region_id").agg(
        samples=("sample_id", "count"),
        total_annotated_pixels=("pixel_annotated", "sum"),
        mean_annotated_pixels=("pixel_annotated", "mean"),
        mean_landslide_ratio=("landslide_pixel_ratio", "mean"),
        max_landslide_ratio=("landslide_pixel_ratio", "max"),
        min_landslide_ratio=("landslide_pixel_ratio", "min"),
    ).to_dict(orient="index")

    audit_report = {
        "dataset_name": "Sen12Landslides (S12LS-LD Harmonized Benchmark)",
        "source": "https://huggingface.co/datasets/paulhoehn/Sen12Landslides",
        "citation": "Höhn et al., Nature Scientific Data (2025)",
        "license": "CC-BY-4.0",
        "total_aligned_samples": len(df),
        "spatial_dimensions": {"height": 128, "width": 128, "resolution_m": 10.0},
        "modalities": {
            "sar_asc": "VV, VH (dB)",
            "sar_dsc": "VV, VH (dB)",
            "optical": "10 bands B02-B12 (Surface Refl.)",
            "dem": "Elevation (m) and derived Slope (deg)",
            "mask": "Binary {0, 1} where MASK > 0"
        },
        "inventory_count": df["region_id"].nunique(),
        "inventories": sorted(df["region_id"].unique().tolist()),
        "official_iid_split_counts": df_iid["split"].value_counts().to_dict(),
        "disjoint_split_counts": df_disjoint["split"].value_counts().to_dict(),
        "disjoint_split_inventories": {
            "train": sorted(list(train_invs)),
            "val": sorted(list(val_invs)),
            "test": sorted(list(test_invs))
        },
        "overall_pixel_prevalence": {
            "total_pixels": len(df) * 128 * 128,
            "total_landslide_pixels": int(df["pixel_annotated"].sum()),
            "mean_landslide_pixel_prevalence_pct": round(float(df["landslide_pixel_ratio"].mean() * 100), 3)
        },
        "inventory_statistics": inv_stats
    }

    out_audit = ROOT_DIR / "docs" / "reports" / "sen12landslides_real_data_audit.json"
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"[Compiler] Saved audit report: {out_audit}")

    return df_iid, df_disjoint, audit_report


if __name__ == "__main__":
    compile_official_manifests()
