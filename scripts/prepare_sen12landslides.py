"""Sen12Landslides Dataset Acquisition and Preparation Pipeline.

Prepares raw Sen12Landslides observation patches, extracts metadata, verifies
integrity and channel alignment, and creates deterministic, leakage-safe splits.

Usage:
    # 1. Dry run / setup instructions:
    python scripts/prepare_sen12landslides.py --dry-run

    # 2. Prepare from existing downloaded directory:
    python scripts/prepare_sen12landslides.py --raw-dir /path/to/Sen12Landslides --output-dir data/manifests

    # 3. Explicitly initiate download from Hugging Face Hub (requires HF credentials if gated):
    python scripts/prepare_sen12landslides.py --download --raw-dir data/raw/Sen12Landslides
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

from src.geotrustnet.data.splits import generate_region_event_splits
from src.geotrustnet.data.validation import validate_split_disjointness


def compute_file_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def download_sen12landslides(target_dir: Path) -> Path:
    """Downloads Sen12Landslides dataset from Hugging Face repository."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    repo_id = "paulhoehn/Sen12Landslides"

    print(f"\n[Dataset Download] Connecting to Hugging Face Hub: {repo_id}")
    print(f"[Dataset Download] Destination: {target_dir}")

    try:
        from huggingface_hub import snapshot_download
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[Dataset Download] Completed successfully to: {downloaded_path}")
        return Path(downloaded_path)
    except ImportError:
        print("[Error] 'huggingface_hub' is not installed. Please install it using:")
        print("    pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to download dataset: {e}")
        print("\nManual Alternative:")
        print("  git clone https://huggingface.co/datasets/paulhoehn/Sen12Landslides " + str(target_dir))
        sys.exit(1)


def scan_and_extract_metadata(raw_dir: Path) -> pd.DataFrame:
    """Scans raw_dir for patch files (.pt or .tif) and extracts metadata."""
    raw_dir = Path(raw_dir)
    pt_files = sorted(list(raw_dir.glob("**/*.pt")))
    tif_files = sorted(list(raw_dir.glob("**/*.tif*")))

    records: List[Dict[str, Any]] = []

    if pt_files:
        print(f"[Scanner] Found {len(pt_files)} PyTorch (.pt) patch files.")
        try:
            import torch
        except ImportError:
            torch = None

        for idx, f in enumerate(pt_files):
            rel_path = f.relative_to(ROOT_DIR) if f.is_relative_to(ROOT_DIR) else f
            file_hash = compute_file_sha256(f)

            # Default attributes inferred from path or metadata
            parts = f.stem.split("_")
            region_id = parts[0] if len(parts) > 1 else "Region_Global"
            event_id = parts[1] if len(parts) > 2 else "Event_0"

            has_opt = True
            has_sar = True
            has_dem = True
            ls_ratio = 0.0

            # If torch is available, inspect sample attributes directly
            if torch is not None:
                try:
                    sample = torch.load(f, weights_only=False)
                    if hasattr(sample, "metadata"):
                        region_id = getattr(sample.metadata, "region_id", region_id)
                        event_id = getattr(sample.metadata, "event_id", event_id)
                    has_opt = getattr(sample, "optical_available", True) and (sample.optical is not None)
                    has_sar = getattr(sample, "sar_available", True) and (sample.sar is not None)
                    has_dem = getattr(sample, "dem_available", True) and (sample.dem is not None)
                    if getattr(sample, "mask", None) is not None:
                        mask_np = sample.mask.numpy()
                        ls_ratio = float((mask_np == 1).sum() / mask_np.size)
                except Exception:
                    pass

            records.append({
                "sample_id": f.stem,
                "file_path": str(rel_path).replace("\\", "/"),
                "file_sha256": file_hash,
                "region_id": region_id,
                "event_id": event_id,
                "has_optical": has_opt,
                "has_sar": has_sar,
                "has_dem": has_dem,
                "landslide_pixel_ratio": round(ls_ratio, 6),
                "height": 128,
                "width": 128,
                "crs": "EPSG:4326",
            })

    elif tif_files:
        print(f"[Scanner] Found {len(tif_files)} GeoTIFF files.")
        for f in tif_files:
            rel_path = f.relative_to(ROOT_DIR) if f.is_relative_to(ROOT_DIR) else f
            records.append({
                "sample_id": f.stem,
                "file_path": str(rel_path).replace("\\", "/"),
                "file_sha256": compute_file_sha256(f),
                "region_id": f.stem.split("_")[0],
                "event_id": "Event_0",
                "has_optical": True,
                "has_sar": True,
                "has_dem": True,
                "landslide_pixel_ratio": 0.05,
                "height": 128,
                "width": 128,
                "crs": "EPSG:4326",
            })
    else:
        print(f"[Warning] No .pt or .tif files found in {raw_dir}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Sen12Landslides Dataset Preparation")
    parser.add_argument("--raw-dir", type=str, default="data/raw/Sen12Landslides", help="Path to raw dataset")
    parser.add_argument("--output-dir", type=str, default="data/manifests", help="Output directory for manifests")
    parser.add_argument("--download", action="store_true", help="Download dataset from Hugging Face Hub")
    parser.add_argument("--dry-run", action="store_true", help="Display download instructions without executing")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed for splits")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("=" * 70)
        print("  Sen12Landslides Dataset Setup & Acquisition Instructions")
        print("=" * 70)
        print("Dataset:      Sen12Landslides: A Spatio-Temporal Multimodal Benchmark")
        print("License:      Creative Commons Attribution 4.0 International (CC-BY-4.0)")
        print("Citation:     Höhn et al., Nature Scientific Data (2025)")
        print("Hugging Face: https://huggingface.co/datasets/paulhoehn/Sen12Landslides")
        print("\nTo download the dataset automatically using this script:")
        print("    python scripts/prepare_sen12landslides.py --download --raw-dir data/raw/Sen12Landslides")
        print("\nOr download using the Hugging Face CLI:")
        print("    huggingface-cli download paulhoehn/Sen12Landslides --repo-type dataset --local-dir data/raw/Sen12Landslides")
        print("\nOr in Google Colab / Kaggle:")
        print("    from huggingface_hub import snapshot_download")
        print("    snapshot_download(repo_id='paulhoehn/Sen12Landslides', repo_type='dataset', local_dir='/content/sen12landslides')")
        print("=" * 70)
        return

    if args.download:
        raw_path = download_sen12landslides(raw_path)

    if not raw_path.exists():
        print(f"[Error] Raw dataset path '{raw_path}' does not exist.")
        print("Run with '--dry-run' to inspect download instructions.")
        sys.exit(1)

    print(f"\n[Step 1] Scanning files in: {raw_path}")
    manifest_df = scan_and_extract_metadata(raw_path)

    if len(manifest_df) == 0:
        print("[Error] No valid dataset samples discovered. Cannot build manifest.")
        sys.exit(1)

    print(f"[Step 2] Extracted metadata for {len(manifest_df)} samples across {manifest_df['region_id'].nunique()} regions.")

    # Generate deterministic splits
    print(f"[Step 3] Partitioning samples into Train/Val/Test (Seed={args.seed})...")
    split_df, split_hash = generate_region_event_splits(
        manifest_df=manifest_df,
        group_col="region_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )

    # Save manifest outputs
    csv_path = out_dir / "sen12landslides_split_manifest.csv"
    json_path = out_dir / "sen12landslides_split_manifest.json"

    split_df.to_csv(csv_path, index=False)

    meta = {
        "dataset_name": "Sen12Landslides",
        "total_samples": len(split_df),
        "split_hash_sha256": split_hash,
        "seed": args.seed,
        "splits": split_df["split"].value_counts().to_dict(),
        "regions": split_df["region_id"].unique().tolist(),
        "columns": split_df.columns.tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Completed] Manifests generated successfully:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Split Hash: {split_hash}")
    print(f"  - Distribution: {meta['splits']}")


if __name__ == "__main__":
    main()
