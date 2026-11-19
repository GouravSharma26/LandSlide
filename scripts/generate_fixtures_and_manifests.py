"""Utility script to generate synthetic fixtures and leakage-safe split manifests for Phase 1 testing."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.geotrustnet.data.synthetic import generate_synthetic_dataset
from src.geotrustnet.data.splits import generate_region_event_splits, save_split_manifest


def main():
    root = Path(__file__).resolve().parent.parent
    synthetic_dir = root / "data" / "synthetic"
    manifest_dir = root / "data" / "manifests"

    print("Generating synthetic multimodal patches...")
    df_raw = generate_synthetic_dataset(
        output_dir=synthetic_dir,
        n_regions=4,
        events_per_region=3,
        patches_per_event=2,
        seed=42,
    )
    print(f"Generated {len(df_raw)} patches across {df_raw['region_id'].nunique()} regions and {df_raw['event_id'].nunique()} events.")

    print("\nGenerating region-held-out splits...")
    df_splits, split_hash = generate_region_event_splits(
        manifest_df=df_raw,
        group_col="region_id",
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    print(f"Split Hash (SHA-256): {split_hash}")
    print(f"Split distribution:\n{df_splits['split'].value_counts()}")

    manifest_csv = manifest_dir / "synthetic_split_manifest.csv"
    save_split_manifest(
        df=df_splits,
        output_path=manifest_csv,
        metadata={
            "description": "Synthetic multi-region test manifest for GeoTrustNet Phase 1",
            "group_column": "region_id",
            "seed": 42,
        },
    )
    print(f"\nManifest successfully saved to: {manifest_csv}")


if __name__ == "__main__":
    main()
