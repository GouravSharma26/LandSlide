"""Deterministic, leakage-safe split generation utilities for GeoTrustNet."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.geotrustnet.data.validation import validate_split_disjointness


def generate_region_event_splits(
    manifest_df: pd.DataFrame,
    group_col: str = "region_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """Partitions dataset samples into Train, Val, and Test by holding out whole geographic regions/events.

    Guarantees zero spatial/event leakage across splits.
    Returns:
        (updated_df, sha256_hash_of_split_assignment)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    df = manifest_df.copy()

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in manifest DataFrame")

    unique_groups = np.sort(df[group_col].dropna().unique())
    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    if n_groups < 3:
        raise ValueError(f"Need at least 3 distinct groups to create train/val/test splits, found {n_groups}")

    n_train = max(1, int(np.round(n_groups * train_ratio)))
    n_val = max(1, int(np.round(n_groups * val_ratio)))
    # Ensure remaining goes to test
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train:n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val:])

    # Assign split column
    def assign_split(g):
        if g in train_groups:
            return "train"
        elif g in val_groups:
            return "val"
        else:
            return "test"

    df["split"] = df[group_col].apply(assign_split)

    # Validate disjointness
    is_disjoint, report = validate_split_disjointness(df, split_col="split", group_cols=[group_col])
    if not is_disjoint:
        raise RuntimeError(f"Generated split failed disjointness check: {report['violations']}")

    # Compute deterministic SHA256 hash of sample-to-split mapping
    hash_str = "".join(f"{s}:{sp};" for s, sp in sorted(zip(df["sample_id"], df["split"])))
    split_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()
    df.attrs["split_hash_sha256"] = split_hash
    df.attrs["group_col"] = group_col
    df.attrs["seed"] = seed

    return df, split_hash


def save_split_manifest(
    df: pd.DataFrame,
    output_path: Path,
    metadata: Dict[str, Any] = None,
) -> Path:
    """Saves the split manifest with provenance metadata and checksum."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is None:
        metadata = {}

    metadata["split_hash_sha256"] = df.attrs.get("split_hash_sha256", "")
    metadata["total_samples"] = len(df)
    metadata["split_distribution"] = df["split"].value_counts().to_dict()

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return output_path
    elif output_path.suffix == ".json":
        records = df.to_dict(orient="records")
        data = {"metadata": metadata, "records": records}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path
    else:
        raise ValueError(f"Unsupported manifest format: {output_path.suffix}")
