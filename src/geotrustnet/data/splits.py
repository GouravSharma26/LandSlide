"""Deterministic, leakage-safe split generation utilities for GeoTrustNet."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.geotrustnet.data.validation import validate_split_disjointness


class DisjointSet:
    """Union-find data structure for computing connected components."""

    def __init__(self):
        self.parent = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def generate_region_event_splits(
    manifest_df: pd.DataFrame,
    group_col: str = "region_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """Partitions dataset samples into Train, Val, and Test by holding out geographic regions/events.

    When both 'region_id' and 'event_id' exist in the manifest, samples are clustered into
    connected components (bipartite graph of regions and events) ensuring dual disjointness:
    neither region_id nor event_id will ever cross split boundaries.

    Guarantees zero spatial/event leakage across splits.
    Returns:
        (updated_df, sha256_hash_of_split_assignment)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    df = manifest_df.copy()

    # Determine whether dual disjointness (region_id + event_id) is applicable
    has_dual_grouping = "region_id" in df.columns and "event_id" in df.columns

    if has_dual_grouping:
        ds = DisjointSet()
        for _, row in df.iterrows():
            r_node = f"r_{row['region_id']}"
            e_node = f"e_{row['event_id']}"
            ds.union(r_node, e_node)

        cluster_series = df["region_id"].apply(lambda r: ds.find(f"r_{r}"))
        unique_groups = np.sort(cluster_series.unique())
        validation_cols = ["region_id", "event_id"]
    else:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in manifest DataFrame")
        cluster_series = df[group_col]
        unique_groups = np.sort(df[group_col].dropna().unique())
        validation_cols = [group_col]

    rng = np.random.RandomState(seed)
    shuffled_groups = rng.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 distinct independent groups/clusters to create train/val/test splits, found {n_groups}"
        )

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

    df["split"] = cluster_series.apply(assign_split)

    # Validate disjointness
    is_disjoint, report = validate_split_disjointness(df, split_col="split", group_cols=validation_cols)
    if not is_disjoint:
        raise RuntimeError(f"Generated split failed disjointness check: {report['violations']}")

    # Compute deterministic SHA256 hash of sample-to-split mapping
    hash_str = "".join(f"{s}:{sp};" for s, sp in sorted(zip(df["sample_id"], df["split"])))
    split_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()
    df.attrs["split_hash_sha256"] = split_hash
    df.attrs["group_col"] = "dual_region_event" if has_dual_grouping else group_col
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
