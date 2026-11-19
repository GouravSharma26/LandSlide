"""Data contracts, validation, split generators, and PyTorch datasets for GeoTrustNet."""

from src.geotrustnet.data.contracts import (
    PatchMetadata,
    PatchSample,
    ManifestEntry,
)
from src.geotrustnet.data.validation import (
    validate_patch_spatial,
    validate_temporal_order,
    validate_future_data_leakage,
    validate_label_alignment,
    validate_split_disjointness,
)
from src.geotrustnet.data.splits import (
    generate_region_event_splits,
    save_split_manifest,
)
from src.geotrustnet.data.synthetic import (
    create_synthetic_patch,
    generate_synthetic_dataset,
)
from src.geotrustnet.data.datasets import (
    GeoTrustNetPatchDataset,
    collate_geotrustnet_patches,
)

__all__ = [
    "PatchMetadata",
    "PatchSample",
    "ManifestEntry",
    "validate_patch_spatial",
    "validate_temporal_order",
    "validate_future_data_leakage",
    "validate_label_alignment",
    "validate_split_disjointness",
    "generate_region_event_splits",
    "save_split_manifest",
    "create_synthetic_patch",
    "generate_synthetic_dataset",
    "GeoTrustNetPatchDataset",
    "collate_geotrustnet_patches",
]
