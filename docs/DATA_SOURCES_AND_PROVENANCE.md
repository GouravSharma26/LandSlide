# Data Sources, Provenance, and Split Protocol — LandSlide × GeoTrustNet

**Document Version:** 1.0.0 (Phase 1 Baseline)  
**Date:** 2026-11-19  

---

## 1. Overview of Data Assets

The **LandSlide — GeoTrustNet Intelligence** platform operates across two distinct, complementary data streams:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                DATA PROVENANCE MATRIX                                  │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│ Stream 1: Tabular Point-Zone  │ Stream 2: Spatio-Temporal Raster Patches               │
│ (Existing Himalayan Baseline) │ (GeoTrustNet Deep Learning Engine)                     │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ • Dataset: NASA COOLR + GEE   │ • Dataset: Sen12Landslides (Höhn et al., 2025)         │
│ • Sample Unit: Single point   │ • Sample Unit: 128 × 128 pixel patches (10m res)       │
│ • Total Size: N=254 samples   │ • Total Size: 39,556 patches across global events      │
│ • Format: CSV feature tables  │ • Format: NetCDF (.nc) / PyTorch (.pt) / GeoTIFF       │
│ • Task: Regional susceptibility│ • Task: High-resolution landslide scar segmentation    │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Sen12Landslides Benchmark Specification

- **Citation:** Höhn et al., *"A Spatio-Temporal Dataset for Satellite-Based Landslide Detection"*, Nature Scientific Data (2025).
- **DOI:** [10.1038/s41597-025-06167-2](https://doi.org/10.1038/s41597-025-06167-2)
- **HuggingFace Repository:** `https://huggingface.co/datasets/paulhoehn/Sen12Landslides`
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0).

### Spatial & Temporal Dimensions
- **Patch Dimensions:** $128 \times 128$ pixels ($1.28\text{ km} \times 1.28\text{ km}$ footprint).
- **Spatial Resolution:** $10.0\text{ meters}$ per pixel.
- **Temporal Sequence:** $T \le 15$ observation timestamps sampled around the target landslide event date.

### Modality Details & Channel Mapping
1. **Sentinel-2 MSI (Optical Multispectral):**
   - 10 surface reflectance channels: `B02` (Blue), `B03` (Green), `B04` (Red), `B05` (RedEdge 1), `B06` (RedEdge 2), `B07` (RedEdge 3), `B08` (NIR Broad), `B8A` (NIR Narrow), `B11` (SWIR 1), `B12` (SWIR 2).
   - Values: Surface reflectance scaled $[0, 10000]$.
   - Quality Layer: Scene Classification Layer (`SCL`) indicating clouds (classes 3, 8, 9, 10, 11).
2. **Sentinel-1 C-Band SAR (Active Microwave Radar):**
   - 2 dual-polarization backscatter channels: `VV` and `VH` (in dB).
   - Day-and-night all-weather observation through monsoon clouds.
3. **Copernicus GLO-30 DSM (Topography):**
   - 2 terrain channels: `elevation` (meters above sea level) and `slope` (degrees).
4. **Ground-Truth Landslide Mask (`MASK`):**
   - Shape: $1 \times 128 \times 128$.
   - Values: Binary $\{0, 1\}$, where $0 = \text{stable ground}$, $1 = \text{confirmed landslide scar}$.

---

## 3. Dataset Download and Setup Instructions

Per project policy, large datasets are **never downloaded automatically** during app startup or testing.

### Manual Download Workflow:
To obtain the full Sen12Landslides benchmark for GPU training:

```bash
# 1. Install huggingface-hub CLI
pip install -q huggingface_hub

# 2. Download a specific task subset (e.g. S12LS-LD detection subset ~15 GB)
huggingface-cli download paulhoehn/Sen12Landslides \
    --repo-type dataset \
    --local-dir data/sen12landslides \
    --include "S12LS-LD/*"
```

For continuous integration, unit testing, and CPU development, the platform provides **deterministic synthetic fixtures** under `data/synthetic/` generated via:
```bash
python scripts/generate_fixtures_and_manifests.py
```

---

## 4. Leakage-Safe Split Protocol

To eliminate spatial autocorrelation and event memorization (**Tobler's First Law of Geography**):

1. **Group Partitioning Rule:**
   - Splitting is performed strictly at the **Geographic Region (`region_id`)** or **Landslide Event (`event_id`)** level.
   - Patches originating from the same geographic valley or same physical mass-wasting event are never split across train and test sets.
2. **Mathematical Disjointness Contract:**
   $$\text{Regions}_{\text{train}} \cap \text{Regions}_{\text{val}} = \emptyset, \quad \text{Regions}_{\text{train}} \cap \text{Regions}_{\text{test}} = \emptyset, \quad \text{Regions}_{\text{val}} \cap \text{Regions}_{\text{test}} = \emptyset$$
3. **Cryptographic Checksumming:**
   - Every split partition generates a SHA-256 hash of the complete `sample_id:split` mapping.
   - Stored in `data/manifests/synthetic_split_manifest.csv` and accompanying JSON metadata to ensure exact experiment reproducibility.
