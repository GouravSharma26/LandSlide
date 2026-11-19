# Phase 10C — Official Sen12Landslides Schema & Data Architecture Audit

**Project:** Himalayan Landslide Intelligence System (GeoTrustNet)  
**Phase:** Phase 10C — Data Integrity & Official Dataset Compatibility  
**Date:** 2026-11-19  
**Reference Specification:** Höhn et al., *Nature Scientific Data* 12, Article 6167 (2025). DOI: `10.1038/s41597-025-06167-2`  
**Repository Source:** Hugging Face Hub (`paulhoehn/Sen12Landslides`), GitHub (`PaulH97/Sen12Landslides`)  
**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)

---

## 1. Executive Summary & Verification Scope

This document establishes the official data schema and contract specifications for the **Sen12Landslides** dataset. 

> [!IMPORTANT]
> **Authentic File Availability Status:** The complete ~50 GB raw satellite imagery archive is not downloaded on this local development machine (consistent with project storage constraints). All schema specifications herein are derived directly from the peer-reviewed dataset publication (Höhn et al., 2025), official normalization artifacts (`configs/data/sen12landslides_norm.json`), and official manifest splits (`data/manifests/sen12landslides_splits_aligned.json`). Authentic file compatibility is implemented to strict specifications and verified against synthetic NetCDF fixtures conforming exactly to the published schema. Genuine satellite data execution will occur upon Cloud GPU environment attachment (Phase 11).

---

## 2. Directory Structure & File Organization

The official harmonized dataset (`tasks/S12LS-LD/harmonized/`) organizes multi-sensor observations into three parallel directory trees:

```text
Sen12Landslides/
├── s1asc/                         # Sentinel-1 Ascending GRD NetCDF files
│   └── <inventory>_s1asc_<id>.nc  # e.g., chimanimani_s1asc_1004.nc
├── s1dsc/                         # Sentinel-1 Descending GRD NetCDF files
│   └── <inventory>_s1dsc_<id>.nc  # e.g., chimanimani_s1dsc_1004.nc
├── s2/                            # Sentinel-2 L2A Surface Reflectance NetCDF files
│   └── <inventory>_s2_<id>.nc     # e.g., chimanimani_s2_1004.nc
└── splits_aligned.json            # Official train/val/test allocation
```

---

## 3. Sensor Variables, Dimensions & Nodata Conventions

### 3.1 Sentinel-1 SAR NetCDF (`s1asc`, `s1dsc`)
| Variable | Data Type | Dimensions | Unit / Range | Nodata Sentinel | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`VV`** | `float32` | `(time, y, x)` | $[-45.0, 10.0]$ dB | $-9999.0$ | Vertical-transmit, Vertical-receive backscatter |
| **`VH`** | `float32` | `(time, y, x)` | $[-45.0, 10.0]$ dB | $-9999.0$ | Vertical-transmit, Horizontal-receive backscatter |
| **`DEM`** | `float32` | `(y, x)` / `(1, y, x)`| $[-500, 9000]$ m | $-9999.0$ | Copernicus GLO-30 DSM elevation |
| **`slope`** | `float32` | `(y, x)` / `(1, y, x)`| $[0.0, 90.0]^\circ$ | $-9999.0$ | Topographic slope (derived or static layer) |
| **`mask` / `label`**| `uint8` | `(y, x)` | $\{0, 1, 255\}$ | $255$ | Target mask: $0$=stable, $1$=landslide, $255$=nodata |

### 3.2 Sentinel-2 Optical NetCDF (`s2`)
| Variable | Data Type | Dimensions | Unit / Range | Nodata Sentinel | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`B02` (Blue)** | `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 490 nm surface reflectance (scaled by 10,000) |
| **`B03` (Green)**| `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 560 nm surface reflectance |
| **`B04` (Red)**  | `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 665 nm surface reflectance |
| **`B05` (RE1)**  | `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 705 nm Red Edge |
| **`B06` (RE2)**  | `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 740 nm Red Edge |
| **`B07` (RE3)**  | `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 783 nm Red Edge |
| **`B08` (NIR)**  | `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 842 nm Broad NIR |
| **`B8A` (Narrow NIR)**| `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 865 nm Water Vapor NIR |
| **`B11` (SWIR1)**| `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 1610 nm Short-wave Infrared |
| **`B12` (SWIR2)**| `uint16` / `float32` | `(time, y, x)` | $[0, 10000]$ | $-9999.0$ | 2190 nm Short-wave Infrared |
| **`SCL`**        | `uint8` | `(time, y, x)` | $\{0, \dots, 11\}$ | $0$ | Scene Classification Layer (cloud/shadow) |
| **`DEM`**        | `float32` | `(y, x)` | $[-500, 9000]$ m | $-9999.0$ | Co-registered elevation layer |

---

## 4. Optical Quality & Cloud Classification (SCL Decoding)

The Sentinel-2 L2A Scene Classification Layer (`SCL`) provides pixel-level categorization:

| SCL Value | Semantic Class | Optical Validity Policy |
| :---: | :--- | :--- |
| `0` | NO_DATA | **Invalid** (Excluded from optical feature extraction) |
| `1` | SATURATED_OR_DEFECTIVE | **Invalid** |
| `2` | DARK_AREA_PIXELS | **Invalid** |
| `3` | CLOUD_SHADOWS | **Invalid** (Cloud contamination) |
| `4` | VEGETATION | **Valid** (Clear observation) |
| `5` | NOT_VEGETATED | **Valid** (Clear observation) |
| `6` | WATER | **Valid** |
| `7` | UNCLASSIFIED | **Valid** |
| `8` | CLOUD_MEDIUM_PROBABILITY | **Invalid** (Cloud contamination) |
| `9` | CLOUD_HIGH_PROBABILITY | **Invalid** (Severe cloud contamination) |
| `10` | THIN_CIRRUS | **Invalid** (Semi-transparent cloud) |
| `11` | SNOW | **Excluded from optical scarp detection** (Ambiguous) |

**Operational Quality Rule:**
A Sentinel-2 pixel $(t, y, x)$ is considered an authentic clear observation if and only if:
$$\text{OpticalValid}(t, y, x) = \mathbb{I}(\text{SCL}(t, y, x) \in \{4, 5, 6, 7\}) \land (\text{Bands}(t, y, x) \neq -9999.0)$$

---

## 5. Official Schema vs. Loader Expectations Matrix

| Schema Dimension | Official Published Specification | Loader Implementation Strategy (Phase 10C) |
| :--- | :--- | :--- |
| **SAR Channels** | `VV`, `VH` backscatter in dB | Case-insensitive discovery; supports `VV`/`vv` and `VH`/`vh`. |
| **SAR Time Horizon** | Multi-temporal ($T \le 15$) | Pads or slices to `fixed_time_steps=5`; retains temporal ordering. |
| **SAR/Optical Co-registration** | Spatial 128×128 aligned grid | Validates spatial shape `(128, 128)` strictly; raises error on mismatch. |
| **SAR/Optical Temporal Alignment** | Unsynchronized orbits | S1 and S2 capture dates are independent; loader preserves sensor-specific time arrays without forcing fake temporal identity. |
| **DEM Representation** | Named `DEM` in `s1` and `s2` | Discovers `DEM` or `elevation`; derives slope if absent; marks `dem_available=False` if completely missing. |
| **Target Mask Nodata** | Sentinel $255$ | Preserved as separate `mask_valid` boolean tensor; strictly excluded from loss & metrics. |
| **Normalization Owner** | Published `norm_aligned.json` statistics | Single-pass owner: `GeoTrustNetPatchDataset` applies normalization; `netcdf_loader.py` leaves data in physical domain by default. |
| **Geospatial Metadata** | `crs` attribute, coordinate variables `x`, `y` | Extracts actual coordinates; sets `None` if absent; never invents synthetic coordinates. |

---

## 6. Audit Conclusion & Compliance Status

1. **Schema Documented:** Official Sen12Landslides variables, dimensions, types, and nodata sentinels are completely codified.
2. **Nodata Preservation Mandate:** The target validity mask must be decoupled from the binary class values to avoid label leakage.
3. **Single Normalization Mandate:** Normalization must be applied exactly once by the dataset, preserving raw physical values in loader returns.
