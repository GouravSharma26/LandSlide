# Phase 10D Completion Report: Authentic NetCDF Verification & Real-Data Loader Qualification

**Project:** LandSlide — GeoTrustNet Intelligence  
**Repository:** `GouravSharma26/LandSlide`  
**Branch:** `geotrustnet-benchmark`  
**Milestone:** Phase 10D — Authentic Data Pipeline Qualification  
**Date:** 2026-11-19  
**Status:** **DATA PIPELINE QUALIFIED ON AUTHENTIC SEN12LANDSLIDES DATA** (Full-scale cloud GPU training pending Phase 11)

---

## 1. Executive Summary & Objective

Phase 10C established data-pipeline integrity safeguards (decoupling target validity masks, prohibiting synthetic benchmark fallbacks, and centralizing normalization ownership). However, as confirmed by research audit, the pipeline had only processed synthetic fixtures.

**Phase 10D accomplishes the critical scientific milestone of acquiring genuine, official Sen12Landslides NetCDF files and qualifying the entire ingestion, quality-masking, normalization, forward-pass, loss, gradient, and evaluation pipeline on authentic satellite imagery.**

### Key Achievements in Phase 10D:
1. **Acquired Authentic Official Triad:** Streamed directly from the official Hugging Face repository `paulhoehn/Sen12Landslides` (`data_harmonized/`) without downloading the entire 50 GB dataset:
   - `chimanimani_s1asc_1004.nc` (1,536,088 bytes)
   - `chimanimani_s1dsc_1004.nc` (1,531,115 bytes)
   - `chimanimani_s2_1004.nc` (2,999,631 bytes)
2. **Schema & Contract Alignment:** Verified exact coordinate arrays, EPSG:32736 CRS, 10m spatial resolution, GeoTransforms, 15 multi-temporal acquisitions, and co-registered target masks.
3. **Decoupled Ground-Truth Target Verification:** Confirmed that the embedded `MASK` has exactly 362 positive landslide pixels and 16,022 negative pixels (prevalence 2.2095%), exactly matching the manifest entry `pixel_annotated: 362`.
4. **Refined Sentinel-2 SCL Optical Quality Policy:** Made clear classes configurable (`(4, 5, 6)` by default); scientifically justified the exclusion of Class 7 ("unclassified") and Class 2 ("dark area pixels"); logged full policy rationale in patch metadata.
5. **Physical Topographic Slope Derivation:** Derived realistic topographic slope in degrees ($0.0^\circ$ to $66.7^\circ$, mean $23.7^\circ$) from the 10m DEM elevation grid.
6. **Single-Pass Normalization Verification:** Ensured single-pass normalization strictly owned by `GeoTrustNetPatchDataset` using training-only statistics, with guards preventing double-normalization.
7. **Model Execution on Authentic Satellite Data:** Verified Model A (`UNet`) and Model D (`GeoTrustNet`) forward passes, `CombinedBCEDiceLoss` masked execution, backward gradient flow (all finite), and decoupled metric calculation.
8. **Geographic Distribution & Prevalence Investigation:** Investigated the regional imbalance explaining why validation prevalence ($4.903\%$) exceeds training prevalence ($3.361\%$), explicitly identifying the Kaikoura earthquake inventory skew in New Zealand ($6.275\%$ prevalence, $76.2\%$ of validation positive pixels).
9. **Full Test Suite & Dashboard Regression:** Added 9 new unit/integration tests in `tests/test_geotrustnet_phase10d.py`. Full test suite achieves **186 / 186 passing tests (100%)**.

---

## 2. Authentic NetCDF Sample Acquisition & Provenance Audit

Rather than downloading the full 50 GB dataset, authentic multi-temporal files corresponding to sample ID `chimanimani_1004` (indexed in `data/manifests/sen12landslides_disjoint_manifest.csv`) were streamed via HTTP range queries directly from official repository tar archives (`s1asc_part01.tar.gz`, `s1dsc_part01.tar.gz`, `s2_part01.tar.gz`).

### Authentic File Triad Specifications

| File Name | Modality | Exact Bytes | SHA-256 Checksum | Dimensions |
| :--- | :--- | :--- | :--- | :--- |
| `chimanimani_s1asc_1004.nc` | Sentinel-1 Ascending | 1,536,088 | `612089fbbebcbeaf543415ef4eb7b794f7d2f9d51ff6a798f0607823f3fcfeeb` | $(T=15, X=128, Y=128)$ |
| `chimanimani_s1dsc_1004.nc` | Sentinel-1 Descending | 1,531,115 | `a281e285a86ebf010fa482200427c3da334a17feaa356d48bc30541fb1ec6e74` | $(T=15, X=128, Y=128)$ |
| `chimanimani_s2_1004.nc` | Sentinel-2 Optical | 2,999,631 | `1e9d1bf77bbba9d38ca80dc1c05d41f5a50785f764a7538be1c4a0fc84e7ec94` | $(T=15, X=128, Y=128)$ |

**Origin Archive:** `https://huggingface.co/datasets/paulhoehn/Sen12Landslides/resolve/main/data_harmonized/`  
**Patch Identification:** Chimanimani, Zimbabwe (Cyclone Idai event, 2019-03-15).

---

## 3. NetCDF Ingestion Schema Audit vs Loader Assumptions

Comprehensive inspection of the authentic NetCDF datasets using `xarray` and `netCDF4` revealed the exact data layout:

### 3.1 Spatial Coordinates & Projection
- **Grid Size:** $128 \times 128$ pixels
- **Pixel Spacing:** $10.0\text{ m} \times 10.0\text{ m}$
- **Coordinate Reference System (CRS):** `EPSG:32736` (WGS 84 / UTM zone 36S)
- **GeoTransform:** `480000.0 10.0 0.0 7793920.0 0.0 -10.0`
- **Spatial Bounds:**
  - $X \in [480000.0\text{ m}, 481280.0\text{ m}]$
  - $Y \in [7792640.0\text{ m}, 7793920.0\text{ m}]$

### 3.2 Temporal Horizon
- **Time steps ($T$):** 15 acquisitions spanning from 2018-09-08 to 2019-08-27.
- **Event Date:** `2019-03-15` (Cyclone Idai).
- **Temporal Alignment:** The archive provides 7 pre-event acquisitions and 8 post-event acquisitions (`pre_post_dates: {'pre': 7, 'post': 8}`).

### 3.3 Variable Layout & Value Distributions

| Modality | Variable | Shape | Data Type | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentinel-1** | `VV` | $(15, 128, 128)$ | `float32` | $[-26.20\text{ dB}, 0.00\text{ dB}]$ | Co-polarized backscatter |
| **Sentinel-1** | `VH` | $(15, 128, 128)$ | `float32` | $[-50.00\text{ dB}, -2.10\text{ dB}]$ | Cross-polarized backscatter |
| **Sentinel-2** | `B02`-`B12` (10 bands) | $(15, 128, 128)$ | `int16` | $[1, 10000]$ | Surface reflectance ($\times 10^4$) |
| **Sentinel-2** | `SCL` | $(15, 128, 128)$ | `int16` | $\{4, 5, 6, 7, 255\}$ | ESA Scene Classification Layer |
| **Topography** | `DEM` | $(15, 128, 128)$ | `int16` | $[1088\text{ m}, 1507\text{ m}]$ | Elevation above sea level |
| **Ground Truth** | `MASK` | $(15, 128, 128)$ | `uint8` | $\{0, 1\}$ | Binary landslide label |

### 3.4 Topographic Slope Derivation
The official Sen12Landslides NetCDF files contain elevation in meters (`DEM`), but do not precalculate slope. In `netcdf_loader.py`, topographic slope is now physically derived using the spatial gradient of the 10m grid:
$$\nabla z = \left(\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y}\right), \quad \theta = \arctan(\|\nabla z\|) \times \frac{180}{\pi}$$
On `chimanimani_1004`, this produces slope values between $0.0^\circ$ and $66.74^\circ$ with a mean of $23.71^\circ$, accurately reflecting the steep, landslide-prone topography of the Chimanimani range.

---

## 4. Optical Quality Policy (ESA SCL) Refinement & Justification

The Scene Classification Layer (`SCL`) provides pixel-level data quality flags produced by ESA's Sen2Cor algorithm.

### 4.1 Configurable SCL Policy
The loader now accepts `clear_scl_classes: Tuple[int, ...] = (4, 5, 6)` as a configurable parameter:
```python
load_sen12landslides_netcdf(..., clear_scl_classes=(4, 5, 6))
```

### 4.2 Scientific Justification for Excluded Classes

1. **Class 4 (Vegetation), Class 5 (Not-vegetated / bare soil), Class 6 (Water):**
   - **Policy:** **CLEAR / VALID**
   - **Rationale:** Unambiguous clear-sky ground surface observations where optical reflectance reliably represents physical land cover and landslide scars.
2. **Class 7 ("Unclassified"):**
   - **Policy:** **EXCLUDED BY DEFAULT (CONFIGURABLE)**
   - **Rationale:** Sen2Cor assigns Class 7 to pixels where spectral classification confidence falls below detection thresholds (frequently noisy pixels, cloud fringe boundaries, or mixed low-contrast surfaces). Including Class 7 introduces spectral noise into feature representations. It is excluded by default, but researchers can enable it by passing `clear_scl_classes=(4, 5, 6, 7)`.
3. **Class 2 ("Dark Area Pixels"):**
   - **Policy:** **EXCLUDED BY DEFAULT**
   - **Rationale:** In steep mountainous terrain, Class 2 denotes deep topographic cast shadows or cloud shadow margins. In cast shadows, incident solar radiation is near zero, causing severe SNR degradation and uninformative reflectance. Since GeoTrustNet's cross-attention is specifically designed to fall back to SAR (which is radar backscatter and immune to optical shadows), masking Class 2 directs the network to rely on SAR backscatter where optical data is physically degraded.
4. **Classes 3, 8, 9, 10, 1, 0, 255 (Shadows, Clouds, Cirrus, Saturated, Nodata):**
   - **Policy:** **EXCLUDED (INVALID)**
   - **Rationale:** Direct obscuration by cloud cover, atmospheric ice, detector saturation, or nodata sentinel.
   - **Empirical Observation:** In `chimanimani_s2_1004.nc`, time steps $t=0, 11, 12$ are completely filled with $255$ ($16,384$ pixels each), representing fully unobserved acquisitions. The optical validity mask correctly flags these timesteps as invalid across all pixels.

The policy choices and rationale are logged in `metadata.extra_metadata["scl_policy"]`.

---

## 5. Geographic Generalization Audit: Split & Inventory Prevalences

The Phase 10C review questioned why the validation split prevalence ($4.903\%$) is higher than the training split prevalence ($3.361\%$). An audit of all 4,392 patches across all 11 inventories in `data/manifests/sen12landslides_disjoint_manifest.csv` was conducted.

### 5.1 Per-Inventory Distribution Summary

| Region | Split | Patches | Annotated Landslide Pixels | Total Pixels | Prevalence (%) | Primary Trigger Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `china` | `train` | 47 | 11,254 | 770,048 | 1.4615% | Rainfall |
| `hiroshima` | `train` | 322 | 127,206 | 5,275,648 | 2.4112% | Rainfall (2018 Japan floods) |
| `indonesia` | `train` | 253 | 168,435 | 4,145,152 | 4.0634% | Rainfall |
| `itogon` | `train` | 82 | 17,171 | 1,343,488 | 1.2781% | Typhoon Mangkhut (2018) |
| `kyrgyzstan1` | `train` | 694 | 506,160 | 11,370,496 | 4.4515% | Rain / snowmelt |
| `kyrgyzstan2` | `train` | 170 | 44,239 | 2,785,280 | 1.5883% | Rain / snowmelt |
| `usa` | `train` | 28 | 4,510 | 458,752 | 0.9831% | Rainfall |
| **Train Split Total** | `train` | **1,596** | **878,975** | **26,148,864** | **3.3614%** | Mixed rainfall / monsoonal |
| `chimanimani` | `val` | 345 | 163,019 | 5,652,480 | 2.8840% | Cyclone Idai (2019) |
| `newzealand` | `val` | 508 | 522,250 | 8,323,072 | 6.2747% | Kaikoura Earthquake (2016) |
| **Val Split Total** | `val` | **853** | **685,269** | **13,975,552** | **4.9033%** | Mixed cyclone + co-seismic |
| `hokkaido` | `test` | 144 | 337,335 | 2,359,296 | 14.2981% | Iburi Earthquake (2018) |
| `italy` | `test` | 1,799 | 792,659 | 29,474,816 | 2.6893% | Rainfall (Central Italy) |
| **Test Split Total** | `test` | **1,943** | **1,129,994** | **31,834,112** | **3.5496%** | Mixed co-seismic + rainfall |

### 5.2 Scientific Finding on Validation Discrepancy
1. **The Kaikoura Earthquake Skew:** Co-seismic landslides triggered by major earthquakes (e.g. New Zealand 2016, Hokkaido 2018) produce extraordinarily dense clustering of scarps compared to rainfall-induced events.
2. New Zealand patches comprise $59.5\%$ ($508 / 853$) of validation patches and $76.2\%$ ($522,250 / 685,269$) of positive validation pixels, with an individual regional prevalence of $6.275\%$.
3. Chimanimani (Cyclone Idai) has a lower prevalence of $2.884\%$, closely matching training inventories like Hiroshima ($2.41\%$) and Kyrgyzstan 2 ($1.59\%$).
4. **Geographic Generalization Bounds:** The elevation of validation prevalence to $4.903\%$ is **not an artifact of data processing or leakage**, but an authentic geological domain shift inherent to disjoint geographic holdout partitioning. Consequently:
   - Models trained on predominantly rainfall-induced events (train prevalence $3.36\%$) encounter significantly higher target density in validation.
   - Any evaluation must account for this domain shift, and claims of universal geographic generalization must be appropriately bounded.

---

## 6. Normalization & Pipeline Verification

1. **Ownership Centralization:** Single-pass normalization is enforced strictly at the `GeoTrustNetPatchDataset` level when `normalize=True`.
2. **Training-Only Parameters:**
   - Sentinel-1 VV: Mean $\mu = -9.3099\text{ dB}$, Std $\sigma = 3.4868\text{ dB}$
   - Sentinel-1 VH: Mean $\mu = -15.8861\text{ dB}$, Std $\sigma = 4.1814\text{ dB}$
   - Sentinel-2: Scaled linearly by $\frac{1}{10000}$ to $[0, 1]$ reflectance
3. **Double-Normalization Prevention:** `load_sen12landslides_netcdf` defaults to `normalize=False`, and `metadata.extra_metadata["is_normalized"]` is tracked to ensure tensors are never normalized twice.
4. **Masked Computation:** Normalization is applied strictly over valid pixels ($v = \text{True}$); nodata/invalid pixels remain zeroed.

---

## 7. Model Forward Pass & Metric Qualification on Authentic Data

Using the authentic `chimanimani_1004` patch, both baseline and multimodal architectures were executed end-to-end:

### 7.1 Model A (SAR UNet)
- **Input:** Authentic VV + VH tensor $[B=1, 2, 128, 128]$
- **Output Logits:** $[1, 1, 128, 128]$ — all finite (`torch.isfinite().all() == True`)
- **Loss:** `CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)` with `valid_mask` = 0.9073 (BCE: 0.8568, Dice: 0.9578)
- **Gradients:** Backward pass successfully computed 44 parameter gradients; all gradient norms are finite.
- **Evaluation Metrics:**
  - `total_pixels`: 16,384
  - `nodata_pixels_excluded`: 0
  - `positive_prevalence`: 0.022095 (matches 362 / 16384)
  - `dice`: 0.030060 (untrained random initialization)
  - `iou`: 0.015260
  - `roc_auc`: 0.391050
  - `pr_auc`: 0.017600

### 7.2 Model D (GeoTrustNet Multimodal Spatio-Temporal)
- **Input:**
  - SAR: $[1, 5, 2, 128, 128]$
  - Optical: $[1, 5, 10, 128, 128]$
  - DEM: $[1, 2, 128, 128]$ (Elevation + derived slope)
  - Quality masks: `sar_valid`, `optical_valid`
- **Output Logits:** $[1, 1, 128, 128]$ — all finite
- **Reliability Weights:**
  - `weight_sar`: $[1, 1, 16, 16]$ — all finite
  - `weight_optical`: $[1, 1, 16, 16]$ — all finite
  - `weight_dem`: $[1, 1, 16, 16]$ — all finite

---

## 8. Qualification Scope vs Full Dataset Benchmark Readiness

| Capability | Phase 10C (Previous) | Phase 10D (Current Milestone) | Phase 11 (Next Milestone) |
| :--- | :--- | :--- | :--- |
| **Pipeline Tested On** | Synthetic Fixtures | **Authentic Sen12Landslides Files** | Full 50 GB Cloud Dataset |
| **Sample Count** | 6 synthetic | 1 authentic triad ($T=15$) | 4,392 authentic patches |
| **Target Validity Decoupling** | Code verified | **Verified on genuine raster** | Full-scale training |
| **SCL Quality Policy** | Hardcoded | **Configurable & Documented** | Configurable across dataset |
| **DEM Slope Calculation** | Missing / zeros | **Physically derived ($10\text{m}$ grid)** | Pre-extracted / on-the-fly |
| **Model A Execution** | Mocked in Phase 8 | **Live forward & backward verified** | Full GPU training & convergence |
| **Scientific Claim Readiness** | No claims | **Data Pipeline Qualified** | Full Benchmark Evaluation |

> [!IMPORTANT]
> **Scientific Integrity Notice:** Qualifying the data ingestion pipeline, loss calculation, and model execution on authentic Sen12Landslides NetCDF files does **not** constitute completing the full scientific benchmark. It proves that the codebase is formally ready for full-scale GPU training without data-loading crashes, shape mismatches, or normalization errors.

---

## 9. Verification & Test Suite Summary

A dedicated test suite `tests/test_geotrustnet_phase10d.py` was implemented, containing 9 comprehensive tests:
- `test_authentic_netcdf_file_integrity`
- `test_authentic_netcdf_loader_schema`
- `test_authentic_scl_configurable_policy`
- `test_authentic_dem_and_slope_derivation`
- `test_authentic_target_validity_mask_alignment`
- `test_authentic_dataset_single_pass_normalization`
- `test_model_a_forward_loss_and_metrics_on_authentic_data`
- `test_model_d_geotrustnet_multimodal_forward_on_authentic_data`
- `test_disjoint_manifest_prevalence_distribution_audit`

### Overall Test Suite Status
- **Test command:** `pytest tests/`
- **Result:** **186 passed in 28.85s (100% pass rate)**
- **Dashboard regression:** **6 passed in 3.06s** (all 7 legacy pages + GeoTrustNet intelligence research page intact)

---

## 10. Conclusion & Recommendation

Phase 10D successfully bridges the gap between synthetic scaffolding and real-world satellite data. The Sen12Landslides ingestion pipeline is now verified against genuine European Space Agency data files, the target validity mask is decoupled, optical cloud masking is configurable and justified, and model forward/backward passes execute without numerical instability.

The repository is now fully prepared for **Phase 11: Full-Scale Cloud GPU Benchmark Execution** on Colab/Kaggle or an attached GPU cluster.
