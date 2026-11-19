# Phase 8 Pre-Flight Audit Report — Real Dataset Validation & Cloud Benchmarking

**Project:** Himalayan Landslide Intelligence System (GeoTrustNet)  
**Document:** Pre-Flight Audit for Real-Data Sen12Landslides Training & Benchmarking  
**Date:** 2026-11-19  
**Status:** Audit Passed — Execution Ready  

---

## 1. Executive Summary

This pre-flight audit inspects the complete end-to-end GeoTrustNet training and evaluation pipeline prior to executing real-data experiments on the **Sen12Landslides** dataset.

Through direct inspection of the official Hugging Face repository (`paulhoehn/Sen12Landslides`) and official GitHub repository (`PaulH97/Sen12Landslides`), the actual native dataset format, band ordering, split structure, and normalization coefficients have been audited against GeoTrustNet's interfaces.

A critical scientific discovery was identified: the official author split (`tasks/S12LS-LD/harmonized/splits_aligned.json`) distributes patches from all 11 regional inventories across Train, Val, and Test splits simultaneously, creating spatial autocorrelation leakage. GeoTrustNet's evaluation protocol remedies this by enforcing strict **disjoint geographic inventory holdouts**.

---

## 2. Model Architecture Input/Output Contracts

All four benchmark models implement strictly validated, compatible tensor input/output interfaces:

| Model ID | Class Name | Input Modalities | Input Tensor Shapes | Output Tensor Shapes |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | `UNet` | Sentinel-1 SAR (VV, VH) | `sar`: $[B, 2, 128, 128]$ | `logits`: $[B, 1, 128, 128]$ |
| **Model B** | `TemporalSARUNet` | Sentinel-1 SAR Sequence | `sar`: $[B, T, 2, 128, 128]$<br>`valid_mask`: $[B, T, 1, 128, 128]$ | `logits`: $[B, 1, 128, 128]$ |
| **Model C** | `MultimodalFusionUNet` | SAR, Optical, DEM | `sar`: $[B, 2, 128, 128]$<br>`optical`: $[B, 10, 128, 128]$<br>`dem`: $[B, 2, 128, 128]$<br>`flags`: $(v_{\text{sar}}, v_{\text{opt}}, v_{\text{dem}})$ | `logits`: $[B, 1, 128, 128]$ |
| **Model D** | `GeoTrustNet` | SAR, Optical, DEM | `sar`: $[B, 2, 128, 128]$<br>`optical`: $[B, 10, 128, 128]$<br>`dem`: $[B, 2, 128, 128]$<br>`flags`: $(v_{\text{sar}}, v_{\text{opt}}, v_{\text{dem}})$ | `logits`: $[B, 1, 128, 128]$<br>`reliability_weights`: $[B, 3, 128, 128]$ |

---

## 3. Dataset Specification Audit: Real Sen12Landslides Format

### 3.1 Native File Packaging & Structure
- **Storage Format:** Compressed tarballs containing NetCDF4 (`.nc`) files partitioned into `data_harmonized/s1asc/`, `data_harmonized/s1dsc/`, and `data_harmonized/s2/`.
- **Dimensions per NetCDF file:** `time` (1 to 15 acquisitions), `y` ($H=128$), `x` ($W=128$).
- **Variables in `s1asc` / `s1dsc`:** `VV`, `VH`, `DEM`, `MASK`, `spatial_ref`.
- **Variables in `s2`:** `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11`, `B12`, `SCL`, `DEM`, `MASK`, `spatial_ref`.

### 3.2 Ground-Truth Target Mask
- Variable: `MASK`
- Type: Integer/Binary
- Rule: $(MASK > 0)$ represents confirmed landslide scarp pixels ($1$); $0$ represents stable terrain.

### 3.3 Official Normalization Parameters (Extracted from `norm_aligned.json`)
The dataset authors provide channel-specific means and standard deviations derived across the global dataset:
- **Sentinel-1 SAR:**
  - $VV$: Mean = $-9.310$ dB, Std = $3.487$ dB
  - $VH$: Mean = $-15.886$ dB, Std = $4.181$ dB
- **Copernicus DEM:**
  - Elevation: Mean = $794.389$ m, Std = $657.544$ m
  - Slope: Calculated on-the-fly via numpy/rasterio gradient or normalized by $90^\circ$.
- **Sentinel-2 Optical (Surface Reflectance):**
  - $B02$: Mean = $1438.1$, Std = $2375.1$
  - $B03$: Mean = $1620.8$, Std = $2226.4$
  - $B04$: Mean = $1543.6$, Std = $2224.8$
  - $B05$: Mean = $2027.7$, Std = $2038.6$
  - $B06$: Mean = $3068.6$, Std = $1877.0$
  - $B07$: Mean = $3391.9$, Std = $1888.8$
  - $B08$: Mean = $3507.5$, Std = $1899.7$
  - $B8A$: Mean = $3589.8$, Std = $1918.4$
  - $B11$: Mean = $2275.7$, Std = $1667.9$
  - $B12$: Mean = $1512.7$, Std = $1481.2$

---

## 4. Critical Scientific Finding: Spatial Leakage in Default Author Splits

An audit of `tasks/S12LS-LD/harmonized/splits_aligned.json` reveals:
- Total aligned samples: 4,392 (Train: 2,810; Val: 703; Test: 879).
- Total inventories: 11 global regions (`china`, `newzealand`, `indonesia`, `itogon`, `kyrgyzstan2`, `hokkaido`, `chimanimani`, `kyrgyzstan1`, `hiroshima`, `usa`, `italy`).
- **Audit Result:** **100% of the 11 inventories appear simultaneously in Train, Val, and Test.**
- **Consequence:** The official baseline splits sample patches randomly across space within the same event footprint. Models evaluated under this split benefit from strong spatial autocorrelation (nearby pixels sharing identical topography and rainfall), leading to over-optimistic evaluation metrics.
- **GeoTrustNet Resolution:** To evaluate true geographic generalization, GeoTrustNet partitions by **holding out complete inventories**:
  - **Train Inventories (7):** `china`, `indonesia`, `itogon`, `kyrgyzstan1`, `kyrgyzstan2`, `hiroshima`, `usa`
  - **Validation Inventories (2):** `newzealand`, `chimanimani`
  - **Held-Out Test Inventories (2):** `italy`, `hokkaido`

---

## 5. Pre-Flight Checklist Status

| Audit Item | Status | Verification Detail |
| :--- | :--- | :--- |
| **Model Architectures** | Verified | All 4 models compile and run forward/backward passes cleanly |
| **Loss Formulations** | Verified | `CombinedBCEDiceLoss` numerically stable with epsilon safeguards |
| **Colab Notebook** | Verified | 44 cells covering all 27 required sections; runs in Colab environment |
| **Kaggle Notebook** | Verified | 44 cells with `/kaggle/input` and `/kaggle/working` directories configured |
| **Dataset Loader** | Ready | Enhanced to support both `.pt` tensors and native `.nc` NetCDF patches |
| **Uncertainty Heads** | Verified | Monte Carlo Dropout active in `model.train()` mode; computes variance and ECE |
| **Leakage Defense** | Verified | Disjoint inventory partitioning active and automated |

**Conclusion:** Pre-flight audit is passed. Ready for pilot dataset verification and controlled baseline execution.
