# Version 2.0 Candidate Hillslope-Control Extraction Protocol & Execution Guide

**Himalayan Landslide Intelligence Research Project**  
*Step 5.2 Scientific Governance, Geomorphometric Specifications & Turnkey Earth Engine Guide*

---

## 1. Executive Context & Scientific Governance

### 1.1 Negative Control Terminology & Inventory Incompleteness
In compliance with rigorous geospatial inventory governance, negative samples in Version 2.0 are designated:
$$\textbf{Candidate Hillslope Control / Unrecorded Failure Status}$$

**Scientific Limitation & Governance:**
- Absence from the NASA Global Landslide Catalog (COOLR) establishes *unrecorded failure status*, not guaranteed absolute stability across geological timescales.
- The NASA catalog is derived predominantly from media, government, and emergency response reports, carrying inherent reporting bias toward populated transport corridors.
- Controls represent geomorphologically matched hillslopes without recorded historical failure events, sampled within identical topographic envelopes as confirmed landslides.

### 1.2 The Flat-Terrain Negative Control Confound in Version 1.0
In the historical Version 1.0 dataset ($N=254$):
- Negative controls were selected using random spatial sampling within an unstratified 80 km adaptive box.
- Over $90\%$ of controls fell onto alluvial plains, valley floors, and agricultural lowlands ($< 5^\circ$ slope), with mean slope $1.43^\circ \pm 4.34^\circ$ and mean TRI $0.57 \pm 1.63\text{ m}$.
- Two-sample Kolmogorov-Smirnov tests confirm extreme divergence from positive landslide terrain:
  - Slope KS statistic: $0.923$ ($p = 1.24 \times 10^{-56}$)
  - TRI KS statistic: $0.907$ ($p = 6.18 \times 10^{-54}$)
  - Elevation KS statistic: $0.590$ ($p = 3.07 \times 10^{-20}$)

This demonstrated that Version 1.0 models ($>97\%$ AUROC) partially learned the trivial classification of mountains versus flat valley bottoms rather than hazard-prone versus stable hillslopes.

### 1.3 Confound Mitigation Policy (Empirical Verification Required)
- **Design Target:** Stratified sampling within the empirical hillslope envelope ($6.1^\circ \le \text{slope} \le 41.4^\circ$, $266\text{ m} \le \text{elevation} \le 3027\text{ m}$) is designed to mitigate the flat-terrain confound.
- **Scientific Rule:** Actual elimination of the terrain confound **cannot be assumed or claimed a priori from the sampling mask alone**. It must be empirically validated via two-sample Kolmogorov-Smirnov tests and feature distribution comparisons once genuine Version 2.0 extracted data is ingested.

---

## 2. Implemented Sampling Filters & Mathematical Specifications

The Colab extraction notebook (`notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb`) implements strict, sequential physical and data-integrity filters applied to every candidate point:

| Filter ID | Filter Name | Mathematical / Dataset Definition | Rejection Category Logged |
| :--- | :--- | :--- | :--- |
| **F1** | **Regional ROI** | $[26.0^\circ\text{N}, 35.0^\circ\text{N}]$, $[74.0^\circ\text{E}, 97.0^\circ\text{E}]$ | `roi_out_of_bounds` |
| **F2** | **Universal Exclusion Buffer** | Geodesic distance $> 2000\text{ m}$ from ALL 127 known positive events | `scar_proximity_under_2km` |
| **F3** | **Mandatory Terrain Features** | Explicit non-null checks on elevation, slope, aspect, curvature, TRI, TWI (no 0.0 fallbacks) | `missing_terrain_data` |
| **F4** | **Water Data Availability** | Explicit non-null check on JRC Global Surface Water (no `unmask(0)` silent pass) | `missing_water_data` |
| **F5** | **Land-Cover Availability** | Explicit non-null check on ESA WorldCover 10m | `missing_landcover_data` |
| **F6** | **Elevation Envelope** | $266\text{ m} \le \text{elevation} \le 3027\text{ m}$ ($P_{10}$ to $P_{95}$ of positive events) | `elevation_out_of_bounds` |
| **F7** | **Slope Envelope** | $6.1^\circ \le \text{slope} \le 41.4^\circ$ ($P_{10}$ to $P_{95}$ of positive events) | `slope_out_of_bounds` |
| **F8** | **Surface Water Exclusion** | JRC GSW Occurrence $\ge 10\%$ OR ESA WorldCover Class 80 (Permanent water) | `water_body_overlap` |
| **F9** | **Permanent Snow/Ice Mask** | ESA WorldCover 10m (v100) Class 70 (Snow and ice) | `snow_ice_overlap` |
| **F10** | **SAR Observation Count** | $N_{\text{acquisitions}} \ge 3$ in pre-event baseline window $[t_{\text{event}}-180\text{d}, t_{\text{event}}-15\text{d}]$ | `sar_insufficient_baseline` |
| **F11** | **SAR Temporal Screening** | Sentinel-1 IW GRD VV standard deviation $\sigma_{\text{VV}} \le 1.5\text{ dB}$ | `sar_temporal_instability` |

### 2.1 SAR Temporal Consistency Screening Criterion
- **Screening Role:** Pre-event backscatter temporal standard deviation ($\le 1.5\text{ dB}$) serves as a *microwave temporal consistency screening criterion*, not geodetic proof of physical ground stability.
- **Confounding Influences Documented:**
  - Seasonal freeze-thaw cycles and winter snow pack metamorphism.
  - Vegetation phenological shifts and seasonal biomass changes.
  - Episodic soil moisture pulses from monsoon rainfall.
  - Steep-slope radar geometry effects (local incidence angle variations, radar layover, and radar shadow).
- **Observation Window:** 180 days to 15 days prior to positive event date ($[t_{\text{event}}-180\text{d}, t_{\text{event}}-15\text{d}]$). The 15-day exclusion prevents leakage of potential precursory mass movement.
- **Sensor:** Sentinel-1 IW GRD, C-Band (5.405 GHz), VV polarization.

### 2.2 Corrected Terrain Ruggedness Index (Riley et al. 1999)
- **Standard Formulation:** Riley et al. (1999) calculates the root-sum-squared elevation differences between the central cell $z_0$ and its 8 neighboring cells ($3 \times 3$ kernel):
  $$\text{TRI}_{\text{Riley}} = \sqrt{ \sum_{i=1}^8 \left(z_i - z_0\right)^2 }$$
- **Mean Absolute Difference:** The Wilson & Gallant average absolute elevation difference across the 8 neighbors is also exported:
  $$\text{TRI}_{\text{mean\_diff}} = \frac{1}{8} \sum_{i=1}^8 |z_i - z_0|$$
- **Focal Mean Difference Proxy:** The legacy calculation $|z_0 - \text{mean}_{3\times 3}(z)|$ is renamed and retained strictly as `tri_focal_diff_proxy`. Unit tests in `tests/test_v2_sampling_and_features.py` prove that the focal mean difference severely underestimates ruggedness.

### 2.3 Corrected Topographic Wetness Index (Beven & Kirkby 1979)
- **Resolution Mismatch & Alignment Approach:**
  - SRTM elevation has a native resolution of 1 arc-second ($\approx 30\text{ m}$).
  - MERIT Hydro flow accumulation (`upa`) has a native resolution of 3 arc-seconds ($\approx 90\text{ m}$).
  - To eliminate spatial misalignment without distorting hydrology, MERIT Hydro `upa` is explicitly resampled via bilinear interpolation to match the 30m SRTM projection and grid:
    $$\text{upa}_{\text{30m}} = \text{reproject}(\text{upa}, \text{crs}=\text{SRTM}, \text{scale}=30, \text{resampling}=\text{'bilinear'})$$
  - At 30m grid cell resolution, contour width is $b = 30\text{ m}$, yielding specific catchment area:
    $$\alpha = \frac{\text{upa}_{\text{30m}} \times 10^6 \text{ m}^2}{30\text{ m}}$$
    $$\text{TWI} = \ln \left( \frac{\alpha}{\tan \beta} \right)$$
    where $\beta$ is slope in radians (bounded below by $\tan \beta \ge 0.001$).
- **Slope-Only Proxy:** The legacy calculation $-\ln(\tan \beta) = \ln(1 / \tan \beta)$ is renamed and retained strictly as `twi_slope_proxy`.

---

## 3. Candidate Selection Strategy & Matching Diagnostics

Rather than arbitrarily accepting the first candidate point that passes broad terrain bounds, Version 2.0 implements an **optimized candidate selection strategy**.

### 3.1 Selection Strategy: Minimizing Topographic Distance $D_{\text{topo}}$
For each positive event $(i)$:
1. A candidate batch is sampled within the $10\text{ km}-40\text{ km}$ ring buffer.
2. All points failing physical or data-integrity filters are rejected with logged tallies.
3. For all surviving valid candidates, a standardized multidimensional topographic distance $D_{\text{topo}}$ to the paired positive event is calculated using empirical positive event standard deviations ($\sigma_{\text{slope}} = 10.5^\circ$, $\sigma_{\text{elev}} = 775.0\text{ m}$, $\sigma_{\text{tri}} = 17.5$):
   $$D_{\text{topo}} = \sqrt{ \left(\frac{|\text{slope}_c - \text{slope}_p|}{10.5}\right)^2 + \left(\frac{|\text{elevation}_c - \text{elevation}_p|}{775.0}\right)^2 + \left(\frac{|\text{TRI}_c - \text{TRI}_p|}{17.5}\right)^2 }$$
4. The candidate control that **minimizes $D_{\text{topo}}$** is selected as the representative hillslope control.

### 3.2 Full Empirical Distribution Reporting
Instead of reporting single summary statistics, Step 5 exports the full distribution (Min, P25, Median, Mean, P75, Max) of matching differences to `v2_matching_distribution_summary.csv` for:
- `spatial_distance_km`
- `delta_slope_deg`
- `delta_elevation_m`
- `delta_aspect_deg`
- `delta_tri`
- `d_topo_composite`

### 3.3 Limitation Governance
> [!IMPORTANT]
> A $10\text{ km} - 40\text{ km}$ spatial ring buffer serves as a regional physiographic surrogate (capturing gross tectonic setting and monsoon climatology). It **does not guarantee identical hydrological catchment boundaries or identical bedrock lithology**. The matching diagnostics provide transparent quantification of remaining geomorphic differences.

---

## 4. Turnkey Google Colab Execution Instructions

Because Google Earth Engine requires interactive browser-based OAuth authentication under GCP project `landslidedetection-506008`, feature extraction is executed in Google Colab:

### Step-by-Step Procedure:
1. **Open Google Colab:** Navigate to [https://colab.research.google.com/](https://colab.research.google.com/).
2. **Open the Corrected Notebook:**
   Select **File -> Upload notebook**, and select:
   [`notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb`](file:///c:/Gourav%20Sharma/work/LANDSLIDE/notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb)
   *(Or open from GitHub: `https://github.com/GouravSharma26/LandSlide/blob/main/notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb`)*
3. **Run Cell 1 (Authentication):**
   Execute Cell 1 and click the authentication link. Log in with your authorized Google account and allow access for `landslidedetection-506008`.
4. **Run All Remaining Cells (Runtime -> Run all):**
   - **Step 2:** Automatically loads the 127 verified positive events from GitHub.
   - **Step 3:** Declares SRTM, Riley TRI, 30m-aligned MERIT Hydro TWI, ESA WorldCover, and JRC water layers (without `unmask(0)`).
   - **Step 4:** Executes candidate sampling with optimized selection minimizing $D_{\text{topo}}$, recording candidate counts, accepted counts, and rejection tallies across all 12 categories.
   - **Step 5:** Executes strict validation assertions and outputs full empirical matching distributions.
   - **Step 6 & 7:** Declares dynamic collections and multi-scale SAR / rainfall / spectral extractors with adaptive window fallbacks.
   - **Step 8:** Extracts dynamic features for all 127 controls (~25 to 35 minutes).
   - **Step 9:** Merges tables into `v2_feature_table_experimental.csv` and compiles `v2_extraction_provenance.json`.
   - **Step 10 & 11:** Triggers automatic browser download of all 5 artifacts:
     - `phase2_v2_hillslope_controls.csv`
     - `v2_control_matching_diagnostics.csv`
     - `v2_matching_distribution_summary.csv`
     - `v2_feature_table_experimental.csv`
     - `v2_extraction_provenance.json`
5. **Place Downloaded Artifacts into Local Repository:**
   Copy the five downloaded files to their target directories:
   - `outputs/phase2_v2_hillslope_controls.csv`
   - `artifacts/susceptibility/v2_control_matching_diagnostics.csv`
   - `artifacts/susceptibility/v2_matching_distribution_summary.csv`
   - `artifacts/susceptibility/v2_feature_table_experimental.csv`
   - `artifacts/metadata/v2_extraction_provenance.json`

---

## 5. Post-Extraction Scientific Validation & Promotion Protocol

Upon receipt of the genuine extracted Version 2.0 files, the following protocol will be executed:
1. **Distribution Validation:** Run Kolmogorov-Smirnov tests comparing positive event distributions against extracted controls. Confound mitigation is established only if slope and TRI distributions align without statistically extreme divergence.
2. **Model Retraining:** Execute `python scripts/build_research_artifacts.py --dataset v2` to train:
   - Workflow A: Event Detection (100 km spatial purge).
   - Workflow B: Susceptibility Core, Optical-Enhanced, and Reduced models under identical purged folds.
3. **Governance Audit:** The 43 automated tests in `tests/` will verify zero prohibited features, zero temporal leakage, and genuine spatial fold separation.
4. **Promotion:** Version 2.0 will be promoted to the primary research benchmark on the dashboard only after this full empirical validation is satisfied.
