# Version 2.0 Candidate Hillslope-Control Extraction Protocol & Execution Guide

**Himalayan Landslide Intelligence Research Project**  
*Step 5.3 Final Sampling Logic, Hydrological Integrity Audit & Turnkey Earth Engine Guide*

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

## 2. Implemented Sampling Filters & Hydrological Specifications

The Colab extraction notebook (`notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb`) implements strict, sequential physical and data-integrity filters applied to every candidate point:

| Filter ID | Filter Name | Mathematical / Dataset Definition | Rejection Category Logged |
| :--- | :--- | :--- | :--- |
| **F1** | **Regional ROI** | $[26.0^\circ\text{N}, 35.0^\circ\text{N}]$, $[74.0^\circ\text{E}, 97.0^\circ\text{E}]$ | `roi_out_of_bounds` |
| **F2** | **Universal Exclusion Buffer** | Geodesic distance $> 2000\text{ m}$ from ALL 127 known positive events | `scar_proximity_under_2km` |
| **F3** | **Mandatory Terrain Features** | Explicit non-null checks on elevation, slope, aspect, curvature, TRI, TWI (no defaults) | `missing_terrain_data` |
| **F4** | **Water Data Availability** | Explicit non-null check on JRC Global Surface Water (no `unmask(0)` silent pass) | `missing_water_data` |
| **F5** | **Land-Cover Availability** | Explicit non-null check on ESA WorldCover 10m | `missing_landcover_data` |
| **F6** | **Elevation Envelope** | $266\text{ m} \le \text{elevation} \le 3027\text{ m}$ ($P_{10}$ to $P_{95}$ of positive events) | `elevation_out_of_bounds` |
| **F7** | **Slope Envelope** | $6.1^\circ \le \text{slope} \le 41.4^\circ$ ($P_{10}$ to $P_{95}$ of positive events) | `slope_out_of_bounds` |
| **F8** | **Surface Water Exclusion** | JRC GSW Occurrence $\ge 10\%$ OR ESA WorldCover Class 80 (Permanent water) | `water_body_overlap` |
| **F9** | **Permanent Snow/Ice Mask** | ESA WorldCover 10m (v100) Class 70 (Snow and ice) | `snow_ice_overlap` |
| **F10** | **SAR Observation Count** | $N_{\text{acquisitions}} \ge 3$ in pre-event baseline window $[t_{\text{event}}-180\text{d}, t_{\text{event}}-15\text{d}]$ | `sar_insufficient_baseline` |
| **F11** | **SAR Temporal Screening** | Sentinel-1 IW GRD VV standard deviation $\sigma_{\text{VV}} \le 1.5\text{ dB}$ | `sar_temporal_instability` |
| **F12** | **Coordinate Deduplication** | Rejection of candidates within 30m of previously evaluated candidate | `duplicate_candidate_coordinates` |

### 2.1 Hydrological Resolution Treatment in TWI (Beven & Kirkby 1979)
- **Resolution Mismatch & Physical Justification:**
  - SRTM digital elevation model has a native resolution of 1 arc-second ($\approx 30\text{ m}$).
  - MERIT Hydro flow accumulation (`upa`) has a native resolution of 3 arc-seconds ($\approx 90\text{ m}$).
  - Flow accumulation represents cumulative upstream topological routing. Artificially interpolating a 90m flow accumulation raster down to 30m using bilinear or nearest-neighbor interpolation does *not* create genuine 30m hydrological flow paths; it merely introduces numerical smoothing artifacts.
- **Scientifically Consistent Native Grid Implementation:**
  - Topographic Wetness Index is calculated on the **native 90m MERIT Hydro hydrological grid**.
  - Total contributing area $A = \text{upa} \times 10^6\text{ m}^2$.
  - Specific catchment area is computed using the native metric contour width $b = 90.0\text{ m}$:
    $$\alpha = \frac{\text{upa} \times 10^6 \text{ m}^2}{90.0\text{ m}} = \frac{A}{b} \quad [\text{units: m}^2/\text{m} = \text{m}]$$
  - SRTM 30m slope is spatially aggregated (mean reduction) to the 90m hydrological cell to represent the mean hillslope gradient $\bar{\beta}$ over that cell:
    $$\beta = \bar{\beta}_{\text{90m}} \times \left(\frac{\pi}{180.0}\right)$$
    $$\tan\beta = \max(\tan\beta, 0.001) \quad [\text{dimensionless}]$$
    $$\text{TWI} = \ln \left( \frac{\alpha}{\tan\beta} \right) \quad [\text{dimensionless topographic index}]$$
  - Slope-only proxy is computed on the same 90m cell for baseline comparison:
    $$\text{TWI}_{\text{slope\_proxy}} = -\ln(\tan\beta)$$

### 2.2 SAR Temporal Consistency Screening & Asymmetry Analysis
- **Screening Role:** Pre-event backscatter temporal standard deviation ($\le 1.5\text{ dB}$) serves as a *microwave temporal consistency screening criterion*, not geodetic proof of physical ground stability.
- **Confounding Influences Documented:**
  - Seasonal freeze-thaw cycles and snow metamorphism.
  - Vegetation phenological shifts and seasonal biomass changes.
  - Episodic soil moisture pulses from monsoon rainfall.
  - Steep-slope radar geometry effects (local incidence angle variations, radar layover, and radar shadow).
- **Matching Asymmetry & Selection Bias Analysis:**
  - Positive events from NASA COOLR are externally reported landslides whose pre-event radar consistency was not screened.
  - Applying this filter only to controls creates potential selection asymmetry: if unstable slopes naturally exhibit higher radar variance prior to failure (e.g. from moisture fluctuations or surface creep), screening controls might favor stable mature forest over bare rock or scree.
  - **Sensitivity Analysis Design:** In post-extraction validation, models will be evaluated both with and without this screening criterion to quantify whether asymmetric screening induces a performance or feature importance artifact.

---

## 3. Strict Positive Validation & Multi-Batch Candidate Optimization

### 3.1 Strict Positive-Event Terrain Validation (No Fallbacks)
Prior to matching:
- Positive events are verified against authoritative ROI bounds $[26.0^\circ, 35.0^\circ\text{N}]$, $[74.0^\circ, 97.0^\circ\text{E}]$.
- Terrain features (`slope`, `elevation`, `aspect`, `tri`, `curvature`, `twi`) are extracted directly from `static_diag` at the positive event coordinate.
- **Zero Fallback Policy:** Any positive event with missing terrain features (e.g. SRTM alpine void or boundary clipping) is strictly excluded from matching. It is recorded in `v2_positive_events_excluded_audit.csv` with the exact missing fields. Arbitrary defaults (e.g. `slope=21.0`, `elev=1200m`) are prohibited.

### 3.2 Multi-Batch Candidate Optimization across All 3 Batches
To ensure representative hillslope controls:
1. Three sequential batches of candidate points (25 points each, total 75 candidates) are sampled within the $10\text{ km}-40\text{ km}$ ring buffer.
2. The loop **does not exit early** upon finding valid points in batch 1; all three batches are fully evaluated.
3. Candidates are deduplicated by coordinate proximity ($30\text{ m}$ tolerance).
4. For all surviving unique valid candidates, the standardized three-feature topographic distance is calculated using positive-event standard deviations ($\sigma_{\text{slope}} = 10.5^\circ$, $\sigma_{\text{elev}} = 775.0\text{ m}$, $\sigma_{\text{tri}} = 17.5$):
   $$\text{three\_feature\_topographic\_distance} = \sqrt{ \left(\frac{|\text{slope}_c - \text{slope}_p|}{10.5^\circ}\right)^2 + \left(\frac{|\text{elevation}_c - \text{elevation}_p|}{775.0\text{ m}}\right)^2 + \left(\frac{|\text{TRI}_c - \text{TRI}_p|}{17.5}\right)^2 }$$
5. The candidate minimizing this standardized distance across the combined 3-batch candidate set is selected.
6. The result is documented as the **minimum-distance candidate among sampled valid candidates**, not a claim of global optimality.

### 3.3 Enhanced Matching Diagnostics & Cross-Event Duplicate Audit
In `v2_control_matching_diagnostics.csv`:
- `three_feature_topographic_distance` is explicitly reported.
- Independent deltas are recorded for `delta_slope_deg`, `delta_elevation_m`, `delta_aspect_deg`, `delta_tri`, `delta_curvature`, and `delta_twi`.
- **Cross-Event Duplicate Audit:** Checks whether the same hillslope candidate was selected for multiple positive events, reporting duplicate count and minimum pairwise separation distance across all accepted controls.
- **Full Empirical Distribution:** Min, P25, Median, Mean, P75, and Max are exported to `v2_matching_distribution_summary.csv`.

---

## 4. Turnkey Google Colab Execution Instructions

Because Google Earth Engine requires interactive OAuth authentication under GCP project `landslidedetection-506008`, extraction is executed in Google Colab:

### Step-by-Step Procedure:
1. **Open Google Colab:** Navigate to [https://colab.research.google.com/](https://colab.research.google.com/).
2. **Open the Corrected Notebook:**
   Select **File -> Upload notebook**, and select:
   [`notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb`](https://github.com/GouravSharma26/LandSlide/blob/main/notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb)
3. **Run Cell 1 (Authentication):**
   Execute Cell 1 and log in with your authorized Google account for `landslidedetection-506008`.
4. **Run All Remaining Cells (Runtime -> Run all):**
   - **Step 2:** Loads confirmed positive events and declares the 2.0 km universal exclusion zone.
   - **Step 3:** Declares SRTM, Riley TRI, native 90m MERIT Hydro TWI, ESA WorldCover, and masked JRC water.
   - **Step 4:** Validates positive events without defaults, screens 3 batches per event, deduplicates coordinates, and selects min-distance controls across all 13 rejection categories.
   - **Step 5:** Executes validation assertions, duplicate control audits, and exports empirical matching percentiles.
   - **Step 6 & 7:** Declares dynamic collections and multi-scale SAR / rainfall / spectral extractors with adaptive window fallbacks.
   - **Step 8:** Extracts dynamic features for all accepted controls (~25 to 35 minutes).
   - **Step 9:** Merges tables into `v2_feature_table_experimental.csv` and compiles `v2_extraction_provenance.json`.
   - **Step 10 & 11:** Triggers automatic browser download of all 6 exported artifacts:
     1. `phase2_v2_hillslope_controls.csv`
     2. `v2_control_matching_diagnostics.csv`
     3. `v2_positive_events_excluded_audit.csv`
     4. `v2_matching_distribution_summary.csv`
     5. `v2_feature_table_experimental.csv`
     6. `v2_extraction_provenance.json`
5. **Place Downloaded Artifacts into Local Repository:**
   Copy the six downloaded files to:
   - `outputs/phase2_v2_hillslope_controls.csv`
   - `artifacts/susceptibility/v2_control_matching_diagnostics.csv`
   - `artifacts/susceptibility/v2_positive_events_excluded_audit.csv`
   - `artifacts/susceptibility/v2_matching_distribution_summary.csv`
   - `artifacts/susceptibility/v2_feature_table_experimental.csv`
   - `artifacts/metadata/v2_extraction_provenance.json`

---

## 5. Test Suite Categorization & Integrity

The test suite in [`tests/`](file:///c:/Gourav%20Sharma/work/LANDSLIDE/tests/) is structured into distinct, transparent categories:

1. **Category 1: Pure Python Unit Tests (Mocked / Offline Logic)**
   - Multi-batch candidate selection helpers (`select_best_candidate_across_batches`)
   - TWI dimensional analysis and monotonicity (`calculate_twi_hydrological_native_grid`)
   - Positive-event terrain validation without fallbacks (`validate_positive_event`)
   - Candidate screening filters (`evaluate_candidate_point`)
   - Cross-event duplicate control detection
2. **Category 2: Notebook Structure & Governance Tests (Static AST / JSON Inspection)**
   - Confirms zero arbitrary fallback defaults exist in the Colab notebook
   - Confirms candidate search does not break prematurely on batch 1
   - Confirms MERIT Hydro 90m native grid and slope aggregation are implemented
   - Confirms all 13 rejection categories are tracked in audit schemas
   - Confirms candidate terminology and immutable V1 historical baseline
3. **Category 3: Earth Engine Remote Integration Tests**
   - Explicitly documented as pending actual authenticated Google Colab execution. No mock GEE execution is claimed locally.
