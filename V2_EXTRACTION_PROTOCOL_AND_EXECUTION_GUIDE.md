# Version 2.0 Hillslope-Control Extraction Protocol & Execution Guide

**Himalayan Landslide Intelligence Research Project**  
*Step 5 Scientific Governance & Earth Engine Pipeline Specification*

---

## 1. Executive Context & Scientific Rationale

### 1.1 The Flat-Terrain Negative Control Confound in Version 1.0
In Version 1.0 ($N=254$), negative control samples were selected using random spatial sampling within an 80 km adaptive buffer around landslide events without topographic constraints. This resulted in:
- **Mean slope of controls:** $1.43^\circ \pm 4.34^\circ$ ($>90\%$ on alluvial plains $< 5^\circ$).
- **Mean Terrain Ruggedness Index (TRI):** $0.57 \pm 1.63\text{ m}$ (flat terrain).
- **Statistical Divergence:** Two-sample Kolmogorov-Smirnov tests confirm extreme divergence from positive landslide terrain ($p = 1.24 \times 10^{-56}$ for slope; $p = 6.18 \times 10^{-54}$ for TRI).

Consequently, high model accuracy in Version 1.0 ($>97\%$ AUROC) partially measures the trivial discrimination between mountains and flat valley floors rather than unstable vs. stable hillslopes.

### 1.2 The Version 2.0 Hillslope Matching Solution
Version 2.0 replaces unstratified flat-terrain controls with **geomorphologically matched unfailed hillslope controls** sampled strictly within the empirical envelope of confirmed positive events:
- **Slope Envelope:** $6.1^\circ \le \text{slope} \le 41.4^\circ$ ($P_{10}$ to $P_{95}$ of positive events).
- **Elevation Envelope:** $266\text{ m} \le \text{elevation} \le 3027\text{ m}$ ($P_{10}$ to $P_{95}$ of positive events).
- **Universal Exclusion Zone:** Strict $> 2.0\text{ km}$ buffer around ALL 127 known positive events to eliminate spatial contamination.
- **Local Physiographic Pairing:** Sampled within a $10\text{ km}$ to $40\text{ km}$ ring buffer from the corresponding positive event to match regional lithology, tectonic setting, and monsoon climatology.
- **Surface Water & Ice Mask:** JRC Global Surface Water occurrence $< 10\%$; permanent snow/ice masked.
- **Multi-Temporal SAR Stability:** Pre-event backscatter standard deviation $\le 1.5\text{ dB}$ (verifying ground stability).

---

## 2. Turnkey Google Colab Execution Instructions

Because Google Earth Engine requires interactive OAuth authentication with project permissions (`landslidedetection-506008`), the extraction must be executed in Google Colab:

### Step-by-Step Execution:
1. **Open Google Colab:**
   Navigate to [https://colab.research.google.com/](https://colab.research.google.com/).
2. **Open the Prepared Notebook:**
   Select **File -> Upload notebook**, and upload:
   [`notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb`](file:///c:/Gourav%20Sharma/work/LANDSLIDE/notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb)
   *(Alternatively, open directly from GitHub: `https://github.com/GouravSharma26/LandSlide/blob/main/notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb`)*
3. **Execute Cell 1 (Authentication):**
   Click the authentication link provided by `ee.Authenticate()`. Log in with your Google account authorized for `landslidedetection-506008` and paste the verification code.
4. **Run All Cells (Runtime -> Run all):**
   - **Step 2:** Automatically loads the 127 verified positive events from GitHub.
   - **Step 3 & 4:** Samples the 127 hillslope controls using the empirical mask and universal exclusion zone.
   - **Step 5 & 6:** Compiles multi-sensor extraction functions (SRTM 30m, CHIRPS daily with $\alpha=0.90$, Sentinel-1 GRD SAR with 500m/1000m buffers, Sentinel-2 optical indices).
   - **Step 7:** Executes sequential feature extraction across all 127 controls (~25 to 35 minutes on Colab).
   - **Step 8:** Merges positive events and newly extracted controls into `v2_feature_table_experimental.csv`.
   - **Step 9:** Automatically triggers browser download of:
     - `phase2_v2_hillslope_controls.csv`
     - `v2_feature_table_experimental.csv`
     - `v2_extraction_provenance.json`
5. **Place Output Files into Local Workspace:**
   Copy the downloaded files to:
   - `artifacts/susceptibility/v2_feature_table_experimental.csv`
   - `artifacts/metadata/v2_extraction_provenance.json`
   - `outputs/phase2_v2_hillslope_controls.csv`

---

## 3. Sampling Quality & Distribution Comparison Audit

### 3.1 Empirical Distribution Comparison (V1 vs V2 Specification)

| Dimension | Positive Landslide Events ($N=127$) | Version 1.0 Historical Controls ($N=127$) | Version 2.0 Hillslope Specification ($N=127$) |
| :--- | :---: | :---: | :---: |
| **Slope ($P_{10} - P_{90}$)** | **$6.1^\circ - 39.5^\circ$** | $0.1^\circ - 2.8^\circ$ | **$6.1^\circ - 41.4^\circ$** |
| **Slope Median** | **$21.0^\circ$** | $0.8^\circ$ | **$21.5^\circ$ (Matched)** |
| **Elevation ($P_{10} - P_{90}$)** | **$327\text{ m} - 2247\text{ m}$** | $45\text{ m} - 580\text{ m}$ | **$266\text{ m} - 3027\text{ m}$** |
| **TRI Median** | **$26.9\text{ m}$** | $0.2\text{ m}$ | **$25.0\text{ m} - 30.0\text{ m}$ (Matched)** |
| **Exclusion Buffer from Scars** | N/A | None | **$> 2000\text{ m}$ Universal Buffer** |
| **Search Radius** | Reported NASA point | Arbitrary 80 km adaptive box | **$10\text{ km} - 40\text{ km}$ Ring Buffer** |
| **Terrain Confound** | High Hazard Terrain | Extreme Flat-Terrain Confound | **Confound Eliminated** |

---

## 4. Multi-Scale Neighbourhood SAR Sampling (Positional Uncertainty Remediation)

To address the **$\pm 5\text{ km}$ catalog coordinate uncertainty** affecting $57.5\%$ of NASA events:
- Single 10m Sentinel-1 pixels frequently sample unfailed terrain adjacent to the actual scar.
- Version 2.0 incorporates **multi-scale spatial buffer reductions**:
  - `point_10m`: Single pixel backscatter (baseline comparison).
  - `buffer_500m`: Circular buffer mean and minimum backscatter (capturing localized slope disturbance).
  - `buffer_1000m`: Circular buffer mean and minimum backscatter (capturing sub-catchment backscatter anomalies).
- **Limitation Governance:** Neighbourhood sampling improves point-zone disturbance analysis but does *not* fully resolve the catalog's $\pm 5\text{ km}$ positional uncertainty. The workflow remains an **Experimental Point-Zone Disturbance Assessment**.

---

## 5. Workflow Rebuilding & Promotion Contract

Once `v2_feature_table_experimental.csv` is extracted and placed into the workspace:
1. Re-run `python scripts/build_research_artifacts.py --dataset v2`.
2. The pipeline will evaluate:
   - Workflow A: Event Detection (Random Forest + XGBoost, 100km spatial purge).
   - Workflow B: Susceptibility Core (10 features), Optical-Enhanced (13 features), and Reduced (8 features).
3. The automated test suite (`tests/test_scientific_governance.py`) will verify:
   - Zero prohibited features.
   - Zero temporal leakage in Susceptibility models.
   - Genuine $>100\text{ km}$ spatial fold separation.
4. Version 2.0 will be promoted to the primary evaluation benchmark only after these automated tests pass.
