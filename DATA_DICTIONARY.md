# Comprehensive Data Dictionary: Himalayan Landslide Intelligence

**Project:** Himalayan Landslide Intelligence – Rainfall-Triggered Landslide Detection & Susceptibility Analytics  
**Coordinate Reference System:** WGS84 (EPSG:4326) for geospatial coordinates; Projected Azimuthal Equidistant centered at $(30.5^\circ\text{N}, 85.5^\circ\text{E})$ for spatial distance and clustering.  
**Spatial Bounding Box:** Latitude $[26.0^\circ\text{N}, 35.0^\circ\text{N}]$, Longitude $[74.0^\circ\text{E}, 97.0^\circ\text{E}]$ (~2.2M $\text{km}^2$).

---

## 1. Primary Datasets Overview

| Dataset Identifier | File Path | Total Rows | Columns | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Landslide Inventory** | `Landslide_Reports.csv` | 14,750 | 32 | Global Landslide Catalog (NASA COOLR) compiled from media, government reports, and regional databases (1956–2018). |
| **Cleaned Observation Sample** | `outputs/phase2_cleaned_dataset.csv` | 254 | 8 | Filtered Himalayan subset (127 positive events + 127 locally buffered hard negatives) with 1:1 balance. |
| **Multisource Feature Table** | `outputs/phase3_feature_table.csv` | 254 | 28 | Combined point-level dataset containing static topographic, optical, SAR backscatter, and precipitation features. |
| **Spatial Model Metrics** | `outputs/phase4_metrics_comparison.csv` | 2 | 8 | Cross-validation performance comparison between Full and Reduced feature sets for RF and XGBoost. |
| **Out-of-Fold Predictions** | `outputs/phase4_oof_predictions.csv` | 254 | 8 | Cross-validation spatial out-of-fold prediction probabilities and cluster fold IDs. |
| **Dynamic Grid Checkpoint** | `outputs/phase5_dynamic_grid_checkpoint_0.5deg.csv` | 828 | 15 | Intermediate regional 0.5-degree grid feature extraction checkpoint from Google Earth Engine. |
| **Regional Susceptibility Grid** | `outputs/phase5_susceptibility_grid.csv` | 765 | 7 | Final 0.5-degree susceptibility inference grid covering the Himalayan belt with ensemble scores and uncertainties. |

---

## 2. Table-by-Table Schema Specifications

### 2.1. Raw Landslide Inventory (`Landslide_Reports.csv`)

| Column Name | Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `OBJECTID` | Integer | 0 | Unique internal sequential database record identifier. |
| `Name of Information Source` | String | 0 | Organization or news agency reporting the incident (e.g., NASA, GSI, NDTV). |
| `Link to Information Source` | String | 843 | URL link to the original news story or scientific report. |
| `Event ID` | Integer | 0 | NASA Global Landslide Catalog unique event identification number. |
| `Event Date` | String | 1,637 | Timestamp of landslide occurrence formatted as `DD/MM/YYYY, HH:MM`. |
| `Event Time (approximate)` | String | 8,169 | Approximate time of day (e.g., morning, overnight, 14:30). |
| `Event Title` | String | 698 | Short descriptive headline summarizing location and incident. |
| `Event Description` | String | 2,511 | Free-text narrative detailing damages, casualties, road closures, and triggers. |
| `Location Description` | String | 1,529 | Specific place name, road stretch, or geographic landmark. |
| `Location Accuracy` | String | 30 | Positional uncertainty category: `known` ($\le 1\text{ km}$), `1km` to `5km`, `10km`, `25km`, `50km`, `100km`. |
| `Landslide Category` | String | 3 | Geomorphological classification (e.g., `mudslide`, `rock_fall`, `debris_flow`, `landslide`). |
| `Landslide Trigger` | String | 4 | Reported causal trigger (e.g., `rain`, `downpour`, `monsoon`, `continuous_rain`, `earthquake`). |
| `Estimated Size` | String | 28 | Approximate scale category (`small`, `medium`, `large`, `very_large`). |
| `Landslide Setting` | String | 33 | Physical environment (e.g., `above_road`, `natural_slope`, `mine`, `urban`). |
| `Number of Fatalities` | Float | 3,864 | Count of confirmed deaths caused by the event. |
| `Number of Injuries` | Float | 8,382 | Count of confirmed injuries caused by the event. |
| `Associated Storm Name` | String | 14,065 | Name of associated tropical cyclone or meteorological depression (if applicable). |
| `Link to Photo` | String | 12,519 | Hyperlink to photographic documentation. |
| `Event Comments` | String | 12,163 | Supplementary remarks logged by NASA curators. |
| `Imported Event Source Catalog` | String | 0 | Ingestion catalog tag (e.g., `GLC`, `Kirschbaum et al.`, `Disaster Alert`). |
| `Imported Event Source Id` | String | 3,061 | External database tracking ID from partner institutions. |
| `Latitude` | Float | 27 | Geographic latitude in decimal degrees (WGS84). Range: $[-46.775, 72.628]$. |
| `Longitude` | Float | 27 | Geographic longitude in decimal degrees (WGS84). Range: $[-179.981, 179.991]$. |
| `Country Name` | String | 82 | ISO administrative country name where event occurred. |
| `Country Code` | String | 84 | Two-letter ISO 3166-1 alpha-2 country code. |
| `Administrative Division` | String | 109 | Primary subnational division (state/province/district). |
| `Closest Gazetteer Point` | String | 53 | Named settlement or gazetteer entity closest to the coordinates. |
| `Distance to Gazetteer Point` | Float | 53 | Distance in kilometers to the closest gazetteer entity. |
| `Submitted Date` | String | 1,156 | Date the report was entered into the database. |
| `last_edited_date` | String | 14,724 | Date of last curator modification or validation. |
| `x` | Float | 0 | Projected or cartographic coordinate (longitude copy). |
| `y` | Float | 0 | Projected or cartographic coordinate (latitude copy). |

---

### 2.2. Cleaned Observation Sample (`outputs/phase2_cleaned_dataset.csv`)

| Column Name | Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `latitude` | Float | 0 | Latitude in decimal degrees (WGS84, $[26.078, 35.428]^\circ\text{N}$). |
| `longitude` | Float | 0 | Longitude in decimal degrees (WGS84, $[74.116, 96.642]^\circ\text{E}$). |
| `event_date` | String | 0 | Timestamp formatted as `YYYY-MM-DD HH:MM:SS` (Range: 2017-01-06 to 2020-08-31). |
| `Location Accuracy` | String | 0 | Positional precision category from NASA inventory ($\le 5\text{ km}$ retained). |
| `Landslide Trigger` | String | 0 | Rainfall trigger category (`rain`, `monsoon`, `downpour`, `continuous_rain`, or synthetic paired match). |
| `Imported Event Source Catalog` | String | 0 | Originating source catalog name (e.g., `GLC`). |
| `label` | Integer | 0 | Ground truth binary indicator: `1` = Confirmed landslide event; `0` = Stable hard negative. |
| `block_id` | String | 0 | $0.5^\circ \times 0.5^\circ$ spatial block identifier (`lon_lat` bin format). |

---

### 2.3. Multisource Feature Table (`outputs/phase3_feature_table.csv`)

| Column Name | Category | Native Scale | Null Count | Units / Formula | Scientific Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `latitude` | Spatial | Coordinate | 0 | Decimal Degrees | Latitude in WGS84. |
| `longitude` | Spatial | Coordinate | 0 | Decimal Degrees | Longitude in WGS84. |
| `event_date` | Temporal | Timestamp | 0 | ISO Date | Timestamp of event occurrence or paired date for negatives. |
| `Location Accuracy` | Metadata | Attribute | 0 | Categorical | Positional accuracy tier ($\le 5\text{ km}$). |
| `Landslide Trigger` | Metadata | Attribute | 0 | Categorical | Specific rain trigger classification. |
| `Imported Event Source Catalog` | Metadata | Attribute | 0 | Categorical | Ingestion catalog source. |
| `label` | Target | Binary | 0 | {0, 1} | Binary class label (127 positive, 127 negative). |
| `block_id` | Spatial | $0.5^\circ$ | 0 | String | Spatial bin code. |
| `elevation` | Topographic | 30 m | 10 | Meters (m) | Orthometric surface height from SRTMGL1_003. |
| `slope` | Topographic | 30 m | 10 | Degrees ($^\circ$) | Maximum rate of elevation change across 30m grid cell. |
| `aspect` | Topographic | 30 m | 10 | Degrees ($^\circ$) | Compass direction of downward slope ($0^\circ - 360^\circ$). |
| `curvature` | Topographic | 30 m | 10 | $\text{m}^{-1}$ (Laplacian) | Second spatial derivative of topography via $3\times 3$ Laplacian kernel. |
| `tri` | Topographic | 30 m | 10 | Meters (m) | Terrain Ruggedness Index: elevation standard deviation in $3\times 3$ window. |
| `twi` | Topographic | 30 m / 90 m | 10 | Dimensionless | Topographic Wetness Index: $\ln(\text{upa} / \tan(\text{slope}))$, using MERIT Hydro upstream area. |
| `BSI` | Optical | 10 m / 20 m | 80 | Index $[-1, 1]$ | Bare Soil Index: $\frac{(B11 + B4) - (B8 + B2)}{(B11 + B4) + (B8 + B2)}$. Higher values indicate exposed soil. |
| `NDVI` | Optical | 10 m | 80 | Index $[-1, 1]$ | Normalized Difference Vegetation Index: $\frac{B8 - B4}{B8 + B4}$. |
| `NDWI` | Optical | 10 m | 80 | Index $[-1, 1]$ | Normalized Difference Water/Moisture Index: $\frac{B3 - B8}{B3 + B8}$. |
| `spectral_window_days` | Optical | Metadata | 80 | Days | Search radius used for clear pixel cloud masking ($30, 60, 90, 120\text{ days}$). |
| `VH_post` | SAR | 10 m | 5 | dB | Post-event (0 to +12d) Sentinel-1 cross-polarization backscatter. |
| `VH_pre` | SAR | 10 m | 5 | dB | Pre-event (-12d to 0d) Sentinel-1 cross-polarization backscatter. |
| `VV_post` | SAR | 10 m | 5 | dB | Post-event (0 to +12d) Sentinel-1 co-polarization backscatter. |
| `VV_pre` | SAR | 10 m | 5 | dB | Pre-event (-12d to 0d) Sentinel-1 co-polarization backscatter. |
| `dVH` | SAR | 10 m | 7 | dB | Cross-polarization change: $\text{VH}_{\text{post}} - \text{VH}_{\text{pre}}$. |
| `dVV` | SAR | 10 m | 7 | dB | Co-polarization change: $\text{VV}_{\text{post}} - \text{VV}_{\text{pre}}$. |
| `API` | Precipitation | ~5.5 km | 0 | Millimeters (mm) | Antecedent Precipitation Index: $\sum_{t=1}^{30} P_t k^t$ with decay constant $k=0.85$. |
| `rain_1d` | Precipitation | ~5.5 km | 0 | Millimeters (mm) | CHIRPS total rainfall on the event date (day $t_0$). |
| `rain_7d` | Precipitation | ~5.5 km | 0 | Millimeters (mm) | 7-day cumulative rainfall prior to and including event date. |
| `rain_30d` | Precipitation | ~5.5 km | 0 | Millimeters (mm) | 30-day cumulative rainfall prior to and including event date. |

---

### 2.4. Spatial Model Metrics (`outputs/phase4_metrics_comparison.csv`)

| Column Name | Type | Value Range | Description |
| :--- | :--- | :--- | :--- |
| `feature_set` | String | Categorical | Description of feature configuration: `FULL` (18 features) or `REDUCED` (16 features). |
| `n_features` | Integer | 16 or 18 | Total number of predictive features in the design matrix. |
| `rf_auroc` | Float | $[0.95, 0.99]$ | Spatial 5-fold cross-validation Area Under ROC Curve for Random Forest. |
| `xgb_auroc` | Float | $[0.97, 0.98]$ | Spatial 5-fold cross-validation Area Under ROC Curve for XGBoost. |
| `rf_prauc` | Float | $[0.96, 0.99]$ | Spatial cross-validation Area Under Precision-Recall Curve for Random Forest. |
| `xgb_prauc` | Float | $[0.97, 0.99]$ | Spatial cross-validation Area Under Precision-Recall Curve for XGBoost. |
| `rf_f1` | Float | $[0.89, 0.96]$ | Macro F1 score at decision threshold $p = 0.5$ for Random Forest. |
| `xgb_f1` | Float | $[0.93, 0.96]$ | Macro F1 score at decision threshold $p = 0.5$ for XGBoost. |

---

### 2.5. Out-of-Fold Predictions (`outputs/phase4_oof_predictions.csv`)

| Column Name | Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `latitude` | Float | 0 | Latitude in decimal degrees (WGS84). |
| `longitude` | Float | 0 | Longitude in decimal degrees (WGS84). |
| `label` | Integer | 0 | True ground truth label ($0$ or $1$). |
| `spatial_fold` | Integer | 0 | Assigned spatial outer cross-validation fold index ($0$ to $4$). |
| `oof_rf_full` | Float | 0 | Random Forest predicted probability under Full feature set. |
| `oof_xgb_full` | Float | 0 | XGBoost predicted probability under Full feature set. |
| `oof_rf_reduced` | Float | 0 | Random Forest predicted probability under Reduced feature set. |
| `oof_xgb_reduced` | Float | 0 | XGBoost predicted probability under Reduced feature set. |

---

### 2.6. Regional Susceptibility Grid (`outputs/phase5_susceptibility_grid.csv`)

| Column Name | Type | Null Count | Summary Statistics | Description |
| :--- | :--- | :--- | :--- | :--- |
| `latitude` | Float | 0 | Min: 26.5, Max: 34.5, Mean: 30.5 | Macro grid cell center latitude ($0.5^\circ$ spacing). |
| `longitude` | Float | 0 | Min: 74.5, Max: 96.5, Mean: 85.5 | Macro grid cell center longitude ($0.5^\circ$ spacing). |
| `susceptibility` | Float | 0 | Min: 0.0004, Max: 0.9988, Mean: 0.3692 | Ensemble susceptibility: $\frac{\text{RF} + \text{XGB}}{2}$. |
| `susceptibility_rf` | Float | 0 | Min: 0.0000, Max: 0.9980, Mean: 0.3860 | Random Forest predicted susceptibility probability. |
| `susceptibility_xgb` | Float | 0 | Min: 0.0003, Max: 0.9997, Mean: 0.3525 | XGBoost predicted susceptibility probability. |
| `rf_tree_uncertainty` | Float | 0 | Min: 0.0000, Max: 0.5000, Mean: 0.3747 | Standard deviation of prediction probabilities across the 500 RF decision trees. |
| `model_disagreement` | Float | 0 | Min: 0.0000, Max: 0.5628, Mean: 0.1474 | Absolute inter-model disparity: $|\text{susceptibility}_{\text{rf}} - \text{susceptibility}_{\text{xgb}}|$. |

---

## 3. Data Integrity & Missingness Policy

1. **Explicit Null Representation:** Missing features are preserved as IEEE 754 `NaN` values and must never be filled with artificial placeholder values (e.g. `0` or `-999`) without transparent visualization.
2. **Missingness Attribution:**
   - Optical gaps (`BSI`, `NDVI`, `NDWI`, 31.5% missing) result from dense, persistent cloud cover during the Himalayan monsoon season.
   - Topographic gaps (3.9% missing) result from SRTM DEM boundaries or invalid void data in extreme alpine terrain.
   - SAR gaps (2.0% - 2.8% missing) result from missing radar tracks or radar shadow/layover masking in steep valleys.
3. **Imputation Policy:**
   - Random Forest uses median imputation (`SimpleImputer(strategy='median')`) fit strictly within spatial training folds to prevent data leakage.
   - XGBoost uses native directional tree routing for missing values without imputation.

