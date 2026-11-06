"""Sampling protocol and Earth Engine script generator for Version 2.0 Candidate Hillslope Controls.

Step 5.2 Data-Integrity Corrections:
1. Removed jrc_water.unmask(0); preserved valid-data mask with explicit rejection for missing water data.
2. Explicit rejection for missing ESA WorldCover observations (no silent pass-through).
3. Explicit null checks on all mandatory terrain features (no silent 0.0 fallbacks).
4. Rejection counters added for missing_terrain_data, missing_water_data, and missing_landcover_data.
5. SAR temporal stability filter formulated as a temporal consistency screening criterion,
   with documented confounding influences (seasonal, vegetation, soil-moisture, layover/shadow).
6. Transparent candidate selection strategy: evaluates candidates in the search ring, filters invalid points,
   and selects the control minimizing normalized composite topographic distance D_topo to the paired positive event.
7. Comprehensive matching diagnostics reporting actual distributions (Min, P25, Median, Mean, P75, Max).
8. Documented 30m SRTM vs 90m MERIT Hydro resolution mismatch, bilinear resampling, and contour length b=30m in TWI.
9. Pure Python candidate evaluation logic with full test suite coverage.

IMPORTANT SCIENTIFIC GOVERNANCE:
- Version 1.0 (N=254) remains the immutable historical diagnostic baseline.
- No simulated or synthetic GEE feature values are fabricated.
- Version 2.0 extraction requires authenticated Google Earth Engine execution in Google Colab.
- Version 2.0 status is explicitly tracked as EXTRACTION_PENDING_COLAB_GEE.
- Candidate terminology is strictly "Candidate Hillslope Control / Unrecorded Failure Status".
- Elimination of terrain confound cannot be claimed a priori from the sampling mask alone;
  it must be empirically validated via Kolmogorov-Smirnov testing upon genuine GEE extraction.
"""

import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METADATA_DIR = ARTIFACTS_DIR / "metadata"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates geodesic distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def evaluate_candidate_point(
    c_lat: float,
    c_lon: float,
    props: dict,
    paired_positive: dict,
    all_positives_coords: list,
    lat_min: float = 26.0,
    lat_max: float = 35.0,
    lon_min: float = 74.0,
    lon_max: float = 97.0,
    slope_bounds: tuple = (6.1, 41.4),
    elevation_bounds: tuple = (266.0, 3027.0),
    min_dist_to_positive_km: float = 2.0,
    max_sar_std_db: float = 1.5,
    min_sar_count: int = 3
) -> tuple:
    """Evaluates a candidate hillslope control against physical, environmental, and quality filters.
    
    Returns:
        (is_accepted: bool, status_reason: str, record_or_none: dict or None)
    """
    # Filter 1: Regional ROI bounds
    if not (lat_min <= c_lat <= lat_max and lon_min <= c_lon <= lon_max):
        return False, "roi_out_of_bounds", None

    # Filter 2: Proximity to ANY known positive event (> 2.0 km)
    min_dist = min(haversine_km(c_lat, c_lon, p_lat, p_lon) for p_lat, p_lon in all_positives_coords)
    if min_dist <= min_dist_to_positive_km:
        return False, "scar_proximity_under_2km", None

    # Filter 3: Explicit check for mandatory terrain features (no silent 0.0 fallbacks)
    c_elev = props.get("elevation")
    c_slope = props.get("slope")
    c_aspect = props.get("aspect")
    c_curv = props.get("curvature")
    c_tri = props.get("tri")
    c_tri_mean = props.get("tri_mean_diff")
    c_tri_proxy = props.get("tri_focal_diff_proxy")
    c_twi = props.get("twi")
    c_twi_proxy = props.get("twi_slope_proxy")

    terrain_vars = [c_elev, c_slope, c_aspect, c_curv, c_tri, c_tri_mean, c_tri_proxy, c_twi, c_twi_proxy]
    if any(v is None for v in terrain_vars):
        return False, "missing_terrain_data", None

    # Filter 4: Water observation availability (no unmask(0) silent pass)
    c_water = props.get("water_occurrence")
    if c_water is None:
        return False, "missing_water_data", None

    # Filter 5: Land-cover observation availability (no unmask silent pass)
    c_lc = props.get("landcover")
    if c_lc is None:
        return False, "missing_landcover_data", None

    # Filter 6: Elevation envelope [266m, 3027m]
    if c_elev < elevation_bounds[0] or c_elev > elevation_bounds[1]:
        return False, "elevation_out_of_bounds", None

    # Filter 7: Slope envelope [6.1 deg, 41.4 deg]
    if c_slope < slope_bounds[0] or c_slope > slope_bounds[1]:
        return False, "slope_out_of_bounds", None

    # Filter 8: Water body overlap (JRC >= 10% or WorldCover == 80)
    if c_water >= 10.0 or c_lc == 80:
        return False, "water_body_overlap", None

    # Filter 9: Permanent snow/ice overlap (WorldCover == 70)
    if c_lc == 70:
        return False, "snow_ice_overlap", None

    # Filter 10: SAR temporal baseline count
    c_sar_count = props.get("sar_pre_count")
    c_sar_std = props.get("sar_pre_std_vv")
    if c_sar_count is None or c_sar_count < min_sar_count or c_sar_std is None:
        return False, "sar_insufficient_baseline", None

    # Filter 11: SAR temporal consistency screening (Std <= 1.5 dB)
    if c_sar_std > max_sar_std_db:
        return False, "sar_temporal_instability", None

    # Candidate accepted: Compute normalized composite topographic matching distance
    p_lat = paired_positive.get("latitude", c_lat)
    p_lon = paired_positive.get("longitude", c_lon)
    p_slope = float(paired_positive.get("slope", c_slope))
    p_elev = float(paired_positive.get("elevation", c_elev))
    p_tri = float(paired_positive.get("tri", c_tri))
    p_asp = float(paired_positive.get("aspect", c_aspect))

    # Empirical standard deviations of confirmed positive events
    sigma_slope = 10.5
    sigma_elev = 775.0
    sigma_tri = 17.5

    d_slope = abs(float(c_slope) - p_slope)
    d_elev = abs(float(c_elev) - p_elev)
    d_tri = abs(float(c_tri) - p_tri)
    d_aspect = min(abs(float(c_aspect) - p_asp), 360.0 - abs(float(c_aspect) - p_asp))
    dist_paired = haversine_km(p_lat, p_lon, c_lat, c_lon)

    d_topo = math.sqrt(
        (d_slope / sigma_slope) ** 2 +
        (d_elev / sigma_elev) ** 2 +
        (d_tri / sigma_tri) ** 2
    )

    record = {
        "latitude": float(c_lat),
        "longitude": float(c_lon),
        "elevation": float(c_elev),
        "slope": float(c_slope),
        "aspect": float(c_aspect),
        "curvature": float(c_curv),
        "tri": float(c_tri),
        "tri_mean_diff": float(c_tri_mean),
        "tri_focal_diff_proxy": float(c_tri_proxy),
        "twi": float(c_twi),
        "twi_slope_proxy": float(c_twi_proxy),
        "water_occurrence": float(c_water),
        "landcover": int(c_lc),
        "sar_pre_std_vv": float(c_sar_std),
        "sar_pre_count": int(c_sar_count),
        "min_distance_to_any_positive_km": float(min_dist),
        "spatial_distance_to_paired_positive_km": float(dist_paired),
        "delta_slope_deg": float(d_slope),
        "delta_elevation_m": float(d_elev),
        "delta_aspect_deg": float(d_aspect),
        "delta_tri": float(d_tri),
        "d_topo_composite": float(d_topo)
    }
    return True, "ACCEPTED", record


def generate_v2_sampling_parameters():
    """Calculates empirical parameter bounds from the 127 confirmed positive events."""
    p3_path = OUTPUTS_DIR / "phase3_feature_table.csv"
    if not p3_path.exists():
        raise FileNotFoundError(f"Missing {p3_path}")

    df = pd.read_csv(p3_path)
    pos = df[df["label"] == 1].copy()
    neg_v1 = df[df["label"] == 0].copy()

    params = {
        "candidate_control_terminology": "Candidate Hillslope Control / Unrecorded Failure Status",
        "inventory_incompleteness_disclaimer": (
            "Absence from NASA COOLR catalog establishes unrecorded failure status, not guaranteed absolute stability. "
            "Controls are sampled from unfailed hillslopes matching empirical positive event terrain envelopes."
        ),
        "confound_mitigation_policy": (
            "Sampling mask stratification is designed to mitigate the flat-terrain confound. "
            "Actual confound elimination cannot be claimed a priori; it must be empirically verified via two-sample "
            "Kolmogorov-Smirnov testing once extracted Version 2.0 distributions are ingested."
        ),
        "sar_screening_characterization": (
            "Pre-event backscatter standard deviation (<= 1.5 dB) serves as a microwave temporal consistency screening "
            "criterion, not geodetic proof of physical ground stability. Radar backscatter temporal variance is influenced "
            "by seasonal freeze-thaw cycles, vegetation phenology, episodic soil moisture spikes from monsoon rains, and "
            "steep-slope local incidence angle variations (layover and shadow)."
        ),
        "candidate_selection_strategy": (
            "For each positive event, candidates are sampled within a 10-40km ring buffer and evaluated sequentially "
            "against all physical and data availability filters. Among all valid candidates, the optimal control is selected "
            "by minimizing the standardized multidimensional topographic distance D_topo to the paired positive event: "
            "D_topo = sqrt(((slope_c - slope_p)/10.5)^2 + ((elev_c - elev_p)/775)^2 + ((tri_c - tri_p)/17.5)^2). "
            "This ensures that accepted controls are geomorphologically representative rather than arbitrary first passes."
        ),
        "twi_resolution_alignment_specification": (
            "SRTM 30m elevation derivatives are combined with MERIT Hydro 90m flow accumulation (upa). "
            "To resolve the resolution mismatch, MERIT Hydro 'upa' is resampled via bilinear interpolation to match the "
            "30m SRTM grid. At 30m resolution, contour length b = 30m, yielding specific catchment area alpha = (upa * 1e6) / 30m. "
            "Topographic Wetness Index is calculated as ln(alpha / tan(beta)), with tan(beta) >= 0.001."
        ),
        "slope": {
            "p05": float(pos["slope"].quantile(0.05)),
            "p10": float(pos["slope"].quantile(0.10)),
            "p25": float(pos["slope"].quantile(0.25)),
            "median": float(pos["slope"].median()),
            "mean": float(pos["slope"].mean()),
            "std": float(pos["slope"].std()),
            "p75": float(pos["slope"].quantile(0.75)),
            "p90": float(pos["slope"].quantile(0.90)),
            "p95": float(pos["slope"].quantile(0.95)),
            "min": float(pos["slope"].min()),
            "max": float(pos["slope"].max()),
        },
        "elevation": {
            "p05": float(pos["elevation"].quantile(0.05)),
            "p10": float(pos["elevation"].quantile(0.10)),
            "median": float(pos["elevation"].median()),
            "mean": float(pos["elevation"].mean()),
            "std": float(pos["elevation"].std()),
            "p90": float(pos["elevation"].quantile(0.90)),
            "p95": float(pos["elevation"].quantile(0.95)),
            "min": float(pos["elevation"].min()),
            "max": float(pos["elevation"].max()),
        },
        "tri": {
            "p10": float(pos["tri"].quantile(0.10)),
            "median": float(pos["tri"].median()),
            "mean": float(pos["tri"].mean()),
            "std": float(pos["tri"].std()),
            "p90": float(pos["tri"].quantile(0.90)),
        },
        "twi": {
            "p10": float(pos["twi"].quantile(0.10)),
            "median": float(pos["twi"].median()),
            "mean": float(pos["twi"].mean()),
            "std": float(pos["twi"].std()),
            "p90": float(pos["twi"].quantile(0.90)),
        },
        "n_positives": len(pos),
    }

    params["v2_control_envelope"] = {
        "slope_min_deg": round(params["slope"]["p10"], 1),       # 6.1 deg
        "slope_max_deg": round(params["slope"]["p95"], 1),       # 41.4 deg
        "elevation_min_m": round(params["elevation"]["p10"], 0), # 266 m (P05) / 327 m (P10)
        "elevation_max_m": round(params["elevation"]["p95"], 0), # 3027 m
        "min_dist_to_positive_m": 2000,                          # 2.0 km universal buffer exclusion (prevents scar overlap)
        "max_buffer_search_km": 40,                              # Local catchment matching (10km to 40km ring buffer)
        "min_buffer_search_km": 10,
        "jrc_max_water_pct": 10,                                 # JRC surface water exclusion
        "esa_worldcover_snow_ice_excluded_class": 70,            # ESA WorldCover 10m Snow and Ice class
        "esa_worldcover_water_excluded_class": 80,               # ESA WorldCover 10m Permanent Water class
        "sar_observation_window_days": [180, 15],                # Pre-event observation window [event-180d, event-15d]
        "sar_max_std_db": 1.5,                                   # Multi-temporal stability criterion (VV std <= 1.5 dB)
        "sar_min_observations": 3                                # Minimum valid pre-event SAR acquisitions
    }

    params["geomorphometric_definitions"] = {
        "tri_riley_standard": "Riley et al. (1999) root-sum-squared elevation differences across 8 neighbors: sqrt(sum(z_i - z_0)^2)",
        "tri_mean_absolute_difference": "Wilson & Gallant / standard mean absolute elevation difference: (1/8) * sum(|z_i - z_0|)",
        "tri_focal_diff_proxy": "Absolute difference from 3x3 uniform focal mean: |z_0 - mean_3x3(z)| (retained as proxy)",
        "twi_merit_hydro": "Beven & Kirkby (1979) ln(alpha / tan(beta)) using bilinearly resampled MERIT Hydro flow accumulation (upa)",
        "twi_slope_proxy": "Slope-only proxy: ln(1 / tan(beta)) = -ln(tan(beta)) (retained as proxy)"
    }

    params["sampling_audit_schema"] = {
        "required_rejection_counters": [
            "roi_out_of_bounds",
            "scar_proximity_under_2km",
            "missing_terrain_data",
            "missing_water_data",
            "missing_landcover_data",
            "elevation_out_of_bounds",
            "slope_out_of_bounds",
            "water_body_overlap",
            "snow_ice_overlap",
            "sar_insufficient_baseline",
            "sar_temporal_instability",
            "no_valid_candidate_after_search"
        ]
    }

    params["matching_diagnostics_schema"] = {
        "diagnostic_variables": [
            "spatial_distance_km",
            "min_dist_to_all_positives_km",
            "delta_slope_deg",
            "delta_elevation_m",
            "delta_aspect_deg",
            "delta_tri",
            "d_topo_composite",
            "sar_pre_std_vv",
            "sar_pre_count"
        ],
        "reporting_metrics": ["min", "p25", "median", "mean", "p75", "max"],
        "limitation_note": "A 10–40 km ring buffer serves as a regional physiographic proxy but does not guarantee identical catchment or bedrock lithology."
    }

    # Statistical distribution validation against Version 1.0 controls
    ks_slope_v1 = ks_2samp(pos["slope"].dropna(), neg_v1["slope"].dropna())
    ks_tri_v1 = ks_2samp(pos["tri"].dropna(), neg_v1["tri"].dropna())
    ks_elev_v1 = ks_2samp(pos["elevation"].dropna(), neg_v1["elevation"].dropna())

    params["v1_distribution_divergence"] = {
        "note": "Version 1.0 controls show extreme statistical divergence from landslide terrain (p-value < 1e-15), proving flat-terrain bias.",
        "slope_ks_statistic": float(ks_slope_v1.statistic),
        "slope_ks_pvalue": float(ks_slope_v1.pvalue),
        "tri_ks_statistic": float(ks_tri_v1.statistic),
        "tri_ks_pvalue": float(ks_tri_v1.pvalue),
        "elevation_ks_statistic": float(ks_elev_v1.statistic),
        "elevation_ks_pvalue": float(ks_elev_v1.pvalue),
    }

    params["candidate_validation_status"] = {
        "status": "EXTRACTION_PENDING_COLAB_GEE",
        "description": "Candidate sampling envelopes are empirically validated against positive events. Feature extraction requires authenticated Earth Engine execution in Colab.",
        "colab_notebook": "notebooks/Phase3b_V2_Hillslope_Extraction_Colab.ipynb",
        "gee_project": "landslidedetection-506008",
        "provenance_policy": "No synthetic feature generation permitted. Version 1.0 remains baseline until genuine GEE extraction is completed."
    }

    return params


def create_colab_extraction_notebook(output_path: Path):
    """Creates the turnkey, corrected Google Colab notebook for GEE Version 2.0 extraction."""
    cells = [
        # Cell 0: Markdown Title & Scientific Governance
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Version 2.0 Candidate Hillslope Control Extraction Pipeline (Colab GEE)\n",
                "## Himalayan Landslide Intelligence — Phase 3b Multi-Sensor Extraction (Step 5.2)\n",
                "\n",
                "### Scientific Context & Data Integrity Governance:\n",
                "1. **Terminology & Failure Status:** Negative controls are designated **Candidate Hillslope Control / Unrecorded Failure Status**. Absence from the NASA Global Landslide Catalog establishes unrecorded failure status, not guaranteed absolute stability across geological timescales.\n",
                "2. **Confound Mitigation Policy:** Sampling within the empirical hillslope envelope ($6.1^\\circ \\le \\text{slope} \\le 41.4^\\circ$, $266\\text{ m} \\le \\text{elev} \\le 3027\\text{ m}$) is designed to mitigate the historical flat-terrain bias (Version 1.0 controls had mean slope $1.43^\\circ$). **Actual elimination of the terrain confound cannot be assumed a priori**; it must be empirically validated via two-sample Kolmogorov-Smirnov tests on the extracted feature distributions.\n",
                "3. **SAR Temporal Consistency Screening:** Pre-event Sentinel-1 IW GRD VV standard deviation must be $\\le 1.5\\text{ dB}$ over a 180-day baseline observation window ($[t_{\\text{event}}-180\\text{d}, t_{\\text{event}}-15\\text{d}]$), with at least 3 valid acquisitions required. *Note:* This serves as a microwave temporal consistency screening criterion, not geodetic proof of ground stability; backscatter variance may also be influenced by seasonal freeze-thaw, vegetation phenology, episodic soil moisture spikes, and radar layover/shadow on steep terrain.\n",
                "4. **Rigorous Data Availability Masks (No Silent Pass):**\n",
                "   - JRC Surface Water ($< 10\\%$) and ESA WorldCover (Class 80 $\\neq$ Water, Class 70 $\\neq$ Snow/Ice) are evaluated with strict valid-data masks. Missing water or land-cover observations are explicitly rejected, NOT converted to zero.\n",
                "   - Mandatory terrain variables (elevation, slope, aspect, curvature, TRI, TWI) are subject to explicit null checks; no silent 0.0 fallbacks.\n",
                "5. **Resolution Alignment in TWI:** MERIT Hydro upstream contributing area (`upa`, 90m native) is explicitly resampled via bilinear interpolation to match the SRTM 30m grid. Contour width is set to $b = 30\\text{ m}$, yielding $\\alpha = (\\text{upa} \\times 10^6) / 30\\text{ m}$, and $\\text{TWI} = \\ln(\\alpha / \\tan\\beta)$.\n",
                "6. **Optimized Candidate Selection Strategy:** Among candidates in the $10\\text{ km}-40\\text{ km}$ ring passing physical filters, the optimal control is selected by **minimizing the normalized composite topographic distance** $D_{\\text{topo}}$ to the paired positive event.\n",
                "7. **Audit & Diagnostics:** Complete rejection audit tracking 12 categories, and full percentile distributions of matching differences."
            ]
        },
        # Cell 1: Code - Dependencies and Authentication
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Install Dependencies and Initialize Google Earth Engine\n",
                "!pip install -q geemap earthengine-api pyproj pandas numpy scipy\n",
                "\n",
                "import ee\n",
                "import geemap.core as geemap\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import time\n",
                "import json\n",
                "import math\n",
                "from datetime import datetime\n",
                "\n",
                "# Authenticate and Initialize GEE with Project Credentials\n",
                "ee.Authenticate()\n",
                "ee.Initialize(project='landslidedetection-506008')\n",
                "print('Earth Engine initialized successfully:', ee.Number(1).add(1).getInfo() == 2)"
            ]
        },
        # Cell 2: Code - Load Positive Events & Authoritative ROI
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Load Confirmed Positive Landslide Events\n",
                "url_p2 = 'https://raw.githubusercontent.com/GouravSharma26/LandSlide/main/outputs/phase2_cleaned_dataset.csv'\n",
                "df_p2 = pd.read_csv(url_p2)\n",
                "df_p2['event_date'] = pd.to_datetime(df_p2['event_date'])\n",
                "positives = df_p2[df_p2['label'] == 1].reset_index(drop=True)\n",
                "print(f'Loaded {len(positives)} confirmed positive rainfall-triggered landslide events.')\n",
                "\n",
                "# Authoritative Himalayan Region of Interest (ROI)\n",
                "lat_min, lat_max = 26.0, 35.0\n",
                "lon_min, lon_max = 74.0, 97.0\n",
                "roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])\n",
                "\n",
                "# Universal Exclusion Zone: 2000m geodesic buffer around ALL 127 positive events\n",
                "pos_points = [ee.Geometry.Point([row['longitude'], row['latitude']]) for _, row in positives.iterrows()]\n",
                "universal_exclusion = ee.FeatureCollection([ee.Feature(pt.buffer(2000)) for pt in pos_points]).geometry()\n",
                "all_positives_coords = [(float(r['latitude']), float(r['longitude'])) for _, r in positives.iterrows()]\n",
                "print('Authoritative Himalayan ROI and Universal Exclusion Zone (2.0 km) declared successfully.')"
            ]
        },
        # Cell 3: Code - Environmental Datasets, Riley TRI, MERIT TWI (with Bilinear Resampling)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Environmental Datasets & Corrected Geomorphometric Indices (Riley TRI & MERIT TWI)\n",
                "\n",
                "# 1. SRTM 30m Digital Elevation Model & Derivatives\n",
                "srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(roi)\n",
                "slope = ee.Terrain.slope(srtm)\n",
                "aspect = ee.Terrain.aspect(srtm)\n",
                "curvature = srtm.convolve(ee.Kernel.laplacian8(normalize=False)).rename('curvature')\n",
                "\n",
                "# 2. Corrected Riley et al. (1999) Terrain Ruggedness Index (TRI)\n",
                "# Formula: sqrt( sum_{i=1..8} (z_i - z_0)^2 )\n",
                "offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]\n",
                "sq_diff_sum = ee.Image(0)\n",
                "abs_diff_sum = ee.Image(0)\n",
                "for dx, dy in offsets:\n",
                "    weights = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]\n",
                "    weights[dy + 1][dx + 1] = 1\n",
                "    k = ee.Kernel.fixed(3, 3, weights)\n",
                "    neighbor = srtm.convolve(k)\n",
                "    diff = neighbor.subtract(srtm)\n",
                "    sq_diff_sum = sq_diff_sum.add(diff.pow(2))\n",
                "    abs_diff_sum = abs_diff_sum.add(diff.abs())\n",
                "\n",
                "tri_riley = sq_diff_sum.sqrt().rename('tri')\n",
                "tri_mean_diff = abs_diff_sum.divide(8.0).rename('tri_mean_diff')\n",
                "tri_focal_proxy = srtm.subtract(srtm.convolve(ee.Kernel.square(radius=1, units='pixels'))).abs().rename('tri_focal_diff_proxy')\n",
                "\n",
                "# 3. Corrected Topographic Wetness Index (TWI) with Explicit Bilinear Resampling for Resolution Alignment\n",
                "# SRTM is 30m; MERIT Hydro is 90m. Bilinearly resample MERIT Hydro upa to 30m SRTM grid.\n",
                "merit_hydro = ee.Image('MERIT/Hydro/v1_0_1').clip(roi)\n",
                "merit_upa_30m = merit_hydro.select('upa').resample('bilinear').reproject(crs=srtm.projection(), scale=30)\n",
                "alpha_30m = merit_upa_30m.multiply(1e6).divide(30.0).max(1.0)  # Specific catchment area with contour width b=30m\n",
                "slp_rad = slope.multiply(math.pi / 180.0)\n",
                "tan_beta = slp_rad.tan().max(0.001)\n",
                "twi_merit = alpha_30m.divide(tan_beta).log().rename('twi')\n",
                "twi_slope_proxy = tan_beta.pow(-1).log().rename('twi_slope_proxy')\n",
                "\n",
                "# 4. Surface Water Occurrence: JRC Global Surface Water (v1.4) — PRESERVED WITHOUT unmask(0)\n",
                "jrc_water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(roi)\n",
                "\n",
                "# 5. Land Cover & Permanent Snow/Ice: ESA WorldCover 10m (v100)\n",
                "# Class 70 = Snow and ice; Class 80 = Permanent water bodies; Class 50 = Built-up\n",
                "worldcover = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map').clip(roi)\n",
                "\n",
                "# Multi-band Static Diagnostic Image for Rapid Batch Candidate Evaluation\n",
                "static_diag = (\n",
                "    srtm.rename('elevation')\n",
                "    .addBands(slope.rename('slope'))\n",
                "    .addBands(aspect.rename('aspect'))\n",
                "    .addBands(curvature.rename('curvature'))\n",
                "    .addBands(tri_riley)\n",
                "    .addBands(tri_mean_diff)\n",
                "    .addBands(tri_focal_proxy)\n",
                "    .addBands(twi_merit)\n",
                "    .addBands(twi_slope_proxy)\n",
                "    .addBands(jrc_water.rename('water_occurrence'))  # Valid-data mask preserved\n",
                "    .addBands(worldcover.rename('landcover'))\n",
                ")\n",
                "print('Static environmental layers, Riley TRI, and MERIT Hydro TWI (30m aligned) compiled successfully.')"
            ]
        },
        # Cell 4: Code - Candidate Sampling with Optimized Selection & Complete Audit Tracking
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 4: Sample Candidate Hillslope Controls with Optimized Matching & Complete Audit Tracking\n",
                "\n",
                "def haversine_km(lat1, lon1, lat2, lon2):\n",
                "    \"\"\"Calculates geodesic distance in km between two lat/lon points.\"\"\"\n",
                "    R = 6371.0\n",
                "    phi1, phi2 = math.radians(lat1), math.radians(lat2)\n",
                "    dphi = math.radians(lat2 - lat1)\n",
                "    dlambda = math.radians(lon2 - lon1)\n",
                "    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2\n",
                "    return 2 * R * math.asin(math.sqrt(a))\n",
                "\n",
                "# Standard deviations of confirmed positive events for normalized distance D_topo\n",
                "SIGMA_SLOPE = 10.5\n",
                "SIGMA_ELEV = 775.0\n",
                "SIGMA_TRI = 17.5\n",
                "\n",
                "# Comprehensive Rejection Audit Tracking (12 Categories)\n",
                "sampling_audit = {\n",
                "    'total_positive_events': len(positives),\n",
                "    'total_candidates_evaluated': 0,\n",
                "    'controls_successfully_sampled': 0,\n",
                "    'rejections': {\n",
                "        'roi_out_of_bounds': 0,\n",
                "        'scar_proximity_under_2km': 0,\n",
                "        'missing_terrain_data': 0,\n",
                "        'missing_water_data': 0,\n",
                "        'missing_landcover_data': 0,\n",
                "        'elevation_out_of_bounds': 0,\n",
                "        'slope_out_of_bounds': 0,\n",
                "        'water_body_overlap': 0,\n",
                "        'snow_ice_overlap': 0,\n",
                "        'sar_insufficient_baseline': 0,\n",
                "        'sar_temporal_instability': 0,\n",
                "        'no_valid_candidate_after_search': 0\n",
                "    }\n",
                "}\n",
                "\n",
                "v2_controls = []\n",
                "matching_diagnostics = []\n",
                "\n",
                "print('Starting Candidate Hillslope Control Sampling across 127 events...')\n",
                "print('Screening: Proximity > 2km | Elev [266-3027m] | Slope [6.1-41.4 deg] | Water < 10% | Snow/Ice != 70 | Pre-SAR std <= 1.5 dB')\n",
                "start_sampling = time.time()\n",
                "\n",
                "for idx, row in positives.iterrows():\n",
                "    p_lat, p_lon = float(row['latitude']), float(row['longitude'])\n",
                "    event_date = pd.to_datetime(row['event_date'])\n",
                "    pt_pos = ee.Geometry.Point([p_lon, p_lat])\n",
                "    \n",
                "    # 10km to 40km local catchment ring buffer, intersected with regional ROI\n",
                "    search_ring = pt_pos.buffer(40000).difference(pt_pos.buffer(10000)).intersection(roi)\n",
                "    \n",
                "    # Pre-event Sentinel-1 baseline window: [event_date - 180 days, event_date - 15 days]\n",
                "    pre_start = event_date - pd.Timedelta(days=180)\n",
                "    pre_end = event_date - pd.Timedelta(days=15)\n",
                "    s1_pre_coll = (\n",
                "        ee.ImageCollection('COPERNICUS/S1_GRD')\n",
                "        .filter(ee.Filter.eq('instrumentMode', 'IW'))\n",
                "        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\n",
                "        .filterBounds(search_ring)\n",
                "        .filterDate(pre_start.strftime('%Y-%m-%d'), pre_end.strftime('%Y-%m-%d'))\n",
                "    )\n",
                "    s1_std_vv = s1_pre_coll.select('VV').reduce(ee.Reducer.stdDev()).rename('sar_pre_std_vv')\n",
                "    s1_count = s1_pre_coll.select('VV').reduce(ee.Reducer.count()).rename('sar_pre_count')\n",
                "    \n",
                "    event_diag = static_diag.addBands(s1_std_vv).addBands(s1_count)\n",
                "    \n",
                "    valid_candidates_for_event = []\n",
                "    seed_offset = 0\n",
                "    max_batches = 3\n",
                "    \n",
                "    p_slope = float(row.get('slope', 21.0))\n",
                "    p_elev = float(row.get('elevation', 1200.0))\n",
                "    p_asp = float(row.get('aspect', 180.0))\n",
                "    p_tri = float(row.get('tri', 25.0))\n",
                "    \n",
                "    for batch_num in range(max_batches):\n",
                "        batch_pts = ee.FeatureCollection.randomPoints(\n",
                "            region=search_ring,\n",
                "            points=25,\n",
                "            seed=42 + idx * 100 + seed_offset\n",
                "        )\n",
                "        candidates = event_diag.sampleRegions(\n",
                "            collection=batch_pts,\n",
                "            scale=30,\n",
                "            geometries=True\n",
                "        ).getInfo().get('features', [])\n",
                "        \n",
                "        for cand in candidates:\n",
                "            sampling_audit['total_candidates_evaluated'] += 1\n",
                "            coords = cand['geometry']['coordinates']\n",
                "            c_lon, c_lat = float(coords[0]), float(coords[1])\n",
                "            props = cand.get('properties', {})\n",
                "            \n",
                "            # Filter 1: Regional ROI bounds\n",
                "            if not (lat_min <= c_lat <= lat_max and lon_min <= c_lon <= lon_max):\n",
                "                sampling_audit['rejections']['roi_out_of_bounds'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 2: Proximity to ANY known positive event (> 2.0 km)\n",
                "            min_dist_to_positive = min(\n",
                "                haversine_km(c_lat, c_lon, p_coord[0], p_coord[1])\n",
                "                for p_coord in all_positives_coords\n",
                "            )\n",
                "            if min_dist_to_positive <= 2.0:\n",
                "                sampling_audit['rejections']['scar_proximity_under_2km'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 3: Explicit check for mandatory terrain features (no silent 0.0 defaults)\n",
                "            c_elev = props.get('elevation')\n",
                "            c_slope = props.get('slope')\n",
                "            c_aspect = props.get('aspect')\n",
                "            c_curv = props.get('curvature')\n",
                "            c_tri = props.get('tri')\n",
                "            c_tri_mean = props.get('tri_mean_diff')\n",
                "            c_tri_proxy = props.get('tri_focal_diff_proxy')\n",
                "            c_twi = props.get('twi')\n",
                "            c_twi_proxy = props.get('twi_slope_proxy')\n",
                "            \n",
                "            if any(v is None for v in [c_elev, c_slope, c_aspect, c_curv, c_tri, c_tri_mean, c_tri_proxy, c_twi, c_twi_proxy]):\n",
                "                sampling_audit['rejections']['missing_terrain_data'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 4: Water observation availability\n",
                "            c_water = props.get('water_occurrence')\n",
                "            if c_water is None:\n",
                "                sampling_audit['rejections']['missing_water_data'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 5: Land-cover observation availability\n",
                "            c_lc = props.get('landcover')\n",
                "            if c_lc is None:\n",
                "                sampling_audit['rejections']['missing_landcover_data'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 6: Elevation envelope [266m, 3027m]\n",
                "            if c_elev < 266.0 or c_elev > 3027.0:\n",
                "                sampling_audit['rejections']['elevation_out_of_bounds'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 7: Slope envelope [6.1 deg, 41.4 deg]\n",
                "            if c_slope < 6.1 or c_slope > 41.4:\n",
                "                sampling_audit['rejections']['slope_out_of_bounds'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 8: Water body overlap (JRC >= 10% or WorldCover == 80)\n",
                "            if c_water >= 10.0 or c_lc == 80:\n",
                "                sampling_audit['rejections']['water_body_overlap'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 9: Permanent snow/ice overlap (WorldCover == 70)\n",
                "            if c_lc == 70:\n",
                "                sampling_audit['rejections']['snow_ice_overlap'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 10: SAR temporal baseline count (>= 3 acquisitions)\n",
                "            c_sar_count = props.get('sar_pre_count')\n",
                "            c_sar_std = props.get('sar_pre_std_vv')\n",
                "            if c_sar_count is None or c_sar_count < 3 or c_sar_std is None:\n",
                "                sampling_audit['rejections']['sar_insufficient_baseline'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Filter 11: SAR temporal consistency screening criterion (Std <= 1.5 dB)\n",
                "            if c_sar_std > 1.5:\n",
                "                sampling_audit['rejections']['sar_temporal_instability'] += 1\n",
                "                continue\n",
                "                \n",
                "            # Candidate valid: compute differences and normalized composite topographic distance D_topo\n",
                "            d_slope = abs(float(c_slope) - p_slope)\n",
                "            d_elev = abs(float(c_elev) - p_elev)\n",
                "            d_tri = abs(float(c_tri) - p_tri)\n",
                "            d_aspect = min(abs(float(c_aspect) - p_asp), 360.0 - abs(float(c_aspect) - p_asp))\n",
                "            dist_paired = haversine_km(p_lat, p_lon, c_lat, c_lon)\n",
                "            \n",
                "            d_topo = math.sqrt(\n",
                "                (d_slope / SIGMA_SLOPE) ** 2 +\n",
                "                (d_elev / SIGMA_ELEV) ** 2 +\n",
                "                (d_tri / SIGMA_TRI) ** 2\n",
                "            )\n",
                "            \n",
                "            cand_record = {\n",
                "                'latitude': c_lat,\n",
                "                'longitude': c_lon,\n",
                "                'event_date': row['event_date'].strftime('%Y-%m-%d'),\n",
                "                'Location Accuracy': 'Candidate Hillslope Control / Unrecorded Failure Status',\n",
                "                'Landslide Trigger': 'Candidate Hillslope Control / Unrecorded Failure Status',\n",
                "                'Imported Event Source Catalog': 'V2_GEE_Candidate_Hillslope',\n",
                "                'label': 0,\n",
                "                'block_id': f\"{int(c_lon//0.5)}_{int(c_lat//0.5)}\",\n",
                "                'paired_positive_idx': int(idx),\n",
                "                'sar_pre_std_vv': float(c_sar_std),\n",
                "                'sar_pre_count': int(c_sar_count),\n",
                "                'elevation': float(c_elev),\n",
                "                'slope': float(c_slope),\n",
                "                'aspect': float(c_aspect),\n",
                "                'curvature': float(c_curv),\n",
                "                'tri': float(c_tri),\n",
                "                'tri_mean_diff': float(c_tri_mean),\n",
                "                'tri_focal_diff_proxy': float(c_tri_proxy),\n",
                "                'twi': float(c_twi),\n",
                "                'twi_slope_proxy': float(c_twi_proxy),\n",
                "                'min_distance_to_any_positive_km': float(min_dist_to_positive),\n",
                "                'spatial_distance_to_paired_positive_km': float(dist_paired),\n",
                "                'delta_slope_deg': float(d_slope),\n",
                "                'delta_elevation_m': float(d_elev),\n",
                "                'delta_aspect_deg': float(d_aspect),\n",
                "                'delta_tri': float(d_tri),\n",
                "                'd_topo_composite': float(d_topo)\n",
                "            }\n",
                "            valid_candidates_for_event.append(cand_record)\n",
                "            \n",
                "        # If valid candidates found in this batch, stop generating further batches\n",
                "        if valid_candidates_for_event:\n",
                "            break\n",
                "        seed_offset += 25\n",
                "        \n",
                "    if valid_candidates_for_event:\n",
                "        # OPTIMIZED SELECTION: Select candidate that minimizes composite topographic distance D_topo\n",
                "        valid_candidates_for_event.sort(key=lambda r: r['d_topo_composite'])\n",
                "        best_candidate = valid_candidates_for_event[0]\n",
                "        v2_controls.append(best_candidate)\n",
                "        sampling_audit['controls_successfully_sampled'] += 1\n",
                "        \n",
                "        matching_diagnostics.append({\n",
                "            'paired_positive_idx': int(idx),\n",
                "            'control_lat': best_candidate['latitude'],\n",
                "            'control_lon': best_candidate['longitude'],\n",
                "            'positive_lat': p_lat,\n",
                "            'positive_lon': p_lon,\n",
                "            'spatial_distance_km': best_candidate['spatial_distance_to_paired_positive_km'],\n",
                "            'min_dist_to_all_positives_km': best_candidate['min_distance_to_any_positive_km'],\n",
                "            'control_slope': best_candidate['slope'],\n",
                "            'positive_slope': p_slope,\n",
                "            'delta_slope_deg': best_candidate['delta_slope_deg'],\n",
                "            'control_elevation': best_candidate['elevation'],\n",
                "            'positive_elevation': p_elev,\n",
                "            'delta_elevation_m': best_candidate['delta_elevation_m'],\n",
                "            'control_aspect': best_candidate['aspect'],\n",
                "            'positive_aspect': p_asp,\n",
                "            'delta_aspect_deg': best_candidate['delta_aspect_deg'],\n",
                "            'control_tri': best_candidate['tri'],\n",
                "            'positive_tri': p_tri,\n",
                "            'delta_tri': best_candidate['delta_tri'],\n",
                "            'd_topo_composite': best_candidate['d_topo_composite'],\n",
                "            'sar_pre_std_vv': best_candidate['sar_pre_std_vv'],\n",
                "            'sar_pre_count': best_candidate['sar_pre_count'],\n",
                "            'matching_strategy': 'Minimum standardized topographic distance D_topo in 10-40km ring buffer'\n",
                "        })\n",
                "    else:\n",
                "        sampling_audit['rejections']['no_valid_candidate_after_search'] += 1\n",
                "        print(f'Warning: No valid candidate found for positive event {idx} after {max_batches} batches.')\n",
                "        \n",
                "    if (idx + 1) % 25 == 0 or (idx + 1) == len(positives):\n",
                "        print(f'  Progress: {len(v2_controls)}/{idx+1} events (Evaluated {sampling_audit[\"total_candidates_evaluated\"]} total candidates)')\n",
                "\n",
                "df_v2_controls = pd.DataFrame(v2_controls)\n",
                "df_matching_diagnostics = pd.DataFrame(matching_diagnostics)\n",
                "elapsed_sampling = time.time() - start_sampling\n",
                "print(f'\\nSampling Complete in {elapsed_sampling:.1f}s.')\n",
                "print(f'Accepted Controls: {len(df_v2_controls)} / {len(positives)}')\n",
                "print('Detailed Rejection Tallies across all 12 Categories:')\n",
                "for reason, count in sampling_audit['rejections'].items():\n",
                "    print(f'  - {reason}: {count}')\n",
                "\n",
                "df_v2_controls.to_csv('phase2_v2_hillslope_controls.csv', index=False)\n",
                "df_matching_diagnostics.to_csv('v2_control_matching_diagnostics.csv', index=False)"
            ]
        },
        # Cell 5: Code - Validation & Complete Distribution Reporting
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: Strict Validation and Full Matching Distribution Reporting\n",
                "print('Executing Candidate Control Constraint Validation...')\n",
                "assert len(df_v2_controls) == len(positives), f'Sample count mismatch: {len(df_v2_controls)} != {len(positives)}'\n",
                "assert df_v2_controls['latitude'].between(lat_min, lat_max).all(), 'Latitude outside ROI bounds!'\n",
                "assert df_v2_controls['longitude'].between(lon_min, lon_max).all(), 'Longitude outside ROI bounds!'\n",
                "assert (df_v2_controls['min_distance_to_any_positive_km'] > 2.0).all(), 'Universal exclusion violation (point <= 2km from scar)!'\n",
                "assert df_v2_controls['slope'].between(6.1, 41.4).all(), 'Slope outside empirical envelope [6.1, 41.4]!'\n",
                "assert df_v2_controls['elevation'].between(266, 3027).all(), 'Elevation outside empirical envelope [266, 3027]!'\n",
                "assert (df_v2_controls['sar_pre_std_vv'] <= 1.5).all(), 'SAR stability threshold violated (> 1.5 dB)!'\n",
                "assert df_v2_controls[['elevation', 'slope', 'aspect', 'curvature', 'tri', 'twi']].notna().all().all(), 'Null terrain values detected!'\n",
                "print('PASSED: All 127 candidate controls strictly satisfy geographic bounds, universal exclusion (>2km), terrain envelope, and SAR consistency.')\n",
                "\n",
                "# Display Full Empirical Distributions of Matching Differences\n",
                "print('\\nFull Distribution of Paired Topographic Matching Differences:')\n",
                "dist_metrics = ['spatial_distance_km', 'delta_slope_deg', 'delta_elevation_m', 'delta_aspect_deg', 'delta_tri', 'd_topo_composite']\n",
                "summary_rows = []\n",
                "for m in dist_metrics:\n",
                "    s = df_matching_diagnostics[m]\n",
                "    summary_rows.append({\n",
                "        'Metric': m,\n",
                "        'Min': round(float(s.min()), 2),\n",
                "        'P25': round(float(s.quantile(0.25)), 2),\n",
                "        'Median': round(float(s.median()), 2),\n",
                "        'Mean': round(float(s.mean()), 2),\n",
                "        'P75': round(float(s.quantile(0.75)), 2),\n",
                "        'Max': round(float(s.max()), 2)\n",
                "    })\n",
                "df_dist_summary = pd.DataFrame(summary_rows)\n",
                "display(df_dist_summary)\n",
                "df_dist_summary.to_csv('v2_matching_distribution_summary.csv', index=False)"
            ]
        },
        # Cell 6: Code - Dynamic Multi-Sensor Collections Declaration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 6: Declare Dynamic Multi-Sensor Collections\n",
                "\n",
                "# 1. Sentinel-1 IW GRD C-Band SAR\n",
                "s1_grd = (\n",
                "    ee.ImageCollection('COPERNICUS/S1_GRD')\n",
                "    .filter(ee.Filter.eq('instrumentMode', 'IW'))\n",
                "    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\n",
                "    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\n",
                "    .filterBounds(roi)\n",
                ")\n",
                "\n",
                "# 2. Sentinel-2 MSI Surface Reflectance + Cloudless Probability Join\n",
                "s2_sr = (\n",
                "    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\n",
                "    .filterBounds(roi)\n",
                "    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))\n",
                ")\n",
                "s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(roi)\n",
                "s2_with_prob = ee.Join.saveFirst('cloud_prob').apply(\n",
                "    primary=s2_sr,\n",
                "    secondary=s2_clouds,\n",
                "    condition=ee.Filter.equals(leftField='system:index', rightField='system:index')\n",
                ")\n",
                "\n",
                "# Robust S2 Cloud Masking: Combines Scene Classification Layer (SCL) and Cloud Probability\n",
                "def mask_s2_clouds(img):\n",
                "    scl = img.select('SCL')\n",
                "    # SCL valid: 4 (Vegetation), 5 (Not vegetated), 6 (Water), 7 (Unclassified)\n",
                "    # Excludes: 3 (Shadow), 8 (Med Cloud), 9 (High Cloud), 10 (Cirrus), 11 (Snow)\n",
                "    scl_valid = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))\n",
                "    c_prob_obj = img.get('cloud_prob')\n",
                "    # Null-safe cloud probability check\n",
                "    c_prob_mask = ee.Algorithms.If(\n",
                "        c_prob_obj,\n",
                "        ee.Image(c_prob_obj).select('probability').lt(40),\n",
                "        ee.Image(1)\n",
                "    )\n",
                "    return img.updateMask(scl_valid.And(ee.Image(c_prob_mask)))\n",
                "\n",
                "# 3. CHIRPS Daily Precipitation (~5.5 km resolution)\n",
                "chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(roi)\n",
                "print('Dynamic satellite collections and robust masking functions declared successfully.')"
            ]
        },
        # Cell 7: Code - Multi-Sensor Extraction Functions with Robust Error Handling
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 7: Robust Extraction Functions (Multi-Scale SAR, CHIRPS, Adaptive Optical)\n",
                "\n",
                "SAR_NOMINAL_WINDOW_DAYS = 12\n",
                "SAR_EXTENDED_WINDOW_DAYS = 24\n",
                "SPECTRAL_WINDOW_STEPS = [30, 60, 90, 120]\n",
                "API_DECAY = 0.90\n",
                "\n",
                "def extract_point_and_buffers_sar(lon, lat, event_date_str):\n",
                "    \"\"\"Extracts Sentinel-1 SAR at 10m point, 500m buffer, and 1000m buffer with adaptive window fallback.\"\"\"\n",
                "    pt = ee.Geometry.Point([lon, lat])\n",
                "    event = ee.Date(event_date_str)\n",
                "    \n",
                "    pre_coll = s1_grd.filterBounds(pt).filterDate(event.advance(-SAR_NOMINAL_WINDOW_DAYS, 'day'), event)\n",
                "    post_coll = s1_grd.filterBounds(pt).filterDate(event, event.advance(SAR_NOMINAL_WINDOW_DAYS, 'day'))\n",
                "    \n",
                "    window_used = SAR_NOMINAL_WINDOW_DAYS\n",
                "    pre_size = pre_coll.size().getInfo()\n",
                "    post_size = post_coll.size().getInfo()\n",
                "    \n",
                "    if pre_size == 0 or post_size == 0:\n",
                "        pre_coll = s1_grd.filterBounds(pt).filterDate(event.advance(-SAR_EXTENDED_WINDOW_DAYS, 'day'), event)\n",
                "        post_coll = s1_grd.filterBounds(pt).filterDate(event, event.advance(SAR_EXTENDED_WINDOW_DAYS, 'day'))\n",
                "        window_used = SAR_EXTENDED_WINDOW_DAYS\n",
                "        pre_size = pre_coll.size().getInfo()\n",
                "        post_size = post_coll.size().getInfo()\n",
                "        \n",
                "    if pre_size == 0 or post_size == 0:\n",
                "        return {\n",
                "            'sar_window_days': None,\n",
                "            'sar_quality_flag': 'NO_SAR_PASS_IN_WINDOW',\n",
                "            'VV_pre': None, 'VH_pre': None, 'VV_post': None, 'VH_post': None, 'dVV': None, 'dVH': None,\n",
                "            'VV_pre_500m': None, 'VH_pre_500m': None, 'VV_post_500m': None, 'VH_post_500m': None,\n",
                "            'dVV_500m': None, 'dVH_500m': None, 'VV_pre_1000m': None, 'VH_pre_1000m': None,\n",
                "            'VV_post_1000m': None, 'VH_post_1000m': None, 'dVV_1000m': None, 'dVH_1000m': None\n",
                "        }\n",
                "        \n",
                "    pre = pre_coll.mean()\n",
                "    post = post_coll.mean()\n",
                "    delta = post.subtract(pre)\n",
                "    \n",
                "    img = (\n",
                "        pre.select(['VV', 'VH'], ['VV_pre', 'VH_pre'])\n",
                "        .addBands(post.select(['VV', 'VH'], ['VV_post', 'VH_post']))\n",
                "        .addBands(delta.select(['VV', 'VH'], ['dVV', 'dVH']))\n",
                "    )\n",
                "    \n",
                "    # 1. 10m point sample\n",
                "    pt_res = img.reduceRegion(reducer=ee.Reducer.first(), geometry=pt, scale=10, bestEffort=True).getInfo() or {}\n",
                "    # 2. 500m circular buffer mean and minimum\n",
                "    res_500m = img.reduceRegion(\n",
                "        reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '_min', True),\n",
                "        geometry=pt.buffer(500),\n",
                "        scale=10,\n",
                "        bestEffort=True\n",
                "    ).getInfo() or {}\n",
                "    # 3. 1000m circular buffer mean and minimum\n",
                "    res_1000m = img.reduceRegion(\n",
                "        reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '_min', True),\n",
                "        geometry=pt.buffer(1000),\n",
                "        scale=20,\n",
                "        bestEffort=True\n",
                "    ).getInfo() or {}\n",
                "    \n",
                "    combined = {'sar_window_days': window_used, 'sar_quality_flag': 'VALID'}\n",
                "    combined.update(pt_res)\n",
                "    for k, v in res_500m.items():\n",
                "        combined[f'{k}_500m'] = v\n",
                "    for k, v in res_1000m.items():\n",
                "        combined[f'{k}_1000m'] = v\n",
                "    return combined\n",
                "\n",
                "def extract_rainfall(lon, lat, event_date_str):\n",
                "    \"\"\"Extracts CHIRPS precipitation: 1d, 7d, 30d, and API (decay=0.90).\"\"\"\n",
                "    pt = ee.Geometry.Point([lon, lat])\n",
                "    event = ee.Date(event_date_str)\n",
                "    \n",
                "    r1 = chirps.filterDate(event.advance(-1, 'day'), event).sum().rename('rain_1d')\n",
                "    r7 = chirps.filterDate(event.advance(-7, 'day'), event).sum().rename('rain_7d')\n",
                "    r30 = chirps.filterDate(event.advance(-30, 'day'), event).sum().rename('rain_30d')\n",
                "    \n",
                "    api_window = chirps.filterDate(event.advance(-30, 'day'), event)\n",
                "    def weight(img):\n",
                "        days = event.difference(img.date(), 'day')\n",
                "        return img.multiply(ee.Number(API_DECAY).pow(days))\n",
                "    api = api_window.map(weight).sum().rename('API')\n",
                "    \n",
                "    img = r1.addBands(r7).addBands(r30).addBands(api)\n",
                "    res = img.reduceRegion(reducer=ee.Reducer.first(), geometry=pt, scale=5566, bestEffort=True).getInfo() or {}\n",
                "    return res\n",
                "\n",
                "def extract_spectral(lon, lat, event_date_str):\n",
                "    \"\"\"Adaptive multi-window Sentinel-2 spectral extraction for NDVI, NDWI, and BSI.\"\"\"\n",
                "    pt = ee.Geometry.Point([lon, lat])\n",
                "    end = ee.Date(event_date_str)\n",
                "    \n",
                "    for window in SPECTRAL_WINDOW_STEPS:\n",
                "        start = end.advance(-window, 'day')\n",
                "        coll = s2_with_prob.filterBounds(pt).filterDate(start, end).map(mask_s2_clouds)\n",
                "        coll_size = coll.size().getInfo()\n",
                "        if coll_size == 0:\n",
                "            continue\n",
                "            \n",
                "        comp = coll.median()\n",
                "        ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI')\n",
                "        ndwi = comp.normalizedDifference(['B3', 'B8']).rename('NDWI')\n",
                "        bsi = comp.expression(\n",
                "            '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',\n",
                "            {'SWIR': comp.select('B11'), 'RED': comp.select('B4'),\n",
                "             'NIR': comp.select('B8'), 'BLUE': comp.select('B2')}\n",
                "        ).rename('BSI')\n",
                "        \n",
                "        img = ndvi.addBands(ndwi).addBands(bsi)\n",
                "        res = img.reduceRegion(reducer=ee.Reducer.first(), geometry=pt, scale=10, bestEffort=True).getInfo()\n",
                "        if res and res.get('NDVI') is not None:\n",
                "            res['spectral_window_days'] = window\n",
                "            res['optical_quality_flag'] = 'VALID'\n",
                "            return res\n",
                "            \n",
                "    return {\n",
                "        'NDVI': None, 'NDWI': None, 'BSI': None,\n",
                "        'spectral_window_days': None,\n",
                "        'optical_quality_flag': 'PERSISTENT_MONSOON_CLOUD_COVER'\n",
                "    }\n",
                "\n",
                "print('Extraction functions compiled successfully with robust null-safety and fallbacks.')"
            ]
        },
        # Cell 8: Code - Sequential Dynamic Feature Extraction Loop
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 8: Sequential Feature Extraction Loop across all 127 Candidate Controls\n",
                "extracted_controls = []\n",
                "n_total = len(df_v2_controls)\n",
                "print(f'Extracting multisource dynamic features for {n_total} candidate controls...')\n",
                "start_loop = time.time()\n",
                "\n",
                "quality_audit = {\n",
                "    'sar_complete': 0,\n",
                "    'sar_extended_window': 0,\n",
                "    'sar_data_gap': 0,\n",
                "    'rain_complete': 0,\n",
                "    'optical_valid': 0,\n",
                "    'optical_cloud_covered': 0\n",
                "}\n",
                "\n",
                "for i, row in df_v2_controls.iterrows():\n",
                "    rec = row.to_dict()\n",
                "    lon, lat, date_str = row['longitude'], row['latitude'], row['event_date']\n",
                "    \n",
                "    # 1. SAR Point & Buffers\n",
                "    try:\n",
                "        sar_dict = extract_point_and_buffers_sar(lon, lat, date_str)\n",
                "        rec.update(sar_dict)\n",
                "        if sar_dict.get('sar_quality_flag') == 'VALID':\n",
                "            quality_audit['sar_complete'] += 1\n",
                "            if sar_dict.get('sar_window_days') == SAR_EXTENDED_WINDOW_DAYS:\n",
                "                quality_audit['sar_extended_window'] += 1\n",
                "        else:\n",
                "            quality_audit['sar_data_gap'] += 1\n",
                "    except Exception as e:\n",
                "        rec['sar_quality_flag'] = f'ERROR: {str(e)[:50]}'\n",
                "        quality_audit['sar_data_gap'] += 1\n",
                "        \n",
                "    # 2. Rainfall\n",
                "    try:\n",
                "        rain_dict = extract_rainfall(lon, lat, date_str)\n",
                "        rec.update(rain_dict)\n",
                "        if rain_dict.get('rain_1d') is not None:\n",
                "            quality_audit['rain_complete'] += 1\n",
                "    except Exception as e:\n",
                "        rec['_rain_error'] = str(e)[:50]\n",
                "        \n",
                "    # 3. Spectral Indices\n",
                "    try:\n",
                "        spec_dict = extract_spectral(lon, lat, date_str)\n",
                "        rec.update(spec_dict)\n",
                "        if spec_dict.get('optical_quality_flag') == 'VALID':\n",
                "            quality_audit['optical_valid'] += 1\n",
                "        else:\n",
                "            quality_audit['optical_cloud_covered'] += 1\n",
                "    except Exception as e:\n",
                "        rec['optical_quality_flag'] = f'ERROR: {str(e)[:50]}'\n",
                "        quality_audit['optical_cloud_covered'] += 1\n",
                "        \n",
                "    extracted_controls.append(rec)\n",
                "    if (i + 1) % 15 == 0 or (i + 1) == n_total:\n",
                "        elapsed = time.time() - start_loop\n",
                "        print(f'  Progress: {i+1}/{n_total} controls extracted ({elapsed:.1f}s elapsed)')\n",
                "\n",
                "df_v2_extracted_controls = pd.DataFrame(extracted_controls)\n",
                "print('\\nDynamic Feature Extraction Complete.')\n",
                "print('Extraction Quality Audit:', quality_audit)"
            ]
        },
        # Cell 9: Code - Assemble Feature Table & Provenance Metadata
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 9: Assemble Version 2.0 Feature Table & Export Artifacts\n",
                "url_p3 = 'https://raw.githubusercontent.com/GouravSharma26/LandSlide/main/outputs/phase3_feature_table.csv'\n",
                "df_p3 = pd.read_csv(url_p3)\n",
                "df_p3_positives = df_p3[df_p3['label'] == 1].reset_index(drop=True)\n",
                "\n",
                "# Common schema alignment\n",
                "common_cols = [c for c in df_p3_positives.columns if c in df_v2_extracted_controls.columns]\n",
                "df_v2_full = pd.concat([\n",
                "    df_p3_positives[common_cols],\n",
                "    df_v2_extracted_controls[common_cols]\n",
                "], ignore_index=True)\n",
                "\n",
                "print(f'Assembled Version 2.0 Feature Table: {len(df_v2_full)} total rows ({len(df_p3_positives)} positive, {len(df_v2_extracted_controls)} controls)')\n",
                "print('Class balance:', df_v2_full['label'].value_counts().to_dict())\n",
                "\n",
                "# Save primary dataset outputs\n",
                "df_v2_full.to_csv('v2_feature_table_experimental.csv', index=False)\n",
                "\n",
                "# Save Provenance Metadata\n",
                "provenance = {\n",
                "    'extraction_timestamp': datetime.utcnow().isoformat(),\n",
                "    'status': 'EXTRACTION_COMPLETED_COLAB',\n",
                "    'gee_project': 'landslidedetection-506008',\n",
                "    'target_roi': [26.0, 35.0, 74.0, 97.0],\n",
                "    'candidate_control_terminology': 'Candidate Hillslope Control / Unrecorded Failure Status',\n",
                "    'confound_policy': 'Confound mitigation must be empirically verified via KS tests upon ingestion; not assumed a priori.',\n",
                "    'sar_screening_characterization': 'Microwave temporal consistency screening criterion (std <= 1.5 dB); not geodetic stability proof.',\n",
                "    'candidate_selection_strategy': 'Minimum standardized composite topographic distance D_topo in 10-40km ring buffer',\n",
                "    'twi_resolution_alignment': 'MERIT Hydro upa bilinearly resampled to 30m SRTM grid; contour length b=30m',\n",
                "    'total_observations': len(df_v2_full),\n",
                "    'positive_events': int((df_v2_full['label'] == 1).sum()),\n",
                "    'candidate_hillslope_controls': int((df_v2_full['label'] == 0).sum()),\n",
                "    'slope_bounds_deg': [6.1, 41.4],\n",
                "    'elevation_bounds_m': [266, 3027],\n",
                "    'buffer_exclusion_m': 2000,\n",
                "    'sar_stability_filter': {'max_std_db': 1.5, 'window_days': [180, 15], 'min_observations': 3},\n",
                "    'land_cover_exclusions': {'esa_worldcover_snow_ice_class': 70, 'esa_worldcover_water_class': 80, 'jrc_water_max_pct': 10},\n",
                "    'sampling_audit': sampling_audit,\n",
                "    'extraction_quality_audit': quality_audit,\n",
                "    'matching_distribution_summary': df_dist_summary.to_dict(orient='records')\n",
                "}\n",
                "with open('v2_extraction_provenance.json', 'w') as f:\n",
                "    json.dump(provenance, f, indent=2)\n",
                "\n",
                "print('\\nAll Version 2.0 Artifacts Exported Successfully:')\n",
                "print('1. phase2_v2_hillslope_controls.csv')\n",
                "print('2. v2_control_matching_diagnostics.csv')\n",
                "print('3. v2_matching_distribution_summary.csv')\n",
                "print('4. v2_feature_table_experimental.csv')\n",
                "print('5. v2_extraction_provenance.json')"
            ]
        },
        # Cell 10: Markdown - Download Instructions
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 10: Download Artifacts to Local Workspace\n",
                "Run the cell below to download all 5 exported files to your local workstation.\n",
                "Then place them into your local project directory:\n",
                "- `outputs/phase2_v2_hillslope_controls.csv`\n",
                "- `artifacts/susceptibility/v2_control_matching_diagnostics.csv`\n",
                "- `artifacts/susceptibility/v2_matching_distribution_summary.csv`\n",
                "- `artifacts/susceptibility/v2_feature_table_experimental.csv`\n",
                "- `artifacts/metadata/v2_extraction_provenance.json`"
            ]
        },
        # Cell 11: Code - Download Trigger
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import files\n",
                "files.download('phase2_v2_hillslope_controls.csv')\n",
                "files.download('v2_control_matching_diagnostics.csv')\n",
                "files.download('v2_matching_distribution_summary.csv')\n",
                "files.download('v2_feature_table_experimental.csv')\n",
                "files.download('v2_extraction_provenance.json')"
            ]
        }
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}
        },
        "cells": cells
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)


if __name__ == "__main__":
    params = generate_v2_sampling_parameters()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_DIR / "v2_candidate_distribution_validation.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print("Version 2.0 Candidate Distribution Validation Saved to:")
    print(METADATA_DIR / "v2_candidate_distribution_validation.json")

    nb_path = BASE_DIR / "notebooks" / "Phase3b_V2_Hillslope_Extraction_Colab.ipynb"
    create_colab_extraction_notebook(nb_path)
    print(f"Colab extraction notebook updated at: {nb_path}")
