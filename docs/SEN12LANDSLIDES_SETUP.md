# Sen12Landslides Dataset Setup, Acquisition & Verification Guide

**Project:** Himalayan Landslide Intelligence System (GeoTrustNet)  
**Dataset:** Sen12Landslides: A Spatio-Temporal Dataset for Satellite-Based Landslide Detection  
**Citation:** Höhn et al., *Nature Scientific Data* (2025)  
**DOI:** [10.1038/s41597-025-06167-2](https://doi.org/10.1038/s41597-025-06167-2)  
**Official Repository:** [Hugging Face: paulhoehn/Sen12Landslides](https://huggingface.co/datasets/paulhoehn/Sen12Landslides)  
**License:** Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/))  

---

## 1. Dataset Overview

The Sen12Landslides dataset provides high-resolution, co-registered Earth Observation patches designed for machine learning-based landslide detection.

### Spatial & Temporal Dimensions
- **Spatial Resolution:** 10 meters ground sampling distance (GSD).
- **Patch Dimensions:** 128 × 128 pixels (1.28 km × 1.28 km per patch).
- **Coordinate Reference System:** EPSG:4326 / Local UTM projections.
- **Temporal Horizon:** Multi-temporal observations (up to 15 acquisitions per event patch).

### Modalities & Spectral Bands
| Sensor / Modality | Data Type | Dimensions | Valid Range | Nodata Value | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentinel-1 SAR** | Level-1 GRD C-Band | `[T, 2, 128, 128]` | $[-45.0, 10.0]$ dB | $-9999.0$ | VV & VH polarizations; all-weather penetration |
| **Sentinel-2 Optical** | Level-2A Surface Refl. | `[T, 10, 128, 128]` | $[0, 10000]$ | $-9999.0$ | Bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 |
| **Copernicus DEM** | GLO-30 DSM | `[2, 128, 128]` | Elev: $[-500, 9000]$ m, Slope: $[0, 90]^\circ$ | $-9999.0$ | Band 0: Elevation, Band 1: Slope |
| **Landslide Mask** | Ground Truth Polygon | `[1, 128, 128]` | $\{0, 1\}$ | $255$ | Binary: $0$ = Non-landslide, $1$ = Landslide |

---

## 2. Acquisition & Download Methods

> [!IMPORTANT]
> The full dataset is approximately 50 GB. The user must initiate the download explicitly; the repository does not download large datasets automatically.

### Method A: Automated Download via Repository Utility
Use the built-in acquisition script:
```bash
# Preview instructions and dry run:
python scripts/prepare_sen12landslides.py --dry-run

# Execute full download to data/raw/Sen12Landslides:
python scripts/prepare_sen12landslides.py --download --raw-dir data/raw/Sen12Landslides
```

### Method B: Hugging Face CLI Download
```bash
# 1. Install Hugging Face Hub:
pip install huggingface-hub

# 2. Login (optional, required if repo requires authentication):
huggingface-cli login

# 3. Download repository snapshot:
huggingface-cli download paulhoehn/Sen12Landslides --repo-type dataset --local-dir data/raw/Sen12Landslides
```

### Method C: Google Colab Interactive Download
In a Colab notebook cell:
```python
from huggingface_hub import snapshot_download

raw_path = snapshot_download(
    repo_id="paulhoehn/Sen12Landslides",
    repo_type="dataset",
    local_dir="/content/sen12landslides",
    resume_download=True
)
print("Downloaded dataset to:", raw_path)
```

### Method D: Kaggle Dataset Attachment
In Kaggle Notebooks:
1. Click **+ Add Data** in the right-hand panel.
2. Search for `Sen12Landslides` or upload from Hugging Face.
3. Path will be mounted read-only at `/kaggle/input/sen12landslides`.

---

## 3. Manifest Generation & Leakage-Safe Splitting

Once raw patches are downloaded or mounted, generate deterministic split manifests:
```bash
python scripts/prepare_sen12landslides.py \
    --raw-dir data/raw/Sen12Landslides \
    --output-dir data/manifests \
    --seed 42
```

This generates:
- `data/manifests/sen12landslides_split_manifest.csv`
- `data/manifests/sen12landslides_split_manifest.json`

The partitioning algorithm enforces **zero geographic leakage** by holding out entire `region_id` groups:
- **Train Split (70%):** Training regions
- **Validation Split (15%):** Hyperparameter tuning and model selection
- **Test Split (15%):** Final out-of-region evaluation

---

## 4. Dataset Inspection & Integrity Verification

To inspect any manifest and verify class balance and zero leakage:
```bash
python scripts/inspect_sen12landslides.py \
    --manifest data/manifests/sen12landslides_split_manifest.csv \
    --output-report docs/reports/dataset_inspection_report.json
```

The inspector will verify:
1. `is_leakage_free_disjoint == True` (Zero region overlap between splits).
2. Number of unique regions and events.
3. Modality availability across all patches.
4. Positive class prevalence (typically $\approx 2.5\% - 4\%$ of all pixels).
5. Duplicate file SHA-256 detection.
