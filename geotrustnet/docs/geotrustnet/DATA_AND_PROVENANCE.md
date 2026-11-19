# Data, Provenance, and Preprocessing Contract

## Principle
Never assume that a dataset contains every modality, every timestamp, or pixel-perfect alignment. Inspect the actual files and metadata first. Maintain provenance from source data to each training sample.

## Candidate research dataset
**Sen12Landslides** is a relevant candidate for satellite time-series landslide segmentation:
- Paper: Höhn et al., “A Spatio-Temporal Dataset for Satellite-Based Landslide Detection,” *Scientific Data* (2025).
- DOI: https://doi.org/10.1038/s41597-025-06167-2
- Paper: https://www.nature.com/articles/s41597-025-06167-2
- Code: https://github.com/PaulH97/Sen12Landslides
- Dataset: https://huggingface.co/datasets/paulhoehn/Sen12Landslides

The dataset description reports Sentinel-1 SAR, Sentinel-2 optical, terrain/DEM-related information, temporal observations, and landslide annotations. Verify the current dataset card, release version, licensing, file layout, available aligned samples, and exact modality coverage before use.

## Existing LandSlide data
The existing repository may include a separate landslide inventory and derived features. Audit and document:
- source/provider and download date;
- license and terms of use;
- geographic bounds and coordinate reference system;
- spatial and temporal resolution;
- label creation method and known label limitations;
- filtering steps and row/pixel counts;
- missingness and invalid-value handling;
- any manual edits;
- train/validation/test assignment.

Do not assume the existing inventory is compatible with Sen12Landslides. Treat them as separate data sources unless a defensible harmonization protocol is developed.

## Required manifest
Create a machine-readable manifest for every dataset version. Suggested fields:
- dataset_id, version, source_url, license, retrieval_date
- region_id, event_id, sample_id
- acquisition_datetime, sensor, modality
- CRS, transform, width, height, pixel_size
- band_names, units, nodata_value
- label_source, label_version
- split, preprocessing_version, checksum

## Preprocessing requirements
1. Validate CRS, affine transform, dimensions, pixel size, and nodata.
2. Align modalities spatially only where valid; document resampling method per data type.
3. Use nearest-neighbor resampling for categorical masks; choose and justify continuous-image resampling.
4. Normalize each modality using training-set statistics only.
5. Preserve nodata/validity masks; do not convert missing pixels into real zero-valued observations without an explicit mask.
6. Keep temporal order and acquisition timestamps.
7. Ensure no post-event/future information enters a forecasting input.
8. Check label alignment visually and programmatically.
9. Store patch-to-region/event lineage.
10. Keep preprocessing deterministic and versioned.

## Split protocol
Preferred:
- Group by region and/or landslide event before creating splits.
- Hold out complete regions/events for final testing.
- Keep validation geography separate from training where feasible.
- Do not randomly split adjacent patches from the same event across train and test.
- Save split manifests and hashes so experiments reuse identical partitions.

## Data quality report
Report per modality:
- sample count and coverage;
- missing/invalid fraction;
- temporal observation count;
- band availability;
- class/pixel imbalance;
- label quality caveats;
- geographic distribution;
- alignment failures.

## Data protection
Raw datasets and large derived rasters should not be committed unless repository policy and licensing explicitly allow it. Document download/setup instructions and provide a small synthetic fixture for tests.
