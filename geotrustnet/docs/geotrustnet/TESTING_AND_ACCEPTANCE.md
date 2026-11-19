# Testing and Acceptance Criteria

## Repository preservation
- Existing application entry point still runs.
- Existing dashboard pages/routes still load.
- Existing data processing commands remain documented and functional.
- No data files or user changes are removed.
- Git status is clean apart from intentional changes at phase boundaries.

## Data tests
- CRS and raster transform checks.
- Modality dimensions and alignment checks.
- Nodata and validity-mask handling.
- Label values and mask alignment.
- Timestamp ordering and future-data leakage checks.
- Region/event split disjointness.
- Dataset manifest reproducibility.

## Model tests
- CPU forward pass.
- Input/output shape checks.
- Finite logits/loss.
- Missing modality behavior.
- Temporal padding/masking behavior.
- Fusion weight normalization.
- Checkpoint save/load.
- Deterministic inference where supported.

## Evaluation tests
- Metrics verified on a tiny synthetic example.
- No test-set use for model selection.
- Per-region aggregation correct.
- Calibration metric implementation documented.
- Threshold selection uses validation data only.

## Dashboard tests
- Existing dashboard smoke test.
- New map layers have CRS and bounds.
- Uncertainty legend is clear and not described as certainty.
- Model version and data provenance visible.
- Empty/missing inference results handled gracefully.

## Acceptance report
Each phase report must include:
- summary;
- changed files;
- commands run;
- pass/fail counts;
- screenshots only if they add value;
- known limitations;
- exact next action.
