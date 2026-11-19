# LandSlide — GeoTrustNet Intelligence

## Vision
Evolve the existing LandSlide project into a unified geospatial intelligence application with a reproducible deep-learning research engine. Keep the current dashboard and useful pipelines; introduce GeoTrustNet as an independently testable model subsystem.

## Research title
**GeoTrustNet: Uncertainty-Aware Spatio-Temporal Multimodal Deep Learning for Reliable Landslide Detection and Geographic Generalization**

This title describes the intended research direction, not a proven result or confirmed novelty claim.

## Product capabilities
1. **Landslide event detection:** pixel-level segmentation from satellite observations where suitable labels exist.
2. **Landslide susceptibility:** spatial susceptibility estimates from terrain/environmental predictors, retained as a distinct prediction task.
3. **Uncertainty and reliability:** show where predictions are uncertain and which input modalities contributed.
4. **Geospatial exploration:** retain the existing map/dashboard experience and connect it to validated model outputs.
5. **Research evaluation:** compare baselines and proposed components using leakage-safe geographic/event splits.

## Initial scope boundaries
- First deliver a reliable detection/segmentation pipeline.
- Use Sentinel-1, Sentinel-2, and DEM only where data are available, aligned, and documented.
- Add rainfall as a temporal predictor only when timestamps, spatial resolution, and target horizon are defensibly aligned.
- Do not claim operational early warning or disaster-risk forecasting without prospective/future-time validation.
- Do not merge susceptibility and event segmentation into one metric or one label definition.

## Expected user-facing outputs
- Landslide segmentation mask and probability map.
- Optional susceptibility layer from its separate model.
- Predictive uncertainty map and summary.
- Modality availability/quality indicators.
- Region/event metadata, data provenance, model version, and inference timestamp.

## Suggested stack
- Python, PyTorch
- Rasterio, GeoPandas, Xarray, NumPy, Pandas
- scikit-learn for classical baselines
- TorchMetrics or equivalent for metrics
- MLflow or a lightweight structured experiment logger
- Streamlit for the existing dashboard; FastAPI only if an API boundary is useful
- Docker for reproducible deployment

## Repository integration principle
Do not replace the existing project structure blindly. After audit, map these conceptual modules into the repository's current conventions. Add new modules only where they have a clear owner and interface.

## First action for Antigravity
Read `AGENTS.md` and all files in `docs/geotrustnet/`. Then perform Phase 0 only: audit the repository and write `docs/REPOSITORY_AUDIT.md`, `docs/IMPLEMENTATION_PLAN.md`, and a preservation/rollback plan. Do not begin broad refactoring until the audit is reviewed.
