# Implementation Plan — LandSlide × GeoTrustNet

**Project Title:** LandSlide — GeoTrustNet Intelligence  
**Document:** Phased Implementation Roadmap and File-Level Plan  
**Status:** Phase 0 (Planning & Architecture)  

---

## 1. Architectural Strategy: The Dual-Track Integration

To honor the non-negotiable rules in `AGENTS.md` and avoid breaking the existing operational Streamlit application, the system adopts a **Dual-Track Modular Architecture**:

```text
                                LANDSLIDE PLATFORM
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
   TRACK 1: Classical Intelligence                           TRACK 2: GeoTrustNet Deep Learning
   (Diagnostic & Regional Susceptibility)                    (Spatio-Temporal Pixel Segmentation)
   ─────────────────────────────────────                     ───────────────────────────────────
   • Paradigm: Tabular Point-Zone Features                   • Paradigm: Multimodal Spatio-Temporal Rasters
   • Data: N=254 points, 28 multi-sensor bands               • Data: Aligned Sentinel-1/2 time series & DEM
   • Models: Random Forest (500 trees), XGBoost              • Models: U-Net, U-TAE, GeoTrustNet (Reliability)
   • Validation: 100 km Purged Spatial K-Fold                • Validation: Region/Event Held-Out Holdouts
   • Role: Macro-regional hazard planning                    • Role: High-resolution landslide scar mapping
   • Status: 100% complete, 59/59 tests pass                 • Status: New research engine under `src/`
```

---

## 2. Target File-Level Mapping

We map the proposed GeoTrustNet modules into the existing repository without creating duplicate modules or altering existing dashboard paths:

```text
LANDSLIDE/
├── dashboard/                              # [PRESERVED] Existing Streamlit application
│   ├── app.py                             # Preserved; optionally add GeoTrustNet tab in Phase 6
│   └── pages/                             # 7 existing pages preserved 100%
├── outputs/                                # [PRESERVED] 6 production CSV tables
├── artifacts/                              # [PRESERVED] 21 standardized artifacts
├── tests/                                  # [EXTENDED] Existing tests + new GeoTrustNet tests
│   ├── test_data_loader.py                 # [PRESERVED]
│   ├── test_maps.py                        # [PRESERVED]
│   ├── test_metrics.py                     # [PRESERVED]
│   ├── test_scientific_governance.py       # [PRESERVED]
│   ├── test_v2_sampling_and_features.py    # [PRESERVED]
│   ├── test_validation.py                  # [PRESERVED]
│   ├── test_geotrustnet_data.py            # [NEW: Phase 1] Data contracts & synthetic fixtures
│   ├── test_geotrustnet_models.py          # [NEW: Phase 2-5] Forward pass & shape tests
│   └── test_geotrustnet_metrics.py         # [NEW: Phase 1-6] Metric math & calibration checks
├── src/                                    # [NEW] GeoTrustNet research package
│   └── geotrustnet/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py                 # Spatio-temporal patch dataset reader
│       │   ├── transforms.py               # Normalization, augmentation, nodata masking
│       │   ├── geospatial.py               # CRS, affine transforms, raster alignment
│       │   └── splits.py                   # Event & region-held-out manifest generator
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoders.py                 # SAR (CNN), Optical (CNN/ResNet), Terrain (CNN)
│       │   ├── temporal.py                 # Temporal attention / temporal pooling
│       │   ├── fusion.py                   # Concatenation, equal-weight, reliability-aware
│       │   ├── decoder.py                  # U-Net multi-scale segmentation decoder
│       │   ├── uncertainty.py              # MC Dropout / Deep Ensemble heads
│       │   └── geotrustnet.py              # Complete unified model class
│       ├── training/
│       │   ├── __init__.py
│       │   ├── engine.py                   # PyTorch training loop, CPU smoke-tests, checkpointing
│       │   └── losses.py                   # Combined BCE + Dice + Focal loss
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── segmentation.py             # IoU, Dice/F1, Precision, Recall, PR-AUC
│       │   ├── calibration.py              # ECE (binned), Brier score, NLL
│       │   └── geographic.py               # Grouped cross-region evaluation
│       └── inference/
│           ├── __init__.py
│           └── predict.py                  # Georeferenced inference pipeline & mask export
├── configs/                                # [NEW] YAML/JSON experiment configurations
│   ├── data/
│   │   └── sen12landslides_spec.json       # Dataset contracts & band mapping
│   ├── models/
│   │   ├── unet_baseline.json              # Spatial single-modality baseline
│   │   └── geotrustnet_default.json        # Full multimodal reliability architecture
│   └── experiments/
│       └── ablation_matrix.json            # Structured experiment ledger
├── data/
│   ├── manifests/                          # Split manifests with SHA-256 hashes
│   └── synthetic/                          # Small CPU synthetic fixtures for unit tests
└── docs/
    ├── REPOSITORY_AUDIT.md                 # Complete Phase 0 audit report
    ├── IMPLEMENTATION_PLAN.md              # This plan
    ├── PRESERVATION_AND_ROLLBACK.md        # Rollback & safety protocol
    └── reports/                            # Concise phase reports (Phase 1 through Phase 7)
```

---

## 3. Phased Execution Roadmap

### Phase 1: Data Contracts, Provenance, and Leakage-Safe Splits
- **Goal:** Establish deterministic data contracts, raster fixtures, and leakage-safe splits.
- **Actions:**
  1. Define `configs/data/sen12landslides_spec.json` documenting band names, resolutions, nodata values, and licenses.
  2. Implement `src/geotrustnet/data/datasets.py` and `geospatial.py`.
  3. Create `src/geotrustnet/data/splits.py` supporting strictly grouped region/event holdout.
  4. Generate synthetic mini-raster fixtures under `data/synthetic/` so CPU tests run without downloading gigabytes of satellite data.
  5. Add unit tests in `tests/test_geotrustnet_data.py` (validating CRS, transforms, shapes, and zero train-test leakage).
- **Exit Criteria:** Deterministic data loader verified; 100% of unit tests pass.

### Phase 2: Baseline Segmentation (Single-Modality U-Net)
- **Goal:** Build and verify an end-to-end segmentation baseline (U-Net) with training, validation, and metrics.
- **Actions:**
  1. Implement `src/geotrustnet/models/decoder.py` and a standard U-Net in `src/geotrustnet/models/encoders.py`.
  2. Implement `src/geotrustnet/training/engine.py` and `losses.py` (BCE + Dice).
  3. Implement `src/geotrustnet/evaluation/segmentation.py` (IoU, Dice, Precision, Recall).
  4. Run CPU smoke-test to verify end-to-end forward/backward passes, metric calculation, and checkpoint saving.
  5. Add unit tests in `tests/test_geotrustnet_models.py`.
- **Exit Criteria:** End-to-end training runs; checkpointing and metrics verified on synthetic fixtures.

### Phase 3: Temporal Modeling Baseline (U-TAE / Temporal Attention)
- **Goal:** Implement temporal sequence modeling over multi-date observations.
- **Actions:**
  1. Implement `src/geotrustnet/models/temporal.py` (temporal attention / pooling supporting variable observation lengths and observation masks).
  2. Compare single-date spatial U-Net vs temporal U-Net on a common evaluation subset.
  3. Add tests verifying that temporal padding and missing timestamps do not cause NaNs.
- **Exit Criteria:** Temporal module executes cleanly on variable sequence lengths with valid masking.

### Phase 4: Multimodal Encoders & Naïve Fusion
- **Goal:** Ingest Sentinel-1 SAR (VV/VH), Sentinel-2 optical, and SRTM DEM rasters into dedicated branches.
- **Actions:**
  1. Implement SAR encoder, optical encoder, and terrain encoder in `src/geotrustnet/models/encoders.py`.
  2. Implement concatenation and equal-weight fusion in `src/geotrustnet/models/fusion.py`.
  3. Support modality availability masks (enabling graceful forward pass when optical imagery is cloud-covered).
  4. Add tests for missing modality robustness (e.g. forward pass with SAR-only or optical-only).
- **Exit Criteria:** Multimodal forward pass functions with any individual modality missing.

### Phase 5: GeoTrustNet Contribution (Reliability-Aware Fusion & Uncertainty)
- **Goal:** Implement learned reliability-aware cross-modal fusion and predictive uncertainty quantification.
- **Actions:**
  1. Implement learned reliability scoring \(r_m\) and normalized gating in `src/geotrustnet/models/fusion.py`.
  2. Implement Monte Carlo Dropout / Deep Evidential uncertainty in `src/geotrustnet/models/uncertainty.py`.
  3. Implement calibration evaluation in `src/geotrustnet/evaluation/calibration.py` (Expected Calibration Error, Brier score, Negative Log-Likelihood).
  4. Compare learned reliability fusion against naïve concatenation and equal-weight baselines.
- **Exit Criteria:** Reliability fusion produces normalized weights over available modalities; uncertainty maps correlate with segmentation errors.

### Phase 6: Existing Dashboard Integration (Non-Destructive)
- **Goal:** Connect GeoTrustNet inference outputs to the existing Streamlit dashboard without modifying existing pages.
- **Actions:**
  1. Create `src/geotrustnet/inference/predict.py` producing georeferenced output rasters (probability mask, binary decision, uncertainty map, modality weights).
  2. Add a new dedicated Section / Page in `dashboard/pages/` (e.g. `dashboard/pages/geotrustnet_explorer.py`) or an expandable tab that displays:
     - Predicted landslide segmentation mask.
     - Predictive uncertainty heatmap.
     - Modality reliability weights (showing how much SAR vs Optical was trusted).
     - Model lineage, version, and clear decision-support caveats.
  3. Verify all 7 original dashboard pages remain 100% functional and unmodified.
- **Exit Criteria:** Live dashboard displays GeoTrustNet outputs; existing test suite (59 tests) passes.

### Phase 7: Research Release & Reproducibility Package
- **Goal:** Package experiment ledgers, model cards, literature review, and paper-ready figures.
- **Actions:**
  1. Complete `LITERATURE_REVIEW.md` and formal research gap statement based on empirical evidence.
  2. Generate `MODEL_CARD.md` and dataset provenance statements.
  3. Export reproducibility configs and checkpoint release guides.
- **Exit Criteria:** All claims are backed by logged experiment artifacts; no unverified claims.
