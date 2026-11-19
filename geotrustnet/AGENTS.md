# AGENTS.md — LandSlide × GeoTrustNet

## Mission
Upgrade the existing `LandSlide` repository into a research-grade, reproducible geospatial deep-learning system. Preserve working functionality and existing data assets. Integrate the proposed **GeoTrustNet** research model into the existing application rather than creating a disconnected replacement.

**Working product identity**
- Application/project: **LandSlide — GeoTrustNet Intelligence**
- Research model: **GeoTrustNet**
- Research focus: uncertainty-aware spatio-temporal multimodal learning for landslide detection and geographic generalization.

## Non-negotiable operating rules
1. **Audit before editing.** Inspect the complete repository, current branch, git status, README, dependency files, data layout, dashboard pages, existing models, tests, and deployment configuration. Produce `docs/REPOSITORY_AUDIT.md` before implementation.
2. Do not delete, rename, or rewrite existing project components until their purpose and dependencies are understood.
3. Preserve existing dashboard behavior and current working deployment unless a planned migration is approved.
4. Do not fabricate datasets, labels, metrics, citations, model results, or completed work.
5. Do not claim that GeoTrustNet is novel or publication-ready until literature review and experiments support those claims.
6. Keep raw data, credentials, API keys, and large model weights out of Git. Use documented data download instructions and `.gitignore`.
7. Use reproducible configuration files, fixed random seeds, explicit dataset versions, and logged experiment metadata.
8. Prefer small, reviewable phases. Run relevant tests after each phase and report exact commands and outcomes.
9. Avoid gratuitous UI redesign, emojis, exaggerated marketing language, and generic AI-generated filler. Keep the interface clean and professional.
10. Never use random patch-level splitting where nearby patches or the same landslide event can leak between train and test. Prefer event/region-held-out evaluation.
11. Treat all performance targets as unknown until measured. Report confidence intervals where feasible.
12. Do not silently change the existing data schema or dashboard routes.

## Existing project context to verify
The existing LandSlide project has previously been described as a Himalayan landslide intelligence dashboard with event detection and susceptibility analysis, a Streamlit entry point at `dashboard/app.py`, and multiple dashboard pages. Historical reports mentioned a COOLR inventory filtering pipeline and tests. These are **leads for the audit, not permission to assume the current checkout matches them**. Verify every path, count, page, metric, and test against the actual repository.

## Research direction
GeoTrustNet should initially focus on:
- pixel-level landslide segmentation/detection;
- multimodal satellite input (Sentinel-1 SAR, Sentinel-2 optical, terrain/DEM);
- temporal modeling where time-series observations are available;
- reliability-aware fusion that can account for noisy or missing modalities;
- uncertainty estimation and calibration;
- geographic generalization to held-out regions/events.

Susceptibility mapping is a related but distinct task. Retain the existing susceptibility workflow and integrate/upgrade it as a separate task with its own labels, split protocol, and metrics. Do not present susceptibility as equivalent to event detection. Rainfall-based forecasting is a later extension unless the dataset has correctly aligned historical rainfall and future-event targets.

## Required development workflow
- Phase 0: repository audit and preservation plan.
- Phase 1: data contracts, provenance, and leakage-safe splits.
- Phase 2: reproduce a simple baseline and validate the pipeline.
- Phase 3: implement benchmark baseline(s), including U-Net and, where compatible, U-TAE.
- Phase 4: implement multimodal encoders and fusion.
- Phase 5: add reliability-aware fusion and uncertainty.
- Phase 6: ablations, robustness, geographic holdout, calibration.
- Phase 7: integrate inference and visualizations into the existing dashboard.
- Phase 8: paper-ready reproducibility package and documentation.

## Engineering standards
- Python type hints for public interfaces.
- Clear module boundaries: data, models, training, evaluation, inference, dashboard.
- Configuration-driven experiments (YAML/TOML/JSON as appropriate).
- Structured logging; save configs, seeds, code revision, dataset version, and metrics with each run.
- Unit tests for data shapes, masks, missing modalities, fusion weights, metrics, and checkpoint loading.
- CPU smoke tests must work even if full GPU training is unavailable.
- GPU training must fail with a clear diagnostic if CUDA is unavailable; do not silently pretend to train.
- Use mixed precision only when supported and numerically safe.
- Avoid downloading large datasets automatically during ordinary app startup.

## Definition of done for every phase
1. Files changed are listed.
2. Tests/commands are listed with exact results.
3. Any unimplemented items and known limitations are disclosed.
4. No unverified performance claims are added.
5. A concise phase report is saved under `docs/reports/`.
