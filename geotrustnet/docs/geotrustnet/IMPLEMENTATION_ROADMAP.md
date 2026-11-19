# Implementation Roadmap for Antigravity

## Phase 0 — Audit and preservation (no feature implementation)
Deliver:
- `docs/REPOSITORY_AUDIT.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/PRESERVATION_AND_ROLLBACK.md`
- current branch/status and baseline test report
- map of existing pages, pipelines, models, data, configs, and deployment

Exit criteria: no destructive changes; current functionality documented.

## Phase 1 — Data and evaluation foundations
- inspect actual datasets and licenses;
- define data manifest and schema;
- implement geospatial validation and label checks;
- define event/region-grouped splits;
- add synthetic test fixtures;
- verify metrics against small hand-computed examples.

Exit criteria: deterministic data loading and leakage-safe split manifests.

## Phase 2 — Baseline segmentation
- implement or adapt U-Net;
- train a small smoke-test run;
- implement validation and checkpointing;
- record resource requirements;
- establish a reproducible baseline.

Exit criteria: end-to-end training/inference works and metrics are verified.

## Phase 3 — Temporal baseline
- inspect whether samples have aligned timestamps and repeated observations;
- implement U-TAE if the data schema supports it, or a documented temporal alternative;
- compare against the spatial baseline.

Exit criteria: fair comparison on a common split/sample subset.

## Phase 4 — Multimodal baseline
- add SAR, optical, and terrain encoders incrementally;
- start with concatenation/equal-weight fusion;
- add modality masks and missing-data handling.

Exit criteria: robust forward pass with individual modalities missing.

## Phase 5 — GeoTrustNet contribution
- implement learned reliability-aware fusion;
- add uncertainty estimation and calibration;
- log modality weights and uncertainty diagnostics.

Exit criteria: complete ablation and robustness experiments.

## Phase 6 — Existing dashboard integration
- preserve existing dashboard pages and routes;
- add model inference as a separate service/module;
- show probability, mask, uncertainty, and modality availability;
- show model/dataset version and caveats;
- avoid claiming a live warning system.

Exit criteria: existing pages remain functional and new outputs are traceable.

## Phase 7 — Research release
- finalize literature review and gap statement;
- generate tables/figures directly from experiment logs;
- document limitations and reproducibility;
- prepare paper draft only from verified results;
- add model card and data statement.

## Phase control
At the end of each phase:
1. stop;
2. write a phase report under `docs/reports/`;
3. list changed files and tests;
4. list blockers and next phase;
5. do not proceed to the next phase without user approval.
