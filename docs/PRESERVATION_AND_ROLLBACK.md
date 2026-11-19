# Preservation and Rollback Plan — LandSlide × GeoTrustNet

**Document:** System Safety, Backward Compatibility, and Rollback Protocol  
**Date:** 2026-11-19  
**Status:** Phase 0 (Baseline Governance)  

---

## 1. Core Preservation Directives

To guarantee that the existing operational product is never compromised during GeoTrustNet research development, the following non-negotiable preservation policies are enforced:

### Policy A: Zero Deletions in Existing Data and Models
- **Existing Tabular Assets:** `Landslide_Reports.csv`, all files in `outputs/` (`cleaned_observation_records.csv`, `multisource_feature_matrix.csv`, `regional_susceptibility_grid.csv`, etc.), and all 21 artifacts in `artifacts/` must NEVER be deleted, modified, or overwritten.
- **Existing ML Models:** `artifacts/detection/detection_models.joblib` and `artifacts/susceptibility/susceptibility_models.joblib` remain the immutable production baselines for Tasks A and B in the current dashboard.

### Policy B: Zero Regression in Existing Dashboard Behavior
- **Streamlit Routes & Pages:** All 7 existing dashboard pages (`overview`, `geographic_explorer`, `data_quality`, `environmental_analysis`, `event_detection`, `susceptibility_prediction`, `explainability_uncertainty`) must remain functional with zero breaking changes.
- **Component Integrity:** Existing visual components (`charts.py`, `maps.py`, `sidebar.py`, `kpi_cards.py`) must continue to serve the current pages. GeoTrustNet will introduce new components or interfaces rather than altering existing functions.

### Policy C: Test Suite Invariance
- The existing 59 unit and regression tests in `tests/` (`test_data_loader.py`, `test_maps.py`, `test_metrics.py`, `test_scientific_governance.py`, `test_v2_sampling_and_features.py`, `test_validation.py`) must pass 100% at every single commit and phase boundary.
- Any change that causes even 1 existing test to fail is considered an immediate blocker and must be reverted.

---

## 2. Protected Files Matrix

| File / Directory | Protection Level | Purpose | Modification Policy |
|---|---|---|---|
| `dashboard/app.py` | **High** | Primary application router | Only non-breaking menu extension allowed in Phase 6 |
| `dashboard/pages/*.py` (7 files) | **Locked** | Production research dashboard pages | Strictly READ-ONLY; no refactoring allowed |
| `dashboard/components/*.py` | **Locked** | Visual widgets and Plotly/PyDeck maps | Strictly READ-ONLY |
| `dashboard/utils/*.py` | **Locked** | Data loaders, metrics, preprocessing | Strictly READ-ONLY |
| `artifacts/**/*` (21 files) | **Locked** | Standardized research artifacts | Strictly READ-ONLY |
| `outputs/*.csv` (6 files) | **Locked** | Cleaned datasets & model predictions | Strictly READ-ONLY |
| `Landslide_Reports.csv` | **Locked** | Raw NASA COOLR ground truth (10.5 MB) | Strictly READ-ONLY |
| `tests/test_*.py` (6 files) | **Locked** | 59 regression tests | Strictly READ-ONLY (only add new test files) |
| `scripts/build_research_artifacts.py` | **Locked** | Artifact generation pipeline | Strictly READ-ONLY |

---

## 3. Sandboxed Workspaces for GeoTrustNet

All new GeoTrustNet development is strictly quarantined into dedicated namespaces:
1. **Source Code:** Quarantined under `src/geotrustnet/` (completely isolated from `dashboard/` and existing `scripts/`).
2. **Configurations:** Quarantined under `configs/` (`configs/data/`, `configs/models/`, `configs/experiments/`).
3. **Data Fixtures & Manifests:** Quarantined under `data/synthetic/` and `data/manifests/`.
4. **New Unit Tests:** Quarantined under dedicated test files (`tests/test_geotrustnet_*.py`).
5. **Phase Documentation:** Saved under `docs/reports/` and `docs/geotrustnet/`.

---

## 4. Phase-by-Phase Verification & Rollback Protocol

At every phase boundary (Phases 1 through 7):

1. **Pre-Commit Verification Checklist:**
   ```bash
   # 1. Verify all 59 existing regression tests pass
   python -m pytest tests/ -k "not geotrustnet" -v

   # 2. Verify all new GeoTrustNet tests pass
   python -m pytest tests/ -k "geotrustnet" -v

   # 3. Verify dashboard loads cleanly without runtime errors
   python -c "import dashboard.app; print('Dashboard OK')"

   # 4. Check git status to ensure no locked files were touched
   git status
   ```

2. **Rollback Trigger Conditions:**
   - Any failure in the existing 59 tests.
   - Any runtime error or layout breakage on the 7 existing dashboard pages.
   - Any inadvertent modification or deletion of files in `artifacts/`, `outputs/`, or `dashboard/pages/`.
   - Any unhandled memory spike or ungraceful crash during CPU smoke-tests.

3. **Rollback Execution Command:**
   If a phase introduces breaking regressions, revert to the last verified commit immediately:
   ```bash
   # Hard revert of uncommitted working directory changes
   git checkout -- dashboard/ outputs/ artifacts/ tests/
   git clean -fd src/geotrustnet/ configs/  # (if needed to restore clean state)

   # Or revert to the Phase 0 baseline commit (323bb80)
   git reset --hard 323bb80
   ```

4. **Phase Sign-Off Requirement:**
   - No phase will be marked complete without a signed-off report under `docs/reports/phaseX_report.md` stating exact commands run, test pass counts, and explicit user approval before advancing to the next phase.
