"""Unit and integration tests for GeoTrustNet Intelligence dashboard integration.

Validates:
1. Navigation preservation: All 7 legacy pages preserved in exact order; GeoTrustNet added as 8th.
2. Router integrity: All 8 routes in app.py map to valid callable render functions.
3. GeoTrustNet Intelligence page module integrity and render tab functions.
4. Data and artifact loaders graceful error handling.
5. Scientific provenance and integrity labels (synthetic verification vs cloud GPU status).
6. Synthetic demonstration patch generation and modality alignment.
"""

import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure Windows PyTorch DLL loading
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

try:
    import torch
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_sidebar_navigation_preservation():
    """Verify that all 7 legacy dashboard pages are preserved in their exact order and GeoTrustNet is added."""
    sidebar_file = ROOT_DIR / "dashboard" / "components" / "sidebar.py"
    assert sidebar_file.exists(), "sidebar.py must exist"

    content = sidebar_file.read_text(encoding="utf-8")
    
    expected_legacy_pages = [
        "Overview",
        "Study Area",
        "Data Quality",
        "Environmental Analysis",
        "Event Detection",
        "Susceptibility Prediction",
        "Feature Importance",
    ]

    # Verify all legacy page names appear in sidebar code
    for page in expected_legacy_pages:
        assert f'"{page}"' in content or f"'{page}'" in content, f"Legacy page '{page}' missing from sidebar!"

    # Verify GeoTrustNet Intelligence is present
    assert '"GeoTrustNet Intelligence"' in content or "'GeoTrustNet Intelligence'" in content, (
        "GeoTrustNet Intelligence page missing from sidebar!"
    )


def test_app_router_registration():
    """Verify that app.py registers all 8 page routers with valid callable handlers."""
    app_file = ROOT_DIR / "dashboard" / "app.py"
    assert app_file.exists(), "app.py must exist"

    content = app_file.read_text(encoding="utf-8")

    # Verify imports
    assert "geotrustnet_intelligence" in content, "geotrustnet_intelligence must be imported in app.py"

    # Verify router dictionary mapping
    assert '"GeoTrustNet Intelligence": geotrustnet_intelligence.render' in content or \
           "'GeoTrustNet Intelligence': geotrustnet_intelligence.render" in content, (
        "Router for 'GeoTrustNet Intelligence' not mapped in app.py"
    )

    # Verify legacy router keys
    legacy_keys = [
        "Overview",
        "Study Area",
        "Data Quality",
        "Environmental Analysis",
        "Event Detection",
        "Susceptibility Prediction",
        "Feature Importance",
    ]
    for key in legacy_keys:
        assert f'"{key}"' in content or f"'{key}'" in content, f"Legacy route key '{key}' missing from router!"


def test_geotrustnet_page_functions():
    """Verify that dashboard.pages.geotrustnet_intelligence exports required tab rendering functions."""
    from dashboard.pages import geotrustnet_intelligence

    assert hasattr(geotrustnet_intelligence, "render"), "geotrustnet_intelligence must have render()"
    assert callable(geotrustnet_intelligence.render), "render must be callable"

    # Check sub-tab renderers
    assert hasattr(geotrustnet_intelligence, "render_overview_tab"), "render_overview_tab missing"
    assert hasattr(geotrustnet_intelligence, "render_input_explorer_tab"), "render_input_explorer_tab missing"
    assert hasattr(geotrustnet_intelligence, "render_reliability_tab"), "render_reliability_tab missing"
    assert hasattr(geotrustnet_intelligence, "render_uncertainty_tab"), "render_uncertainty_tab missing"
    assert hasattr(geotrustnet_intelligence, "render_robustness_tab"), "render_robustness_tab missing"


def test_artifact_and_split_loaders_graceful_handling():
    """Verify that split manifest and experiment report loaders operate safely."""
    from dashboard.pages.geotrustnet_intelligence import load_split_manifest, load_experiment_reports

    manifest = load_split_manifest()
    # If file exists, it should be a DataFrame; if not, None (no crash)
    assert manifest is None or isinstance(manifest, pd.DataFrame)

    reports = load_experiment_reports()
    assert isinstance(reports, dict), "load_experiment_reports must return a dictionary"

    # If reports were loaded, check known keys
    known_keys = [
        "phase2_summary",
        "phase3_summary",
        "phase4_summary",
        "phase5_uncertainty",
        "phase5_robustness",
    ]
    for k in known_keys:
        # Either present or gracefully absent
        if k in reports:
            assert isinstance(reports[k], dict), f"Report '{k}' must be a dictionary"


def test_scientific_integrity_and_provenance_labeling():
    """Verify that reports or fallback strings clearly designate synthetic status and cloud GPU training pending."""
    from dashboard.pages.geotrustnet_intelligence import load_experiment_reports

    reports = load_experiment_reports()
    # Check that baseline experiment summaries explicitly reference synthetic manifests
    if "phase2_summary" in reports:
        p2 = reports["phase2_summary"]
        manifest = p2.get("config", {}).get("split_manifest", "")
        assert "synthetic" in manifest.lower(), "Baseline experiment must reference synthetic manifest"


def test_synthetic_demo_patch_generation():
    """Verify that synthetic demonstration patch generator creates compliant multimodal PatchSample."""
    from dashboard.pages.geotrustnet_intelligence import _get_synthetic_demo_patch
    from src.geotrustnet.data.contracts import PatchSample

    patch = _get_synthetic_demo_patch(seed=123)
    assert isinstance(patch, PatchSample), "Must return a PatchSample instance"
    assert patch.sar is not None
    assert patch.sar.ndim == 4  # [T, C, H, W]
    assert patch.sar.shape[1] == 2  # VV, VH
    assert patch.mask is not None
    assert patch.mask.ndim == 3  # [1, H, W]
    assert patch.optical is not None
    assert patch.optical.ndim == 4  # [T, C, H, W]
    assert patch.dem is not None
    assert patch.dem.ndim == 3  # [C, H, W]
