"""Unit and integration tests for GeoTrustNet Phase 8 deliverables.

Verifies:
1. Real Sen12Landslides dataset audit completeness (4,392 aligned samples, 11 inventories).
2. Official IID and strict disjoint inventory split manifests integrity.
3. Model A Pilot training execution artifacts and checkpoint verification.
4. Pre-flight audit and real-data experimental reports completeness.
5. Dashboard report loader integration with Phase 8 real-data audit and pilot artifacts.
"""

import json
import os
import sys
from pathlib import Path
import pytest
import pandas as pd

# Windows DLL initialization for torch
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_real_data_audit_integrity():
    """Verify that sen12landslides_real_data_audit.json contains verified real dataset statistics."""
    audit_file = ROOT_DIR / "docs" / "reports" / "sen12landslides_real_data_audit.json"
    assert audit_file.exists(), "sen12landslides_real_data_audit.json must exist"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_aligned_samples"] == 4392, "Must contain exactly 4,392 aligned samples"
    assert data["inventory_count"] == 11, "Must contain 11 geographic inventories"
    assert "overall_pixel_prevalence" in data
    assert 2.0 <= data["overall_pixel_prevalence"]["mean_landslide_pixel_prevalence_pct"] <= 5.0
    assert "inventory_statistics" in data
    assert len(data["inventory_statistics"]) == 11


def test_manifest_files_validity():
    """Verify that official IID and disjoint inventory manifests exist and have correct schemas."""
    iid_csv = ROOT_DIR / "data" / "manifests" / "sen12landslides_iid_manifest.csv"
    disjoint_csv = ROOT_DIR / "data" / "manifests" / "sen12landslides_disjoint_manifest.csv"

    assert iid_csv.exists(), "sen12landslides_iid_manifest.csv must exist"
    assert disjoint_csv.exists(), "sen12landslides_disjoint_manifest.csv must exist"

    df_iid = pd.read_csv(iid_csv)
    df_disjoint = pd.read_csv(disjoint_csv)

    assert len(df_iid) == 4392
    assert len(df_disjoint) == 4392

    for col in ["sample_id", "region_id", "split", "pixel_annotated", "landslide_pixel_ratio"]:
        assert col in df_iid.columns
        assert col in df_disjoint.columns

    # Verify disjointness in disjoint manifest
    train_invs = set(df_disjoint[df_disjoint["split"] == "train"]["region_id"])
    test_invs = set(df_disjoint[df_disjoint["split"] == "test"]["region_id"])
    assert len(train_invs.intersection(test_invs)) == 0, "Disjoint manifest must have 0 region overlap between train and test"


def test_pilot_training_artifacts():
    """Verify that Model A pilot training produced valid checkpoint and summary."""
    pilot_dir = ROOT_DIR / "artifacts" / "geotrustnet" / "experiments" / "pilot_model_a_sar_unet"
    assert pilot_dir.exists(), "Pilot directory must exist"

    summary_file = pilot_dir / "pilot_summary.json"
    checkpoint_file = pilot_dir / "best_checkpoint.pt"
    history_file = pilot_dir / "training_history.csv"

    assert summary_file.exists(), "pilot_summary.json must exist"
    assert checkpoint_file.exists(), "best_checkpoint.pt must exist"
    assert history_file.exists(), "training_history.csv must exist"

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["experiment_id"] == "pilot_model_a_sar_unet"
    assert summary["checkpoint_integrity_verified"] is True
    assert "hardware" in summary
    assert "test_metrics" in summary
    assert "dice" in summary["test_metrics"]
    assert "iou" in summary["test_metrics"]
    assert "calibration" in summary


def test_dashboard_report_loader_integration():
    """Verify that dashboard loads Phase 8 pilot and real-data audit reports cleanly."""
    from dashboard.pages.geotrustnet_intelligence import load_experiment_reports

    reports = load_experiment_reports()
    assert isinstance(reports, dict)
    assert "phase8_pilot" in reports, "load_experiment_reports must include 'phase8_pilot'"
    assert "real_data_audit" in reports, "load_experiment_reports must include 'real_data_audit'"


def test_phase8_documentation_files():
    """Verify that Phase 8 audit and completion reports exist and have substantial content."""
    preflight = ROOT_DIR / "docs" / "reports" / "phase8_preflight_audit.md"
    experiments = ROOT_DIR / "docs" / "reports" / "phase8_real_data_experiments.md"

    assert preflight.exists()
    assert experiments.exists()
    assert preflight.stat().st_size > 1000
    assert experiments.stat().st_size > 1000
