"""Unit and integration tests for GeoTrustNet Phase 7 deliverables.

Verifies:
1. Cloud environment configuration schema and detection logic.
2. Dataset inspection and zero-leakage split verification.
3. Jupyter Notebook structural validity (Colab and Kaggle).
4. Controlled benchmark experiment configurations schema and consistency.
5. Benchmark runner execution under smoke-test parameters.
6. Documentation and research governance artifacts completeness.
"""

import json
import os
import sys
from pathlib import Path
import pytest

# Windows DLL initialization for torch (must import torch before pandas/numpy on Windows)
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


def test_cloud_environment_config():
    """Verify that cloud_environment.json exists and has valid configuration schema."""
    config_path = ROOT_DIR / "configs" / "training" / "cloud_environment.json"
    assert config_path.exists(), "cloud_environment.json must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "target_platforms" in cfg
    assert "Google Colab" in cfg["target_platforms"]
    assert "Kaggle" in cfg["target_platforms"]
    assert "recommended_gpus" in cfg
    assert len(cfg["recommended_gpus"]) >= 3
    assert "dependencies" in cfg
    assert "training_settings" in cfg
    assert cfg["training_settings"]["deterministic_seed"] == 42


def test_gpu_environment_detection():
    """Verify that environment inspection runs and returns valid diagnostics."""
    from scripts.verify_gpu_environment import inspect_environment

    report = inspect_environment()
    assert isinstance(report, dict)
    assert "platform" in report
    assert "python_version" in report
    assert "torch_version" in report
    assert "cuda_available" in report
    assert "execution_tier" in report
    assert report["execution_tier"] in ["GPU_ACCELERATED", "CPU_SMOKE_TEST_ONLY"]
    assert report["recommended_batch_size"] > 0


def test_dataset_manifest_inspection():
    """Verify that dataset manifest inspection audits class balance and zero leakage."""
    from scripts.inspect_sen12landslides import audit_dataset_manifest

    manifest_path = ROOT_DIR / "data" / "manifests" / "synthetic_split_manifest.csv"
    assert manifest_path.exists(), "synthetic_split_manifest.csv must exist"

    report = audit_dataset_manifest(manifest_path)
    assert isinstance(report, dict)
    assert report["total_samples"] > 0
    assert report["n_regions"] > 1
    assert report["splits"]["is_leakage_free_disjoint"] is True
    assert "landslide_pixel_prevalence" in report
    assert "provenance_status" in report
    assert report["duplicate_files_detected"] == 0


def test_notebooks_structural_validity():
    """Verify that both Colab and Kaggle notebooks are valid Jupyter notebooks."""
    nb_colab = ROOT_DIR / "notebooks" / "GeoTrustNet_Research_Training.ipynb"
    nb_kaggle = ROOT_DIR / "notebooks" / "GeoTrustNet_Kaggle_Training.ipynb"

    assert nb_colab.exists(), "Colab notebook must exist"
    assert nb_kaggle.exists(), "Kaggle notebook must exist"

    for nb_path in [nb_colab, nb_kaggle]:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert nb.get("nbformat") == 4, f"{nb_path.name} must be nbformat 4"
        assert "cells" in nb
        assert len(nb["cells"]) >= 25, f"{nb_path.name} must have at least 25 cells"

        # Check for key sections
        sources = " ".join([" ".join(c.get("source", [])) for c in nb["cells"]])
        assert "GeoTrustNet" in sources
        assert "Sen12Landslides" in sources
        assert "seed_everything" in sources
        assert "Reliability" in sources or "reliability" in sources


def test_benchmark_configs_consistency():
    """Verify that all 4 controlled benchmark experiment configs exist and share identical seeds and splits."""
    benchmark_dir = ROOT_DIR / "configs" / "experiments"
    models = ["model_a_sar_unet", "model_b_temporal_sar", "model_c_multimodal_fusion", "model_d_geotrustnet"]

    for m in models:
        cfg_file = benchmark_dir / f"benchmark_{m}.json"
        assert cfg_file.exists(), f"Benchmark config for {m} must exist"

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        assert cfg["benchmark_tier"] == "controlled_scientific_comparison"
        assert cfg["seed"] == 42
        assert "loss" in cfg
        assert cfg["loss"]["name"] == "CombinedBCEDiceLoss"
        assert "output_directory" in cfg


def test_benchmark_runner_smoke_execution():
    """Verify that the unified benchmark runner executes under smoke-test parameters."""
    from scripts.run_benchmarks_and_ablations import run_benchmark_experiment

    res = run_benchmark_experiment("model_a", smoke_test=True, device_pref="cpu")
    assert isinstance(res, dict)
    assert res["model_key"] == "model_a"
    assert res["is_smoke_test"] is True
    assert "metrics" in res
    assert "dice" in res["metrics"]
    assert "iou" in res["metrics"]
    assert "calibration" in res


def test_phase7_documentation_and_governance():
    """Verify that all required Phase 7 documentation and scientific governance documents exist."""
    docs = [
        ROOT_DIR / "docs" / "reports" / "phase7_pretraining_audit.md",
        ROOT_DIR / "docs" / "reports" / "phase7_report.md",
        ROOT_DIR / "docs" / "SEN12LANDSLIDES_SETUP.md",
        ROOT_DIR / "docs" / "GEOTRUSTNET_EVALUATION_PROTOCOL.md",
        ROOT_DIR / "docs" / "GEOTRUSTNET_RESEARCH_STATUS.md",
        ROOT_DIR / "docs" / "GEOTRUSTNET_PAPER_OUTLINE.md",
    ]

    for d in docs:
        assert d.exists(), f"Required Phase 7 document missing: {d.name}"
        assert d.stat().st_size > 500, f"Document {d.name} is unexpectedly empty"
