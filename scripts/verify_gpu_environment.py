"""GPU and Cloud Execution Environment Verification Utility.

Checks Python version, PyTorch installation, CUDA availability, GPU memory,
mixed-precision support, and core dependencies. Supports local workstations,
Google Colab, and Kaggle.
"""

import json
import os
import platform
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Windows DLL registration for PyTorch if needed
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass


def inspect_environment() -> dict:
    """Inspects the local or cloud runtime environment and returns a diagnostic report."""
    report = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "is_colab": "google.colab" in sys.modules or os.path.exists("/content"),
        "is_kaggle": os.path.exists("/kaggle"),
    }

    # PyTorch inspection
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "device_index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "multi_processor_count": props.multi_processor_count,
                    "major_capability": props.major,
                    "minor_capability": props.minor,
                })
            report["cuda_version"] = torch.version.cuda
            report["devices"] = devices
            report["current_device"] = torch.cuda.current_device()

            # Test mixed-precision support
            try:
                x = torch.randn(2, 64, 16, 16, device="cuda")
                conv = torch.nn.Conv2d(64, 64, 3, padding=1).to("cuda")
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    y = conv(x)
                report["amp_float16_supported"] = True
            except Exception as e:
                report["amp_float16_supported"] = False
                report["amp_error"] = str(e)
        else:
            report["cuda_version"] = None
            report["devices"] = []
            report["current_device"] = "cpu"
            report["amp_float16_supported"] = False

    except ImportError as e:
        report["torch_version"] = None
        report["cuda_available"] = False
        report["torch_import_error"] = str(e)

    # Core dependencies verification
    deps = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "rasterio"]:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[pkg] = "not_installed"
    report["dependencies"] = deps

    # Recommended batch size based on available compute
    if report.get("cuda_available") and report.get("devices"):
        vram = report["devices"][0]["total_memory_gb"]
        if vram >= 32.0:
            rec_batch = 16
        elif vram >= 14.0:
            rec_batch = 8
        else:
            rec_batch = 4
        report["recommended_batch_size"] = rec_batch
        report["execution_tier"] = "GPU_ACCELERATED"
    else:
        report["recommended_batch_size"] = 2
        report["execution_tier"] = "CPU_SMOKE_TEST_ONLY"

    return report


def main():
    report = inspect_environment()
    print("=" * 60)
    print("  GeoTrustNet Cloud & GPU Execution Environment Audit")
    print("=" * 60)
    print(f"Platform:            {report['platform']}")
    print(f"Python Version:      {report['python_version']}")
    print(f"Google Colab:        {report['is_colab']}")
    print(f"Kaggle:              {report['is_kaggle']}")
    print(f"PyTorch Version:     {report.get('torch_version')}")
    print(f"CUDA Available:      {report['cuda_available']}")

    if report["cuda_available"]:
        for dev in report["devices"]:
            print(f"  GPU {dev['device_index']}: {dev['name']} ({dev['total_memory_gb']} GB VRAM)")
        print(f"AMP (FP16) Support:  {report['amp_float16_supported']}")
    else:
        print("  Notice: No CUDA GPU detected. Running in CPU smoke-test tier.")

    print(f"Execution Tier:      {report['execution_tier']}")
    print(f"Rec. Batch Size:     {report['recommended_batch_size']}")
    print("\nInstalled Dependencies:")
    for pkg, ver in report["dependencies"].items():
        print(f"  - {pkg:<12}: {ver}")
    print("=" * 60)

    # Save environment snapshot to config
    out_dir = ROOT_DIR / "configs" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "detected_environment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved environment profile to: {out_path}")


if __name__ == "__main__":
    main()
