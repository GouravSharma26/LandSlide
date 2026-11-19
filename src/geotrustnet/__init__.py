"""GeoTrustNet: Uncertainty-Aware Spatio-Temporal Multimodal Deep Learning for Reliable Landslide Detection.

Research model package integrating with the LandSlide platform.
"""

import os
import sys
from pathlib import Path

# On Windows, resolve torch DLL directory search path
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

__version__ = "0.5.0-phase5"
__all__ = ["data", "models", "training", "evaluation"]
