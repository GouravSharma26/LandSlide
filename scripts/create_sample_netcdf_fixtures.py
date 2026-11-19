"""Generate miniature valid NetCDF fixtures for Sen12Landslides testing.

Produces co-registered NetCDF (.nc) files with authentic band structures:
  - sample_sar.nc: VV, VH, elevation, slope, MASK (including sentinel nodata=255)
  - sample_optical.nc: B02-B12 spectral bands, SCL cloud mask
"""

from pathlib import Path
import numpy as np
import xarray as xr


def generate_sample_netcdf(
    output_dir: Path,
    height: int = 128,
    width: int = 128,
    time_steps: int = 5,
    seed: int = 42,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. SAR Dataset
    # VV: mean ~ -9.3, std ~ 3.5 dB
    vv = rng.normal(-9.31, 3.49, size=(time_steps, height, width)).astype(np.float32)
    # VH: mean ~ -15.89, std ~ 4.18 dB
    vh = rng.normal(-15.89, 4.18, size=(time_steps, height, width)).astype(np.float32)
    # Terrain
    elev = rng.uniform(200.0, 2500.0, size=(height, width)).astype(np.float32)
    slope = rng.uniform(5.0, 55.0, size=(height, width)).astype(np.float32)

    # Mask: 0 (stable, 92%), 1 (landslide, 5%), 255 (nodata, 3%)
    mask = rng.choice([0, 1, 255], size=(height, width), p=[0.92, 0.05, 0.03]).astype(np.uint8)

    # Inject a few nodata (-9999.0) in SAR to test validity masking
    vv[0, :5, :5] = -9999.0

    sar_ds = xr.Dataset(
        data_vars={
            "VV": (("time", "y", "x"), vv),
            "VH": (("time", "y", "x"), vh),
            "elevation": (("y", "x"), elev),
            "slope": (("y", "x"), slope),
            "MASK": (("y", "x"), mask),
        },
        coords={
            "time": np.arange(time_steps),
            "y": np.arange(height),
            "x": np.arange(width),
        },
        attrs={"title": "Synthetic Sen12Landslides SAR Fixture for Testing"},
    )
    sar_path = output_dir / "sample_sar.nc"
    sar_ds.to_netcdf(sar_path)

    # 2. Optical Dataset
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    opt_vars = {}
    for b in bands:
        opt_vars[b] = (
            ("time", "y", "x"),
            rng.uniform(500.0, 3000.0, size=(time_steps, height, width)).astype(np.float32),
        )
    # SCL: 4 (vegetation), 9 (high cloud prob), 0 (nodata)
    scl = rng.choice([4, 9, 0], size=(time_steps, height, width), p=[0.85, 0.10, 0.05]).astype(np.uint8)
    opt_vars["SCL"] = (("time", "y", "x"), scl)

    opt_ds = xr.Dataset(
        data_vars=opt_vars,
        coords={
            "time": np.arange(time_steps),
            "y": np.arange(height),
            "x": np.arange(width),
        },
        attrs={"title": "Synthetic Sen12Landslides Optical Fixture for Testing"},
    )
    opt_path = output_dir / "sample_optical.nc"
    opt_ds.to_netcdf(opt_path)

    print(f"Generated NetCDF fixtures in {output_dir}:")
    print(f"  - {sar_path.name}")
    print(f"  - {opt_path.name}")
    return sar_path


if __name__ == "__main__":
    generate_sample_netcdf(Path("data/synthetic_netcdf"))
