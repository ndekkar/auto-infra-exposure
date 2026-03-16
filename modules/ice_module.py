"""
Ice hazard analysis module.

This module processes ERA5-derived ice hazard indicators (NetCDF) and produces:
- Per-indicator GeoTIFF rasters (mean over all years in the file)
- Stats CSV and histogram plots per indicator
- Binary and 4-class classified rasters based on thresholds

The input NetCDF is a single file produced by the ERA5 ice indicator pipeline,
containing the four indicators along a year dimension:
  - freeze_days        (FD)  : days/year where t2m < 0°C
  - freeze_thaw_cycles (FTC) : days/year where t2m crosses 0°C within the same day
  - freezing_rain_days (FRD) : days/year where tp > 0 and t2m < 0°C simultaneously
  - cold_spell_duration(CSD) : maximum consecutive freeze days per year

Config expectations (YAML)
--------------------------
hazards:
  ice:
    active: true
    input: "../data/era5/ice/indicators/era5_ice_indicators_TJK.nc"
    thresholds:
      freeze_days: 60
      freeze_thaw_cycles: 20
      freezing_rain_days: 10
      cold_spell_duration: 14
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd


# ---------------------------
# Constants
# ---------------------------

INDICATORS = {
    "freeze_days":         {"long_name": "Freeze Days",             "units": "days/year"},
    "freeze_thaw_cycles":  {"long_name": "Freeze-Thaw Cycles",      "units": "days/year"},
    "freezing_rain_days":  {"long_name": "Freezing Rain Days",      "units": "days/year"},
    "cold_spell_duration": {"long_name": "Cold Spell Duration",     "units": "days"},
}

DEFAULT_THRESHOLDS = {
    "freeze_days":         60,
    "freeze_thaw_cycles":  20,
    "freezing_rain_days":  10,
    "cold_spell_duration": 14,
}


# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _array_valid(da: xr.DataArray) -> np.ndarray:
    """Return flattened array of valid (non-NaN) values."""
    arr = da.values.astype(np.float64).reshape(-1)
    return arr[~np.isnan(arr)]


def _da_to_geotiff(da: xr.DataArray, out_path: Path) -> None:
    """
    Write a 2D DataArray (latitude/longitude) to a GeoTIFF using rioxarray.
    """
    da_out = da.copy().astype("float32")
    da_out = da_out.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    da_out = da_out.rio.write_crs("EPSG:4326")
    da_out.rio.to_raster(str(out_path))


# ---------------------------
# Per-indicator processing
# ---------------------------

def _process_indicator(
    indicator: str,
    da: xr.DataArray,
    threshold: float,
    output_dir: Path,
    aoi_path: Optional[str],
) -> Dict[str, str]:
    """
    For a single indicator DataArray (2D, already averaged over years):
    - Optionally clip to AOI
    - Export mean GeoTIFF
    - Compute stats CSV
    - Plot histogram
    - Export binary and 4-class rasters
    Returns dict of output paths.
    """
    long_name = INDICATORS[indicator]["long_name"]

    # Clip to AOI if provided
    if aoi_path:
        try:
            aoi = gpd.read_file(aoi_path)
            da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            da = da.rio.write_crs("EPSG:4326")
            if aoi.crs and aoi.crs.to_epsg() != 4326:
                aoi = aoi.to_crs("EPSG:4326")
            da = da.rio.clip(aoi.geometry, aoi.crs, drop=True)
        except Exception as e:
            print(f"  [WARN] Could not clip {indicator} to AOI: {e}")

    # Export mean GeoTIFF
    tif_path = output_dir / f"ice_{indicator}_mean.tif"
    _da_to_geotiff(da, tif_path)

    # Stats
    arr = _array_valid(da)
    if arr.size == 0:
        print(f"  [WARN] No valid pixels for {indicator}")
        return {}

    stats = {
        "indicator":           indicator,
        "min":                 float(np.min(arr)),
        "p05":                 float(np.percentile(arr, 5)),
        "p25":                 float(np.percentile(arr, 25)),
        "median":              float(np.percentile(arr, 50)),
        "mean":                float(np.mean(arr)),
        "p75":                 float(np.percentile(arr, 75)),
        "p95":                 float(np.percentile(arr, 95)),
        "max":                 float(np.max(arr)),
        "count":               int(arr.size),
        "threshold":           float(threshold),
        "pct_above_threshold": float(100.0 * np.mean(arr >= threshold)),
    }

    stats_csv = output_dir / f"ice_{indicator}_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)

    # Histogram
    hist_png = output_dir / f"ice_{indicator}_hist.png"
    plt.figure()
    plt.hist(arr, bins=40, color="steelblue", edgecolor="white")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
    plt.title(f"{long_name} distribution")
    plt.xlabel(INDICATORS[indicator]["units"])
    plt.ylabel("Pixel count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_png, dpi=200)
    plt.close()

    # Binary raster: 1 where indicator >= threshold, 0 elsewhere
    nodata_mask = da.isnull()
    bin_da = xr.where(da >= threshold, 1, 0)
    bin_da = xr.where(nodata_mask, 255, bin_da).astype(np.uint8)
    bin_da = bin_da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    bin_da = bin_da.rio.write_crs("EPSG:4326")
    bin_da = bin_da.rio.write_nodata(255, encoded=True)
    bin_tif = output_dir / f"ice_{indicator}_binary.tif"
    bin_da.rio.to_raster(str(bin_tif))

    # 4-class raster
    # Class 1 = Low     : < threshold * 0.5
    # Class 2 = Medium  : threshold * 0.5 to threshold
    # Class 3 = High    : threshold to threshold * 1.5
    # Class 4 = Extreme : >= threshold * 1.5
    t_low  = threshold * 0.5
    t_high = threshold * 1.5
    class_da = xr.zeros_like(da, dtype=np.uint8)
    class_da = xr.where(da < t_low,                         1, class_da)
    class_da = xr.where((da >= t_low)  & (da < threshold),  2, class_da)
    class_da = xr.where((da >= threshold) & (da < t_high),  3, class_da)
    class_da = xr.where(da >= t_high,                       4, class_da)
    class_da = xr.where(nodata_mask, 255, class_da).astype(np.uint8)
    class_da = class_da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    class_da = class_da.rio.write_crs("EPSG:4326")
    class_da = class_da.rio.write_nodata(255, encoded=True)
    class_tif = output_dir / f"ice_{indicator}_class4.tif"
    class_da.rio.to_raster(str(class_tif))

    print(f"  {indicator}: mean={stats['mean']:.1f}, pct_above_threshold={stats['pct_above_threshold']:.1f}%")

    return {
        f"{indicator}_tif":        str(tif_path),
        f"{indicator}_stats_csv":  str(stats_csv),
        f"{indicator}_hist_png":   str(hist_png),
        f"{indicator}_binary_tif": str(bin_tif),
        f"{indicator}_class4_tif": str(class_tif),
    }


# ---------------------------
# Main entry point
# ---------------------------

def process_ice(config: dict) -> dict:
    """
    Orchestrates ice hazard processing based on config.
    Returns dict of output paths.

    Reads a single NetCDF file specified in config (hazards.ice.input),
    computes the mean over the year dimension for each of the 4 indicators,
    exports GeoTIFFs, stats and classified rasters.

    The primary raster (freeze_days mean GeoTIFF) is returned as 'primary_raster'
    for use by compute_infra_stats_from_overlay in main.py.
    """
    ice_conf   = (config.get("hazards", {}).get("ice", {}) or {})
    nc_path    = ice_conf.get("input")
    thresholds = ice_conf.get("thresholds", {}) or {}
    aoi_path   = config.get("aoi")

    if not nc_path:
        raise ValueError("Ice hazard 'input' is missing in config (hazards.ice.input).")

    if not Path(nc_path).exists():
        raise FileNotFoundError(f"Ice hazard input file not found: {nc_path}")

    output_dir = Path(config.get("output_dir", "output")) / "ice"
    _ensure_dir(output_dir)

    print(f"Loading ice indicators from: {nc_path}")
    ds = xr.open_dataset(nc_path, decode_timedelta=False)

    outputs = {}

    for indicator in INDICATORS:
        if indicator not in ds:
            print(f"  [SKIP] Indicator '{indicator}' not found in dataset")
            continue

        threshold = float(thresholds.get(indicator, DEFAULT_THRESHOLDS[indicator]))
        print(f"\nProcessing ice indicator: {indicator} (threshold={threshold})")

        # Average over year dimension to get a single 2D map
        da = ds[indicator].mean(dim="year").astype("float32")
        da.name = indicator
        da.attrs["long_name"] = INDICATORS[indicator]["long_name"]
        da.attrs["units"]     = INDICATORS[indicator]["units"]

        result = _process_indicator(
            indicator=indicator,
            da=da,
            threshold=threshold,
            output_dir=output_dir,
            aoi_path=aoi_path,
        )
        outputs.update(result)

    ds.close()

    # Primary raster = freeze_days mean GeoTIFF
    # passed to compute_infra_stats_from_overlay in main.py
    outputs["primary_raster"] = str(output_dir / "ice_freeze_days_mean.tif")

    print(f"\nIce module complete. Outputs in: {output_dir}")
    return outputs