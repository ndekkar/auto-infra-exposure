"""
Cold hazard analysis module.

This module processes a cold hazard raster (TNn mean) and produces:
- Clipped raster stats
- Histogram and CDF plots
- Binary and 4-class classified rasters based on a threshold
- (Optional) Frost Days (FD) raster summary if provided

Config expectations (YAML)
-------------------------
hazards:
  cold:
    active: true
    input: "../data/hazards/cold/TNn_mean_2010_2024.tif"
    threshold: -20
    fd_input: null  # or a path like "../data/hazards/cold/FD_mean_2010_2024_TJK.tif"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
import rioxarray  # noqa: F401
import geopandas as gpd


# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _open_raster_clip(tif_path: str, aoi_path: Optional[str] = None) -> xr.DataArray:
    """
    Open raster with rioxarray and optionally clip to AOI polygon.
    """
    da = rioxarray.open_rasterio(tif_path, masked=True).squeeze(drop=True)

    if aoi_path:
        aoi = gpd.read_file(aoi_path)
        # reproject AOI to raster CRS if needed
        if da.rio.crs and aoi.crs and aoi.crs != da.rio.crs:
            aoi = aoi.to_crs(da.rio.crs)

        da = da.rio.clip(aoi.geometry, aoi.crs, drop=True)

    return da


def _nodata_value(da: xr.DataArray) -> Optional[float]:
    """
    Try to retrieve nodata from raster metadata.
    """
    try:
        nd = da.rio.nodata
        return nd
    except Exception:
        return None


def _array_valid(da: xr.DataArray) -> np.ndarray:
    """
    Return flattened array of valid values (nodata/NaN removed).
    """
    arr = da.values.astype(np.float64)
    nd = _nodata_value(da)

    arr = arr.reshape(-1)

    # Remove NaNs
    arr = arr[~np.isnan(arr)]

    # Remove explicit nodata if present
    if nd is not None and not np.isnan(nd):
        arr = arr[arr != nd]

    return arr


# ---------------------------
# Main entry
# ---------------------------

def process_cold(config: dict) -> dict:
    """
    Orchestrates cold hazard processing based on config.
    Returns dict of outputs (paths).
    """
    cold_conf = (config.get("hazards", {}).get("cold", {}) or {})
    tif_path = cold_conf.get("input")
    threshold = cold_conf.get("threshold", -20)
    aoi_path = config.get("aoi_path")

    if not tif_path:
        raise ValueError("Cold hazard 'input' is missing in config (hazards.cold.input).")

    output_dir = Path(config.get("output_dir", "output")) / "cold"
    _ensure_dir(output_dir)

    # 1) Summarize TNn raster (stats + hist + CDF)
    stats_csv, hist_png, cdf_png = summarize_cold_tnn(
        tif_path=tif_path,
        output_dir=output_dir,
        aoi_path=aoi_path,
        threshold=threshold
    )

    # 2) Export classified rasters (binary at threshold + 4-class scheme)
    bin_tif, class_tif = export_cold_class_rasters(
        tif_path=tif_path,
        output_dir=output_dir,
        aoi_path=aoi_path,
        threshold=threshold
    )

    # 3) (Optional) Summarize Frost Days (FD) raster if provided in config (YAML can be null)
    fd_input = cold_conf.get("fd_input")  # expected to be a path or None (YAML: null)

    fd_stats_csv = None
    fd_hist_png = None
    if fd_input:
        fd_stats_csv, fd_hist_png = summarize_frost_days(
            fd_tif_path=fd_input,
            output_dir=output_dir,
            aoi_path=aoi_path
        )

    return {
        "cold_stats_csv": stats_csv,
        "cold_hist_png": hist_png,
        "cold_cdf_png": cdf_png,
        "cold_binary_tif": bin_tif,
        "cold_class4_tif": class_tif,
        "fd_stats_csv": fd_stats_csv,
        "fd_hist_png": fd_hist_png
    }


# ---------------------------
# TNn summary + plots
# ---------------------------

def summarize_cold_tnn(
    tif_path: str,
    output_dir: Path,
    aoi_path: Optional[str],
    threshold: float
) -> Tuple[str, str, str]:
    """
    Summarize cold TNn raster:
    - Basic stats
    - Histogram with threshold line
    - CDF plot
    - CSV summary
    """
    da = _open_raster_clip(tif_path, aoi_path=aoi_path)
    arr = _array_valid(da)

    if arr.size == 0:
        raise ValueError("No valid pixels found in TNn raster after clipping/masking.")

    stats = {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
        "threshold": float(threshold),
        "pct_below_threshold": float(100.0 * np.mean(arr <= threshold)),
    }

    stats_csv = output_dir / "cold_tnn_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)

    # Histogram
    hist_png = output_dir / "cold_tnn_hist.png"
    plt.figure()
    plt.hist(arr, bins=40)
    plt.axvline(threshold)
    plt.title("TNn distribution")
    plt.xlabel("TNn")
    plt.ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(hist_png, dpi=200)
    plt.close()

    # CDF
    cdf_png = output_dir / "cold_tnn_cdf.png"
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    plt.figure()
    plt.plot(x, y)
    plt.axvline(threshold)
    plt.title("TNn CDF")
    plt.xlabel("TNn")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(cdf_png, dpi=200)
    plt.close()

    return str(stats_csv), str(hist_png), str(cdf_png)


# ---------------------------
# Export classified rasters
# ---------------------------

def export_cold_class_rasters(
    tif_path: str,
    output_dir: Path,
    aoi_path: Optional[str],
    threshold: float
) -> Tuple[str, str]:
    """
    Export:
    - Binary raster: 1 where TNn <= threshold, else 0
    - 4-class raster based on fixed bins around threshold

    Notes
    -----
    - Preserves NoData from the input raster by writing 255 for uint8 outputs.
    """
    da = _open_raster_clip(tif_path, aoi_path=aoi_path)

    # Preserve nodata / NaNs from input
    nodata_mask = da.isnull()

    # --- Binary (uint8; nodata=255)
    bin_da = xr.where(da <= threshold, 1, 0)
    bin_da = xr.where(nodata_mask, 255, bin_da).astype(np.uint8)
    bin_da.rio.write_crs(da.rio.crs, inplace=True)
    bin_da.rio.write_nodata(255, encoded=True, inplace=True)

    bin_tif = output_dir / "cold_tnn_binary.tif"
    bin_da.rio.to_raster(bin_tif)

    # --- 4-class scheme (uint8; nodata=255)
    # Classes:
    # 1 = Mild (warmest)      : TNn > threshold + 10
    # 2 = Cold               : threshold + 10 >= TNn > threshold + 5
    # 3 = Very cold          : threshold + 5  >= TNn > threshold
    # 4 = Extreme (<= thr)   : TNn <= threshold
    class_da = xr.zeros_like(da, dtype=np.uint8)
    class_da = xr.where(da > (threshold + 10), 1, class_da)
    class_da = xr.where((da <= (threshold + 10)) & (da > (threshold + 5)), 2, class_da)
    class_da = xr.where((da <= (threshold + 5)) & (da > threshold), 3, class_da)
    class_da = xr.where(da <= threshold, 4, class_da)
    class_da = xr.where(nodata_mask, 255, class_da).astype(np.uint8)

    class_da.rio.write_crs(da.rio.crs, inplace=True)
    class_da.rio.write_nodata(255, encoded=True, inplace=True)

    class_tif = output_dir / "cold_tnn_class4.tif"
    class_da.rio.to_raster(class_tif)

    return str(bin_tif), str(class_tif)


# ---------------------------
# FD (Frost Days) summary + plot
# ---------------------------

def summarize_frost_days(
    fd_tif_path: str,
    output_dir: Path,
    aoi_path: Optional[str]
) -> Tuple[str, str]:
    """
    Summarize Frost Days (FD) raster:
    - Basic stats
    - Histogram plot
    """
    da = _open_raster_clip(fd_tif_path, aoi_path=aoi_path)
    arr = _array_valid(da)

    if arr.size == 0:
        raise ValueError("No valid pixels found in FD raster after clipping/masking.")

    stats = {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }

    stats_csv = output_dir / "cold_fd_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)

    hist_png = output_dir / "cold_fd_hist.png"
    plt.figure()
    plt.hist(arr, bins=40)
    plt.title("Frost Days distribution")
    plt.xlabel("FD")
    plt.ylabel("Pixel count")
    plt.tight_layout()
    plt.savefig(hist_png, dpi=200)
    plt.close()

    return str(stats_csv), str(hist_png)
