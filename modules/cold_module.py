"""
Cold hazard analysis module.

This module processes a cold-hazard raster input (TNn_mean in °C) to compute:
- Area statistics (min/mean/max, quantiles)
- Share of area colder than the YAML threshold (e.g., -20°C)
- Classified rasters: binary (<= threshold) and discrete classes (Mild/Cold/Very/Extreme)
- Simple figures: histogram + CDF with the threshold line
- CSV summary

Notes
-----
- Input is a GeoTIFF already in degrees Celsius (TNn_mean 2010–2024).
- Config YAML (minimal):
    hazards:
      cold:
        active: true
        input: "../data/hazards/cold/TNn_mean_2010_2024.tif"
        threshold: -20
- Optional: if a sibling FD raster exists (../data/hazards/cold/FD_mean_2010_2024.tif),
  it will be summarized too (separate CSV + histogram), but it's not required and
  not referenced in the YAML to keep it minimal.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
from typing import Optional, Dict, Any

# Optional helpers from your project (used only if available)
try:
    from modules.plotting import add_scalebar, add_north_arrow  # noqa: F401
except Exception:
    add_scalebar = None
    add_north_arrow = None


# ---------------------------
# Core API (called from main)
# ---------------------------

def process_cold(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Entry point to process cold hazard if activated in config.

    Returns a dict of key output paths (CSV/PNGs/TIFs). Safe to ignore keys you don't need.
    """
    if "cold" not in config.get("hazards", {}):
        return {}

    cold_conf = config["hazards"]["cold"]
    if not cold_conf.get("active", False):
        return {}

    tif_path = cold_conf["input"]
    threshold = float(cold_conf.get("threshold", -20))
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi_path = config.get("aoi")  # optional, can be None

    # 1) Summarize TNn raster and export CSV + figures
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

    # 3) (Optional) If FD raster exists next to TNn, summarize it too (no YAML key needed)
    fd_summary = {}
    fd_guess = Path(tif_path).parent / "FD_mean_2010_2024.tif"
    if fd_guess.exists():
        fd_csv, fd_hist = summarize_frost_days(
            fd_tif_path=str(fd_guess),
            output_dir=output_dir,
            aoi_path=aoi_path
        )
        fd_summary = {"fd_csv": fd_csv, "fd_hist": fd_hist}

    outputs = {
        "tnn_stats_csv": stats_csv,
        "tnn_hist_png": hist_png,
        "tnn_cdf_png": cdf_png,
        "tnn_binary_tif": bin_tif,
        "tnn_class_tif": class_tif,
        **fd_summary
    }
    # Log succinctly
    print("[cold] outputs:", outputs)
    return outputs


# ---------------------------
# TNn (°C) processing
# ---------------------------

def _open_raster_clip(tif_path: str, aoi_path: Optional[str]) -> xr.DataArray:
    """
    Open GeoTIFF as DataArray and optionally clip to AOI (expects EPSG:4326 or reproj handled by rioxarray).
    """
    da = rioxarray.open_rasterio(tif_path).squeeze("band", drop=True)

    # Ensure spatial dims and CRS
    if "x" in da.dims and "y" in da.dims:
        # already set by rioxarray
        pass
    else:
        # Fallback for (longitude, latitude) naming
        if "longitude" in da.dims and "latitude" in da.dims:
            da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)

    if not da.rio.crs:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    # Optional clip
    if aoi_path:
        aoi = gpd.read_file(aoi_path)
        aoi = aoi.to_crs(da.rio.crs)
        da = da.rio.clip(aoi.geometry.values, aoi.crs, drop=True)

    return da


def summarize_cold_tnn(tif_path: str, output_dir: Path, aoi_path: Optional[str], threshold: float):
    """
    Compute area stats for TNn (°C) and export:
    - CSV with min/mean/max, quantiles, area share below threshold
    - Histogram PNG
    - CDF PNG
    """
    da = _open_raster_clip(tif_path, aoi_path)
    arr = da.values.astype("float32")
    nod = _nodata_value(da)
    mask = np.isfinite(arr) if nod is None else (arr != nod)
    vals = arr[mask]

    if vals.size == 0:
        raise ValueError("No valid data found in TNn raster (after clipping or masking).")

    # Basic stats
    q = np.nanpercentile(vals, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    stats = {
        "min": float(np.nanmin(vals)),
        "mean": float(np.nanmean(vals)),
        "max": float(np.nanmax(vals)),
        "q01": float(q[0]), "q05": float(q[1]), "q10": float(q[2]),
        "q25": float(q[3]), "q50": float(q[4]), "q75": float(q[5]),
        "q90": float(q[6]), "q95": float(q[7]), "q99": float(q[8]),
        "threshold": float(threshold)
    }

    # Area share (<= threshold in °C)
    share = float(np.mean(vals <= threshold))
    stats["share_le_threshold"] = share  # fraction 0..1

    # Save CSV
    df = pd.DataFrame([stats])
    csv_path = output_dir / "cold_tnn_summary.csv"
    df.to_csv(csv_path, index=False)

    # Figures: histogram & CDF with threshold line
    hist_png = output_dir / "cold_tnn_histogram.png"
    cdf_png = output_dir / "cold_tnn_cdf.png"
    _plot_histogram(vals, threshold, title="TNn (°C) — Area distribution", out_path=hist_png, unit="°C")
    _plot_cdf(vals, threshold, title="TNn (°C) — CDF (share ≤ threshold)", out_path=cdf_png, unit="°C")

    # Display inline if in notebook
    try:
        display(Image(filename=str(hist_png)))
        display(Image(filename=str(cdf_png)))
    except Exception:
        pass

    return str(csv_path), str(hist_png), str(cdf_png)


def export_cold_class_rasters(tif_path: str, output_dir: Path, aoi_path: Optional[str], threshold: float):
    """
    Write:
    - Binary raster (1 where TNn <= threshold, else 0)
    - 4-class raster using internal thresholds for communication:
        classes: 1=Mild(>-10), 2=Cold(-10..-20], 3=VeryCold(-20..-30], 4=Extreme(<=-30)
      (These class breaks are internal defaults; you can adjust here later if needed.)
    """
    da = _open_raster_clip(tif_path, aoi_path)
    nod = _nodata_value(da)

    # Binary
    bin_da = xr.ones_like(da, dtype="uint8") * 255  # 255 as nodata default
    mask_valid = np.isfinite(da.values) if nod is None else (da.values != nod)
    bin_arr = np.zeros_like(da.values, dtype="uint8")
    bin_arr[mask_valid] = (da.values[mask_valid] <= threshold).astype("uint8")
    bin_da = bin_da.where(~mask_valid, other=bin_arr)
    bin_da.rio.write_nodata(255, inplace=True)

    # 4 classes
    # breaks high->low cold:  -10, -20, -30
    # class 1: > -10
    # class 2: -10 .. -20
    # class 3: -20 .. -30
    # class 4: <= -30
    breaks = [-10.0, -20.0, -30.0]
    class_da = xr.ones_like(da, dtype="uint8") * 255
    cls = np.zeros_like(da.values, dtype="uint8")
    v = da.values

    cls[(v > breaks[0])] = 1
    cls[(v <= breaks[0]) & (v > breaks[1])] = 2
    cls[(v <= breaks[1]) & (v > breaks[2])] = 3
    cls[(v <= breaks[2])] = 4

    class_da = class_da.where(~mask_valid, other=cls)
    class_da.rio.write_nodata(255, inplace=True)

    # Write rasters
    bin_out = output_dir / "cold_tnn_binary_le_threshold.tif"
    cls_out = output_dir / "cold_tnn_class_4levels.tif"

    # Ensure CRS present
    if not bin_da.rio.crs:
        bin_da = bin_da.rio.write_crs("EPSG:4326", inplace=False)
    if not class_da.rio.crs:
        class_da = class_da.rio.write_crs("EPSG:4326", inplace=False)

    bin_da.rio.to_raster(bin_out)
    class_da.rio.to_raster(cls_out)

    print(f"[cold] wrote: {bin_out.name}, {cls_out.name}")
    return str(bin_out), str(cls_out)


# ---------------------------
# Optional FD (days/year)
# ---------------------------

def summarize_frost_days(fd_tif_path: str, output_dir: Path, aoi_path: Optional[str]):
    """
    If a FD raster (days/year) is available, summarize it:
    - CSV (min/mean/max, quantiles)
    - Histogram PNG (with vertical reference lines at 30/90/180 days)
    """
    da = _open_raster_clip(fd_tif_path, aoi_path)
    arr = da.values.astype("float32")
    nod = _nodata_value(da)
    mask = np.isfinite(arr) if nod is None else (arr != nod)
    vals = arr[mask]

    q = np.nanpercentile(vals, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    stats = {
        "min": float(np.nanmin(vals)),
        "mean": float(np.nanmean(vals)),
        "max": float(np.nanmax(vals)),
        "q01": float(q[0]), "q05": float(q[1]), "q10": float(q[2]),
        "q25": float(q[3]), "q50": float(q[4]), "q75": float(q[5]),
        "q90": float(q[6]), "q95": float(q[7]), "q99": float(q[8]),
        "unit": "days/year"
    }
    df = pd.DataFrame([stats])
    csv_path = output_dir / "cold_fd_summary.csv"
    df.to_csv(csv_path, index=False)

    hist_png = output_dir / "cold_fd_histogram.png"
    _plot_histogram(vals, None, title="Frost Days (days/year) — Area distribution", out_path=hist_png, unit="days/year",
                    vlines=[30, 90, 180])

    try:
        display(Image(filename=str(hist_png)))
    except Exception:
        pass

    return str(csv_path), str(hist_png)


# ---------------------------
# Plot helpers
# ---------------------------

def _plot_histogram(values, threshold: Optional[float], title: str, out_path: Path, unit: str, vlines: Optional[list] = None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=60, stat="density", kde=True)
    if threshold is not None:
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold} {unit})")
    if vlines:
        for v in vlines:
            plt.axvline(v, color="gray", linestyle=":", linewidth=1)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(unit, fontsize=12, fontweight="bold")
    plt.ylabel("Density", fontsize=12)
    if threshold is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_cdf(values, threshold: Optional[float], title: str, out_path: Path, unit: str):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sorted_vals = np.sort(values[np.isfinite(values)])
    y = np.linspace(0, 1, sorted_vals.size, endpoint=True)
    plt.plot(sorted_vals, y, linewidth=2)
    if threshold is not None:
        # Share <= threshold
        share = float(np.mean(values <= threshold))
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold} {unit})")
        plt.axhline(share, color="red", linestyle=":", linewidth=1)
        plt.text(threshold, share, f"  {share:.1%}", va="bottom", ha="left")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(unit, fontsize=12, fontweight="bold")
    plt.ylabel("Cumulative share", fontsize=12)
    if threshold is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _nodata_value(da: xr.DataArray):
    try:
        return da.rio.nodata
    except Exception:
        return None
