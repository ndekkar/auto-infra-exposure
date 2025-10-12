"""
Heat hazard analysis module.

This module processes ERA5 Tmax data from NetCDF to compute:
- Annual maximum daily temperature (Tmax)
- Annual number of hot days (threshold from YAML)

It saves the statistics as CSV and generates plots with smoothing.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import rioxarray
from IPython.display import Image, display
from statsmodels.nonparametric.smoothers_lowess import lowess
from modules.plotting import add_scalebar, add_north_arrow



def _to_celsius(da, convert_kelvin=True):
    """
    Ensure the DataArray is in degrees Celsius.

    - If units are Kelvin (or values look like Kelvin), convert to °C.
    - Always set attrs['units'] = 'degC' on the returned DataArray.
    """
    if not convert_kelvin:
        # Still normalize the units attribute if missing
        if not da.attrs.get("units"):
            da.attrs["units"] = "degC"
        return da

    units = (da.attrs.get("units") or "").lower()
    if units in ("k", "kelvin"):
        out = da - 273.15
        out.attrs["units"] = "degC"
        return out

    # Heuristic: if first value looks like Kelvin (>150), convert
    try:
        v = float(np.asarray(da.values).ravel()[0])
        if v > 150:
            out = da - 273.15
            out.attrs["units"] = "degC"
            return out
    except Exception:
        pass

    # Assume already in Celsius
    da.attrs["units"] = "degC"
    return da



def _get_time_dim(dataarray):
    if "time" in dataarray.dims:
        return "time"
    if "valid_time" in dataarray.dims:
        return "valid_time"
    raise ValueError("No time coordinate found (expected 'time' or 'valid_time').")




def process_heat(config):
    """
    Entry point to process heat hazard if activated in config.
    """
    if "heat" in config["hazards"]:
        heat_conf = config["hazards"]["heat"]
        if heat_conf.get("active", False):
            thr = float(heat_conf.get("threshold", 29))  # default only if YAML missing
            nc_path = heat_conf["input"]
            output_dir = config["output_dir"]
            aoi_path = config.get("aoi")  # may be None
            return process_heat_from_netcdf(nc_path, output_dir, aoi_path, threshold=thr)
    return None


def process_heat_from_netcdf(nc_path, output_dir, aoi_path, threshold=29):
    """
    Process heat data from NetCDF file and generate statistics rasters and plots.
    """
    annual_stats = compute_heat_statistics(nc_path, convert_kelvin=True, threshold=threshold)
    plot_heat_statistics(annual_stats, output_dir, threshold=threshold)
    raster_path = export_max_tmax_raster(nc_path, output_dir, aoi_path)
    return raster_path


def compute_heat_statistics(nc_path, convert_kelvin=True, threshold=29):
    """
    Compute annual heat statistics from a NetCDF file.

    Returns:
        DataFrame with columns:
          - 'Year' (int)
          - 'Tmax' (float, °C): annual maximum of daily Tmax
          - 'HotD' (int, days): count of days with Tmax >= threshold (°C)
    """
    variable = "t2m"
    ds = xr.open_dataset(nc_path)

    # 1) Convert to Celsius and give an explicit name to avoid confusion
    data = _to_celsius(ds[variable], convert_kelvin=convert_kelvin)
    data.name = "t2m_c"  # explicit Celsius variable name

    # 2) Flatten to a DataFrame; keep a robust time column name
    df = data.to_dataframe(name="t2m_c").reset_index()
    time_col = "valid_time" if "valid_time" in df.columns else ("time" if "time" in df.columns else None)
    if time_col is None:
        raise ValueError("No time column found after to_dataframe().")
    df["date"] = pd.to_datetime(df[time_col]).dt.date

    # 3) Daily Tmax over the AOI/grid (max across all pixels for each day)
    daily_max = df.groupby("date")["t2m_c"].max().reset_index()
    daily_max.columns = ["date", "Tmax"]
    daily_max["Year"] = pd.to_datetime(daily_max["date"]).dt.year

    # 4) Keep only years with enough coverage (>= 180 days) to avoid bias
    day_counts = daily_max.groupby("Year")["date"].count().rename("n_days")

    # 5) Annual stats:
    #    - Tmax: maximum daily Tmax within the year (°C)
    #    - HotD: number of days with Tmax >= threshold (°C)
    annual_stats = daily_max.groupby("Year").agg(
        Tmax=("Tmax", "max"),
        HotD=("Tmax", lambda x: (x >= float(threshold)).sum())
    ).reset_index()

    # 6) Filter by coverage and years in scope
    annual_stats = annual_stats.merge(day_counts, on="Year")
    annual_stats = annual_stats[annual_stats["n_days"] >= 180][["Year", "Tmax", "HotD"]]

    return annual_stats[annual_stats["Year"] >= 1960]



def plot_heat_statistics(annual_stats, output_dir, threshold=29):
    """
    Generate and save plots for annual Tmax and number of hot days.
    """
    os.makedirs(output_dir, exist_ok=True)
    annual_stats.to_csv(os.path.join(output_dir, "annual_tmax_stats.csv"), index=False)

    if annual_stats.empty:
        print("Warning: annual_stats is empty. Check time coverage and threshold.")
        return

    # --- Tmax plot ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=annual_stats, x="Year", y="Tmax", color="black", label="Annual Tmax")
    sns.regplot(data=annual_stats, x="Year", y="Tmax", lowess=True, scatter=False,
                label="Local fit (25-year window)", line_kws={"color": "blue"})
    lowess_12 = lowess(annual_stats["Tmax"], annual_stats["Year"], frac=0.3)
    plt.plot(lowess_12[:, 0], lowess_12[:, 1], color="red", label="Local fit (12-year window)")
    plt.title("Time series of annual maximum daily temperature", fontsize=20, fontweight="bold")
    plt.xlabel("Year", fontsize=20, fontweight="bold")
    plt.ylabel("Tmax (°C)", fontsize=20, fontweight="bold")
    plt.legend()
    tmax_path = os.path.join(output_dir, "Tmax2.png")
    plt.savefig(tmax_path, dpi=200, bbox_inches="tight")
    display(Image(filename=tmax_path))
    plt.close()

    # --- Hot days plot ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=annual_stats, x="Year", y="HotD", color="black", label="Hot Days")
    sns.regplot(data=annual_stats, x="Year", y="HotD", lowess=True, scatter=False,
                label="Local fit (25-year window)", line_kws={"color": "blue"})
    lowess_hd = lowess(annual_stats["HotD"], annual_stats["Year"], frac=0.3)
    plt.plot(lowess_hd[:, 0], lowess_hd[:, 1], color="red", label="Local fit (12-year window)")
    plt.title(f"Annual number of days with Tmax ≥ {threshold}°C", fontsize=20, fontweight="bold")
    plt.xlabel("Year", fontsize=20, fontweight="bold")
    plt.ylabel(f"Days ≥ {threshold}°C", fontsize=20, fontweight="bold")
    plt.legend()
    hotd_path = os.path.join(output_dir, "HotD2.png")
    plt.savefig(hotd_path, dpi=200, bbox_inches="tight")
    display(Image(filename=hotd_path))
    plt.close()


def export_max_tmax_raster(nc_path, output_dir, aoi_path, convert_kelvin=True):
    """
    Export max temperature per pixel over time to GeoTIFF (optionally clipped to AOI).
    """
    variable = "t2m"
    ds = xr.open_dataset(nc_path)
    data = _to_celsius(ds[variable], convert_kelvin=convert_kelvin)

    # Get time dimension name (robust)
    time_dim = _get_time_dim(data)

    # Compute max temperature per pixel over time
    max_tmax = data.max(dim=time_dim)

    # Set CRS if not present
    if not max_tmax.rio.crs:
        max_tmax = max_tmax.rio.write_crs("EPSG:4326")

    # Clip to AOI if provided
    if aoi_path:
        aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")
        max_tmax = max_tmax.rio.clip(aoi.geometry.values, aoi.crs, drop=True)

    # Write unit metadata
    max_tmax.attrs["units"] = "degC"

    # Export raster
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "heat_max_tmax.tif")
    max_tmax.rio.to_raster(output_path)
    return output_path
