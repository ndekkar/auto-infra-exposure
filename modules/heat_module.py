"""
Heat hazard analysis module.

This module processes ERA5 Tmax data from NetCDF to compute:
- Annual maximum daily temperature (Tmax)
- Annual number of hot days (Tmax ≥ 29°C)

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

def process_heat(config):
    """
    Entry point to process heat hazard if activated in config.

    Parameters:
        config (dict): Full YAML configuration dictionary.
    """
    if "heat" in config["hazards"]:
        heat_conf = config["hazards"]["heat"]
        if heat_conf.get("active", False):
            return process_heat_from_netcdf(heat_conf["input"], config["output_dir"], config["aoi"])

def process_heat_from_netcdf(nc_path, output_dir, aoi_path):
    """
    Process heat data from NetCDF file and generate statistics rasters and plots.

    Parameters:
        nc_path (str): Path to the NetCDF file containing 2m temperature (t2m).
        output_dir (str): Directory to save plots and output CSV.
    """
    annual_stats = compute_heat_statistics(nc_path, convert_kelvin=True)
    plot_heat_statistics(annual_stats, output_dir)
    raster_path = export_max_tmax_raster(nc_path, output_dir, aoi_path) 
    return raster_path

def compute_heat_statistics(nc_path, convert_kelvin=True):
    """
    Compute annual heat statistics from a NetCDF file.

    Parameters:
        nc_path (str): Path to NetCDF file containing ERA5 't2m' variable.
        convert_kelvin (bool): If True, convert temperature from Kelvin to Celsius.

    Returns:
        DataFrame: Annual statistics with columns 'Year', 'Tmax', and 'HotD'.
    """
    variable = "t2m"
    ds = xr.open_dataset(nc_path)
    data = ds[variable]

    if convert_kelvin:
        data = data - 273.15

    df = data.to_dataframe().reset_index()
    df['date'] = pd.to_datetime(df['valid_time']).dt.date

    daily_max = df.groupby('date')[variable].max().reset_index()
    daily_max.columns = ['date', 'Tmax']
    daily_max['Year'] = pd.to_datetime(daily_max['date']).dt.year

    annual_stats = daily_max.groupby('Year').agg(
        Tmax=('Tmax', 'max'),
        HotD=('Tmax', lambda x: (x >= 29).sum())
    ).reset_index()

    return annual_stats[annual_stats['Year'] >= 1960]

def plot_heat_statistics(annual_stats, output_dir):
    """
    Generate and save plots for annual Tmax and number of hot days.

    Parameters:
        annual_stats (DataFrame): Output from compute_heat_statistics().
        output_dir (str): Directory to save PNG plots and CSV file.
    """
    annual_stats.to_csv(os.path.join(output_dir, 'annual_tmax_stats.csv'), index=False)

    # --- Tmax plot ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=annual_stats, x='Year', y='Tmax', color="black", label="Annual Tmax")
    sns.regplot(data=annual_stats, x='Year', y='Tmax', lowess=True, scatter=False,
                label="Local fit (25-year window)", line_kws={"color": "blue"})
    lowess_12 = lowess(annual_stats['Tmax'], annual_stats['Year'], frac=0.3)
    plt.plot(lowess_12[:, 0], lowess_12[:, 1], color="red", label="Local fit (12-year window)")
    plt.title("Time series of annual maximum daily temperature", fontsize=20, fontweight="bold")
    plt.xlabel("Year", fontsize=20, fontweight="bold")
    plt.ylabel("Tmax (°C)", fontsize=20, fontweight="bold")
    plt.legend()
    tmax_path = os.path.join(output_dir, 'Tmax2.png')
    #add_scalebar(ax, loc='lower left')      # or use length_km=5 for fixed scale
    #add_north_arrow(ax, loc='upper left')   # change location if needed
    plt.savefig(tmax_path)
    display(Image(filename=tmax_path))
    plt.close()

    # --- Hot days plot ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=annual_stats, x='Year', y='HotD', color="black", label="Hot Days")
    sns.regplot(data=annual_stats, x='Year', y='HotD', lowess=True, scatter=False,
                label="Local fit (25-year window)", line_kws={"color": "blue"})
    lowess_hd = lowess(annual_stats['HotD'], annual_stats['Year'], frac=0.3)
    plt.plot(lowess_hd[:, 0], lowess_hd[:, 1], color="red", label="Local fit (12-year window)")
    plt.title("Annual number of days with Tmax ≥ 29°C", fontsize=20, fontweight="bold")
    plt.xlabel("Year", fontsize=20, fontweight="bold")
    plt.ylabel("", fontsize=20, fontweight="bold")
    plt.legend()
    hotd_path = os.path.join(output_dir, 'HotD2.png')
    #add_scalebar(ax, loc='lower left')      # or use length_km=5 for fixed scale
    #add_north_arrow(ax, loc='upper left')   # change location if needed
    plt.savefig(hotd_path)
    display(Image(filename=hotd_path))
    plt.close()

def export_max_tmax_raster(nc_path, output_dir, aoi_path, convert_kelvin=True):
    """
    Export annual maximum temperature per pixel from NetCDF to GeoTIFF,
    optionally clipped to an Area of Interest (AOI).

    Parameters:
        nc_path (str): Path to the input NetCDF file (variable 't2m')
        output_dir (str): Output directory to save the raster
        aoi_path (str, optional): Path to AOI shapefile or GeoJSON to clip the raster
        convert_kelvin (bool): Convert temperature from Kelvin to Celsius (default: True)

    Output:
        Saves 'heat_max_tmax.tif' in the output directory.
    """
    variable = "t2m"
    ds = xr.open_dataset(nc_path)
    data = ds[variable]

    if convert_kelvin:
        data = data - 273.15

    # Get time dimension name
    time_dim = "time" if "time" in data.dims else "valid_time"

    # Compute max temperature per pixel over time
    max_tmax = data.max(dim=time_dim)

    # Set CRS if not present
    if not max_tmax.rio.crs:
        max_tmax = max_tmax.rio.write_crs("EPSG:4326")
    # Clip to AOI if provided
    if aoi_path:
        aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")
        max_tmax = max_tmax.rio.clip(aoi.geometry.values, aoi.crs, drop=True)

    # Export raster
    output_path = os.path.join(output_dir, "heat_max_tmax.tif")
    max_tmax.rio.to_raster(output_path)
    return output_path