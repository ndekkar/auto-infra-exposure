import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
from statsmodels.nonparametric.smoothers_lowess import lowess

def process_heat(config):
    """
    Wrapper to compute and plot annual Tmax and hot days from NetCDF.
    """
    if "heat" in config["hazards"]:
        heat_conf = config["hazards"]["heat"]
        if heat_conf.get("active", False):
            print("\n--- Processing heat (NetCDF Tmax analysis) ---")
            process_heat_from_netcdf(heat_conf["input"], config["output_dir"])

def process_heat_from_netcdf(nc_path, output_dir):
    annual_stats = compute_heat_statistics(nc_path, convert_kelvin=True)
    plot_heat_statistics(annual_stats, output_dir)

def compute_heat_statistics(nc_path, convert_kelvin=True):
    """
    Extract annual Tmax and number of hot days (Tmax ≥ 29°C) from NetCDF.
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
    Plot and save temperature statistics: Tmax and number of hot days.
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
    plt.savefig(hotd_path)
    display(Image(filename=hotd_path))
    plt.close()
