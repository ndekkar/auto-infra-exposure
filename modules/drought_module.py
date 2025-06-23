"""
Drought module: Computes and visualizes the NDMI drought index using Google Earth Engine (MODIS).
"""

import os
import datetime
import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geemap
import ee
from rasterio.mask import mask

def process_drought(config):
    """
    Entry point for drought processing based on the pipeline configuration.

    Parameters:
        config (dict): Dictionary loaded from the YAML configuration file.
    """
    if "drought" in config["hazards"]:
        drought_conf = config["hazards"]["drought"]
        if drought_conf.get("active", False):
            print("\n--- Processing drought index (NDMI) ---")
            generate_drought_index(config["aoi"], config["output_dir"], drought_conf)

def generate_drought_index(aoi_path, output_dir, drought_conf):
    """
    Generate NDMI drought index and plot the results.

    Parameters:
        aoi_path (str): Path to the Area of Interest (AOI) shapefile.
        output_dir (str): Directory to store output files.
        drought_conf (dict): Drought configuration (start date, end date).
    """
    out_path = os.path.join(output_dir, "drought_index_ndmi.tif")
    fetch_ndmi_from_gee(aoi_path, drought_conf, out_path)
    plot_ndmi_raster(out_path, output_dir, aoi_path)

def fetch_ndmi_from_gee(aoi_path, drought_conf, output_path):
    """
    Fetch NDMI imagery from MODIS via Google Earth Engine and export to GeoTIFF.

    Parameters:
        aoi_path (str): Path to the AOI shapefile.
        drought_conf (dict): Contains start and end date info for NDMI calculation.
        output_path (str): Output path for the exported raster.
    """
    ee.Initialize()

    # Define date range for analysis
    start = datetime.datetime(drought_conf["start_year"], drought_conf["start_month"], 1)
    end = datetime.datetime(drought_conf["end_year"], drought_conf["end_month"], 28)

    # Convert AOI shapefile to Earth Engine geometry
    aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
    #bounds = aoi.total_bounds
    #geom = ee.Geometry.BBox(*bounds)
    #geom = geemap.geopandas_to_ee(gpd.GeoDataFrame(geometry=[aoi.unary_union], crs="EPSG:4326"))
    ee_fc = geemap.geopandas_to_ee(aoi)
    ee_geom = ee_fc.geometry()
    
    # Build NDMI image collection from MODIS
    collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(start, end) \
        .map(lambda img: img.normalizedDifference(['sur_refl_b02', 'sur_refl_b07']).rename('NDMI'))

    # Export the mean NDMI image
    mean_image = collection.select('NDMI').mean().clip(ee_geom)
    geemap.ee_export_image(mean_image, filename=output_path, scale=250, region=ee_geom)


def plot_ndmi_raster(tif_path, output_dir, aoi_path):
    """
    Plot the exported NDMI raster, clipped visually to the AOI, and save as PNG.

    Parameters:
        tif_path (str): Path to the NDMI raster GeoTIFF.
        output_dir (str): Directory to save the plot.
        aoi_path (str): Path to the AOI shapefile.
    """
    try:
        with rasterio.open(tif_path) as src:
            
            aoi = gpd.read_file(aoi_path).to_crs(src.crs)
            shapes = [feature["geometry"] for feature in aoi.__geo_interface__["features"]]

            
            data, transform = mask(src, shapes, crop=True)
            data = data[0]  # extraire la première bande

            
            vmin, vmax = np.nanmin(data), np.nanmax(data)

            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(data, cmap='RdYlGn', vmin=vmin, vmax=vmax)
            ax.set_title("Drought Index (NDMI)")
            ax.axis('off')

            # Colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, pad=0.02, aspect=25)
            cbar.set_label("NDMI [-1 to 1]")

            plt.tight_layout()
            map_path = os.path.join(output_dir, "drought_map_ndmi.png")
            plt.savefig(map_path, dpi=300)
            plt.show()

    except Exception as e:
        print(f"Unable to plot drought map: {e}")
