import os
import datetime
import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geemap
import ee

def process_drought(config):
    """
    Wrapper function to generate and plot NDMI drought index from GEE.
    """
    if "drought" in config["hazards"]:
        drought_conf = config["hazards"]["drought"]
        if drought_conf.get("active", False):
            print("\n--- Processing drought index (NDMI) ---")
            generate_drought_index(config["aoi"], config["output_dir"], drought_conf)

def generate_drought_index(aoi_path, output_dir, drought_conf):
    """
    Generate NDMI drought index using Earth Engine and export + plot it.
    """
    out_path = os.path.join(output_dir, "drought_index_ndmi.tif")
    fetch_ndmi_from_gee(aoi_path, drought_conf, out_path)
    plot_ndmi_raster(out_path, output_dir)

def fetch_ndmi_from_gee(aoi_path, drought_conf, output_path):
    """
    Fetch NDMI from MODIS on Earth Engine and export it to GeoTIFF.
    """
    ee.Initialize()

    start = datetime.datetime(drought_conf["start_year"], drought_conf["start_month"], 1)
    end = datetime.datetime(drought_conf["end_year"], drought_conf["end_month"], 28)

    aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
    bounds = aoi.total_bounds
    geom = ee.Geometry.BBox(*bounds)

    collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(start, end) \
        .map(lambda img: img.normalizedDifference(['sur_refl_b02', 'sur_refl_b07']).rename('NDMI'))

    mean_image = collection.select('NDMI').mean().clip(geom)
    geemap.ee_export_image(mean_image, filename=output_path, scale=250, region=geom)

def plot_ndmi_raster(tif_path, output_dir):
    """
    Plot NDMI raster from a GeoTIFF file.
    """
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            vmin, vmax = np.nanmin(data), np.nanmax(data)

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(data, cmap='RdYlGn', vmin=vmin, vmax=vmax)
            ax.set_title("Drought Index (NDMI)")
            ax.axis('off')

            cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, pad=0.02, aspect=25)
            cbar.set_label("NDMI [-1 to 1]")

            plt.tight_layout()
            map_path = os.path.join(output_dir, "drought_map_ndmi.png")
            plt.savefig(map_path, dpi=300)
            plt.show()
    except Exception as e:
        print(f"Unable to plot drought map: {e}")
