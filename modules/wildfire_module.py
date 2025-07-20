"""
Module for processing wildfire hazard using GlobFire burned area data.

This module extracts wildfire centroids from shapefiles within an Area of Interest (AOI)
and generates a kernel density estimate (KDE) heatmap of wildfire occurrence.

Functions:
- process_wildfire: Main entry point to control wildfire hazard processing.
- extract_burned_area_centroids: Reads and filters GlobFire shapefiles to extract burned area centroids.
- plot_fire_density: Generates and saves a KDE heatmap of wildfire centroids.
"""

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import contextily as ctx

def process_wildfire(config):
    """
    Process wildfire hazard using burned area shapefiles and generate a density map.

    Parameters:
    - config (dict): Configuration dictionary loaded from YAML.
    """
    if "wildfire" in config["hazards"]:
        wildfire_conf = config["hazards"]["wildfire"]
        if wildfire_conf.get("active", False):
            centroids_path = os.path.join(config["output_dir"], "wildfire_centroids.gpkg")
            raster_path = os.path.join(config["output_dir"], "wildfire_centroids.tif")
            
            extract_burned_area_centroids(
                aoi_path=config["aoi"],
                globfire_dir=wildfire_conf["input"],
                output_path=os.path.join(config["output_dir"], "wildfire_centroids.gpkg")
            )
            plot_fire_density(
                gpkg_path=os.path.join(config["output_dir"], "wildfire_centroids.gpkg"),
                save_path=os.path.join(config["output_dir"], "wildfire_density.png"),
                aoi_path=config["aoi"]
            )
            return rasterize_fire_centroids(centroids_path, raster_path)

def extract_burned_area_centroids(aoi_path, globfire_dir, output_path):
    """
    Extract centroids of burned areas from GlobFire shapefiles within AOI.

    Parameters:
    - aoi_path (str): Path to the AOI shapefile.
    - globfire_dir (str): Directory containing burned area shapefiles (GlobFire).
    - output_path (str): Path where the extracted centroids will be saved as a GeoPackage.
    """
    aoi = gpd.read_file(aoi_path).to_crs(epsg=3857)
    aoi_buffered = aoi.buffer(10000)  # 10 km buffer
    aoi_geom = aoi_buffered.unary_union

    all_points = []

    for file in os.listdir(globfire_dir):
        if file.endswith('.shp'):
            shp_path = os.path.join(globfire_dir, file)
            try:
                gdf = gpd.read_file(shp_path).to_crs(epsg=3857)
                filtered = gdf[gdf.intersects(aoi_geom)]
                centroids = filtered.centroid
                centroids = centroids.to_crs(epsg=4326)
                all_points.extend([Point(pt.x, pt.y) for pt in centroids])
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not all_points:
        print(" No burned area centroids found.")
        return

    output_gdf = gpd.GeoDataFrame(geometry=all_points, crs='EPSG:4326')
    output_gdf.to_file(output_path, driver='GPKG', layer='burned_area_centroids')

def plot_fire_density(gpkg_path, save_path=None, aoi_path=None):
    """
    Generate and save a KDE heatmap of wildfire centroids, styled like other exposure maps.

    Parameters:
    - gpkg_path (str): Path to the GeoPackage file containing wildfire centroids.
    - save_path (str): Path to save the output PNG image.
    - aoi_path (str): Path to the AOI shapefile for extent and clipping.
    """
    gdf = gpd.read_file(gpkg_path)
    gdf = gdf.to_crs(epsg=3857)

    # Load AOI for bounds and plotting
    aoi = gpd.read_file(aoi_path).to_crs(epsg=3857) if aoi_path else None
    bounds = aoi.total_bounds if aoi is not None else gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds

    df = pd.DataFrame({'x': gdf.geometry.x, 'y': gdf.geometry.y})

    fig, ax = plt.subplots(figsize=(12, 12)) 
    sns.kdeplot(
        data=df, x='x', y='y',
        fill=True, cmap='Reds', bw_adjust=0.5,
        levels=100, thresh=0.05, ax=ax
    )

    if aoi is not None:
        aoi.boundary.plot(ax=ax, color="black", linewidth=1, zorder=2)

    # Remove axis graduations
    ax.axis("off")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    # Add basemap
    try:
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"Basemap error: {e}")

    ax.set_title("Wildfire Density", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    plt.close()

def rasterize_fire_centroids(gpkg_path, output_tif_path, resolution=0.01):
    """
    Rasterize centroid points from GeoPackage into a binary raster.

    Parameters:
        gpkg_path (str): Path to the GeoPackage with burned area centroids.
        output_tif_path (str): Path to save the output raster.
        resolution (float): Raster resolution in degrees.

    Returns:
        str: Path to the output .tif file.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    gdf = gpd.read_file(gpkg_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_origin(minx, maxy, resolution, resolution)

    shapes = [(geom, 1) for geom in gdf.geometry]
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    with rasterio.open(
        output_tif_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(raster, 1)
    return output_tif_path
