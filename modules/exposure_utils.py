"""
Raster-based hazard exposure analysis.

This module includes functions to:
- Extract raster values at infrastructure point locations.
- Assess line exposure by sampling along line geometries.
- Process exposure for all active raster-based hazards.
"""

import numpy as np
import rasterio
import os
import geopandas as gpd
from shapely.geometry import Point
from modules.plotting import plot_and_save_exposure_map

def extract_values_to_points(points_gdf, raster, threshold=0.0):
    """
    Extract raster values at each point location and determine exposure.

    Parameters:
        points_gdf (GeoDataFrame): GeoDataFrame containing point geometries.
        raster (rasterio.io.DatasetReader): Opened raster object.
        threshold (float): Value above which a point is considered exposed.

    Returns:
        GeoDataFrame: Copy of input with added 'haz_val' and 'exposed' columns.
    """
    points_gdf = points_gdf.copy()
    values = []
    for geom in points_gdf.geometry:
        try:
            val = next(raster.sample([(geom.x, geom.y)]))[0]
        except:
            val = None
        values.append(val)

    points_gdf["haz_val"] = values
    points_gdf["exposed"] = points_gdf["haz_val"].apply(
        lambda x: x is not None and x >= threshold
    )
    return points_gdf

def check_line_exposure(line_geom, raster, n_points=10, threshold=0.0):
    """
    Determine if a line is exposed by sampling points along its length.

    Parameters:
        line_geom (shapely.geometry.LineString): Line geometry to test.
        raster (rasterio.io.DatasetReader): Opened raster object.
        n_points (int): Number of points to sample along the line.
        threshold (float): Hazard threshold value.

    Returns:
        bool: True if any sampled point is exposed, else False.
    """
    if line_geom.is_empty or line_geom.length == 0:
        return False
    distances = np.linspace(0, 1, n_points)
    points = [line_geom.interpolate(d, normalized=True) for d in distances]
    coords = [(p.x, p.y) for p in points]
    try:
        values = [val[0] for val in raster.sample(coords)]
        return any(v is not None and v >= threshold for v in values)
    except:
        return False

def process_raster_exposures(config, aoi, points, lines, sample_points_per_line):
    """
    Process all standard raster-based hazards defined in the config.

    Parameters:
        config (dict): Pipeline configuration dictionary.
        aoi (GeoDataFrame): Area of Interest polygon(s).
        points (GeoDataFrame): Infrastructure point features.
        lines (GeoDataFrame): Infrastructure line features.
        sample_points_per_line (int): Number of samples to interpolate per line.

    Returns:
        dict: Dictionary mapping hazard names to their raster path and threshold.
    """
    from .crs_utils import assign_or_reproject_to_wgs84

    hazard_rasters = {}

    for hazard_name, hazard_conf in config["hazards"].items():
        if not hazard_conf.get("active", False) or hazard_name in ["drought", "heat", "wildfire", "cold"]:
            print(f"Skipping hazard: {hazard_name}")
            continue

        print(f"\n--- Processing hazard: {hazard_name} ---")
        raster_path_wgs84 = assign_or_reproject_to_wgs84(hazard_conf["input"])
        raster = rasterio.open(raster_path_wgs84)
        threshold = hazard_conf["threshold"]

        hazard_rasters[hazard_name] = (raster_path_wgs84, threshold)

        points_hazard = extract_values_to_points(points, raster, threshold)
        points_out = os.path.join(config["output_dir"], f"points_exposure_{hazard_name}.shp")
        points_hazard.to_file(points_out)

        lines_hazard = lines.copy()
        lines_hazard["exposed"] = lines_hazard["geometry"].apply(
            lambda geom: check_line_exposure(geom, raster, sample_points_per_line, threshold)
        )
        lines_out = os.path.join(config["output_dir"], f"lines_exposure_{hazard_name}.shp")
        lines_hazard.to_file(lines_out)

        plot_and_save_exposure_map(
            aoi, points_hazard, lines_hazard, hazard_name, config["output_dir"], raster_path_wgs84
        )

    return hazard_rasters
