"""
Module for computing exposure of infrastructure to raster-based hazards.

This module processes each active raster hazard (excluding drought, heat, and wildfire),
evaluates point and line exposure based on thresholds, and generates corresponding
shapefiles and maps.

Functions:
- process_raster_exposures: Main function to handle exposure analysis for all applicable hazards.
"""

import os
import rasterio
import geopandas as gpd
from modules.exposure_utils import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map
from modules.crs_utils import assign_or_reproject_to_wgs84

def process_raster_exposures(config, aoi, points, lines, sample_points_per_line):
    """
    Process exposure of infrastructure (points and lines) to all standard raster-based hazards.

    This function:
    - Filters active raster hazards (excluding non-raster ones like drought, heat, wildfire)
    - Ensures rasters are in WGS84
    - Extracts exposure values for points and lines
    - Saves exposure results to Shapefiles
    - Generates and saves exposure maps

    Parameters:
        config (dict): Parsed YAML configuration.
        aoi (GeoDataFrame): Area of interest.
        points (GeoDataFrame): Infrastructure points.
        lines (GeoDataFrame): Infrastructure lines.
        sample_points_per_line (int): Number of interpolated points used per line for exposure check.

    Returns:
        dict: Dictionary mapping each hazard name to a tuple: (raster_path_wgs84, threshold)
    """
    hazard_rasters = {}

    for hazard_name, hazard_conf in config["hazards"].items():
        if not hazard_conf.get("active", False) or hazard_name in ["drought", "wildfire"]:
            continue

        print(f"\n--- Processing hazard: {hazard_name} ---")
        raster_path_wgs84 = assign_or_reproject_to_wgs84(hazard_conf["input"])
        raster = rasterio.open(raster_path_wgs84)
        threshold = hazard_conf["threshold"]

        hazard_rasters[hazard_name] = (raster_path_wgs84, threshold)

        # Points exposure
        points_hazard = extract_values_to_points(points, raster, threshold)
        points_out = os.path.join(config["output_dir"], f"points_exposure_{hazard_name}.shp")
        points_hazard.to_file(points_out)

        # Lines exposure
        lines_hazard = lines.copy()
        lines_hazard["exposed"] = lines_hazard["geometry"].apply(
            lambda geom: check_line_exposure(geom, raster, sample_points_per_line, threshold)
        )
        lines_out = os.path.join(config["output_dir"], f"lines_exposure_{hazard_name}.shp")
        lines_hazard.to_file(lines_out)

        # Plot exposure map
        plot_and_save_exposure_map(aoi, points_hazard, lines_hazard, hazard_name, config["output_dir"], raster_path_wgs84)

    return hazard_rasters
