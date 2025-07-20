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
import pandas as pd
from modules.exposure_utils import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map
from modules.crs_utils import assign_or_reproject_to_wgs84
from rasterio.mask import mask


def process_raster_exposures(config, aoi, points_by_type, lines_by_type, sample_points_per_line):
    """
    Process exposure of all infrastructures (merged points and lines) to raster-based hazards.

    Parameters:
        config (dict): YAML configuration.
        aoi (GeoDataFrame): Area of interest.
        points_by_type (dict): Dict of {name: GeoDataFrame} for points.
        lines_by_type (dict): Dict of {name: GeoDataFrame} for lines.
        sample_points_per_line (int): Number of sample points per line for exposure.

    Returns:
        dict: hazard_name -> (raster_path, threshold or None)
    """

    # Tag each infrastructure point and line with its type
    for infra_type, gdf in points_by_type.items():
        gdf["infra_type"] = infra_type
    for infra_type, gdf in lines_by_type.items():
        gdf["infra_type"] = infra_type

    # Merge all points and lines into unified GeoDataFrames
    all_points = gpd.GeoDataFrame(pd.concat(points_by_type.values(), ignore_index=True), crs=aoi.crs)
    all_lines = gpd.GeoDataFrame(pd.concat(lines_by_type.values(), ignore_index=True), crs=aoi.crs)

    hazard_rasters = {}

    for hazard_name, hazard_conf in config["hazards"].items():
        if not hazard_conf.get("active", False) or hazard_name in ["drought", "heat", "wildfire"]:
            continue

        # Special case for earthquake: only clip and visualize the raster
        if hazard_name == "earthquake":
            print(f"\n--- Clipping and saving earthquake raster ---")
            raster_path = hazard_conf["input"]
            raster_path_wgs84 = assign_or_reproject_to_wgs84(raster_path)

            with rasterio.open(raster_path_wgs84) as src:
                geoms = [f["geometry"] for f in aoi.to_crs(src.crs).__geo_interface__["features"]]
                clipped, transform = mask(src, geoms, crop=True)
                meta = src.meta.copy()
                meta.update({
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": transform
                })

            out_path = os.path.join(config["output_dir"], "earthquake_clipped.tif")
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(clipped)

            plot_and_save_exposure_map(
                aoi=aoi,
                points=gpd.GeoDataFrame(geometry=[]),
                lines=gpd.GeoDataFrame(geometry=[]),
                hazard_name=hazard_name,
                output_dir=config["output_dir"],
                raster_path=out_path
            )

            hazard_rasters[hazard_name] = (out_path, None)
            continue

        # Standard raster hazard exposure processing

        raster_path = hazard_conf["input"]
        threshold = hazard_conf["threshold"]
        raster_path_wgs84 = assign_or_reproject_to_wgs84(raster_path)

        with rasterio.open(raster_path_wgs84) as raster:
            hazard_rasters[hazard_name] = (raster_path_wgs84, threshold)

            # Points exposure
            points_exposed = extract_values_to_points(all_points.copy(), raster, threshold)
            points_out = os.path.join(config["output_dir"], f"points_exposure_{hazard_name}.shp")
            points_exposed.to_file(points_out)

            # Lines exposure
            lines_exposed = all_lines.copy()
            lines_exposed["exposed"] = lines_exposed["geometry"].apply(
                lambda geom: check_line_exposure(geom, raster, sample_points_per_line, threshold)
            )
            lines_out = os.path.join(config["output_dir"], f"lines_exposure_{hazard_name}.shp")
            lines_exposed.to_file(lines_out)

            # Plot exposure map
            plot_and_save_exposure_map(
                aoi=aoi,
                points=points_exposed,
                lines=lines_exposed,
                hazard_name=hazard_name,
                output_dir=config["output_dir"],
                raster_path=raster_path_wgs84
            )

    return hazard_rasters