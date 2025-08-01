"""
Flood combination module.

This module combines fluvial and pluvial flood rasters into a single raster using a specified method,
then computes exposure for infrastructure points and lines, and generates output maps and shapefiles.
"""

import os
import rasterio
from modules.raster_exposure import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map
import rasterio
import numpy as np

def combine_rasters(raster_paths, output_path, method='max'):
    """
    Combine multiple raster layers using a specified method.

    Parameters:
        raster_paths (list): List of paths to raster files to be combined.
        output_path (str): Path where the combined raster will be saved.
        method (str): Method for combination: 'max', 'sum', or 'mean'.

    Raises:
        ValueError: If an unsupported method is specified.
    """
    with rasterio.open(raster_paths[0]) as src_ref:
        meta = src_ref.meta.copy()
        data = src_ref.read(1).astype(float)

    for path in raster_paths[1:]:
        with rasterio.open(path) as src:
            data_new = src.read(1).astype(float)
            if method == 'max':
                data = np.maximum(data, data_new)
            elif method == 'sum':
                data = np.nan_to_num(data) + np.nan_to_num(data_new)
            elif method == 'mean':
                data = (data + data_new) / 2
            else:
                raise ValueError("Invalid combination method.")

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)


def process_combined_flood(config, hazard_rasters, aoi, points, lines, sample_points_per_line):
    """
    Process combined flood hazard (fluvial + pluvial) and generate exposure outputs.
    
    Parameters:
        config (dict): Loaded YAML configuration.
        hazard_rasters (dict): Dict with hazard_name -> (raster_path, threshold).
        aoi (GeoDataFrame): Area of interest.
        points (GeoDataFrame): Merged infrastructure points (with 'infra_type').
        lines (GeoDataFrame): Merged infrastructure lines (with 'infra_type').
        sample_points_per_line (int): Sampling density for line exposure.
    """
    if "pluvial_flood" in hazard_rasters and "fluvial_flood" in hazard_rasters:
        print("\n--- Processing combined flood ---")

        pluvial_path, pluvial_threshold = hazard_rasters["pluvial_flood"]
        fluvial_path, fluvial_threshold = hazard_rasters["fluvial_flood"]

        combined_path = os.path.join(config["output_dir"], "combined_flood.tif")
        combined_threshold = max(pluvial_threshold, fluvial_threshold)

        # Combine rasters
        combine_rasters([pluvial_path, fluvial_path], combined_path, method='max')
        combined_raster = rasterio.open(combined_path)

        # Points exposure
        points_combined = extract_values_to_points(points.copy(), combined_raster, combined_threshold)
        points_combined.to_file(os.path.join(config["output_dir"], "points_exposure_combined_flood.shp"))

        # Lines exposure
        lines_combined = lines.copy()
        lines_combined["exposed"] = lines_combined["geometry"].apply(
            lambda geom: check_line_exposure(geom, combined_raster, sample_points_per_line, combined_threshold)
        )
        lines_combined.to_file(os.path.join(config["output_dir"], "lines_exposure_combined_flood.shp"))

        # Plot combined map
        plot_and_save_exposure_map(
            aoi=aoi,
            points=points_combined,
            lines=lines_combined,
            hazard_name="combined_flood",
            output_dir=config["output_dir"],
            raster_path=combined_path,
            group_by_type=True 
        )

