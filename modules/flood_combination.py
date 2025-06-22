import os
import rasterio
from modules.raster_exposure import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map
import rasterio
import numpy as np

def combine_rasters(raster_paths, output_path, method='max'):
    """
    Combines multiple rasters using the specified method: 'max', 'sum', or 'mean'.
    Saves the result to output_path.
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

    print(f"✅ Combined raster written to: {output_path}")


def process_combined_flood(config, hazard_rasters, aoi, points, lines, sample_points_per_line):
    """
    Combines pluvial and fluvial flood rasters, analyzes exposure, and generates outputs.
    """
    if "pluvial_flood" in hazard_rasters and "fluvial_flood" in hazard_rasters:
        print("\n--- Processing combined flood ---")

        # Input paths and thresholds
        pluvial_path, pluvial_threshold = hazard_rasters["pluvial_flood"]
        fluvial_path, fluvial_threshold = hazard_rasters["fluvial_flood"]
        combined_path = os.path.join(config["output_dir"], "combined_flood.tif")

        # Combine rasters using max value
        combine_rasters([pluvial_path, fluvial_path], combined_path, method='max')
        combined_raster = rasterio.open(combined_path)
        combined_threshold = max(pluvial_threshold, fluvial_threshold)

        # Points exposure
        points_combined = extract_values_to_points(points, combined_raster, combined_threshold)
        points_combined.to_file(os.path.join(config["output_dir"], "points_exposure_combined_flood.shp"))

        # Lines exposure
        lines_combined = lines.copy()
        lines_combined["exposed"] = lines_combined["geometry"].apply(
            lambda geom: check_line_exposure(geom, combined_raster, sample_points_per_line, combined_threshold)
        )
        lines_combined.to_file(os.path.join(config["output_dir"], "lines_exposure_combined_flood.shp"))

        # Plot map
        plot_and_save_exposure_map(
            aoi, points_combined, lines_combined,
            "combined_flood", config["output_dir"], combined_path
        )
