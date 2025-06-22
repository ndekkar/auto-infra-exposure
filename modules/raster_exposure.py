import os
import rasterio
import geopandas as gpd
from modules.exposure_utils import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map
from modules.crs_utils import assign_or_reproject_to_wgs84

def process_raster_exposures(config, aoi, points, lines, sample_points_per_line):
    """
    Process exposure of infrastructure to all standard raster-based hazards.
    Returns a dictionary of hazard_name -> (raster_path_wgs84, threshold).
    """
    hazard_rasters = {}

    for hazard_name, hazard_conf in config["hazards"].items():
        if not hazard_conf.get("active", False) or hazard_name in ["drought", "heat", "wildfire"]:
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
