"""
Main script to run the multi-hazard exposure pipeline.

This script loads the configuration, processes infrastructure and hazard data,
and performs exposure analysis for various natural hazards, including:
- Drought
- Heat
- Wildfire
- Flood (pluvial, fluvial, and combined)
- Earthquake
- Landslide
"""

import sys
from modules.config_utils import load_config
from modules.crs_utils import harmonize_crs
from modules.raster_exposure import process_raster_exposures
from modules.flood_combination import process_combined_flood
from modules.drought_module import process_drought
from modules.heat_module import process_heat
from modules.wildfire_module import process_wildfire
import geopandas as gpd

def run_multi_hazard_pipeline(config_path):
    """
    Orchestrates the multi-hazard exposure analysis using the provided configuration.

    Parameters:
        config_path (str): Path to the YAML configuration file.
    """
    # Load configuration
    config = load_config(config_path)

    # Load AOI and infrastructure layers
    aoi = gpd.read_file(config["aoi"])
    aoi_union = aoi.union_all()
    points = gpd.read_file(config["infra_points_input"])
    lines = gpd.read_file(config["infra_lines_input"])

    # Harmonize CRS and clip infrastructure
    sample_points_per_line = 10
    points, lines = harmonize_crs([points, lines], aoi.crs)
    points = points[points.geometry.within(aoi_union)]
    lines = lines[lines.geometry.intersects(aoi_union)]

    # Process non-raster hazards
    process_drought(config)
    heat_raster_path = process_heat(config)
    config["hazards"]["heat"]["input"] = heat_raster_path
    
    wildfire_raster_path = process_wildfire(config)
    config["hazards"]["wildfire"]["input"] = wildfire_raster_path
    
    # Process standard raster-based hazards
    hazard_rasters = process_raster_exposures(config, aoi, points, lines, sample_points_per_line)

    # Process derived hazard: combined flood
    process_combined_flood(config, hazard_rasters, aoi, points, lines, sample_points_per_line)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    run_multi_hazard_pipeline(config_path)
