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



import os
import sys
import geopandas as gpd
import pandas as pd
from modules.config_utils import load_config
from modules.crs_utils import harmonize_crs
from modules.plotting import plot_initial_map_by_type
from modules.raster_exposure import process_raster_exposures
from modules.flood_combination import process_combined_flood
from modules.drought_module import process_drought
from modules.heat_module import process_heat
from modules.wildfire_module import process_wildfire


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
    
    # Load Infrastructure data
    points_by_type = {}
    lines_by_type = {}

    for name, path in config["infrastructure_inputs"]["points"].items():
        if path is not None:
            points_by_type[name] = gpd.read_file(path)
    
    for name, path in config["infrastructure_inputs"]["lines"].items():
        if path is not None:
            lines_by_type[name] = gpd.read_file(path)

    # Harmonize CRS and clip infrastructure
    for name, gdf in points_by_type.items():
        gdf, = harmonize_crs([gdf], aoi.crs)
        gdf = gdf[gdf.geometry.within(aoi_union)]
        points_by_type[name] = gdf
    
    for name, gdf in lines_by_type.items():
        gdf, = harmonize_crs([gdf], aoi.crs)
        gdf = gdf[gdf.geometry.intersects(aoi_union)]
        lines_by_type[name] = gdf


    # Plot initial map
    plot_initial_map_by_type(
    aoi,
    points_by_type=points_by_type,
    lines_by_type=lines_by_type,
    output_path=os.path.join(config["output_dir"], "initial_context_map.png")
    )
   
    
    # Process non-raster hazards
    # Drought is disabled cause not relevant for Energy Infra exposure
    #process_drought(config)


    heat_raster_path = process_heat(config)
    config["hazards"]["heat"]["input"] = heat_raster_path


    wildfire_raster_path = process_wildfire(config)
    config["hazards"]["wildfire"]["input"] = wildfire_raster_path
    
    # Process standard raster-based hazards  
    for name, gdf in points_by_type.items():
        gdf["infra_type"] = name
    for name, gdf in lines_by_type.items():
        gdf["infra_type"] = name
    
    all_points = gpd.GeoDataFrame(pd.concat(points_by_type.values(), ignore_index=True), crs=aoi.crs)
    all_lines = gpd.GeoDataFrame(pd.concat(lines_by_type.values(), ignore_index=True), crs=aoi.crs)

    sample_points_per_line = 10
    hazard_rasters = process_raster_exposures(config=config,aoi=aoi,points_by_type=points_by_type,lines_by_type=lines_by_type,
        sample_points_per_line=sample_points_per_line)
    

    # Process derived hazard: combined flood
    process_combined_flood(config=config, hazard_rasters=hazard_rasters,  aoi=aoi, points=all_points, lines=all_lines,     
                           sample_points_per_line=sample_points_per_line)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    run_multi_hazard_pipeline(config_path)
