"""
Module for computing exposure of infrastructure to raster-based hazards.

This module processes each active raster hazard (excluding drought, heat, and wildfire),
evaluates point and line exposure based on thresholds, and generates corresponding
shapefiles and maps.

Functions:
- process_raster_exposures: Main function to handle exposure analysis for all applicable hazards.
"""

import os
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask

from modules.exposure_utils import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map
from modules.crs_utils import assign_or_reproject_to_wgs84


# -----------------------------
# Small helpers
# -----------------------------
def _empty_gdf(crs) -> gpd.GeoDataFrame:
    """Create an empty GeoDataFrame with the given CRS."""
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _ensure_exposed_bool(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure the GeoDataFrame has a boolean 'exposed' column.
    If already present, cast to bool. If missing, try to infer from existing fields (no-op otherwise).
    """
    if gdf is None or len(gdf) == 0:
        # Still create column for schema stability
        gdf = gdf.copy()
        if "exposed" not in gdf.columns:
            gdf["exposed"] = pd.Series(dtype=bool)
        return gdf

    gdf = gdf.copy()
    if "exposed" in gdf.columns:
        gdf["exposed"] = gdf["exposed"].astype(bool)
        return gdf

    # Heuristic fallback: if there is a column like 'value' and a threshold was already applied upstream,
    # extract_values_to_points/check_line_exposure should already have set 'exposed'; so usually we do nothing here.
    # But we still create the column to avoid schema issues.
    gdf["exposed"] = pd.Series(dtype=bool)
    return gdf


# -----------------------------
# Main function
# -----------------------------
def process_raster_exposures(config,
                             aoi: gpd.GeoDataFrame,
                             points_by_type: Dict[str, gpd.GeoDataFrame],
                             lines_by_type: Dict[str, gpd.GeoDataFrame],
                             sample_points_per_line: int) -> Dict[str, Dict[str, gpd.GeoDataFrame]]:
    """
    Process exposure of all infrastructures (merged points and lines) to raster-based hazards.

    Parameters:
        config (dict): YAML configuration.
        aoi (GeoDataFrame): Area of interest.
        points_by_type (dict): Dict of {name: GeoDataFrame} for points.
        lines_by_type (dict): Dict of {name: GeoDataFrame} for lines.
        sample_points_per_line (int): Number of sample points per line for exposure.

    Returns:
        dict: A normalized mapping:
            {
              hazard_name: {
                "points_exposed": GeoDataFrame,   # has boolean column 'exposed'
                "lines_exposed":  GeoDataFrame,   # has boolean column 'exposed'
                "raster_path":    "<path/to/raster>",
                "threshold":      <float | int | None>
              },
              ...
            }
        Notes:
            - For hazards without an exposure study (e.g., earthquake here), we return empty GDFs
              but still provide 'raster_path' for downstream visualization or overlay stats if needed.
    """

    # Tag each infrastructure point and line with its type (preserve the user's fields)
    for infra_type, gdf in points_by_type.items():
        gdf["infra_type"] = infra_type
    for infra_type, gdf in lines_by_type.items():
        gdf["infra_type"] = infra_type

    # Merge all points and lines into unified GeoDataFrames
    all_points = gpd.GeoDataFrame(pd.concat(points_by_type.values(), ignore_index=True), crs=aoi.crs) \
        if points_by_type else _empty_gdf(aoi.crs)
    all_lines = gpd.GeoDataFrame(pd.concat(lines_by_type.values(), ignore_index=True), crs=aoi.crs) \
        if lines_by_type else _empty_gdf(aoi.crs)

    exposure_results: Dict[str, Dict[str, gpd.GeoDataFrame]] = {}

    for hazard_name, hazard_conf in config.get("hazards", {}).items():
        # Skip inactive hazards and those handled by overlay modules elsewhere
        if not hazard_conf.get("active", False):
            continue
        if hazard_name in ["drought", "heat", "wildfire"]:
            # These hazards are handled by dedicated modules; we don't compute exposure here.
            continue

        # ---------------------------------------------
        # Special case: earthquake → clip & map only
        # ---------------------------------------------
        if hazard_name == "earthquake":
            print(f"\n--- Clipping and saving earthquake raster ---")
            raster_path = hazard_conf["input"]
            raster_path_wgs84 = assign_or_reproject_to_wgs84(raster_path)

            # Clip by AOI
            with rasterio.open(raster_path_wgs84) as src:
                geoms = [f["geometry"] for f in aoi.to_crs(src.crs).__geo_interface__["features"]]
                clipped, transform = mask(src, geoms, crop=True)
                meta = src.meta.copy()
                meta.update({"height": clipped.shape[1], "width": clipped.shape[2], "transform": transform})

            out_path = os.path.join(config["output_dir"], "earthquake_clipped.tif")
            Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(clipped)

            # Plot (no exposure features for earthquake here)
            plot_and_save_exposure_map(
                aoi=aoi,
                points=_empty_gdf(aoi.crs),
                lines=_empty_gdf(aoi.crs),
                hazard_name=hazard_name,
                output_dir=config["output_dir"],
                raster_path=out_path
            )

            # Normalized result with empty GDFs
            exposure_results[hazard_name] = {
                "points_exposed": _ensure_exposed_bool(_empty_gdf(aoi.crs)),
                "lines_exposed":  _ensure_exposed_bool(_empty_gdf(aoi.crs)),
                "raster_path": out_path,
                "threshold": None
            }
            continue

        # ---------------------------------------------
        # Standard raster-based hazards (e.g., pluvial_flood, fluvial_flood, landslide)
        # ---------------------------------------------
        raster_path = hazard_conf["input"]
        threshold = hazard_conf.get("threshold", None)
        raster_path_wgs84 = assign_or_reproject_to_wgs84(raster_path)

        with rasterio.open(raster_path_wgs84) as raster:
            # Points exposure (function is expected to add 'exposed' boolean)
            points_exposed = extract_values_to_points(all_points.copy(), raster, threshold)
            points_exposed = _ensure_exposed_bool(points_exposed)

            # Lines exposure (we explicitly create 'exposed' boolean)
            lines_exposed = all_lines.copy()
            lines_exposed["exposed"] = lines_exposed["geometry"].apply(
                lambda geom: check_line_exposure(geom, raster, sample_points_per_line, threshold)
            )
            lines_exposed = _ensure_exposed_bool(lines_exposed)

            # Save shapefiles (same behavior as before)
            Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
            points_out = os.path.join(config["output_dir"], f"points_exposure_{hazard_name}.shp")
            lines_out  = os.path.join(config["output_dir"], f"lines_exposure_{hazard_name}.shp")
            points_exposed.to_file(points_out)
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

            # ---- Normalized return payload for stats & downstream steps ----
            exposure_results[hazard_name] = {
                "points_exposed": points_exposed,
                "lines_exposed":  lines_exposed,
                "raster_path":    raster_path_wgs84,
                "threshold":      threshold,
                # Optional: keep file paths if someone needs them later
                "outputs": {
                    "points_path": points_out,
                    "lines_path":  lines_out
                }
            }

    return exposure_results
