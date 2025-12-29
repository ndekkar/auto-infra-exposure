"""
Module for computing exposure of infrastructure to raster-based hazards.

This module processes each active raster hazard (excluding drought, heat, and wildfire),_clip_and_plot_raster_only
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
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from affine import Affine

from modules.exposure_utils import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map, plot_initial_map_by_type
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
    If already present, cast to bool. If missing, create it (False / empty).
    """
    if gdf is None or len(gdf) == 0:
        gdf = gdf.copy()
        if "exposed" not in gdf.columns:
            gdf["exposed"] = pd.Series(dtype=bool)
        return gdf

    gdf = gdf.copy()
    if "exposed" in gdf.columns:
        gdf["exposed"] = gdf["exposed"].astype(bool)
    else:
        gdf["exposed"] = pd.Series(dtype=bool, index=gdf.index)
    return gdf


def _clip_and_plot_raster_only(
    hazard_name: str,
    config: dict,
    aoi: gpd.GeoDataFrame,
    points_by_type: dict,
    lines_by_type: dict,
    raster_path: str,
) -> dict:
    """
    Clip raster and display Network.

    Used for hazards like earthquake / heat / cold where we want a raster-only view.
    """

    raster_path_wgs84 = assign_or_reproject_to_wgs84(raster_path)

    # Clip by AOI
    with rasterio.open(raster_path_wgs84) as src:
        aoi_in_raster_crs = aoi.to_crs(src.crs)
        geoms = [f["geometry"] for f in aoi_in_raster_crs.__geo_interface__["features"]]
        clipped, transform = mask(src, geoms, crop=True)
        meta = src.meta.copy()
        meta.update({"height": clipped.shape[1], "width": clipped.shape[2], "transform": transform})

    out_path = os.path.join(config["output_dir"], f"{hazard_name}_clipped.tif")
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)

    all_points_dummy = (
        gpd.GeoDataFrame(pd.concat(points_by_type.values(), ignore_index=True), crs=aoi.crs)
        if points_by_type else _empty_gdf(aoi.crs)
    )
    all_lines_dummy = (
        gpd.GeoDataFrame(pd.concat(lines_by_type.values(), ignore_index=True), crs=aoi.crs)
        if lines_by_type else _empty_gdf(aoi.crs)
    )

    # Plot using helper, without exposure classification

    output_path = Path(config["output_dir"]) / f"{hazard_name}_initial.png"
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    plot_initial_map_by_type(
        aoi=aoi,
        points_by_type=points_by_type,
        lines_by_type=lines_by_type,
        output_path=output_path,     
        raster_path=out_path,
        hazard_name=hazard_name,     
    )



    # Return an empty exposure GDF but keep the raster path for potential overlay statistics
    return {
        "points_exposed": _ensure_exposed_bool(_empty_gdf(aoi.crs)),
        "lines_exposed": _ensure_exposed_bool(_empty_gdf(aoi.crs)),
        "raster_path": out_path,
        "threshold": None,
    }


def _estimate_aoi_area_km2(aoi: gpd.GeoDataFrame) -> float:
    """
    Estimate AOI area in km² using an equal-area projection.
    If anything fails, return None.
    """
    try:
        aoi_eq = aoi.to_crs("EPSG:6933")  # equal-area projection
        area_m2 = aoi_eq.geometry.area.sum()
        return float(area_m2) / 1e6
    except Exception as exc:  # defensive
        print(f"[WARN] Could not estimate AOI area: {exc}")
        return None


def _build_lowres_raster_for_plot(
    raster_path_wgs84: str,
    aoi: gpd.GeoDataFrame,
    out_dir: str,
    hazard_name: str,
    max_pixels: int = 5_000_000,
) -> str:
    """
    Create a temporary low-resolution raster clipped to the AOI, for plotting only.

    This avoids loading a huge full-resolution raster (e.g. Kazakhstan) just for the PNG.

    max_pixels: maximum number of pixels (width * height) for the plotted raster.
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"{hazard_name}_plot_lowres.tif")

    with rasterio.open(raster_path_wgs84) as src:
        # Reproject AOI to raster CRS, work with bounding box (lighter than full mask)
        aoi_raster_crs = aoi.to_crs(src.crs)
        minx, miny, maxx, maxy = aoi_raster_crs.total_bounds

        # Build a window covering the AOI
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        window = window.round_offsets().round_lengths()

        # Original window size
        win_width = int(window.width)
        win_height = int(window.height)
        if win_width <= 0 or win_height <= 0:
            # Fallback: use full raster extent
            print(f"[WARN] AOI window has non-positive size for {hazard_name}; using full raster extent.")
            window = None
            win_width = src.width
            win_height = src.height

        # Determine scaling factor to keep number of pixels <= max_pixels
        n_pixels = win_width * win_height
        if n_pixels <= 0:
            scale = 1.0
        elif n_pixels <= max_pixels:
            scale = 1.0
        else:
            # isotropic scaling factor
            scale = (n_pixels / max_pixels) ** 0.5

        # Compute output height/width
        out_width = max(1, int(win_width / scale))
        out_height = max(1, int(win_height / scale))

        # Read & resample window in one go
        data = src.read(
            out_shape=(src.count, out_height, out_width),
            window=window,
            resampling=Resampling.average,
        )

        # Compute new transform
        if window is not None:
            base_transform = src.window_transform(window)
        else:
            base_transform = src.transform

        scale_x = win_width / out_width
        scale_y = win_height / out_height
        new_transform = base_transform * Affine.scale(scale_x, scale_y)

        meta = src.meta.copy()
        meta.update(
            {
                "height": out_height,
                "width": out_width,
                "transform": new_transform,
            }
        )

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)

    print(
        f"[INFO] Built low-res raster for plotting '{hazard_name}': "
        f"{out_width}x{out_height} pixels (original window {win_width}x{win_height})."
    )
    return out_path


# -----------------------------
# Main function
# -----------------------------
def process_raster_exposures(
    config,
    aoi: gpd.GeoDataFrame,
    points_by_type: Dict[str, gpd.GeoDataFrame],
    lines_by_type: Dict[str, gpd.GeoDataFrame],
    sample_points_per_line: int,
) -> Dict[str, Dict[str, gpd.GeoDataFrame]]:
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
    """

    # Tag each infrastructure point and line with its type (preserve the user's fields)
    for infra_type, gdf in points_by_type.items():
        gdf["infra_type"] = infra_type
    for infra_type, gdf in lines_by_type.items():
        gdf["infra_type"] = infra_type

    # Merge all points and lines into unified GeoDataFrames
    all_points = (
        gpd.GeoDataFrame(pd.concat(points_by_type.values(), ignore_index=True), crs=aoi.crs)
        if points_by_type else _empty_gdf(aoi.crs)
    )
    all_lines = (
        gpd.GeoDataFrame(pd.concat(lines_by_type.values(), ignore_index=True), crs=aoi.crs)
        if lines_by_type else _empty_gdf(aoi.crs)
    )

    # Estimate AOI size for plotting strategy
    aoi_area_km2 = _estimate_aoi_area_km2(aoi)
    LARGE_AOI_THRESHOLD_KM2 = 1_000_000.0  # threshold for "huge" AOIs (e.g. Kazakhstan)
    use_lowres_for_plot = False
    if aoi_area_km2 is not None:
        if aoi_area_km2 > LARGE_AOI_THRESHOLD_KM2:
            use_lowres_for_plot = True
            print(
                f"[INFO] AOI area ≈ {aoi_area_km2:,.0f} km² (> {LARGE_AOI_THRESHOLD_KM2:,.0f}); "
                "will use a low-resolution raster for plotting to save memory."
            )
        else:
            print(
                f"[INFO] AOI area ≈ {aoi_area_km2:,.0f} km²; using full-resolution raster for plotting."
            )
    else:
        print("[INFO] AOI area unknown; defaulting to full-resolution plotting.")

    exposure_results: Dict[str, Dict[str, gpd.GeoDataFrame]] = {}

    for hazard_name, hazard_conf in config.get("hazards", {}).items():
        # Skip inactive hazards and those handled by overlay modules elsewhere
        if not hazard_conf.get("active", False):
            continue
        if hazard_name in ["drought", "wildfire"]:
            # These hazards are handled by dedicated modules; we don't compute exposure here.
            continue

        # ---------------------------------------------
        # Special case: earthquake + heat + cold → clip & map only
        # ---------------------------------------------
        RASTER_ONLY = {"earthquake", "heat", "cold"}
        if hazard_name in RASTER_ONLY:
            raster_path = hazard_conf["input"]
            exposure_results[hazard_name] = _clip_and_plot_raster_only(
                hazard_name=hazard_name,
                config=config,
                aoi=aoi,
                points_by_type=points_by_type,
                lines_by_type=lines_by_type,
                raster_path=raster_path,
            )
            continue

        # ---------------------------------------------
        # Standard raster-based hazards (e.g., pluvial_flood, fluvial_flood, landslide)
        # ---------------------------------------------
        raster_path = hazard_conf["input"]
        threshold = hazard_conf.get("threshold", None)
        raster_path_wgs84 = assign_or_reproject_to_wgs84(raster_path)

        # ---- Exposure computation on FULL-RES raster ----
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

        # Save shapefiles
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        points_out = os.path.join(config["output_dir"], f"points_exposure_{hazard_name}.shp")
        lines_out = os.path.join(config["output_dir"], f"lines_exposure_{hazard_name}.shp")
        points_exposed.to_file(points_out)
        lines_exposed.to_file(lines_out)

        # ---- Choose raster path for plotting: full-res or low-res ----
        if use_lowres_for_plot:
            raster_path_for_plot = _build_lowres_raster_for_plot(
                raster_path_wgs84=raster_path_wgs84,
                aoi=aoi,
                out_dir=config["output_dir"],
                hazard_name=hazard_name,
                max_pixels=5_000_000,  # tu peux ajuster si besoin
            )
        else:
            raster_path_for_plot = raster_path_wgs84

        # PNG avec fond raster (low-res si AOI énorme)
        plot_and_save_exposure_map(
            aoi=aoi,
            points=points_exposed,
            lines=lines_exposed,
            hazard_name=hazard_name,
            output_dir=config["output_dir"],
            raster_path=raster_path_for_plot,
        )

        # ---- Normalized return payload for stats & downstream steps ----
        exposure_results[hazard_name] = {
            "points_exposed": points_exposed,
            "lines_exposed": lines_exposed,
            "raster_path": raster_path_wgs84,  # on garde le full-res comme référence
            "threshold": threshold,
            "outputs": {
                "points_path": points_out,
                "lines_path": lines_out,
            },
        }

    return exposure_results
