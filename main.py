"""
Main script to run the multi-hazard exposure pipeline (robust to tuple/dict returns).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import geopandas as gpd
import pandas as pd
from modules.config_utils import load_config
from modules.crs_utils import harmonize_crs
from modules.plotting import plot_initial_map_by_type
from modules.raster_exposure import process_raster_exposures
from modules.flood_combination import process_combined_flood
# from modules.drought_module import process_drought
from modules.heat_module import process_heat
from modules.wildfire_module import process_wildfire
from modules.cold_module import process_cold
from modules.stats import (
    compute_infra_stats_from_results,
    compute_infra_stats_from_overlay,
)

# ------------------------------
# Helpers
# ------------------------------
def _is_active(config: dict, hazard_key: str) -> bool:
    try:
        return bool((config.get("hazards", {}).get(hazard_key, {}) or {}).get("active", False))
    except Exception:
        return False

def _dedupe_points_per_type(gdf: gpd.GeoDataFrame, decimals: int = 6) -> gpd.GeoDataFrame:
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = gdf.copy()
    if "type" not in gdf.columns:
        gdf["type"] = None
    gdf["_x"] = gdf.geometry.x.round(decimals)
    gdf["_y"] = gdf.geometry.y.round(decimals)
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["_x", "_y", "type"]).drop(columns=["_x", "_y"])
    removed = before - len(gdf)
    if removed > 0:
        print(f"[DEDUPE] points: removed {removed} duplicate feature(s)")
    return gdf

def _read_if_exists(path: Path) -> Optional[gpd.GeoDataFrame]:
    try:
        return gpd.read_file(path) if path.exists() else None
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None

def _extract_raster_threshold(entry: Any) -> Tuple[Optional[str], Optional[float]]:
    """
    Accepts:
      - dict with keys 'raster_path'/'path' and 'threshold'
      - tuple/list like (raster_path, threshold) or (raster_path,)
      - string path
    Returns (raster_path, threshold)
    """
    rp, th = None, None
    if isinstance(entry, dict):
        rp = entry.get("raster_path") or entry.get("path") or entry.get("output") or entry.get("raster")
        th = entry.get("threshold")
    elif isinstance(entry, (list, tuple)):
        if len(entry) >= 1:
            rp = entry[0]
        if len(entry) >= 2:
            th = entry[1]
    elif isinstance(entry, (str, Path)):
        rp = str(entry)
    return (str(rp) if rp else None, th)

def _stats_from_outputs_or_overlay(
    hz: str,
    config: dict,
    aoi: gpd.GeoDataFrame,
    all_points: Optional[gpd.GeoDataFrame],
    all_lines: Optional[gpd.GeoDataFrame],
    points_by_type: Dict[str, gpd.GeoDataFrame],
    lines_by_type: Dict[str, gpd.GeoDataFrame],
    hazard_results: Dict[str, Any],
) -> None:
    """
    Preferred order:
      1) If shapefiles written by exposure step exist -> read them and compute stats.
      2) Else if hazard_results contains GDFs (rare in your current setup) -> compute stats.
      3) Else if we can find a raster path -> do overlay stats.
      4) Else warn.
    """
    outdir = Path(config["output_dir"])
    shp_points = outdir / f"points_exposure_{hz}.shp"
    shp_lines  = outdir / f"lines_exposure_{hz}.shp"

    pts_gdf = _read_if_exists(shp_points)
    ln_gdf  = _read_if_exists(shp_lines)

    if (pts_gdf is not None and len(pts_gdf) > 0) or (ln_gdf is not None and len(ln_gdf) > 0):
        compute_infra_stats_from_results(
            hazard_name=hz,
            aoi_gdf=aoi,
            points_exposed_gdf=pts_gdf,
            lines_exposed_gdf=ln_gdf,
            out_root=os.path.join(config["output_dir"], "stats")
        )
        print(f"[INFO] Stats (from shapefiles) written for {hz}")
        return

    # Attempt to find a raster path in hazard_results or config
    rp, _ = _extract_raster_threshold(hazard_results.get(hz)) if isinstance(hazard_results, dict) else (None, None)
    if not rp:
        rp = (config.get("hazards", {}).get(hz, {}) or {}).get("input")

    if rp and Path(str(rp)).exists():
        compute_infra_stats_from_overlay(
            hazard_name=hz,
            raster_path=rp,
            aoi_gdf=aoi,
            all_points_gdf=all_points,
            all_lines_gdf=all_lines,
            points_by_type=points_by_type,
            lines_by_type=lines_by_type,
            out_root=os.path.join(config["output_dir"], "stats")
        )
        print(f"[INFO] Overlay stats written for {hz}")
    else:
        print(f"[WARN] {hz}: no shapefiles, no GDFs, and no raster found for stats.")

# ------------------------------
# Main
# ------------------------------
def run_multi_hazard_pipeline(config_path: str):
    config = load_config(config_path)

    # AOI
    aoi = gpd.read_file(config["aoi"])
    aoi_union = aoi.union_all()

    # Infrastructure
    points_by_type: Dict[str, gpd.GeoDataFrame] = {}
    lines_by_type: Dict[str, gpd.GeoDataFrame] = {}

    for name, path in config["infrastructure_inputs"]["points"].items():
        if path is not None:
            points_by_type[name] = gpd.read_file(path)

    for name, path in config["infrastructure_inputs"]["lines"].items():
        if path is not None:
            lines_by_type[name] = gpd.read_file(path)

    # Harmonize, clip, tag, dedupe (points)
    for name, gdf in points_by_type.items():
        gdf, = harmonize_crs([gdf], aoi.crs)
        gdf = gdf[gdf.geometry.within(aoi_union)]
        gdf["type"] = name
        gdf = _dedupe_points_per_type(gdf)
        points_by_type[name] = gdf

    # Lines
    for name, gdf in lines_by_type.items():
        gdf, = harmonize_crs([gdf], aoi.crs)
        gdf = gdf[gdf.geometry.intersects(aoi_union)]
        gdf["type"] = name
        lines_by_type[name] = gdf
    
    config["_points_by_type"] = points_by_type
    config["_lines_by_type"]  = lines_by_type
    # Union layers
    all_points = gpd.GeoDataFrame(pd.concat(points_by_type.values(), ignore_index=True), crs=aoi.crs) if points_by_type else None
    all_lines  = gpd.GeoDataFrame(pd.concat(lines_by_type.values(),  ignore_index=True), crs=aoi.crs) if lines_by_type else None

    # Map
    plot_initial_map_by_type(
        aoi,
        points_by_type=points_by_type,
        lines_by_type=lines_by_type,
        output_path=os.path.join(config["output_dir"], "initial_context_map.png")
    )

    # Non-raster overlays if active
    if _is_active(config, "heat"):
        heat_raster_path = process_heat(config)
        if heat_raster_path and Path(heat_raster_path).exists():
            compute_infra_stats_from_overlay(
                hazard_name="heat",
                raster_path=heat_raster_path,
                aoi_gdf=aoi,
                all_points_gdf=all_points,
                all_lines_gdf=all_lines,
                points_by_type=points_by_type,
                lines_by_type=lines_by_type,
                out_root=os.path.join(config["output_dir"], "stats")
            )
            config["hazards"]["heat"]["input"] = str(heat_raster_path)
    
    if _is_active(config, "wildfire"):
        wildfire_raster_path = process_wildfire(config)
        if wildfire_raster_path and Path(wildfire_raster_path).exists():
            compute_infra_stats_from_overlay(
                hazard_name="wildfire",
                raster_path=wildfire_raster_path,
                aoi_gdf=aoi,
                all_points_gdf=all_points,
                all_lines_gdf=all_lines,
                points_by_type=points_by_type,
                lines_by_type=lines_by_type,
                out_root=os.path.join(config["output_dir"], "stats")
            )


       
    if _is_active(config, "cold"):
        cold_outputs = process_cold(config)  
        cold_raster_path = (config.get("hazards", {}).get("cold", {}) or {}).get("input")
        if cold_raster_path and Path(cold_raster_path).exists():
            compute_infra_stats_from_overlay(
                hazard_name="cold",
                raster_path=cold_raster_path,
                aoi_gdf=aoi,
                all_points_gdf=all_points,
                all_lines_gdf=all_lines,
                points_by_type=points_by_type,
                lines_by_type=lines_by_type,
                out_root=os.path.join(config["output_dir"], "stats")
            )
    
    if _is_active(config, "earthquake"):
        eq_path = (config.get("hazards", {}).get("earthquake", {}) or {}).get("input")
        if eq_path and Path(eq_path).exists():
            compute_infra_stats_from_overlay(
                hazard_name="earthquake",
                raster_path=eq_path,
                aoi_gdf=aoi,
                all_points_gdf=all_points,
                all_lines_gdf=all_lines,
                points_by_type=points_by_type,
                lines_by_type=lines_by_type,
                out_root=os.path.join(config["output_dir"], "stats")
            )

    # Raster exposure hazards
    sample_points_per_line = 10
    hazard_results = process_raster_exposures(
        config=config,
        aoi=aoi,
        points_by_type=points_by_type,
        lines_by_type=lines_by_type,
        sample_points_per_line=sample_points_per_line
    )
    print(f"[DEBUG] process_raster_exposures keys: {list(hazard_results.keys()) if isinstance(hazard_results, dict) else type(hazard_results)}")

    # Stats for flood/landslide (from shapefiles or overlay fallback)
    for hz in ("pluvial_flood", "fluvial_flood", "landslide"):
        if _is_active(config, hz):
            _stats_from_outputs_or_overlay(
                hz=hz,
                config=config,
                aoi=aoi,
                all_points=all_points,
                all_lines=all_lines,
                points_by_type=points_by_type,
                lines_by_type=lines_by_type,
                hazard_results=hazard_results if isinstance(hazard_results, dict) else {},
            )
        else:
            print(f"[INFO] {hz} inactive; stats skipped.")

    # Combined flood (requires active pluvial + fluvial)
    if _is_active(config, "pluvial_flood") and _is_active(config, "fluvial_flood"):
        # Build dict {hz: (raster_path, threshold)} robustly from hazard_results or config
        combined_inputs: Dict[str, Tuple[str, Optional[float]]] = {}
        for hz in ("pluvial_flood", "fluvial_flood"):
            rp, th = (None, None)
            if isinstance(hazard_results, dict) and hz in hazard_results:
                rp, th = _extract_raster_threshold(hazard_results[hz])
            if not rp:
                # fallback to config input
                rp = (config.get("hazards", {}).get(hz, {}) or {}).get("input")
                th = (config.get("hazards", {}).get(hz, {}) or {}).get("threshold")
            if rp and Path(str(rp)).exists():
                combined_inputs[hz] = (rp, th)

        if len(combined_inputs) == 2:
            combined_output = process_combined_flood(
                config=config,
                hazard_rasters=combined_inputs,
                aoi=aoi,
                points=all_points,
                lines=all_lines,
                sample_points_per_line=sample_points_per_line
            )

            # Try stats from shapefiles written by the combined module
            outdir = Path(config["output_dir"])
            c_pts = _read_if_exists(outdir / "points_exposure_combined_flood.shp")
            c_lin = _read_if_exists(outdir / "lines_exposure_combined_flood.shp")

            if (c_pts is not None and len(c_pts) > 0) or (c_lin is not None and len(c_lin) > 0):
                compute_infra_stats_from_results(
                    hazard_name="combined_flood",
                    aoi_gdf=aoi,
                    points_exposed_gdf=c_pts,
                    lines_exposed_gdf=c_lin,
                    out_root=os.path.join(config["output_dir"], "stats")
                )
                print("[INFO] Stats written for combined_flood (from shapefiles).")
            else:
                # Try to pull a raster path from the combined output for overlay fallback
                c_raster = None
                if isinstance(combined_output, dict):
                    cobj = combined_output.get("combined_flood") if "combined_flood" in combined_output else combined_output
                    if isinstance(cobj, dict):
                        c_raster = cobj.get("raster_path") or cobj.get("output") or cobj.get("raster")

                if c_raster and Path(str(c_raster)).exists():
                    compute_infra_stats_from_overlay(
                        hazard_name="combined_flood",
                        raster_path=c_raster,
                        aoi_gdf=aoi,
                        all_points_gdf=all_points,
                        all_lines_gdf=all_lines,
                        points_by_type=points_by_type,
                        lines_by_type=lines_by_type,
                        out_root=os.path.join(config["output_dir"], "stats")
                    )
                    print("[INFO] Overlay stats written for combined_flood.")
                else:
                    print("[WARN] Combined flood: no exposure outputs or raster available for stats.")
        else:
            print("[WARN] Combined flood skipped: missing pluvial/fluvial raster inputs.")
    else:
        print("[INFO] Combined flood requires pluvial_flood AND fluvial_flood active; skipped.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/config.yaml")
        sys.exit(1)
    run_multi_hazard_pipeline(sys.argv[1])
