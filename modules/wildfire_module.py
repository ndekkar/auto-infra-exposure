"""
Wildfire processing: build a raster from burned-area centroids and plot
ONE map with the raster underlay + energy network by type.

Behavior:
- Extract centroids within AOI (from GlobFire polygons)
- Rasterize centroids to a TIF
- Plot a single map: wildfire TIF + network (points/lines by type) + basemap
"""

import os
from typing import Dict, Optional

import geopandas as gpd
from shapely.geometry import Point

from modules.plotting import plot_initial_map_by_type


# -------------------------------
# Public entry
# -------------------------------
def process_wildfire(config) -> Optional[str]:
    """
    Main entry point for wildfire.
    - Builds wildfire centroids (GPKG) and a raster (TIF)
    - Plots a single map with the raster + network by type
    - Returns the raster path (or None on failure)
    """
    wf = (config.get("hazards", {}).get("wildfire", {}) or {})
    if not wf.get("active", False):
        print("[INFO] Wildfire inactive; skipping.")
        return None

    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    aoi_path = config["aoi"]
    gpkg_centroids = os.path.join(out_dir, "wildfire_centroids.gpkg")
    tif_raster = os.path.join(out_dir, "wildfire_centroids.tif")
    png_map = os.path.join(out_dir, "wildfire_w_network.png")

    # 1) Extract centroids in AOI
    try:
        extract_burned_area_centroids(
            aoi_path=aoi_path,
            globfire_dir=wf["input"],
            output_path=gpkg_centroids
        )
    except Exception as e:
        print(f"[ERROR] Wildfire centroids failed: {e}")
        return None

    # 2) Rasterize centroids → TIF
    try:
        rasterize_fire_centroids(gpkg_centroids, tif_raster, resolution=0.01)
    except Exception as e:
        print(f"[ERROR] Wildfire rasterization failed: {e}")
        return None

    # 3) Prepare AOI and network dicts (points_by_type / lines_by_type)
    try:
        aoi_gdf = gpd.read_file(aoi_path)
        aoi_gdf = aoi_gdf.to_crs(aoi_gdf.crs or "EPSG:3857")  # ensure CRS set

        points_by_type = config.get("_points_by_type")
        lines_by_type  = config.get("_lines_by_type")

        # Fallback: rebuild dicts from configured inputs if cache not present
        if not points_by_type or not lines_by_type:
            points_by_type, lines_by_type = _load_network_from_config_inputs(config)

        # Reproject network dicts to AOI CRS (critical for display)
        target_crs = aoi_gdf.crs
        points_by_type = _reproj_dict(points_by_type, target_crs)
        lines_by_type  = _reproj_dict(lines_by_type, target_crs)
    except Exception as e:
        print(f"[WARN] Wildfire: could not prep network dicts: {e}")
        points_by_type, lines_by_type = {}, {}

    # 4) Plot ONE map: raster + network by type
    try:
        plot_initial_map_by_type(
            aoi=aoi_gdf,
            points_by_type=points_by_type or {},
            lines_by_type=lines_by_type or {},
            output_path=png_map,
            raster_path=tif_raster,
            hazard_name="wildfire"
        )
        print(f"[INFO] Wildfire map written: {png_map}")
    except Exception as e:
        print(f"[WARN] Wildfire map failed: {e}")

    # Return raster path for downstream overlay stats if needed
    return tif_raster


# -------------------------------
# Helpers
# -------------------------------
def _reproj_dict(d: Optional[Dict[str, gpd.GeoDataFrame]], target_crs):
    """Reproject all non-empty GeoDataFrames in a dict to target_crs."""
    out = {}
    if isinstance(d, dict):
        for k, g in d.items():
            if g is None or len(g) == 0:
                continue
            if g.crs is None:
                g = g.set_crs(target_crs, allow_override=True)
            elif g.crs != target_crs:
                g = g.to_crs(target_crs)
            out[k] = g
    return out


def _load_network_from_config_inputs(config):
    """
    Fallback: build points_by_type / lines_by_type from file paths in config:
      config["infrastructure_inputs"]["points"][type] -> path
      config["infrastructure_inputs"]["lines"][type]  -> path
    Adds a 'type' column to each GDF.
    """
    pts_dict, lns_dict = {}, {}
    infra = config.get("infrastructure_inputs", {})
    pts_spec = (infra.get("points") or {})
    lns_spec = (infra.get("lines") or {})

    # Points (e.g., substations, transformer, tower, existing)
    for k, pth in pts_spec.items():
        if not pth:
            continue
        try:
            g = gpd.read_file(pth)
            g["type"] = k
            pts_dict[k] = g
        except Exception as e:
            print(f"[WARN] Cannot read points '{k}' from {pth}: {e}")

    # Lines (e.g., hv, lv, existing)
    for k, pth in lns_spec.items():
        if not pth:
            continue
        try:
            g = gpd.read_file(pth)
            g["type"] = k
            lns_dict[k] = g
        except Exception as e:
            print(f"[WARN] Cannot read lines '{k}' from {pth}: {e}")

    return pts_dict, lns_dict


def extract_burned_area_centroids(aoi_path, globfire_dir, output_path):
    """
    Read burned-area polygons (GlobFire), keep those intersecting the AOI (buffered),
    and write their centroids as a GeoPackage (layer='burned_area_centroids').
    """
    aoi = gpd.read_file(aoi_path).to_crs(epsg=3857)
    aoi_buffered = aoi.buffer(10000)  # 10 km buffer
    aoi_geom = aoi_buffered.unary_union

    all_pts = []
    for fname in os.listdir(globfire_dir):
        if not fname.lower().endswith(".shp"):
            continue
        shp_path = os.path.join(globfire_dir, fname)
        try:
            gdf = gpd.read_file(shp_path).to_crs(epsg=3857)
            filtered = gdf[gdf.intersects(aoi_geom)]
            centroids = filtered.centroid.to_crs(epsg=4326)
            all_pts.extend([Point(pt.x, pt.y) for pt in centroids])
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")

    if not all_pts:
        print("[INFO] No burned area centroids found.")
        # still write empty layer so pipeline is consistent
        gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(
            output_path, driver="GPKG", layer="burned_area_centroids"
        )
        return

    out_gdf = gpd.GeoDataFrame(geometry=all_pts, crs="EPSG:4326")
    out_gdf.to_file(output_path, driver="GPKG", layer="burned_area_centroids")


def rasterize_fire_centroids(gpkg_path, output_tif_path, resolution=0.01):
    """
    Rasterize centroid points (1 = burned, 0 = background) at a given degree resolution.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    gdf = gpd.read_file(gpkg_path, layer="burned_area_centroids").to_crs("EPSG:4326")
    if gdf.empty:
        # write an empty raster with tiny extent to avoid crashes (optional)
        raise ValueError("No wildfire centroids to rasterize.")

    minx, miny, maxx, maxy = gdf.total_bounds
    width = max(1, int((maxx - minx) / resolution))
    height = max(1, int((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    shapes = [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty]
    if not shapes:
        raise ValueError("No valid centroid geometries to rasterize.")

    arr = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    with rasterio.open(
        output_tif_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        compress="LZW"
    ) as dst:
        dst.write(arr, 1)

    return output_tif_path
