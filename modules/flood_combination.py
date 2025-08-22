# modules/flood_combination.py
"""
Flood combination module.

This module combines fluvial and pluvial flood rasters into a single **binary**
raster using per-hazard thresholds with a logical OR:
    exposed := (pluvial >= thr_pluvial) OR (fluvial >= thr_fluvial)

It then computes exposure flags for ALL infrastructure layers (points & lines)
and writes audit shapefiles + a map. The map is rendered here (no dependency
on external plotting quirks), so assets always appear even when none is exposed.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling

from modules.exposure_utils import extract_values_to_points, check_line_exposure
from modules.plotting import plot_and_save_exposure_map


def _ensure_infra_type(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    """Ensure 'infra_type' column exists for plotting styles."""
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = gdf.copy()
    if "infra_type" not in gdf.columns:
        if "type" in gdf.columns:
            gdf = gdf.rename(columns={"type": "infra_type"})
        else:
            gdf["infra_type"] = "infra"
    return gdf

def _reproject_raster_to_crs(in_path: Path, out_path: Path, dst_crs) -> Path:
    """Reproject a single-band uint8 raster to dst_crs (nearest)."""
    with rasterio.open(in_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": rasterio.uint8,
            "nodata": 0,
            "count": 1,
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
    return out_path


def _align_to_ref(src_ds: rasterio.io.DatasetReader,
                  ref_ds: rasterio.io.DatasetReader):
    """
    Reproject src_ds band-1 to the grid of ref_ds; return (array, transform, crs).
    """
    if (src_ds.crs == ref_ds.crs and
        src_ds.transform == ref_ds.transform and
        src_ds.width == ref_ds.width and
        src_ds.height == ref_ds.height):
        return src_ds.read(1), ref_ds.transform, ref_ds.crs

    dst = np.empty((ref_ds.height, ref_ds.width), dtype=src_ds.dtypes[0])
    reproject(
        source=rasterio.band(src_ds, 1),
        destination=dst,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=ref_ds.transform,
        dst_crs=ref_ds.crs,
        resampling=Resampling.nearest
    )
    return dst, ref_ds.transform, ref_ds.crs


def _plot_combined_map(
    raster_path: Path,
    aoi_plot: gpd.GeoDataFrame,
    points_plot: Optional[gpd.GeoDataFrame],
    lines_plot: Optional[gpd.GeoDataFrame],
    out_png: Path,
    title: str = "Combined Flood Exposure"
):
    """
    Robust plotting:
      - draw basemap (if contextily available)
      - draw AOI boundary
      - draw combined raster mask (0 transparent, 1 blue)
      - draw ALL assets in grey; overlay 'exposed' in red if present
    """
    import matplotlib.pyplot as plt
    from rasterio.plot import show as rio_show
    import rasterio
    import numpy as np

    fig, ax = plt.subplots(figsize=(16, 9))

    # Open raster first to know CRS/extent
    data = None
    transform = None
    r_crs = None
    try:
        with rasterio.open(raster_path) as src:
            band = src.read(1)
            # Make zeros transparent
            data = np.ma.masked_where(band == 0, band)
            transform = src.transform
            r_crs = src.crs
    except Exception as e:
        print(f"[WARN] Cannot open combined raster '{raster_path}': {e}")

    # Basemap (optional): works in any CRS by passing crs=...
    try:
        import contextily as cx
        cx.add_basemap(ax, crs=r_crs if r_crs is not None else aoi_plot.crs,
                       source=cx.providers.CartoDB.Positron, attribution_size=6)
    except Exception as e:
        print(f"[INFO] Basemap skipped (contextily not available or error: {e})")

    # AOI boundary
    try:
        aoi_plot.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=5)
    except Exception as e:
        print(f"[WARN] Cannot plot AOI boundary: {e}")

    # Combined raster in blue, transparent outside mask
    if data is not None and transform is not None:
        try:
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(["none", "#6baed6"])  # 0 -> transparent, 1 -> blue
            rio_show(data, transform=transform, ax=ax, cmap=cmap, alpha=0.55, zorder=6)
        except Exception as e:
            print(f"[WARN] Cannot render combined raster: {e}")

    # Lines base + highlight
    if isinstance(lines_plot, gpd.GeoDataFrame) and len(lines_plot) > 0:
        try:
            lines_plot.plot(ax=ax, linewidth=0.6, color="#9aa0a6", zorder=10)
            if "exposed" in lines_plot.columns:
                exp = lines_plot[lines_plot["exposed"]]
                if len(exp) > 0:
                    exp.plot(ax=ax, linewidth=1.6, color="#d32f2f", zorder=11)
        except Exception as e:
            print(f"[WARN] Cannot plot lines: {e}")

    # Points base + highlight
    if isinstance(points_plot, gpd.GeoDataFrame) and len(points_plot) > 0:
        try:
            points_plot.plot(ax=ax, markersize=10, color="#9aa0a6", zorder=12)
            if "exposed" in points_plot.columns:
                exp = points_plot[points_plot["exposed"]]
                if len(exp) > 0:
                    exp.plot(ax=ax, markersize=28, color="#d32f2f", zorder=13)
        except Exception as e:
            print(f"[WARN] Cannot plot points: {e}")

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_axis_off()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.show()
    plt.close(fig)


def process_combined_flood(
    config: dict,
    hazard_rasters: Dict[str, Tuple[str, Optional[float]]],
    aoi: gpd.GeoDataFrame,
    points: Optional[gpd.GeoDataFrame],
    lines: Optional[gpd.GeoDataFrame],
    sample_points_per_line: int = 10
) -> dict:
    """
    Combine pluvial & fluvial with an OR on their own thresholds,
    reproject the combined raster to AOI CRS (so plotting matches other hazards),
    compute exposure on ALL assets, write shapefiles, then call plot_and_save_exposure_map.
    """
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs & thresholds
    p_path, p_thr = hazard_rasters.get("pluvial_flood", (None, None))
    f_path, f_thr = hazard_rasters.get("fluvial_flood", (None, None))
    if not p_path or not f_path:
        print("[WARN] process_combined_flood: missing pluvial or fluvial raster path.")
        return {}

    p_thr = 0 if p_thr is None else p_thr
    f_thr = 0 if f_thr is None else f_thr

    # 1) Align fluvial to pluvial grid, then OR on thresholds
    with rasterio.open(p_path) as Rref, rasterio.open(f_path) as R2:
        band_p = Rref.read(1)

        # reproject fluvial band to pluvial grid
        dst_f = np.empty((Rref.height, Rref.width), dtype=R2.dtypes[0])
        reproject(
            source=rasterio.band(R2, 1),
            destination=dst_f,
            src_transform=R2.transform,
            src_crs=R2.crs,
            dst_transform=Rref.transform,
            dst_crs=Rref.crs,
            resampling=Resampling.nearest,
        )

        valid = np.ones_like(band_p, dtype=bool)
        valid &= (band_p != Rref.nodata) if Rref.nodata is not None else ~np.isnan(band_p)
        valid &= (dst_f != R2.nodata)    if R2.nodata   is not None else ~np.isnan(dst_f)

        combined = valid & ((band_p >= p_thr) | (dst_f >= f_thr))  # binary mask 0/1

        meta = Rref.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1, nodata=0)
        combined_ref = out_dir / "combined_flood.tif"  # in pluvial CRS
        with rasterio.open(combined_ref, "w", **meta) as dst:
            dst.write(combined.astype(np.uint8), 1)

    # 2) Reproject the combined raster to the AOI CRS (so plotting is identical)
    combined_aoi = out_dir / "combined_flood_aoi_crs.tif"
    _reproject_raster_to_crs(combined_ref, combined_aoi, aoi.crs)

    # 3) Ensure plotting-required columns on vectors
    points = _ensure_infra_type(points)
    lines  = _ensure_infra_type(lines)

    # 4) Compute exposure on ALL assets using the binary raster (threshold=1)
    with rasterio.open(combined_aoi) as rr:
        points_exposed = None
        if isinstance(points, gpd.GeoDataFrame) and len(points) > 0:
            pe = extract_values_to_points(points.copy(), rr, threshold=1)
            if pe is None or len(pe) == 0 or ("exposed" not in pe.columns):
                pe = points.copy(); pe["exposed"] = False
            points_exposed = pe

        lines_exposed = None
        if isinstance(lines, gpd.GeoDataFrame) and len(lines) > 0:
            le = lines.copy()
            le["exposed"] = le.geometry.apply(
                lambda geom: check_line_exposure(geom, rr, sample_points_per_line, threshold=1)
            )
            lines_exposed = le

    # 5) Write shapefiles (stats module will pick them)
    if isinstance(points_exposed, gpd.GeoDataFrame) and len(points_exposed) > 0:
        points_exposed.to_file(out_dir / "points_exposure_combined_flood.shp")
    if isinstance(lines_exposed, gpd.GeoDataFrame) and len(lines_exposed) > 0:
        lines_exposed.to_file(out_dir / "lines_exposure_combined_flood.shp")

    # 6) EXACT SAME plotting as the other hazards
    plot_and_save_exposure_map(
        aoi=aoi,
        points=points_exposed if points_exposed is not None else points,
        lines=lines_exposed   if lines_exposed   is not None else lines,
        hazard_name="combined_flood",
        output_dir=str(out_dir),
        raster_path=str(combined_aoi),
        group_by_type=True
    )

    return {
        "combined_flood": {
            "raster_path": str(combined_aoi),
            "points_exposed": points_exposed,
            "lines_exposed": lines_exposed,
        }
    }
