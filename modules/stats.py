# modules/stats.py
# One-stop infrastructure stats module (robust against missing rasters).
# - compute_infra_stats_from_results(): consume your existing exposed GDFs (NO raster work).
# - compute_infra_stats_from_overlay(): lightweight AOI-masked thresholding ONLY IF a valid raster is provided;
#   otherwise it gracefully skips (but still writes the ONEFILE with NaNs for exposure).
# - OUTPUT POLICY (your request): keep ONLY a single tidy stats file per hazard:
#     <hazard>_stats_onefile.csv
#   -> all other per-hazard stats CSVs are no longer written.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

# ----------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------

EXPOSED_COL_CANDIDATES = [
    "exposed",
    "is_exposed",
    "isExposed",
    "Exposure",
    "exposure",
    "exp",
    "flag_exposed",
    "haz_exposed",
]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_percent(n: float, d: float) -> float:
    return float(n) / float(d) * 100.0 if d else 0.0


def _metric_crs_from_aoi(aoi_gdf: gpd.GeoDataFrame, fallback_crs=None):
    """
    Best-effort choice of a metric CRS for length computations:
    - If AOI is in a projected CRS, use it.
    - Else try to use UTM based on AOI centroid (if in EPSG:4326).
    - Else fallback to provided CRS.
    """
    try:
        if aoi_gdf is not None and aoi_gdf.crs is not None:
            # If projected, assume meters
            if aoi_gdf.crs.is_projected:
                return aoi_gdf.crs

            # If geographic, derive UTM zone from centroid (works best for EPSG:4326)
            aoi_ll = aoi_gdf.to_crs(4326)
            c = aoi_ll.unary_union.centroid
            lon, lat = float(c.x), float(c.y)
            zone = int((lon + 180) // 6) + 1
            epsg = 32600 + zone if lat >= 0 else 32700 + zone
            return epsg
    except Exception:
        pass

    return fallback_crs


def _clip_to_aoi(gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clip GDF to AOI (keeps only features intersecting AOI).
    """
    if gdf is None or len(gdf) == 0:
        return gdf
    if aoi_gdf is None or len(aoi_gdf) == 0:
        return gdf

    try:
        if gdf.crs != aoi_gdf.crs:
            aoi = aoi_gdf.to_crs(gdf.crs)
        else:
            aoi = aoi_gdf
        return gpd.clip(gdf, aoi)
    except Exception:
        # Fallback to intersects filter if clip fails
        try:
            if gdf.crs != aoi_gdf.crs:
                aoi = aoi_gdf.to_crs(gdf.crs)
            else:
                aoi = aoi_gdf
            geom = aoi.unary_union
            return gdf[gdf.intersects(geom)]
        except Exception:
            return gdf


def _normalize_exposed_col(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure there is a boolean column named 'exposed' in the GeoDataFrame.
    If a known exposure flag exists, coerce it to boolean.
    Otherwise, try to derive it from common class/level/value columns.
    As a last resort, mark all features as exposed (since these layers
    typically come from '*_exposed' exports).
    """
    if gdf is None or len(gdf) == 0:
        return gdf

    # 1) Use an existing exposure-like column if present
    for col in EXPOSED_COL_CANDIDATES:
        if col in gdf.columns:
            # Coerce to boolean robustly (handles strings/numbers)
            gdf["exposed"] = (
                gdf[col]
                .replace({"True": True, "False": False, "true": True, "false": False})
                .apply(
                    lambda v: bool(int(v))
                    if isinstance(v, (np.integer, int, np.int64, np.int32, np.int16))
                    else bool(v)
                )
            )
            return gdf

    # 2) Try class/level categorical columns (any non-null category counts as exposed)
    class_like = [
        c
        for c in [
            "class",
            "hazard_class",
            "risk_class",
            "flood_class",
            "drought_class",
            "heat_class",
            "landslide_class",
            "category",
            "level",
            "hazard_level",
        ]
        if c in gdf.columns
    ]
    for c in class_like:
        # Treat categories (e.g., 'Low', 'Medium', 'High', 'Very High', 1..4) as exposed when not null and not 'none'
        s = gdf[c].astype(str).str.strip().str.lower()
        gdf["exposed"] = s.notna() & (s != "") & (s != "none") & (s != "nan")
        return gdf

    # 3) Try numeric intensity/value columns (> 0 counts as exposed)
    numeric_like = [
        c
        for c in [
            "value",
            "intensity",
            "depth",
            "mmi",
            "pga",
            "hazard_value",
            "score",
        ]
        if c in gdf.columns
    ]
    for c in numeric_like:
        try:
            v = pd.to_numeric(gdf[c], errors="coerce")
            gdf["exposed"] = v.notna() & (v > 0)
            return gdf
        except Exception:
            pass

    # 4) Last resort: mark all as exposed
    gdf["exposed"] = True
    return gdf


def _lines_total_and_exposed_km(lines: gpd.GeoDataFrame) -> Tuple[float, float]:
    """
    If your pipeline already computed length columns (total_length_m / exposed_length_m), reuse them.
    Otherwise, derive from geometry and the 'exposed' flag.
    """
    if "exposed_length_m" in lines.columns and "total_length_m" in lines.columns:
        total_km = float(lines["total_length_m"].sum()) / 1000.0
        exposed_km = float(lines["exposed_length_m"].sum()) / 1000.0
        return total_km, exposed_km

    total_km = float(lines.length.sum()) / 1000.0
    exposed_km = float(lines.loc[lines.get("exposed", False)].length.sum()) / 1000.0
    return total_km, exposed_km


# ----------------------------------------------------------------------
# Single-file ("onefile") writer (ONLY output we keep)
# ----------------------------------------------------------------------

def _make_onefile_df(
    hazard_name: str,
    points_global: Optional[Tuple[float, float]] = None,
    points_by_type_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    lines_global_km: Optional[Tuple[float, float]] = None,
    lines_by_type_km_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Build a single tidy stats table.

    Columns:
      hazard, geometry(point|line), scope(global|by_type), type, total, exposed, percent, unit(count|km)

    Notes:
      - If 'exposed' is unknown (e.g., overlay skipped), it will be NaN and percent will be NaN.
      - For lines, totals/exposed are expressed in km.
      - For points, totals/exposed are expressed in counts.
    """
    rows: List[dict] = []

    def add_row(geometry: str, scope: str, typ: Optional[str], total: float, exposed: float, unit: str) -> None:
        pct = (
            _safe_percent(exposed, total)
            if (total not in (0, 0.0) and not (np.isnan(exposed) or np.isnan(total)))
            else (np.nan if (np.isnan(exposed) or np.isnan(total)) else 0.0)
        )
        rows.append(
            {
                "hazard": hazard_name,
                "geometry": geometry,
                "scope": scope,
                "type": typ,
                "total": total,
                "exposed": exposed,
                "percent": pct,
                "unit": unit,
            }
        )

    # Points (counts)
    if points_global is not None:
        t, e = points_global
        add_row("point", "global", None, float(t), float(e), "count")
    if points_by_type_stats:
        for typ, (t, e) in points_by_type_stats.items():
            add_row("point", "by_type", str(typ), float(t), float(e), "count")

    # Lines (km)
    if lines_global_km is not None:
        t, e = lines_global_km
        add_row("line", "global", None, float(t), float(e), "km")
    if lines_by_type_km_stats:
        for typ, (t, e) in lines_by_type_km_stats.items():
            add_row("line", "by_type", str(typ), float(t), float(e), "km")

    return pd.DataFrame(
        rows,
        columns=["hazard", "geometry", "scope", "type", "total", "exposed", "percent", "unit"],
    )


def _write_onefile(
    hazard_name: str,
    outdir: Path,
    points_global: Optional[Tuple[float, float]] = None,
    points_by_type_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    lines_global_km: Optional[Tuple[float, float]] = None,
    lines_by_type_km_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Path:
    """Write <hazard>_stats_onefile.csv and return its path."""
    tidy = _make_onefile_df(
        hazard_name=hazard_name,
        points_global=points_global,
        points_by_type_stats=points_by_type_stats,
        lines_global_km=lines_global_km,
        lines_by_type_km_stats=lines_by_type_km_stats,
    )
    out_path = outdir / f"{hazard_name}_stats_onefile.csv"
    tidy.to_csv(out_path, index=False)
    return out_path


# ----------------------------------------------------------------------
# A) Use existing exposure RESULTS (no raster recomputation)
# ----------------------------------------------------------------------

def compute_infra_stats_from_results(
    hazard_name: str,
    aoi_gdf: gpd.GeoDataFrame,
    points_exposed_gdf: Optional[gpd.GeoDataFrame] = None,
    lines_exposed_gdf: Optional[gpd.GeoDataFrame] = None,
    out_root: str = "output/stats"
) -> dict:
    """
    Aggregate infrastructure stats from ALREADY-EXPOSED feature layers.

    Output (under output/stats/<hazard>/):
      <hazard>_stats_onefile.csv    (ONLY file kept)
    """
    outdir = Path(out_root) / hazard_name
    _ensure_dir(outdir)

    points_global: Optional[Tuple[float, float]] = None
    points_by_type_stats: Dict[str, Tuple[float, float]] = {}

    lines_global_km: Optional[Tuple[float, float]] = None
    lines_by_type_km_stats: Dict[str, Tuple[float, float]] = {}

    # Points
    if points_exposed_gdf is not None and len(points_exposed_gdf) > 0:
        pts = _clip_to_aoi(points_exposed_gdf, aoi_gdf).copy()
        if len(pts) > 0:
            pts = _normalize_exposed_col(pts)
            total = float(len(pts))
            exp_n = float(pts["exposed"].sum())
            points_global = (total, exp_n)

            if "type" in pts.columns:
                grp = pts.groupby("type")["exposed"].agg(["sum", "count"]).reset_index()
                for _, r in grp.iterrows():
                    points_by_type_stats[str(r["type"])] = (float(r["count"]), float(r["sum"]))

    # Lines
    if lines_exposed_gdf is not None and len(lines_exposed_gdf) > 0:
        ln = _clip_to_aoi(lines_exposed_gdf, aoi_gdf).copy()
        if len(ln) > 0:
            metric_crs = _metric_crs_from_aoi(aoi_gdf, ln.crs)
            if metric_crs:
                ln = ln.to_crs(metric_crs)
            ln = _normalize_exposed_col(ln)

            total_km, exp_km = _lines_total_and_exposed_km(ln)
            lines_global_km = (float(total_km), float(exp_km))

            if "type" in ln.columns:
                # For lines, compute by type using LENGTH (km), not counts
                for typ, sub in ln.groupby("type"):
                    tk = float(sub.length.sum()) / 1000.0
                    ek = float(sub.loc[sub.get("exposed", False)].length.sum()) / 1000.0
                    lines_by_type_km_stats[str(typ)] = (tk, ek)

    # ---- write single tidy file (pure stats) ----
    onefile_path = _write_onefile(
        hazard_name=hazard_name,
        outdir=outdir,
        points_global=points_global,
        points_by_type_stats=points_by_type_stats if points_by_type_stats else None,
        lines_global_km=lines_global_km,
        lines_by_type_km_stats=lines_by_type_km_stats if lines_by_type_km_stats else None,
    )

    return {"hazard": hazard_name, "onefile": str(onefile_path)}


# ----------------------------------------------------------------------
# Overlay helpers
# ----------------------------------------------------------------------

def _aoi_mask(rds: rasterio.io.DatasetReader, aoi_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Raster mask True inside AOI."""
    if aoi_gdf is None or len(aoi_gdf) == 0:
        return np.ones((rds.height, rds.width), dtype=bool)

    if aoi_gdf.crs != rds.crs:
        aoi = aoi_gdf.to_crs(rds.crs)
    else:
        aoi = aoi_gdf

    return features.geometry_mask(
        [geom.__geo_interface__ for geom in aoi.geometry],
        transform=rds.transform,
        invert=True,
        out_shape=(rds.height, rds.width),
    )


def _auto_is_categorical(vals: np.ndarray, max_unique=20) -> bool:
    if vals.size == 0:
        return False
    sample = vals if vals.size <= 2_000_000 else vals[np.random.choice(vals.size, 2_000_000, replace=False)]
    uniq = np.unique(sample[~np.isnan(sample)])
    return (uniq.size > 0) and (uniq.size <= max_unique) and np.all(np.isclose(uniq, np.round(uniq)))


def _build_exposed_polygons(
    band: np.ndarray,
    valid_mask: np.ndarray,
    transform,
    crs,
    threshold_val: float,
) -> gpd.GeoSeries:
    """Vectorize a binary 'exposed' mask (band >= threshold_val) into polygons."""
    exposed = valid_mask & (band >= threshold_val)
    if not np.any(exposed):
        return gpd.GeoSeries([], crs=crs)
    bin_arr = exposed.astype(np.uint8)
    geoms = []
    for geom, val in shapes(bin_arr, mask=exposed, transform=transform):
        if val == 1:
            geoms.append(shape(geom))
    return gpd.GeoSeries(geoms, crs=crs) if geoms else gpd.GeoSeries([], crs=crs)


# ----------------------------------------------------------------------
# B) Overlay reporting (threshold-based), still ONLY writes ONEFILE
# ----------------------------------------------------------------------

def compute_infra_stats_from_overlay(
    hazard_name: str,
    raster_path: Optional[str],
    aoi_gdf: gpd.GeoDataFrame,
    all_points_gdf: Optional[gpd.GeoDataFrame] = None,
    all_lines_gdf: Optional[gpd.GeoDataFrame] = None,
    points_by_type: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    lines_by_type: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    out_root: str = "output/stats",
) -> dict:
    """
    Minimal 'reporting overlay' for hazards WITHOUT exposure studies.

    It defines an exposed zone from the AOI-masked raster:
      - Continuous rasters: threshold = 90th percentile (P90) of valid AOI pixels.
      - Categorical rasters: threshold = 75th percentile of present integer classes (top quartile classes).

    Output (under output/stats/<hazard>/):
      <hazard>_stats_onefile.csv    (ONLY file kept)

    If raster_path is missing/invalid, we still write the onefile with totals,
    but exposure-related fields are NaN.
    """
    outdir = Path(out_root) / hazard_name
    _ensure_dir(outdir)

    metric_crs = _metric_crs_from_aoi(
        aoi_gdf,
        (
            all_points_gdf.crs
            if all_points_gdf is not None
            else (all_lines_gdf.crs if all_lines_gdf is not None else None)
        ),
    )

    points_global: Optional[Tuple[float, float]] = None
    points_by_type_stats: Dict[str, Tuple[float, float]] = {}

    lines_global_km: Optional[Tuple[float, float]] = None
    lines_by_type_km_stats: Dict[str, Tuple[float, float]] = {}

    # --- If raster missing/invalid: totals only, exposure unknown (NaN) ---
    if not raster_path or not Path(raster_path).exists():
        # Points totals
        if all_points_gdf is not None and len(all_points_gdf) > 0:
            pts = _clip_to_aoi(all_points_gdf, aoi_gdf)
            if len(pts) > 0:
                points_global = (float(len(pts)), float("nan"))
                if "type" in pts.columns:
                    for typ, sub in pts.groupby("type"):
                        sub = _clip_to_aoi(sub, aoi_gdf)
                        if len(sub) == 0:
                            continue
                        points_by_type_stats[str(typ)] = (float(len(sub)), float("nan"))

        # Lines totals
        if all_lines_gdf is not None and len(all_lines_gdf) > 0:
            ln = _clip_to_aoi(all_lines_gdf, aoi_gdf)
            if len(ln) > 0:
                ln_m = ln.to_crs(metric_crs) if metric_crs else ln
                total_km = float(ln_m.length.sum()) / 1000.0
                lines_global_km = (total_km, float("nan"))
                if "type" in ln.columns:
                    for typ, sub in ln.groupby("type"):
                        sub = _clip_to_aoi(sub, aoi_gdf)
                        if len(sub) == 0:
                            continue
                        sub_m = sub.to_crs(metric_crs) if metric_crs else sub
                        tk = float(sub_m.length.sum()) / 1000.0
                        lines_by_type_km_stats[str(typ)] = (tk, float("nan"))

        onefile_path = _write_onefile(
            hazard_name=hazard_name,
            outdir=outdir,
            points_global=points_global,
            points_by_type_stats=points_by_type_stats if points_by_type_stats else None,
            lines_global_km=lines_global_km,
            lines_by_type_km_stats=lines_by_type_km_stats if lines_by_type_km_stats else None,
        )
        return {"hazard": hazard_name, "mode": "skipped", "threshold": np.nan, "onefile": str(onefile_path)}

    # --- Raster exists: compute exposed polygons in AOI ---
    with rasterio.open(raster_path) as rds:
        band = rds.read(1).astype(float)
        aoi_mask = _aoi_mask(rds, aoi_gdf)

        nodata = rds.nodata
        valid = aoi_mask.copy()
        if nodata is not None and not np.isnan(nodata):
            valid &= ~np.isclose(band, nodata)
        valid &= ~np.isnan(band)

        vals = band[valid]
        if vals.size == 0:
            # No valid pixels in AOI -> totals only, exposures unknown (NaN)
            if all_points_gdf is not None and len(all_points_gdf) > 0:
                pts = _clip_to_aoi(all_points_gdf, aoi_gdf)
                if len(pts) > 0:
                    points_global = (float(len(pts)), float("nan"))
                    if "type" in pts.columns:
                        for typ, sub in pts.groupby("type"):
                            sub = _clip_to_aoi(sub, aoi_gdf)
                            if len(sub) == 0:
                                continue
                            points_by_type_stats[str(typ)] = (float(len(sub)), float("nan"))

            if all_lines_gdf is not None and len(all_lines_gdf) > 0:
                ln = _clip_to_aoi(all_lines_gdf, aoi_gdf)
                if len(ln) > 0:
                    ln_m = ln.to_crs(metric_crs) if metric_crs else ln
                    total_km = float(ln_m.length.sum()) / 1000.0
                    lines_global_km = (total_km, float("nan"))
                    if "type" in ln.columns:
                        for typ, sub in ln.groupby("type"):
                            sub = _clip_to_aoi(sub, aoi_gdf)
                            if len(sub) == 0:
                                continue
                            sub_m = sub.to_crs(metric_crs) if metric_crs else sub
                            tk = float(sub_m.length.sum()) / 1000.0
                            lines_by_type_km_stats[str(typ)] = (tk, float("nan"))

            onefile_path = _write_onefile(
                hazard_name=hazard_name,
                outdir=outdir,
                points_global=points_global,
                points_by_type_stats=points_by_type_stats if points_by_type_stats else None,
                lines_global_km=lines_global_km,
                lines_by_type_km_stats=lines_by_type_km_stats if lines_by_type_km_stats else None,
            )
            return {"hazard": hazard_name, "mode": "no_valid_pixels", "threshold": np.nan, "onefile": str(onefile_path)}

        is_cat = _auto_is_categorical(vals)
        if is_cat:
            uniq = np.unique(vals)
            thr = float(np.quantile(uniq, 0.75))
            mode = "categorical_q75_classes"
        else:
            thr = float(np.quantile(vals, 0.90))
            mode = "continuous_p90"

        exposed_polys = _build_exposed_polygons(band, valid, rds.transform, rds.crs, thr)

    # --- Points: exposure by containment in exposed polygons ---
    if all_points_gdf is not None and len(all_points_gdf) > 0:
        pts = _clip_to_aoi(all_points_gdf, aoi_gdf)
        if len(pts) > 0:
            if exposed_polys.empty:
                exp_n = 0.0
                total = float(len(pts))
            else:
                exp_union = unary_union(exposed_polys)
                pts2 = pts.to_crs(exposed_polys.crs) if pts.crs != exposed_polys.crs else pts
                inside = pts2.geometry.within(exp_union)
                total = float(len(pts2))
                exp_n = float(inside.sum())
            points_global = (total, exp_n)

            if points_by_type:
                for typ, sub in points_by_type.items():
                    sub = _clip_to_aoi(sub, aoi_gdf)
                    if len(sub) == 0:
                        continue
                    if exposed_polys.empty:
                        points_by_type_stats[str(typ)] = (float(len(sub)), 0.0)
                    else:
                        sub2 = sub.to_crs(exposed_polys.crs) if sub.crs != exposed_polys.crs else sub
                        inside = sub2.geometry.within(exp_union)
                        points_by_type_stats[str(typ)] = (float(len(sub2)), float(inside.sum()))
            elif "type" in pts.columns:
                for typ, sub in pts.groupby("type"):
                    if exposed_polys.empty:
                        points_by_type_stats[str(typ)] = (float(len(sub)), 0.0)
                    else:
                        sub2 = sub.to_crs(exposed_polys.crs) if sub.crs != exposed_polys.crs else sub
                        inside = sub2.geometry.within(exp_union)
                        points_by_type_stats[str(typ)] = (float(len(sub2)), float(inside.sum()))

    # --- Lines: exposure by intersection length with exposed union ---
    if all_lines_gdf is not None and len(all_lines_gdf) > 0:
        ln = _clip_to_aoi(all_lines_gdf, aoi_gdf)
        if len(ln) > 0:
            ln_m = ln.to_crs(metric_crs) if metric_crs else ln
            total_km = float(ln_m.length.sum()) / 1000.0
            if exposed_polys.empty:
                exp_km = 0.0
            else:
                exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
                exp_km = float(ln_m.intersection(exp_union_m).length.sum()) / 1000.0
            lines_global_km = (total_km, exp_km)

            if lines_by_type:
                for typ, sub in lines_by_type.items():
                    sub = _clip_to_aoi(sub, aoi_gdf)
                    if len(sub) == 0:
                        continue
                    sub_m = sub.to_crs(metric_crs) if metric_crs else sub
                    tk = float(sub_m.length.sum()) / 1000.0
                    if exposed_polys.empty:
                        ek = 0.0
                    else:
                        exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
                        ek = float(sub_m.intersection(exp_union_m).length.sum()) / 1000.0
                    lines_by_type_km_stats[str(typ)] = (tk, ek)
            elif "type" in ln.columns:
                for typ, sub in ln.groupby("type"):
                    sub = _clip_to_aoi(sub, aoi_gdf)
                    if len(sub) == 0:
                        continue
                    sub_m = sub.to_crs(metric_crs) if metric_crs else sub
                    tk = float(sub_m.length.sum()) / 1000.0
                    if exposed_polys.empty:
                        ek = 0.0
                    else:
                        exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
                        ek = float(sub_m.intersection(exp_union_m).length.sum()) / 1000.0
                    lines_by_type_km_stats[str(typ)] = (tk, ek)

    # ---- write single tidy file (pure stats) ----
    onefile_path = _write_onefile(
        hazard_name=hazard_name,
        outdir=outdir,
        points_global=points_global,
        points_by_type_stats=points_by_type_stats if points_by_type_stats else None,
        lines_global_km=lines_global_km,
        lines_by_type_km_stats=lines_by_type_km_stats if lines_by_type_km_stats else None,
    )

    return {"hazard": hazard_name, "mode": mode, "threshold": thr, "onefile": str(onefile_path)}
