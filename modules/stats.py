# modules/stats.py
# One-stop infrastructure stats module (robust against missing rasters).
# - compute_infra_stats_from_results(): consume your existing exposed GDFs (NO raster work).
# - compute_infra_stats_from_overlay(): lightweight AOI-masked thresholding ONLY IF a valid raster is provided;
#   otherwise it gracefully skips and writes a small '..._overlay_skipped.csv' (no paths, stats-only intent).
# - NEW: write a single tidy stats file per hazard: <hazard>_stats_onefile.csv (no paths/meta, stats only).

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

EXPOSED_COL_CANDIDATES = ["exposed", "is_exposed", "exposed_flag", "exp", "EXPOSED", "Exposed"]

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _clip_to_aoi(gdf: Optional[gpd.GeoDataFrame], aoi_gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or len(gdf) == 0:
        return gdf
    if aoi_gdf is None or len(aoi_gdf) == 0:
        return gdf
    if gdf.crs != aoi_gdf.crs:
        gdf = gdf.to_crs(aoi_gdf.crs)
    return gpd.clip(gdf, aoi_gdf)

def _normalize_exposed_col(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure a boolean 'exposed' column exists, reusing common aliases if needed."""
    for c in EXPOSED_COL_CANDIDATES:
        if c in gdf.columns:
            if c != "exposed":
                gdf = gdf.rename(columns={c: "exposed"})
            gdf["exposed"] = gdf["exposed"].astype(bool)
            return gdf
    raise ValueError(f"No exposure boolean column found. Tried: {EXPOSED_COL_CANDIDATES}")






def _normalize_exposed_col(gdf: "gpd.GeoDataFrame"):
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
            gdf['exposed'] = (gdf[col].replace({'True': True, 'False': False, 'true': True, 'false': False, 'YES': True, 'Yes': True, 'No':                               False, 'NO': False, 'Y': True, 'N': False}).apply(lambda v: bool(int(v)) if isinstance(v, (np.integer, int,                                 np.int64, np.int32, np.int16)) else bool(v)))
            return gdf

    # 2) Try class/level categorical columns (any non-null category counts as exposed)
    class_like = [c for c in ['class', 'hazard_class', 'risk_class', 'flood_class', 'landslide_class', 'category', 'level', 'hazard_level']
                  if c in gdf.columns]
    for c in class_like:
        # Treat categories (e.g., 'Low', 'Medium', 'High', 'Very High', 1..4) as exposed when not null and not 'none'
        s = gdf[c].astype(str).str.strip().str.lower()
        gdf['exposed'] = s.notna() & (s != '') & (s != 'none') & (s != 'nan')
        return gdf

    # 3) Try numeric intensity/value columns (> 0 counts as exposed)
    numeric_like = [c for c in ['value', 'intensity', 'depth', 'mmi', 'pga', 'hazard_value', 'score']
                    if c in gdf.columns]
    for c in numeric_like:
        with pd.option_context('mode.use_inf_as_na', True):
            vals = pd.to_numeric(gdf[c], errors='coerce').fillna(0.0)
        gdf['exposed'] = vals > 0
        return gdf

    # 4) Fallback: if this layer is already an "*_exposed" product, consider everything exposed
    # (prevents breaking downstream stats; adjust if you prefer to be stricter)
    gdf['exposed'] = True
    return gdf


def _is_metric_crs(crs) -> bool:
    try:
        return crs and crs.axis_info and any(ai.unit_name and ai.unit_name.lower().startswith("metre") for ai in crs.axis_info)
    except Exception:
        return False

def _metric_crs_from_aoi(aoi_gdf: gpd.GeoDataFrame, fallback_crs=None):
    """Pick a metric CRS for length/area computation (UTM from AOI; fallback to given CRS if already metric)."""
    try:
        return aoi_gdf.estimate_utm_crs()
    except Exception:
        return fallback_crs if _is_metric_crs(fallback_crs) else None

def _safe_percent(n: float, d: float) -> float:
    return float(n) / float(d) * 100.0 if d else 0.0

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
# Single-file ("onefile") writer
# ----------------------------------------------------------------------

def _write_onefile_pure(hazard_name: str, outdir: Path) -> Path:
    """
    Build <hazard>_stats_onefile.csv (pure stats, no paths/meta) from the 4 CSVs if present:
      - <hazard>_points_exposure.csv
      - <hazard>_lines_exposure.csv
      - <hazard>_points_by_type.csv
      - <hazard>_lines_by_type.csv
    Rows format (tidy):
      hazard, geometry(point|line), scope(global|by_type), type(nullable), total, exposed, percent, unit(count|km)
    """
    def safe_read(p: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

    px = safe_read(outdir / f"{hazard_name}_points_exposure.csv")
    lx = safe_read(outdir / f"{hazard_name}_lines_exposure.csv")
    pb = safe_read(outdir / f"{hazard_name}_points_by_type.csv")
    lb = safe_read(outdir / f"{hazard_name}_lines_by_type.csv")

    rows: List[dict] = []

    # Global points
    if not px.empty:
        rows.append({
            "hazard": hazard_name, "geometry": "point", "scope": "global", "type": None,
            "total": float(px.iloc[0].get("total_points_in_aoi", 0)),
            "exposed": float(px.iloc[0].get("exposed_points_in_aoi", 0)),
            "percent": float(px.iloc[0].get("percent_exposed_points", 0.0)),
            "unit": "count",
        })

    # Global lines
    if not lx.empty:
        rows.append({
            "hazard": hazard_name, "geometry": "line", "scope": "global", "type": None,
            "total": float(lx.iloc[0].get("total_length_km_in_aoi", 0.0)),
            "exposed": float(lx.iloc[0].get("exposed_length_km_in_aoi", 0.0)),
            "percent": float(lx.iloc[0].get("percent_exposed_length", 0.0)),
            "unit": "km",
        })

    # Points by type
    if not pb.empty:
        for _, r in pb.iterrows():
            rows.append({
                "hazard": hazard_name, "geometry": "point", "scope": "by_type", "type": r.get("type"),
                "total": float(r.get("total_points_in_aoi", 0)),
                "exposed": float(r.get("exposed_points_in_aoi", 0)),
                "percent": float(r.get("percent_exposed_points", 0.0)),
                "unit": "count",
            })

    # Lines by type
    if not lb.empty:
        for _, r in lb.iterrows():
            rows.append({
                "hazard": hazard_name, "geometry": "line", "scope": "by_type", "type": r.get("type"),
                "total": float(r.get("total_length_km_in_aoi", 0.0)),
                "exposed": float(r.get("exposed_length_km_in_aoi", 0.0)),
                "percent": float(r.get("percent_exposed_length", 0.0)),
                "unit": "km",
            })

    tidy = pd.DataFrame(rows, columns=["hazard","geometry","scope","type","total","exposed","percent","unit"])
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
    This does NOT touch rasters; it only consumes your exposure results.

    Outputs (under output/stats/<hazard>/):
      <hazard>_points_exposure.csv
      <hazard>_lines_exposure.csv
      <hazard>_points_by_type.csv   (if 'type' exists)
      <hazard>_lines_by_type.csv    (if 'type' exists)
      <hazard>_stats_onefile.csv    (pure, tidy, no paths/meta)
    """
    outdir = Path(out_root) / hazard_name
    _ensure_dir(outdir)

    # Points
    if points_exposed_gdf is not None and len(points_exposed_gdf) > 0:
        pts = _clip_to_aoi(points_exposed_gdf, aoi_gdf).copy()
        if len(pts) > 0:
            pts = _normalize_exposed_col(pts)
            total = int(len(pts))
            exp_n = int(pts["exposed"].sum())
            pct = _safe_percent(exp_n, total)
            pd.DataFrame([{
                "total_points_in_aoi": total,
                "exposed_points_in_aoi": exp_n,
                "percent_exposed_points": pct
            }]).to_csv(outdir / f"{hazard_name}_points_exposure.csv", index=False)

            if "type" in pts.columns:
                grp = pts.groupby("type")["exposed"].agg(["sum", "count"]).reset_index()
                grp = grp.rename(columns={"sum": "exposed_points_in_aoi", "count": "total_points_in_aoi"})
                grp["percent_exposed_points"] = grp.apply(
                    lambda r: _safe_percent(r["exposed_points_in_aoi"], r["total_points_in_aoi"]),
                    axis=1
                )
                grp.to_csv(outdir / f"{hazard_name}_points_by_type.csv", index=False)

    # Lines
    if lines_exposed_gdf is not None and len(lines_exposed_gdf) > 0:
        ln = _clip_to_aoi(lines_exposed_gdf, aoi_gdf).copy()
        if len(ln) > 0:
            metric_crs = _metric_crs_from_aoi(aoi_gdf, ln.crs)
            if metric_crs:
                ln = ln.to_crs(metric_crs)
            ln = _normalize_exposed_col(ln)

            total_km, exp_km = _lines_total_and_exposed_km(ln)
            pct = _safe_percent(exp_km, total_km)
            pd.DataFrame([{
                "total_length_km_in_aoi": total_km,
                "exposed_length_km_in_aoi": exp_km,
                "percent_exposed_length": pct
            }]).to_csv(outdir / f"{hazard_name}_lines_exposure.csv", index=False)

            if "type" in ln.columns:
                rows = []
                for typ, sub in ln.groupby("type"):
                    tk, ek = _lines_total_and_exposed_km(sub)
                    rows.append({
                        "type": typ,
                        "total_length_km_in_aoi": tk,
                        "exposed_length_km_in_aoi": ek,
                        "percent_exposed_length": _safe_percent(ek, tk)
                    })
                pd.DataFrame(rows).to_csv(outdir / f"{hazard_name}_lines_by_type.csv", index=False)

    # ---- NEW: write single tidy file (pure stats) ----
    onefile_path = _write_onefile_pure(hazard_name, outdir)

    return {"hazard": hazard_name, "onefile": str(onefile_path)}

# ----------------------------------------------------------------------
# B) Reporting overlay for hazards WITHOUT exposure studies
#     (robust: gracefully skip if raster_path is None/missing)
# ----------------------------------------------------------------------

def _mask_aoi(rds, aoi_gdf: gpd.GeoDataFrame):
    aoi = aoi_gdf if aoi_gdf.crs == rds.crs else aoi_gdf.to_crs(rds.crs)
    return features.geometry_mask(
        [geom.__geo_interface__ for geom in aoi.geometry],
        transform=rds.transform, invert=True, out_shape=(rds.height, rds.width)
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
    threshold_val: float
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

def compute_infra_stats_from_overlay(
    hazard_name: str,
    raster_path: Optional[str],
    aoi_gdf: gpd.GeoDataFrame,
    all_points_gdf: Optional[gpd.GeoDataFrame] = None,
    all_lines_gdf: Optional[gpd.GeoDataFrame] = None,
    points_by_type: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    lines_by_type: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    out_root: str = "output/stats"
) -> dict:
    """
    Minimal 'reporting overlay' for hazards WITHOUT exposure studies.
    Defines an exposed zone from the AOI-masked raster:
      - Continuous rasters: threshold = 90th percentile (P90) of valid AOI pixels.
      - Categorical rasters: threshold = 75th percentile of present integer classes (top quartile classes).
    Then computes the SAME infrastructure stats (global + by type).

    Robust behavior:
      - If raster_path is None/empty/missing, the function SKIPS gracefully
        and writes <hazard>_overlay_skipped.csv with the reason (no paths/meta).
    """
    outdir = Path(out_root) / hazard_name
    _ensure_dir(outdir)

    # --- Robust guard: skip gracefully if raster is not provided or missing ---
    if raster_path is None or str(raster_path).strip() == "" or not Path(str(raster_path)).exists():
        pd.DataFrame([{
            "hazard": hazard_name,
            "skipped": True,
            "reason": "Missing or non-existent raster_path for overlay reporting."
        }]).to_csv(outdir / f"{hazard_name}_overlay_skipped.csv", index=False)
        print(f"[WARN] Overlay stats skipped for '{hazard_name}': raster not found.")
        return {"hazard": hazard_name, "skipped": True, "reason": "missing raster"}

    # --- Open raster and compute threshold from AOI-masked values ---
    with rasterio.open(raster_path) as rds:
        mask = _mask_aoi(rds, aoi_gdf)
        band = rds.read(1)
        nodata = rds.nodata
        valid = (band != nodata) & mask if nodata is not None else (~np.isnan(band) & mask)
        vals = band[valid].astype(float)

        if vals.size == 0:
            pd.DataFrame([{
                "hazard": hazard_name,
                "note": "No valid pixels in AOI"
            }]).to_csv(outdir / f"{hazard_name}_aoi_threshold_used.csv", index=False)
            return {"hazard": hazard_name, "mode": "empty"}

        is_categorical = _auto_is_categorical(vals)
        if is_categorical:
            classes = np.unique(vals[~np.isnan(vals)])
            thr = float(np.percentile(classes, 75))  # top quartile classes
            mode = "categorical_top_quartile_classes"
        else:
            thr = float(np.nanpercentile(vals, 90))  # P90
            mode = "continuous_p90"

        # Save threshold provenance (stats only, no paths)
        pd.DataFrame([{"hazard": hazard_name, "mode": mode, "threshold_used": thr}]).to_csv(
            outdir / f"{hazard_name}_aoi_threshold_used.csv", index=False
        )

        # Build exposed polygons and pick a metric CRS
        exposed_polys = _build_exposed_polygons(band, valid, rds.transform, rds.crs, thr)
        metric_crs = _metric_crs_from_aoi(aoi_gdf, rds.crs)

    # ---------- Global stats ----------
    if all_points_gdf is not None and len(all_points_gdf) > 0:
        pts = _clip_to_aoi(all_points_gdf, aoi_gdf)
        if len(pts) > 0:
            pts_m = pts.to_crs(metric_crs) if metric_crs else pts
            exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
            flags = pts_m.intersects(exp_union_m)
            total = int(len(pts_m))
            exp_n = int(flags.sum())
            pd.DataFrame([{
                "total_points_in_aoi": total,
                "exposed_points_in_aoi": exp_n,
                "percent_exposed_points": _safe_percent(exp_n, total)
            }]).to_csv(outdir / f"{hazard_name}_points_exposure.csv", index=False)

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
            pd.DataFrame([{
                "total_length_km_in_aoi": total_km,
                "exposed_length_km_in_aoi": exp_km,
                "percent_exposed_length": _safe_percent(exp_km, total_km)
            }]).to_csv(outdir / f"{hazard_name}_lines_exposure.csv", index=False)

    # ---------- By-type stats ----------
    if points_by_type:
        rows: List[pd.DataFrame] = []
        for typ, g in points_by_type.items():
            if g is None or len(g) == 0:
                continue
            sub = _clip_to_aoi(g, aoi_gdf)
            if len(sub) == 0:
                continue
            sub_m = sub.to_crs(metric_crs) if metric_crs else sub
            exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
            flags = sub_m.intersects(exp_union_m)
            total = int(len(sub_m))
            exp_n = int(flags.sum())
            rows.append(pd.DataFrame([{
                "type": typ,
                "total_points_in_aoi": total,
                "exposed_points_in_aoi": exp_n,
                "percent_exposed_points": _safe_percent(exp_n, total)
            }]))
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(outdir / f"{hazard_name}_points_by_type.csv", index=False)
    elif all_points_gdf is not None and "type" in all_points_gdf.columns:
        rows: List[pd.DataFrame] = []
        for typ, sub in all_points_gdf.groupby("type"):
            sub = _clip_to_aoi(sub, aoi_gdf)
            if len(sub) == 0:
                continue
            sub_m = sub.to_crs(metric_crs) if metric_crs else sub
            exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
            flags = sub_m.intersects(exp_union_m)
            total = int(len(sub_m))
            exp_n = int(flags.sum())
            rows.append(pd.DataFrame([{
                "type": typ,
                "total_points_in_aoi": total,
                "exposed_points_in_aoi": exp_n,
                "percent_exposed_points": _safe_percent(exp_n, total)
            }]))
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(outdir / f"{hazard_name}_points_by_type.csv", index=False)

    if lines_by_type:
        rows: List[pd.DataFrame] = []
        for typ, g in lines_by_type.items():
            if g is None or len(g) == 0:
                continue
            sub = _clip_to_aoi(g, aoi_gdf)
            if len(sub) == 0:
                continue
            sub_m = sub.to_crs(metric_crs) if metric_crs else sub
            total_km = float(sub_m.length.sum()) / 1000.0
            if exposed_polys.empty:
                exp_km = 0.0
            else:
                exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
                exp_km = float(sub_m.intersection(exp_union_m).length.sum()) / 1000.0
            rows.append(pd.DataFrame([{
                "type": typ,
                "total_length_km_in_aoi": total_km,
                "exposed_length_km_in_aoi": exp_km,
                "percent_exposed_length": _safe_percent(exp_km, total_km)
            }]))
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(outdir / f"{hazard_name}_lines_by_type.csv", index=False)
    elif all_lines_gdf is not None and "type" in all_lines_gdf.columns:
        rows: List[pd.DataFrame] = []
        for typ, sub in all_lines_gdf.groupby("type"):
            sub = _clip_to_aoi(sub, aoi_gdf)
            if len(sub) == 0:
                continue
            sub_m = sub.to_crs(metric_crs) if metric_crs else sub
            total_km = float(sub_m.length.sum()) / 1000.0
            if exposed_polys.empty:
                exp_km = 0.0
            else:
                exp_union_m = unary_union(exposed_polys.to_crs(metric_crs) if metric_crs else exposed_polys)
                exp_km = float(sub_m.intersection(exp_union_m).length.sum()) / 1000.0
            rows.append(pd.DataFrame([{
                "type": typ,
                "total_length_km_in_aoi": total_km,
                "exposed_length_km_in_aoi": exp_km,
                "percent_exposed_length": _safe_percent(exp_km, total_km)
            }]))
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(outdir / f"{hazard_name}_lines_by_type.csv", index=False)

    # ---- NEW: write single tidy file (pure stats) ----
    onefile_path = _write_onefile_pure(hazard_name, outdir)

    return {"hazard": hazard_name, "mode": mode, "threshold": thr, "onefile": str(onefile_path)}
