"""
Plotting utilities for visualizing hazard exposure maps.

This module provides functions to:
- Add raster overlays (continuous or discrete) with custom symbology.
- Safely plot GeoDataFrames if not empty.
- Generate and save infrastructure exposure maps with background basemap.
"""

import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import os
import rasterio
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from rasterio.mask import mask
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .config_utils import get_hazard_display_spec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from pyproj import CRS
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from pathlib import Path
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image

# ---- Flood classification spec (depth in meters) ----
FLOOD_BREAKS = [0.0, 0.3, 1.0, 2.0, np.inf]
FLOOD_LABELS = ["Low (≤0.3 m)", "Medium (0.3–1 m)", "High (1–2 m)", "Very High (>2 m)"]
FLOOD_COLORS = [
    "#e0f3ff",  # Low (light)
    "#9ec9ff",  # Medium
    "#5596ff",  # High
    "#08306b",  # Very High (dark)
]

# ---- Seismic (PGA in g) — discrete classes in merged legend ----
SEISMIC_BREAKS = [0.0, 0.1, 0.2, 0.3, np.inf]
SEISMIC_LABELS = ["0.0–0.1 g", "0.1–0.2 g", "0.2–0.3 g", "≥ 0.3 g"]
SEISMIC_NAMES = {"earthquake", "seismic", "eq", "pga"}

def _seismic_palette(cmap_name: str = "plasma_r"):
    cmap = plt.get_cmap(cmap_name)
    samples = np.linspace(0.25, 0.95, 4)
    return [mcolors.to_hex(cmap(s)) for s in samples]

# ---- Heat (t2m in °C) — discrete classes for legend ----
HEAT_NAMES  = {"heat", "temperature", "t2m", "era5_heat"}
HEAT_BREAKS = [-float("inf"), 30.0, 35.0, 40.0, 45.0, float("inf")]  # <30, 30–35, 35–40, 40–45, ≥45
HEAT_LABELS = ["< 30 °C", "30–35 °C", "35–40 °C", "40–45 °C", "≥ 45 °C"]
HEAT_COLORS = [
    "#fff7bc",
    "#fee391",
    "#fdb863",
    "#ef6548",
    "#b30000",
]
def _classify_heat_degC(masked_temp: np.ma.MaskedArray) -> np.ndarray:
    arr = masked_temp.filled(np.nan).astype(float)
    classes = np.digitize(arr, HEAT_BREAKS, right=False)  # 1..5
    classes[np.isnan(arr)] = 0
    return classes

# ---- Cold (TNn-type °C) — discrete classes for legend (darker = colder) ----
COLD_NAMES  = {"cold", "tnn", "era5_cold", "cold_extreme"}
COLD_BREAKS = [-float("inf"), -40.0, -30.0, -20.0, -10.0, float("inf")]  # < -40, -40–-30, -30–-20, -20–-10, ≥ -10
COLD_LABELS = ["< -40 °C", "-40–-30 °C", "-30–-20 °C", "-20–-10 °C", "≥ -10 °C"]
COLD_COLORS = [
    "#08306b",  # < -40 (darkest)
    "#2171b5",  # -40–-30
    "#6baed6",  # -30–-20
    "#c6dbef",  # -20–-10
    "#f7fbff",  # ≥ -10 (lightest)
]
def _classify_cold_degC(masked_temp: np.ma.MaskedArray) -> np.ndarray:
    arr = masked_temp.filled(np.nan).astype(float)
    classes = np.digitize(arr, COLD_BREAKS, right=False)  # 1..5
    classes[np.isnan(arr)] = 0
    return classes

def classify_flood_depth_array(depth_array: np.ndarray, nodata_value: float | int | None) -> np.ndarray:
    arr = depth_array.astype(float).copy()
    if nodata_value is not None:
        nodata_mask = (arr == nodata_value) | np.isnan(arr)
    else:
        nodata_mask = np.isnan(arr) | (arr < 0)
    classes = np.zeros(arr.shape, dtype=np.uint8)
    classes[(arr > 0) & (arr <= 0.3)] = 1
    classes[(arr > 0.3) & (arr <= 1.0)] = 2
    classes[(arr > 1.0) & (arr <= 2.0)] = 3
    classes[(arr > 2.0)] = 4
    classes[nodata_mask] = 0
    return classes

def render_flood_raster_styled(
    raster_path: str,
    title: str = "Flood depth (classified)",
    out_png: str | None = None,
    aoi_gdf=None,
):
    with rasterio.open(raster_path) as src:
        depth = src.read(1, out_shape=(src.height, src.width), resampling=Resampling.nearest)
        nodata = src.nodata
        classes = classify_flood_depth_array(depth, nodata)
        cmap = ListedColormap(["none"] + FLOOD_COLORS)
        bounds = [0, 1, 2, 3, 4, 5]
        norm = BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        ax.imshow(classes, extent=extent, origin="upper", cmap=cmap, norm=norm)
        if aoi_gdf is not None and len(aoi_gdf) > 0:
            aoi_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        legend_handles = [
            Patch(facecolor=FLOOD_COLORS[0], label=FLOOD_LABELS[0]),
            Patch(facecolor=FLOOD_COLORS[1], label=FLOOD_LABELS[1]),
            Patch(facecolor=FLOOD_COLORS[2], label=FLOOD_LABELS[2]),
            Patch(facecolor=FLOOD_COLORS[3], label=FLOOD_LABELS[3]),
        ]
        ax.legend(handles=legend_handles, title="Flood hazard (depth)", loc="lower left")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        plt.tight_layout()
        if out_png:
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

class _TextOnlyHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        t = Text(xdescent, ydescent + height * 0.5, orig_handle.text,
                 va="center", ha="left", transform=trans)
        return [t]

class _TextOnly:
    def __init__(self, text):
        self.text = text

class _NoSymbolHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        artist = Line2D([], [], linewidth=0, alpha=0)
        artist.set_transform(trans)
        return [artist]

def safe_plot(gdf, ax, **kwargs):
    if (gdf is not None) and (not gdf.empty):
        gdf.plot(ax=ax, **kwargs)

class _NoHandle(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        return []

def build_raster_legend_handles(spec, data_masked=None, vmin=None, vmax=None,  labels_override=None):
    handles = []
    labels = labels_override or spec.get("labels", ["Very low", "Low", "Medium", "High", "Very high"])
    if spec.get("type", "discrete") == "discrete":
        colors = spec.get("palette", [])
        labels = spec.get("labels", [])
        for color, lbl in zip(colors, labels):
            handles.append(Patch(facecolor=color, edgecolor="none", label=lbl))
        return handles
    if vmin is None or vmax is None:
        if data_masked is None:
            return handles
        arr = np.asarray(data_masked, dtype="float64")
        finite = np.isfinite(arr)
        if not finite.any():
            vmin, vmax = 0.0, 1.0
        else:
            lo = np.nanpercentile(arr[finite], 2)
            hi = np.nanpercentile(arr[finite], 98)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                vmin, vmax = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
                if vmin == vmax:
                    vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = float(lo), float(hi)
    bins = np.linspace(vmin, vmax, 6)
    centers = 0.5 * (bins[:-1] + bins[1:])
    norm = Normalize(vmin=vmin, vmax=vmax)
    raw_cmap = spec.get("cmap", "viridis")
    cmap = plt.get_cmap(raw_cmap) if isinstance(raw_cmap, str) else raw_cmap
    for c, lbl in zip(centers, labels):
        handles.append(Patch(facecolor=cmap(norm(c)), edgecolor="none", label=lbl))
    return handles

def _merge_and_draw_legend(ax, spec, legend_raster_handles):
    handles, labels = ax.get_legend_handles_labels()
    handler_map = {}
    if legend_raster_handles:
        raster_name = (
            spec.get("legend_title")
            or spec.get("label")
            or "Hazard"
        )
        name_row = _TextOnly(raster_name)
        handler_map[name_row] = _TextOnlyHandler()
        handles = list(handles) + [name_row] + list(legend_raster_handles)
        labels = list(labels) + [""] + [h.get_label() for h in legend_raster_handles]
    else:
        handles = list(handles)
        labels = list(labels)
    seen, merged = set(), []
    for h, l in zip(handles, labels):
        key = (id(h), l) if l == "" else l
        if key not in seen:
            merged.append((h, l))
            seen.add(key)
    if not merged:
        return
    handles, labels = zip(*merged)
    ax.legend(
        handles, labels,
        title=None, loc="upper right", frameon=True,
        handler_map=handler_map
    )

def add_raster_to_ax(ax, raster_path, aoi, hazard_name):
    """
    Add a styled raster overlay to a Matplotlib axis.
    """
    spec = get_hazard_display_spec(hazard_name)
    with rasterio.open(raster_path) as src:
        aoi_proj = aoi.to_crs(src.crs)
        geometries = [feat["geometry"] for feat in aoi_proj.__geo_interface__["features"]]
        out_image, out_transform = mask(
            src,
            geometries,
            crop=True,
            nodata=np.nan,
            filled=True
        )
        arr = out_image[0]
    masked = np.ma.masked_invalid(arr)
    height, width = arr.shape
    bounds = rasterio.transform.array_bounds(height, width, out_transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # Heat: discrete bands
    if hazard_name in {"heat", "temperature", "t2m", "era5_heat"}:
        classes = _classify_heat_degC(masked)
        cmap = ListedColormap(["none"] + HEAT_COLORS)
        norm = BoundaryNorm(range(0, len(HEAT_COLORS)+1), len(HEAT_COLORS)+1)
        ax.imshow(classes, cmap=cmap, norm=norm, extent=extent, origin="upper", alpha=0.9)
        legend_handles = [Patch(facecolor=HEAT_COLORS[i], label=HEAT_LABELS[i]) for i in range(len(HEAT_COLORS))]
        _merge_and_draw_legend(ax, {"legend_title": "Heat (daily max, °C)"}, legend_handles)
        return extent

    # Cold: discrete bands
    if hazard_name in {"cold", "tnn", "era5_cold", "cold_extreme"}:
        classes = _classify_cold_degC(masked)
        cmap = ListedColormap(["none"] + COLD_COLORS)
        norm = BoundaryNorm(range(0, len(COLD_COLORS)+1), len(COLD_COLORS)+1)
        ax.imshow(classes, cmap=cmap, norm=norm, extent=extent, origin="upper", alpha=0.9)
        legend_handles = [Patch(facecolor=COLD_COLORS[i], label=COLD_LABELS[i]) for i in range(len(COLD_COLORS))]
        _merge_and_draw_legend(ax, {"legend_title": "Cold (cold extreme, °C)"}, legend_handles)
        return extent

    # Landslide: continuous raster but legend merged as 5 bins (no side colorbar)
    if hazard_name == "landslide" and spec.get("type") == "continuous":
        vmin = float(masked.min())
        vmax = float(masked.max())
        ax.imshow(
            masked,
            cmap=spec["cmap"],
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8
        )
        legend_handles = build_raster_legend_handles(
            spec, data_masked=masked, vmin=vmin, vmax=vmax
        )
        _merge_and_draw_legend(ax, {"legend_title": "Landslide"}, legend_handles)
        return extent

    # Seismic: discrete PGA classes in merged legend
    if spec.get("type") == "continuous" and (hazard_name in SEISMIC_NAMES):
        palette = _seismic_palette(spec.get("cmap", "plasma_r"))
        cmap = ListedColormap(palette)
        norm = BoundaryNorm(SEISMIC_BREAKS, len(palette))
        ax.imshow(
            masked, cmap=cmap, norm=norm,
            extent=extent, origin="upper", alpha=0.85
        )
        legend_handles = [Patch(facecolor=palette[i], label=SEISMIC_LABELS[i]) for i in range(len(palette))]
        _merge_and_draw_legend(ax, {"legend_title": "Seismic Hazard — PGA (g)"}, legend_handles)
        return extent

    # Generic continuous / discrete (kept for other hazards)
    if spec["type"] == "continuous":
        vmin = masked.min()
        vmax = masked.max()
        im = ax.imshow(
            masked,
            cmap=spec["cmap"],
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8
        )
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=spec["cmap"], norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(spec["label"])
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['Low', 'High'])
    else:
        cmap = ListedColormap(spec["palette"])
        norm = BoundaryNorm(spec["breaks"], len(spec["palette"]))
        im = ax.imshow(
            masked,
            cmap=cmap,
            norm=norm,
            extent=extent,
            origin='upper',
            alpha=0.8
        )
        tick_pos = [
            (spec["breaks"][i] + spec["breaks"][i+1]) / 2
            for i in range(len(spec["breaks"]) - 1)
        ]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, ticks=tick_pos)
        cbar.ax.set_yticklabels(spec["labels"])
        cbar.set_label(spec["label"])
    return extent

def plot_and_save_exposure_map(
    aoi,
    points,
    lines,
    hazard_name,
    output_dir,
    raster_path=None,
    resolution=300,
    suffix="",
    group_by_type=True
):
    """
    Build an exposure map with:
      - basemap
      - optional raster underlay (styled by hazard spec)
      - AOI boundary
      - vectors (two-pass: not-exposed first, then exposed on top)
      - merged legend with a raster name row before bins (no global title)
    """
    spec = get_hazard_display_spec(hazard_name)
    legend_raster_handles = []

    # Hollow points (towers are skipped later)
    custom_point_styles = {
        "substations": {"marker": "^", "label": "Substation",
                        "facecolor": "none", "edgecolor": "#666666", "linewidth": 0.7},
        "transformer": {"marker": "o", "label": "Transformer",
                        "facecolor": "none", "edgecolor": "#666666", "linewidth": 0.7},
        "existing": {"marker": "^", "label": "Existing substation",
                     "facecolor": "none", "edgecolor": "#1f78b4", "linewidth": 0.9},
    }
    custom_line_styles = {
        "hv": {"color": "yellow", "linestyle": "-", "linewidth": 2.5, "label": "High transmission line"},
        "lv": {"color": "#970499", "linestyle": "-", "linewidth": 1.5, "label": "Low transmission line"},
        "existing": {"color": "black", "linestyle": "--", "linewidth": 1.5, "label": "Existing line"},
    }

    xmin = ymin = xmax = ymax = None
    masked = None
    nodata_val = None
    if raster_path:
        with rasterio.open(raster_path) as src:
            nodata_val = src.nodata
            geoms = [f["geometry"] for f in aoi.to_crs(src.crs).__geo_interface__["features"]]
            masked_arr, transform = mask(src, geoms, crop=True, filled=False)
            masked_arr = masked_arr[0]
        masked = np.ma.masked_array(masked_arr.data, mask=masked_arr.mask)
        h, w = masked.shape
        b0, b1, b2, b3 = rasterio.transform.array_bounds(h, w, transform)
        xmin, xmax, ymin, ymax = b0, b2, b1, b3
    else:
        xmin, ymin, xmax, ymax = aoi.total_bounds

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
    ax.set_ylim(min(ymin, ymax), max(ymin, ymax))

    try:
        ctx.add_basemap(ax, crs=aoi.crs.to_string(), source=ctx.providers.CartoDB.Positron, attribution_size=6, zorder=0)
    except Exception as e:
        print(f"[!] Could not add basemap: {e}")

    if masked is not None:
        if hazard_name in {"heat", "temperature", "t2m", "era5_heat"}:
            classes = _classify_heat_degC(masked)
            cmap = ListedColormap(["none"] + HEAT_COLORS)
            norm = BoundaryNorm(range(0, len(HEAT_COLORS)+1), len(HEAT_COLORS)+1)
            ax.imshow(
                classes,
                cmap=cmap,
                norm=norm,
                extent=(xmin, xmax, ymin, ymax),
                origin="upper",
                alpha=0.9,
                zorder=1,
            )
            legend_raster_handles = [Patch(facecolor=HEAT_COLORS[i], label=HEAT_LABELS[i]) for i in range(len(HEAT_COLORS))]
        elif hazard_name in {"cold", "tnn", "era5_cold", "cold_extreme"}:
            classes = _classify_cold_degC(masked)
            cmap = ListedColormap(["none"] + COLD_COLORS)
            norm = BoundaryNorm(range(0, len(COLD_COLORS)+1), len(COLD_COLORS)+1)
            ax.imshow(
                classes,
                cmap=cmap,
                norm=norm,
                extent=(xmin, xmax, ymin, ymax),
                origin="upper",
                alpha=0.9,
                zorder=1,
            )
            legend_raster_handles = [Patch(facecolor=COLD_COLORS[i], label=COLD_LABELS[i]) for i in range(len(COLD_COLORS))]
        elif hazard_name in {"pluvial_flood", "fluvial_flood", "combined_flood"}:
            classes = classify_flood_depth_array(masked.filled(np.nan), nodata_val)
            flood_cmap = ListedColormap(["none"] + FLOOD_COLORS)
            ax.imshow(
                classes,
                cmap=flood_cmap,
                extent=(xmin, xmax, ymin, ymax),
                origin="upper",
                alpha=1.0,
                zorder=1,
            )
            legend_raster_handles = [
                Patch(facecolor=FLOOD_COLORS[0], label=FLOOD_LABELS[0]),
                Patch(facecolor=FLOOD_COLORS[1], label=FLOOD_LABELS[1]),
                Patch(facecolor=FLOOD_COLORS[2], label=FLOOD_LABELS[2]),
                Patch(facecolor=FLOOD_COLORS[3], label=FLOOD_LABELS[3]),
            ]
        elif hazard_name == "landslide" and spec.get("type") == "continuous":
            vmin, vmax = float(masked.min()), float(masked.max())
            ax.imshow(
                masked,
                cmap=spec["cmap"],
                extent=(xmin, xmax, ymin, ymax),
                vmin=vmin,
                vmax=vmax,
                alpha=0.8,
                zorder=1,
            )
            legend_raster_handles = build_raster_legend_handles(
                spec, data_masked=masked, vmin=vmin, vmax=vmax
            )
        else:
            if spec["type"] == "continuous":
                if hazard_name in SEISMIC_NAMES:
                    palette = _seismic_palette(spec.get("cmap", "plasma_r"))
                    cmap = ListedColormap(palette)
                    norm = BoundaryNorm(SEISMIC_BREAKS, len(palette))
                    ax.imshow(
                        masked,
                        cmap=cmap,
                        norm=norm,
                        extent=(xmin, xmax, ymin, ymax),
                        origin="upper",
                        alpha=0.85,
                        zorder=1,
                    )
                    legend_raster_handles = [Patch(facecolor=palette[i], label=SEISMIC_LABELS[i]) for i in range(len(palette))]
                else:
                    vmin, vmax = float(masked.min()), float(masked.max())
                    im = ax.imshow(
                        masked,
                        cmap=spec["cmap"],
                        extent=(xmin, xmax, ymin, ymax),
                        vmin=vmin,
                        vmax=vmax,
                        alpha=0.8,
                        zorder=1,
                    )
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="3%", pad=0.1)
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    sm = ScalarMappable(cmap=spec["cmap"], norm=norm)
                    cbar = plt.colorbar(sm, cax=cax)
                    cbar.set_label(spec["label"])
                    cbar.set_ticks([vmin, vmax])
                    cbar.set_ticklabels(['Low', 'High'])
            else:
                cmap = ListedColormap(spec["palette"])
                norm = BoundaryNorm(spec["breaks"], len(spec["palette"]))
                im = ax.imshow(
                    masked,
                    cmap=cmap,
                    norm=norm,
                    extent=(xmin, xmax, ymin, ymax),
                    origin="upper",
                    alpha=0.8,
                    zorder=1,
                )
                legend_raster_handles = build_raster_legend_handles(spec, data_masked=masked)

    aoi.boundary.plot(ax=ax, color="black", linewidth=1, zorder=2, label="_nolegend_")

    used_labels = set()
    def _label_once(lbl: str) -> str:
        if lbl in used_labels:
            return "_nolegend_"
        used_labels.add(lbl)
        return lbl

    # Lines (NOT exposed)
    if (lines is not None) and (len(lines) > 0) and ("infra_type" in lines.columns) and ("exposed" in lines.columns):
        for ln_type in lines["infra_type"].unique():
            lns = lines[lines["infra_type"] == ln_type]
            style = custom_line_styles.get(
                ln_type, {"color": "gray", "linestyle": "-", "linewidth": 1.5, "label": ln_type}
            )
            subset = lns[~lns["exposed"]]
            if len(subset) > 0:
                safe_plot(
                    subset,
                    ax,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=_label_once(f"{style['label']} (not exposed)"),
                    zorder=3,
                )

    # Points (NOT exposed) — skip towers; hollow subs/transformers
    if (points is not None) and (len(points) > 0) and ("infra_type" in points.columns) and ("exposed" in points.columns):
        for pt_type in points["infra_type"].unique():
            if pt_type == "tower":
                continue
            pts = points[points["infra_type"] == pt_type]
            style = custom_point_styles.get(pt_type, {"marker": "o", "label": pt_type,
                                                      "facecolor": "none", "edgecolor": "#666666", "linewidth": 0.7})
            subset = pts[~pts["exposed"]]
            if len(subset) > 0:
                safe_plot(
                    subset,
                    ax,
                    marker=style["marker"],
                    markersize=30,
                    facecolor=style.get("facecolor", "none"),
                    edgecolor=style.get("edgecolor", "#666666"),
                    linewidth=style.get("linewidth", 0.7),
                    label=_label_once(f"{style['label']} (not exposed)"),
                    zorder=5,
                )

    # Lines (EXPOSED)
    if (lines is not None) and (len(lines) > 0) and ("infra_type" in lines.columns) and ("exposed" in lines.columns):
        for ln_type in lines["infra_type"].unique():
            lns = lines[lines["infra_type"] == ln_type]
            style = custom_line_styles.get(
                ln_type, {"color": "gray", "linestyle": "-", "linewidth": 1.5, "label": ln_type}
            )
            subset = lns[lns["exposed"]]
            if len(subset) > 0:
                safe_plot(
                    subset,
                    ax,
                    color="red",
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=_label_once(f"{style['label']} (exposed)"),
                    zorder=4,
                )

    # Points (EXPOSED) — skip towers; hollow subs/transformers with red outline
    if (points is not None) and (len(points) > 0) and ("infra_type" in points.columns) and ("exposed" in points.columns):
        for pt_type in points["infra_type"].unique():
            if pt_type == "tower":
                continue
            pts = points[points["infra_type"] == pt_type]
            style = custom_point_styles.get(pt_type, {"marker": "o", "label": pt_type,
                                                      "facecolor": "none", "edgecolor": "#666666", "linewidth": 0.7})
            subset = pts[pts["exposed"]]
            if len(subset) > 0:
                safe_plot(
                    subset,
                    ax,
                    marker=style["marker"],
                    markersize=30,
                    facecolor="none",
                    edgecolor="red",          # red outline when exposed
                    linewidth=style.get("linewidth", 0.7),
                    label=_label_once(f"{style['label']} (exposed)"),
                    zorder=6,
                )

    ax.set_aspect("equal")
    ax.axis("off")

    _merge_and_draw_legend(ax, spec, legend_raster_handles)

    output_filename = f"exposure_map_{hazard_name}_{suffix}.png" if suffix else f"exposure_map_{hazard_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=resolution, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_initial_map(aoi, points, lines, output_path=None):
    """
    Plot a basemap with AOI and energy network (points and lines).
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    aoi.boundary.plot(ax=ax, color='black', linewidth=1, label='_nolegend_')
    if lines is not None:
        lines.plot(ax=ax, color='blue', linewidth=1, label='Lines')

    # Remove towers; keep points hollow by default
    if points is not None:
        pts_to_plot = points
        if "infra_type" in points.columns:
            pts_to_plot = points[points["infra_type"] != "tower"]
        pts_to_plot.plot(
            ax=ax, marker="o", markersize=20, facecolor="none", edgecolor="#666666", linewidth=0.7, label="Points"
        )

    try:
        ctx.add_basemap(ax, crs=aoi.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"[!] Could not add basemap: {e}")

    ax.axis('off')
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_initial_map_by_type(
    aoi, points_by_type, lines_by_type, output_path,
    *, raster_path=None, hazard_name=None, frame_buffer: float = 0.03, basemap: bool = True
):
    """
    Plot all infrastructure layers (points and lines) by type on a single map,
    optionally overlaying a hazard raster (clipped to AOI), with a small frame buffer.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    spec = get_hazard_display_spec(hazard_name) if (raster_path and hazard_name) else None
    legend_raster_handles = []
    current_crs = aoi.crs
    xmin = ymin = xmax = ymax = None
    nodata_val = None

    if raster_path and spec is not None:
        with rasterio.open(raster_path) as src:
            current_crs = src.crs
            nodata_val = src.nodata
            aoi_proj = aoi.to_crs(current_crs)
            geoms = [f["geometry"] for f in aoi_proj.__geo_interface__["features"]]
            masked_arr, transform = mask(src, geoms, crop=True, filled=False)
            masked_arr = masked_arr[0]

        masked = np.ma.masked_array(masked_arr.data, mask=masked_arr.mask)

        xmin, ymin, xmax, ymax = aoi_proj.total_bounds
        dx, dy = (xmax - xmin), (ymax - ymin)
        bx, by = dx * float(frame_buffer), dy * float(frame_buffer)
        ax.set_xlim(xmin - bx, xmax + bx)
        ax.set_ylim(ymin - by, ymax + by)

        h, w = masked.shape
        rxmin, rymin, rxmax, rymax = rasterio.transform.array_bounds(h, w, transform)

        if hazard_name in {"heat", "temperature", "t2m", "era5_heat"}:
            classes = _classify_heat_degC(masked)
            cmap = ListedColormap(["none"] + HEAT_COLORS)
            norm = BoundaryNorm(range(0, len(HEAT_COLORS)+1), len(HEAT_COLORS)+1)
            ax.imshow(
                classes, cmap=cmap, norm=norm,
                extent=(rxmin, rxmax, rymin, rymax),
                origin="upper", alpha=0.9, zorder=1
            )
            legend_raster_handles = [Patch(facecolor=HEAT_COLORS[i], label=HEAT_LABELS[i]) for i in range(len(HEAT_COLORS))]
        elif hazard_name in {"cold", "tnn", "era5_cold", "cold_extreme"}:
            classes = _classify_cold_degC(masked)
            cmap = ListedColormap(["none"] + COLD_COLORS)
            norm = BoundaryNorm(range(0, len(COLD_COLORS)+1), len(COLD_COLORS)+1)
            ax.imshow(
                classes, cmap=cmap, norm=norm,
                extent=(rxmin, rxmax, rymin, rymax),
                origin="upper", alpha=0.9, zorder=1
            )
            legend_raster_handles = [Patch(facecolor=COLD_COLORS[i], label=COLD_LABELS[i]) for i in range(len(COLD_COLORS))]
        elif hazard_name in {"pluvial_flood", "fluvial_flood", "combined_flood"}:
            classes = classify_flood_depth_array(masked.filled(np.nan), nodata_val)
            flood_cmap = ListedColormap(["none"] + FLOOD_COLORS)
            ax.imshow(
                classes, cmap=flood_cmap,
                extent=(rxmin, rxmax, rymin, rymax),
                origin="upper", alpha=1.0, zorder=1
            )
            legend_raster_handles = [
                Patch(facecolor=FLOOD_COLORS[0], label=FLOOD_LABELS[0]),
                Patch(facecolor=FLOOD_COLORS[1], label=FLOOD_LABELS[1]),
                Patch(facecolor=FLOOD_COLORS[2], label=FLOOD_LABELS[2]),
                Patch(facecolor=FLOOD_COLORS[3], label=FLOOD_LABELS[3]),
            ]
        elif hazard_name == "landslide" and spec.get("type") == "continuous":
            vmin, vmax = float(masked.min()), float(masked.max())
            cmap = plt.get_cmap(spec["cmap"]).copy()
            cmap.set_bad(alpha=0)
            ax.imshow(
                masked, cmap=cmap,
                extent=(rxmin, rxmax, rymin, rymax),
                vmin=vmin, vmax=vmax, alpha=0.8, zorder=1
            )
            legend_raster_handles = build_raster_legend_handles(
                spec, data_masked=masked, vmin=vmin, vmax=vmax
            )
        else:
            if spec["type"] == "continuous":
                vmin, vmax = float(masked.min()), float(masked.max())
                cmap = plt.get_cmap(spec["cmap"]).copy()
                cmap.set_bad(alpha=0)
                if hazard_name in SEISMIC_NAMES:
                    palette = _seismic_palette(spec.get("cmap", "plasma_r"))
                    cmap_disc = ListedColormap(palette)
                    norm = BoundaryNorm(SEISMIC_BREAKS, len(palette))
                    ax.imshow(
                        masked, cmap=cmap_disc, norm=norm,
                        extent=(rxmin, rxmax, rymin, rymax),
                        origin="upper", alpha=0.85, zorder=1
                    )
                    legend_raster_handles = [Patch(facecolor=palette[i], label=SEISMIC_LABELS[i]) for i in range(len(palette))]
                else:
                    im = ax.imshow(
                        masked, cmap=cmap,
                        extent=(rxmin, rxmax, rymin, rymax),
                        vmin=vmin, vmax=vmax, alpha=0.8, zorder=1
                    )
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="3%", pad=0.1)
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    sm = ScalarMappable(cmap=spec["cmap"], norm=norm)
                    cbar = plt.colorbar(sm, cax=cax)
                    cbar.set_label(spec["label"])
                    cbar.set_ticks([vmin, vmax])
                    cbar.set_ticklabels(['Low', 'High'])
            else:
                cmap = ListedColormap(spec["palette"])
                cmap.set_bad((0, 0, 0, 0))
                norm = BoundaryNorm(spec["breaks"], len(spec["palette"]))
                im = ax.imshow(
                    masked, cmap=cmap, norm=norm,
                    extent=(rxmin, rxmax, rymin, rymax),
                    origin="upper", alpha=0.8, zorder=1
                )
                legend_raster_handles = build_raster_legend_handles(spec, data_masked=masked)
    else:
        xmin, ymin, xmax, ymax = aoi.total_bounds
        dx, dy = (xmax - xmin), (ymax - ymin)
        bx, by = dx * float(frame_buffer), dy * float(frame_buffer)
        ax.set_xlim(xmin - bx, xmax + bx)
        ax.set_ylim(ymin - by, ymax + by)

    if basemap:
        try:
            ctx.add_basemap(
                ax, crs=current_crs.to_string(),
                source=ctx.providers.CartoDB.Positron,
                attribution_size=6, zorder=0
            )
        except Exception as e:
            print(f"[!] Could not load basemap: {e}")

    aoi_to_plot = aoi if aoi.crs == current_crs else aoi.to_crs(current_crs)
    aoi_to_plot.boundary.plot(ax=ax, color="black", linewidth=1, label="_nolegend_", zorder=2)

    # Hollow styles & tower removal
    custom_point_styles = {
        "substations": {"marker": "^", "label": "Substation",
                        "facecolor": "none", "edgecolor": "#666666", "linewidth": 0.7},
        "transformer": {"marker": "o", "label": "Transformer",
                        "facecolor": "none", "edgecolor": "#666666", "linewidth": 0.7},
        "existing": {"marker": "^", "label": "Existing substation",
                     "facecolor": "none", "edgecolor": "#1f78b4", "linewidth": 0.9},
    }
    custom_line_styles = {
        "hv": {"color": "yellow", "linestyle": "-", "linewidth": 2.5, "label": "High transmission line"},
        "lv": {"color": "#970499", "linestyle": "-", "linewidth": 1.5, "label": "Low transmission line"},
        "existing": {"color": "black", "linestyle": "--", "linewidth": 1.5, "label": "Existing line"},
    }

    used_labels = set()
    def _label_once(lbl: str) -> str:
        if lbl in used_labels:
            return "_nolegend_"
        used_labels.add(lbl)
        return lbl

    for name, gdf in (lines_by_type or {}).items():
        if gdf is None or gdf.empty:
            continue
        gdfp = gdf if gdf.crs == current_crs else gdf.to_crs(current_crs)
        style = custom_line_styles.get(
            name, {"color": "gray", "linestyle": "-", "linewidth": 1.0, "label": f"Line: {name}"}
        )
        gdfp.plot(
            ax=ax,
            color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"],
            label=_label_once(style["label"]), zorder=3
        )

    for name, gdf in (points_by_type or {}).items():
        if name == "tower":
            continue
        if gdf is None or gdf.empty:
            continue
        gdfp = gdf if gdf.crs == current_crs else gdf.to_crs(current_crs)
        style = custom_point_styles.get(name, {"marker": "o", "label": f"Point: {name}",
                                               "facecolor": "none", "edgecolor": "#666666", "linewidth": 0.7})
        gdfp.plot(
            ax=ax,
            marker=style["marker"], markersize=30,
            facecolor=style.get("facecolor", "none"),
            edgecolor=style.get("edgecolor", "#666666"),
            linewidth=style.get("linewidth", 0.7),
            label=_label_once(style["label"]), zorder=5
        )

    ax.set_axis_off()
    if spec is not None:
        _merge_and_draw_legend(ax, spec, legend_raster_handles)
    else:
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def add_scalebar(ax, *, loc='lower left', pad=0.4, borderpad=0.5, sep=5, length_km=None):
    """
    Add a scale bar (metric) to the map. Assumes axis is in meters.
    """
    x0, x1 = ax.get_xlim()
    width_m = abs(x1 - x0)
    if length_km is None:
        raw = width_m / 5.0
        nice_steps = [1, 2, 5]
        pow10 = 10 ** int(np.floor(np.log10(raw)))
        best = min(nice_steps, key=lambda s: abs(raw - s * pow10))
        length_m = best * pow10
    else:
        length_m = float(length_km) * 1000.0
    if length_m >= 1000:
        label = f"{length_m/1000:.0f} km" if (length_m/1000) >= 1 else f"{length_m/1000:.1f} km"
    else:
        label = f"{length_m:.0f} m"
    fontprops = fm.FontProperties(size=9)
    scalebar = AnchoredSizeBar(
        ax.transData,
        length_m, label, loc,
        pad=pad, borderpad=borderpad, sep=sep,
        frameon=True, size_vertical=length_m * 0.02, fontproperties=fontprops
    )
    ax.add_artist(scalebar)

def add_north_arrow(ax, *, loc='upper left', size=0.08, pad=0.02):
    """
    Add a simple North arrow to the map.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0
    if loc == 'upper left':
        x = x0 + dx * (0.08 + pad)
        y = y1 - dy * (0.08 + pad)
    elif loc == 'upper right':
        x = x1 - dx * (0.08 + pad)
        y = y1 - dy * (0.08 + pad)
    elif loc == 'lower left':
        x = x0 + dx * (0.08 + pad)
        y = y0 + dy * (0.08 + pad)
    else:
        x = x1 - dx * (0.08 + pad)
        y = y0 + dy * (0.08 + pad)
    L = dy * size
    ax.annotate(
        '', xy=(x, y + L), xytext=(x, y),
        arrowprops=dict(arrowstyle='-|>', linewidth=1.5)
    )
    ax.text(x, y + L + dy * 0.02, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold')
