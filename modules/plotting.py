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

class _TextOnlyHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Texte dessiné DANS la "colonne symbole", donc démarre tout à gauche
        t = Text(xdescent, ydescent + height * 0.5, orig_handle.text,
                 va="center", ha="left", transform=trans)
        return [t]

class _TextOnly:
    """Petit objet-proxy qui porte juste le texte à afficher sur la ligne."""
    def __init__(self, text):
        self.text = text
        
class _NoSymbolHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        artist = Line2D([], [], linewidth=0, alpha=0)  # invisible
        artist.set_transform(trans)
        return [artist]

def safe_plot(gdf, ax, **kwargs):
    """
    Plot a GeoDataFrame only if it is not empty.

    Parameters:
        gdf (GeoDataFrame): Layer to plot.
        ax (matplotlib.axes.Axes): Axes on which to plot.
        kwargs: Additional keyword arguments passed to .plot().
    """
    if not gdf.empty:
        gdf.plot(ax=ax, **kwargs)

class _NoHandle(HandlerBase):
    """Legend handler that renders nothing (no marker/box), so text is left-aligned."""
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        return []

def build_raster_legend_handles(spec, data_masked=None, vmin=None, vmax=None):
    """
    Build legend handles (Patch objects) for a raster, so we can merge them
    into the same legend as vector layers.

    - For discrete rasters (e.g., floods, landslide): use spec["palette"] + spec["labels"].
    - For continuous rasters (e.g., earthquake, wildfire): discretize into 5 bins
      and label as Very low ... Very high using the colormap.
    """
    handles = []

    # Discrete style: palette + labels provided by spec
    if spec.get("type", "discrete") == "discrete":
        colors = spec.get("palette", [])
        labels = spec.get("labels", [])
        for color, lbl in zip(colors, labels):
            handles.append(Patch(facecolor=color, edgecolor="none", label=lbl))
        return handles

    # Continuous style: build 5 bins and sample the colormap
    # Use data range if not explicitly provided
    if vmin is None or vmax is None:
        if data_masked is None:
            return handles
        arr = np.asarray(data_masked, dtype="float64")
        # Use robust percentiles so flat/NaN edges don't kill the legend
        finite = np.isfinite(arr)
        if not finite.any():
            # fall back to dummy range
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

    bins = np.linspace(vmin, vmax, 6)               # 5 classes
    centers = 0.5 * (bins[:-1] + bins[1:])
    labels_vlh = ["Very low", "Low", "Medium", "High", "Very high"]

    # spec["cmap"] is already a colormap (used in imshow below)
    norm = Normalize(vmin=vmin, vmax=vmax)
    raw_cmap = spec.get("cmap", "viridis")
    cmap = plt.get_cmap(raw_cmap) if isinstance(raw_cmap, str) else raw_cmap

    for c, lbl in zip(centers, labels_vlh):
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

    Parameters:
        ax (matplotlib.axes.Axes): Axis object to add raster to.
        raster_path (str): Path to raster (.tif) file.
        aoi (GeoDataFrame): Area of interest used to crop the raster.
        hazard_name (str): Hazard name used to determine symbology.

    Returns:
        list: The extent [xmin, xmax, ymin, ymax] for setting plot limits.
    """
    spec = get_hazard_display_spec(hazard_name)

    # Read + mask to AOI, forcing nodata to NaN
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
        arr = out_image[0]  # single band

    # Mask all invalid (nan) pixels
    masked = np.ma.masked_invalid(arr)

    # Compute extent for plotting
    height, width = arr.shape
    bounds = rasterio.transform.array_bounds(height, width, out_transform)
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # Plot based on type
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
        # colorbar
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=spec["cmap"], norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(spec["label"])
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['Low', 'High'])

    else:  # discrete
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
      - merged legend with a raster name row before bins (no global legend title)
    """
    spec = get_hazard_display_spec(hazard_name)
    legend_raster_handles = []

    # Symbology
    custom_point_styles = {
        "substations": {"color": "#b2df8a", "marker": "^", "label": "Substation"},
        "transformer": {"color": "black", "marker": "o", "label": "Transformer"},
        "tower": {"color": "gray", "marker": "s", "label": "Tower"},
        "existing": {"color": "blue", "marker": "^", "label": "Existing substation"},
    }
    custom_line_styles = {
        "hv": {"color": "yellow", "linestyle": "-", "linewidth": 2.5, "label": "High transmission line"},
        "lv": {"color": "#970499", "linestyle": "-", "linewidth": 1.5, "label": "Low transmission line"},
        "existing": {"color": "black", "linestyle": "--", "linewidth": 1.5, "label": "Existing line"},
    }

    # --- Raster clip (optional) ---
    xmin = ymin = xmax = ymax = None
    masked = None
    if raster_path:
        with rasterio.open(raster_path) as src:
            geoms = [f["geometry"] for f in aoi.to_crs(src.crs).__geo_interface__["features"]]
            masked_arr, transform = mask(src, geoms, crop=True, filled=False)
            masked_arr = masked_arr[0]
        masked = np.ma.masked_array(masked_arr.data, mask=masked_arr.mask)
        h, w = masked.shape
        b0, b1, b2, b3 = rasterio.transform.array_bounds(h, w, transform)
        xmin, xmax, ymin, ymax = b0, b2, b1, b3
    else:
        xmin, ymin, xmax, ymax = aoi.total_bounds

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
    ax.set_ylim(min(ymin, ymax), max(ymin, ymax))

    # Basemap
    try:
        ctx.add_basemap(ax, crs=aoi.crs.to_string(), source=ctx.providers.CartoDB.Positron, attribution_size=6, zorder=0)
    except Exception as e:
        print(f"[!] Could not add basemap: {e}")

    # Raster underlay
    if masked is not None:
        if spec["type"] == "continuous":
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
            # legend handles for continuous bins (Very low..Very high)
            legend_raster_handles = build_raster_legend_handles(spec, data_masked=masked, vmin=vmin, vmax=vmax)
        else:
            cmap = ListedColormap(spec["palette"])
            norm = BoundaryNorm(spec["breaks"], len(spec["palette"]))
            ax.imshow(
                masked,
                cmap=cmap,
                norm=norm,
                extent=(xmin, xmax, ymin, ymax),
                origin="upper",
                alpha=0.8,
                zorder=1,
            )
            # legend handles for discrete bins (use palette/labels)
            legend_raster_handles = build_raster_legend_handles(spec, data_masked=masked)

    # AOI
    aoi.boundary.plot(ax=ax, color="black", linewidth=1, zorder=2, label="_nolegend_")

    # ---------- VECTORS (two-pass ordering) ----------
    used_labels = set()

    def _label_once(lbl: str) -> str:
        if lbl in used_labels:
            return "_nolegend_"
        used_labels.add(lbl)
        return lbl

    # Lines: pass 1 (NOT exposed)
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

    # Points: pass 1 (NOT exposed)
    if (points is not None) and (len(points) > 0) and ("infra_type" in points.columns) and ("exposed" in points.columns):
        for pt_type in points["infra_type"].unique():
            pts = points[points["infra_type"] == pt_type]
            style = custom_point_styles.get(pt_type, {"color": "gray", "marker": "o", "label": pt_type})
            subset = pts[~pts["exposed"]]
            if len(subset) > 0:
                safe_plot(
                    subset,
                    ax,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=30,
                    label=_label_once(f"{style['label']} (not exposed)"),
                    zorder=5,
                )

    # Lines: pass 2 (EXPOSED)
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

    # Points: pass 2 (EXPOSED) — always on very top
    if (points is not None) and (len(points) > 0) and ("infra_type" in points.columns) and ("exposed" in points.columns):
        for pt_type in points["infra_type"].unique():
            pts = points[points["infra_type"] == pt_type]
            style = custom_point_styles.get(pt_type, {"color": "gray", "marker": "o", "label": pt_type})
            subset = pts[pts["exposed"]]
            if len(subset) > 0:
                safe_plot(
                    subset,
                    ax,
                    color="red",
                    marker=style["marker"],
                    markersize=30,
                    label=_label_once(f"{style['label']} (exposed)"),
                    zorder=6,
                )

    # ---------- Legend & save ----------
    ax.set_aspect("equal")
    ax.axis("off")

    # merged legend with raster "name line" before bins, no global title
    _merge_and_draw_legend(ax, spec, legend_raster_handles)

    # Save
    output_filename = f"exposure_map_{hazard_name}_{suffix}.png" if suffix else f"exposure_map_{hazard_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=resolution, bbox_inches="tight")
    plt.show()
    plt.close()



def plot_initial_map(aoi, points, lines, output_path=None):
    """
    Plot a basemap with AOI and energy network (points and lines).

    Parameters:
        aoi (GeoDataFrame): Area of interest.
        points (GeoDataFrame): Infrastructure points.
        lines (GeoDataFrame): Infrastructure lines.
        output_path (str): Optional path to save the PNG file.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot layers
    aoi.boundary.plot(ax=ax, color='black', linewidth=1, label='_nolegend_')
    lines.plot(ax=ax, color='blue', linewidth=1, label='Lines')
    points.plot(ax=ax, color='red', markersize=10, label='Points')

    # Add basemap
    try:
        ctx.add_basemap(ax, crs=aoi.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"[!] Could not add basemap: {e}")

    #ax.set_title("Initial Map: Energy Infrastructure", fontsize=15, fontweight="bold")
    ax.axis('off')
    ax.legend()

    if output_path:
        #add_scalebar(ax, loc='lower left')      # or use length_km=5 to force fixed size
        #add_north_arrow(ax, loc='upper left')   # change loc if it overlaps with legend
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_initial_map_by_type(
    aoi, points_by_type, lines_by_type, output_path,
    *, raster_path=None, hazard_name=None, frame_buffer: float = 0.03, basemap: bool = True
):
    """
    Plot all infrastructure layers (points and lines) by type on a single map,
    optionally overlaying a hazard raster (clipped to AOI), with a small frame
    buffer around the extent to avoid tight-crop visual artifacts.

    frame_buffer: percentage of width/height to add around the map frame (e.g., 0.03 = 3%)
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # 0) Hazard spec (if any)
    spec = get_hazard_display_spec(hazard_name) if (raster_path and hazard_name) else None
    legend_raster_handles = []
    current_crs = aoi.crs  # will be set to raster CRS if a raster is used

    xmin = ymin = xmax = ymax = None

    # 1) If raster provided: clip & draw it FIRST (under network)
    if raster_path and spec is not None:
        with rasterio.open(raster_path) as src:
            current_crs = src.crs
            # Mask raster by AOI in the raster CRS
            aoi_proj = aoi.to_crs(current_crs)
            geoms = [f["geometry"] for f in aoi_proj.__geo_interface__["features"]]
            masked_arr, transform = mask(src, geoms, crop=True, filled=False)
            masked_arr = masked_arr[0]

        # Convert to a masked array (masked=True outside AOI)
        masked = np.ma.masked_array(masked_arr.data, mask=masked_arr.mask)

        # --- AOI-based frame (NOT raster bounds) ---
        xmin, ymin, xmax, ymax = aoi_proj.total_bounds
        dx, dy = (xmax - xmin), (ymax - ymin)
        bx, by = dx * float(frame_buffer), dy * float(frame_buffer)
        ax.set_xlim(xmin - bx, xmax + bx)
        ax.set_ylim(ymin - by, ymax + by)

        # Raster extent still comes from the masked raster transform
        h, w = masked.shape
        rxmin, rymin, rxmax, rymax = rasterio.transform.array_bounds(h, w, transform)

        # Draw raster with transparent "bad" (masked) pixels
        if spec["type"] == "continuous":
            vmin, vmax = float(masked.min()), float(masked.max())
            cmap = plt.get_cmap(spec["cmap"]).copy()
            cmap.set_bad(alpha=0)  # outside AOI fully transparent
            ax.imshow(
                masked, cmap=cmap,
                extent=(rxmin, rxmax, rymin, rymax),
                vmin=vmin, vmax=vmax, alpha=0.8, zorder=1
            )
            legend_raster_handles = build_raster_legend_handles(
                spec, data_masked=masked, vmin=vmin, vmax=vmax
            )
        else:
            cmap = ListedColormap(spec["palette"])
            cmap.set_bad((0, 0, 0, 0))
            norm = BoundaryNorm(spec["breaks"], len(spec["palette"]))
            ax.imshow(
                masked, cmap=cmap, norm=norm,
                extent=(rxmin, rxmax, rymin, rymax),
                origin="upper", alpha=0.8, zorder=1
            )
            legend_raster_handles = build_raster_legend_handles(spec, data_masked=masked)

    else:
        # No raster: set view to AOI bounds + buffer (in AOI CRS)
        xmin, ymin, xmax, ymax = aoi.total_bounds
        dx, dy = (xmax - xmin), (ymax - ymin)
        bx, by = dx * float(frame_buffer), dy * float(frame_buffer)
        ax.set_xlim(xmin - bx, xmax + bx)
        ax.set_ylim(ymin - by, ymax + by)

    # 2) Basemap under everything (use the *current* CRS of the axis)
    if basemap:
        try:
            ctx.add_basemap(
                ax, crs=current_crs.to_string(),
                source=ctx.providers.CartoDB.Positron,
                attribution_size=6, zorder=0
            )
        except Exception as e:
            print(f"[!] Could not load basemap: {e}")

    # 3) AOI boundary (no legend) — reproject to current_crs if needed
    aoi_to_plot = aoi if aoi.crs == current_crs else aoi.to_crs(current_crs)
    aoi_to_plot.boundary.plot(ax=ax, color="black", linewidth=1, label="_nolegend_", zorder=2)

    # 4) Styles
    custom_point_styles = {
        "substations": {"color": "#b2df8a", "marker": "^", "label": "Substation"},
        "transformer": {"color": "black", "marker": "o", "label": "Transformer"},
        "tower": {"color": "gray", "marker": "s", "label": "Tower"},
        "existing": {"color": "blue", "marker": "^", "label": "Existing substation"},
    }
    custom_line_styles = {
        "hv": {"color": "yellow", "linestyle": "-", "linewidth": 2.5, "label": "High transmission line"},
        "lv": {"color": "#970499", "linestyle": "-", "linewidth": 1.5, "label": "Low transmission line"},
        "existing": {"color": "black", "linestyle": "--", "linewidth": 1.5, "label": "Existing line"},
    }

    # 5) Draw lines by type (below points) — reproject if needed
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

    # 6) Draw points by type (on top of lines) — reproject if needed
    for name, gdf in (points_by_type or {}).items():
        if gdf is None or gdf.empty:
            continue
        gdfp = gdf if gdf.crs == current_crs else gdf.to_crs(current_crs)
        style = custom_point_styles.get(name, {"color": "gray", "marker": "o", "label": f"Point: {name}"})
        gdfp.plot(
            ax=ax,
            color=style["color"], marker=style["marker"], markersize=30,
            label=_label_once(style["label"]), zorder=5
        )

    # 7) Final formatting + merged legend
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
    Add a scale bar (metric) to the map. Assumes axis is in meters (e.g., EPSG:3857 or any projected CRS in meters).
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to which the scale bar will be added.
    loc : str
        Location of the scale bar ('lower left', 'lower right', etc.).
    pad : float
        Padding between the bar and the axis.
    borderpad : float
        Padding between the bar and the frame.
    sep : float
        Separation between the bar and the label text.
    length_km : float or None
        Fixed length of the scale bar in kilometers. If None, it will be calculated automatically as ~1/5 of map width.
    """
    # Get axis extent in data coordinates
    x0, x1 = ax.get_xlim()
    width_m = abs(x1 - x0)

    # Compute length of the scale bar
    if length_km is None:
        # Automatically choose a nice round number
        raw = width_m / 5.0
        nice_steps = [1, 2, 5]
        pow10 = 10 ** int(np.floor(np.log10(raw)))
        best = min(nice_steps, key=lambda s: abs(raw - s * pow10))
        length_m = best * pow10
    else:
        length_m = float(length_km) * 1000.0

    # Choose label (m or km)
    if length_m >= 1000:
        label = f"{length_m/1000:.0f} km" if (length_m/1000) >= 1 else f"{length_m/1000:.1f} km"
    else:
        label = f"{length_m:.0f} m"

    # Draw scale bar
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
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to which the arrow will be added.
    loc : str
        Location of the arrow ('upper left', 'upper right', 'lower left', 'lower right').
    size : float
        Arrow length relative to the map height (0.08 = 8% of map height).
    pad : float
        Padding from the edges.
    """
    # Axis extent
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0

    # Choose base position depending on loc
    if loc == 'upper left':
        x = x0 + dx * (0.08 + pad)
        y = y1 - dy * (0.08 + pad)
    elif loc == 'upper right':
        x = x1 - dx * (0.08 + pad)
        y = y1 - dy * (0.08 + pad)
    elif loc == 'lower left':
        x = x0 + dx * (0.08 + pad)
        y = y0 + dy * (0.08 + pad)
    else:  # 'lower right'
        x = x1 - dx * (0.08 + pad)
        y = y0 + dy * (0.08 + pad)

    # Arrow length in data coordinates
    L = dy * size

    # Draw the arrow
    ax.annotate(
        '', xy=(x, y + L), xytext=(x, y),
        arrowprops=dict(arrowstyle='-|>', linewidth=1.5)
    )
    ax.text(x, y + L + dy * 0.02, 'N', ha='center', va='bottom', fontsize=11, fontweight='bold')


