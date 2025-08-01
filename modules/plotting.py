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
from rasterio.mask import mask
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .config_utils import get_hazard_display_spec


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


def plot_and_save_exposure_map(aoi, points, lines, hazard_name, output_dir, raster_path=None, resolution=300, suffix="",group_by_type=True):
    """
    Generate and save an exposure map showing the AOI, energy infrastructure, and optional raster hazard.

    The map includes:
    - AOI boundary
    - Infrastructure points and lines (colored by exposure)
    - Hazard raster overlay if provided
    - Background basemap (CartoDB Positron)

    Parameters:
        aoi (GeoDataFrame): Area of interest polygon.
        points (GeoDataFrame): Infrastructure points with "exposed" column.
        lines (GeoDataFrame): Infrastructure lines with "exposed" column.
        hazard_name (str): Name of the hazard (used for labeling and symbology).
        output_dir (str): Directory where the PNG will be saved.
        raster_path (str, optional): Path to raster (.tif) file to overlay.
        resolution (int): Resolution in DPI for the saved image.
    """
    spec = get_hazard_display_spec(hazard_name)
    # Custom symbology for exposure map
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
    xmin = ymin = xmax = ymax = None
    masked = None  # masked raster array to display

    # Step 1: Load and clip raster to AOI if provided
    if raster_path:
        with rasterio.open(raster_path) as src:
            geoms = [f["geometry"] for f in aoi.to_crs(src.crs).__geo_interface__["features"]]
            #arr, transform = mask(src, geoms, crop=True, nodata=np.nan, filled=True)
            #arr = arr[0]
            masked_arr, transform = mask(src, geoms, crop=True, filled=False)
            masked_arr = masked_arr[0]                         
        masked = np.ma.masked_array(masked_arr.data,mask=masked_arr.mask)
        #masked = np.ma.masked_invalid(arr)
        height, width = masked.shape
        bounds = rasterio.transform.array_bounds(height, width, transform)
        xmin, xmax, ymin, ymax = bounds[0], bounds[2], bounds[1], bounds[3]
    else:
        xmin, ymin, xmax, ymax = aoi.total_bounds

    # Step 2: Initialize the figure and set limits
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
    ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
    
    # Step 3: Add basemap (under everything)
    try:
        ctx.add_basemap(ax, crs=aoi.crs.to_string(), source=ctx.providers.CartoDB.Positron, attribution_size=6, zorder=0)
    except Exception as e:
        print(f"[!] Could not add basemap: {e}")

    # Step 4: Plot raster above the basemap
    if masked is not None:
        if spec["type"] == "continuous":
            vmin, vmax = float(masked.min()), float(masked.max())
            im = ax.imshow(masked, cmap=spec["cmap"], extent=(xmin, xmax, ymin, ymax),
                           vmin=vmin, vmax=vmax, alpha=0.8, zorder=1)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(cmap=spec["cmap"], norm=norm)
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="3%", pad=0.1)
            #cbar = plt.colorbar(sm, cax=cax)
            #cbar.set_label(spec["label"])
        else:
            cmap = ListedColormap(spec["palette"])
            norm = BoundaryNorm(spec["breaks"], len(spec["palette"]))
            im = ax.imshow(masked, cmap=cmap, norm=norm,
                           extent=(xmin, xmax, ymin, ymax), origin='upper',
                           alpha=0.8, zorder=1)
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="3%", pad=0.1)
            #ticks = [(spec["breaks"][i] + spec["breaks"][i+1]) / 2 for i in range(len(spec["breaks"]) - 1)]
            #cbar = plt.colorbar(im, cax=cax, ticks=ticks)
            # cbar.ax.set_yticklabels(spec["labels"])
            #cbar.set_label(spec["label"])

    # Step 5: Plot vector data above raster
    aoi.boundary.plot(ax=ax, color="black", linewidth=1, zorder=2)

    if "infra_type" in points.columns and "exposed" in points.columns:
        for pt_type in points["infra_type"].unique():
            pts = points[points["infra_type"] == pt_type]
            style = custom_point_styles.get(pt_type, {"color": "gray", "marker": "o", "label": pt_type})
    
            # Not exposed: original color
            safe_plot(
                pts[~pts["exposed"]],
                ax,
                color=style["color"],
                marker=style["marker"],
                markersize=30,
                label=f"{style['label']} (not exposed)",
                zorder=5,
            )
            # Exposed: red
            safe_plot(
                pts[pts["exposed"]],
                ax,
                color="red",
                marker=style["marker"],
                markersize=30,
                label=f"{style['label']} (exposed)",
                zorder=6,
            )


    if "infra_type" in lines.columns and "exposed" in lines.columns:
        for ln_type in lines["infra_type"].unique():
            lns = lines[lines["infra_type"] == ln_type]
            style = custom_line_styles.get(ln_type, {"color": "gray", "linestyle": "-", "linewidth": 1.5, "label": ln_type})
    
            # Not exposed: original color
            safe_plot(
                lns[~lns["exposed"]],
                ax,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=f"{style['label']} (not exposed)",
                zorder=3,
            )
    
            # Exposed: red 
            safe_plot(
                lns[lns["exposed"]],
                ax,
                color="red",
                linewidth=style["linewidth"],
                label=f"{style['label']} (exposed)",
                zorder=4,
            )



    # Step 6: Final formatting
    ax.set_aspect("equal")
    ax.set_title(f"{hazard_name.replace('_', ' ').title()} Exposure", fontsize=15, fontweight="bold")
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend() 

    # Step 7: Save the figure
    if suffix:
        output_filename = f"exposure_map_{hazard_name}_{suffix}.png"
    else:
        output_filename = f"exposure_map_{hazard_name}.png"
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
    aoi.boundary.plot(ax=ax, color='black', linewidth=1, label='AOI')
    lines.plot(ax=ax, color='blue', linewidth=1, label='Lines')
    points.plot(ax=ax, color='red', markersize=10, label='Points')

    # Add basemap
    try:
        ctx.add_basemap(ax, crs=aoi.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"[!] Could not add basemap: {e}")

    ax.set_title("Initial Map: Energy Infrastructure", fontsize=15, fontweight="bold")
    ax.axis('off')
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_initial_map_by_type(aoi, points_by_type, lines_by_type, output_path):
    """
    Plot all infrastructure layers (points and lines) by type on a single map, with legend.

    Parameters:
        aoi (GeoDataFrame): Area of interest.
        points_by_type (dict): Dict of {type_name: GeoDataFrame} for point features.
        lines_by_type (dict): Dict of {type_name: GeoDataFrame} for line features.
        output_path (str): Path to save the output map image.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    # Custom styles for points
    custom_point_styles = {
        "substations": {"color": "#b2df8a", "marker": "^", "label": "Substation"},
        "transformer": {"color": "black", "marker": "o", "label": "Transformer"},
        "tower": {"color": "gray", "marker": "s", "label": "Tower"},
        "existing": {"color": "blue", "marker": "^", "label": "Existing substation"},
    }
    
    # Custom styles for lines
    custom_line_styles = {
        "hv": {"color": "yellow", "linestyle": "-", "linewidth": 2.5, "label": "High transmission line"},
        "lv": {"color": "#970499", "linestyle": "-", "linewidth": 1.5, "label": "Low transmission line"},
        "existing": {"color": "black", "linestyle": "--", "linewidth": 1.5, "label": "Existing line"},
    }
    
    # Plot AOI
    aoi.boundary.plot(ax=ax, color="black", linewidth=1, label="AOI", zorder=1)


    # Plot all point types
    for name, gdf in points_by_type.items():
        if gdf.empty or gdf.crs is None:
            continue
        style = custom_point_styles.get(name, {"color": "gray", "marker": "o", "label": f"Point: {name}"})
        gdf.plot(ax=ax, color=style["color"], marker=style["marker"],
                 markersize=30, label=style["label"], zorder=2)


    # Plot all line types
    for name, gdf in lines_by_type.items():
        if gdf.empty or gdf.crs is None:
            continue
        style = custom_line_styles.get(name, {"color": "gray", "linestyle": "-", "linewidth": 1.0, "label": f"Line: {name}"})
        gdf.plot(ax=ax, color=style["color"], linestyle=style["linestyle"],
                 linewidth=style["linewidth"], label=style["label"], zorder=3)


    # Add basemap
    try:
        ctx.add_basemap(ax, crs=aoi.crs.to_string(), source=ctx.providers.CartoDB.Positron, attribution_size=6)
    except Exception as e:
        print(f"[!] Could not load basemap: {e}")

    # Final styling
    ax.set_title("Initial Infrastructure Map", fontsize=16, fontweight="bold")
    ax.set_axis_off()
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


