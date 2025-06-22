import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import os
import rasterio
from rasterio.mask import mask
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .config_utils import get_hazard_display_spec

def safe_plot(gdf, ax, **kwargs):
    """Plot GeoDataFrame only if it’s not empty."""
    if not gdf.empty:
        gdf.plot(ax=ax, **kwargs)

def add_raster_to_ax(ax, raster_path, aoi, hazard_name):
    """
    Add a stylized raster overlay to a Matplotlib axis.
    """
    spec = get_hazard_display_spec(hazard_name)

    with rasterio.open(raster_path) as src:
        aoi_proj = aoi.to_crs(src.crs)
        geometries = [feature["geometry"] for feature in aoi_proj.__geo_interface__["features"]]
        out_image, out_transform = mask(src, geometries, crop=True)
        out_image = out_image[0]
        out_image = np.where(out_image == src.nodata, np.nan, out_image)
        height, width = out_image.shape
        bounds = rasterio.transform.array_bounds(height, width, out_transform)
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    if spec["type"] == "continuous":
        vmin = np.nanmin(out_image)
        vmax = np.nanmax(out_image)
        im = ax.imshow(out_image, cmap=spec["cmap"], extent=extent, vmin=vmin, vmax=vmax, alpha=0.6)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=spec["cmap"], norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(spec["label"])
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['Low', 'High'])

    elif spec["type"] == "discrete":
        cmap = ListedColormap(spec["palette"])
        norm = BoundaryNorm(spec["breaks"], len(spec["palette"]))
        im = ax.imshow(out_image, cmap=cmap, norm=norm, extent=extent, origin='upper', alpha=0.6)
        tick_pos = [(spec["breaks"][i] + spec["breaks"][i+1]) / 2 for i in range(len(spec["breaks"]) - 1)]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, ticks=tick_pos)
        cbar.ax.set_yticklabels(spec["labels"])
        cbar.set_label(spec["label"])

    return extent

def plot_and_save_exposure_map(aoi, points, lines, hazard_name, output_dir, raster_path=None):
    """
    Plot infrastructure exposure map with optional raster background.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ctx.add_basemap(ax, crs=aoi.crs.to_string())

    if raster_path:
        extent = add_raster_to_ax(ax, raster_path, aoi, hazard_name)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    aoi.boundary.plot(ax=ax, color="black", linewidth=1)
    safe_plot(lines[~lines["exposed"]], ax, color="gray", label="Lines not exposed")
    safe_plot(lines[lines["exposed"]], ax, color="orange", label="Exposed lines")
    safe_plot(points[~points["exposed"]], ax, color="green", markersize=10, label="Points not exposed")
    safe_plot(points[points["exposed"]], ax, color="red", markersize=10, label="Exposed points")

    ax.set_aspect('equal')
    handles, labels = ax.get_legend_handles_labels()
    if handles and any(lbl for lbl in labels):
        ax.legend()
    ax.set_title(f"{hazard_name.replace('_',' ').title()} Exposure")

    output_path = os.path.join(output_dir, f"exposure_map_{hazard_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
