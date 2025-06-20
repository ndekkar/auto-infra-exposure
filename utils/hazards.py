import geopandas as gpd
import rasterio
import yaml
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import os
import pandas as pd
from shapely.geometry import Point
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geemap
import ee
import datetime
import rasterio.plot

# ---------------- CONFIG LOADER ----------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------- CRS HANDLING ----------------
def harmonize_crs(layers, target_crs):
    return [layer.to_crs(target_crs) for layer in layers]

def assign_or_reproject_to_wgs84(input_tif, output_tif=None, default_crs='EPSG:4326'):
    with rasterio.open(input_tif, 'r+') as src:
        if src.crs is None:
            print(f"No CRS detected in {input_tif}. Assigning {default_crs}.")
            src.crs = default_crs
            if output_tif is None:
                return input_tif
            else:
                import shutil
                shutil.copy(input_tif, output_tif)
                return output_tif

        if src.crs.to_string() == default_crs:
            if output_tif is None:
                return input_tif
            else:
                import shutil
                shutil.copy(input_tif, output_tif)
                return output_tif

        print(f"Reprojecting {input_tif} to {default_crs}...")
        transform, width, height = calculate_default_transform(
            src.crs, default_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': default_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        if output_tif is None:
            base, ext = os.path.splitext(input_tif)
            output_tif = f"{base}_wgs84{ext}"

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=default_crs,
                    resampling=Resampling.nearest)

        print(f"Reprojection complete: {output_tif}")
        return output_tif

# ---------------- EXPOSURE CALCULATION ----------------
def extract_values_to_points(points_gdf, raster, threshold=0.0):
    points_gdf = points_gdf.copy()
    values = []
    for geom in points_gdf.geometry:
        try:
            val = next(raster.sample([(geom.x, geom.y)]))[0]
        except:
            val = None
        values.append(val)
    points_gdf["haz_val"] = values
    points_gdf["exposed"] = points_gdf["haz_val"].apply(
        lambda x: x is not None and x >= threshold
    )
    return points_gdf

def check_line_exposure(line_geom, raster, n_points=10, threshold=0.0):
    if line_geom.is_empty or line_geom.length == 0:
        return False
    distances = np.linspace(0, 1, n_points)
    points = [line_geom.interpolate(d, normalized=True) for d in distances]
    coords = [(p.x, p.y) for p in points]
    try:
        values = [val[0] for val in raster.sample(coords)]
        return any(v is not None and v >= threshold for v in values)
    except:
        return False

# ---------------- RASTER COMBINATION ----------------
def combine_rasters(raster_paths, output_path, method='max'):
    with rasterio.open(raster_paths[0]) as src_ref:
        meta = src_ref.meta
        data = src_ref.read(1).astype(float)

    for path in raster_paths[1:]:
        with rasterio.open(path) as src:
            data_new = src.read(1).astype(float)
            if method == 'max':
                data = np.maximum(data, data_new)
            elif method == 'sum':
                data = np.nan_to_num(data) + np.nan_to_num(data_new)
            elif method == 'mean':
                data = (data + data_new) / 2
            else:
                raise ValueError("Invalid combination method.")

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)

    print(f"✅ Combined raster written to: {output_path}")

# ---------------- SAFE PLOTTING AND SAVING ----------------
def safe_plot(gdf, ax, **kwargs):
    if not gdf.empty:
        gdf.plot(ax=ax, **kwargs)

def plot_and_save_exposure_map(aoi, points, lines, hazard_name, output_dir):
    fig, ax = plt.subplots(figsize=(12, 12))
    aoi.boundary.plot(ax=ax, color="black", linewidth=1)
    safe_plot(lines[~lines["exposed"]], ax, color="gray", label="Lines not exposed")
    safe_plot(lines[lines["exposed"]], ax, color="orange", label="Exposed lines")
    safe_plot(points[~points["exposed"]], ax, color="green", markersize=10, label="Points not exposed")
    safe_plot(points[points["exposed"]], ax, color="red", markersize=10, label="Exposed points")
    ctx.add_basemap(ax, crs=aoi.crs.to_string())
    plt.legend()
    plt.title(f"{hazard_name.replace('_',' ').title()} Exposure")
    output_path = os.path.join(output_dir, f"exposure_map_{hazard_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ---------------- DROUGHT NDMI GENERATION ----------------
def generate_drought_index(aoi_path, output_dir, drought_conf):
    ee.Initialize()

    # Dates
    start = datetime.datetime(drought_conf["start_year"], drought_conf["start_month"], 1)
    end = datetime.datetime(drought_conf["end_year"], drought_conf["end_month"], 28)

    # AOI
    aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
    bounds = aoi.total_bounds
    geom = ee.Geometry.BBox(*bounds)

    # Utilisation de MODIS/061/MOD13Q1 et calcul NDMI
    collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(start, end) \
        .map(lambda img: img.normalizedDifference(['sur_refl_b02', 'sur_refl_b07']).rename('NDMI'))

    mean_image = collection.select('NDMI').mean().clip(geom)
    scale = 250

    out_path = os.path.join(output_dir, "drought_index_ndmi.tif")
    geemap.ee_export_image(mean_image, filename=out_path, scale=scale, region=geom)
    print(f"✅ Exported drought index raster to: {out_path}")

    # --- Affichage avec symbologie continue ---
    try:
        with rasterio.open(out_path) as src:
            fig, ax = plt.subplots(figsize=(12, 10))
            rasterio.plot.show(src, ax=ax, cmap='RdYlGn', vmin=-1, vmax=1)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.colors as colors

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            norm = colors.Normalize(vmin=-1, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
            sm.set_array([])
            fig.colorbar(sm, cax=cax, label="NDMI")

            plt.title("Drought Index (NDMI)")
            plt.axis('off')
            plt.tight_layout()

            map_path = os.path.join(output_dir, "drought_map_ndmi.png")
            plt.savefig(map_path, dpi=300)
            plt.show() 
            print(f"✅ NDMI map saved to: {map_path}")
            plt.close()

    except Exception as e:
        print(f"❌ Unable to plot drought map: {e}")

# ---------------- MAIN PIPELINE ----------------
def run_multi_hazard_pipeline(config_path):
    config = load_config(config_path)

    aoi = gpd.read_file(config["aoi_shapefile"])
    aoi_union = aoi.union_all()
    points = gpd.read_file(config["power_points_shapefile"])
    lines = gpd.read_file(config["power_lines_shapefile"])

    points, lines = harmonize_crs([points, lines], aoi.crs)
    points = points[points.geometry.within(aoi_union)]
    lines = lines[lines.geometry.intersects(aoi_union)]

    if "drought" in config["hazards"]:
        drought_conf = config["hazards"]["drought"]
        if drought_conf.get("active", False):
            print("\n--- Processing drought index (NDMI) ---")
            generate_drought_index(config["aoi_shapefile"], config["output_dir"], drought_conf)

    hazard_rasters = {}

    for hazard_name, hazard_conf in config["hazards"].items():
        if not hazard_conf.get("active", False) or hazard_name == "drought":
            print(f"Skipping hazard: {hazard_name}")
            continue

        print(f"\n--- Processing hazard: {hazard_name} ---")
        raster_path_wgs84 = assign_or_reproject_to_wgs84(hazard_conf["raster"])
        raster = rasterio.open(raster_path_wgs84)
        threshold = hazard_conf["threshold"]

        hazard_rasters[hazard_name] = (raster_path_wgs84, threshold)

        points_hazard = extract_values_to_points(points, raster, threshold)
        points_out = f"{config['output_dir']}/points_exposure_{hazard_name}.shp"
        points_hazard.to_file(points_out)

        lines_hazard = lines.copy()
        lines_hazard["exposed"] = lines_hazard["geometry"].apply(
            lambda geom: check_line_exposure(geom, raster, config["sample_points_per_line"], threshold)
        )
        lines_out = f"{config['output_dir']}/lines_exposure_{hazard_name}.shp"
        lines_hazard.to_file(lines_out)

        plot_and_save_exposure_map(aoi, points_hazard, lines_hazard, hazard_name, config["output_dir"])

    if "pluvial_flood" in hazard_rasters and "fluvial_flood" in hazard_rasters:
        print("\n--- Processing combined flood ---")
        pluvial_path, pluvial_threshold = hazard_rasters["pluvial_flood"]
        fluvial_path, fluvial_threshold = hazard_rasters["fluvial_flood"]

        combined_path = f"{config['output_dir']}/combined_flood.tif"
        combine_rasters([pluvial_path, fluvial_path], combined_path, method='max')
        combined_raster = rasterio.open(combined_path)
        combined_threshold = max(pluvial_threshold, fluvial_threshold)

        points_combined = extract_values_to_points(points, combined_raster, combined_threshold)
        points_out = f"{config['output_dir']}/points_exposure_combined_flood.shp"
        points_combined.to_file(points_out)

        lines_combined = lines.copy()
        lines_combined["exposed"] = lines_combined["geometry"].apply(
            lambda geom: check_line_exposure(geom, combined_raster, config["sample_points_per_line"], combined_threshold)
        )
        lines_out = f"{config['output_dir']}/lines_exposure_combined_flood.shp"
        lines_combined.to_file(lines_out)

        plot_and_save_exposure_map(aoi, points_combined, lines_combined, "combined_flood", config["output_dir"])
