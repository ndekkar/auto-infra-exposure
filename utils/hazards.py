import geopandas as gpd
import rasterio
import yaml
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def harmonize_crs(layers, target_crs):
    return [layer.to_crs(target_crs) for layer in layers]

def extract_flood_values_to_points(points_gdf, raster, threshold=0.0):
    points_gdf = points_gdf.copy()
    values = []
    for geom in points_gdf.geometry:
        try:
            val = next(raster.sample([(geom.x, geom.y)]))[0]
        except:
            val = None
        values.append(val)
    points_gdf["flood_value"] = values
    points_gdf["exposed"] = points_gdf["flood_value"].apply(
        lambda x: x is not None and x > threshold
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
        return any(v is not None and v > threshold for v in values)
    except:
        return False

def plot_exposure_map(country, points, lines):
    fig, ax = plt.subplots(figsize=(12, 12))
    country.boundary.plot(ax=ax, color="black", linewidth=1)
    lines[~lines["exposed"]].plot(ax=ax, color="gray", label="Lines not exposed")
    lines[lines["exposed"]].plot(ax=ax, color="orange", label="Exposed lines")
    points[~points["exposed"]].plot(ax=ax, color="green", markersize=10, label="Points not exposed")
    points[points["exposed"]].plot(ax=ax, color="red", markersize=10, label="Exposed points")
    ctx.add_basemap(ax, crs=country.crs.to_string())
    plt.legend()
    plt.title("Flood Exposure of Energy Network")
    plt.show()


