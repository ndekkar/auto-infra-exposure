import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import contextily as ctx

def process_wildfire(config):
    """
    Extract burned area centroids and generate a wildfire density map.
    """
    if "wildfire" in config["hazards"]:
        wildfire_conf = config["hazards"]["wildfire"]
        if wildfire_conf.get("active", False):
            print("\n--- Processing wildfire hazard ---")
            extract_burned_area_centroids(
                aoi_path=config["aoi"],
                globfire_dir=wildfire_conf["input"],
                output_path=os.path.join(config["output_dir"], "wildfire_centroids.gpkg")
            )
            plot_fire_density(
                gpkg_path=os.path.join(config["output_dir"], "wildfire_centroids.gpkg"),
                save_path=os.path.join(config["output_dir"], "wildfire_density.png")
            )

def extract_burned_area_centroids(aoi_path, globfire_dir, output_path):
    """
    Extract centroids of burned areas from GlobFire shapefiles within AOI.
    """
    aoi = gpd.read_file(aoi_path).to_crs(epsg=3857)
    aoi_buffered = aoi.buffer(10000)  # 10 km buffer
    aoi_geom = aoi_buffered.unary_union

    all_points = []

    for file in os.listdir(globfire_dir):
        if file.endswith('.shp'):
            shp_path = os.path.join(globfire_dir, file)
            try:
                gdf = gpd.read_file(shp_path).to_crs(epsg=3857)
                filtered = gdf[gdf.intersects(aoi_geom)]
                centroids = filtered.centroid
                centroids = centroids.to_crs(epsg=4326)
                all_points.extend([Point(pt.x, pt.y) for pt in centroids])
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not all_points:
        print(" No burned area centroids found.")
        return

    output_gdf = gpd.GeoDataFrame(geometry=all_points, crs='EPSG:4326')
    output_gdf.to_file(output_path, driver='GPKG', layer='burned_area_centroids')
    print(f" Burned area centroids saved to: {output_path}")

def plot_fire_density(gpkg_path, save_path=None):
    """
    Generate a KDE density heatmap of wildfire centroids.
    """
    gdf = gpd.read_file(gpkg_path)
    gdf_web = gdf.to_crs(epsg=3857)
    df = pd.DataFrame({
        'x': gdf_web.geometry.x,
        'y': gdf_web.geometry.y
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    kde = sns.kdeplot(
        data=df, x='x', y='y', fill=True,
        cmap='Reds', bw_adjust=0.5, levels=100, thresh=0.05, ax=ax
    )

    mappable = kde.collections[0]
    cbar = plt.colorbar(mappable, ax=ax, label="Wildfire Density")
    ctx.add_basemap(ax, crs='EPSG:3857')
    ax.set_title("Wildfire Density Heatmap")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
