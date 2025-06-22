import shutil
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def harmonize_crs(layers, target_crs):
    """
    Reproject a list of GeoDataFrames to a common CRS.
    """
    return [layer.to_crs(target_crs) for layer in layers]

def assign_or_reproject_to_wgs84(input_tif, output_tif=None, default_crs='EPSG:4326'):
    """
    Ensure a raster has a WGS84 CRS; reproject if needed. Optionally write to new file.
    """
    def copy_or_return_input():
        if output_tif is None:
            return input_tif
        shutil.copy(input_tif, output_tif)
        return output_tif

    with rasterio.open(input_tif, 'r+') as src:
        if src.crs is None:
            print(f"No CRS detected in {input_tif}. Assigning {default_crs}.")
            src.crs = default_crs
            return copy_or_return_input()

        if src.crs.to_string() == default_crs:
            return copy_or_return_input()

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
            base, ext = input_tif.rsplit('.', 1)
            output_tif = f"{base}_wgs84.{ext}"

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
