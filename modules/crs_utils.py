"""
CRS (Coordinate Reference System) utilities for geospatial harmonization.

This module provides functions to:
- Reproject GeoDataFrames to a common CRS
- Ensure raster files are assigned or reprojected to WGS84 (EPSG:4326)
"""

import shutil
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def harmonize_crs(layers, target_crs):
    """
    Reproject a list of GeoDataFrames to a specified target CRS.

    Parameters:
        layers (list): List of GeoDataFrames to reproject.
        target_crs (str or CRS object): Target coordinate reference system (e.g., 'EPSG:4326').

    Returns:
        list: List of GeoDataFrames reprojected to the target CRS.
    """
    return [layer.to_crs(target_crs) for layer in layers]

def assign_or_reproject_to_wgs84(input_tif, output_tif=None, default_crs='EPSG:4326'):
    """
    Ensure a raster has WGS84 as its CRS. If needed, reproject it and write to a new file.

    Parameters:
        input_tif (str): Path to the input raster.
        output_tif (str, optional): Path to save the reprojected raster. 
                                    If None and reprojection is needed, creates a new file with '_wgs84' suffix.
        default_crs (str): CRS to assign or reproject to (default is 'EPSG:4326').

    Returns:
        str: Path to the raster file with the assigned or reprojected CRS.
    """
    def copy_or_return_input():
        """Return original file or copy to the specified output path if provided."""
        if output_tif is None:
            return input_tif
        shutil.copy(input_tif, output_tif)
        return output_tif

    with rasterio.open(input_tif, 'r+') as src:
        # If no CRS is defined, assign the default (WGS84)
        if src.crs is None:
            print(f"No CRS detected in {input_tif}. Assigning {default_crs}.")
            src.crs = default_crs
            return copy_or_return_input()
        
        # If already in the correct CRS, return original or copied file
        if src.crs.to_string() == default_crs:
            return copy_or_return_input()
         
        # Reproject raster to WGS84
        transform, width, height = calculate_default_transform(
            src.crs, default_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': default_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        # Define output path if not provided
        if output_tif is None:
            base, ext = input_tif.rsplit('.', 1)
            output_tif = f"{base}_wgs84.{ext}"

        # Perform reprojection and write output
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

        return output_tif
