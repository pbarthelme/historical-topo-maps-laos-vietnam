import glob
import pandas as pd
import rasterio
from rasterio.merge import merge
from shapely.geometry import box

from shapely.geometry import Polygon


def create_polygon(lon_min, lat_min, lon_max, lat_max):
    """Creates a polygon from four corner coordinates."""
    return Polygon([
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_max),
        (lon_max, lat_min),
        (lon_min, lat_min)
    ])

def get_mapping_from_csv(path, col_key, col_value, convert_rgba=False):
    """
    Reads a CSV file and creates a mapping between two specified columns.

    Parameters:
        path (str): Path to the CSV file.
        col_key (str): Column name to use as keys in the mapping.
        col_value (str): Column name to use as values in the mapping.
        convert_rgba (bool, optional): If True, converts string RGBA values in the `col_value` column 
                                       to tuples of integers. Defaults to False.

    Returns:
        dict: A dictionary mapping `col_key` values to `col_value` values.
    """
    df = pd.read_csv(path)
    if convert_rgba:
        df[col_value] = df[col_value].apply(lambda x: tuple(map(int, x.strip('()').split(','))))
    mapping = {key: value for key, value in zip(df[col_key], df[col_value])}

    return mapping

def merge_tifs_within_study_area(input_folder, output_file, study_area, file_prefix="", indexes=[1]):
    """
    Merges all TIFF files in the input folder that intersect with the study area into a single raster and saves it.

    Parameters:
        input_folder (str): Path to the folder containing tiled TIFF files.
        output_file (str): Path to save the merged raster file.
        study_area (GeoDataFrame): GeoDataFrame containing the study area geometry.
        file_prefix (str, optional): Prefix to filter TIFF files in the input folder. Defaults to an empty string, meaning all files are considered.
        indexes (list, optional): List of band indexes to include in the merged raster. Defaults to [1] meaning only the first band is merged.

    Returns:
        None
    """
    study_area_bounds = study_area.union_all()

    # Find all .tif files in the directory
    tif_files = glob.glob(f"{input_folder}/{file_prefix}*.tif")
    if not tif_files:
        print("No TIFF files found in the specified directory.")
        return

    # Filter TIFF files by intersection with the study area
    intersecting_files = []
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            # Check if CRS matches
            if src.crs != study_area.crs:
                raise ValueError(
                    f"CRS mismatch: Raster file {tif_file} has CRS {src.crs}, "
                    f"but the study area has CRS {study_area.crs}."
                )
            tif_bounds = box(*src.bounds)
            if tif_bounds.intersects(study_area_bounds):
                intersecting_files.append(tif_file)

    if not intersecting_files:
        print("No TIFF files intersect with the study area.")
        return

    # Open all intersecting TIFF files as datasets
    datasets = [rasterio.open(f) for f in intersecting_files]

    # Merge the datasets
    merged, transform = merge(datasets, indexes=indexes)

    # Copy metadata from the first dataset
    profile = datasets[0].profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": len(indexes),
        "height": merged.shape[1],
        "width": merged.shape[2],
        "transform": transform
    })

    # Write the merged dataset to a new file
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(merged)

    # Close all opened datasets
    for ds in datasets:
        ds.close()

    print(f"Merged TIFF saved as {output_file}")