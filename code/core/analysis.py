import geopandas as gpd
import rasterio
import rioxarray

from exactextract import exact_extract
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape


def calc_luc_frac(raster, vector, include_cols, mapping):
    """
    Calculates the land use/land cover (LUC) fractions for a given raster and vector.

    Parameters:
        raster (str): Path to the raster file.
        vector (GeoDataFrame): GeoDataFrame containing the vector data.
        include_cols (str or list): Column(s) to include in the output.
        mapping (dict): Mapping of raster values to LUC classes.

    Returns:
        DataFrame: A DataFrame containing LUC fractions and pixel counts across each polygon in the vector file.
    """
    df = exact_extract(raster, vector, ["count(default_value=0)", "unique(default_value=0)", "frac(default_value=0)"], include_cols=include_cols, output="pandas")
    
    # Get pixel counts
    pixels = df[[include_cols, "count"]] if isinstance(include_cols, str) else df[include_cols + ["count"]]
    pixels = pixels.set_index(include_cols)

    df = df.explode(["unique", "frac"])
    df["frac"] = df["frac"].astype("float")
    df["luc"] = df["unique"].map(mapping)
    # Create dummy column for NaN values that is dropped later
    # always ensure mapping includes all possible values 
    # otherwise this will lead to duplicates for NaN values
    df.loc[df["unique"].isna(), ["luc"]] = "nodata"

    # aggregate by luc type in case multiple raster values were mapped to the same luc type
    group_cols = [include_cols, "luc", "count"] if isinstance(include_cols, str) else [*include_cols, "luc", "count"]
    df = df.groupby(group_cols, as_index=False).agg({"frac": "sum"})
    
    # Pivot from long to wide format and fill missing values with a 
    # fraction of 0 (unless all values are missing than keep NaN)
    df = df.pivot(index=include_cols, columns="luc", values="frac")
    df.where(df.isnull().all(axis=1), df.fillna(0), inplace=True)

    # Drop dummy column for grid cells that don't overlap the raster
    if "nodata" in df.columns:
        df.drop("nodata", axis=1, inplace=True)

    # Join pixel counts back to frac data
    df = df.join(pixels)
    
    return df.astype("float")

def calc_lucc_stats(gdf, index_cols, src_raster, dst_raster, src_mapping, dst_mapping, src_class, dst_class, pixel_area):
    """
    Calculates land use/land cover change (LUCC) statistics for a given GeoDataFrame and two rasters.

    Parameters:
        gdf (GeoDataFrame): GeoDataFrame containing the study areas.
        index_cols (str or list): Column(s) to use as the index.
        src_raster (str): Path to the source raster file.
        dst_raster (str): Path to the destination raster file.
        src_mapping (dict): Mapping of source raster values to LUC classes.
        dst_mapping (dict): Mapping of destination raster values to LUC classes.
        src_class (str): Source LUC class to analyze.
        dst_class (str): Destination LUC class to analyze.
        pixel_area (float): Area of a single pixel in square kilometers.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with LUCC statistics.
    """    
    gdf = gdf.copy()
    gdf = gdf.set_index(index_cols)
    gdf[index_cols] = gdf.index
    gdf["area"] = gdf.geometry.area / 1e6 # area in km2
    
    # Calculate fractions for source and destination rasters
    src_luc_frac = calc_luc_frac(src_raster, gdf, index_cols, src_mapping)
    dst_luc_frac = calc_luc_frac(dst_raster, gdf, index_cols, dst_mapping)

    # Merge class percentage and calculate percentage change
    gdf["src_perc"] = gdf.join(src_luc_frac)[src_class] * 100
    gdf["dst_perc"] = gdf.join(dst_luc_frac)[dst_class] * 100
    gdf["change_perc_total_area"] = gdf["dst_perc"] - gdf["src_perc"]
    gdf["change_perc_src_area"] = (gdf["dst_perc"] / gdf["src_perc"] - 1) * 100

    # Calculate area and area change
    gdf["src_pixels"] = gdf.join(src_luc_frac)["count"]
    gdf["dst_pixels"] = gdf.join(dst_luc_frac)["count"]
    gdf["src_area"] = gdf["src_perc"] * gdf["src_pixels"] * pixel_area / 100
    gdf["dst_area"] = gdf["dst_perc"] * gdf["dst_pixels"] * pixel_area / 100
    gdf["change_area"] = gdf["dst_area"] - gdf["src_area"]

    return gdf

def create_change_map(src_raster_path, dst_raster_path, output_path, src_raster_vals, dst_raster_exclude=None, dst_raster_include=None, nodata_value=0, cmap=None):
    """
    Creates a change map by comparing two rasters and applying a mask.

    Parameters:
        src_raster_path (str): Path to the source raster file.
        dst_raster_path (str): Path to the destination raster file.
        output_path (str): Path to save the output raster file.
        src_raster_vals (list): List of values to include from the source raster.
        dst_raster_exclude (list, optional): List of values to exclude from the destination raster.
        dst_raster_include (list, optional): List of values to include from the destination raster.
        nodata_value (int, optional): Value to use for nodata pixels. Defaults to 0.
        cmap (dict, optional): Colormap to apply to the output raster.

    Returns:
        None
    """    
    # Open the rasters with chunking
    src_raster = rioxarray.open_rasterio(src_raster_path, chunks="auto")
    dst_raster = rioxarray.open_rasterio(dst_raster_path, chunks="auto")

    # Create the mask for the desired values
    if dst_raster_exclude:
        mask = (src_raster.isin(src_raster_vals)) & (dst_raster.isin(dst_raster_exclude) == False)
    elif dst_raster_include:
        mask = (src_raster.isin(src_raster_vals)) & (dst_raster.isin(dst_raster_include))
    else:
        mask = src_raster.isin(src_raster_vals)

    # Create a copy of dst_raster
    dst_raster_output = dst_raster.copy()

    # Apply the mask to dst_raster_output: Set values to nodata where mask is False
    dst_raster_output = dst_raster_output.where(mask == 1, nodata_value)

    # Write the modified dst_raster to the output path
    (dst_raster_output
        .rio.write_nodata(input_nodata=nodata_value)
        .rio.to_raster(output_path, compute=True, compress="LZW", tiled=True, dtype="uint8"))

    if cmap:
        # Write colormap 
        with rasterio.open(output_path, "r+") as dst:
            dst.write_colormap(1, cmap)
            
    print(f"Output written to {output_path}")

def clip_and_polygonize_raster(raster_path, raster_idx, extent, extent_crs="EPSG:4326"):
    """
    Clips a raster to a study area and polygonizes the specified index.

    Parameters:
        raster_path (str): Path to the raster file.
        raster_idx (int): The raster value to polygonize.
        extent (Polygon): The study area polygon.
        extent_crs (str, optional): The CRS for the extent polygon. Defaults to "EPSG:4326".

    Returns:
        GeoDataFrame: A GeoDataFrame containing the polygonized areas.
    """
    with rioxarray.open_rasterio(raster_path) as raster:

        # Open and clip the raster using rioxarray
        clipped_raster = raster.rio.clip(
            [extent], crs=extent_crs, from_disk=True, drop=True
        )

        # Convert the raster to a NumPy array and extract the first band
        raster_data = clipped_raster.values[0]  # Assuming a single-band raster

        # Create a mask for the specified index
        mask = raster_data == raster_idx

        # Polygonize the raster using rasterio.features.shapes
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for s, v in shapes(raster_data, mask=mask, transform=clipped_raster.rio.transform())
        )

        # Convert the polygons to a GeoDataFrame
        geoms = [shape(feature['geometry']) for feature in results]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': geoms}, crs=clipped_raster.rio.crs)

    return gdf

def reproject_raster(input_path, output_path, target_crs, resolution, resampling=Resampling.nearest):
    """
    Reprojects a raster to a target CRS and resolution.

    Parameters:
        input_path (str): Path to the input raster file.
        output_path (str): Path to save the reprojected raster file.
        target_crs (str): Target CRS for the reprojected raster.
        resolution (float): Target resolution for the reprojected raster.
        resampling (Resampling, optional): Resampling method. Defaults to Resampling.nearest.

    Returns:
        None
    """    
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        
        kwargs = src.profile.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })
        
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling
                )
                
def reproject_align_raster(src_path, ref_path, dst_path, resampling=Resampling.mode, colormap=None):
    """
    Reprojects a raster to match the resolution, CRS, and alignment of a reference raster.

    Parameters:
        src_path (str): Path to the source raster file.
        ref_path (str): Path to the reference raster file.
        dst_path (str): Path to save the aligned raster file.
        resampling (Resampling, optional): Resampling method. Defaults to Resampling.mode.
        colormap (dict, optional): Colormap to apply to the output raster.

    Returns:
        None
    """    
    with rasterio.open(ref_path) as ref:
        with rasterio.open(src_path) as src:
            src_profile = src.profile.copy()
            transform, width, height = calculate_default_transform(
                src.crs, ref.crs, ref.width, ref.height, *ref.bounds
            )
            transform, width, height = rasterio.warp.aligned_target(transform, width, height, ref.res)    

            # Update profile based on refernce file, keep some of the src file attributes
            dst_profile = ref.profile.copy()
            dst_profile["nodata"] = src_profile["nodata"]
            dst_profile["dtype"] = src_profile["dtype"]
            dst_profile["compress"] = src_profile["compress"] 

            with rasterio.open(dst_path, "w", **dst_profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=ref.crs,
                    resampling=resampling, 
                )

                if colormap:
                    print("Writing colormap")
                    dst.write_colormap(1, colormap)