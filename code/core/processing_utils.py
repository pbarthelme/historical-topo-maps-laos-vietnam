import numpy as np
import rasterio

from rasterio.mask import geometry_mask
from scipy import ndimage
from skimage.measure import label, regionprops


def mask_raster(raster, polygon, transform, nodata):
    """
    Masks a raster image using a polygon.

    Parameters:
        raster: numpy array of shape (height, width) or (channels, height, width)
        polygon: Shapely polygon used for masking
        transform: Affine transform of the raster
        nodata: Value to assign to masked areas

    Returns:
        Masked raster as a numpy array.
    """
    if len(raster.shape) == 2:
        height, width = raster.shape[0], raster.shape[1]
    else:
        height, width = raster.shape[1], raster.shape[2]

    mask = geometry_mask(
        [polygon],
        transform=transform,
        invert=True,
        out_shape=(height, width)
        )
    masked_raster = np.where(mask, raster, nodata)

    return masked_raster

def split_into_tiles(image, tile_size=256, overlap=64):
    """
    Splits the input image into overlapping windows.

    Parameters:
        image: numpy array of shape (channels, height, width)
        tile_size: Size of each tile (in pixels)
        overlap: Number of overlapping pixels between adjacent tiles

    Returns:
        List of window arrays.
    """
    height, width = image.shape[1], image.shape[2]
    
    row_offsets = list(range(0, height - tile_size, tile_size - 2*overlap)) + [height-tile_size]
    col_offsets = list(range(0, width - tile_size, tile_size - 2*overlap)) + [width-tile_size]

    tiles = []
    for i in row_offsets:
        for j in col_offsets:
            tile = image[..., i:i+tile_size, j:j+tile_size]
            tiles.append(tile)

    return tiles

def stitch_tiles(windows, image_shape, tile_size=256, overlap=64):
    """
    Stitches windows back together to form the original image.

    Parameters:
        windows: List of processed window arrays
        image_shape: Tuple (channels, height, width)
        tile_size: Size of each tile (in pixels)
        overlap: Number of overlapping pixels between adjacent tiles

    Returns:
        Numpy array representing the stitched image.
    """
    height, width = image_shape[1], image_shape[2]
    if len(windows[0].shape) == 2:
        out_shape = (height, width)
    else:
        out_shape = (windows[0].shape[0], height, width)
    image = np.empty(out_shape, dtype=windows[0].dtype)

    row_offsets = list(range(0, height - tile_size, tile_size - 2*overlap)) + [height-tile_size]
    col_offsets = list(range(0, width - tile_size, tile_size - 2*overlap)) + [width-tile_size]

    index = 0
    for i in row_offsets:
        for j in col_offsets:
            # only write center of each predicted tile 
            i_start, w_i_start = (i, 0) if i == row_offsets[0] else (i + overlap, overlap)
            i_end, w_i_end = (i + tile_size, tile_size)  if i == row_offsets[-1] else (i + tile_size - overlap, tile_size - overlap) 
            j_start, w_j_start = (j, 0) if j == col_offsets[0] else (j + overlap, overlap)
            j_end, w_j_end = (j + tile_size, tile_size) if j == col_offsets[-1] else (j + tile_size - overlap, tile_size - overlap)
                
            image[..., i_start:i_end, j_start:j_end] = windows[index][..., w_i_start:w_i_end, w_j_start:w_j_end]
            index += 1

    return image

def extend_to_full_map_sheet(img, map_sheet, nodata):
    """
    Extends an image to match the full size of a map sheet.

    Parameters:
        img: numpy array of shape (channels, height, width) or (height, width)
        map_sheet: MapSheet object containing the full map extent
        nodata: Value to assign to areas outside the image

    Returns:
        Extended image as a numpy array.
    """
    if len(map_sheet.shape) == 2:
        height, width = map_sheet.shape[0], map_sheet.shape[1]
    else:
        height, width = map_sheet.shape[1], map_sheet.shape[2]

    if len(img.shape) == 2:
        out_shape = (height, width)
    else:
        out_shape = (img.shape[0], height, width)
    
    img_full = np.full(out_shape, nodata, dtype=img.dtype)
    min_row, max_row, min_col, max_col = map_sheet.extract_map_extent_rowcol()

    img_full[..., min_row:(max_row+1), min_col:(max_col+1)] = img

    return img_full

def process_map_sheet(
        map_sheet,
        tile_proc_fn=None,
        img_proc_fn=None,
        crop_to_map_extent=False, 
        proc_tiles_together=False,
        mask_map_extent=False,
        tile_size=None,
        overlap=None,
        nodata=0,
        ):
    """
    Applies processing functions to a map sheet with options for cropping, tiling and masking.

    Parameters:
        map_sheet: MapSheet object to process
        tile_proc_fn: Function to process individual tiles
        img_proc_fn: Function to process the entire image
        crop_to_map_extent: Whether to crop the image to the map extent
        proc_tiles_together: Whether to process tiles together as a batch
        mask_map_extent: Whether to mask the image to the map extent
        tile_size: Size of each tile (in pixels)
        overlap: Number of overlapping pixels between adjacent tiles
        nodata: Value to assign to areas outside the image

    Returns:
        Processed image as a numpy array.
    """
    if crop_to_map_extent:
        img, transform = map_sheet.extract_map_extent(nodata=nodata)
    else:
        img = map_sheet.read_map_sheet()
        transform = map_sheet.transform

    if tile_proc_fn:
        if not tile_size or not overlap:
            raise ValueError("tile_size and overlap must be specified when processing tiles.")
        img_tiles = split_into_tiles(img, tile_size, overlap)
        if proc_tiles_together:
            tiles_processed = tile_proc_fn(img_tiles)
        else:
            tiles_processed = [tile_proc_fn(tile) for tile in img_tiles]
        img = stitch_tiles(tiles_processed, img.shape, tile_size, overlap)

    if img_proc_fn:
        img = img_proc_fn(img)

    if crop_to_map_extent:
        img = extend_to_full_map_sheet(img, map_sheet, nodata)
        # set transform back to original map sheet transform 
        # in case it was changed previously to cropped extent
        transform = map_sheet.transform

    if mask_map_extent:
        img = mask_raster(
            img,
            map_sheet.map_extent_polygon,
            transform, 
            nodata=nodata
            )
        
    return np.squeeze(img)

def write_processed_image(map_sheet, img, output_file, profile_update=None, cmap=None):
    """
    Writes a processed image to a file.

    Parameters:
        map_sheet: MapSheet object containing metadata
        img: numpy array of the processed image
        output_file: Path to save the output file
        profile_update: Dictionary of updates to the raster profile
        cmap: Colormap to apply to the output file

    Returns:
        None
    """
    with rasterio.open(map_sheet.path) as src:
        profile = src.profile.copy()
        if profile_update:
            profile.update(profile_update)
    
    with rasterio.open(output_file, 'w', **profile) as dst:
        if len(img.shape) == 2:
            dst.write(img, 1)
        else:
            dst.write(img)

    if cmap:
        with rasterio.open(output_file, "r+") as dst:
            dst.write_colormap(1, cmap)

def map_array_values(array, value_map):
    """
    Maps values in a NumPy array according to a given mapping.

    Parameters:
        array: NumPy array of integers
        value_map: Dictionary mapping from integer to integer

    Returns:
        Mapped NumPy array.
    """
    # Create a copy of the array to avoid modifying the original array
    mapped_array = np.copy(array)
    
    # Create a vectorized function with a default value
    def map_value(x):
        return value_map.get(x, x)
    
    vectorized_map = np.vectorize(map_value)
    mapped_array = vectorized_map(mapped_array)
    
    return mapped_array


def impute_nearest_nonvalue(image, values):
    """
    Replaces specified values in an image by imputing the nearest non-value.

    Parameters:
        image: numpy array of the image
        values: List of values to replace

    Returns:
        Imputed image as a numpy array.
    """    
    # Create a binary mask of where the values are zero
    mask = np.isin(image, values)

    # Use distance transform to find nearest non-zero value for each zero-valued pixel
    distance, nearest_nonzero_idx = ndimage.distance_transform_edt(mask, return_indices=True)

    # Use nearest non-zero index to get values from the original image
    nearest_values = image[tuple(nearest_nonzero_idx)]

    # Replace the zero pixels with their nearest non-zero values
    imputed_image = image.copy()
    imputed_image[mask] = nearest_values[mask]

    return imputed_image

def remove_small_connected_areas_all_values(image, min_size):
    """
    Removes connected areas of any value smaller than a given size.

    Parameters:
        image: numpy array of the image
        min_size: Minimum size of connected areas to retain

    Returns:
        Image with small connected areas removed.
    """
    # Label all connected components in the image at once
    labeled_image, _ = label(image, connectivity=1, return_num=True)
    
    # Get the properties of each labeled region (size, area, etc.)
    regions = regionprops(labeled_image)

    # Copy the image to modify
    result_image = image.copy()

    # Iterate over each labeled region and remove small ones
    for region in regions:
        if region.area < min_size:
            # Set all pixels in this small region to 0 (or background)
            result_image[labeled_image == region.label] = 0

    return result_image

def impute_polygon_overlap(img, polygon, transform, impute_value, mask_class_id=None):
    """
    Imputes a specified value for areas overlapping with a polygon.

    Parameters:
        img: numpy array of the image
        polygon: Shapely polygon used for overlap
        transform: Affine transform of the image
        impute_value: Value to assign to overlapping areas
        mask_class_id: Class ID to restrict the imputation (optional)

    Returns:
        Image with imputed values as a numpy array.
    """
    if len(img.shape) == 2:
        height, width = img.shape[0], img.shape[1]
    else:
        height, width = img.shape[1], img.shape[2]

    overlap = geometry_mask(
        [polygon],
        transform=transform,
        invert=True,
        out_shape=(height, width)
        )
    
    if mask_class_id is not None:
        mask_class = img == mask_class_id
        overlap = np.logical_and(overlap, mask_class)

    img_imputed = np.where(overlap, impute_value, img)
    return img_imputed