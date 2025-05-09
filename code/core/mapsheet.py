import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely

from pyproj import Transformer
from rasterio.mask import mask, geometry_mask
from rasterio.transform import rowcol
from shapely import intersection
from shapely.geometry import box, Polygon


class MapSheet:
    """
    Represents a map sheet with associated raster data and geographic extent, providing tools for processing and analysis.

    This class provides methods to read and process GeoTIFF files, extract map extents,
    split the map into tiles, and perform transformations between coordinate reference systems.

    Parameters (used during initialization):
        sheet_id (int): Unique identifier for the map sheet.
        path (str): Path to the corresponding GeoTIFF file.
        polygon (Polygon): Polygon representing the map sheet extent.
        polygon_crs (str): Coordinate Reference System of the polygon (e.g., 'EPSG:4326').

    Attributes:
        id (int): Unique identifier for the map sheet.
        path (str): Path to the corresponding GeoTIFF file.
        polygon (Polygon): Polygon representing the map sheet extent.
        polygon_crs (str): Coordinate Reference System of the polygon (e.g., 'EPSG:4326').
        crs (str): CRS of the GeoTIFF file.
        resolution (tuple): Resolution of the raster data.
        transform (Affine): Affine transformation matrix for the raster data.
        shape (tuple): Shape of the raster data (rows, columns).
        map_extent_polygon (Polygon): Polygon representing the map extent in the GeoTIFF CRS.
    """
    def __init__(self, sheet_id: int, path: str, polygon: Polygon, polygon_crs: str):
        """Initializes a MapSheet instance."""
        self.id = sheet_id
        self.path = path
        self.polygon = polygon
        self.polygon_crs = polygon_crs  # Set the polygon CRS

        # Load CRS and resolution from the GeoTIFF file
        with rasterio.open(path) as dataset:
            self.crs = dataset.crs.to_string()  
            self.resolution = dataset.res
            self.transform = dataset.transform
            self.shape = dataset.shape
            self.map_extent_polygon = self.get_polygon(target_crs=self.crs)

    def read_map_sheet(self):
        """Reads the map sheet and returns the numpy data array."""
        with rasterio.open(self.path) as src:
            data = src.read()
        return data

    def get_polygon(self, target_crs: str = None) -> Polygon:
        """Returns the polygon representing the map sheet extent, optionally transformed to a specified CRS."""
        if target_crs is not None and target_crs != self.polygon_crs:
            transformer = Transformer.from_crs(self.polygon_crs, target_crs, always_xy=True)
            transformed_polygon = shapely.ops.transform(transformer.transform, self.polygon)
            return transformed_polygon
        return self.polygon
    
    def extract_map_extent_rowcol(self):
        """Extracts the map sheet extent based on the polygon in row and col values.
        
        Returns:
            min_row (int): Minimum row index.
            max_row (int): Maximum row index.
            min_col (int): Minimum column index.
            max_col (int): Maximum column index.
        """
        map_extent_polygon = self.get_polygon(target_crs=self.crs)
        
        # Get the bounds of the cropped area
        min_x, min_y, max_x, max_y = map_extent_polygon.bounds

        # Use rowcol to get pixel indices
        min_row, min_col = rowcol(self.transform, min_x, max_y)
        max_row, max_col = rowcol(self.transform, max_x, min_y)

        return min_row, max_row, min_col, max_col

    def extract_map_extent(self, nodata=0):
        """
        Crops and masks the map sheet scan to its map extent.

        Parameters:
            nodata (int, optional): Value to use for nodata areas. Defaults to 0.

        Returns:
            tuple: Cropped image (numpy.ndarray) and its transformation (Affine).
       
        """
        with rasterio.open(self.path) as src:
            map_extent_polygon = self.get_polygon(target_crs=self.crs)
            cropped_image, cropped_transform = mask(src, [map_extent_polygon], crop=True, filled=True, nodata=nodata)

            return cropped_image, cropped_transform
        
    def extract_map_tile_info(self, tile_size):
        """
        Splits the map sheet extent into smaller tiles and extracts metadata for each tile.

        Parameters:
            tile_size (int): Size of each tile in pixels.

        Returns:
            GeoDataFrame: A GeoDataFrame containing tile metadata, geometries and random numbers.
        """
        min_row, max_row, min_col, max_col = self.extract_map_extent_rowcol()
        tiles = []
        # loop through the raster tiles
        for col_off in range(min_col, max_col, tile_size):
            for row_off in range(min_row, max_row, tile_size):
                # define the window for the current tile
                window = rasterio.windows.Window(col_off, row_off, tile_size, tile_size) 
                # get bounds in local map sheet projection
                window_bounds = rasterio.windows.bounds(window, self.transform)
                # crop windows at map sheet boundaries 
                window_polygon = intersection(box(*window_bounds), self.map_extent_polygon)
                
                # if there is no intersection with the map sheet extent continue with next tile
                # otherwise append tile to tile list
                if window_polygon.is_empty:
                    continue
                else:
                    tiles.append((self.id, window.col_off, window.row_off, tile_size, window_polygon))
        df_tiles = gpd.GeoDataFrame(tiles, columns=["map_id", "col_off", "row_off", "tile_size", "geometry"], crs=self.crs)

        # assign random number to each tile to use for sampling
        # use the map sheet id as seed for reproducibility
        rng = np.random.default_rng(seed=self.id)
        df_tiles["random_number"] = rng.uniform(size=len(df_tiles))
        return df_tiles

    def extract_map_tile(self, col_off, row_off, tile_size):
        """
        Extracts a specific tile from the map sheet.

        Args:
            col_off (int): Column offset of the tile.
            row_off (int): Row offset of the tile.
            tile_size (int): Size of the tile in pixels.

        Returns:
            numpy.ndarray: The raster data for the tile masked to the map sheet extent.
        """
        window = rasterio.windows.Window(col_off, row_off, tile_size, tile_size) 

        with rasterio.open(self.path) as src:
            data = src.read(window=window)
        
        # Mask out areas that are outside of the map extent
        # keeping the same shape and setting outside values to 0
        mask = geometry_mask([self.map_extent_polygon],
                             transform=src.window_transform(window),
                             invert=True,
                             out_shape=(tile_size, tile_size)
                             )

        masked_data = np.where(mask, data, 0)
        
        return masked_data

    def plot_map_sheet(self):
        """Plots the entire map sheet scan including the legend."""
        data = self.read_map_sheet()

        plt.figure(1, figsize=(10, 10))
        plt.imshow(data.transpose((1, 2, 0)))

    def plot_map_extent(self):
        """Plots the extent of the map sheet."""
        data, _ = self.extract_map_extent()

        plt.figure(1, figsize=(10, 10))
        plt.imshow(data.transpose((1, 2, 0)))

    def __repr__(self):
        """Returns a string representation of the MapSheet object for easier debugging."""
        return (f"MapSheet(id={self.id}, path={self.path}, crs={self.crs}, resolution={self.resolution}, "
                f"polygon_crs={self.polygon_crs}, polygon={self.polygon})")