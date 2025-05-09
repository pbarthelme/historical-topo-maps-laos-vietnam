import argparse
import concurrent.futures
import functools
import gc
import glob
import geopandas as gpd
import numpy as np
import os
import shutil
import torch

from config import Config
from core.dataloader import TopoMapsDataModule
from core.mapsheet import MapSheet
from core.model import SegmentationModelLightning
from core.processing_utils import process_map_sheet, map_array_values, remove_small_connected_areas_all_values, impute_nearest_nonvalue, impute_polygon_overlap, write_processed_image
from core.utils import get_mapping_from_csv


def setup_crop_map_sheet(config):
    """Loads parameters for cropping of map sheets."""
    return {
        "nodata": config.nodata,
        "compress": "DEFLATE"
    }

def setup_pred_map_sheet(config):
    """Loads parameters and data for map sheet prediction."""
    cmap = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="color_rgba", convert_rgba=True)
    checkpoint_files = glob.glob(f"{config.final_model_path}/*.ckpt")
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file found in {config.final_model_path}")
    model = SegmentationModelLightning.load_from_checkpoint(checkpoint_files[0])
    model.eval()

    return {
        "model": model,
        "cmap": cmap,
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "nodata": config.nodata,
        "compress": config.compress
    }

def setup_postprocess1(config):
    """Loads parameters and data for postprocessing step 1."""
    msi = gpd.read_file(config.map_sheet_index_geo_overedge_path)
    mapping = get_mapping_from_csv(config.topo_legend_path, col_key="class", col_value="pixel")
    cmap = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="color_rgba", convert_rgba=True)
    keys_no_mangrove = msi[msi["laos"] == 1]["key"].values
    polygons_water = gpd.read_file(config.polygons_imputation_water)
    polygons_forest = gpd.read_file(config.polygons_imputation_forest)
    polygons_impute = gpd.read_file(config.polygons_imputation_other)
    polygons_dmz = gpd.read_file(config.polygons_dmz)
    
    return {
        "keys_no_mangrove": keys_no_mangrove,
        "keys_brushwood_to_forest": config.keys_brushwood_to_forest,
        "polygons_water": polygons_water,
        "polygons_forest": polygons_forest,
        "polygons_impute": polygons_impute,
        "polygons_dmz": polygons_dmz,
        "mapping": mapping,
        "cmap": cmap,
        "nodata": config.nodata,
    }

def setup_postprocess2(config):
    """Loads parameters and data for postprocessing step 2."""
    cmap = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="color_rgba", convert_rgba=True)
    mapping_post2 = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="pixel_post2")
    
    return {
        "cmap": cmap,
        "mapping": mapping_post2,
        "min_area_impute": config.min_area_impute,
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "nodata": config.nodata,
    }

def setup_postprocess3(config):
    """Loads parameters and data for postprocessing step 3."""
    cmap = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="color_rgba", convert_rgba=True)
    mapping_post3 = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="pixel_post3")
    
    return {
        "cmap": cmap,
        "mapping": mapping_post3,
        "nodata": config.nodata,
    }

def setup_postprocess4(config):
    """Loads parameters and data for postprocessing step 4."""
    cmap = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="color_rgba", convert_rgba=True)
    mapping_l2_to_l1 = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="pixel_l1")
    
    return {
        "cmap": cmap,
        "mapping": mapping_l2_to_l1,
        "nodata": config.nodata,
    }

def crop_map_sheet(map_sheet, output_file, nodata, compress=None):
    """Crops scanned map sheet to map extent."""
    img_processed = process_map_sheet(
        map_sheet,
        mask_map_extent=True,
        nodata=nodata
    )
    
    profile_update = {"nodata": nodata, "compress": compress}
    write_processed_image(map_sheet, img_processed, output_file, profile_update=profile_update)


def predict_map_tiles(tiles, model):
    """Predicts map tiles using the trained model."""
    gc.collect()
    tiles = [tile.transpose(1, 2, 0) for tile in tiles]
    data_module = TopoMapsDataModule(pred_data=tiles)
    data_module.setup(stage="predict")

    pred_list = []
    with torch.no_grad():
        for batch in data_module.predict_dataloader():
            inputs = batch.to(model.device)
            preds = model.predict_step(inputs)  
            pred_list.append(preds)

    return np.vstack(pred_list)


def pred_map_sheet(map_sheet, output_file, model, tile_size, overlap, nodata, compress, cmap):
    """Predicts map sheet using the trained model."""
    img_processed = process_map_sheet(
        map_sheet=map_sheet,
        tile_proc_fn=lambda x: predict_map_tiles(x, model),
        proc_tiles_together=True,
        tile_size=tile_size,
        overlap=overlap,
        crop_to_map_extent=True,
        mask_map_extent=True,
        nodata=nodata
    )

    profile_update = {"count": 1, "nodata": nodata, "compress": compress, "dtype": "uint8"}
    write_processed_image(map_sheet, img_processed, output_file, profile_update=profile_update, cmap=cmap)

def postprocess1(
        map_sheet,
        output_file,
        polygons_water,
        polygons_forest,
        polygons_impute,
        polygons_dmz,
        keys_no_mangrove,
        keys_brushwood_to_forest,
        mapping,
        cmap,
        nodata
        ):
    """Applies postprocessing step 1 to map sheet."""
    # extract the correct transform
    _, transform = map_sheet.extract_map_extent(nodata=nodata)

    def img_proc_fn(x):
        # map all water polygons to water class
        x = impute_polygon_overlap(
            x,
            polygon=polygons_water.to_crs(map_sheet.crs).union_all(),
            transform=transform,
            impute_value=mapping.get("water")
            )
        
        # map all forest polygons to forest class
        x = impute_polygon_overlap(
            x,
            polygon=polygons_forest.to_crs(map_sheet.crs).union_all(),
            transform=transform,
            impute_value=mapping.get("forest")
            )

        # impute all areas with missing information (e.g. misaligned areas)
        x = impute_polygon_overlap(
            x,
            polygon=polygons_impute.to_crs(map_sheet.crs).union_all(),
            transform=transform,
            impute_value=nodata,
            )
        
        # impute all areas predicted as built-up in DMZ area
        x = impute_polygon_overlap(
            x,
            polygon=polygons_dmz.to_crs(map_sheet.crs).union_all(),
            transform=transform,
            impute_value=nodata,
            mask_class_id=mapping.get("built-up")
            )
        
        # map brushwood to forest for two map sheets with different legend symbols
        if map_sheet.id in keys_brushwood_to_forest:
            x = map_array_values(x, {mapping.get("brushwood"): mapping.get("forest")})

        # impute all mangrove predictions for map sheets with legend types 6 (Laos) as not included in legend
        if map_sheet.id in keys_no_mangrove:
            x = map_array_values(x, {mapping.get("mangrove"): nodata})

        # impute all previously exlcuded areas as well as predicted classes for text/other symbols and boundaries
        x = impute_nearest_nonvalue(x, values=[nodata, mapping.get("boundary"), mapping.get("other")])
        return x

    img_processed = process_map_sheet(
        map_sheet,
        img_proc_fn=img_proc_fn,
        crop_to_map_extent=True,
        mask_map_extent=True,
        nodata=nodata
    )

    write_processed_image(map_sheet, img_processed, output_file, cmap=cmap)

def postprocess2(
        map_sheet,
        output_file,
        mapping,
        min_area_impute,
        cmap,
        nodata,
        tile_size,
        overlap
        ):
    """Applies postprocessing step 2 to map sheet."""
    def img_proc_fn(x, mapping):
        x = map_array_values(x, mapping)
        x = impute_nearest_nonvalue(x, values=[nodata])
        return x

    img_processed = process_map_sheet(
        map_sheet,
        tile_proc_fn=lambda x: remove_small_connected_areas_all_values(x, min_size=min_area_impute), # runs faster when tiled
        img_proc_fn=lambda x: img_proc_fn(x, mapping=mapping),
        crop_to_map_extent=True,
        mask_map_extent=True,
        nodata=nodata,
        tile_size=tile_size,
        overlap=overlap
    )

    write_processed_image(map_sheet, img_processed, output_file, cmap=cmap)

def postprocess_mapping(map_sheet, output_file, mapping, cmap, nodata):
    """Applies integer mapping to map sheet predictions."""
    img_processed = process_map_sheet(
        map_sheet,
        img_proc_fn=lambda x: map_array_values(x, mapping),
        crop_to_map_extent=True,
        mask_map_extent=True,
        nodata=nodata
    )

    write_processed_image(map_sheet, img_processed, output_file, cmap=cmap)


def main(stage, parallel, max_workers):
    """
    Process map sheets through various stages of cropping, prediction and postprocessing.

    Parameters:
        stage (str): The processing stage to execute. Options include:
            "crop": Crops scanned map sheets to their extents.
            "pred": Predicts map sheet content using a trained model.
            "post1": Applies the first postprocessing step.
            "post2": Applies the second postprocessing step.
            "post3": Applies integer mapping for postprocessing step 3.
            "post4": Applies integer mapping for postprocessing step 4.
        parallel (bool): Whether to enable parallel processing.
        max_workers (int): The maximum number of workers to use for parallel processing.

    Returns:
        None
    """
    config = Config.Config()

    # Define a dictionary to map stages to their corresponding functions and setup steps
    stages = {
        "crop": {
            "function": crop_map_sheet,
            "setup": setup_crop_map_sheet,
            "input_stage": "raw",
        },
        "pred": {
            "function": pred_map_sheet,
            "setup": setup_pred_map_sheet,
            "input_stage": "raw",
        },
        "post1": {
            "function": postprocess1,
            "setup": setup_postprocess1,
            "input_stage": "pred",
        },
        "post2": {
            "function": postprocess2,
            "setup": setup_postprocess2,
            "input_stage": "post1",
        },
        "post3": {
            "function": postprocess_mapping,
            "setup": setup_postprocess3,
            "input_stage": "post2",
        },
        "post4": {
            "function": postprocess_mapping,
            "setup": setup_postprocess4,
            "input_stage": "post3",
        }
    }

    if stage not in stages:
        raise ValueError(f"Invalid stage: {stage}")

    # Perform setup for the specified stage
    print(f"Setting up {stage} processing ...")
    params = stages[stage]["setup"](config)

    # get input and output folders
    if stages[stage]["input_stage"] == "raw":
        input_folder = config.map_sheet_folder
    else:
        input_folder = f"{config.map_sheet_processed_folder}/{stages[stage]["input_stage"]}" 
    
    output_folder = f"{config.map_sheet_processed_folder}/{stage}" 
    
    # delete existing files in output folder
    if os.path.exists(output_folder):
        print(f"Deleting existing files in output folder {output_folder}")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Load map sheet index
    map_sheet_index = gpd.read_file(config.map_sheet_index_geo_overedge_path)
    # Apply small buffer to avoid gaps between map sheets especially after reprojecting
    map_sheet_index["geometry"] = map_sheet_index.geometry.buffer(config.buffer_map_sheet_extent)

    # Define the processing functions
    postprocess_fns = []
    for _, row in map_sheet_index.iterrows():
        map_id = row["key"]
        path = f"{input_folder}/{map_id}.tif"

        # Set up map sheet object
        map_sheet = MapSheet(
            sheet_id=row.key,
            path = path,
            polygon=row.geometry,
            polygon_crs=map_sheet_index.crs
        )
        output_file = f"{output_folder}/{map_id}.tif"
        # Wrapper function around main processing function allowing to pass parameters 
        postprocess_fn = functools.partial(stages[stage]["function"], map_sheet, output_file, **params)
        postprocess_fns.append(postprocess_fn) 

    # Run the processing functions in a loop or in parallel
    print(f"Applying {stage} processing to map sheets...")
    if parallel:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for postprocess_fn in postprocess_fns:
                futures.append(executor.submit(postprocess_fn))
                    
            # Wait for all tasks to finish
            for future in futures:
                future.result()
    else:
        for postprocess_fn in postprocess_fns:
            postprocess_fn()

    print(f"Finished processing stage {stage}. Processed files saved to folder {output_folder}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the script with specific model and batch size.")
    
    # Add arguments for model and batch_size
    parser.add_argument("--stage", type=str, default="pred", help="One of 'crop', 'pred', 'post1', 'post2', 'post3' or 'post4'")
    parser.add_argument("--parallel", action="store_true", help="Run the processing in parallel")
    parser.add_argument("--max_workers", type=int, default=7, help="Number of workers to use for parallel processing. Deafults to 7")

    # Parse arguments
    args = parser.parse_args()
    # Call main function with parsed arguments
    main(stage=args.stage, parallel=args.parallel, max_workers=args.max_workers)