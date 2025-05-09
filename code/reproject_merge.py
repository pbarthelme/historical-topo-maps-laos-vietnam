import argparse
import os
import concurrent.futures
import shutil
import subprocess
import rasterio

from config import Config

def reproject_ms_gdal(src_path, dst_path, compress, dst_crs, resolution):
    """
    Reprojects a raster file using GDAL.

    Parameters:
        src_path (str): Path to the source raster file.
        dst_path (str): Path to save the reprojected raster file.
        compress (str): Compression options for the output file.
        dst_crs (str): Destination coordinate reference system.
        resolution (float): Target resolution for the output raster.

    Returns:
        None
    """
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        
        if src_crs == "EPSG:3148":
            utm_zone = 48
        elif src_crs == "EPSG:3149":
            utm_zone = 49
        else:
            utm_zone = 47 

        # manual projection pipeline neede as the default conversion does properly convert the Indian 1960 datum,
        # likely due to a missing towgs84 parameter, e.g. https://github.com/OSGeo/PROJ/issues/1799
        # if not addressed this leads to a shift of about 500m
        proj_pipeline = (
            "+proj=pipeline "
            "+step +inv +proj=utm +zone={utm_zone} +ellps=evrst30 "
            "+step +proj=push +v_3 "
            "+step +proj=cart +ellps=evrst30 "
            "+step +proj=helmert +x=198 +y=881 +z=317 +rx=0 +ry=0 +rz=0 +s=0 "
            "+convention=position_vector "
            "+step +inv +proj=cart +ellps=WGS84 "
            "+step +proj=pop +v_3 "
            "+step +proj=aea +lat_0=-15 +lon_0=125 +lat_1=7 +lat_2=-32 +x_0=0 +y_0=0 +ellps=WGS84"
        ).format(utm_zone=utm_zone)
        
        # Base command
        command = [
            "gdalwarp",
            "--config", "GDAL_CACHEMAX", "4000",
            "-t_srs", dst_crs,
            "-ct", f"'{proj_pipeline}'",
            "-tr", str(resolution), str(resolution),
            compress,
            "-co", "TILED=YES",
            "-r", "nearest",
            "-overwrite",
            src_path,
            dst_path
        ]
        command = " ".join(command)
        subprocess.run(command, check=True, shell=True)


def merge_with_vrt(input_folder, output_file, compress, max_workers, overviews=True):
    """
    Merges raster files into a single output file using GDAL VRT.

    Parameters:
        input_folder (str): Path to the folder containing input raster files.
        output_file (str): Path to save the merged raster output.
        compress (str): Compression options for the output file.
        max_workers (int): Number of threads to use for processing.
        overviews (bool): Whether to generate overviews for the output file (default: True).

    Returns:
        None
    """
    input_files = f"{input_folder}/*.tif"
    output_file_vrt = output_file.replace(".tif", ".vrt")

    # Build VRT
    command_vrt = f"gdalbuildvrt -overwrite {output_file_vrt} {input_files}"
    subprocess.run(command_vrt, check=True, shell=True)

    # combined map
    command_translate = [
        "gdal_translate",
        "--config", "GDAL_CACHEMAX", "4000",
        "-of", "GTiff",
        "-co", "BIGTIFF=YES",
        compress,
        "-co", "TILED=YES",
        "-co", f"NUM_THREADS={max_workers}",
        "-r", "nearest",
        output_file_vrt,
        output_file
    ]
    command_translate = " ".join(command_translate)
    print(command_translate)
    subprocess.run(command_translate, check=True, shell=True)
    
    # delete intermediate vrt file
    os.remove(output_file_vrt)

    if overviews:
        command_overviews = f"gdaladdo --config GDAL_CACHEMAX 4000 {output_file}"
        print(command_overviews)
        subprocess.run(command_overviews, check=True, shell=True)

def main(stage, crop_to_geom, parallel, max_workers):
    """
    Reprojects, merges and crops raster files based on the specified stage.

    Parameters:
        stage (str): Processing stage, e.g., "crop", "pred", "post1", etc.
        crop_to_geom (bool): Whether to crop the output to specific geometries.
        parallel (bool): Whether to enable parallel processing.
        max_workers (int): Number of workers to use for parallel processing.

    Returns:
        None
    """
    config = Config.Config()

    # use JPEG compression for scanned map sheet images and LZW compression for output LUC maps
    if stage == "crop":
        compress = "-co COMPRESS=JPEG -co JPEG_QUALITY=50"
    else:
        compress = f"-co COMPRESS={config.compress}"
    
    input_folder = f"{config.map_sheet_processed_folder}/{stage}" 
    merged_path = f"{input_folder}.tif"

    # reproject file in input folder before merging
    tmp_folder = f"{input_folder}_reprojected"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    # reproject the files in the input folder
    files = [file for file in os.listdir(input_folder) if file.endswith(".tif")]
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in sorted(files):
                src_path = f"{input_folder}/{file}"
                dst_path = f"{tmp_folder}/{file}"
                
                futures.append(executor.submit(
                    reproject_ms_gdal,
                    src_path,
                    dst_path,
                    compress,
                    config.output_crs,
                    config.res_detailed,
                    ))
                        
            # Wait for all tasks to finish
            for future in futures:
                future.result()
    else:
        for file in sorted(files):
            src_path = f"{input_folder}/{file}"
            dst_path = f"{tmp_folder}/{file}"
            reproject_ms_gdal(
                src_path,
                dst_path,
                compress,
                config.output_crs,
                config.res_detailed,
                )

    # merge reprojected files using intermediate VRT
    merge_with_vrt(tmp_folder, merged_path, compress, max_workers, overviews=True)

    # delete tmp folder
    shutil.rmtree(tmp_folder)

    # reproject to 30m resolution
    merged_path_30m = f"{input_folder}_30m.tif"
    command = [
        "gdalwarp",
        "--config", "GDAL_CACHEMAX", "4000",
        "-tr", str(config.res_coarse), str(config.res_coarse),
        compress,
        "-co", "TILED=YES",
        "-co", f"NUM_THREADS={max_workers}",
        "-r", "mode",
        "-overwrite",
        merged_path,
        merged_path_30m
    ]
    command = " ".join(command)
    subprocess.run(command, check=True, shell=True)
    
    command_overviews = f"gdaladdo --config GDAL_CACHEMAX 4000 {merged_path_30m}"
    print(command_overviews)
    subprocess.run(command_overviews, check=True, shell=True)

    # create country-specifc files 
    if crop_to_geom:
        for name, geom in config.cropping_geoms.items():
            cropped_path = merged_path.replace(".tif", f"_{name}.tif")
            command_crop = [
                "gdalwarp",
                "--config", "GDAL_CACHEMAX", "500",
                "-wm", "500",
                "-wo", f"{max_workers}",
                "-of", "GTiff",
                "-co", "BIGTIFF=YES",
                compress,
                "-co", "TILED=YES",
                "-cutline", f"{geom}",
                "-crop_to_cutline",
                "-r", "nearest",
                "-overwrite",
                merged_path,
                cropped_path
            ]
            command_crop = " ".join(command_crop)
            print(command_crop)
            subprocess.run(command_crop, check=True, shell=True)

            command_overviews = f"gdaladdo --config GDAL_CACHEMAX 4000 {cropped_path}"
            print(command_overviews)
            subprocess.run(command_overviews, check=True, shell=True)

            # reproject to 30m resolution
            cropped_path_30m = cropped_path.replace(".tif", f"_30m.tif")
            command = [
                "gdalwarp",
                "--config", "GDAL_CACHEMAX", "4000",
                "-tr", str(config.res_coarse), str(config.res_coarse),
                compress,
                "-co", "TILED=YES",
                "-co", f"NUM_THREADS={max_workers}",
                "-r", "mode",
                "-overwrite",
                cropped_path,
                cropped_path_30m
            ]
            command = " ".join(command)
            subprocess.run(command, check=True, shell=True)
            
            command_overviews = f"gdaladdo --config GDAL_CACHEMAX 4000 {cropped_path_30m}"
            print(command_overviews)
            subprocess.run(command_overviews, check=True, shell=True)
            

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the script with specific model and batch size.")
    
    # Add arguments for model and batch_size
    parser.add_argument("--stage", type=str, default="pred", help="One of 'crop', 'pred', 'post1', 'post2', 'post3' or 'post4'")
    parser.add_argument("--crop_to_geom", action="store_true", help="Include cropping to country outlines")
    parser.add_argument("--parallel", action="store_true", help="Run the processing in parallel")
    parser.add_argument("--max_workers", type=int, default=7, help="Number of workers to use for parallel processing. Deafults to 7")

    # Parse arguments
    args = parser.parse_args()
    # Call main function with parsed arguments
    main(
        stage=args.stage,
        crop_to_geom=args.crop_to_geom,
        parallel=args.parallel,
        max_workers=args.max_workers
        )