# Large-Scale Historical Land Use Mapping in Vietnam and Laos using military topographic maps
This is the code repository for the manuscript **Large-Scale Historical Land Use Mapping in Vietnam and Laos using military topographic maps** (in submission).

## Setup
This implementation uses Python 3.12.4 in combination with multiple other packages specified in the `./environment.yml` file. A conda environment with the corresponding packages can be created using the command `conda env create -f environment.yml`. For exact package versions used during the analysis, refer to `./environment_detailed.yml`.

## Structure
The repository is organized into Jupyter notebooks and bash scripts located in the `code/` folder. The files are numbered to indicate the order in which they should be executed, as they rely on outputs from previous steps. Core functions are implemented in reusable modules for better organization. Configuration parameters and file paths are defined in `code/config/Config.py`.

The code for this analysis was run on the Forth compute cluster at the University of Edinburgh which uses a SLURM workload manager. This is reflected in the bash scripts [3_training.sh](code/3_training.sh), [4_prediction.sh](code/4_prediction.sh) and [5_postprocessing.sh](code/5_postprocessing.sh) which include settings specific to the Forth cluster. These settings might need to be adjusted when running the code on a different cluster/machine. Model training and prediction was run on a NVIDIA L4 GPU. Parts of the raster reprojection code executes gdal commands in the shell using the `subprocess` Python library which requires GDAL (version 3.4.1) to be installed. 

The code also includes one R script for the calculation of accuracy metrics and area estimates based on Stehman (2014) utilising an existing implementation in R instead of duplicating code in Python.

## Data Availability
The data used for this analysis can be accessed via the links specified below. The georeferenced map sheets are currently only available on request until details on the declassification/release for public distribution of the source files are clarified.
* **data/raw**: download this folder from the [data repository](https://doi.org/10.5281/zenodo.15357898)

Add the following input data from other sources:
* **data/raw/boundaries**: [The union of world country boundaries and EEZs, version 4](https://marineregions.org/download_file.php?name=EEZ_land_union_v4_202410.zip) 
* **data/raw/boundaries**: [GADM boundaries](https://gadm.org) ([Vietnam Level 0](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_0.json), [Vietnam Level 1](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_1.json), [Lao PDR Level 0](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LAO_0.json), [Lao PDR Level 1](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LAO_1.json), [Cambodia Level 0](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KHM_0.json), [China Level 0](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_CHN_0.json), [Myanmar Level 0](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MMR_0.json), [Thailand Level 0](https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_THA_0.json))
* **data/raw/luc**: [Global Mangrove Watch Data Version 3.0](https://doi.org/10.5281/zenodo.6894273) folder [gmw_v3_1996_gtiff.zip](https://zenodo.org/records/6894273/files/gmw_v3_1996_gtiff.zip?download=1)
* **data/raw/luc**: [GLC_FCS30D Data Version 1](https://doi.org/10.5281/zenodo.8239305) folder [GLC_FCS30D_19852022maps_E100-E105.zip](https://zenodo.org/records/8239305/files/GLC_FCS30D_19852022maps_E100-E105.zip?download=1)
* **data/raw/map_sheets**: the georeferenced map sheets are currently only available on request until details on the limited distribution restrictions of the source files are clarified.


## File Overview

### [0_study_areas.ipynb](code/0_study_areas.ipynb)
Defines the study areas by intersecting country boundaries with map sheet coverage. Outputs GeoJSON files for study areas.

### [1_preprocessing.ipynb](code/1_preprocessing.ipynb)
Generates image tiles for labeling using stratified random sampling and adaptive sampling. Outputs a sample tile catalog and writes out sample images.

### [1.5_labelbox.ipynb](code/1.5_labelbox.ipynb)
Handles image labeling using the Labelbox API. Includes code for uploading image tiles to Labelbox, downloading annotations, and processing them into masks. Running this would require Labelbox API access so the code is only included for reference. The labelled image masks are provided in the data repository.  

### [2_training_data.ipynb](code/2_training_data.ipynb)
Prepares training datasets by combining image and label tiles to generate input-output pairs for model training. Splits data into training and validation sets.

### [3_training.sh](code/3_training.sh)
Runs model training experiments using different architectures, loss functions, and augmentation settings. Configurable via command-line arguments. Calls the [train_model.py](code/train_model.py) scripts.

### [4_prediction.sh](code/4_prediction.sh)
Generates predictions for the entire dataset using the trained model. Handles tiling and merging of prediction outputs. Calls the [process_map_sheets.py](code/process_map_sheets.py) and [reproject_merge.py](code/reproject_merge.py) scripts.

### [5_postprocessing.sh](code/5_postprocessing.sh)
Post-processes model predictions, reprojects them, and merges them into final outputs. Calls the [process_map_sheets.py](code/process_map_sheets.py) and [reproject_merge.py](code/reproject_merge.py) scripts.

### [6_accuracy_eval.ipynb](code/6_accuracy_eval.ipynb)
Evaluates the accuracy of predictions using test samples. Outputs confusion matrices and other evaluation metrics.

### [6_forest_cover.ipynb](code/6_forest_cover.ipynb)
Analyzes forest cover changes by aggregating results across study areas and map sheets. Outputs change statistics and vector files.

### [6_mangroves.ipynb](code/6_mangroves.ipynb)
Analyzes mangrove cover changes by aggregating results across zones and map sheets. Outputs change statistics and vector files.

### [7_accuracy_eval_stehman.R](code/7_accuracy_eval_stehman.R)
Calculates overall accuracy, class-based user and producer accuracies and area estimates of the final L2 LUC maps for Laos and Vietnam using outputs from the [6_accuracy_eval.ipynb](code/6_accuracy_eval.ipynb) notebook and methods described in Stehman (2014).

### [7_plots.ipynb](code/7_plots.ipynb)
Creates all Figures in the paper using outputs from previous steps.
