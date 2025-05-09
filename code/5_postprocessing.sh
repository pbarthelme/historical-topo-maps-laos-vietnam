#!/bin/bash
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate topo-maps

# Post-process map sheet predictions
python process_map_sheets.py --stage "post1" --parallel
python reproject_merge.py --stage "post1" --parallel --crop_to_geom
python process_map_sheets.py --stage "post2" --parallel
python reproject_merge.py --stage "post2" --parallel --crop_to_geom
python process_map_sheets.py --stage "post3" --parallel
python reproject_merge.py --stage "post3" --parallel --crop_to_geom
python process_map_sheets.py --stage "post4" --parallel
python reproject_merge.py --stage "post4" --parallel --crop_to_geom

# Crop, reproject and merge original map sheets
python process_map_sheets.py --stage "crop" --parallel
python reproject_merge.py --stage "crop" --parallel --crop_to_geom
