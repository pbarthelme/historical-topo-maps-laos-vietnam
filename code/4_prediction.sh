#!/bin/bash
#SBATCH --account geos_gpgpu
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Activate conda environment
module load cuda
source /opt/conda/etc/profile.d/conda.sh
conda activate topo-maps

python process_map_sheets.py --stage "pred"
python reproject_merge.py --stage "pred" --parallel
