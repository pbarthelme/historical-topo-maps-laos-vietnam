#!/bin/bash
#SBATCH --account geos_gpgpu
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00

# Activate conda environment
module load cuda
source /opt/conda/etc/profile.d/conda.sh
conda activate topo-maps

### standard settings - model architecture comparison
python train_model.py --model_type "unet" --encoder_name "resnet50" --loss "focal" --weight_alpha 0 --aug_setting "no_color_jitter"
python train_model.py --model_type "unet++" --encoder_name "resnet50" --loss "focal" --weight_alpha 0 --aug_setting "no_color_jitter"
python train_model.py --model_type "deeplabv3plus" --encoder_name "resnet50" --loss "focal" --weight_alpha 0 --aug_setting "no_color_jitter"
python train_model.py --model_type "segformer" --encoder_name "mit_b2" --loss "focal" --weight_alpha 0 --aug_setting "no_color_jitter"

### experiments with best model architecture
# cross-entropy loss
python train_model.py --model_type "unet++" --encoder_name "resnet50" --loss "cross-entropy" --weight_alpha 0 --aug_setting "no_color_jitter"

# class weights
python train_model.py --model_type "unet++" --encoder_name "resnet50" --loss "focal" --weight_alpha 1 --aug_setting "no_color_jitter"
python train_model.py --model_type "unet++" --encoder_name "resnet50" --loss "cross-entropy" --weight_alpha 1 --aug_setting "no_color_jitter"

# augmentation with color jitter
python train_model.py --model_type "unet++" --encoder_name "resnet50" --loss "focal" --weight_alpha 0 --aug_setting "low_color_jitter"
python train_model.py --model_type "unet++" --encoder_name "resnet50" --loss "focal" --weight_alpha 0 --aug_setting "high_color_jitter"
