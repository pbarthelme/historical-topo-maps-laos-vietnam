import argparse
import glob
import os
import shutil
import torch
import wandb

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config import Config
from core.dataloader import TopoMapsDataModule
from core.losses import FocalLoss, DeterministicCrossEntropyLoss
from core.model import SegmentationModelLightning
from core.utils import get_mapping_from_csv
from core.training import calc_class_report, compute_smoothed_weights, pred_val_data

    
def main(model_type, encoder_name, loss, weight_alpha, aug_setting):
    """
    Trains and evaluates a semantic segmentation model.

    This function sets up the data module, model, loss function, and training pipeline using PyTorch Lightning.
    It also handles logging, checkpointing, and evaluation of the model.

    Parameters:
        model_type (str): The type of segmentation model to use (e.g., 'unet', 'deeplabv3').
        encoder_name (str): The name of the encoder to use (e.g., 'resnet50').
        loss (str): The loss function to use ('cross-entropy' or 'focal').
        weight_alpha (float): The alpha parameter for calculating smoothed class weights. 
                              A value of 0 means equal class weights, while 1 means proportional class weights.
        aug_setting (str): The augmentation setting to use ('no_color_jitter', 'low_color_jitter', or 'high_color_jitter').
    """
    # Load config and set seeds
    config = Config.Config()
    seed_everything(config.seed, workers=True)
    torch.use_deterministic_algorithms(True)
    
    # Create name for run
    run_name = f"{model_type}_{encoder_name}_{loss}_walpha_{weight_alpha}_aug_{aug_setting}"

    # Define augmentation parameters
    aug_params = config.aug_settings.get(aug_setting)

    # Define class weights
    weights = compute_smoothed_weights(config.expected_class_counts, alpha=weight_alpha).tolist()
    weights.insert(config.ignore_index, 0) # add weight of zero for ignored index which is not in the expected class counts
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    # Define loss functions
    if loss == "focal":
        loss_fn = FocalLoss(alpha=weights, gamma=2.0, ignore_index=config.ignore_index)
    elif loss == "cross-entropy":
        loss_fn = DeterministicCrossEntropyLoss(weight=weights, ignore_index=config.ignore_index)
    else:
        raise ValueError("Invalid loss function specified. Please choose 'cross-entropy' or 'focal'.")

    # Load data module with augmentation parameters
    data_module = TopoMapsDataModule(
        data_path=config.training_data_path,
        batch_size=config.batch_size,
        rot_deg=config.rot_deg,
        rot_fill=config.ignore_index,
        **aug_params
        )
    
    # Load model 
    model = SegmentationModelLightning(
        model_name=model_type,
        encoder_name=encoder_name,
        encoder_weights=config.encoder_weights,
        classes=config.classes,
        loss_fn=loss_fn,
        lr=config.lr,
        ignore_index=config.ignore_index,
        )
        
    # Set up checkpoints and early stopping
    checkpoint_path = f"{config.checkpoint_folder}/{run_name}"
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, save_top_k=1, monitor="valid_macro_f1", mode="max")
    callbacks=[checkpoint_callback]

    # Set up wandb logging and log hyperparameters
    wandb_logger = WandbLogger(project=config.wandb_project, name=run_name)
    wandb_logger.experiment.config["model_type"] = model_type
    wandb_logger.experiment.config["weight_alpha"] = weight_alpha
    wandb_logger.experiment.config["aug_setting"] = aug_setting
    wandb_logger.experiment.config["lr"] = config.lr
    wandb_logger.experiment.config["batch_size"] = config.batch_size

    # Set up trainer and fit model
    trainer = Trainer(
        callbacks=callbacks,
        deterministic=True,
        devices=1,
        accelerator="gpu",
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        logger=wandb_logger
        )

    trainer.fit(model, datamodule=data_module)

    # Log lowest validation loss
    lowest_val_loss = checkpoint_callback.best_model_score.cpu().numpy()
    wandb_logger.log_metrics({"lowest_val_loss": lowest_val_loss})

    # Evaluate best model saved in checkpoint path on full validation dataset
    checkpoint_files = glob.glob(f"{checkpoint_path}/*.ckpt")
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
    model = SegmentationModelLightning.load_from_checkpoint(checkpoint_files[0])

    data_module.setup()
    y_pred, y_true = pred_val_data(model, data_module.val_dataloader())
    class_mapping = get_mapping_from_csv(config.topo_legend_path, col_key="pixel", col_value="class")

    class_report_df = calc_class_report(y_pred, y_true, config.ignore_index, class_mapping)

    if not os.path.exists(config.eval_folder):
        os.makedirs(config.eval_folder)
    
    output_path = f"{config.eval_folder}/{run_name}.csv"
    class_report_df.to_csv(output_path, index=True)

    wandb_logger.log_metrics({"overall_acc": class_report_df.loc["accuracy"]["f1-score"]})
    wandb_logger.log_metrics({"avg_f1_score": class_report_df.loc["macro avg"]["f1-score"]})

    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the script with specific model type and parameters.")
    
    # Add arguments for model and batch_size
    parser.add_argument("--model_type", type=str, required=True, help="Specify the model.")
    parser.add_argument("--encoder_name", type=str, default="resnet50", help="Specify the name of the model encoder.")
    parser.add_argument("--loss", type=str, default="cross-entropy", help="Setting for loss function, either 'cross-entropy' or 'focal'.")
    parser.add_argument("--weight_alpha", type=float, default=0, help="Specify the alpha parameter for calculation of smoothed class weights. Default of 0 means equal class weights while 1 means proportional class weights.")
    parser.add_argument("--aug_setting", type=str, default="no_color_jitter", help="Specify the augmentation setting to use, either 'no_color_jitter', 'low_color_jitter' or 'high_color_jitter'")

    # Parse arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(
        model_type=args.model_type,
        encoder_name=args.encoder_name,
        loss=args.loss,
        weight_alpha=args.weight_alpha,
        aug_setting=args.aug_setting
        )